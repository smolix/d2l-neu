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
that the slides read), plus ``img/mdl-opt-dual-envelope.svg`` and
``img/mdl-opt-svm-margin.svg`` (book only).
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
WHITE = dict(facecolor="white", edgecolor="none", pad=1.5)


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


def fig_dual_envelope():
    """The dual function as the lower envelope of affine functions.

    Clipping problem of the section: f(x) = |x - x0|^2 / 2 with x0 = (3, 4),
    g(x) = (|x|^2 - 1) / 2.  Each fixed x gives the line
    lam -> L(x, lam) = f(x) + lam g(x).  The lines drawn belong to the
    Lagrangian minimizers x(lam_k) = x0 / (1 + lam_k), so each one touches
    q(lam) = 12.5 lam / (1 + lam) - lam / 2 at lam_k; q is their pointwise
    minimum.  The maximum of q, at lam = 4, equals p* = 8.
    """
    x0 = np.array([3.0, 4.0])
    f = lambda x: 0.5 * np.sum((x - x0) ** 2)
    g = lambda x: 0.5 * (np.sum(x ** 2) - 1.0)
    q = lambda lam: 12.5 * lam / (1 + lam) - lam / 2
    lr, vr = (0.0, 12.6), (0.0, 11.0)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    lams = np.linspace(0.0, 12.0, 600)
    for lk in (0.0, 1.0, 2.5, 4.0, 7.0, 11.0):
        xk = x0 / (1 + lk)
        line = f(xk) + lams * g(xk)
        keep = (line <= vr[1]) & (line >= vr[0] + 0.05)
        ax.plot(lams[keep], line[keep], color=GRAY, lw=1.4,
                ls=(0, (5, 3)), zorder=2)
        ax.plot(lk, q(lk), "o", color=GRAY, ms=5, zorder=5)
    ax.plot(lams, q(lams), color=BLUE, lw=2.6, zorder=4)
    ax.plot(4.0, 8.0, "o", color="black", ms=6.5, zorder=6)

    # x(4) = (0.6, 0.8) is feasible with g = 0: its line is flat at p* = 8
    ax.text(12.0, 8.12, r"$\mathbf{x}^\star$: flat at $p^\star=8$",
            color="black", fontsize=LBL, ha="right", va="bottom")
    ax.text(4.0, 7.75, r"$d^\star=q(4)$", color="black", fontsize=LBL,
            ha="center", va="top", bbox=WHITE, zorder=7)
    ax.text(9.4, 4.75, r"$q(\lambda)=\inf_{\mathbf{x}}\,"
            r"\mathcal{L}(\mathbf{x},\lambda)$", color=BLUE,
            fontsize=LBL + 1, ha="center", va="center", bbox=WHITE, zorder=7)
    ax.plot([5.3, 6.1], [2.4, 2.4], color=GRAY, lw=1.4, ls=(0, (5, 3)))
    ax.text(6.3, 2.4, r"one line $\lambda\mapsto\mathcal{L}(\mathbf{x},"
            r"\lambda)$ per $\mathbf{x}$", color=GRAY, fontsize=LBL,
            ha="left", va="center")

    fl.axis_cross(ax, lr, vr, color="black")
    ax.text(lr[1], -0.2, r"$\lambda$", color="black", fontsize=LBL + 1,
            ha="right", va="top")
    ax.text(-0.3, vr[1], "value", color="black", fontsize=LBL,
            ha="right", va="top")
    fl.clean_axes(ax, lim=((-1.4, lr[1] + 0.1), (-0.9, vr[1] + 0.1)),
                  hide=True, equal=False)
    fl.save(fig, "mdl-opt-dual-envelope")


def fig_svm_margin():
    """The hard-margin SVM of the section's code cell.

    Eight points; the offset-folded solution w = (4/7, 4/7), b = -3/7 gives
    the decision line x1 + x2 = 3/4 and the margin lines x1 + x2 = 5/2 and
    x1 + x2 = -1.  The four points with margin exactly 1 are ringed: they are
    the only points that can carry alpha_i > 0.  The margin lines are
    2 / |w| apart.
    """
    X = np.array([[1.0, 1.5], [2.0, 0.5], [2.5, 2.0], [1.5, 3.0],
                  [-1.0, -0.5], [-2.0, 1.0], [-1.5, -2.0], [0.5, -1.5]])
    y = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    w, b = np.array([4.0, 4.0]) / 7, -3.0 / 7
    margins = y * (X @ w + b)
    xr, yr = (-3.0, 3.6), (-3.05, 3.7)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    t = np.linspace(-4.5, 4.5, 2)
    u = np.array([1.0, -1.0]) / np.sqrt(2)          # direction of the lines
    n = w / np.linalg.norm(w)                       # unit normal
    for level, color, ls in ((0.0, "black", "-"), (1.0, BLUE, (0, (5, 3))),
                             (-1.0, ORANGE, (0, (5, 3)))):
        p0 = n * (level - b) / np.linalg.norm(w)    # point on w.x + b = level
        pts = p0[None, :] + t[:, None] * u[None, :]
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.0 if level else 2.4,
                ls=ls, zorder=2)

    for xi, yi, mi in zip(X, y, margins):
        ax.plot(*xi, "o", color=BLUE if yi > 0 else ORANGE, ms=8, zorder=5)
        if abs(mi - 1.0) < 1e-9:
            ax.plot(*xi, "o", mfc="none", mec="black", mew=1.6, ms=17,
                    zorder=6)

    def on(level, along):
        """The point of w.x + b = level at coordinate ``along`` on u."""
        return n * (level - b) / np.linalg.norm(w) + along * u

    # the margin width 2 / |w|, drawn across the empty upper-left strip
    ax.annotate("", xy=on(1.0, -3.0), xytext=on(-1.0, -3.0),
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.3,
                                shrinkA=0, shrinkB=0, mutation_scale=12),
                zorder=4)
    lab = on(0.5, -2.45)
    ax.text(lab[0], lab[1], r"$2/\|\mathbf{w}\|$", color="black",
            fontsize=LBL + 1, ha="center", va="center", zorder=7)

    # line labels along the lines, inside the empty margin strip, clear of
    # the ringed points and of the width arrow
    for level, color, off in ((1.0, BLUE, -0.42), (0.0, "black", -0.4),
                              (-1.0, ORANGE, 0.42)):
        p0 = on(level, {1.0: 2.2, 0.0: -1.4, -1.0: 2.6}[level]) + off * n
        text = (r"$\mathbf{w}^{\top}\mathbf{x}+b=%s$"
                % {1.0: "1", 0.0: "0", -1.0: "-1"}[level])
        ax.text(p0[0], p0[1], text, color=color, fontsize=LBL, ha="center",
                va="center", rotation=-45, rotation_mode="anchor", zorder=7)

    fl.clean_axes(ax, lim=(xr, yr), hide=True, equal=True)
    fl.save(fig, "mdl-opt-svm-margin")


FIGURES = [fig_conjugate_intercept, fig_dual_envelope, fig_svm_margin]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
