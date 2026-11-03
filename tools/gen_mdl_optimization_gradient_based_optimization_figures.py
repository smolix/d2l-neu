#!/usr/bin/env python3
"""Generate the new illustrative figures of "Gradient-Based Optimization"
(``chapter_mdl-optimization/mdl-gradient-based-optimization.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's other three figures (``mdl-opt-gd-bowl-vs-valley``,
``mdl-opt-eta-tent`` and ``mdl-opt-momentum-damping``) stay in
``gen_mdl_optimization_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_optimization_gradient_based_optimization_figures.py

Writes ``img/mdl-opt-descent-lemma-slice.svg``,
``img/mdl-opt-hb-eigenvalues.svg`` and ``img/mdl-opt-trust-region-saddle.svg``
(and copies in ``img/auto/``).  Every curve and point is computed: the
objective and its ceiling along the gradient step, the roots of the heavy-ball
characteristic polynomial, and the Newton and trust-region iterates on
f(x, y) = x^2/2 + cos y.
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

LBL = 13          # the chapter's in-figure label size
BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none")


def _copy_to_auto(name):
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


# --------------------------------------------------------------------------- #
# 1. The descent lemma along one gradient step                               #
# --------------------------------------------------------------------------- #

def fig_descent_lemma_slice():
    """f(x - eta g) for f = (x^2 + 10 y^2)/2 at (2, 1), g = (2, 10).

    phi(eta) = 7 - 104 eta + 502 eta^2   (g^T A g = 1004)
    ceiling  = 7 - 104 eta + 520 eta^2   (L/2 |g|^2 = 520)
    The ceiling is tangent at eta = 0, minimized at 1/L = 0.1 (value 1.8) and
    back at 7 at 2/L = 0.2.  Dots at the opening's step sizes 0.01, 0.1, 0.25.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    er, fr = (0.0, 0.272), (0.0, 14.6)
    eta = np.linspace(0.0, 0.265, 400)
    phi = 7 - 104 * eta + 502 * eta ** 2
    ceil = 7 - 104 * eta + 520 * eta ** 2

    # reference lines: the start value f = 7 and the ceiling's return at 2/L
    ax.plot([0, 0.265], [7, 7], color=GRAY, lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.plot([0.2, 0.2], [0, 7], color=GRAY, lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.plot([0.1, 0.1], [0, 1.62], color=GRAY, lw=1.0, ls=(0, (2, 3)),
            zorder=1)

    ax.plot(eta, ceil, color=ORANGE, lw=2.0, ls=(0, (6, 3)), zorder=3)
    ax.plot(eta, phi, color=BLUE, lw=2.2, zorder=4)

    # the three steps of the opening
    for e in (0.01, 0.1, 0.25):
        ax.plot(e, 7 - 104 * e + 502 * e * e, "o", color=BLUE, ms=6.5,
                zorder=6)
    ax.plot(0.1, 1.8, "o", mfc="white", mec=ORANGE, mew=1.8, ms=8.5,
            zorder=5)

    ax.text(0.019, 6.0, r"$\eta=0.01$: $6.01$", color="black", fontsize=LBL,
            ha="left", va="center", bbox=BOX, zorder=7)
    ax.annotate(r"bound $1.8$", xy=(0.1, 1.8), xytext=(0.05, 4.1),
                color=ORANGE, fontsize=LBL, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9,
                                shrinkA=2, shrinkB=5), zorder=7)
    ax.annotate(r"actual $1.62$", xy=(0.1, 1.62), xytext=(0.118, 0.5),
                color=BLUE, fontsize=LBL, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9,
                                shrinkA=2, shrinkB=4), zorder=7)
    ax.text(0.257, 12.1, r"$\eta=0.25$: $12.4$", color="black",
            fontsize=LBL, ha="left", va="center", bbox=BOX, zorder=7)

    ax.text(0.218, 10.6, "ceiling", color=ORANGE, fontsize=LBL,
            ha="right", va="center", bbox=BOX)
    ax.text(0.236, 8.2, r"$f(\mathbf{x}-\eta\mathbf{g})$", color=BLUE,
            fontsize=LBL, ha="left", va="center", bbox=BOX)
    ax.text(0.003, 7.25, r"$f(\mathbf{x})=7$", color="black", fontsize=LBL,
            ha="left", va="bottom")
    ax.text(0.2, -0.25, r"$2/L$", color="black", fontsize=LBL, ha="center",
            va="top")
    ax.text(0.1, -0.25, r"$1/L$", color="black", fontsize=LBL, ha="center",
            va="top")
    ax.text(er[1], -0.25, r"$\eta$", color="black", fontsize=LBL + 1,
            ha="right", va="top")
    ax.text(-0.004, fr[1], r"$f$", color="black", fontsize=LBL + 1,
            ha="right", va="top")

    fl.axis_cross(ax, er, fr, color="black")
    fl.clean_axes(ax, lim=((-0.012, 0.318), (-1.3, fr[1] + 0.2)),
                  hide=True, equal=False)
    fl.save(fig, "mdl-opt-descent-lemma-slice")
    _copy_to_auto("mdl-opt-descent-lemma-slice")


# --------------------------------------------------------------------------- #
# 2. Heavy-ball roots: flat modulus on an interval, a circle in the plane   #
# --------------------------------------------------------------------------- #

def _hb_roots(eta_lam, beta):
    return np.roots([1.0, -(1.0 + beta - eta_lam), beta])


def fig_hb_eigenvalues():
    """Heavy ball at the tuning eta*, beta* for kappa = 10.

    (a) max |z| over the two roots of z^2 - (1 + beta - eta lam) z + beta as a
        function of eta*lam, for beta = beta* = 0.2699: flat at sqrt(beta) =
        0.519 on [(1 - sqrt(beta))^2, (1 + sqrt(beta))^2] = [0.231, 2.309],
        rising outside; gradient descent's |1 - eta lam| for comparison; dots
        at the two extreme modes eta* lam_min and eta* lam_max.
    (b) The complex plane: unit circle (faint), circle of radius 0.519, the
        root pairs for lam in {1, 2, 3, 5, 10} at the tuning, and gradient
        descent's real factors 1 - (2/11) lam for the same lam.
    """
    kappa = 10.0
    sk = np.sqrt(kappa)
    beta = ((sk - 1) / (sk + 1)) ** 2            # 0.2699
    eta = (2 / (1 + sk)) ** 2                    # 0.2309 (lam_min = 1)
    r = np.sqrt(beta)                            # 0.5195
    lo, hi = (1 - r) ** 2, (1 + r) ** 2          # 0.2309, 2.3091

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), gridspec_kw=dict(width_ratios=[1.32, 1.0]))

    # ============================ (a) ============================ #
    xs = np.linspace(0.0, 2.75, 1200)
    mod = np.array([np.abs(_hb_roots(x, beta)).max() for x in xs])
    keep = mod <= 1.12
    xt = np.linspace(0.0, 2.08, 300)           # the tent, up to |1-x| = 1.08
    axa.plot([0, 2.75], [1, 1], color=GRAY, lw=1.1, ls=(0, (5, 3)), zorder=1)
    axa.plot(xt, np.abs(1 - xt), color=GRAY, lw=1.8, zorder=2)
    axa.plot(xs[keep], mod[keep], color=GREEN, lw=2.4, zorder=3)
    for x in (lo, hi):
        axa.plot([x, x], [0, r], color=GRAY, lw=0.9, ls=(0, (2, 3)), zorder=1)
        axa.plot(x, r, "o", color=GREEN, ms=7.5, zorder=5)
    axa.text(lo, -0.04, r"$\eta\lambda_{\min}$", color="black",
             fontsize=LBL, ha="center", va="top")
    axa.text(hi, -0.04, r"$\eta\lambda_{\max}$", color="black",
             fontsize=LBL, ha="center", va="top")
    axa.text(1.0, r + 0.035, r"$\sqrt{\beta^{\star}}=0.52$", color=GREEN,
             fontsize=LBL, ha="center", va="bottom")
    axa.text(1.9, 0.42, "complex roots", color=GREEN, fontsize=LBL,
             ha="center", va="center")
    axa.text(2.37, 0.36, "double\nroot", color="black", fontsize=LBL,
             ha="left", va="center")
    axa.text(1.0, 0.84, "gradient descent", color=GRAY, fontsize=LBL,
             ha="center", va="center")
    axa.text(1.0, 0.75, r"$|1-\eta\lambda|$", color=GRAY, fontsize=LBL,
             ha="center", va="center")
    axa.text(1.2, 1.03, "unstable", color="black", fontsize=LBL,
             ha="center", va="bottom")
    xr, yr = (0.0, 2.8), (0.0, 1.22)
    fl.axis_cross(axa, xr, yr, color="black")
    axa.text(xr[1], -0.04, r"$\eta\lambda$", color="black", fontsize=LBL + 1,
             ha="right", va="top")
    axa.text(0.07, yr[1], r"largest $|z|$", color="black",
             fontsize=LBL + 1, ha="left", va="top")
    fl.clean_axes(axa, lim=((-0.08, 2.86), (-0.2, 1.25)), hide=True,
                  equal=False)

    # ============================ (b) ============================ #
    t = np.linspace(0, 2 * np.pi, 400)
    axb.plot(np.cos(t), np.sin(t), color=LIGHT, lw=1.2, zorder=1)
    axb.plot(r * np.cos(t), r * np.sin(t), color=GREEN, lw=1.6, zorder=2)
    lams = (1.0, 2.0, 3.0, 5.0, 10.0)
    for lam in lams:
        z = _hb_roots(eta * lam, beta)
        if abs(z[0].imag) < 1e-6:                # double real root
            x0 = z.real.mean()
            axb.plot(x0, 0, "o", color=GREEN, ms=7.5, zorder=5)
            axb.plot(x0, 0, "o", mfc="none", mec=GREEN, mew=1.4, ms=13,
                     zorder=5)
        else:
            axb.plot(z.real, z.imag, "o", color=GREEN, ms=7.5, zorder=5)
            zu = z[np.argmax(z.imag)]            # label the upper root
            u = np.array([zu.real, zu.imag]) / abs(zu)
            axb.text(*(0.69 * u), f"{lam:.0f}", color=GREEN, fontsize=LBL,
                     ha="center", va="center")
    gd = 1 - (2 / 11) * np.array(lams)
    for v in gd:
        axb.plot([v, v], [-0.045, 0.045], color=ORANGE, lw=2.2, zorder=4)
    xr, yr = (-1.12, 1.18), (-1.1, 1.14)
    fl.axis_cross(axb, xr, yr, color="black")
    axb.text(xr[1], -0.05, r"Re $z$", color="black", fontsize=LBL,
             ha="right", va="top")
    axb.text(0.05, yr[1], r"Im $z$", color="black", fontsize=LBL,
             ha="left", va="top")
    axb.text(0.0, -1.2, r"heavy ball: radius $0.52$", color=GREEN,
             fontsize=LBL, ha="center", va="center")
    axb.text(0.0, -1.37, r"gradient descent: $[-0.82, 0.82]$",
             color=ORANGE, fontsize=LBL, ha="center", va="center")
    axb.annotate("10", xy=(-r, 0), xytext=(-0.72, 0.3),
                 color=GREEN, fontsize=LBL, ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8,
                                 shrinkA=1, shrinkB=7))
    axb.annotate("1", xy=(r, 0), xytext=(0.72, -0.3),
                 color=GREEN, fontsize=LBL, ha="center", va="top",
                 arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8,
                                 shrinkA=1, shrinkB=7))
    fl.clean_axes(axb, lim=((-1.15, 1.2), (-1.46, 1.16)), hide=True)

    fig.subplots_adjust(wspace=0.08)
    fl.save(fig, "mdl-opt-hb-eigenvalues")
    _copy_to_auto("mdl-opt-hb-eigenvalues")


# --------------------------------------------------------------------------- #
# 3. Newton walks into a saddle; a trust region walks out                   #
# --------------------------------------------------------------------------- #

def _f(v):
    return 0.5 * v[0] ** 2 + np.cos(v[1])


def _grad(v):
    return np.array([v[0], -np.sin(v[1])])


def _hess(v):
    return np.diag([1.0, -np.cos(v[1])])


def _tr_step(g, B, Delta):
    """Exact trust-region step by bisection on the multiplier tau."""
    eig_min = np.linalg.eigvalsh(B).min()
    if eig_min > 0:
        p = np.linalg.solve(B, -g)
        if np.linalg.norm(p) <= Delta:
            return p
    lo, hi = max(0.0, -eig_min), max(0.0, -eig_min) + 1e3
    for _ in range(200):
        tau = 0.5 * (lo + hi)
        p = np.linalg.solve(B + tau * np.eye(len(g)), -g)
        lo, hi = (tau, hi) if np.linalg.norm(p) > Delta else (lo, tau)
    return np.linalg.solve(B + hi * np.eye(len(g)), -g)


def _trust_region_path(x0, Delta0, steps):
    x, Delta, xs, radii = np.array(x0, float), Delta0, [], []
    for _ in range(steps):
        xs.append(x.copy())
        radii.append(Delta)
        g, B = _grad(x), _hess(x)
        p = _tr_step(g, B, Delta)
        pred = -(g @ p + 0.5 * p @ B @ p)
        r = (_f(x) - _f(x + p)) / pred
        if r > 0.1:
            x = x + p
        if r < 0.25:
            Delta /= 4
        elif r > 0.75 and np.linalg.norm(p) > 0.99 * Delta:
            Delta *= 2
    xs.append(x.copy())
    return np.array(xs), radii


def fig_trust_region_saddle():
    """(a) The quadratic model at (0.5, 0.3) in step coordinates p, with the
    ball |p| <= 0.5, the Newton point (the model's saddle), the trust-region
    step and the gradient step -g.  (b) f(x, y) = x^2/2 + cos y with the
    saddle (0, 0) and the minimum (0, pi), the Newton iterates and the
    trust-region iterates from (0.5, 0.3), the first two radii drawn faintly.
    """
    x0 = np.array([0.5, 0.3])
    g, B = _grad(x0), _hess(x0)
    Delta = 0.5
    p_tr = _tr_step(g, B, Delta)                 # (-0.19, 0.46)
    p_n = np.linalg.solve(B, -g)                 # (-0.5, -0.31)

    xa, ya = (-1.02, 0.72), (-0.62, 0.74)
    xb, yb = (-1.05, 1.05), (-0.7, 4.05)
    ra = (xa[1] - xa[0]) / (ya[1] - ya[0])
    rb = (xb[1] - xb[0]) / (yb[1] - yb[0])
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(9.6, 5.6), gridspec_kw=dict(width_ratios=[ra, rb]))

    # ============================ (a) ============================ #
    P1, P2 = np.meshgrid(np.linspace(*xa, 300), np.linspace(*ya, 300))
    M = g[0] * P1 + g[1] * P2 + 0.5 * (B[0, 0] * P1 ** 2 + B[1, 1] * P2 ** 2)
    axa.contour(P1, P2, M, levels=np.linspace(-0.6, 0.6, 13), colors=[LIGHT],
                linewidths=0.9, linestyles="solid", zorder=1)
    t = np.linspace(0, 2 * np.pi, 300)
    axa.plot(Delta * np.cos(t), Delta * np.sin(t), color=GREEN, lw=1.6,
             zorder=2)
    fl.axis_cross(axa, xa, ya, color="black")
    for p, col in ((p_n, ORANGE), (p_tr, GREEN), (-g, BLUE)):
        fl.arrow(axa, (0, 0), p, color=col, lw=2.0)
    axa.plot(*p_n, "o", color=ORANGE, ms=7, zorder=6)
    axa.plot(*p_tr, "o", color=GREEN, ms=7, zorder=6)
    axa.text(p_n[0] - 0.05, p_n[1] - 0.02, "Newton\n(model's saddle)",
             color=ORANGE, fontsize=LBL, ha="right", va="top", bbox=BOX,
             zorder=7)
    axa.text(p_tr[0] - 0.03, p_tr[1] + 0.07, "trust-region step",
             color=GREEN, fontsize=LBL, ha="center", va="bottom", bbox=BOX,
             zorder=7)
    axa.text(-g[0] - 0.02, -g[1] + 0.04, r"$-\mathbf{g}$", color=BLUE,
             fontsize=LBL + 1, ha="right", va="bottom", zorder=7)
    axa.text(0.37, -0.42, r"$\|\mathbf{p}\|=\Delta$", color=GREEN,
             fontsize=LBL, ha="left", va="center", bbox=BOX)
    axa.text(xa[1], -0.03, r"$p_1$", color="black", fontsize=LBL + 1,
             ha="right", va="top")
    axa.text(0.03, ya[1], r"$p_2$", color="black", fontsize=LBL + 1,
             ha="left", va="top")
    fl.clean_axes(axa, lim=(xa, ya), hide=True)

    # ============================ (b) ============================ #
    Xg, Yg = np.meshgrid(np.linspace(*xb, 300), np.linspace(*yb, 600))
    F = 0.5 * Xg ** 2 + np.cos(Yg)
    axb.contour(Xg, Yg, F, levels=np.linspace(-0.9, 1.5, 13), colors=[LIGHT],
                linewidths=0.9, linestyles="solid", zorder=1)
    fl.axis_cross(axb, xb, yb, color="black")
    # Newton iterates: (0.5, 0.3) -> (0, -0.0093) -> (0, 2.7e-7)
    xn = [x0]
    for _ in range(2):
        xn.append(xn[-1] - np.linalg.solve(_hess(xn[-1]), _grad(xn[-1])))
    xn = np.array(xn)
    axb.plot(xn[:, 0], xn[:, 1], "-o", color=ORANGE, ms=5, lw=1.8, zorder=4)
    path, radii = _trust_region_path(x0, Delta, 6)
    for c, rad in zip(path[:1], radii[:1]):
        axb.plot(c[0] + rad * np.cos(t), c[1] + rad * np.sin(t), color=GREEN,
                 lw=0.9, alpha=0.45, zorder=2)
    axb.plot(path[:, 0], path[:, 1], "-o", color=GREEN, ms=5, lw=1.8,
             zorder=5)
    axb.plot(*x0, "o", color="black", ms=6, zorder=6)
    axb.plot(0, 0, "o", mfc="white", mec="black", mew=1.6, ms=9, zorder=6)
    axb.plot(0, np.pi, "*", color="black", ms=14, zorder=7)
    axb.text(0.6, 0.3, r"start $(0.5, 0.3)$", color="black", fontsize=LBL,
             ha="left", va="center", bbox=BOX, zorder=8)
    axb.text(-0.1, -0.12, "saddle\n$f=1$", color="black", fontsize=LBL,
             ha="right", va="top", bbox=BOX, zorder=8)
    axb.text(-0.15, np.pi, "minimum\n$f=-1$", color="black", fontsize=LBL,
             ha="right", va="center", bbox=BOX, zorder=8)
    axb.text(0.12, -0.42, "Newton", color=ORANGE, fontsize=LBL, ha="left",
             va="center", bbox=BOX, zorder=8)
    axb.text(0.4, 1.9, "trust\nregion", color=GREEN, fontsize=LBL,
             ha="left", va="center", bbox=BOX, zorder=8)
    axb.text(xb[1], -0.06, r"$x$", color="black", fontsize=LBL + 1,
             ha="right", va="top")
    axb.text(-0.06, yb[1], r"$y$", color="black", fontsize=LBL + 1,
             ha="right", va="top")
    fl.clean_axes(axb, lim=(xb, yb), hide=True)

    fig.subplots_adjust(wspace=0.06)
    fl.save(fig, "mdl-opt-trust-region-saddle")
    _copy_to_auto("mdl-opt-trust-region-saddle")


FIGURES = [fig_descent_lemma_slice, fig_hb_eigenvalues,
           fig_trust_region_saddle]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
