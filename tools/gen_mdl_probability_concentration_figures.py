#!/usr/bin/env python3
"""Generate the illustrative figures of "Concentration Inequalities"
(``chapter_mdl-probability-statistics/mdl-concentration-generalization.md``)
in the chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_probability_concentration_figures.py

Writes ``img/mdl-prob-conc-majorants.svg`` and ``img/mdl-prob-conc-shell.svg``
(and the copies in ``img/auto/`` that the slides read).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

from scipy.special import gammaln  # noqa: E402

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.INK
LIGHT = fl.LIGHT


def _black_axes(ax):
    """Left/bottom spines in black, no top/right spines."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")
    ax.tick_params(colors="black", labelsize=13)


def _save_both(fig, name):
    fl.save(fig, name)
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def fig_majorants():
    """(a) The indicator 1{x >= t} at t = 2 and three functions above it:
    Markov x/t, Chebyshev (x/t)^2 and Chernoff e^{lambda (x - t)} with
    lambda = 1.5.  A faint bump centred at 1 shows where the mass of X sits;
    each bound is the expectation of its majorant, so what the majorant does
    under the bump decides the bound.  (b) The Chernoff exponent
    -lambda n t + n lambda^2 / 8 for the fair-coin mean at n = 100, t = 0.1:
    the parabola -10 lambda + 12.5 lambda^2 with minimum -2 at lambda = 0.4,
    and the valid but weaker value -0.875 at lambda = 0.1."""
    t, lam = 2.0, 1.5
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.3),
                                 gridspec_kw=dict(wspace=0.55))

    # (a) majorants
    xs = np.linspace(0, 2.45, 400)
    bump = 0.8 * np.exp(-(xs - 1.0) ** 2 / (2 * 0.35 ** 2))
    ax.fill_between(xs, 0, bump, color=fl.T.BLUE.tint, lw=0,
                    zorder=1)
    ax.text(1.0, 0.83, "mass of $X$", color=INK, fontsize=13,
            ha="center", va="bottom", zorder=2)
    ax.plot(xs, xs / t, color=BLUE, lw=2.4, zorder=3)
    ax.plot(xs, (xs / t) ** 2, color=ORANGE, lw=2.4, zorder=3)
    ax.plot(xs, np.exp(lam * (xs - t)), color=GREEN, lw=2.4, zorder=3)
    # the indicator: 0 left of t, 1 from t on
    ax.plot([0, t], [0, 0], color=INK, lw=3.4, solid_capstyle="butt",
            zorder=4)
    ax.plot([t, 2.45], [1, 1], color=INK, lw=3.4, solid_capstyle="butt",
            zorder=4)
    ax.plot([t], [0], "o", ms=7, mfc="white", mec=INK, mew=1.8, zorder=5,
            clip_on=False)
    ax.plot([t], [1], "o", ms=7, color=INK, zorder=5)
    ax.plot([t, t], [0, 2.05], ":", color=GRAY, lw=1.2, zorder=0)
    # labels at the right ends of the curves, outside the panel
    xe = 2.45
    for y, text, col in ((np.exp(lam * (xe - t)), r"Chernoff $e^{\lambda(x-t)}$",
                          GREEN),
                         ((xe / t) ** 2, r"Chebyshev $(x/t)^2$", ORANGE),
                         (xe / t, r"Markov $x/t$", BLUE),
                         (1.0, r"$\mathbf{1}\{x\geq t\}$", INK)):
        y_txt = y - 0.06 if col is INK else y
        ax.text(xe + 0.04, y_txt, text, color=col, fontsize=14, ha="left",
                va="center", clip_on=False)
    ax.set_xlim(0, 2.45)
    ax.set_ylim(0, 2.05)
    ax.set_xticks([0, 1, t])
    ax.set_xticklabels(["0", "1", r"$t$"])
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel(r"$x$", fontsize=15)
    _black_axes(ax)

    # (b) the Chernoff exponent for the fair coin, n = 100, t = 0.1
    n, tc = 100, 0.1
    ls = np.linspace(0, 0.9, 400)
    f = -ls * n * tc + n * ls ** 2 / 8
    keep = f <= 0.6
    bx.plot(ls[keep], f[keep], color=BLUE, lw=2.6, zorder=3)
    bx.axhline(0, color="black", lw=0.9, zorder=1)
    l_opt = 4 * tc
    f_opt = -l_opt * n * tc + n * l_opt ** 2 / 8
    l_1 = 0.1
    f_1 = -l_1 * n * tc + n * l_1 ** 2 / 8
    bx.plot([l_opt], [f_opt], "o", color=BLUE, ms=8, zorder=5)
    bx.plot([l_1], [f_1], "o", color=ORANGE, ms=8, zorder=5)
    bx.text(l_opt, f_opt - 0.1,
            r"$\lambda=4t$: bound $e^{-2}=0.14$", color=INK, fontsize=14,
            ha="center", va="top")
    bx.text(l_1 + 0.04, f_1 + 0.3,
            r"$\lambda=0.1$: $e^{-0.875}=0.42$", color=INK,
            fontsize=14, ha="left", va="center")
    bx.text(0.02, 0.1, r"exponent $0$: bound $1$", color=INK, fontsize=13,
            ha="left", va="bottom")
    bx.set_xlim(0, 0.9)
    bx.set_ylim(-2.45, 0.6)
    bx.set_xticks([0, 0.1, 0.4, 0.8])
    bx.set_xticklabels(["0", "0.1", "0.4", "0.8"])
    bx.set_yticks([-2, -1, 0])
    bx.set_xlabel(r"$\lambda$", fontsize=15)
    bx.set_ylabel("exponent", fontsize=14, labelpad=2)
    _black_axes(bx)

    for a, tag in ((ax, "(a)"), (bx, "(b)")):
        a.text(-0.02, 1.02, tag, transform=a.transAxes, fontsize=14,
               color=INK, ha="right", va="bottom")
    _save_both(fig, "mdl-prob-conc-majorants")


def _chi_pdf(r, d):
    """Density of the norm of a standard Gaussian in R^d (chi with d dof)."""
    with np.errstate(divide="ignore"):
        logf = ((d - 1) * np.log(r) - r ** 2 / 2
                - (d / 2 - 1) * np.log(2) - gammaln(d / 2))
    return np.exp(logf)


def fig_shell():
    """(a) Mode versus mass at d = 20: the density factor e^{-r^2/2} falls,
    the shell-volume factor r^{d-1} grows, and their product, the density of
    the norm, peaks at sqrt(d - 1).  Each curve is scaled to maximum 1 on
    [0, 8].  (b) The exact density of ||x|| for d = 2, 8, 32, 128, 512: the
    centres move out like sqrt(d), the widths stay at about 1/sqrt(2)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.8, 4.3),
                                 gridspec_kw=dict(wspace=0.22,
                                                  width_ratios=[1, 1.6]))
    d = 20
    rs = np.linspace(0, 8, 500)
    dens = np.exp(-rs ** 2 / 2)
    vol = (rs / 8) ** (d - 1)
    prod = _chi_pdf(rs, d)
    prod = prod / prod.max()
    ax.plot(rs, dens, color=BLUE, lw=2.4)
    ax.plot(rs, vol, color=ORANGE, lw=2.4)
    ax.plot(rs, prod, color=GREEN, lw=3.4)
    rm = np.sqrt(d - 1)
    ax.plot([rm, rm], [0, 1.0], ":", color=GRAY, lw=1.2, zorder=0)
    ax.text(0.25, 1.06, r"density $e^{-r^2/2}$", color=BLUE, fontsize=14,
            ha="left", va="bottom")
    ax.text(7.75, 1.06, r"shell volume $r^{d-1}$", color=ORANGE,
            fontsize=14, ha="right", va="bottom")
    ax.text(rm + 0.95, 0.80, "product", color=GREEN, fontsize=14,
            ha="left", va="center")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.22)
    ax.set_xticks([0, 2, rm, 6, 8])
    ax.set_xticklabels(["0", "2", r"$\sqrt{d-1}$", "6", "8"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"radius $r$ ($d=20$)", fontsize=15)
    _black_axes(ax)

    rr = np.linspace(1e-4, 25, 2000)
    for dd in (2, 8, 32, 128, 512):
        p = _chi_pdf(rr, dd)
        lw = 3.4 if dd == 512 else 2.2
        bx.plot(rr, p, color=BLUE, lw=lw)
        i = int(np.argmax(p))
        # centred above each peak; the d = 2 label is raised to clear d = 8's
        if dd == 2:
            bx.text(0.45, p[i] + 0.09, rf"$d={dd}$", color=INK, fontsize=14,
                    ha="left", va="bottom")
        else:
            bx.text(rr[i], p[i] + 0.025, rf"$d={dd}$", color=INK,
                    fontsize=14, ha="center", va="bottom")
    bx.set_xlim(0, 25)
    bx.set_ylim(0, 0.8)
    bx.set_xticks([0, 5, 10, 15, 20, 25])
    bx.set_yticks([0, 0.2, 0.4, 0.6])
    bx.set_xlabel(r"norm $\|\mathbf{x}\|$", fontsize=15)
    bx.set_ylabel("density", fontsize=14)
    _black_axes(bx)

    for a, tag in ((ax, "(a)"), (bx, "(b)")):
        a.text(-0.02, 1.02, tag, transform=a.transAxes, fontsize=14,
               color=INK, ha="right", va="bottom")
    _save_both(fig, "mdl-prob-conc-shell")


def fig_slerp_vs_lerp():
    """Interpolating between two latent codes z1, z2 ~ N(0, I_d).  (a) The
    plane through 0, z1 and z2, drawn to scale: both codes sit on the shell
    of radius sqrt(d) (shaded band) and are nearly orthogonal, so the chord
    (lerp) passes through the interior, its midpoint at radius about
    sqrt(d/2), while the arc (slerp) stays on the shell.  (b) The norm along
    each path, divided by sqrt(d), for one seeded pair at d = 512: the lerp
    dips to sqrt(1/2) = 0.71 at gamma = 1/2; the slerp stays near 1."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                                 gridspec_kw=dict(wspace=0.18,
                                                  width_ratios=[1, 1.25]))
    # (a) schematic in the plane of the two codes, radius sqrt(d) -> 1
    a1, a2 = np.deg2rad(100.0), np.deg2rad(10.0)
    z1 = np.array([np.cos(a1), np.sin(a1)])
    z2 = np.array([np.cos(a2), np.sin(a2)])
    th = np.linspace(np.deg2rad(-12), np.deg2rad(122), 300)
    for r0, r1 in ((0.93, 1.07),):
        ax.fill(np.r_[r1 * np.cos(th), r0 * np.cos(th[::-1])],
                np.r_[r1 * np.sin(th), r0 * np.sin(th[::-1])],
                color=BLUE, alpha=0.12, lw=0)
    arc = np.linspace(a2, a1, 200)
    ax.plot(np.cos(arc), np.sin(arc), color=GREEN, lw=3.4, zorder=3)
    ax.plot([z1[0], z2[0]], [z1[1], z2[1]], "--", color=ORANGE, lw=2.4,
            zorder=3)
    mid = (z1 + z2) / 2
    ax.plot([0, mid[0]], [0, mid[1]], ":", color=GRAY, lw=1.2, zorder=1)
    ax.plot([0, z2[0]], [0, z2[1]], ":", color=GRAY, lw=1.2, zorder=1)
    for pt in (z1, z2):
        ax.plot(*pt, "o", ms=8, color=BLUE, zorder=5)
    ax.plot(*mid, "o", ms=7, color=ORANGE, zorder=5)
    ax.plot(0, 0, "o", ms=6, color=INK, zorder=5)
    ax.text(z1[0] - 0.06, z1[1] + 0.06, r"$\mathbf{z}_1$", fontsize=15,
            color=INK, ha="right", va="bottom")
    ax.text(z2[0] + 0.07, z2[1], r"$\mathbf{z}_2$", fontsize=15, color=INK,
            ha="left", va="center")
    ax.text(0.0, -0.06, r"$\mathbf{0}$", fontsize=14, color=INK,
            ha="center", va="top")
    lp = z1 + 0.74 * (z2 - z1)
    ax.text(lp[0] - 0.05, lp[1] - 0.05, "lerp", color=fl.T.ORANGE.dark,
            fontsize=14, ha="right", va="top")
    ax.text(0.86, 0.86, "slerp", color=fl.T.GREEN.dark, fontsize=14,
            ha="left", va="bottom")
    ax.text(0.66, 0.02, r"$\sqrt{d}$", color=INK, fontsize=14,
            ha="center", va="top")
    ax.text(0.14, 0.40, r"$\approx\!\sqrt{d/2}$", color=INK, fontsize=14,
            ha="right", va="center")
    fl.clean_axes(ax, lim=((-0.45, 1.3), (-0.3, 1.25)), hide=True)

    # (b) norms along the two paths for one seeded pair at d = 512
    d = 512
    rng = np.random.default_rng(0)
    u, v = rng.standard_normal(d), rng.standard_normal(d)
    om = np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
    g = np.linspace(0, 1, 201)[:, None]
    lerp = (1 - g) * u + g * v
    slerp = (np.sin((1 - g) * om) * u + np.sin(g * om) * v) / np.sin(om)
    sd = np.sqrt(d)
    gg = g[:, 0]
    bx.plot(gg, np.linalg.norm(slerp, axis=1) / sd, color=GREEN, lw=2.6)
    bx.plot(gg, np.linalg.norm(lerp, axis=1) / sd, "--", color=ORANGE,
            lw=2.4)
    bx.axhline(np.sqrt(0.5), color=LIGHT, lw=1.0, ls=":", zorder=0)
    bx.text(0.5, np.linalg.norm(slerp[100]) / sd + 0.03, "slerp",
            color=fl.T.GREEN.dark, fontsize=14, ha="center", va="bottom")
    bx.text(0.5, np.linalg.norm(lerp[100]) / sd + 0.03, "lerp",
            color=fl.T.ORANGE.dark, fontsize=14, ha="center", va="bottom")
    bx.set_xlim(0, 1)
    bx.set_ylim(0.6, 1.12)
    bx.set_xticks([0, 0.5, 1])
    bx.set_yticks([np.sqrt(0.5), 1.0])
    bx.set_yticklabels([r"$\sqrt{1/2}$", "$1$"])
    bx.set_xlabel(r"interpolation weight $\gamma$ ($d=512$)", fontsize=14)
    bx.set_ylabel(r"norm $/\sqrt{d}$", fontsize=14)
    _black_axes(bx)
    ax.text(-0.45, 1.25 + 0.02 * 1.55, "(a)", fontsize=14, color=INK,
            ha="left", va="bottom")
    bx.text(-0.02, 1.02, "(b)", transform=bx.transAxes, fontsize=14,
            color=INK, ha="right", va="bottom")
    fl.save(fig, "mdl-prob-conc-slerp-vs-lerp")


FIGURES = [fig_majorants, fig_shell, fig_slerp_vs_lerp]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
