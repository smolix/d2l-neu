#!/usr/bin/env python3
"""Generate the new illustrative figures of "Constrained Optimization"
(``chapter_mdl-optimization/mdl-constrained-optimization-duality.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's other two figures (``mdl-opt-lagrange-tangency`` and
``mdl-opt-kkt-active-set``) stay in ``gen_mdl_optimization_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_optimization_constrained_figures.py

Writes ``img/mdl-opt-projection-inequality.svg`` and
``img/mdl-opt-simplex-threshold.svg`` (and the copies in ``img/auto/`` that
the slides read).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY

LBL = 13          # the chapter's in-figure label size


def _copy_to_auto(name):
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def fig_projection_inequality():
    """Projection onto the unit disk C, the section's running example.

    (a) The variational inequality: y = (1.5, 2) projects to (0.6, 0.8); the
        residual y - Pi(y) makes an obtuse angle with the direction to any
        point x of C (here x = (-0.5, 0.3), angle about 151 degrees), and C lies
        on the far side of the supporting line 3 x1 + 4 x2 = 5.
    (b) Nonexpansiveness: x = (1.5, 2) and x' = (2.2, -0.6) project radially
        to u = (0.6, 0.8) and v = x'/||x'||; ||u - v|| = 1.07 < ||x - x'|| = 2.69.
    Both panels share one window and equal aspect.
    """
    from matplotlib.patches import Arc

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.0, 4.9))
    lim = ((-1.35, 2.65), (-1.3, 2.45))
    t = np.linspace(0, 2 * np.pi, 400)

    def disk(ax):
        ax.add_patch(plt.Circle((0, 0), 1.0, facecolor=BLUE, alpha=0.12,
                                lw=0, zorder=1))
        ax.plot(np.cos(t), np.sin(t), color=BLUE, lw=2.0, zorder=3)
        ax.text(-0.35, -0.45, "$C$", color=BLUE, fontsize=LBL + 3,
                ha="center", va="center")

    # ======================= (a) the obtuse angle ======================= #
    disk(axa)
    y = np.array([1.5, 2.0])
    p = y / np.linalg.norm(y)                  # (0.6, 0.8)
    x = np.array([-0.5, 0.3])
    # supporting line 3 x1 + 4 x2 = 5 through p, dashed blue
    s = np.linspace(-1.15, 1.15, 2)
    tang = np.array([-0.8, 0.6])
    seg = p[None, :] + s[:, None] * tang[None, :]
    axa.plot(seg[:, 0], seg[:, 1], color=BLUE, lw=1.6, ls=(0, (5, 3)),
             zorder=2)
    axa.text(*(p + 1.12 * tang + np.array([-0.02, 0.10])), "supporting line",
             color=BLUE, fontsize=LBL, ha="center", va="bottom")
    # residual y - p (orange) and the direction to a feasible point (green)
    fl.arrow(axa, p, y, color=ORANGE, lw=2.2)
    fl.arrow(axa, p, x, color=GREEN, lw=2.2)
    # angle between them, drawn the short way round (about 151 degrees)
    a0 = np.degrees(np.arctan2(*(y - p)[::-1]))
    a1 = np.degrees(np.arctan2(*(x - p)[::-1])) % 360
    axa.add_patch(Arc(p, 0.6, 0.6, theta1=a0, theta2=a1, color="black",
                      lw=1.1, zorder=4))
    # between the residual and the supporting line, clear of both
    axa.text(*(p + 0.47 * np.array([np.cos(np.radians(96)),
                                    np.sin(np.radians(96))])),
             "≥ 90°", color="black", fontsize=LBL + 1, ha="center",
             va="center", bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                    ec="none"), zorder=5)
    for q in (y, p, x):
        axa.plot(*q, "o", color="black", ms=6, zorder=6)
    axa.text(y[0] + 0.10, y[1], r"$\mathbf{y}$", color="black",
             fontsize=LBL + 1, ha="left", va="center")
    axa.text(*(y - p) * 0.5 + p + np.array([0.12, -0.12]),
             r"$\mathbf{y}-\Pi_C(\mathbf{y})$", color=ORANGE, fontsize=LBL,
             ha="left", va="center")
    axa.text(p[0] - 0.06, p[1] - 0.26, r"$\Pi_C(\mathbf{y})$",
             color="black", fontsize=LBL, ha="center", va="top")
    axa.text(x[0] - 0.08, x[1] - 0.12, r"$\mathbf{x}$", color="black",
             fontsize=LBL + 1, ha="right", va="top")
    fl.clean_axes(axa, lim=lim, hide=True)

    # ===================== (b) nonexpansiveness ===================== #
    disk(axb)
    xa = np.array([1.5, 2.0])
    xb = np.array([2.2, -0.6])
    u, v = xa / np.linalg.norm(xa), xb / np.linalg.norm(xb)
    for q, w in ((xa, u), (xb, v)):
        axb.plot([q[0], w[0]], [q[1], w[1]], color=GRAY, lw=1.0, zorder=2)
    axb.plot([xa[0], xb[0]], [xa[1], xb[1]], color=GRAY, lw=1.8,
             ls=(0, (5, 3)), zorder=2)
    axb.plot([u[0], v[0]], [u[1], v[1]], color=BLUE, lw=2.4, zorder=4)
    for q in (xa, xb, u, v):
        axb.plot(*q, "o", color="black", ms=6, zorder=6)
    axb.text(xa[0] + 0.10, xa[1], r"$\mathbf{x}$", color="black",
             fontsize=LBL + 1, ha="left", va="center")
    axb.text(xb[0] + 0.10, xb[1], r"$\mathbf{x}'$", color="black",
             fontsize=LBL + 1, ha="left", va="center")
    axb.text(u[0] - 0.14, u[1] + 0.16, r"$\mathbf{u}$", color="black",
             fontsize=LBL + 1, ha="right", va="bottom")
    axb.text(v[0] + 0.12, v[1] - 0.14, r"$\mathbf{v}$", color="black",
             fontsize=LBL + 1, ha="left", va="top")
    mid = 0.5 * (xa + xb)
    axb.text(mid[0] + 0.14, mid[1] + 0.10, r"$\|\mathbf{x}-\mathbf{x}'\|$",
             color="black", fontsize=LBL, ha="left", va="center")
    mid = 0.5 * (u + v)
    axb.text(mid[0] - 0.12, mid[1] - 0.02, r"$\|\mathbf{u}-\mathbf{v}\|$",
             color=BLUE, fontsize=LBL, ha="right", va="center",
             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    fl.clean_axes(axb, lim=lim, hide=True)

    fig.subplots_adjust(wspace=0.06)
    fl.save(fig, "mdl-opt-projection-inequality")
    _copy_to_auto("mdl-opt-projection-inequality")


def fig_simplex_threshold():
    """Projection of y = (0.9, 0.5, -0.2) onto the simplex, drawn like the
    water-filling figure of the Duality section (same bar width and gap,
    dashed orange level, baseline, hidden axes).

    (a) Sparsemax: every score is shifted down by tau = 0.2 and clipped at
        zero; the blue parts above the level are x* = (0.7, 0.3, 0).
    (b) Softmax of the same scores: (0.50, 0.33, 0.17), nowhere zero.
    The threshold is computed by the section's sort-and-threshold rule, and
    the softmax values by exp/sum, so the figure matches the text.
    """
    yv = np.array([0.9, 0.5, -0.2])
    u = np.sort(yv)[::-1]
    tau_k = (np.cumsum(u) - 1) / np.arange(1, 4)
    tau = tau_k[np.nonzero(u > tau_k)[0][-1]]          # 0.2
    xs = np.maximum(yv - tau, 0.0)
    soft = np.exp(yv) / np.exp(yv).sum()

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    width, gap = 0.84, 0.16
    ylim = (-0.5, 1.08)

    # ======================= (a) sparsemax ======================= #
    for i, (s, x) in enumerate(zip(yv, xs)):
        x0 = i + gap / 2
        # grey up to the level (or down to a negative score), blue above it
        axa.add_patch(plt.Rectangle((x0, 0), width, min(s, tau),
                                    facecolor=GRAY, alpha=0.45, lw=0))
        axa.plot([x0, x0 + width], [s, s], color=GRAY, lw=1.6)
        if x > 0:                              # the part above the level
            axa.add_patch(plt.Rectangle((x0, tau), width, x,
                                        facecolor=BLUE, alpha=0.35, lw=0))
            axa.text(i + 0.5, tau + 0.5 * x, f"${x:.1f}$", color="black",
                     fontsize=LBL + 1, ha="center", va="center")
        else:
            axa.text(i + 0.5, tau + 0.05, "$0$", color="black",
                     fontsize=LBL + 1, ha="center", va="bottom")
        lab_y = min(s, 0.0) - 0.06
        axa.text(i + 0.5, lab_y, rf"$y_{{{i + 1}}}={s:g}$", color="black",
                 fontsize=LBL, ha="center", va="top")
    axa.plot([0.0, 3.0], [tau, tau], color=ORANGE, lw=1.7, ls=(0, (5, 3)),
             zorder=4)
    axa.text(3.06, tau, rf"$\tau={tau:.1f}$", color=ORANGE, fontsize=LBL,
             ha="left", va="center")
    axa.plot([0.0, 3.0], [0.0, 0.0], color="black", lw=1.2)
    axa.text(3.06, 0.62, "blue parts\nsum to $1$", color=BLUE,
             fontsize=LBL, ha="left", va="center")
    fl.clean_axes(axa, lim=((-0.05, 4.1), ylim), hide=True, equal=False)

    # ==================== (b) softmax, same scores ==================== #
    for i, s in enumerate(soft):
        x0 = i + gap / 2
        axb.add_patch(plt.Rectangle((x0, 0), width, s, facecolor=BLUE,
                                    alpha=0.35, lw=0))
        axb.plot([x0, x0 + width], [s, s], color=BLUE, lw=1.6)
        axb.text(i + 0.5, s + 0.04, f"${s:.2f}$", color="black",
                 fontsize=LBL + 1, ha="center", va="bottom")
        axb.text(i + 0.5, -0.06, rf"$y_{{{i + 1}}}={yv[i]:g}$",
                 color="black", fontsize=LBL, ha="center", va="top")
    axb.plot([0.0, 3.0], [0.0, 0.0], color="black", lw=1.2)
    axb.text(3.06, 0.62, "all three\npositive", color=BLUE, fontsize=LBL,
             ha="left", va="center")
    fl.clean_axes(axb, lim=((-0.05, 4.1), ylim), hide=True, equal=False)

    fig.subplots_adjust(wspace=0.08)
    fl.save(fig, "mdl-opt-simplex-threshold")
    _copy_to_auto("mdl-opt-simplex-threshold")


FIGURES = [fig_projection_inequality, fig_simplex_threshold]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
