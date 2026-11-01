#!/usr/bin/env python3
"""Generate the illustrative figures for the "Multilayer Perceptrons ->
Generalization in Deep Learning" section (``chapter_multilayer-perceptrons/
generalization-deep.md``) in the one shared house style.

The section measures double descent in its notebook; the figures here show the
two implicit-bias results it states, with no drawing code in the notebook:

* ``mdl-mlp-min-norm``: gradient descent on least squares, started at zero,
  moves only along the rows of the design matrix and stops at the
  minimum-norm exact fit; a start with a component along the line of exact
  fits ends at a different, longer fit.
* ``mdl-mlp-max-margin``: gradient descent on the logistic loss for separable
  data; the direction of the weights turns toward the maximum-margin
  separator, slowly (the angle shrinks like 1/log t).

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` + ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette and the
drawing helpers.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_doubledescent_figures.py

The figures are written to ``img/mdl-mlp-min-norm.svg`` and
``img/mdl-mlp-max-margin.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


def fig_min_norm():
    """Gradient descent on least squares picks the minimum-norm exact fit.

    One equation h^T w = y in two unknowns: the exact fits form a line.  The
    gradient h (h^T w - y) is always a multiple of h, so every step moves
    perpendicular to that line.  Started at w = 0, the iterates walk along h and
    stop at the foot of the perpendicular, the point of the line closest to
    the origin (the dashed circle through it touches the line there).  Started
    at w0, they walk parallel to h and stop at a different, longer exact
    fit: the component of w0 along the line is never changed.
    """
    h = np.array([1.0, 2.0])
    y = 4.0
    w_star = y * h / (h @ h)                        # (0.8, 1.6)
    w0 = np.array([-1.9, 0.4])
    w0_inf = w0 + (y - h @ w0) / (h @ h) * h         # where GD from w0 stops
    eta = 0.45 / (h @ h)                            # each step: 45% of the way
    print(f"  |w*| = {np.linalg.norm(w_star):.2f}, "
          f"|w_inf from w0| = {np.linalg.norm(w0_inf):.2f}")

    def iterates(w, steps=7):
        out = [w]
        for _ in range(steps):
            w = w - eta * h * (h @ w - y)
            out.append(w)
        return np.array(out)

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    fl.axis_cross(ax, (-2.3, 4.3), (-0.5, 3.2), color=fl.T.INK, lw=1.2)
    ax.text(4.3, -0.1, r"$w_1$", fontsize=17, ha="right", va="top")
    ax.text(0.1, 3.2, r"$w_2$", fontsize=17, ha="left", va="top")

    # the line of exact fits h^T w = y
    d = np.array([-h[1], h[0]]) / np.linalg.norm(h)          # along the line
    seg = np.array([w_star + s * d for s in (-3.35, 1.7)])
    ax.plot(seg[:, 0], seg[:, 1], color=GREEN, lw=3.0, zorder=2)
    ax.text(2.55, 0.95, "exact fits\n" r"$\mathbf{h}^\top\mathbf{w} = y$",
            color=fl.T.GREEN.dark, fontsize=15, ha="left", va="bottom",
            linespacing=0.95)

    # an arc of the circle of radius |w*|: it touches the line at w*
    r = np.linalg.norm(w_star)
    a = np.radians(np.linspace(-15, 175, 300))
    ax.plot(r * np.cos(a), r * np.sin(a), ls="--", color=fl.T.FAINT, lw=1.4,
            zorder=1)
    fl.right_angle(ax, w_star, -h, d, size=0.2, color=fl.T.INK)

    # gradient descent from zero (blue) and from w0 (orange)
    for start, color in ((np.zeros(2), BLUE), (w0, ORANGE)):
        path = iterates(start)
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.6, zorder=3)
        ax.plot(path[:-1, 0], path[:-1, 1], "o", color=color, ms=5, zorder=4)
    ax.plot(*w_star, "o", color=fl.T.BLUE.dark, ms=8, zorder=5)
    ax.plot(*w0_inf, "o", color=fl.T.ORANGE.dark, ms=8, zorder=5)

    ax.text(0.12, -0.1, r"$\mathbf{0}$", fontsize=16, ha="left", va="top")
    ax.text(w_star[0] + 0.25, w_star[1] + 0.1, "minimum-norm\nfit",
            color=fl.T.BLUE.dark, fontsize=15, ha="left", va="bottom",
            linespacing=0.9)
    ax.text(w0[0] - 0.15, w0[1], r"$\mathbf{w}_0$",
            color=fl.T.ORANGE.dark, fontsize=16, ha="right", va="center")
    ax.text(w0_inf[0] + 0.05, w0_inf[1] + 0.2, "a longer fit",
            color=fl.T.ORANGE.dark, fontsize=15, ha="right", va="bottom")

    fl.clean_axes(ax, lim=((-2.9, 4.4), (-0.6, 3.3)), hide=True)
    fl.save(fig, "mdl-mlp-min-norm")


def _max_margin_direction(X, y):
    """Unit w maximizing min_i y_i w^T x_i, by search over angles (2-D)."""
    th = np.linspace(0, 2 * np.pi, 200001)
    W = np.stack([np.cos(th), np.sin(th)])
    margins = (y[:, None] * (X @ W)).min(axis=0)
    k = int(np.argmax(margins))
    return W[:, k], margins[k]


def fig_max_margin():
    """Logistic gradient descent turns toward the maximum-margin separator.

    (a) Separable 2-D data, separators through the origin.  Decision lines of
    gradient descent on the mean logistic loss after t = 1, 10^2, 10^4 steps
    (gray to blue) approach the maximum-margin line (heavy black), whose
    margin band is dashed; the support vectors are ringed.  Every point is
    classified correctly from the first step on.  (b) The angle between w_t
    and the maximum-margin direction against t on a log axis: it keeps
    shrinking long after the training error is zero, but slowly.
    """
    rng = np.random.default_rng(5)                  # fixed: an illustration
    pos = rng.normal([1.3, 1.1], 0.4, (10, 2))
    neg = rng.normal([-1.2, -0.9], 0.4, (10, 2))
    pos = np.vstack([pos, [[1.2, -0.45]]])          # one near the boundary
    neg = np.vstack([neg, [[-0.8, 0.75]]])          # and one on the other side
    c, s_ = np.cos(np.radians(-35)), np.sin(np.radians(-35))
    rot = np.array([[c, -s_], [s_, c]])             # tilt the separator
    pos, neg = pos @ rot.T, neg @ rot.T
    X = np.vstack([pos, neg])
    y = np.r_[np.ones(len(pos)), -np.ones(len(neg))]
    u, gamma = _max_margin_direction(X, y)
    sv = np.abs(y * (X @ u) - gamma) < 1e-3
    assert gamma > 0 and set(y[sv]) == {-1.0, 1.0}

    T_max = 10 ** 5
    snap = (1, 100, 10 ** 4)
    ts = np.unique(np.logspace(0, 5, 80).astype(int))
    w = np.zeros(2)
    ws, angles = {}, []
    angle = lambda v: np.degrees(np.arccos(np.clip(v @ u / np.linalg.norm(v),
                                                   -1, 1)))
    for t in range(1, T_max + 1):
        # one gradient step (step size 1) on the mean logistic loss
        w = w + (X * (y / (1 + np.exp(y * (X @ w))))[:, None]).mean(axis=0)
        assert t > 1 or (np.sign(X @ w) == y).all()
        if t in snap:
            ws[t] = w.copy()
        if t in ts:
            angles.append(angle(w))
    print("  angle to max-margin (deg): "
          + ", ".join(f"t={t}: {angle(v):.1f}" for t, v in ws.items())
          + f", t={T_max}: {angles[-1]:.1f}")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.0),
                                   gridspec_kw=dict(width_ratios=[1.2, 1.0],
                                                 wspace=0.38))
    perp = lambda v: np.array([-v[1], v[0]]) / np.linalg.norm(v)
    up = lambda v: v if v[1] >= 0 else -v           # the end pointing up
    colors = {1: fl.T.FAINT, 100: "#6FA6D2", 10 ** 4: fl.T.BLUE.base}
    for t, wt in ws.items():                        # gradient descent
        p = up(perp(wt)) * 1.95
        axa.plot([-p[0], p[0]], [-p[1], p[1]], color=colors[t], lw=2.2,
                 zorder=2)
    p = up(perp(u)) * 2.45                          # maximum margin
    axa.plot([-p[0], p[0]], [-p[1], p[1]], color=fl.T.INK, lw=3.2, zorder=3)
    axa.text(p[0] + 0.08, p[1] + 0.05, "max margin", fontsize=14,
             ha="left", va="bottom", color=fl.T.INK)
    for s in (+1, -1):
        off = s * gamma * u
        axa.plot([off[0] - p[0], off[0] + p[0]], [off[1] - p[1], off[1] + p[1]],
                 ls="--", color=fl.T.INK, lw=1.3, zorder=3)
    axa.plot(0, 0, "o", color=fl.T.INK, ms=5, zorder=7)
    axa.scatter(pos[:, 0], pos[:, 1], s=36, color=BLUE, zorder=5)
    axa.scatter(neg[:, 0], neg[:, 1], s=36, marker="s", color=ORANGE, zorder=5)
    axa.scatter(X[sv, 0], X[sv, 1], s=180, facecolors="none",
                edgecolors=fl.T.INK, lw=1.4, zorder=6)
    fl.clean_axes(axa, lim=((-2.45, 3.2), (-2.25, 2.2)), hide=True)

    axb.plot(ts[:len(angles)], angles, color=fl.T.MUTED, lw=2.0, zorder=2)
    for t, wt in ws.items():                        # the snapshots of (a)
        axb.plot(t, angle(wt), "o", color=colors[t], ms=10, zorder=3,
                 markeredgecolor=fl.T.INK, markeredgewidth=0.8)
        label = "$t = 1$" if t == 1 else f"$t = 10^{int(np.log10(t))}$"
        axb.text(t * 1.6, angle(wt) + 1.2, label, fontsize=14, ha="left",
                 va="bottom", color=fl.T.INK)
    axb.set_xscale("log")
    axb.set_xlabel("gradient steps $t$", fontsize=15, color=fl.T.INK)
    axb.set_ylabel("angle to the max-margin\ndirection (degrees)",
                   fontsize=15, color=fl.T.INK, linespacing=0.95)
    axb.set_ylim(0, 40)
    axb.set_xlim(0.7, T_max)
    axb.set_xticks([10 ** k for k in range(6)])
    for side in ("left", "bottom"):
        axb.spines[side].set_color(fl.T.INK)
    axb.tick_params(colors=fl.T.INK, labelsize=13)

    # centre the scene of (a) on the whole of (b), tick labels and axis
    # label included, rather than on (b)'s plotting box alone; otherwise the
    # scene floats high above an empty band at the bottom left
    fig.canvas.draw()
    tb = axb.get_tightbbox().transformed(fig.transFigure.inverted())
    pa = axa.get_position()
    axa.set_position([pa.x0, tb.y0, pa.width, tb.height])

    fl.save(fig, "mdl-mlp-max-margin")


FIGURES = [fig_min_norm, fig_max_margin]


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks the Linear Algebra figures, which we don't run here).
    # Optional arguments name the figures to draw (e.g. fig_min_norm).
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        if len(sys.argv) > 1 and fn.__name__ not in sys.argv[1:]:
            continue
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert os.path.exists(p), f"missing: {p}"
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):32s} {size:>8,d} bytes")
    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
