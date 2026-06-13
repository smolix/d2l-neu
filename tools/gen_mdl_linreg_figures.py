#!/usr/bin/env python3
"""Generate the illustrative figures for the "Linear Neural Networks for
Regression" chapter (``chapter_linear-regression``) in the one shared house
style defined in ``gen_mdl_figures.py``.

The notebooks / prose reference the generated files with no drawing code (like
the slide SVGs).  Figures that show a computed result use real numerical
computation so the pictures are exact, not sketched.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_linreg_figures.py

All figures are written to ``img/mdl-linreg-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Circle, Polygon


def fig_ridge_geometry():
    """Weight decay as a constraint.  Elliptical squared-loss contours, centred
    on the unconstrained least-squares optimum w-hat, meet the constraint
    region: the L2 ball (ridge) is touched *tangentially*, off-axis, while the
    L1 diamond (lasso) is touched at a *corner*, driving a coordinate to zero
    --- the sparsity that distinguishes the two penalties."""
    bhat = np.array([2.6, 0.5])                # unconstrained least-squares optimum
    A = np.array([[1.0, 0.12], [0.12, 1.0]])   # loss curvature: L = (w-bhat)^T A (w-bhat)
    t = 1.2                                     # L2 radius / L1 budget

    def L(W):
        d = W - bhat
        return np.einsum("...i,ij,...j->...", d, A, d)

    # Exact constrained optima by dense boundary search (the figure must be true).
    th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    circle_pts = t * np.c_[np.cos(th), np.sin(th)]
    w_ridge = circle_pts[np.argmin(L(circle_pts))]
    verts = np.array([[t, 0], [0, t], [-t, 0], [0, -t]])
    edges = np.concatenate(
        [np.linspace(verts[i], verts[(i + 1) % 4], 1000, endpoint=False)
         for i in range(4)]
    )
    w_lasso = edges[np.argmin(L(edges))]

    # Loss contours via the eigendecomposition of A: {w : (w-bhat)^T A (w-bhat)=c}.
    evals, V = np.linalg.eigh(A)
    ang = np.linspace(0, 2 * np.pi, 400)
    u = np.c_[np.cos(ang), np.sin(ang)]

    def contour(c):
        return bhat + np.sqrt(c) * (u / np.sqrt(evals)) @ V.T

    print(f"  ridge  w*=({w_ridge[0]:.3f}, {w_ridge[1]:.3f})  "
          f"|w|_2={np.linalg.norm(w_ridge):.3f}")
    print(f"  lasso  w*=({w_lasso[0]:.3f}, {w_lasso[1]:.3f})  "
          f"-> w2={w_lasso[1]:.3f} (sparse when ~0)")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    specs = [
        (r"(a) ridge: $\|\mathbf{w}\|_2 \leq t$", w_ridge, "circle"),
        (r"(b) lasso: $\|\mathbf{w}\|_1 \leq t$", w_lasso, "diamond"),
    ]
    m_x, m_y = (-1.7, 3.5), (-2.0, 2.2)
    for ax, (title, wstar, shape) in zip(axes, specs):
        if shape == "circle":
            ax.add_patch(Circle((0, 0), t, fc=BLUE, ec=BLUE, alpha=0.12, lw=1.8))
        else:
            ax.add_patch(Polygon(verts, closed=True, fc=BLUE, ec=BLUE,
                                  alpha=0.12, lw=1.8))
        for c in (0.3, 1.2, 2.7):                 # faint background contours
            e = contour(c)
            ax.plot(e[:, 0], e[:, 1], color=LIGHT, lw=1.1)
        e = contour(L(wstar[None])[0])            # the contour that just touches the region
        ax.plot(e[:, 0], e[:, 1], color=ORANGE, lw=1.7)
        fl.axis_cross(ax, m_x, m_y, color=GRAY, lw=0.8)
        ax.plot(*bhat, "o", color=GRAY, ms=6)
        ax.text(bhat[0] + 0.08, bhat[1] + 0.30, r"$\hat{\mathbf{w}}$",
                color=GRAY, ha="center", va="center", fontsize=11)
        ax.plot(*wstar, "o", color=ORANGE, ms=7)
        ax.text(wstar[0] + 0.12, wstar[1] - 0.42, r"$\mathbf{w}^\star$",
                color=ORANGE, ha="center", va="center", fontsize=11)
        ax.text(m_x[1], -0.26, r"$w_1$", color=GRAY, ha="center",
                va="center", fontsize=10)
        ax.text(0.26, m_y[1], r"$w_2$", color=GRAY, ha="center",
                va="center", fontsize=10)
        ax.set_title(title, fontsize=11)
        fl.clean_axes(ax, lim=(m_x, m_y), hide=True)
    fig.subplots_adjust(wspace=0.10)
    fl.save(fig, "mdl-linreg-ridge-geometry")


def fig_oo_classes():
    """Orientation diagram for the OO design: the three base classes and how
    Trainer.fit drives a Module over data served by a DataModule."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    def classbox(cx, top, title, methods, color):
        w, lh = 3.7, 0.46
        h = 0.85 + lh * len(methods)
        x, y = cx - w / 2, top - h
        for fc, a in [(color, 0.10), ("none", 1.0)]:
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=1.8, edgecolor=color, facecolor=fc, alpha=a))
        ax.text(cx, top - 0.38, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=color)
        ax.plot([x + 0.3, x + w - 0.3], [top - 0.64, top - 0.64],
                color=color, lw=1.0)
        for i, m in enumerate(methods):
            ax.text(cx, top - 1.04 - i * lh, m, ha="center", va="center",
                    fontsize=10, family="monospace")
        return dict(l=x, r=x + w, mid=y + h / 2)

    mod = classbox(2.3, 4.7, "Module",
                   ["forward(X)", "loss(y_hat, y)",
                    "configure_optimizers()", "training_step(batch)"], BLUE)
    dm = classbox(12.7, 4.7, "DataModule",
                  ["train_dataloader()", "val_dataloader()"], GREEN)
    tr = classbox(7.5, 4.15, "Trainer", ["fit(model, data)"], ORANGE)

    y0 = tr["mid"]
    fl.arrow(ax, (tr["l"], y0), (mod["r"], y0), color=GRAY, lw=1.8)
    fl.arrow(ax, (tr["r"], y0), (dm["l"], y0), color=GRAY, lw=1.8)
    ax.text(7.5, 1.05,
            "New models subclass Module; new datasets subclass DataModule,\n"
            "often extended cell by cell with @add_to_class.",
            ha="center", va="center", fontsize=10)
    fl.save(fig, "mdl-linreg-oo-classes")


FIGURES = [fig_ridge_geometry, fig_oo_classes]


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks the Linear Algebra figures, which we don't run here).
    start = len(fl.WRITTEN)
    for fn in FIGURES:
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
