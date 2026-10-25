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

import figstyle.mpl as M
from figstyle import tokens as T


def fig_ridge_geometry():
    """Weight decay as a constraint.  Elliptical squared-loss contours, centred
    on the unconstrained least-squares optimum w-hat, meet the constraint
    region: the L2 ball (ridge) is touched where both coordinates are
    nonzero, while the L1 diamond (lasso) is touched at a *corner*, driving
    w2 to zero --- the sparsity that distinguishes the two penalties.  The
    contours are ellipses, elongated along the direction the data determine
    poorly (curvature ratio 4, long axis at 110 degrees); the regularized
    optimum is labelled w-hat_lambda (w* is reserved for the true weights)."""
    bhat = np.array([2.0, 1.2])                # unconstrained least-squares optimum
    th0 = np.deg2rad(110)                      # long axis of the ellipses
    R = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])
    A = R @ np.diag([0.25, 1.0]) @ R.T         # loss curvature: L = (w-bhat)^T A (w-bhat)
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

    print(f"  ridge  w=({w_ridge[0]:.3f}, {w_ridge[1]:.3f})")
    print(f"  lasso  w=({w_lasso[0]:.3f}, {w_lasso[1]:.3f})  (sparse when w2 ~ 0)")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    # the caption already names each panel ("Left: the l2 ball ... Right: the
    # l1 diamond ..."), so no redundant per-panel "(a)/(b)" title is drawn
    specs = [(w_ridge, "circle"), (w_lasso, "diamond")]
    m_x, m_y = (-1.5, 3.9), (-1.5, 3.9)         # same window: same scale
    for ax, (wopt, shape) in zip(axes, specs):
        if shape == "circle":
            ax.add_patch(Circle((0, 0), t, fc=T.BLUE.tint, ec=T.BLUE.base,
                                lw=1.8))
        else:
            ax.add_patch(Polygon(verts, closed=True, fc=T.BLUE.tint,
                                 ec=T.BLUE.base, lw=1.8))
        for c in (0.15, 0.6):                     # faint inner contours
            e = contour(c)
            ax.plot(e[:, 0], e[:, 1], color=LIGHT, lw=1.1)
        e = contour(L(wopt[None])[0])  # the contour that just touches the region
        ax.plot(e[:, 0], e[:, 1], color=ORANGE, lw=1.7)
        fl.axis_cross(ax, m_x, m_y, color=T.INK, lw=1.2)
        ax.plot(*bhat, "o", color=T.INK, ms=6)
        ax.text(bhat[0] + 0.32, bhat[1] + 0.10, r"$\hat{\mathbf{w}}$",
                color=T.INK, ha="center", va="center", fontsize=15)
        ax.plot(*wopt, "o", color=ORANGE, ms=7, zorder=5)
        # inside the constraint region, up-left of the dot: no contour or
        # axis crosses it there
        lab = wopt + (np.array([-0.62, 0.22]) if shape == "circle"
                      else np.array([-0.72, 0.26]))
        ax.text(lab[0], lab[1], r"$\hat{\mathbf{w}}_\lambda$",
                color=T.ORANGE.dark, ha="center", va="center", fontsize=15)
        ax.text(m_x[1], -0.28, r"$w_1$", color=T.INK, ha="center",
                va="center", fontsize=15)
        ax.text(0.30, m_y[1], r"$w_2$", color=T.INK, ha="center",
                va="center", fontsize=15)
        fl.clean_axes(ax, lim=(m_x, m_y), hide=True)
    fig.subplots_adjust(wspace=0.06)
    fl.save(fig, "mdl-linreg-ridge-geometry")


def fig_ridge_shrinkage():
    """Ridge regression multiplies the coordinate of the weights along an
    eigenvector of X^T X / n with eigenvalue mu by mu / (mu + lambda): a
    direction is kept while lambda << mu and suppressed once lambda >> mu,
    with the factor 1/2 at lambda = mu (dots).  Directions with mu = 0, which
    the training inputs do not see at all, are set to zero for every
    lambda > 0 (dashed)."""
    lam = np.logspace(-3, 4, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    # mu = 0 first and thin, so the curves' tails cover it where they merge
    ax.plot(lam, np.zeros_like(lam), color=T.MUTED, lw=T.SW_HAIR, ls="--",
            zorder=1)
    for mu, trio in zip((0.01, 1.0, 100.0), (T.BLUE, T.GREEN, T.ORANGE)):
        ax.plot(lam, mu / (mu + lam), color=trio.base, lw=T.SW_LINE, zorder=3)
        ax.plot([mu], [0.5], "o", color=trio.base, ms=6, zorder=4)
        # right of the dot and above the curve, which falls below 0.4 there
        ax.text(mu * 1.6, 0.56, rf"$\mu = {mu:g}$", color=trio.dark,
                fontsize=T.FS_SMALL, ha="left", va="bottom")
    ax.text(1.3e-3, 0.06, r"$\mu = 0$", color=T.MUTED, fontsize=T.FS_SMALL,
            ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$\lambda$", color=T.INK)
    ax.set_ylabel(r"$\mu / (\mu + \lambda)$", color=T.INK)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T.INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=T.INK)
    fl.save(fig, "mdl-linreg-ridge-shrinkage")


def fig_oo_classes():
    """The three base classes of the OO design and the contract between them:
    the Trainer takes minibatches from the DataModule and hands each to the
    Module, whose step returns the loss; the Trainer keeps the history.  The
    caption says that new models and datasets are subclasses, so the figure
    carries no note of its own."""
    from matplotlib.patches import FancyBboxPatch
    LH, MID = 0.46, 0.0          # member line height, common centre line
    W_MOD, W_TR, W_DM = 4.7, 4.5, 4.0   # each box clears its longest member
    GAP_L, GAP_R = 1.75, 2.05    # room for the arrow labels between boxes
    PT = 35.0                    # points per data unit: fixes the text/box ratio
    xr = (-0.1, W_MOD + GAP_L + W_TR + GAP_R + W_DM + 0.1)
    yr = (MID - 1.7, MID + 1.7)
    fig = plt.figure(figsize=((xr[1] - xr[0]) * PT / 72,
                              (yr[1] - yr[0]) * PT / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect("equal")
    ax.axis("off")

    def classbox(x, w, title, members, color):
        """A class box from x to x + w, centred on MID; returns x + w."""
        h = 0.85 + LH * len(members)
        cx, top = x + w / 2, MID + h / 2
        for fc, a in [(color, 0.10), ("none", 1.0)]:
            ax.add_patch(FancyBboxPatch(
                (x, top - h), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=1.8, edgecolor=color, facecolor=fc, alpha=a))
        ax.text(cx, top - 0.38, title, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=color)
        ax.plot([x + 0.3, x + w - 0.3], [top - 0.64, top - 0.64],
                color=color, lw=1.0)
        for i, m in enumerate(members):
            ax.text(cx, top - 1.04 - i * LH, m, ha="center", va="center",
                    fontsize=11, family="monospace", color=fl.INK)
        return x + w

    mod_r = classbox(0.0, W_MOD, "Module",
                     ["forward(X)", "loss(y_hat, y)", "training_step(batch)",
                      "validation_step(batch)", "configure_optimizers()"], BLUE)
    tr_l = mod_r + GAP_L
    tr_r = classbox(tr_l, W_TR, "Trainer",
                    ["fit(model, data)", "evaluate(model, data)",
                     "history", "board"], ORANGE)
    dm_l = tr_r + GAP_R
    classbox(dm_l, W_DM, "DataModule",
             ["train_dataloader()", "val_dataloader()"], GREEN)

    lab = dict(ha="center", fontsize=11.5, color=fl.INK)
    # DataModule -> Trainer: the loaders yield minibatches
    fl.arrow(ax, (dm_l, MID), (tr_r, MID), color=GRAY, lw=1.8)
    ax.text((tr_r + dm_l) / 2, MID + 0.16, "minibatches", va="bottom", **lab)
    # Trainer -> Module: one minibatch per step; Module -> Trainer: the loss
    xm = (mod_r + tr_l) / 2
    fl.arrow(ax, (tr_l, MID + 0.32), (mod_r, MID + 0.32), color=GRAY, lw=1.8)
    ax.text(xm, MID + 0.48, "minibatch", va="bottom", **lab)
    fl.arrow(ax, (mod_r, MID - 0.32), (tr_l, MID - 0.32), color=GRAY, lw=1.8)
    ax.text(xm, MID - 0.48, "loss", va="top", **lab)
    fl.save(fig, "mdl-linreg-oo-classes")


def fig_loss_menu():
    """Companion to the "menu of losses" recipe (Section 3.1).  Left: Gaussian
    vs. Laplace noise densities with *matched variance* (sigma = 1 vs. scale
    b = 1/sqrt(2)), with a log-scale inset exposing the tails -- an error at
    r = 4 is orders of magnitude less surprising under the Laplace model.
    Right: the per-error penalty each noise model induces via its negative
    log-likelihood -- squared error r^2/2, absolute error |r|, and the Huber
    compromise between them."""
    r = np.linspace(-5, 5, 1001)
    gauss = np.exp(-0.5 * r**2) / np.sqrt(2 * np.pi)           # N(0, 1)
    b = 1 / np.sqrt(2)                                          # Var = 2 b^2 = 1
    laplace = np.exp(-np.abs(r) / b) / (2 * b)

    delta = 1.0                                                 # Huber threshold
    sq = 0.5 * r**2
    abs_ = np.abs(r)
    huber = np.where(np.abs(r) <= delta, 0.5 * r**2,
                     delta * (np.abs(r) - 0.5 * delta))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

    # the caption already names each panel ("Left: ... densities ... Right:
    # the per-error penalties ..."), so no redundant per-panel "(a)/(b)"
    # title is drawn; the inset's "log scale" title stays -- it is the only
    # place that tells the reader this small axes uses a different y-scale.
    ax = axes[0]
    ax.plot(r, gauss, color=BLUE, label="Gaussian")
    ax.plot(r, laplace, color=ORANGE, label="Laplace")
    ax.set_xlabel(r"noise $\epsilon$")
    ax.set_ylabel(r"$p(\epsilon)$")
    ax.legend(loc="upper left", fontsize=13)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.75)
    # inset fonts match the main tick labels (>= 14 px on a slide at 64%
    # width); the inset sits right of x = 1.2 so that its tick labels clear
    # the two densities
    axins = ax.inset_axes([0.72, 0.40, 0.27, 0.52])
    axins.plot(r, gauss, color=BLUE, lw=1.6)
    axins.plot(r, laplace, color=ORANGE, lw=1.6)
    axins.set_yscale("log")
    axins.set_xlim(0, 5)
    axins.set_ylim(1e-6, 1)
    axins.set_yticks([1e-5, 1])
    axins.set_xticks([0, 5])
    axins.minorticks_off()
    axins.tick_params(labelsize=14, length=3, pad=3)
    axins.set_title("log scale", fontsize=T.FS_SMALL, pad=9)

    ax = axes[1]
    ax.plot(r, sq, color=BLUE, label=r"squared $\frac{1}{2}r^2$")
    ax.plot(r, abs_, color=ORANGE, label=r"absolute $|r|$")
    ax.plot(r, huber, color=GREEN, label=r"Huber ($\delta=1$)")
    ax.set_xlabel(r"error $r = \hat{y} - y$")
    ax.set_ylabel("penalty")
    ax.legend(loc="upper center", fontsize=13)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 6)

    fig.subplots_adjust(wspace=0.25)
    fl.save(fig, "mdl-linreg-loss-menu")


def fig_projection():
    """Least squares as projection.  The predictions Xw sweep the column space
    of X (a plane through the origin); the fitted values y-hat = X w-hat are
    the foot of the perpendicular from the label vector y, so the error
    r = y-hat - y meets the plane at a right angle, and any other prediction
    Xw is farther from y (the dotted hypotenuse of the right triangle
    y, y-hat, Xw).  A schematic in oblique
    projection: 3-D points (x right, d depth, z up) map to the page by P."""
    def P(p):
        x, d, z = p
        return np.array([x + 0.75 * d, z + 0.5 * d])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    corners = [(-0.6, -0.4, 0), (4.9, -0.4, 0), (4.9, 3.0, 0), (-0.6, 3.0, 0)]
    ax.add_patch(Polygon([P(c) for c in corners], closed=True,
                         fc=T.BLUE.tint, ec=T.BLUE.base, lw=1.4, zorder=1))
    origin = np.zeros(3)
    y_hat = np.array([2.0, 1.7, 0.0])
    y = y_hat + np.array([0.0, 0.0, 2.5])
    other = np.array([3.5, 0.8, 0.0])           # some other prediction Xw

    dotted = (0, (1.5, 2.5))
    for a, b in ((y, other), (y_hat, other)):
        ax.plot(*zip(P(a), P(b)), color=T.MUTED, lw=1.5, ls=dotted, zorder=2)
    ax.plot(*P(other), "o", color=T.MUTED, ms=5, zorder=4)
    ax.plot(*zip(P(y_hat), P(y)), color=T.RED.base, lw=2.5, zorder=3)
    fl.arrow(ax, P(origin), P(y_hat), color=T.GREEN.base, lw=2.5, mut=16)
    fl.arrow(ax, P(origin), P(y), color=T.BLUE.base, lw=2.5, mut=16)
    # right-angle mark at y-hat between the error (up) and the leg
    # y-hat -> Xw of the Pythagoras triangle y, y-hat, Xw used in the prose
    s, v = 0.4, np.array([0.0, 0.0, 1.0])
    u = (other - y_hat) / np.linalg.norm(other - y_hat)
    mark = [y_hat + s * u, y_hat + s * (u + v), y_hat + s * v]
    ax.plot(*zip(*[P(p) for p in mark]), color=T.INK, lw=1.3, zorder=4)
    ax.plot(*P(origin), "o", color=T.INK, ms=4.5, zorder=5)

    lab = dict(fontsize=17, ha="center", va="center")
    ax.text(*(P(y) + [0.1, 0.32]), r"$\mathbf{y}$", color=T.BLUE.dark, **lab)
    ax.text(*(P(y_hat) + [-1.0, -0.62]),
            r"$\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$",
            color=T.GREEN.dark, **lab)
    ax.text(*(P((y + y_hat) / 2) + [-0.3, 0.0]), r"$\mathbf{r}$",
            color=T.RED.dark, **lab)
    ax.text(*(P(other) + [0.52, -0.02]), r"$\mathbf{X}\mathbf{w}$",
            color=T.MUTED, **lab)
    ax.text(*(P(origin) + [-0.3, 0.04]), r"$\mathbf{0}$", color=T.INK, **lab)
    ax.text(*P((4.75, 2.2, 0)), "column space\n" r"of $\mathbf{X}$",
            color=T.BLUE.dark, fontsize=T.FS_SMALL, ha="right", va="center",
            multialignment="right", linespacing=1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-linreg-projection")


def fig_gd_sgd():
    """Gradient descent against minibatch SGD on a two-parameter
    least-squares problem (features with correlation 0.6, labels
    y = x^T w* + noise, w* = (2, -3.4) the chapter's running weights, noise
    sd 2, n = 1000, minibatches of 10).  GD converges to w-hat; SGD with the
    same learning rate ends in a cloud of fluctuations around it.  Quartering
    the learning rate (right) halves the width of the cloud: over a long run
    the root-mean-square distance from w-hat is 0.30 at eta = 0.2 and 0.14 at
    eta = 0.05, the sqrt(eta) scaling of the noise ball."""
    rng = np.random.default_rng(0)
    n, batch, sigma = 1000, 10, 2.0
    w_star = np.array([2.0, -3.4])
    X = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]], size=n)
    y = X @ w_star + sigma * rng.standard_normal(n)
    w_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    H = X.T @ X / n                              # Hessian of the mean loss / 2

    def run(eta, steps, b, seed):
        r = np.random.default_rng(seed)
        w = np.array(w0)
        path = [w.copy()]
        for _ in range(steps):
            idx = r.choice(n, b, replace=False) if b else np.arange(n)
            w = w - eta * X[idx].T @ (X[idx] @ w - y[idx]) / len(idx)
            path.append(w.copy())
        return np.array(path)

    w0 = (-0.5, 0.5)
    lim = ((-1.2, 3.6), (-4.9, 0.9))
    g1, g2 = np.meshgrid(np.linspace(*lim[0], 300), np.linspace(*lim[1], 300))
    D = np.stack([g1 - w_hat[0], g2 - w_hat[1]], -1)
    loss = 0.5 * np.einsum("...i,ij,...j->...", D, H, D)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.7), sharey=True)
    for ax, eta, steps, color, seed in [(axes[0], 0.2, 300, T.ORANGE.base, 1),
                                        (axes[1], 0.05, 1200, T.BLUE.base, 2)]:
        gd, sgd = run(eta, steps, None, 0), run(eta, steps, batch, seed)
        ax.contour(g1, g2, loss, levels=[0.1, 0.4, 1, 2, 3.5, 5.5],
                   colors=T.HAIRLINE, linewidths=1.2)
        ax.plot(sgd[:, 0], sgd[:, 1], color=color, lw=1.0, zorder=2)
        ax.plot(gd[:, 0], gd[:, 1], color=T.INK, lw=2.2, zorder=3)
        ax.plot(*w_hat, "o", color=T.INK, ms=4, zorder=5)
        ax.plot(*w0, "o", color=T.INK, ms=6, mfc="white", zorder=5)
        clear = dict(boxstyle="square,pad=0.15", fc="white", ec="none")
        ax.text(w_hat[0] + 0.75, w_hat[1] + 0.05, r"$\hat{\mathbf{w}}$",
                color=T.INK, fontsize=17, ha="center", va="center",
                bbox=clear, zorder=6)
        ax.text(-1.05, -4.2, "gradient descent", color=T.INK,
                fontsize=T.FS_SMALL, ha="left", va="center", bbox=clear)
        ax.text(-1.05, -4.62, "minibatch SGD", color=color,
                fontsize=T.FS_SMALL, ha="left", va="center", bbox=clear)
        ax.set_title(rf"$\eta = {eta}$", fontsize=T.FS_LABEL)
        ax.set_xlabel(r"$w_1$")
        ax.set_aspect("equal")
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
    axes[0].set_ylabel(r"$w_2$")
    fig.subplots_adjust(wspace=0.08)
    fl.save(fig, "mdl-linreg-gd-sgd")


def fig_house_prices():
    """The regression problem of Section 2.1, for the opening slide of its
    deck: recent sales scatter around a rising line in a plot of price
    against area, and the line is read off at the 150 m^2 house that has not
    been sold.  The line is the section's model at age 20,
    3,000 $/m^2 * area - 1,000 $/yr * 20 + 50,000 $ = 3 * area + 30 (in
    thousand dollars), so it predicts the 480,000 dollars of the prose.  The
    scatter is orthogonalized against (1, area), so the line is exactly the
    least-squares fit to the plotted sales."""
    rng = np.random.default_rng(5)
    area = rng.uniform(65, 235, 40)
    area = np.sort(area[np.abs(area - 150) > 15][:14])  # none sold at 150
    line = lambda a: 3.0 * a + 30.0
    A = np.stack([np.ones_like(area), area], 1)
    noise = 45.0 * rng.standard_normal(area.size)
    noise -= A @ np.linalg.lstsq(A, noise, rcond=None)[0]
    price = line(area) + noise

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    q, lo = 150.0, (50.0, 120.0)                 # the house; axis origin
    dash = dict(color=T.ORANGE.base, lw=1.8, ls=(0, (4, 3)), zorder=2)
    ax.plot([q, q], [lo[1], line(q)], **dash)
    ax.plot([lo[0], q], [line(q), line(q)], **dash)
    ax.plot([55, 245], [line(55), line(245)], color=T.INK, lw=2.4, zorder=3)
    ax.plot(area, price, "o", mfc="white", mec=T.BLUE.base, mew=2.0, ms=7,
            zorder=4)
    ax.plot(q, line(q), "o", color=T.ORANGE.base, ms=9, zorder=5)
    ax.text(q - 6, line(q) + 28, "?", color=T.ORANGE.dark, fontsize=22,
            fontweight="bold", ha="right", va="bottom")
    ax.set_xlim(lo[0], 250)
    ax.set_ylim(lo[1], 800)
    ax.set_xticks([50, 100, 150, 200, 250])
    ax.set_yticks([200, 400, 600, 800])
    ax.set_xlabel("area (m²)")
    ax.set_ylabel("price (thousand dollars)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fl.save(fig, "mdl-linreg-house-prices")


def fig_gaussian_noise():
    """The Gaussian noise model behind squared loss: labels scatter around the
    line wx + b, and at every x the conditional density p(y | x) is a normal
    bell centered on the line with the same width.  Same composition and
    axes as fit-linreg.svg (tools/figstyle/remake_ch03.py)."""
    line = lambda x: 0.52 * x + 0.9
    sigma = 0.45
    rng = np.random.default_rng(3)
    xs = np.sort(rng.uniform(0.3, 5.0, 24))
    ys = line(xs) + sigma * rng.standard_normal(xs.size)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    M.axis_cross(ax, (-0.25, 5.6), (-0.3, 4.6), color=T.INK)
    ax.plot([0.1, 4.95], [line(0.1), line(4.95)], color=T.INK, lw=2.4,
            zorder=3)
    ax.plot(xs, ys, "o", mfc="white", mec=T.BLUE.base, mew=1.8, ms=6,
            zorder=4)
    for x0 in (1.0, 2.6, 4.2):
        m = line(x0)
        yy = np.linspace(m - 3 * sigma, m + 3 * sigma, 200)
        bell = x0 + 0.75 * np.exp(-0.5 * ((yy - m) / sigma) ** 2)
        ax.fill_betweenx(yy, x0, bell, color=T.ORANGE.tint, zorder=1)
        ax.plot([x0, x0], [yy[0], yy[-1]], color=T.FAINT, lw=1.2, zorder=2)
        ax.plot(bell, yy, color=T.ORANGE.base, lw=2.0, zorder=2)
    ax.text(4.25, line(4.2) - 3 * sigma - 0.05, r"$p(y \mid x)$",
            color=T.ORANGE.dark, fontsize=16, ha="left", va="top")
    ax.text(5.05, line(4.95) + 0.05, r"$wx + b$", color=T.INK, fontsize=16,
            ha="left", va="center")
    ax.text(5.6, -0.45, r"$x$", color=T.INK, fontsize=17, ha="center")
    ax.text(-0.45, 4.5, r"$y$", color=T.INK, fontsize=17, va="center")
    M.clean_axes(ax, lim=((-0.6, 6.4), (-0.6, 4.7)), hide=True)
    fl.save(fig, "mdl-linreg-gaussian-noise")


FIGURES = [fig_ridge_geometry, fig_ridge_shrinkage, fig_oo_classes,
           fig_loss_menu, fig_projection, fig_gd_sgd, fig_house_prices,
           fig_gaussian_noise]


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
