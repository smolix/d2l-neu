#!/usr/bin/env python3
"""Generate the illustrative figures for the Generative Adversarial Networks
chapter (ch. 16) in the one shared house style defined in ``gen_mdl_figures.py``.

Four figures (see ``reviews/gan-implementation-brief.md`` §8):

  * ``mdl-gan-architecture`` -- the adversarial loop: a latent draw through the
    generator, the generated and real batches meeting at the discriminator, and
    the two updates with their opposite goals (replaces the legacy hand-drawn
    ``img/gan.svg``, now deleted);
  * ``mdl-gan-template`` -- the two knobs of the adversarial-objective template
    ``d(p,q) = sup_T {E_p[a(T)] - E_q[b(T)]}``, with GAN/f-GAN, W_1 and MMD
    placed on the (scoring functions, critic class) plane and RpGAN lifted off it;
  * ``mdl-gan-pairing`` -- single-sample classification against a fixed
    threshold versus pair ranking, with the additive shift flipping the
    verdict on the left and cancelling on the right;
  * ``mdl-gan-exits`` -- the three exits from unstable minimax training, each
    with its named steps and its cost.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_gan_figures.py

or via ``make figures`` (picked up by the ``gen_mdl_*_figures.py`` glob).  All
figures are written to ``img/mdl-gan-<id>.svg``.  The generator is
byte-idempotent: seeded RNGs only, no timestamps (``fl.save`` fixes the SVG
hash salt and nulls the date).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
PURPLE = "#9467bd"

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


# --------------------------------------------------------------------------- #
# Local helpers                                                               #
# --------------------------------------------------------------------------- #

def _box(ax, cx, cy, w, h, text, color, fontsize=13.5, ls="-", lw=1.7,
         fc="white", tc="black", zorder=3):
    """A rounded box (faint fill, coloured edge) with centred text."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        fc=fc, ec=color, lw=lw, linestyle=ls, zorder=zorder))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                color=tc, zorder=zorder + 1)


def _curved(ax, p, q, rad, color, lw=1.8, ls="-", mut=16, zorder=4):
    """A curved arrow between two points."""
    ax.add_patch(FancyArrowPatch(
        p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
        mutation_scale=mut, color=color, lw=lw, linestyle=ls,
        shrinkA=0, shrinkB=0, zorder=zorder))


def _batch(ax, cx, cy, w, h, pts, color, fontsize=12.5, label=None,
           label_below=True, ms=4.2):
    """A sample box: rounded frame with a small scatter of samples inside.

    ``pts`` are box-local coordinates, clipped to the inner margin so no dot
    ever touches the frame.
    """
    _box(ax, cx, cy, w, h, "", color, fc="white", lw=1.7)
    mx, my = w / 2 - 0.22, h / 2 - 0.22
    p = np.clip(np.asarray(pts, float), [-mx, -my], [mx, my])
    ax.plot(cx + p[:, 0], cy + p[:, 1], "o", color=color, ms=ms, mec="none",
            alpha=0.85, zorder=5)
    if label:
        dy = -h / 2 - 0.22 if label_below else h / 2 + 0.22
        ax.text(cx, cy + dy, label, ha="center",
                va="top" if label_below else "bottom", fontsize=fontsize,
                color="black")


# =========================================================================== #
# 16.1  The adversarial loop                                                  #
# =========================================================================== #

def fig_architecture():
    """The game of :numref:`fig_gan`: a latent draw z through the generator G
    makes a batch of generated samples; the discriminator D sees it alongside a
    batch of real data and scores every sample; the two updates push in
    opposite directions."""
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    rng = np.random.default_rng(0)

    y_g, y_r = 2.30, 5.35                       # generated lane / real lane

    def _network(cx, cy, w, sym, name, color):
        """A network: its symbol above its name, both inside one box (so the
        feedback path can leave the box without crossing a caption)."""
        _box(ax, cx, cy, w, 1.42, "", color, lw=1.9)
        ax.text(cx, cy + 0.26, sym, ha="center", va="center", fontsize=17,
                color="black", zorder=4)
        ax.text(cx, cy - 0.34, name, ha="center", va="center", fontsize=12,
                color="black", zorder=4)

    # --- latent draw -> generator -> generated batch ------------------------
    z = rng.normal(0.0, 0.34, (16, 2)) * np.array([1.15, 0.95])
    _batch(ax, 1.45, y_g, 2.0, 1.7, z, GRAY, label=r"latent draw $z$")
    fl.arrow(ax, (2.50, y_g), (3.42, y_g), color="black", lw=1.6)
    _network(4.65, y_g, 2.2, r"$G$", "generator", ORANGE)
    fl.arrow(ax, (5.80, y_g), (6.85, y_g), color="black", lw=1.6)
    gen = rng.normal(0.0, 1.0, (24, 2)) * np.array([0.46, 0.33])
    _batch(ax, 8.10, y_g, 2.4, 2.0, gen, ORANGE,
           label=r"generated batch $G(z)$")

    # --- real batch ---------------------------------------------------------
    c = np.array([[-0.52, 0.30], [0.50, -0.28]])
    real = (c[rng.integers(0, 2, 24)]
            + rng.normal(0.0, 1.0, (24, 2)) * np.array([0.19, 0.14]))
    _batch(ax, 8.10, y_r, 2.4, 2.0, real, BLUE, label_below=False,
           label=r"real batch $x \sim p_{\mathrm{data}}$")

    # --- both batches into the discriminator --------------------------------
    _curved(ax, (9.35, y_r), (10.66, 4.28), 0.12, "black", lw=1.6)
    _curved(ax, (9.35, y_g), (10.66, 3.62), -0.12, "black", lw=1.6)
    _network(12.00, 3.95, 2.62, r"$D$", "discriminator", GREEN)

    # --- the score it produces ----------------------------------------------
    xs = 14.35
    # the output arrow ends where it meets the logit axis's 0 tick
    fl.arrow(ax, (13.35, 3.95), (xs - 0.17, 3.95), color="black", lw=1.6)
    fl.arrow(ax, (xs, 2.55), (xs, 5.42), color="black", lw=1.4, mut=12)
    ax.plot([xs - 0.16, xs + 0.16], [3.98, 3.98], color="black", lw=1.3)
    ax.text(xs + 0.24, 3.98, r"$0$", ha="left", va="center", fontsize=12.5,
            color="black")
    ax.plot([xs], [4.92], "o", color=BLUE, ms=8, zorder=5)
    ax.text(xs + 0.24, 4.92, "real", ha="left", va="center", fontsize=12.5,
            color=BLUE)
    ax.plot([xs], [3.02], "o", color=ORANGE, ms=8, zorder=5)
    ax.text(xs + 0.24, 3.02, "generated", ha="left", va="center", fontsize=12.5,
            color=ORANGE)
    # the axis name sits below the axis, clear of the D-update label above
    ax.text(xs, 2.40, "realness logit", ha="center", va="top", fontsize=12.5,
            color="black")

    # --- the two updates, pushing opposite ways -----------------------------
    _curved(ax, (11.52, 4.68), (12.48, 4.68), -1.05, GREEN, lw=1.8, ls="--")
    ax.text(12.00, 5.40, "update $D$: raise scores on real,\n"
                         "lower them on generated",
            ha="center", va="bottom", fontsize=12.5, color=GREEN)
    # the G-update path runs below the batch captions, its label beneath it
    yb = 0.10
    ax.plot([12.90, 12.90], [3.24, yb], ls="--", color=ORANGE, lw=1.8)
    ax.plot([12.90, 4.65], [yb, yb], ls="--", color=ORANGE, lw=1.8)
    fl.arrow(ax, (4.65, yb), (4.65, 1.55), color=ORANGE, lw=1.8, ls="--")
    ax.text(8.78, yb - 0.16, r"update $G$: gradient flows back through $D$",
            ha="center", va="top", fontsize=12.5, color=ORANGE)

    fl.clean_axes(ax, lim=((0.20, 16.70), (-0.75, 7.05)), hide=True)
    fl.save(fig, "mdl-gan-architecture")


# =========================================================================== #
# 16.2  The template's two knobs                                              #
# =========================================================================== #

def fig_template():
    """One template, two choices: which scoring functions (a, b) and which critic class.
    GAN/f-GAN, W_1 and MMD sit at three corners of that plane; RpGAN does not
    sit on it at all, because its objective is quadratic in (p, q)."""
    fig, ax = plt.subplots(figsize=(9.6, 6.0))

    SH = np.array([0.44, 0.40])                  # shear: one unit of the scoring-function axis
    U1, V1 = 7.4, 4.9                            # plane extent

    def P(u, v):
        return np.array([u + SH[0] * v, SH[1] * v])

    # the plane itself
    ax.add_patch(Polygon([P(0, 0), P(U1, 0), P(U1, V1), P(0, V1)],
                         closed=True, fc=LIGHT, alpha=0.22, ec=GRAY, lw=1.2,
                         zorder=1))
    # the linear-scoring row: every integral probability metric lives here
    ax.add_patch(Polygon([P(0, 0.35), P(U1, 0.35), P(U1, 1.75), P(0, 1.75)],
                         closed=True, fc=GRAY, alpha=0.13, ec="none", zorder=2))
    ax.text(*(P(U1, 1.05) + np.array([0.22, 0.0])),
            "integral probability\nmetrics: $a(T)=b(T)=T$", ha="left",
            va="center", fontsize=12.5, color="black")

    # the two axes
    fl.arrow(ax, P(0, 0), P(U1 + 0.55, 0), color="black", lw=1.5, mut=15)
    fl.arrow(ax, P(0, 0), P(0, V1 + 0.45), color="black", lw=1.5, mut=15)
    ax.text(*(P(U1 / 2, 0) + np.array([0.0, -0.62])), r"critic class $\mathcal{T}$",
            ha="center", va="top", fontsize=14.5, color="black")
    ax.text(*(P(0, V1 + 0.55) + np.array([-0.28, 0.10])), r"scoring functions $(a,b)$",
            ha="right", va="bottom", fontsize=14.5, color="black")

    # ticks along each axis
    for u, name in [(1.15, "unrestricted"), (4.05, "Lipschitz ball"),
                    (6.55, "RKHS ball")]:
        ax.plot(*np.stack([P(u, 0), P(u, 0) - np.array([0.0, 0.16])]).T,
                color="black", lw=1.2)
        ax.text(*(P(u, 0) + np.array([0.0, -0.28])), name, ha="center",
                va="top", fontsize=12.5, color="black")
    for v, name in [(1.05, "linear"), (3.85, "nonlinear")]:
        ax.plot(*np.stack([P(0, v), P(0, v) - np.array([0.16, 0.0])]).T,
                color="black", lw=1.2)
        ax.text(*(P(0, v) + np.array([-0.28, 0.0])), name, ha="right",
                va="center", fontsize=12.5, color="black")

    # the three template members
    members = [
        (1.15, 3.85, BLUE, "GAN, $f$-GAN ($f$-divergences)", (0.32, -0.02),
         "left", "center"),
        (4.05, 1.05, GREEN, "$W_1$", (0.0, 0.36), "center", "bottom"),
        (6.55, 1.05, ORANGE, "MMD", (0.0, 0.36), "center", "bottom"),
    ]
    for u, v, col, name, off, ha, va in members:
        p = P(u, v)
        ax.plot(*p, "o", color=col, ms=11, zorder=6)
        ax.text(p[0] + off[0], p[1] + off[1], name, ha=ha, va=va,
                fontsize=13, color=col, zorder=7,
                bbox=dict(fc="white", ec="none", pad=1.6, alpha=0.85))

    # RpGAN: the same two knobs as the GAN, lifted clean off the plane -- its
    # critic scores a pair, so the objective is not of the template's form.
    foot = P(1.15, 3.85)
    lifted = foot + np.array([0.0, 2.15])
    ax.plot([foot[0], lifted[0]], [foot[1] + 0.14, lifted[1]], ls="--",
            color=PURPLE, lw=1.6, zorder=5)
    ax.plot(*lifted, "o", color=PURPLE, ms=11, zorder=6)
    ax.text(lifted[0] + 0.32, lifted[1],
            "relativistic pairing: the same two knobs,"
            "\nbut the critic scores a pair, and the objective"
            "\nis quadratic in $(p,q)$ — off the plane",
            ha="left", va="center", fontsize=13, color=PURPLE, zorder=7)

    fl.clean_axes(ax, lim=((-0.7, 11.4), (-0.95, 4.4)), hide=True, equal=True)
    fl.save(fig, "mdl-gan-template")


# =========================================================================== #
# 16.3  One sample versus a pair                                              #
# =========================================================================== #

def fig_pairing():
    """Left: a single sample scored against the fixed threshold 0 -- adding a
    constant b to the critic flips the verdict.  Right: a pair, judged by the
    difference of the two scores -- the same constant cancels."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.8, 4.4))

    C = 1.5                                   # the additive shift
    DX, DY = 0.45, -0.95                      # the two critic values
    XL, XR = 2.20, 5.00                       # unshifted / shifted column
    XA = 0.95                                 # the shared score axis
    LIM = ((-0.15, 7.05), (-2.45, 2.55))

    def frame(ax, title):
        fl.arrow(ax, (XA, -1.95), (XA, 2.25), color="black", lw=1.4, mut=12)
        ax.text(0.26, 0.30, "critic score", ha="center", va="center",
                fontsize=13, color="black", rotation=90)
        ax.plot([XA - 0.13, XA + 0.13], [0.0, 0.0], color="black", lw=1.3)
        ax.text(XA - 0.20, 0.0, r"$0$", ha="right", va="center", fontsize=13,
                color="black")
        # both columns hold the same samples; only the critic differs
        ax.text((XL + XR) / 2, -1.62, r"every score $+\,b$", ha="center",
                va="center", fontsize=12.5, color="black")
        for x, lab in [(XL, r"$D$"), (XR, r"$D+b$")]:
            ax.text(x, -2.02, lab, ha="center", va="top", fontsize=14,
                    color="black")
        ax.text((XL + XR) / 2, 2.50, title, ha="center", va="top",
                fontsize=13.5, color="black")
        fl.clean_axes(ax, lim=LIM, hide=True, equal=False)

    # --- (a) one sample against a fixed threshold: the verdict flips --------
    axa.plot([XA, 6.95], [0.0, 0.0], ls="--", color="black", lw=1.2)
    axa.text(6.95, 0.14, "threshold", ha="right", va="bottom", fontsize=12.5,
             color="black")
    for x, y, lab, verdict in [(XL, -0.70, r"$D(x)$", "generated"),
                               (XR, -0.70 + C, r"$D(x)+b$", "real")]:
        # one sample of unknown origin: a neutral colour, not the data blue
        axa.plot([x], [y], "o", color=GRAY, ms=11, zorder=5)
        axa.text(x - 0.24, y, lab, ha="right", va="center", fontsize=13.5,
                 color="black")
        axa.text(x + 0.24, y, f"$\\to$ {verdict}", ha="left", va="center",
                 fontsize=13, color="black")
    frame(axa, "one sample: is $D(x)$ above $0$?")

    # --- (b) a pair: the difference, and the same difference after the shift -
    for k, (x, sh) in enumerate([(XL, 0.0), (XR, C)]):
        for y, col, lab in [(DX + sh, BLUE, r"$D(x)$"),
                            (DY + sh, ORANGE, r"$D(x')$")]:
            axb.plot([x], [y], "o", color=col, ms=11, zorder=5)
            if not k:                       # the header names the shift; the
                axb.text(x - 0.24, y, lab, ha="right", va="center",
                         fontsize=13.5, color=col)   # second column needs no
        xb = x + 0.26                                # relabelling
        axb.plot([xb, xb], [DY + sh, DX + sh], color="black", lw=1.5)
        for y in (DY + sh, DX + sh):
            axb.plot([xb - 0.10, xb + 0.10], [y, y], color="black", lw=1.5)
        axb.text(xb + 0.18, (DX + DY) / 2 + sh,
                 r"$D(x)-D(x')$" if not k else "unchanged",
                 ha="left", va="center", fontsize=13, color="black")
    frame(axb, "a pair: is $D(x)$ above $D(x')$?")

    fl.save(fig, "mdl-gan-pairing")


# =========================================================================== #
# 16.6  Three exits from instability                                          #
# =========================================================================== #

def fig_exits():
    """The de-heuristicization map: regularize the dynamics, constrain the
    critic, or remove the game -- each branch with its named steps and its
    cost.  (That the adversarial term returns at distillation is an
    observation, not a step, so it lives in the caption.)"""
    fig, ax = plt.subplots(figsize=(12.6, 5.4))

    branches = [
        (BLUE, "1. regularize the dynamics",
         ["zero-centered\n$R_1$, $R_2$", "convergent\nobjective (RpGAN)",
          "R3GAN"],
         "cost: one more knob ($\\gamma$) and a minimax loop that remains"),
        (ORANGE, "2. constrain the critic",
         ["spectral norm", "frozen pretrained\nfeatures",
          "controlled\naugmentation (ADA)"],
         "cost: the critic is no longer free and inherits the features' bias"),
        (GREEN, "3. remove the game",
         ["MMD, IMLE", "diffusion and flow\nas regression"],
         "cost: a fixed discrepancy weakens in high dimension; slow sampling"),
    ]

    xs = [4.60, 8.65, 12.70]                     # the three step columns
    ys = [5.25, 3.10, 0.95]                      # the three branch lanes
    bw, bh = 3.34, 0.90

    _box(ax, 1.05, ys[1], 1.80, 1.35, "unstable\nminimax\ntraining", GRAY,
         fontsize=13.5, lw=1.9)

    # a short trunk leaves the source box; each branch turns off it with a
    # rounded corner and arrives horizontally at its lane
    xt = 2.38                                   # x of the branching trunk
    ax.plot([1.99, xt], [ys[1], ys[1]], color=GRAY, lw=1.7, zorder=4,
            solid_capstyle="butt")
    for (col, name, steps, cost), y in zip(branches, ys):
        ax.add_patch(FancyArrowPatch(
            (xt, ys[1]), (xs[0] - bw / 2 - 0.10, y),
            connectionstyle=("arc3,rad=0" if y == ys[1] else
                             "angle,angleA=90,angleB=0,rad=14"),
            arrowstyle="-|>", mutation_scale=15, color=col, lw=1.7,
            shrinkA=0, shrinkB=0, zorder=4))
        ax.text(xs[0] - bw / 2, y + bh / 2 + 0.16, name, ha="left",
                va="bottom", fontsize=13.5, color=col)
        for x, s in zip(xs, steps):
            _box(ax, x, y, bw, bh, s, col, fontsize=12.5, fc="white")
        for xa, xb in zip(xs[:len(steps) - 1], xs[1:len(steps)]):
            fl.arrow(ax, (xa + bw / 2 + 0.08, y), (xb - bw / 2 - 0.08, y),
                     color=col, lw=1.5, mut=13)
        ax.text(xs[0] - bw / 2, y - bh / 2 - 0.18, cost, ha="left", va="top",
                fontsize=12.5, color="black")

    fl.clean_axes(ax, lim=((0.0, 14.55), (-0.05, 6.35)), hide=True, equal=False)
    fl.save(fig, "mdl-gan-exits")


# =========================================================================== #
# 16.1  Saturating and non-saturating weights                                 #
# =========================================================================== #

def _sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def fig_weights():
    """The two generator weights of eq_gan_weights against the critic score
    D(x') on a log axis: w_sat = sigma(D) dies exponentially for a rejected
    sample, w_ns = sigma(-D) stays near one; their ratio is exactly e^{-D}."""
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    d = np.linspace(-7.0, 4.0, 401)
    w_sat, w_ns = _sigmoid(d), _sigmoid(-d)
    ax.semilogy(d, w_ns, color=GREEN, lw=2.5, zorder=3)
    ax.semilogy(d, w_sat, color=PURPLE, lw=2.5, zorder=3)

    for d0 in (-3.0, -5.0):
        lo, hi = _sigmoid(d0), _sigmoid(-d0)
        ax.plot([d0, d0], [lo, hi], color="black", lw=1.4, zorder=2)
        for y in (lo, hi):
            ax.plot([d0 - 0.12, d0 + 0.12], [y, y], color="black", lw=1.4,
                    zorder=2)
        ax.plot([d0], [hi], "o", color=GREEN, ms=7.5, zorder=4)
        ax.plot([d0], [lo], "o", color=PURPLE, ms=7.5, zorder=4)
        ratio = hi / lo                      # = e^{-d0}, computed, not typed
        ax.text(d0 + 0.18, 0.42 if d0 == -3.0 else 0.075,
                f"$e^{{{-d0:.0f}}} \\approx {ratio:.0f}$",
                ha="left", va="center", fontsize=14, color="black",
                bbox=dict(fc="white", ec="none", pad=1.0))

    ax.text(-2.75, 1.22, r"$w_{\mathrm{ns}} = \sigma(-D)$", ha="left",
            va="bottom", fontsize=15, color=GREEN)
    ax.text(-4.55, 0.0024, r"$w_{\mathrm{sat}} = \sigma(D)$", ha="left",
            va="top", fontsize=15, color=PURPLE)

    ax.set_xlim(-7.2, 4.2)
    ax.set_ylim(5e-4, 3.6)
    ax.set_xlabel(r"critic score $D(x')$ of a generated sample", fontsize=14)
    ax.set_ylabel("weight on the sample's gradient", fontsize=14)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
    ax.tick_params(colors="black", labelsize=13)
    fl.save(fig, "mdl-gan-weights")


# =========================================================================== #
# 16.2  Optimal critics on separated supports                                 #
# =========================================================================== #

def fig_critics_separated():
    """Two densities with disjoint supports (data p left, generated q right)
    and the optimal critic of three objectives.  JS: log(p/q) is +inf on p's
    support and -inf on q's, flat on both, so the generator gets no slope.
    W_1: a slope-one ramp.  MMD: the witness E_p k(x,.) - E_q k(x,.), a smooth
    difference of kernel bumps whose slope at q still points towards p.  The
    orange arrow at q is the critic's slope there (the generator's push)."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), sharey=True)
    plt.subplots_adjust(wspace=0.08)

    delta, s = 4.0, 2.0                  # separation, kernel bandwidth
    x = np.linspace(-2.2, delta + 2.2, 1201)

    def bump(c):                         # a smooth density on [c-1, c+1]
        u = np.clip(x - c, -1.0, 1.0)
        f = (1.0 - u ** 2) ** 2
        return f / (f.sum() * (x[1] - x[0]))

    p, q = bump(0.0), bump(delta)
    dens_scale = 1.15 / p.max()          # densities drawn 1.15 units tall
    dx = x[1] - x[0]
    K = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * s ** 2))
    witness = K @ (p - q) * dx
    witness *= 2.0 / np.abs(witness).max()   # display scale is arbitrary
    YL = (-2.55, 3.15)

    def base(ax, title):
        ax.fill_between(x, 0, p * dens_scale, color=BLUE, alpha=0.22, lw=0,
                        zorder=1)
        ax.fill_between(x, 0, q * dens_scale, color=ORANGE, alpha=0.22, lw=0,
                        zorder=1)
        ax.plot(x, p * dens_scale, color=BLUE, lw=1.6, zorder=2)
        ax.plot(x, q * dens_scale, color=ORANGE, lw=1.6, zorder=2)
        fl.arrow(ax, (x[0], 0), (x[-1] + 0.15, 0), color="black", lw=1.3,
                 mut=11)
        ax.text(x[-1] + 0.15, -0.22, r"$x$", ha="right", va="top",
                fontsize=14, color="black")
        ax.text(-1.15, 0.55, r"$p$", ha="right", va="center", fontsize=15,
                color=BLUE)
        ax.text(delta + 1.15, 0.55, r"$q$", ha="left", va="center",
                fontsize=15, color=ORANGE)
        ax.text(x[0], YL[1], title, ha="left", va="top", fontsize=14,
                color="black")
        fl.clean_axes(ax, lim=((x[0] - 0.1, x[-1] + 0.25), YL), hide=True,
                      equal=False)

    def push(ax, slope, y=1.55):
        """The generator's push at q: an arrow whose length is the drawn
        critic's slope there (so W_1 and MMD compare on the page)."""
        if abs(slope) < 1e-9:
            ax.text(delta, y, "no slope", ha="center", va="center",
                    fontsize=13, color="black")
        else:
            L = 1.6 * abs(slope)
            fl.arrow(ax, (delta + L / 2, y), (delta - L / 2, y),
                     color=ORANGE, lw=2.2, mut=14)

    # --- JS: log-ratio critic, +inf on supp p, -inf on supp q --------------
    ax = axes[0]
    base(ax, r"JS: $\log(p/q)$")
    top, bot = 1.75, -1.75
    for c, yv, sgn in [(0.0, top, 1), (delta, bot, -1)]:
        ax.plot([c - 1, c + 1], [yv, yv], color=GREEN, lw=2.6, zorder=3)
        fl.arrow(ax, (c, yv), (c, yv + 0.5 * sgn), color=GREEN, lw=1.8,
                 mut=12)
        ax.text(c + 0.2, yv + 0.32 * sgn, r"$+\infty$" if sgn > 0 else
                r"$-\infty$", ha="left", va="center", fontsize=14,
                color="black")
    push(ax, 0.0)

    # --- W_1: the slope-one ramp -------------------------------------------
    ax = axes[1]
    base(ax, r"$W_1$: 1-Lipschitz critic")
    ramp = delta / 2 - x
    keep = np.abs(ramp) <= 2.35
    ax.plot(x[keep], ramp[keep], color=GREEN, lw=2.6, zorder=3)
    push(ax, 1.0)

    # --- MMD: the witness ---------------------------------------------------
    ax = axes[2]
    base(ax, "MMD: kernel witness")
    ax.plot(x, witness, color=GREEN, lw=2.6, zorder=3)
    i_q = np.argmin(np.abs(x - delta))
    slope = np.gradient(witness, x)[i_q]  # < 0: uphill is towards p
    assert slope < 0
    push(ax, slope)

    fl.save(fig, "mdl-gan-critics-separated")


# =========================================================================== #
# 16.5  DCGAN and the minimal backbone                                        #
# =========================================================================== #

def _side(res):
    """Drawn side of a res x res feature map (log scale, so 1 and 64 fit)."""
    return 0.30 + 0.165 * np.log2(res)


def _fmap(ax, cx, cy, res, label=None, color=GRAY, fc="#F0F2F5", fs=12.5,
          ly=None):
    """A feature map: a square of side ~ log(res), its shape label below."""
    a = _side(res)
    ax.add_patch(FancyBboxPatch((cx - a / 2, cy - a / 2), a, a,
                                boxstyle="square,pad=0", fc=fc, ec=color,
                                lw=1.5, zorder=3))
    if label:
        ax.text(cx, cy - a / 2 - 0.12 if ly is None else ly, label,
                ha="center", va="top", fontsize=fs, color="black",
                linespacing=1.2)
    return a


def _vec(ax, cx, cy, h, color=GRAY, fc="#F0F2F5", w=0.26):
    """A vector: a thin upright bar."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="square,pad=0", fc=fc, ec=color,
                                lw=1.5, zorder=3))


def _op(ax, x0, x1, y, top=None, bottom=None, hl=False, fs=12.5):
    """A data-flow arrow, its operation above (gold wash if hl), its
    parameter count below."""
    fl.arrow(ax, (x0, y), (x1, y), color="black", lw=1.5, mut=12)
    if top:
        bb = dict(fc="#FBF2D4", ec="none", pad=2.0) if hl else None
        ax.text((x0 + x1) / 2, y + 0.14, top, ha="center", va="bottom",
                fontsize=fs, color="black", bbox=bb, zorder=5,
                linespacing=1.15)
    if bottom:
        ax.text((x0 + x1) / 2, y - 0.14, bottom, ha="center", va="top",
                fontsize=fs - 0.5, color="black", zorder=5)


def _shape(r, c):
    return f"${r}{{\\times}}{r}{{\\times}}{c}$"


def fig_dcgan():
    """DCGAN (Radford et al., 2015) as two rows of feature maps.  Generator:
    1x1x100 -> 4 -> 8 -> 16 -> 32 -> 64, transposed 4x4 convolutions, batch
    norm + ReLU except at the output (tanh, no BN).  Discriminator: 64 -> 32 ->
    16 -> 8 -> 4 -> logit, strided 4x4 convolutions, batch norm + leaky ReLU
    except at the input block (no BN).  The two no-BN blocks are washed gold."""
    fig, ax = plt.subplots(figsize=(13.0, 6.4))
    dx = 2.6
    xs = [0.75 + k * dx for k in range(6)]
    yg, yd = 4.75, 0.95
    LY = lambda y: y - _side(64) / 2 - 0.12     # one baseline per row

    # Parameter counts of the network as Radford et al. specify it (bias-free
    # 4x4 kernels; BN scale and shift on every normalized block).
    n = 64
    k4 = lambda ci, co: ci * co * 16
    g_params = (k4(100, 8 * n) + k4(8 * n, 4 * n) + k4(4 * n, 2 * n)
                + k4(2 * n, n) + k4(n, 3) + 2 * (8 + 4 + 2 + 1) * n)
    d_params = (k4(3, n) + k4(n, 2 * n) + k4(2 * n, 4 * n)
                + k4(4 * n, 8 * n) + k4(8 * n, 1) + 2 * (2 + 4 + 8) * n)
    assert (g_params, d_params) == (3_576_704, 2_765_568)

    def row(y, maps, ops, hls):
        sides = []
        for x, (res, ch, lab) in zip(xs, maps):
            text = "1 logit" if ch == 1 else _shape(res, ch)
            sides.append(_fmap(ax, x, y, res, text + (f"\n{lab}" if lab
                                                      else ""), ly=LY(y)))
        for k, (op, hl) in enumerate(zip(ops, hls)):
            _op(ax, xs[k] + sides[k] / 2 + 0.08,
                xs[k + 1] - sides[k + 1] / 2 - 0.08, y, op, hl=hl)

    x_left = xs[0] - _side(64) / 2
    ax.text(x_left, yg + 1.2,
            f"generator $G$: transposed convolutions $4{{\\times}}4$, stride 2"
            f" (the first: stride 1)    {g_params:,} parameters",
            ha="left", va="bottom", fontsize=13.5, color=ORANGE)
    row(yg, [(1, 100, "latent $z$"), (4, 8 * n, ""), (8, 4 * n, ""),
             (16, 2 * n, ""), (32, n, ""), (64, 3, "image")],
        ["BN, ReLU"] * 4 + ["tanh\nno BN"], [False] * 4 + [True])

    ax.text(x_left, yd + 1.2,
            f"discriminator $D$: convolutions $4{{\\times}}4$, stride 2"
            f" (the last: no padding)    {d_params:,} parameters",
            ha="left", va="bottom", fontsize=13.5, color=GREEN)
    row(yd, [(64, 3, "image"), (32, n, ""), (16, 2 * n, ""), (8, 4 * n, ""),
             (4, 8 * n, ""), (1, 1, "")],
        ["leaky ReLU\nno BN"] + ["BN\nleaky ReLU"] * 3 + [""],
        [True] + [False] * 4)

    fl.clean_axes(ax, lim=((x_left - 0.1, xs[-1] + 0.8),
                           (yd - 1.55, yg + 1.6)), hide=True, equal=True)
    fl.save(fig, "mdl-gan-dcgan")


def fig_backbone():
    """The minimal modern backbone of 16.5 with its parameter counts.
    Generator: a learned 4x4x128 constant and a linear projection of z to
    4x4x100, concatenated and mixed by a 3x3 convolution to 4x4x512; four
    stages of bilinear x2 upsampling + 3x3 conv + leaky ReLU to 64x64x32; a
    3x3 conv to RGB, no tanh.  Critic: four stages of 3x3 conv + leaky ReLU +
    bilinear /2 from 64x64x3 to 4x4x256; a mixing conv to 4x4x512; a linear
    head to one logit.  No normalization anywhere."""
    fig, ax = plt.subplots(figsize=(14.6, 6.6))

    def conv(ci, co, k=3):
        return ci * co * k * k + co

    # counts from the layer shapes, asserted against the code's totals
    g = dict(const=128 * 16, proj=100 * 1600 + 1600, mix=conv(228, 512),
             s1=conv(512, 256), s2=conv(256, 128), s3=conv(128, 64),
             s4=conv(64, 32), rgb=conv(32, 3))
    d = dict(s1=conv(3, 32), s2=conv(32, 64), s3=conv(64, 128),
             s4=conv(128, 256), mix=conv(256, 512), head=512 * 16 + 1)
    G_TOTAL, D_TOTAL = sum(g.values()), sum(d.values())
    assert (G_TOTAL, D_TOTAL) == (2_782_851, 1_576_769)

    def fmt(n):
        if n >= 1e6:
            return f"{n / 1e6:.2f} M"
        if n >= 1e4:
            return f"{n / 1e3:.0f} k"
        if n >= 1e3:
            return f"{n / 1e3:.1f} k"
        return f"{n}"

    yg, yd = 4.6, 0.95
    LY = lambda y: y - _side(64) / 2 - 0.12
    dx = 2.05

    # ---------------- generator ----------------
    x_main = [5.0 + k * dx for k in range(6)]          # 4x4x512 ... 64x64x3
    res = [4, 8, 16, 32, 64, 64]
    chans = [512, 256, 128, 64, 32, 3]
    sides = [_side(r) for r in res]
    for k in range(6):
        _fmap(ax, x_main[k], yg, res[k], _shape(res[k], chans[k]), ly=LY(yg))
    # entry: the constant (top) and the projected latent (bottom)
    xz, xe = 0.45, 2.35
    yc, yz = yg + 0.62, yg - 0.62
    a4 = _side(4)
    _vec(ax, xz, yz, 0.8)
    ax.text(xz - 0.22, yz, r"$z$", ha="right", va="center", fontsize=15,
            color="black")
    _fmap(ax, xe, yc, 4, fc="#FBE8D3", color=ORANGE)
    ax.text(xe, yc + a4 / 2 + 0.08, "learned constant\n" + _shape(4, 128),
            ha="center", va="bottom", fontsize=12.5, color="black",
            linespacing=1.2)
    _fmap(ax, xe, yz, 4)
    ax.text(xe, yz - a4 / 2 - 0.10, _shape(4, 100), ha="center", va="top",
            fontsize=12.5, color="black")
    _op(ax, xz + 0.21, xe - a4 / 2 - 0.08, yz, "linear", fmt(g["proj"]))
    # the two maps meet (channel concatenation), then the mixing conv
    xj = xe + a4 / 2 + 0.55
    for y0 in (yc, yz):
        ax.plot([xe + a4 / 2 + 0.06, xj], [y0, yg], color="black", lw=1.5,
                solid_capstyle="round", zorder=2)
    ax.plot([xj], [yg], "o", color="black", ms=4.5, zorder=4)
    _op(ax, xj, x_main[0] - sides[0] / 2 - 0.08, yg, "concat,\nconv",
        fmt(g["mix"]))
    for k, key in enumerate(("s1", "s2", "s3", "s4")):
        _op(ax, x_main[k] + sides[k] / 2 + 0.08,
            x_main[k + 1] - sides[k + 1] / 2 - 0.08, yg, r"$\uparrow 2$",
            fmt(g[key]))
    _op(ax, x_main[4] + sides[4] / 2 + 0.08, x_main[5] - sides[5] / 2 - 0.08,
        yg, "conv", fmt(g["rgb"]))
    ax.text(x_main[5], yg + sides[5] / 2 + 0.10, "image, no tanh",
            ha="center", va="bottom", fontsize=12.5, color="black")
    ax.text(0.2, yg + 1.8, f"generator $G$    {G_TOTAL:,} parameters",
            ha="left", va="bottom", fontsize=13.5, color=ORANGE)

    # ---------------- critic ----------------
    xd = [1.0 + k * dx for k in range(6)]
    rd = [64, 32, 16, 8, 4, 4]
    cd = [3, 32, 64, 128, 256, 512]
    sd = [_side(r) for r in rd]
    for k in range(6):
        _fmap(ax, xd[k], yd, rd[k], _shape(rd[k], cd[k]), ly=LY(yd))
    ax.text(xd[0], yd + sd[0] / 2 + 0.10, "image", ha="center", va="bottom",
            fontsize=12.5, color="black")
    for k, key in enumerate(("s1", "s2", "s3", "s4")):
        _op(ax, xd[k] + sd[k] / 2 + 0.08, xd[k + 1] - sd[k + 1] / 2 - 0.08,
            yd, r"$\downarrow 2$", fmt(d[key]))
    _op(ax, xd[4] + sd[4] / 2 + 0.08, xd[5] - sd[5] / 2 - 0.08, yd, "conv",
        fmt(d["mix"]))
    x_logit = xd[5] + dx
    _vec(ax, x_logit, yd, 0.26, w=0.26)
    ax.text(x_logit, LY(yd), "1 logit", ha="center", va="top", fontsize=12.5,
            color="black")
    _op(ax, xd[5] + sd[5] / 2 + 0.08, x_logit - 0.13 - 0.08, yd, "linear",
        fmt(d["head"]))
    ax.text(0.2, yd + 1.25, f"critic $D$    {D_TOTAL:,} parameters",
            ha="left", va="bottom", fontsize=13.5, color=GREEN)

    # ---------------- key ----------------
    ax.text(x_logit + 0.8, yd + 0.55,
            r"$\uparrow 2$, $\downarrow 2$:" "\nbilinear resampling,\n"
            r"conv $3{\times}3$, leaky ReLU" "\n\nno normalization\n"
            "in either network",
            ha="left", va="center", fontsize=12.5, color="black",
            linespacing=1.25)

    fl.clean_axes(ax, lim=((0.0, x_main[-1] + 0.85),
                           (yd - 1.25, yg + 2.3)), hide=True, equal=True)
    fl.save(fig, "mdl-gan-backbone")


# =========================================================================== #
# 16.6  Three ways a condition enters                                         #
# =========================================================================== #

def fig_conditioning():
    """Concatenation at the input, modulation a(c) h + b(c) of a feature map,
    and the projection critic's compatibility head e_c^T phi(x) + psi_0(x).
    Condition-derived quantities are purple (the embedding role)."""
    E_FC, E_EC = "#F2EDF7", PURPLE
    fig, ax = plt.subplots(figsize=(14.6, 4.6))
    ytop, yf = 4.15, 0.15                 # panel names / formula baseline

    def circ(x, y, sym):
        ax.add_patch(plt.Circle((x, y), 0.24, fc="white", ec="black", lw=1.5,
                                zorder=4))
        ax.text(x, y, sym, ha="center", va="center", fontsize=15,
                color="black", zorder=5)

    # ---------------- (a) concatenation ----------------
    x0 = 0.0
    ax.text(x0 + 0.2, ytop, "concatenation", ha="left", va="bottom",
            fontsize=14, color="black")
    _vec(ax, x0 + 0.65, 2.55, 1.4, w=0.30)
    ax.text(x0 + 0.42, 2.55, r"$z$", ha="right", va="center", fontsize=15,
            color="black")
    _vec(ax, x0 + 0.65, 1.20, 0.80, w=0.30, fc=E_FC, color=E_EC)
    ax.text(x0 + 0.42, 1.20, r"$e_c$", ha="right", va="center", fontsize=15,
            color="black")
    xc = x0 + 2.0                          # the concatenated vector
    _vec(ax, xc, 2.25, 1.4, w=0.30)
    _vec(ax, xc, 1.15, 0.80, w=0.30, fc=E_FC, color=E_EC)
    fl.arrow(ax, (x0 + 0.88, 2.55), (xc - 0.23, 2.30), color="black", lw=1.5,
             mut=12)
    fl.arrow(ax, (x0 + 0.88, 1.20), (xc - 0.23, 1.20), color="black", lw=1.5,
             mut=12)
    _box(ax, xc + 1.55, 1.95, 1.2, 1.0, r"$G$", ORANGE, fontsize=17)
    fl.arrow(ax, (xc + 0.23, 1.95), (xc + 0.87, 1.95), color="black", lw=1.5,
             mut=12)
    ax.text(x0 + 2.4, yf, r"$G([z;\, e_c])$", ha="center", va="center",
            fontsize=16, color="black")

    # ---------------- (b) modulation ----------------
    x0 = 5.15
    ax.text(x0 + 0.2, ytop, "modulation", ha="left", va="bottom",
            fontsize=14, color="black")
    a16 = _side(16)
    xh, xm, xp = x0 + 0.65, x0 + 2.25, x0 + 3.3
    xo = xp + 1.35
    _fmap(ax, xh, 1.75, 16)
    ax.text(xh, 1.75 - a16 / 2 - 0.12, r"$h$", ha="center", va="top",
            fontsize=15, color="black")
    circ(xm, 1.75, r"$\times$")
    circ(xp, 1.75, r"$+$")
    _fmap(ax, xo, 1.75, 16)
    fl.arrow(ax, (xh + a16 / 2 + 0.08, 1.75), (xm - 0.26, 1.75),
             color="black", lw=1.5, mut=12)
    fl.arrow(ax, (xm + 0.26, 1.75), (xp - 0.26, 1.75), color="black", lw=1.5,
             mut=12)
    fl.arrow(ax, (xp + 0.26, 1.75), (xo - a16 / 2 - 0.08, 1.75),
             color="black", lw=1.5, mut=12)
    yc = 3.35
    _box(ax, (xm + xp) / 2, yc, 0.9, 0.55, r"$c$", E_EC, fontsize=15,
         fc=E_FC)
    for xx, lab, ha, sgn in [(xm, r"$a(c)$", "right", -1),
                             (xp, r"$b(c)$", "left", 1)]:
        fl.arrow(ax, ((xm + xp) / 2 + 0.25 * sgn, yc - 0.30),
                 (xx, 1.75 + 0.27), color=E_EC, lw=1.5, mut=12)
        ax.text(xx + 0.22 * sgn, 2.55, lab, ha=ha, va="center", fontsize=15,
                color="black")
    ax.text((xh + xo) / 2, yf, r"$a(c) \odot h + b(c)$", ha="center",
            va="center", fontsize=16, color="black")

    # ---------------- (c) compatibility head ----------------
    x0 = 10.85
    ax.text(x0 + 0.2, ytop, "compatibility head", ha="left", va="bottom",
            fontsize=14, color="black")
    a32 = _side(32)
    xx_, xphi, xh2 = x0 + 0.6, x0 + 1.95, x0 + 3.65
    _fmap(ax, xx_, 1.95, 32)
    ax.text(xx_, 1.95 - a32 / 2 - 0.12, r"$x$", ha="center", va="top",
            fontsize=15, color="black")
    _box(ax, xphi, 1.95, 0.8, 1.0, r"$\varphi$", GREEN, fontsize=17)
    fl.arrow(ax, (xx_ + a32 / 2 + 0.08, 1.95), (xphi - 0.44, 1.95),
             color="black", lw=1.5, mut=12)
    bw = 1.45
    _box(ax, xh2, 2.75, bw, 0.6, r"$e_c^{\top} \varphi(x)$", E_EC,
         fontsize=15, fc=E_FC)
    _box(ax, xh2, 1.15, bw, 0.6, r"$\psi_0(x)$", GREEN, fontsize=15)
    for yy in (2.75, 1.15):
        fl.arrow(ax, (xphi + 0.42, 1.95 + (0.25 if yy > 2 else -0.25)),
                 (xh2 - bw / 2 - 0.06, yy), color="black", lw=1.5, mut=12)
    # the condition enters only at the inner product
    _box(ax, xh2, 3.65, 0.6, 0.48, r"$c$", E_EC, fontsize=15, fc=E_FC)
    fl.arrow(ax, (xh2, 3.65 - 0.26), (xh2, 2.75 + 0.33), color=E_EC, lw=1.5,
             mut=12)
    xs_ = xh2 + bw / 2 + 0.6
    circ(xs_, 1.95, r"$+$")
    for yy in (2.75, 1.15):
        fl.arrow(ax, (xh2 + bw / 2 + 0.04, yy),
                 (xs_ - 0.19, 1.95 + (0.17 if yy > 2 else -0.17)),
                 color="black", lw=1.5, mut=12)
    fl.arrow(ax, (xs_ + 0.26, 1.95), (xs_ + 0.8, 1.95), color="black",
             lw=1.5, mut=12)
    ax.text(xs_ + 0.88, 1.95, r"$D(x, c)$", ha="left", va="center",
            fontsize=15, color="black")
    ax.text((xx_ + xs_) / 2, yf, r"$D(x, c) = e_c^{\top} \varphi(x) + \psi_0(x)$",
            ha="center", va="center", fontsize=16, color="black")

    fl.clean_axes(ax, lim=((-0.1, xs_ + 2.0), (-0.25, 4.6)), hide=True,
                  equal=True)
    fl.save(fig, "mdl-gan-conditioning")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_architecture,
    fig_weights,
    fig_template,
    fig_critics_separated,
    fig_pairing,
    fig_dcgan,
    fig_backbone,
    fig_conditioning,
    fig_exits,
]


def main():
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
