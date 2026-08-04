#!/usr/bin/env python3
"""Generate the illustrative figures for the Generative Adversarial Networks
chapter (ch. 16) in the one shared house style defined in ``gen_mdl_figures.py``.

Four figures (see ``reviews/gan-implementation-brief.md`` §8):

  * ``mdl-gan-architecture`` -- the adversarial loop: a latent draw through the
    generator, the generated and real batches meeting at the discriminator, and
    the two updates with their opposite goals (replaces the legacy hand-drawn
    ``img/gan.svg``);
  * ``mdl-gan-template`` -- the two knobs of the adversarial-objective template
    ``d(p,q) = sup_T {E_p[a(T)] - E_q[b(T)]}``, with GAN/f-GAN, W_1 and MMD
    placed on the (payoff, critic class) plane and RpGAN lifted off it;
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
    fl.arrow(ax, (13.35, 3.95), (13.92, 3.95), color="black", lw=1.6)
    xs = 14.35
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
    ax.text(xs, 5.58, "realness logit", ha="center", va="bottom", fontsize=12.5,
            color="black")

    # --- the two updates, pushing opposite ways -----------------------------
    _curved(ax, (11.52, 4.68), (12.48, 4.68), -1.05, GREEN, lw=1.8, ls="--")
    ax.text(13.05, 6.05, "update $D$: raise scores on real,\n"
                         "lower them on generated",
            ha="center", va="bottom", fontsize=12.5, color=GREEN)
    ax.plot([12.90, 12.90], [3.24, 0.30], ls="--", color=ORANGE, lw=1.8)
    ax.plot([12.90, 4.65], [0.30, 0.30], ls="--", color=ORANGE, lw=1.8)
    fl.arrow(ax, (4.65, 0.30), (4.65, 1.55), color=ORANGE, lw=1.8, ls="--")
    ax.text(8.20, 0.46, r"update $G$: gradient flows back through $D$",
            ha="center", va="bottom", fontsize=12.5, color=ORANGE)

    fl.clean_axes(ax, lim=((0.20, 16.90), (-0.05, 7.05)), hide=True)
    fl.save(fig, "mdl-gan-architecture")


# =========================================================================== #
# 16.2  The template's two knobs                                              #
# =========================================================================== #

def fig_template():
    """One template, two choices: which payoff (a, b) and which critic class.
    GAN/f-GAN, W_1 and MMD sit at three corners of that plane; RpGAN does not
    sit on it at all, because its objective is quadratic in (p, q)."""
    fig, ax = plt.subplots(figsize=(9.6, 6.0))

    SH = np.array([0.44, 0.40])                  # shear: one unit of payoff
    U1, V1 = 7.4, 4.9                            # plane extent

    def P(u, v):
        return np.array([u + SH[0] * v, SH[1] * v])

    # the plane itself
    ax.add_patch(Polygon([P(0, 0), P(U1, 0), P(U1, V1), P(0, V1)],
                         closed=True, fc=LIGHT, alpha=0.22, ec=GRAY, lw=1.2,
                         zorder=1))
    # the linear-payoff row: every integral probability metric lives here
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
    ax.text(*(P(0, V1 + 0.55) + np.array([-0.28, 0.10])), r"payoff $(a,b)$",
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

    fl.clean_axes(ax, lim=((-1.9, 13.6), (-1.05, 5.2)), hide=True, equal=True)
    fl.save(fig, "mdl-gan-template")


# =========================================================================== #
# 16.3  One sample versus a pair                                              #
# =========================================================================== #

def fig_pairing():
    """Left: a single sample scored against the fixed threshold 0 -- adding a
    constant c to the critic flips the verdict.  Right: a pair, judged by the
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
        fl.arrow(ax, (XL + 0.10, -1.62), (XR - 0.10, -1.62), color=GRAY,
                 lw=1.6, mut=13)
        ax.text((XL + XR) / 2, -1.50, r"every score $+\,c$", ha="center",
                va="bottom", fontsize=12.5, color=GRAY)
        for x, lab in [(XL, r"critic $D$"), (XR, r"shifted critic $D+c$")]:
            ax.text(x, -2.02, lab, ha="center", va="top", fontsize=13,
                    color="black")
        ax.text((XL + XR) / 2, 2.50, title, ha="center", va="top",
                fontsize=13.5, color="black")
        fl.clean_axes(ax, lim=LIM, hide=True, equal=False)

    # --- (a) one sample against a fixed threshold: the verdict flips --------
    axa.plot([XA, 6.95], [0.0, 0.0], ls="--", color="black", lw=1.2)
    axa.text(6.95, 0.14, "threshold", ha="right", va="bottom", fontsize=12.5,
             color="black")
    for x, y, lab, verdict in [(XL, -0.70, r"$D(x)$", "generated"),
                               (XR, -0.70 + C, r"$D(x)+c$", "real")]:
        axa.plot([x], [y], "o", color=BLUE, ms=11, zorder=5)
        axa.text(x - 0.16, y, lab, ha="right", va="center", fontsize=13.5,
                 color=BLUE)
        axa.text(x + 0.18, y, f"$\\to$ {verdict}", ha="left", va="center",
                 fontsize=13, color="black")
    frame(axa, "one sample: is $D(x)$ above $0$?")

    # --- (b) a pair: the difference, and the same difference after the shift -
    for k, (x, sh) in enumerate([(XL, 0.0), (XR, C)]):
        for y, col, lab in [(DX + sh, BLUE, r"$D(x)$"),
                            (DY + sh, ORANGE, r"$D(x')$")]:
            axb.plot([x], [y], "o", color=col, ms=11, zorder=5)
            if not k:                       # the header names the shift; the
                axb.text(x - 0.16, y, lab, ha="right", va="center",
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
    """The de-heuristicization map: fix the game, constrain the critic, or
    remove the game -- each branch with its named steps and its cost."""
    fig, ax = plt.subplots(figsize=(12.6, 5.4))

    branches = [
        (BLUE, "1. fix the game",
         ["Mescheder\n$R_1 + R_2$", "convergent\nobjective (RpGAN)", "R3GAN"],
         "cost: one more knob ($\\gamma$) and a minimax loop that remains"),
        (ORANGE, "2. constrain the critic",
         ["spectral norm", "frozen pretrained\nfeatures",
          "controlled\naugmentation (ADA)"],
         "cost: the critic is no longer free — it inherits the features' bias"),
        (GREEN, "3. remove the game",
         ["MMD, IMLE", "diffusion and flow\nas regression",
          "the adversarial term\nreturns at distillation"],
         "cost: a fixed discrepancy weakens in high dimension; slow sampling"),
    ]

    xs = [4.60, 8.65, 12.70]                     # the three step columns
    ys = [5.25, 3.10, 0.95]                      # the three branch lanes
    bw, bh = 3.34, 0.90

    _box(ax, 1.35, ys[1], 2.40, 1.35, "unstable\nminimax\ntraining", GRAY,
         fontsize=13.5, lw=1.9)

    for (col, name, steps, cost), y in zip(branches, ys):
        _curved(ax, (2.58, ys[1] + (0.30 if y > ys[1] else
                                    (-0.30 if y < ys[1] else 0.0))),
                (xs[0] - bw / 2 - 0.10, y), 0.0 if y == ys[1] else
                (-0.18 if y > ys[1] else 0.18), col, lw=1.7)
        ax.text(xs[0] - bw / 2, y + bh / 2 + 0.16, name, ha="left",
                va="bottom", fontsize=13.5, color=col)
        for x, s in zip(xs, steps):
            _box(ax, x, y, bw, bh, s, col, fontsize=12.5, fc="white")
        for xa, xb in zip(xs[:-1], xs[1:]):
            fl.arrow(ax, (xa + bw / 2 + 0.08, y), (xb - bw / 2 - 0.08, y),
                     color=col, lw=1.5, mut=13)
        ax.text(xs[0] - bw / 2, y - bh / 2 - 0.18, cost, ha="left", va="top",
                fontsize=12.5, color="black")

    fl.clean_axes(ax, lim=((0.0, 14.55), (-0.05, 6.35)), hide=True, equal=False)
    fl.save(fig, "mdl-gan-exits")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_architecture,
    fig_template,
    fig_pairing,
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
