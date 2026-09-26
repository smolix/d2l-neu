#!/usr/bin/env python3
"""Generate the illustrative figures for the Diffusion Models chapter (ch. 17)
in the shared house style defined in ``gen_mdl_figures.py``.

Figures:

  * ``mdl-diffusion-energy-density`` -- a one-dimensional energy with two wells
    and the density exp(-E)/Z it defines (limits.md);
  * ``mdl-diffusion-contrastive`` -- the two phases of the maximum-likelihood
    gradient of an energy-based model: data samples push the energy down,
    model samples push it up (energy-training.md);
  * ``mdl-diffusion-score-field`` -- a bimodal density with its score in one
    dimension, and the two-dimensional score field of a Gaussian mixture
    (score-matching.md);
  * ``mdl-diffusion-low-density`` -- accurate score estimates near the data and
    inaccurate ones in the low-density region between modes (langevin.md);
  * ``mdl-diffusion-noise-ladder`` -- the running-example mixture under
    increasing Gaussian smoothing; the barrier between modes disappears
    (annealed-langevin.md);
  * ``mdl-diffusion-forward-reverse`` -- the graphical model of a denoising
    diffusion probabilistic model: the fixed forward chain q and the learned
    reverse chain p_theta (ddpm.md);
  * ``mdl-diffusion-unet`` -- the noise-prediction U-Net with its time
    embedding and skip connections (annealed-langevin.md);
  * ``mdl-diffusion-guidance`` -- classifier-free guidance as an extrapolation
    from the unconditional through the conditional noise prediction (image-diffusion.md);
  * ``mdl-diffusion-paths`` -- the variance-preserving diffusion path and the
    linear flow-matching path as coefficient curves and as interpolations
    (flow-matching.md);
  * ``mdl-diffusion-masking`` -- the absorbing (masking) forward process on a
    token sequence and its reverse (discrete-diffusion.md).

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_diffusion_figures.py

or via ``make figures`` (picked up by the ``gen_mdl_*_figures.py`` glob).  All
figures are written to ``img/mdl-diffusion-<id>.svg``.  The generator is
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
INK = fl.INK
PURPLE = "#9467bd"

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# --------------------------------------------------------------------------- #
# Shared analytic objects                                                     #
# --------------------------------------------------------------------------- #

# The chapter's running example: a mixture of three Gaussians with unequal
# weights (same parameters as the notebooks).
MIX_MEANS = np.array([[-2.5, -1.5], [2.5, -1.5], [0.0, 2.5]])
MIX_WEIGHTS = np.array([0.5, 0.3, 0.2])
MIX_STD = 0.5


def mixture_logpdf(x, sigma=0.0):
    """log p_sigma(x) for the running example smoothed by N(0, sigma^2 I)."""
    var = MIX_STD ** 2 + sigma ** 2
    d2 = ((x[:, None, :] - MIX_MEANS[None]) ** 2).sum(-1)
    logs = np.log(MIX_WEIGHTS)[None] - d2 / (2 * var) - np.log(2 * np.pi * var)
    m = logs.max(1, keepdims=True)
    return (m + np.log(np.exp(logs - m).sum(1, keepdims=True)))[:, 0]


def mixture_score(x, sigma=0.0):
    """The score of p_sigma: the responsibility-weighted average of -(x-mu)/var."""
    var = MIX_STD ** 2 + sigma ** 2
    d2 = ((x[:, None, :] - MIX_MEANS[None]) ** 2).sum(-1)
    logs = np.log(MIX_WEIGHTS)[None] - d2 / (2 * var)
    r = np.exp(logs - logs.max(1, keepdims=True))
    r = r / r.sum(1, keepdims=True)
    return (r[:, :, None] * (MIX_MEANS[None] - x[:, None, :])).sum(1) / var


def _black_axes(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")
    ax.tick_params(colors="black", labelsize=11)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")


# --------------------------------------------------------------------------- #
# limits.md: energy -> density                                                #
# --------------------------------------------------------------------------- #

def fig_energy_density():
    """Left: a smooth one-dimensional energy with two wells and a confining
    quadratic term.  Right: the density exp(-E)/Z computed by quadrature."""
    x = np.linspace(-4.5, 4.5, 1201)
    E = 0.22 * x ** 2 - 1.6 * np.exp(-(x + 1.8) ** 2 / 0.9) \
        - 2.4 * np.exp(-(x - 1.6) ** 2 / 0.7) + 2.2
    p = np.exp(-E)
    Z = p.sum() * (x[1] - x[0])
    p = p / Z

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.3))
    a.plot(x, E, color=BLUE, lw=2.2)
    a.set_xlabel("$x$", fontsize=14)
    a.set_ylabel("$E(x)$", fontsize=14)
    a.set_xlim(-4.5, 4.5)
    a.set_ylim(-0.5, 6.4)
    fl.clean_axes(a, equal=False)
    _black_axes(a)
    # mark the two wells
    for xm, dy in ((-1.75, 0.55), (1.6, 0.55)):
        i = np.argmin(np.abs(x - xm))
        a.plot(x[i], E[i], "o", color=ORANGE, ms=6)
    a.text(-1.75, 0.15 + E[np.argmin(np.abs(x + 1.75))] - 0.95, "well",
           ha="center", va="top", fontsize=13, color="black")
    a.text(1.6, E[np.argmin(np.abs(x - 1.6))] - 0.35, "deeper well",
           ha="center", va="top", fontsize=13, color="black")

    b.plot(x, p, color=BLUE, lw=2.2)
    b.fill_between(x, 0, p, color=BLUE, alpha=0.12, lw=0)
    b.set_xlabel("$x$", fontsize=14)
    b.set_ylabel(r"$p(x) = \exp(-E(x))\,/\,Z$", fontsize=14)
    b.set_xlim(-4.5, 4.5)
    b.set_ylim(0, p.max() * 1.18)
    fl.clean_axes(b, equal=False)
    _black_axes(b)
    i1, i2 = np.argmax(p * (x < 0)), np.argmax(p * (x > 0))
    b.text(x[i1], p[i1] + 0.02, "mode", ha="center", va="bottom",
           fontsize=13, color="black")
    b.text(x[i2], p[i2] + 0.02, "taller mode", ha="center", va="bottom",
           fontsize=13, color="black")
    b.text(3.0, p.max() * 0.62, r"area $= 1$", ha="center", fontsize=13,
           color="black")
    fig.tight_layout(w_pad=2.5)
    fl.save(fig, "mdl-diffusion-energy-density")


# --------------------------------------------------------------------------- #
# energy-training.md: the two phases of the maximum-likelihood gradient       #
# --------------------------------------------------------------------------- #

def fig_contrastive():
    """A current model energy (blue) with data points (filled) sitting where
    the energy should be lower and model samples (open) where it is currently
    too low.  Arrows show the direction in which each phase moves the energy."""
    rng = np.random.default_rng(3)
    x = np.linspace(-4.5, 4.5, 801)
    # current model: one well too shallow at the left, and a spurious well at
    # the right deep enough (E(3) < E(-2)) that the model puts real mass there
    E = 0.25 * x ** 2 - 1.2 * np.exp(-(x + 2.0) ** 2 / 0.8) \
        - 2.0 * np.exp(-(x - 0.6) ** 2 / 0.6) - 3.0 * np.exp(-(x - 3.0) ** 2 / 0.5) + 2.6
    Ef = lambda q: np.interp(q, x, E)

    data_pts = np.concatenate([rng.normal(-2.0, 0.28, 5), rng.normal(0.6, 0.25, 4)])
    model_pts = np.concatenate([rng.normal(0.6, 0.3, 3), rng.normal(3.0, 0.25, 4),
                                rng.normal(-2.0, 0.3, 1)])

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(x, E, color=BLUE, lw=2.2, zorder=2)
    ax.plot(data_pts, Ef(data_pts), "o", color=ORANGE, ms=7, mec=ORANGE,
            zorder=4, label="data $\\mathbf{x} \\sim p_{\\mathrm{data}}$")
    ax.plot(model_pts, Ef(model_pts), "o", mfc="white", mec=GREEN, mew=1.8,
            ms=7, zorder=4, label="model samples $\\mathbf{x} \\sim p_{\\theta}$")
    # phase arrows: down at the data clusters, up at the model clusters
    for cx, dy, col in ((-2.0, -0.9, ORANGE), (3.0, 0.9, GREEN)):
        y0 = Ef(cx)
        ax.annotate("", xy=(cx, y0 + dy), xytext=(cx, y0 + 0.12 * np.sign(dy)),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                    mutation_scale=16), zorder=5)
    ax.text(-2.0, Ef(-2.0) - 1.15, "positive phase:\nlower $E$ at data",
            ha="center", va="top", fontsize=12.5, color="black")
    ax.text(3.0, 4.4, "negative phase:\nraise $E$ at model samples",
            ha="right", va="bottom", fontsize=12.5, color="black")
    ax.text(0.6, Ef(0.6) - 0.55, "both phases:\nno net change",
            ha="center", va="top", fontsize=12.5, color="black")
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel("$E_{\\theta}(x)$", fontsize=14)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-1.6, 6.3)
    ax.legend(loc="lower left", fontsize=11.5, frameon=False)
    fl.clean_axes(ax, equal=False)
    _black_axes(ax)
    fig.tight_layout()
    fl.save(fig, "mdl-diffusion-contrastive")


# --------------------------------------------------------------------------- #
# score-matching.md: the score in one and two dimensions                      #
# --------------------------------------------------------------------------- #

def fig_score_field():
    """Left column: a one-dimensional bimodal density and its score (a signed
    slope that crosses zero at each mode and at the valley).  Right: the
    running example's density contours with its score field."""
    fig = plt.figure(figsize=(10.0, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], hspace=0.12,
                          wspace=0.28)
    a_top = fig.add_subplot(gs[0, 0])
    a_bot = fig.add_subplot(gs[1, 0], sharex=a_top)
    b = fig.add_subplot(gs[:, 1])

    # one-dimensional mixture: weights 0.6 / 0.4, means -2 / 2, stds 0.6 / 0.8
    x = np.linspace(-5, 5, 1001)
    w, mu, sd = np.array([0.6, 0.4]), np.array([-2.0, 2.0]), np.array([0.6, 0.8])
    comp = w[None] * np.exp(-(x[:, None] - mu[None]) ** 2 / (2 * sd[None] ** 2)) \
        / (np.sqrt(2 * np.pi) * sd[None])
    p = comp.sum(1)
    score = (comp * (-(x[:, None] - mu[None]) / sd[None] ** 2)).sum(1) / p

    a_top.plot(x, p, color=BLUE, lw=2.2)
    a_top.fill_between(x, 0, p, color=BLUE, alpha=0.12, lw=0)
    a_top.set_ylabel("$p(x)$", fontsize=14)
    a_top.set_ylim(0, p.max() * 1.25)
    a_top.tick_params(labelbottom=False)
    fl.clean_axes(a_top, equal=False)
    _black_axes(a_top)

    a_bot.axhline(0, color="black", lw=0.9)
    a_bot.plot(x, score, color=ORANGE, lw=2.2)
    a_bot.set_ylim(-9, 9)
    a_bot.set_xlabel("$x$", fontsize=14)
    a_bot.set_ylabel(r"$\frac{d}{dx}\log p(x)$", fontsize=15)
    fl.clean_axes(a_bot, equal=False)
    _black_axes(a_bot)
    for m in mu:  # dotted guides at the modes, through both panels
        a_top.axvline(m, color=GRAY, lw=0.9, ls=":")
        a_bot.axvline(m, color=GRAY, lw=0.9, ls=":")
    # labels centered between the dotted mode guides, clear of the curve
    box = dict(fc="white", ec="none", pad=0.4)  # the guides pass behind the text
    a_bot.text(0.0, 6.3, "score $> 0$:\ntoward the mode", fontsize=11.5,
               ha="center", va="center", color="black", bbox=box, zorder=3)
    a_bot.text(0.0, -6.6, "score $< 0$:\ntoward the mode", fontsize=11.5,
               ha="center", va="center", color="black", bbox=box, zorder=3)

    # two-dimensional running example: contours + clipped score arrows
    g = np.linspace(-5, 5, 161)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    dens = np.exp(mixture_logpdf(pts)).reshape(X.shape)
    b.contourf(X, Y, dens, levels=10, cmap=fl.BLUE_CMAP, alpha=0.85)
    q = np.linspace(-4.6, 4.6, 15)
    QX, QY = np.meshgrid(q, q, indexing="ij")
    qp = np.stack([QX.ravel(), QY.ravel()], 1)
    s = mixture_score(qp)
    n = np.linalg.norm(s, axis=1, keepdims=True)
    s = s * np.minimum(1.4 / n, 1.0)  # clip long arrows in the tails
    b.quiver(qp[:, 0], qp[:, 1], s[:, 0], s[:, 1], angles="xy",
             scale_units="xy", scale=2.2, width=0.0045, color=ORANGE)
    b.set_xlabel("$x_1$", fontsize=14)
    b.set_ylabel("$x_2$", fontsize=14)
    b.set_xlim(-5, 5)
    b.set_ylim(-5, 5)
    fl.clean_axes(b, equal=True)
    _black_axes(b)
    fl.save(fig, "mdl-diffusion-score-field")


# --------------------------------------------------------------------------- #
# langevin.md: the score estimate is accurate only where the data are         #
# --------------------------------------------------------------------------- #

def fig_low_density():
    """Schematic.  Left: the true score of the running example with training
    samples.  Right: a caricature of a learned score, equal to the truth where
    the density is high and rotated by a random angle that grows as the density
    falls.  A dashed contour marks the region holding most of the mass."""
    rng = np.random.default_rng(7)
    # training samples
    k = rng.choice(3, size=400, p=MIX_WEIGHTS)
    samples = MIX_MEANS[k] + MIX_STD * rng.standard_normal((400, 2))

    q = np.linspace(-4.6, 4.6, 15)
    QX, QY = np.meshgrid(q, q, indexing="ij")
    qp = np.stack([QX.ravel(), QY.ravel()], 1)
    s = mixture_score(qp)
    dens = np.exp(mixture_logpdf(qp))
    dmax = np.exp(mixture_logpdf(MIX_MEANS[:1]))[0]
    rel = dens / dmax
    # rotate low-density arrows by a random angle; high-density arrows untouched
    ang = rng.uniform(-np.pi, np.pi, len(qp)) * np.clip(1.0 - 4.0 * rel ** 0.35, 0, 1)
    c, si = np.cos(ang), np.sin(ang)
    s_est = np.stack([c * s[:, 0] - si * s[:, 1], si * s[:, 0] + c * s[:, 1]], 1)

    g = np.linspace(-5, 5, 201)
    X, Y = np.meshgrid(g, g, indexing="ij")
    D = np.exp(mixture_logpdf(np.stack([X.ravel(), Y.ravel()], 1))).reshape(X.shape)
    level = 0.02 * dmax  # roughly the 99% region

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, field, title in ((axes[0], s, "true score"),
                             (axes[1], s_est, "estimated score (schematic)")):
        n = np.linalg.norm(field, axis=1, keepdims=True)
        f = field * np.minimum(1.4 / n, 1.0)
        ax.plot(samples[:, 0], samples[:, 1], "o", color=GRAY, ms=2.6,
                alpha=0.55, zorder=1)
        ax.contour(X, Y, D, levels=[level], colors="black", linestyles="--",
                   linewidths=1.1, zorder=2)
        ax.quiver(qp[:, 0], qp[:, 1], f[:, 0], f[:, 1], angles="xy",
                  scale_units="xy", scale=2.2, width=0.0045,
                  color=[BLUE if r > 0.02 else ORANGE for r in rel], zorder=3)
        ax.set_title(title, fontsize=13, color="black")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("$x_1$", fontsize=14)
        fl.clean_axes(ax, equal=True)
        _black_axes(ax)
    axes[0].set_ylabel("$x_2$", fontsize=14)
    fig.tight_layout(w_pad=2.0)
    fl.save(fig, "mdl-diffusion-low-density")


# --------------------------------------------------------------------------- #
# annealed-langevin.md: the noise ladder                                      #
# --------------------------------------------------------------------------- #

def fig_noise_ladder():
    """The running example's density after Gaussian perturbation at four noise
    levels.  Barriers between modes fill in as sigma grows, until one broad
    bump remains."""
    g = np.linspace(-6, 6, 201)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    sigmas = [0.0, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for ax, sg in zip(axes, sigmas):
        D = np.exp(mixture_logpdf(pts, sigma=sg)).reshape(X.shape)
        ax.contourf(X, Y, D, levels=12, cmap=fl.BLUE_CMAP)
        ax.plot(MIX_MEANS[:, 0], MIX_MEANS[:, 1], "+", color="black", ms=8, mew=1.3)
        title = r"$\sigma = 0$ (data)" if sg == 0 else rf"$\sigma = {sg:g}$"
        ax.set_title(title, fontsize=13.5, color="black")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        fl.clean_axes(ax, equal=True)
        _black_axes(ax)
    fig.tight_layout(w_pad=1.2)
    fl.save(fig, "mdl-diffusion-noise-ladder")


# --------------------------------------------------------------------------- #
# ddpm.md: the graphical model of a diffusion model                           #
# --------------------------------------------------------------------------- #

def fig_forward_reverse():
    """A chain of latent variables x_0 ... x_T.  Forward arrows (top, blue) are
    the fixed noising kernels q(x_t | x_{t-1}); reverse arrows (bottom, orange)
    are the learned denoising kernels p_theta(x_{t-1} | x_t)."""
    labels = [r"$\mathbf{x}_0$", r"$\mathbf{x}_1$", r"$\cdots$",
              r"$\mathbf{x}_{t-1}$", r"$\mathbf{x}_t$", r"$\cdots$", r"$\mathbf{x}_T$"]
    xs = np.arange(len(labels)) * 1.6
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    r = 0.42
    for x, lab in zip(xs, labels):
        if lab == r"$\cdots$":
            ax.text(x, 0, lab, ha="center", va="center", fontsize=16, color="black")
            continue
        # graphical-model convention: the observed variable (the data x_0) is
        # shaded; x_1 ... x_T are latent
        observed = lab == r"$\mathbf{x}_0$"
        ax.add_patch(plt.Circle((x, 0), r, fc=BLUE if observed else "white",
                                ec="black", lw=1.4, zorder=3,
                                alpha=0.25 if observed else 1))
        if observed:
            ax.add_patch(plt.Circle((x, 0), r, fc="none", ec="black", lw=1.4, zorder=4))
        ax.text(x, 0, lab, ha="center", va="center", fontsize=15, color="black", zorder=5)
    # forward arrows above, reverse arrows below
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        gap0 = r if labels[i] != r"$\cdots$" else 0.35
        gap1 = r if labels[i + 1] != r"$\cdots$" else 0.35
        ax.annotate("", xy=(x1 - gap1, 0.55), xytext=(x0 + gap0, 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.9,
                                    mutation_scale=14, connectionstyle="arc3,rad=-0.35"))
        ax.annotate("", xy=(x0 + gap0, -0.55), xytext=(x1 - gap1, -0.55),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.9,
                                    mutation_scale=14, connectionstyle="arc3,rad=-0.35"))
    mid = 0.5 * (xs[3] + xs[4])
    ax.text(mid, 1.55, r"forward: $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t I)$, fixed",
            ha="center", va="center", fontsize=13, color=fl.T.BLUE.dark)
    ax.text(mid, -1.55, r"reverse: $p_{\theta}(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mu_{\theta}(\mathbf{x}_t, t),\ \sigma_t^2 I)$, learned",
            ha="center", va="center", fontsize=13, color=fl.T.ORANGE.dark)
    ax.text(xs[0], -0.95, "data", ha="center", va="top", fontsize=12.5, color="black")
    ax.text(xs[-1], -0.95, "noise", ha="center", va="top", fontsize=12.5, color="black")
    ax.set_xlim(xs[0] - 0.9, xs[-1] + 0.9)
    ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-diffusion-forward-reverse")


# --------------------------------------------------------------------------- #
# annealed-langevin.md and image-diffusion.md: the U-Net and guidance        #
# --------------------------------------------------------------------------- #

def _ubox(ax, x, y, w, h, text, color, fs=11.5, fc="white", zorder=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=color, lw=1.6, zorder=zorder))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="black",
                zorder=zorder + 1)


def fig_unet():
    """The U-Net used for noise prediction: an encoder that halves the resolution
    twice, a bottleneck, and a decoder that doubles it twice, with skip
    connections joining equal resolutions and a time embedding added inside
    every block."""
    fig, ax = plt.subplots(figsize=(11.6, 4.4))
    # column positions and box heights proportional to resolution
    cols = [("input\n$28{\\times}28{\\times}1$", 0.0, 1.55, GRAY),
            ("block\n$28{\\times}28{\\times}32$", 2.1, 1.55, BLUE),
            ("block\n$14{\\times}14{\\times}64$", 4.2, 1.0, BLUE),
            ("bottleneck\n$7{\\times}7{\\times}64$", 6.3, 0.62, PURPLE),
            ("block\n$14{\\times}14{\\times}32$", 8.4, 1.0, ORANGE),
            ("block\n$28{\\times}28{\\times}32$", 10.5, 1.55, ORANGE),
            ("output\n$\\hat{\\epsilon}$, $28{\\times}28{\\times}1$", 12.6, 1.55, GRAY)]
    w = 1.36
    for text, x, h, color in cols:
        _ubox(ax, x, 0.0, w, h, text, color, fs=11.5)
    # main path arrows with labels
    steps = [("conv", 0.0, 2.1), ("down\n$\\div 2$", 2.1, 4.2), ("down\n$\\div 2$", 4.2, 6.3),
             ("up\n$\\times 2$", 6.3, 8.4), ("up\n$\\times 2$", 8.4, 10.5), ("conv", 10.5, 12.6)]
    for lab, x0, x1 in steps:
        ax.annotate("", xy=(x1 - w / 2, 0), xytext=(x0 + w / 2, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4,
                                    mutation_scale=12))
        ax.text((x0 + x1) / 2, 0.16, lab, ha="center", va="bottom", fontsize=10,
                color="black", linespacing=1.0)
    # skip connections (dashed arcs over the top)
    for (x0, x1, h, rad) in ((2.1, 10.5, 1.55, -0.16), (4.2, 8.4, 1.0, -0.3)):
        ax.annotate("", xy=(x1, h / 2 + 0.05), xytext=(x0, h / 2 + 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5, ls="--",
                                    mutation_scale=12,
                                    connectionstyle=f"arc3,rad={rad}"))
    ax.text(6.3, 2.05, "skip connections: encoder features are concatenated with decoder features at the same resolution",
            ha="center", va="center", fontsize=11.5, color=fl.T.GREEN.dark)
    # time embedding: the box spans every arrow's origin (x from 4.2 to 8.4)
    _ubox(ax, 6.3, -1.8, 6.0, 0.5,
          "time embedding: $t \\mapsto$ sinusoidal features $\\mapsto$ MLP", GRAY, fs=11.5)
    for x in (2.1, 4.2, 6.3, 8.4, 10.5):
        ax.annotate("", xy=(x, -0.55 if x != 6.3 else -0.36), xytext=(6.3 - (6.3 - x) * 0.5, -1.55),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.1,
                                    mutation_scale=10))
    ax.text(9.5, -1.8, "added inside every block", ha="left", va="center",
            fontsize=11.5, color="black")
    ax.set_xlim(-0.9, 13.6)
    ax.set_ylim(-2.2, 2.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-diffusion-unet")


def fig_guidance():
    """Classifier-free guidance in noise-prediction space: the guided prediction
    extrapolates from the unconditional prediction through the conditional one
    by a factor gamma."""
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    o = np.array([0.0, 0.0])
    e_u = np.array([2.2, 0.5])      # unconditional
    e_c = np.array([2.6, 1.7])      # conditional
    gamma = 2.5
    e_g = e_u + gamma * (e_c - e_u)
    fl.arrow(ax, o, e_u, color=GRAY, lw=2.2)
    fl.arrow(ax, o, e_c, color=BLUE, lw=2.2)
    fl.arrow(ax, o, e_g, color=ORANGE, lw=2.4)
    ax.plot([e_u[0], e_g[0]], [e_u[1], e_g[1]], ls=":", color="black", lw=1.0)
    ax.plot(*o, "o", color="black", ms=5)
    ax.text(-0.1, -0.1, r"$\mathbf{x}_t$", ha="right", va="top", fontsize=14, color="black")
    ax.text(e_u[0] + 0.15, e_u[1] - 0.05, r"$\epsilon_{\theta}(\mathbf{x}_t, t, \varnothing)$ (unconditional)",
            ha="left", va="top", fontsize=12, color="black")
    ax.text(e_c[0] + 0.15, e_c[1] + 0.08, r"$\epsilon_{\theta}(\mathbf{x}_t, t, c)$ (conditional)",
            ha="left", va="bottom", fontsize=12, color="black")
    ax.text(e_g[0] - 0.12, e_g[1] + 0.1,
            r"guided: $\epsilon_{\varnothing} + \gamma\,(\epsilon_c - \epsilon_{\varnothing})$, $\gamma > 1$",
            ha="right", va="bottom", fontsize=12, color="black")
    ax.text(0.5 * (e_u[0] + e_c[0]) + 0.22, 0.5 * (e_u[1] + e_c[1]) - 0.02,
            r"$\epsilon_c - \epsilon_{\varnothing}$", ha="left", va="center", fontsize=12, color="black")
    fl.arrow(ax, e_u, e_c, color=BLUE, lw=1.4, ls="--", mut=11)
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylim(-0.8, 4.5)
    fl.clean_axes(ax, equal=True, hide=True)
    fl.save(fig, "mdl-diffusion-guidance")


# --------------------------------------------------------------------------- #
# flow-matching.md: two Gaussian paths from noise to data                     #
# --------------------------------------------------------------------------- #

def fig_paths():
    """Left: the coefficient pairs (sigma_t, alpha_t) of the variance-preserving
    diffusion path (a quarter circle) and of the linear flow-matching path (a
    straight segment), both running from pure noise (sigma, alpha) = (1, 0)
    to clean data (0, 1) on the flow-matching clock.  Right: the resulting one-dimensional
    interpolation x_t = alpha_t x_1 + sigma_t eps for one data value and one
    noise draw."""
    s = np.linspace(0, 1, 401)                    # flow-matching clock: 0 noise, 1 data
    # VP diffusion schedule of Ho et al.: T = 1000, beta linear 1e-4 .. 0.02;
    # reparameterized so that s = 1 is data (t = 0) and s = 0 is noise (t = T)
    beta = np.linspace(1e-4, 0.02, 1000)
    abar = np.concatenate([[1.0], np.cumprod(1 - beta)])          # abar_0 .. abar_T
    tt = (1 - s) * 1000
    a_vp = np.sqrt(np.interp(tt, np.arange(1001), abar))
    sg_vp = np.sqrt(1 - a_vp ** 2)
    a_lin, sg_lin = s, 1 - s

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    a.plot(sg_vp, a_vp, color=BLUE, lw=2.4)
    a.plot(sg_lin, a_lin, color=ORANGE, lw=2.4)
    a.plot([0], [1], "o", color="black", ms=6)
    a.plot([1], [0], "o", color="black", ms=6)
    # endpoint labels above the data point and below the noise point, clear of
    # both curves; direct curve labels in the empty regions outside the arc and
    # below the segment (no legend, which the curves would cross)
    a.text(0.0, 1.07, "data:  $(\\sigma, \\alpha) = (0, 1)$", ha="left", va="bottom",
           fontsize=12.5, color="black")
    a.text(1.0, -0.06, "noise:  $(1, 0)$", ha="right", va="top", fontsize=12.5,
           color="black")
    a.text(0.79, 0.84, "diffusion\n(variance preserving)", ha="left", va="bottom",
           fontsize=12, color=fl.T.BLUE.dark)
    a.text(0.05, 0.42, "linear path\n(flow matching)", ha="left", va="top",
           fontsize=12, color=fl.T.ORANGE.dark)
    a.set_xlabel("noise coefficient $\\sigma_t$", fontsize=13)
    a.set_ylabel("signal coefficient $\\alpha_t$", fontsize=13)
    a.set_xlim(-0.05, 1.45)
    a.set_ylim(-0.2, 1.3)
    fl.clean_axes(a, equal=True)
    _black_axes(a)

    x1, eps = 1.5, -1.0
    b.plot(s, a_vp * x1 + sg_vp * eps, color=BLUE, lw=2.4)
    b.plot(s, a_lin * x1 + sg_lin * eps, color=ORANGE, lw=2.4)
    b.axhline(x1, color=GRAY, lw=0.9, ls=":")
    b.axhline(eps, color=GRAY, lw=0.9, ls=":")
    b.text(0.02, x1 + 0.12, "$x_1$ (data)", fontsize=12.5, color="black", va="bottom")
    b.text(0.02, eps - 0.12, "$\\epsilon$ (noise)", fontsize=12.5, color="black", va="top")
    b.text(0.72, -0.5, "diffusion", fontsize=12.5, color=fl.T.BLUE.dark, ha="center")
    b.text(0.36, 0.55, "linear", fontsize=12.5, color=fl.T.ORANGE.dark, ha="center")
    b.set_xlabel("flow-matching time $t$ (0 = noise, 1 = data)", fontsize=13)
    b.set_ylabel("$x_t = \\alpha_t x_1 + \\sigma_t \\epsilon$", fontsize=13)
    b.set_xlim(0, 1)
    b.set_ylim(-1.6, 2.0)
    fl.clean_axes(b, equal=False)
    _black_axes(b)
    fig.tight_layout(w_pad=2.5)
    fl.save(fig, "mdl-diffusion-paths")


# --------------------------------------------------------------------------- #
# discrete-diffusion.md: the absorbing (masking) forward process              #
# --------------------------------------------------------------------------- #

def fig_masking():
    """Four snapshots of a nine-token sequence under an absorbing forward
    process: each token is independently replaced by the mask symbol, masked
    tokens stay masked, and at the end every token is masked.  The reverse
    process reveals tokens in the opposite order."""
    rng = np.random.default_rng(11)
    letters = list("diffusion")
    n = len(letters)
    # time of masking for each token: reveal order is its reverse
    mask_time = rng.permutation(n) + 1           # 1..n, the step at which each token is masked
    stages = [0, 3, 6, 9]                         # number of masked tokens shown per row
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    w, h, gap = 0.78, 0.78, 0.22
    y0 = 3.0
    for r, k in enumerate(stages):
        y = y0 - r * 1.15
        for i, ch in enumerate(letters):
            x = i * (w + gap)
            masked = mask_time[i] <= k
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                        fc=(fl.T.MUTED if masked else "white"),
                                        ec="black", lw=1.2))
            ax.text(x + w / 2, y + h / 2, "?" if masked else ch, ha="center", va="center",
                    fontsize=15, color="white" if masked else "black", family="monospace")
        label = {0: r"$\mathbf{x}_0$: data", 3: r"$\mathbf{x}_{T/3}$", 6: r"$\mathbf{x}_{2T/3}$",
                 9: r"$\mathbf{x}_T$: all masked"}[k]
        ax.text(n * (w + gap) + 0.15, y + h / 2, label, ha="left", va="center",
                fontsize=13, color="black")
    # forward and reverse arrows
    xa = -0.55
    ax.annotate("", xy=(xa, y0 - 3 * 1.15 + 0.1), xytext=(xa, y0 + h - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0, mutation_scale=14))
    ax.text(xa - 0.15, y0 - 1.15 * 1.5 + h / 2, "forward:\nmask each token\nwith probability $\\beta_t$;\nmasked stays masked",
            ha="right", va="center", fontsize=11.5, color=fl.T.BLUE.dark)
    xb = n * (w + gap) + 3.4  # right of the longest row label ("all masked")
    ax.annotate("", xy=(xb, y0 + h - 0.1), xytext=(xb, y0 - 3 * 1.15 + 0.1),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0, mutation_scale=14))
    ax.text(xb + 0.15, y0 - 1.15 * 1.5 + h / 2, "reverse:\nreveal masked tokens\nfrom $p_{\\theta}(x_0^i \\mid \\mathbf{x}_t)$;\nrevealed stays revealed",
            ha="left", va="center", fontsize=11.5, color=fl.T.ORANGE.dark)
    ax.set_xlim(-4.3, n * (w + gap) + 7.9)
    ax.set_ylim(y0 - 3 * 1.15 - 0.3, y0 + h + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-diffusion-masking")


FIGURES = [
    fig_energy_density,
    fig_contrastive,
    fig_score_field,
    fig_low_density,
    fig_noise_ladder,
    fig_forward_reverse,
    fig_unet,
    fig_guidance,
    fig_paths,
    fig_masking,
]


def main():
    for f in FIGURES:
        f()
    for p in fl.WRITTEN:
        print("wrote", os.path.relpath(p))


if __name__ == "__main__":
    main()
