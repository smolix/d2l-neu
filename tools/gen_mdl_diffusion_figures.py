#!/usr/bin/env python3
"""Generate the illustrative figures for the Diffusion Models chapter (ch. 17)
in the shared house style defined in ``gen_mdl_figures.py``.

The chapter's notation throughout: a Gaussian path x_t = alpha_t x0 + sigma_t eps
on one clock, data at t = 0 and noise at t = 1; the VP path has
log abar_t = -(0.1 t + 9.95 t^2), the linear path alpha_t = 1 - t, sigma_t = t;
lambda_t = log(alpha_t^2 / sigma_t^2).  Every figure that shows a quantity of
the three-mode mixture computes it exactly here, in NumPy.

Figures, by section:

  17.1 denoising.md
  * ``mdl-diffusion-score-field`` -- a bimodal density and its score in one
    dimension; the mixture's score field in two;
  * ``mdl-diffusion-energy-density`` -- an energy with two wells and the
    density exp(-E)/Z it defines;
  * ``mdl-diffusion-noise-ladder`` -- the mixture under growing Gaussian noise.

  17.2 gaussian-paths.md
  * ``mdl-diffusion-paths`` -- alpha_t, sigma_t and lambda_t of the VP and the
    linear path;
  * ``mdl-diffusion-forward-reverse`` -- the forward and reverse chains;
  * ``mdl-diffusion-blueprint`` -- conditional and marginal path and field of
    the linear path on the mixture, exact.

  17.3 image-diffusion.md
  * ``mdl-diffusion-unet`` -- the U-Net with its skips and its time/label
    embedding;
  * ``mdl-diffusion-guidance`` -- classifier-free guidance on the mixture as an
    extrapolation of posterior means.

  17.4 samplers.md
  * ``mdl-diffusion-two-samplers`` -- DDIM at eta = 0 and eta = 1 on the
    mixture with the exact predictor, from shared starts.

  17.5 design-space.md
  * ``mdl-diffusion-weightings`` -- weightings w(lambda) and the densities of
    lambda under uniform t;
  * ``mdl-diffusion-schedules`` -- lambda_t against t for four schedules.

  17.6 latent-diffusion.md
  * ``mdl-diffusion-latent-pipeline`` -- encoder, latent diffusion, decoder;
  * ``mdl-diffusion-dit-block`` -- patchify, one adaLN-Zero block, depatchify.

  17.7 few-step.md
  * ``mdl-diffusion-flow-map`` -- probability-flow trajectories of a 1-D
    two-Gaussian mixture, a jump of the flow map, and its three identities.

  17.8 discrete-diffusion.md
  * ``mdl-diffusion-masking`` -- the absorbing chain on a token sequence.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_diffusion_figures.py

or via ``make figures``.  All figures are written to
``img/mdl-diffusion-<id>.svg``.  The generator is byte-idempotent: seeded RNGs
only, no timestamps (``fl.save`` fixes the SVG hash salt and nulls the date).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
T = fl.T
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
INK = fl.INK
PURPLE = T.PURPLE.base
RED = T.RED.base
GOLD = T.GOLD.base

from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

LBL = 14      # axis labels and in-figure math
TXT = 13      # annotations and tick labels

# --------------------------------------------------------------------------- #
# Shared analytic objects                                                     #
# --------------------------------------------------------------------------- #

# The chapter's running example (d2l.GaussianMixture.three_modes()).
MIX_MEANS = np.array([[-2.5, -1.5], [2.5, -1.5], [0.0, 2.5]])
MIX_WEIGHTS = np.array([0.5, 0.3, 0.2])
MIX_STD = 0.5


def mixture_logpdf(x, sigma=0.0, alpha=1.0, means=MIX_MEANS,
                   weights=MIX_WEIGHTS, std=MIX_STD):
    """log p(x) for x = alpha x0 + sigma eps, x0 from the mixture."""
    return mix_posterior(x, alpha, sigma, means, weights, std)[1]


def mix_posterior(x, alpha, sigma, means=MIX_MEANS, weights=MIX_WEIGHTS,
                  std=MIX_STD):
    """For x = alpha x0 + sigma eps: responsibilities, log p(x), E[x0 | x] and
    E[eps | x], all exact (each mode is Gaussian)."""
    var = (alpha * std) ** 2 + sigma ** 2
    r = x[:, None, :] - alpha * means[None]
    d = means.shape[1]
    logs = (np.log(weights)[None] - (r ** 2).sum(-1) / (2 * var)
            - d / 2 * np.log(2 * np.pi * var))
    m = logs.max(1, keepdims=True)
    lse = m + np.log(np.exp(logs - m).sum(1, keepdims=True))
    resp = np.exp(logs - lse)
    x0 = (resp[..., None] * (means[None] + alpha * std ** 2 / var * r)).sum(1)
    eps = (resp[..., None] * (sigma / var * r)).sum(1)
    return resp, lse[:, 0], x0, eps


def mixture_score(x, sigma=0.0):
    """The score of the mixture smoothed by N(0, sigma^2 I)."""
    var = MIX_STD ** 2 + sigma ** 2
    resp = mix_posterior(x, 1.0, sigma)[0]
    return (resp[:, :, None] * (MIX_MEANS[None] - x[:, None, :])).sum(1) / var


def vp_log_abar(t):
    return -(0.1 * t + 9.95 * t ** 2)


def vp(t):
    """(alpha_t, sigma_t) of the VP path, beta from 0.1 to 20."""
    la = vp_log_abar(np.asarray(t, float))
    return np.exp(la / 2), np.sqrt(-np.expm1(la))


def cosine(t, s=0.008):
    """(alpha_t, sigma_t) of Nichol and Dhariwal's cosine path."""
    t = np.asarray(t, float)
    f = np.cos(np.pi / 2 * (t + s) / (1 + s)) / np.cos(np.pi / 2 * s / (1 + s))
    f = np.where(t < 1, f, 0.0)
    return f, np.sqrt(1 - f ** 2)


def linear(t):
    t = np.asarray(t, float)
    return 1 - t, t


def log_snr(path, t):
    a, s = path(t)
    with np.errstate(divide="ignore"):
        return 2 * (np.log(a) - np.log(s))


def data_cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.choice(3, size=n, p=MIX_WEIGHTS)
    return MIX_MEANS[k] + MIX_STD * rng.standard_normal((n, 2))


def _black_axes(ax, ticks=TXT):
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")
    ax.tick_params(colors="black", labelsize=ticks)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")


def _square(ax, lim, ticks=(-4, -2, 0, 2, 4)):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fl.clean_axes(ax, equal=True)
    _black_axes(ax)


def _quiver(ax, pts, field, scale, color, zorder=3):
    ax.quiver(pts[:, 0], pts[:, 1], field[:, 0], field[:, 1], angles="xy",
              scale_units="xy", scale=scale, width=0.006, headwidth=4.0,
              headlength=4.5, color=color, zorder=zorder)


WHITE_BOX = dict(fc="white", ec="none", pad=0.25, alpha=0.9)


# =========================================================================== #
# 17.1 denoising.md                                                           #
# =========================================================================== #

def fig_energy_density():
    """Left: a smooth one-dimensional energy with two wells and a confining
    quadratic term.  Right: the density exp(-E)/Z computed by quadrature."""
    x = np.linspace(-4.5, 4.5, 1201)
    E = 0.22 * x ** 2 - 1.6 * np.exp(-(x + 1.8) ** 2 / 0.9) \
        - 2.4 * np.exp(-(x - 1.6) ** 2 / 0.7) + 2.2
    p = np.exp(-E)
    Z = p.sum() * (x[1] - x[0])
    p = p / Z

    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    a.plot(x, E, color=BLUE, lw=2.2)
    a.set_xlabel("$x$", fontsize=LBL)
    a.set_ylabel("$E(x)$", fontsize=LBL)
    a.set_xlim(-4.5, 4.5)
    a.set_ylim(-1.0, 6.4)
    fl.clean_axes(a, equal=False)
    _black_axes(a)
    for xm in (-1.75, 1.6):
        i = np.argmin(np.abs(x - xm))
        a.plot(x[i], E[i], "o", color=ORANGE, ms=6)
    a.text(-1.75, E[np.argmin(np.abs(x + 1.75))] - 0.8, "well",
           ha="center", va="top", fontsize=TXT, color="black")
    a.text(1.6, E[np.argmin(np.abs(x - 1.6))] - 0.35, "deeper well",
           ha="center", va="top", fontsize=TXT, color="black")

    b.plot(x, p, color=BLUE, lw=2.2)
    b.fill_between(x, 0, p, color=BLUE, alpha=0.12, lw=0)
    b.set_xlabel("$x$", fontsize=LBL)
    b.set_ylabel(r"$p(x) = \exp(-E(x))\,/\,Z$", fontsize=LBL)
    b.set_xlim(-4.5, 4.5)
    b.set_ylim(0, p.max() * 1.18)
    fl.clean_axes(b, equal=False)
    _black_axes(b)
    i1, i2 = np.argmax(p * (x < 0)), np.argmax(p * (x > 0))
    b.text(x[i1], p[i1] + 0.02, "mode", ha="center", va="bottom",
           fontsize=TXT, color="black")
    b.text(x[i2], p[i2] + 0.02, "taller mode", ha="center", va="bottom",
           fontsize=TXT, color="black")
    b.text(-0.1, 0.045, "area = 1", ha="center", va="center", fontsize=TXT,
           color="black")
    fig.tight_layout(w_pad=2.5)
    fl.save(fig, "mdl-diffusion-energy-density")


def fig_score_field():
    """Left column: a one-dimensional bimodal density and its score (a signed
    slope that crosses zero at each mode and at the valley).  Right: the
    running example's density contours with its score field."""
    fig = plt.figure(figsize=(9.0, 3.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], hspace=0.12,
                          wspace=0.3)
    a_top = fig.add_subplot(gs[0, 0])
    a_bot = fig.add_subplot(gs[1, 0], sharex=a_top)
    b = fig.add_subplot(gs[:, 1])

    x = np.linspace(-5, 5, 1001)
    w, mu, sd = np.array([0.6, 0.4]), np.array([-2.0, 2.0]), np.array([0.6, 0.8])
    comp = w[None] * np.exp(-(x[:, None] - mu[None]) ** 2 / (2 * sd[None] ** 2)) \
        / (np.sqrt(2 * np.pi) * sd[None])
    p = comp.sum(1)
    score = (comp * (-(x[:, None] - mu[None]) / sd[None] ** 2)).sum(1) / p

    a_top.plot(x, p, color=BLUE, lw=2.2)
    a_top.fill_between(x, 0, p, color=BLUE, alpha=0.12, lw=0)
    a_top.set_ylabel("$p(x)$", fontsize=LBL)
    a_top.set_ylim(0, p.max() * 1.25)
    a_top.tick_params(labelbottom=False)
    fl.clean_axes(a_top, equal=False)
    _black_axes(a_top)

    a_bot.axhline(0, color="black", lw=0.9)
    a_bot.plot(x, score, color=ORANGE, lw=2.2)
    a_bot.set_ylim(-10.9, 10)
    a_bot.set_yticks([-10, 0, 10])
    a_bot.set_xlabel("$x$", fontsize=LBL)
    a_bot.set_ylabel(r"$\frac{d}{dx}\log p(x)$", fontsize=15)
    fl.clean_axes(a_bot, equal=False)
    _black_axes(a_bot)
    for m in mu:
        a_top.axvline(m, color=GRAY, lw=0.9, ls=":")
        a_bot.axvline(m, color=GRAY, lw=0.9, ls=":")
    box = dict(fc="white", ec="none", pad=0.4)
    a_bot.text(0.0, 6.4, "score > 0:\npoints right", fontsize=TXT - 0.5,
               ha="center", va="center", color="black", bbox=box, zorder=3,
               linespacing=1.05)
    a_bot.text(0.0, -7.0, "score < 0:\npoints left", fontsize=TXT - 0.5,
               ha="center", va="center", color="black", bbox=box, zorder=3,
               linespacing=1.05)

    g = np.linspace(-5, 5, 161)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    dens = np.exp(mixture_logpdf(pts)).reshape(X.shape)
    levels = np.linspace(0.0, dens.max(), 11)
    levels[0] = 1e-2 * dens.max()        # empty space stays white
    b.contourf(X, Y, dens, levels=levels, cmap=fl.BLUE_CMAP, alpha=0.85)
    q = np.linspace(-4.6, 4.6, 15)
    QX, QY = np.meshgrid(q, q, indexing="ij")
    qp = np.stack([QX.ravel(), QY.ravel()], 1)
    s = mixture_score(qp)
    n = np.linalg.norm(s, axis=1, keepdims=True)
    s = s * np.minimum(1.4 / n, 1.0)  # clip long arrows in the tails
    b.quiver(qp[:, 0], qp[:, 1], s[:, 0], s[:, 1], angles="xy",
             scale_units="xy", scale=2.2, width=0.0045, color=ORANGE)
    b.set_xlabel("$x_1$", fontsize=LBL)
    b.set_ylabel("$x_2$", fontsize=LBL)
    _square(b, 5)
    fl.save(fig, "mdl-diffusion-score-field")


def fig_noise_ladder():
    """The running example's density after Gaussian perturbation at four noise
    levels.  Barriers between modes fill in as sigma grows, until one broad
    bump remains."""
    g = np.linspace(-9, 9, 301)          # past the window: no cut contours
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    sigmas = [0.0, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.35))
    for ax, sg in zip(axes, sigmas):
        D = np.exp(mixture_logpdf(pts, sigma=sg)).reshape(X.shape)
        levels = np.linspace(0.0, D.max(), 13)
        levels[0] = 1e-2 * D.max()
        ax.contourf(X, Y, D, levels=levels, cmap=fl.BLUE_CMAP)
        ax.plot(MIX_MEANS[:, 0], MIX_MEANS[:, 1], "+", color="black", ms=8, mew=1.3)
        title = r"$\sigma = 0$ (data)" if sg == 0 else rf"$\sigma = {sg:g}$"
        ax.set_title(title, fontsize=13.5, color="black")
        _square(ax, 6, ticks=(-5, 0, 5))
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    fig.tight_layout(w_pad=0.8)
    fl.save(fig, "mdl-diffusion-noise-ladder")


# =========================================================================== #
# 17.2 gaussian-paths.md                                                      #
# =========================================================================== #

def fig_paths():
    """(a) alpha_t (solid) and sigma_t (dashed) of the VP and the linear path;
    (b) their log-SNR, clipped to [-10, 10]."""
    t = np.linspace(0, 1, 801)
    a_vp, s_vp = vp(t)
    a_li, s_li = linear(t)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for (al, sg, c) in ((a_vp, s_vp, BLUE), (a_li, s_li, ORANGE)):
        a.plot(t, al, color=c, lw=2.4)
        a.plot(t, sg, color=c, lw=2.4, ls=(0, (4, 2.5)))
    a.set_xlim(0, 1)
    a.set_ylim(0, 1.04)
    a.set_xlabel("$t$", fontsize=LBL)
    a.set_ylabel(r"$\alpha_t,\ \sigma_t$", fontsize=LBL)
    fl.clean_axes(a, equal=False)
    _black_axes(a)

    tt = np.linspace(1e-4, 1 - 1e-4, 4001)
    b.axhline(0, color="black", lw=0.9)
    b.plot(tt, np.clip(log_snr(vp, tt), -10, 10), color=BLUE, lw=2.4)
    b.plot(tt, np.clip(log_snr(linear, tt), -10, 10), color=ORANGE, lw=2.4)
    b.set_xlim(0, 1)
    b.set_ylim(-10.5, 10.5)
    b.set_yticks([-10, -5, 0, 5, 10])
    b.set_xlabel("$t$", fontsize=LBL)
    b.set_ylabel(r"$\lambda_t = \log(\alpha_t^2/\sigma_t^2)$", fontsize=LBL)
    fl.clean_axes(b, equal=False)
    _black_axes(b)

    handles = [Line2D([], [], color=BLUE, lw=2.4, label="VP path"),
               Line2D([], [], color=ORANGE, lw=2.4, label="linear path"),
               Line2D([], [], color="black", lw=2.0, label=r"$\alpha_t$"),
               Line2D([], [], color="black", lw=2.0, ls=(0, (4, 2.5)),
                      label=r"$\sigma_t$")]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.94), fontsize=TXT, frameon=False,
               handlelength=2.2, columnspacing=1.8)
    _panel_tag(a, "a", y=1.04)
    _panel_tag(b, "b", y=1.04)
    fig.tight_layout(w_pad=3.0)
    fl.save(fig, "mdl-diffusion-paths")


def fig_forward_reverse():
    """A chain of latent variables x_0 ... x_T.  Forward arrows (top, blue) are
    the fixed noising kernels q(x_k | x_{k-1}); reverse arrows (bottom, orange)
    are the learned denoising kernels p_theta(x_{k-1} | x_k)."""
    labels = [r"$\mathbf{x}_0$", r"$\mathbf{x}_1$", r"$\cdots$",
              r"$\mathbf{x}_{k-1}$", r"$\mathbf{x}_k$", r"$\cdots$", r"$\mathbf{x}_T$"]
    xs = np.arange(len(labels)) * 1.6
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    r = 0.47
    for x, lab in zip(xs, labels):
        if lab == r"$\cdots$":
            ax.text(x, 0, lab, ha="center", va="center", fontsize=16, color="black")
            continue
        observed = lab == r"$\mathbf{x}_0$"
        ax.add_patch(plt.Circle((x, 0), r, fc=BLUE if observed else "white",
                                ec="black", lw=1.4, zorder=3,
                                alpha=0.25 if observed else 1))
        if observed:
            ax.add_patch(plt.Circle((x, 0), r, fc="none", ec="black", lw=1.4, zorder=4))
        ax.text(x, 0, lab, ha="center", va="center", fontsize=15, color="black", zorder=5)
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
    ax.text(mid, 1.12, r"forward: $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}) = "
            r"\mathcal{N}(\sqrt{1-\beta_k}\,\mathbf{x}_{k-1},\ \beta_k \mathbf{I})$, fixed",
            ha="center", va="center", fontsize=TXT, color=T.BLUE.dark)
    ax.text(mid, -1.12, r"reverse: $p_{\boldsymbol{\theta}}(\mathbf{x}_{k-1} \mid \mathbf{x}_k) = "
            r"\mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_k, k),\ \nu_k \mathbf{I})$, learned",
            ha="center", va="center", fontsize=TXT, color=T.ORANGE.dark)
    ax.text(xs[0], -1.12, "data", ha="center", va="center", fontsize=TXT, color="black")
    ax.text(xs[-1], -1.12, "noise", ha="center", va="center", fontsize=TXT, color="black")
    ax.set_xlim(xs[0] - 0.9, xs[-1] + 0.9)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-diffusion-forward-reverse")


BP_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def _time_colors(n=5):
    """A dark-to-light blue ramp: t = 0 (data) darkest, t = 1 (noise) lightest."""
    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list("t", ["#0B3C66", BLUE, "#9CC3E0"])
    return [ramp(v) for v in np.linspace(0, 1, n)]


def _clipped(field, k=0.2, cap=0.7):
    """Arrows of length k |v|, capped at `cap` (about one grid spacing)."""
    n = np.linalg.norm(field, axis=1, keepdims=True)
    return field * k * np.minimum(1.0, cap / np.maximum(k * n, 1e-12))


def _panel_tag(ax, tag, x=-0.02, y=1.03):
    ax.text(x, y, f"({tag})", transform=ax.transAxes, fontsize=LBL,
            ha="left", va="bottom", color="black")


def fig_blueprint():
    """The linear path x_t = (1 - t) x0 + t eps on the mixture, exact.
    (a) conditional path of one x0: circles of radius 2 sigma_t around
    alpha_t x0; (b) marginal path: the contour of p_t that holds 80% of its
    mass; (c) the conditional field -(x - x0)/t at t = 0.5; (d) the marginal
    field -u_0.5(x) = -E[eps - x0 | x_t = x]."""
    lim = 4.5
    cols = _time_colors()
    data = data_cloud(400, seed=1)
    x0 = np.array([-2.3, -1.75])          # one clean point of the heaviest mode
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.9))
    (a, b), (c, d) = axes

    # (a) conditional path: centers alpha_t x0 slide to 0, radii 2 sigma_t grow
    a.plot([x0[0], 0], [x0[1], 0], color=GRAY, lw=1.0, ls=":", zorder=1)
    for t, col in zip(BP_TIMES, cols):
        al, sg = linear(t)
        ctr = al * x0
        if sg > 0:
            a.add_patch(Circle(ctr, 2 * sg, fc="none", ec=col, lw=2.2, zorder=2))
        a.plot(*ctr, "o", color=col, ms=5.5, zorder=3)
    a.text(x0[0] - 0.2, x0[1] - 0.15, r"$\mathbf{x}_0$", fontsize=LBL,
           ha="right", va="top", color="black", zorder=5)

    # (b) marginal path: the contour of p_t holding 80% of its mass
    g = np.linspace(-lim, lim, 241)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    cell = (g[1] - g[0]) ** 2
    for t, col in zip(BP_TIMES, cols):
        al, sg = linear(t)
        D = np.exp(mixture_logpdf(pts, sigma=sg, alpha=al))
        srt = np.sort(D)[::-1]
        level = srt[np.searchsorted(np.cumsum(srt) * cell, 0.8)]
        b.contour(X, Y, D.reshape(X.shape), levels=[level], colors=[col],
                  linewidths=2.2)

    # (c), (d): the fields at t = 0.5, the direction a sampler moves
    t = 0.5
    q = np.linspace(-4.0, 4.0, 11)
    QX, QY = np.meshgrid(q, q, indexing="ij")
    qp = np.stack([QX.ravel(), QY.ravel()], 1)
    cond = -(qp - x0) / t
    _, _, x0hat, epshat = mix_posterior(qp, *linear(t))
    marg = -(epshat - x0hat)
    for ax, field in ((c, cond), (d, marg)):
        ax.scatter(*data.T, s=5, color=LIGHT, zorder=1, lw=0)
        _quiver(ax, qp, _clipped(field), 1.0, ORANGE)
    c.plot(*x0, "*", color="black", ms=12, zorder=4)
    c.text(x0[0] - 0.2, x0[1] - 0.25, r"$\mathbf{x}_0$", fontsize=LBL,
           ha="right", va="top", color="black", bbox=WHITE_BOX, zorder=5)

    for ax, tag in zip(axes.ravel(), "abcd"):
        _square(ax, lim)
        _panel_tag(ax, tag)
    handles = [Line2D([], [], color=col, lw=2.4, label=rf"$t = {t:g}$")
               for t, col in zip(BP_TIMES, cols)]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.985), fontsize=TXT, frameon=False,
               handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(w_pad=1.5, h_pad=1.8)
    fl.save(fig, "mdl-diffusion-blueprint")


# =========================================================================== #
# 17.3 image-diffusion.md                                                     #
# =========================================================================== #

def _ubox(ax, x, y, w, h, text, color, fs=TXT, fc="white", zorder=3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=color, lw=1.6, zorder=zorder))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="black",
                zorder=zorder + 1, linespacing=1.15)


def fig_unet():
    """d2l.UNet(ch=64) drawn as a U: two down-sampling stages, a 7 x 7
    bottleneck, two up-sampling stages, skips that concatenate, and one
    embedding of the time and the label added inside every block."""
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    w, h = 2.5, 1.35
    y28, y14, y7 = 0.0, -2.3, -4.6
    B = {"in": (-0.4, y28, "input\n28×28×1", GRAY),
         "e28": (2.9, y28, "block\n28×28×64", BLUE),
         "e14": (4.4, y14, "block\n14×14×128", BLUE),
         "mid": (7.0, y7, "bottleneck\n7×7×128", PURPLE),
         "d14": (9.6, y14, "block\n14×14×64", ORANGE),
         "d28": (11.1, y28, "block\n28×28×64", ORANGE),
         "out": (14.4, y28, "velocity $\\hat{\\mathbf{u}}$\n28×28×1", GRAY)}
    for x, y, text, col in B.values():
        _ubox(ax, x, y, w, h, text, col)

    def flow(a, b, lab=None, side="left"):
        (xa, ya, *_), (xb, yb, *_) = B[a], B[b]
        if ya == yb:                       # horizontal: edge to edge
            p, q = (xa + w / 2, ya), (xb - w / 2, yb)
        elif yb < ya:                      # down: bottom edge to top edge
            p, q = (xa, ya - h / 2), (xb - 0.35, yb + h / 2)
        else:                              # up
            p, q = (xa + 0.35, ya + h / 2), (xb, yb - h / 2)
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(
            arrowstyle="-|>", color="black", lw=1.4, mutation_scale=12,
            shrinkA=1, shrinkB=1))
        if lab:
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            off = -0.15 if side == "left" else 0.15
            ax.text(mx + off, my, lab, ha="right" if side == "left" else "left",
                    va="center", fontsize=TXT, color="black")
    flow("in", "e28")
    flow("e28", "e14", "down", "left")
    flow("e14", "mid", "down", "left")
    flow("mid", "d14", "up", "right")
    flow("d14", "d28", "up", "right")
    flow("d28", "out")
    # skips: horizontal, concatenated into the decoder block
    for a_, b_ in (("e28", "d28"), ("e14", "d14")):
        (xa, ya, *_), (xb, _, *_) = B[a_], B[b_]
        ax.annotate("", xy=(xb - w / 2, ya), xytext=(xa + w / 2, ya),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.7,
                                    ls="--", mutation_scale=12, shrinkA=1,
                                    shrinkB=1))
    ax.text(7.0, y28 + 0.15, "skip connections\n(concatenate)", ha="center",
            va="bottom", fontsize=TXT, color=T.GREEN.dark, linespacing=1.1)
    # the embedding, added inside every block
    yb = -7.1
    _ubox(ax, 7.0, yb, 13.0, 1.75,
          "time $t$ → sinusoidal features,  label $c$ → embedding table (11 rows)\n"
          "summed → MLP → added inside every block", GRAY)
    for k in ("e28", "e14", "mid", "d14", "d28"):
        x, y, *_ = B[k]
        xs_ = {"e28": 1.75, "e14": 3.3, "mid": 7.0, "d14": 10.7, "d28": 12.25}[k]
        ax.annotate("", xy=(xs_, y - h / 2), xytext=(xs_, yb + 0.9),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2,
                                    mutation_scale=10, shrinkA=0, shrinkB=1))
    ax.set_xlim(-1.75, 15.75)
    ax.set_ylim(yb - 1.0, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-diffusion-unet")


def fig_guidance():
    """Classifier-free guidance on the mixture, VP path at t = 0.5: from a
    noisy x_t, the posterior mean of all the data, the posterior mean given
    the top mode c, and the guided x0_null + gamma (x0_c - x0_null) at
    gamma = 2, extrapolated past x0_c.  Filled: p_t; orange: p_t(. | c)."""
    t = 0.5
    al, sg = vp(t)
    xlim, ylim = (-3.2, 4.2), (-2.6, 4.6)
    gx = np.linspace(*xlim, 241)
    gy = np.linspace(*ylim, 241)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    D = np.exp(mixture_logpdf(pts, sigma=sg, alpha=al)).reshape(X.shape)
    top = MIX_MEANS[2:3]
    Dc = np.exp(mixture_logpdf(pts, sigma=sg, alpha=al, means=top,
                               weights=np.ones(1))).reshape(X.shape)
    xt = np.array([[-2.0, 2.5]])
    x_u = mix_posterior(xt, al, sg)[2][0]
    x_c = mix_posterior(xt, al, sg, means=top, weights=np.ones(1))[2][0]
    gamma = 2.0
    x_g = x_u + gamma * (x_c - x_u)
    xt = xt[0]

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    lv = np.linspace(0, D.max(), 7)[1:]
    ax.contourf(X, Y, D, levels=np.r_[lv, D.max() * 1.01], cmap=fl.BLUE_CMAP,
                alpha=0.6, zorder=0)
    ax.contour(X, Y, Dc, levels=Dc.max() * np.array([0.3, 0.6, 0.9]),
               colors=[ORANGE], linewidths=1.3, zorder=1)
    ax.plot([x_u[0], x_g[0]], [x_u[1], x_g[1]], ls=":", color="black", lw=1.1,
            zorder=2)
    for tip, col in ((x_u, BLUE), (x_c, ORANGE), (x_g, RED)):
        fl.arrow(ax, xt, tip, color=col, lw=2.2)
    ax.plot(*xt, "o", color="black", ms=6, zorder=5)
    ax.text(xt[0] - 0.2, xt[1], r"$\mathbf{x}_t$", fontsize=LBL + 1,
            ha="right", va="center", color="black")
    ax.text(x_u[0] - 0.2, x_u[1] - 0.1, r"$\hat{\mathbf{x}}_0^{\varnothing}$",
            fontsize=LBL + 1, ha="right", va="top", color="black", bbox=WHITE_BOX)
    ax.text(x_c[0] + 0.35, x_c[1] + 0.05, r"$\hat{\mathbf{x}}_0^{c}$",
            fontsize=LBL + 1, ha="left", va="center", color="black", bbox=WHITE_BOX)
    ax.text(x_g[0] + 0.25, x_g[1] + 0.1, r"$\hat{\mathbf{x}}_0^{\varnothing} + "
            r"2(\hat{\mathbf{x}}_0^{c} - \hat{\mathbf{x}}_0^{\varnothing})$",
            fontsize=TXT, ha="left", va="center", color="black", bbox=WHITE_BOX)
    ax.text(1.75, -1.85, r"$p_t$", fontsize=LBL, ha="center", va="center",
            color=T.BLUE.dark)
    ax.text(1.75, 1.55, r"$p_t(\cdot \mid c)$", fontsize=LBL, ha="left",
            va="center", color=T.ORANGE.dark)
    ax.set_xlabel("$x_1$", fontsize=LBL)
    ax.set_ylabel("$x_2$", fontsize=LBL)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_yticks([-2, 0, 2, 4])
    fl.clean_axes(ax, equal=True)
    _black_axes(ax)
    fl.save(fig, "mdl-diffusion-guidance")


# =========================================================================== #
# 17.4 samplers.md                                                            #
# =========================================================================== #

def _ddim_paths(starts, eta, steps=200, seed=0):
    """DDIM with the exact mixture predictor on the VP path; every iterate."""
    rng = np.random.default_rng(seed)
    t = np.linspace(1, 0, steps + 1)
    a, s = vp(t)
    x = starts.copy()
    out = [x.copy()]
    for i in range(steps):
        _, _, x0, eps = mix_posterior(x, a[i], s[i])
        rho = s[i + 1] * a[i] / (a[i + 1] * s[i])
        var = (eta * s[i + 1]) ** 2 * (1 - rho ** 2)
        x = a[i + 1] * x0 + np.sqrt(max(s[i + 1] ** 2 - var, 0.0)) * eps
        if var > 0:
            x = x + np.sqrt(var) * rng.standard_normal(x.shape)
        out.append(x.copy())
    return np.stack(out)          # (steps + 1, n, 2)


def fig_two_samplers():
    """Left: DDIM at eta = 0 from six starts; right: eta = 1 from the same
    starts.  Exact predictor, VP path, 200 steps."""
    starts = np.random.default_rng(4).standard_normal((6, 2)) * vp(1.0)[1]
    data = data_cloud(500, seed=2)
    colors = T.CYCLE[:6]
    lim = 4.5
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
    for ax, eta, seed in ((axes[0], 0.0, 0), (axes[1], 1.0, 5)):
        P = _ddim_paths(starts, eta, seed=seed)
        ax.scatter(*data.T, s=4, color=LIGHT, zorder=1, lw=0)
        for j, col in enumerate(colors):
            ax.plot(P[:, j, 0], P[:, j, 1], color=col, lw=1.6 if eta == 0 else 0.8,
                    zorder=2)
            ax.plot(*P[0, j], "o", mfc="white", mec=col, mew=1.8, ms=7, zorder=4)
            ax.plot(*P[-1, j], "x", color=col, mew=2.2, ms=8, zorder=4)
        ax.set_xlabel("$x_1$", fontsize=LBL)
        _square(ax, lim)
    axes[0].set_ylabel("$x_2$", fontsize=LBL)
    axes[1].tick_params(labelleft=False)
    for ax, tag in zip(axes, "ab"):
        ax.text(0.02, 0.98, f"({tag})", transform=ax.transAxes, fontsize=LBL,
                ha="left", va="top", color="black")
    fig.tight_layout(w_pad=1.2)
    fl.save(fig, "mdl-diffusion-two-samplers")


# =========================================================================== #
# 17.5 design-space.md                                                        #
# =========================================================================== #

def _vp_dt_dlam(lam):
    """|dt / d lambda| of the VP path, as a function of lambda."""
    t = np.linspace(1e-6, 1, 200001)
    L = log_snr(vp, t)                       # decreasing in t
    tl = np.interp(lam, L[::-1], t[::-1], left=np.nan, right=np.nan)
    la = vp_log_abar(tl)
    return -np.expm1(la) / (0.1 + 19.9 * tl)


def _density_of_lambda(path, lam):
    """The density of lambda_t under t ~ U(0, 1), numerically."""
    t = np.linspace(1e-7, 1 - 1e-7, 400001)
    L = log_snr(path, t)
    dL = np.gradient(L, t)
    return np.interp(lam, L[::-1], (1 / np.abs(dL))[::-1], left=0, right=0)


def fig_weightings():
    """(a) w(lambda) of five objectives, each scaled to a maximum of 1 on
    [-10, 10], log scale; (b) the density of lambda under uniform t."""
    lam = np.linspace(-10, 10, 2001)
    w_vp = _vp_dt_dlam(lam)
    w_v = np.exp(-lam / 2)
    w_edm = np.exp(-(lam - 2.4) ** 2 / (2 * 2.4 ** 2)) * (np.exp(-lam) + 0.25)
    curves_a = [(np.ones_like(lam), INK, "ELBO", (0.0, 1.0), "bottom"),
                (w_vp / w_vp.max(), BLUE, r"$\boldsymbol{\epsilon}$ loss, VP", None, None),
                (w_v / w_v.max(), ORANGE, "v-pred. (cosine) =\nflow matching", None, None),
                (w_edm / w_edm.max(), GREEN, "EDM", None, None)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for y, col, *_ in curves_a:
        a.plot(lam, y, color=col, lw=2.4)
    a.set_yscale("log")
    a.set_xlim(-10, 10)
    a.set_ylim(3e-5, 3.0)
    a.set_xlabel(r"log-SNR $\lambda$", fontsize=LBL)
    a.set_ylabel(r"$w(\lambda)$, scaled", fontsize=LBL)
    a.set_xticks([-10, -5, 0, 5, 10])
    fl.clean_axes(a, equal=False)
    _black_axes(a)
    a.text(6.0, 1.45, "ELBO", color="black", fontsize=TXT, ha="center", va="bottom")
    a.text(6.6, 0.3, r"$\boldsymbol{\epsilon}$ loss, VP", color=T.BLUE.dark,
           fontsize=TXT, ha="center", va="center")
    a.text(-5.6, 2.5e-3, "v on cosine,\nflow matching", color=T.ORANGE.dark,
           fontsize=TXT, ha="center", va="center")
    a.text(1.8, 0.035, "EDM", color=T.GREEN.dark, fontsize=TXT,
           ha="center", va="center")

    p_vp = _density_of_lambda(vp, lam)
    p_cos = _density_of_lambda(cosine, lam)
    p_lin = 1 / (8 * np.cosh(lam / 4) ** 2)
    p_ln = np.exp(-lam ** 2 / 8) / np.sqrt(8 * np.pi)
    b.plot(lam, p_vp, color=BLUE, lw=2.4)
    b.plot(lam, p_cos, color=PURPLE, lw=2.4)
    b.plot(lam, p_lin, color=ORANGE, lw=2.4)
    b.plot(lam, p_ln, color=ORANGE, lw=2.4, ls=(0, (4, 2.5)))
    b.set_xlim(-10, 10)
    b.set_ylim(0, 0.24)
    b.set_xticks([-10, -5, 0, 5, 10])
    b.set_xlabel(r"log-SNR $\lambda$", fontsize=LBL)
    b.set_ylabel(r"$p(\lambda)$", fontsize=LBL)
    fl.clean_axes(b, equal=False)
    _black_axes(b)
    b.text(-8.6, 0.105, "VP", color=T.BLUE.dark, fontsize=TXT, ha="center")
    b.text(5.0, 0.105, "cosine", color=T.PURPLE.dark, fontsize=TXT, ha="left")
    b.text(5.0, 0.075, "linear", color=T.ORANGE.dark, fontsize=TXT, ha="left")
    b.text(2.4, 0.19, "logit-normal", color=T.ORANGE.dark, fontsize=TXT, ha="left")
    for ax, tag in ((a, "a"), (b, "b")):
        ax.text(-0.02, 1.04, f"({tag})", transform=ax.transAxes, fontsize=LBL,
                ha="left", va="bottom", color="black")
    fig.tight_layout(w_pad=2.5)
    fl.save(fig, "mdl-diffusion-weightings")


def fig_schedules():
    """lambda_t against t for the VP, cosine, linear and shifted cosine paths;
    the band where noise hides the image above 6 cycles per image."""
    t = np.linspace(1e-5, 1, 20001)
    shifted = lambda tt: log_snr(cosine, tt) - np.log(16)
    curves = [(log_snr(vp, t), BLUE, "VP (linear $\\beta$)", "-"),
              (log_snr(cosine, t), PURPLE, "cosine", "-"),
              (log_snr(linear, t), ORANGE, "linear", "-"),
              (shifted(t), PURPLE, "cosine, shifted by $-\\log 16$", (0, (4, 2.5)))]
    lo, hi = -14, 12
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhspan(1.2, 1.6, color=T.GOLD.tint, zorder=0, lw=0)
    ax.axhline(0, color="black", lw=0.9)
    for y, col, lab, ls in curves:
        ax.plot(t, np.clip(y, lo - 1, hi + 1), color=col, lw=2.4, ls=ls, label=lab)
    ax.plot([1.0], [vp_log_abar(1.0) - np.log(-np.expm1(vp_log_abar(1.0)))], "o",
            color=BLUE, ms=8, zorder=5, clip_on=False)
    ax.annotate("", xy=(1.0, lo), xytext=(1.0, lo + 1.6),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2,
                                mutation_scale=12), annotation_clip=False)
    ax.text(0.975, lo + 0.9, r"$-\infty$", fontsize=TXT, ha="right",
            va="center", color="black")
    ax.text(0.97, 1.75, "noise hides the image above\n6 cycles per image",
            fontsize=TXT, ha="right", va="bottom", color="black")
    ax.set_xlim(0, 1)
    ax.set_ylim(lo, hi)
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_xlabel("$t$", fontsize=LBL)
    ax.set_ylabel(r"$\lambda_t$", fontsize=LBL)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=TXT, frameon=False, handlelength=2.2, columnspacing=1.6)
    fl.clean_axes(ax, equal=False)
    _black_axes(ax)
    fl.save(fig, "mdl-diffusion-schedules")


# =========================================================================== #
# 17.6 latent-diffusion.md                                                    #
# =========================================================================== #

def _shirt(n=28):
    """A 28 x 28 toy garment silhouette in [0, 1] (illustration only)."""
    r, c = np.mgrid[0:n, 0:n]
    body = (r >= 7) & (r <= 25) & (c >= 8) & (c <= 19)
    sleeves = (r >= 5) & (r <= 13) & (np.abs(c - 13.5) <= 3 + (r - 3) * 1.1) \
        & (np.abs(c - 13.5) <= 11)
    neck = (r <= 8) & (np.abs(c - 13.5) <= 2.5 - (r - 5) * 0.3)
    shoulders = (r >= 4) & (r <= 8) & (np.abs(c - 13.5) <= 8)
    img = ((body | sleeves | shoulders) & ~neck).astype(float)
    shade = 0.75 + 0.25 * np.cos((c - 13.5) / 5.0)
    return img * shade


def _blur(img, k=1.0):
    x = np.arange(-3, 4)
    g = np.exp(-x ** 2 / (2 * k ** 2))
    g /= g.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, g, "same"), 0, img)
    return np.apply_along_axis(lambda v: np.convolve(v, g, "same"), 1, out)


def _latent_stack(ax, x0, y0, size, arrays, step=0.11, zorder=3):
    """Draw channel arrays (7 x 7 each) as stacked, offset squares."""
    for i, A in enumerate(arrays[::-1]):
        k = len(arrays) - 1 - i
        xo, yo = x0 + k * step, y0 + k * step
        ax.imshow(A, extent=(xo, xo + size, yo, yo + size), cmap=fl.BLUE_CMAP,
                  vmin=-2.2, vmax=2.2, zorder=zorder + i, interpolation="nearest")
        ax.add_patch(Rectangle((xo, yo), size, size, fc="none", ec=INK, lw=1.0,
                               zorder=zorder + i + 0.5))


def fig_latent_pipeline():
    """The two stages of latent diffusion: an encoder from 28 x 28 x 1 pixels
    to a 7 x 7 x 4 latent grid, diffusion on the grid between z_0 and the
    noise z_1, and a decoder back to pixels; the sizes for our images and for
    a 512 x 512 x 3 image below."""
    rng = np.random.default_rng(3)
    img = _shirt()
    rec = _blur(img, 0.7)
    small = img.reshape(7, 4, 7, 4).mean((1, 3))
    z0 = [2.0 * (small - 0.5) * sgn + 0.3 * rng.standard_normal((7, 7))
          for sgn in (1, -1, 0.6, -0.4)]
    z1 = [rng.standard_normal((7, 7)) for _ in range(4)]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    S = 1.7                                   # image side
    L = 1.05                                  # latent side
    ax.imshow(img, extent=(0, S, 0, S), cmap="gray_r", vmin=0, vmax=1.1, zorder=2)
    ax.add_patch(Rectangle((0, 0), S, S, fc="none", ec=INK, lw=1.0, zorder=3))
    ax.imshow(rec, extent=(8.3, 8.3 + S, 0, S), cmap="gray_r", vmin=0, vmax=1.1,
              zorder=2)
    ax.add_patch(Rectangle((8.3, 0), S, S, fc="none", ec=INK, lw=1.0, zorder=3))
    # encoder and decoder as trapezoids
    from matplotlib.patches import Polygon
    yc = S / 2
    enc = [(2.1, 0.05), (3.4, yc - 0.45), (3.4, yc + 0.45), (2.1, S - 0.05)]
    dec = [(6.6, yc - 0.45), (7.9, 0.05), (7.9, S - 0.05), (6.6, yc + 0.45)]
    for poly, label in ((enc, "encoder"), (dec, "decoder")):
        ax.add_patch(Polygon(poly, closed=True, fc=T.SLATE.tint, ec=GRAY, lw=1.6,
                             zorder=2))
        cx = (poly[0][0] + poly[1][0]) / 2
        ax.text(cx, yc, label, ha="center", va="center", fontsize=TXT,
                color="black", zorder=3)
    zx, zy = 4.3, yc - L / 2 - 0.15
    _latent_stack(ax, zx, zy, L, z0)
    ax.annotate("", xy=(2.1, yc), xytext=(S + 0.05, yc), arrowprops=dict(
        arrowstyle="-|>", color="black", lw=1.4, mutation_scale=12))
    ax.annotate("", xy=(zx - 0.05, yc), xytext=(3.4, yc), arrowprops=dict(
        arrowstyle="-|>", color="black", lw=1.4, mutation_scale=12))
    ax.annotate("", xy=(6.6, yc), xytext=(zx + L + 0.45, yc), arrowprops=dict(
        arrowstyle="-|>", color="black", lw=1.4, mutation_scale=12))
    ax.annotate("", xy=(8.25, yc), xytext=(7.9, yc), arrowprops=dict(
        arrowstyle="-|>", color="black", lw=1.4, mutation_scale=12))
    # the diffusion loop on the latent grid
    zy1 = 2.75
    _latent_stack(ax, zx, zy1, L, z1)
    xm = zx + L / 2 + 0.17
    ax.annotate("", xy=(xm - 0.32, zy1 - 0.08), xytext=(xm - 0.32, zy + L + 0.42),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0,
                                mutation_scale=14))
    ax.annotate("", xy=(xm + 0.32, zy + L + 0.42), xytext=(xm + 0.32, zy1 - 0.08),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0,
                                mutation_scale=14))
    ax.text(xm - 0.5, (zy + L + zy1) / 2 + 0.17, "add noise\n(training)",
            ha="right", va="center", fontsize=TXT, color=T.BLUE.dark)
    ax.text(xm + 0.5, (zy + L + zy1) / 2 + 0.17, "denoise\n(sampling)",
            ha="left", va="center", fontsize=TXT, color=T.ORANGE.dark)
    # names
    ax.text(S / 2, S + 0.12, r"$\mathbf{x}$", ha="center", va="bottom",
            fontsize=LBL + 1, color="black")
    ax.text(8.3 + S / 2, S + 0.12, r"$\hat{\mathbf{x}}$", ha="center",
            va="bottom", fontsize=LBL + 1, color="black")
    ax.text(zx + L + 0.5, zy - 0.02, r"$\mathbf{z}_0$", ha="left",
            va="center", fontsize=LBL + 1, color="black")
    ax.text(zx + L + 0.42, zy1 + L / 2, r"$\mathbf{z}_1$", ha="left",
            va="center", fontsize=LBL + 1, color="black")
    # sizes: ours, then a 512 x 512 x 3 image
    rows = [(-0.38, ["28×28×1 = 784", "7×7×4 = 196", "28×28×1 = 784"]),
            (-0.85, ["512×512×3 = 786,432", "64×64×4 = 16,384",
                     "512×512×3 = 786,432"])]
    for y, (l, m, r) in rows:
        ax.text(S / 2, y, l, ha="center", va="center", fontsize=TXT, color="black")
        ax.text(zx + L / 2 + 0.17, y, m, ha="center", va="center", fontsize=TXT,
                color="black")
        ax.text(8.3 + S / 2, y, r, ha="center", va="center", fontsize=TXT,
                color="black")
    ax.plot([-0.6, 10.6], [-0.6, -0.6], color=LIGHT, lw=1.0)
    ax.set_xlim(-0.75, 10.75)
    ax.set_ylim(-1.15, zy1 + L + 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-diffusion-latent-pipeline")


def _arr(ax, p, q, color="black", lw=1.4, mut=12, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(
        arrowstyle="-|>", color=color, lw=lw, mutation_scale=mut, ls=ls,
        shrinkA=0, shrinkB=0))


def _op(ax, x, y, sym, r=0.2, fc="white", ec=INK):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=1.4, zorder=4))
    ax.text(x, y, sym, ha="center", va="center", fontsize=LBL, color="black",
            zorder=5)


def fig_dit_block():
    """d2l.DiT: patchify a 28 x 28 image into 49 tokens; one adaLN-Zero block,
    whose scale a, shift b and gate g for each sublayer come from
    the condition c = time embedding + label embedding through a
    zero-initialized linear map; then a final adaLN, a zero-initialized
    linear layer and depatchify."""
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    img = _shirt()
    # ---- left: patchify -------------------------------------------------- #
    S = 2.0
    ix, iy = 0.1, 3.7
    ax.imshow(img, extent=(ix, ix + S, iy, iy + S), cmap="gray_r", vmin=0,
              vmax=1.1, zorder=2)
    for k in range(8):
        ax.plot([ix + k * S / 7] * 2, [iy, iy + S], color=ORANGE, lw=1.0, zorder=3)
        ax.plot([ix, ix + S], [iy + k * S / 7] * 2, color=ORANGE, lw=1.0, zorder=3)
    ax.text(ix + S / 2, iy + S + 0.12, "28×28 image, 4×4 patches", ha="center",
            va="bottom", fontsize=TXT, color="black")
    _arr(ax, (ix + S / 2, iy - 0.08), (ix + S / 2, 1.05))
    ax.text(ix + S / 2 + 0.12, (iy + 1.05) / 2, "patchify,\nlinear", ha="left",
            va="center", fontsize=TXT, color="black")
    tw = 0.27
    for k in range(6):
        x = ix + 0.05 + k * (tw + 0.06)
        if k == 4:
            ax.text(x + tw / 2, 0.75, "…", ha="center", va="center",
                    fontsize=LBL, color="black")
            continue
        ax.add_patch(Rectangle((x, 0.62), tw, tw, fc=T.BLUE.tint, ec=BLUE,
                               lw=1.2, zorder=3))
    ax.text(ix + S / 2, 0.45, "49 tokens\n+ positions", ha="center", va="top",
            fontsize=TXT, color="black", linespacing=1.1)

    # ---- middle: one adaLN-Zero block ------------------------------------ #
    cx, bw, bh = 4.75, 2.55, 0.5
    px0, px1, py0, py1 = 2.95, 6.55, 0.2, 6.55
    ax.add_patch(FancyBboxPatch((px0, py0), px1 - px0, py1 - py0,
                                boxstyle="round,pad=0,rounding_size=0.15",
                                fc=T.PANEL, ec=T.HAIRLINE, lw=1.4, zorder=0))
    ax.text(px0 + 0.12, py1 - 0.1, "DiT block", ha="left", va="top",
            fontsize=TXT, color="black")
    y_in, y_ln1, y_att, y_g1, y_p1 = 0.65, 1.3, 2.05, 2.75, 3.25
    y_ln2, y_mlp, y_g2, y_p2, y_out = 3.95, 4.7, 5.4, 5.9, 6.75
    ax.plot([ix + S + 0.05, cx, cx], [0.75, 0.75, 0.95], color="black", lw=1.4)
    _arr(ax, (cx, 0.95), (cx, y_ln1 - bh / 2))
    _ubox(ax, cx, y_ln1, bw, bh, "norm, scale, shift", GRAY)
    _ubox(ax, cx, y_att, bw, bh, "self-attention", BLUE)
    _ubox(ax, cx, y_ln2, bw, bh, "norm, scale, shift", GRAY)
    _ubox(ax, cx, y_mlp, bw, bh, "MLP", BLUE)
    for y in (y_g1, y_g2):
        _op(ax, cx, y, r"$\times$", fc=T.ORANGE.tint, ec=ORANGE)
    for y in (y_p1, y_p2):
        _op(ax, cx, y, "+")
    for a_, b_ in ((y_ln1 + bh / 2, y_att - bh / 2), (y_att + bh / 2, y_g1 - 0.2),
                   (y_g1 + 0.2, y_p1 - 0.2), (y_p1 + 0.2, y_ln2 - bh / 2),
                   (y_ln2 + bh / 2, y_mlp - bh / 2), (y_mlp + bh / 2, y_g2 - 0.2),
                   (y_g2 + 0.2, y_p2 - 0.2)):
        _arr(ax, (cx, a_), (cx, b_))
    # residual paths on the left
    rx = 3.25
    for y0_, y1_ in ((0.95, y_p1), (y_p1 + 0.3, y_p2)):
        ax.plot([cx, rx, rx], [y0_, y0_, y1_], color="black", lw=1.4)
        _arr(ax, (rx, y1_), (cx - 0.2, y1_))
    ax.plot([cx, cx], [y_p1 + 0.2, y_p1 + 0.3], color="black", lw=1.4)

    # ---- the condition and its six modulation vectors -------------------- #
    bx = 7.3
    _ubox(ax, 8.15, -0.75, 2.4, 0.95,
          "$\\mathbf{c}$ = time emb.\n+ label emb.", GRAY)
    _ubox(ax, bx, 0.4, 1.2, 0.9, "linear\n(zero)", RED, fc=T.RED.tint)
    _arr(ax, (bx, -0.27), (bx, 0.4 - 0.47))
    ax.plot([bx, bx], [0.4 + 0.47, 6.45], color=GRAY, lw=1.6, zorder=1)
    for y, lab, xt in ((y_ln1, r"$\mathbf{a}_1, \mathbf{b}_1$", cx + bw / 2),
                       (y_g1, r"$\mathbf{g}_1$", cx + 0.22),
                       (y_ln2, r"$\mathbf{a}_2, \mathbf{b}_2$", cx + bw / 2),
                       (y_g2, r"$\mathbf{g}_2$", cx + 0.22)):
        _arr(ax, (bx, y), (xt + 0.02, y), color=GRAY, lw=1.4, mut=11)
        ax.text(6.55, y + 0.07, lab,
                ha="center", va="bottom", fontsize=TXT, color="black")

    # ---- right: final adaLN, linear, depatchify -------------------------- #
    fx = 9.35
    ax.plot([cx, cx, fx], [y_p2 + 0.2, 7.25, 7.25], color="black", lw=1.4)
    yf1, yf2, yf3 = 6.45, 5.55, 4.65
    _arr(ax, (fx, 7.25), (fx, yf1 + bh / 2))
    _ubox(ax, fx, yf1, 2.0, bh, "final adaLN", GRAY)
    _ubox(ax, fx, yf2, 2.0, bh, "linear (zero)", RED, fc=T.RED.tint)
    _ubox(ax, fx, yf3, 2.0, bh, "depatchify", GRAY)
    _arr(ax, (fx, yf1 - bh / 2), (fx, yf2 + bh / 2))
    _arr(ax, (fx, yf2 - bh / 2), (fx, yf3 + bh / 2))
    _arr(ax, (bx, yf1), (fx - 1.02, yf1), color=GRAY, lw=1.4, mut=11)
    oy = 1.85
    _arr(ax, (fx, yf3 - bh / 2), (fx, oy + 2.0 + 0.08))
    ax.add_patch(Rectangle((fx - 1.0, oy), 2.0, 2.0, fc="white", ec=INK, lw=1.0,
                           zorder=2))
    ax.text(fx, oy + 1.0, "velocity $\\hat{\\mathbf{u}}$\n28×28", ha="center",
            va="center", fontsize=TXT, color="black")
    ax.text(fx + 0.2, oy - 0.15, "zero init: every\nblock starts as the\n"
            "identity, and the\nnetwork outputs 0", ha="center", va="top", fontsize=TXT,
            color=T.RED.dark, linespacing=1.1)
    ax.set_xlim(-0.05, 10.5)
    ax.set_ylim(-1.35, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-diffusion-dit-block")


# =========================================================================== #
# 17.7 few-step.md                                                            #
# =========================================================================== #

FM_MEANS = np.array([-1.5, 1.5])
FM_WEIGHTS = np.array([0.6, 0.4])
FM_STD = 0.35


def _fm_cdf(x, t):
    """CDF of x_t = (1 - t) x0 + t eps for the 1-D two-Gaussian mixture."""
    from scipy.special import erf
    a, s = (np.asarray(v, float)[..., None] for v in (1 - t, t))
    sd = np.sqrt((a * FM_STD) ** 2 + s ** 2)
    z = (np.asarray(x)[..., None] - a * FM_MEANS) / (sd * np.sqrt(2))
    return (FM_WEIGHTS * 0.5 * (1 + erf(z))).sum(-1)


def _fm_quantile(q, t, iters=60):
    """x with CDF_t(x) = q, by bisection: the 1-D flow keeps quantiles."""
    lo = np.full(np.broadcast(q, t).shape, -8.0)
    hi = np.full_like(lo, 8.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        below = _fm_cdf(mid, t) < q
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def _fm_velocity(x, t):
    """u_t(x) = E[eps - x0 | x_t = x] on the linear path."""
    m = FM_MEANS[:, None]
    res = mix_posterior(np.atleast_1d(x)[:, None], 1 - t, t, means=m,
                        weights=FM_WEIGHTS, std=FM_STD)
    return (res[3] - res[2])[:, 0]


def _chord(ax, p, q, color=ORANGE, lw=2.2, alpha=1.0, mut=13):
    ax.annotate("", xy=q, xytext=p, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                                shrinkA=3, shrinkB=3, mutation_scale=mut))


def fig_flow_map():
    """Left: probability-flow trajectories of a 1-D two-Gaussian mixture on
    the linear path (exact: the 1-D flow keeps quantiles), t = 1 at the top,
    with one jump of the flow map.  Right: the compositional, Lagrangian and
    Eulerian identities on one trajectory."""
    ts = np.linspace(0, 1, 401)
    qs = (np.arange(16) + 0.5) / 16
    traj = _fm_quantile(qs[:, None], ts[None, :])          # (n_q, n_t)
    xg = np.linspace(-3.0, 3.0, 301)
    a_, s_ = linear(ts)
    sd = np.sqrt((a_[:, None] * FM_STD) ** 2 + s_[:, None] ** 2)
    dens = (FM_WEIGHTS * np.exp(-(xg[:, None, None] - a_[None, :, None] * FM_MEANS) ** 2
                                / (2 * sd[None] ** 2)) / sd[None]).sum(-1)
    dens = dens / dens.max(0, keepdims=True)                # (n_x, n_t), per row

    q = 0.56                                   # a trajectory that bends
    tr = _fm_quantile(np.full_like(ts, q), ts)
    at = lambda tt: (float(np.interp(tt, ts, tr)), tt)

    fig = plt.figure(figsize=(8.4, 5.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1.0], wspace=0.1,
                          hspace=0.4)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(dens.T, origin="lower", extent=(-3, 3, 0, 1), aspect="auto",
              cmap=fl.BLUE_CMAP, alpha=0.45, vmin=0, vmax=1.5, zorder=0)
    for row in traj:
        ax.plot(row, ts, color=T.FAINT, lw=1.0, zorder=1)
    t1, s1 = 0.9, 0.15
    (xt, _), (xs, _) = at(t1), at(s1)
    ax.plot(tr, ts, color=INK, lw=1.8, zorder=2)
    _chord(ax, (xt, t1), (xs, s1))
    for (x, y) in ((xt, t1), (xs, s1)):
        ax.plot(x, y, "o", color=INK, ms=6, zorder=5)
    ax.text(xt + 0.15, t1 + 0.01, r"$(\mathbf{x}_t, t)$", fontsize=LBL,
            ha="left", va="bottom", color="black", bbox=WHITE_BOX, zorder=6)
    ax.text(xs - 0.15, s1 - 0.01, r"$(\mathbf{x}_s, s)$", fontsize=LBL,
            ha="right", va="top", color="black", bbox=WHITE_BOX, zorder=6)
    ax.text(-1.0, 0.58, r"$\Phi_{t \to s}$", fontsize=LBL + 1,
            ha="right", va="center", color=T.ORANGE.dark, bbox=WHITE_BOX, zorder=6)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("$x$", fontsize=LBL)
    ax.set_ylabel("$t$", fontsize=LBL)
    fl.clean_axes(ax, equal=False)
    _black_axes(ax)

    # the three identities, on a schematic bowed trajectory (t up)
    bow = lambda tt: 0.45 * np.sin(np.pi * tt) + 0.6 * tt - 0.3
    at = lambda tt: (float(bow(tt)), tt)
    names = ["compositional", "Lagrangian", "Eulerian"]
    axs = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for axi, name in zip(axs, names):
        axi.plot(bow(ts), ts, color=INK, lw=1.5, zorder=1)
        axi.set_xlim(-0.55, 0.85)
        axi.set_ylim(-0.05, 1.05)
        axi.axis("off")
        axi.set_title(name, fontsize=TXT, loc="left", color="black", pad=3)
    dot = lambda axi, pt: axi.plot(*pt, "o", color=INK, ms=4.5, zorder=5)
    def lab(axi, pt, txt, where):
        dx, dy, ha, va = {"t": (0.05, 0.03, "left", "bottom"),
                          "r": (0.07, 0.0, "left", "center"),
                          "s": (0.05, -0.04, "left", "top")}[where]
        axi.text(pt[0] + dx, pt[1] + dy, txt, fontsize=TXT, ha=ha, va=va,
                 color="black")
    # compositional: t -> r and r -> s make t -> s
    a0 = axs[0]
    P, R, S = at(0.92), at(0.5), at(0.08)
    _chord(a0, P, R, color=BLUE, lw=1.8, mut=11)
    _chord(a0, R, S, color=BLUE, lw=1.8, mut=11)
    _chord(a0, P, S, color=ORANGE, lw=2.0, mut=11)
    for pt, txt in ((P, "t"), (R, "r"), (S, "s")):
        dot(a0, pt)
        lab(a0, pt, f"${txt}$", txt)
    # Lagrangian: the start fixed, the end moves along the trajectory
    a1 = axs[1]
    P = at(0.92)
    for sv, al in ((0.5, 0.4), (0.28, 0.7), (0.08, 1.0)):
        E = at(sv)
        _chord(a1, P, E, color=ORANGE, lw=2.0, alpha=al, mut=11)
        dot(a1, E)
    dot(a1, P)
    lab(a1, P, "$t$", "t")
    lab(a1, at(0.08), "$s$", "s")
    # Eulerian: the end fixed, the start slides along the trajectory
    a2 = axs[2]
    E = at(0.08)
    for tv, al in ((0.92, 1.0), (0.72, 0.7), (0.5, 0.4)):
        P = at(tv)
        _chord(a2, P, E, color=ORANGE, lw=2.0, alpha=al, mut=11)
        dot(a2, P)
    dot(a2, E)
    lab(a2, at(0.92), "$t$", "t")
    lab(a2, E, "$s$", "s")
    fl.save(fig, "mdl-diffusion-flow-map")


# =========================================================================== #
# 17.8 discrete-diffusion.md                                                  #
# =========================================================================== #

def fig_masking():
    """Four snapshots of a nine-token sequence under an absorbing forward
    process: each token is independently replaced by the mask symbol, masked
    tokens stay masked, and at the end every token is masked.  The reverse
    process reveals tokens in the opposite order."""
    rng = np.random.default_rng(11)
    letters = list("diffusion")
    n = len(letters)
    mask_time = rng.permutation(n) + 1
    stages = [0, 3, 6, 9]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    w, h, gap = 0.78, 0.78, 0.22
    y0 = 3.0
    for r, k in enumerate(stages):
        y = y0 - r * 1.15
        for i, ch in enumerate(letters):
            x = i * (w + gap)
            masked = mask_time[i] <= k
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                        fc=(T.MUTED if masked else "white"),
                                        ec="black", lw=1.2))
            ax.text(x + w / 2, y + h / 2, "?" if masked else ch, ha="center", va="center",
                    fontsize=15, color="white" if masked else "black", family="monospace")
        label = {0: r"$\mathbf{x}_0$: data", 3: r"$\mathbf{x}_{T/3}$", 6: r"$\mathbf{x}_{2T/3}$",
                 9: r"$\mathbf{x}_T$: all masked"}[k]
        ax.text(n * (w + gap) + 0.15, y + h / 2, label, ha="left", va="center",
                fontsize=LBL, color="black")
    xa = -0.55
    ax.annotate("", xy=(xa, y0 - 3 * 1.15 + 0.1), xytext=(xa, y0 + h - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0, mutation_scale=14))
    ax.text(xa - 0.15, y0 - 1.15 * 1.5 + h / 2,
            "forward:\nmask each token\nwith probability $\\beta_t$;\nmasked stays masked",
            ha="right", va="center", fontsize=TXT, color=T.BLUE.dark)
    xb = n * (w + gap) + 3.9
    ax.annotate("", xy=(xb, y0 + h - 0.1), xytext=(xb, y0 - 3 * 1.15 + 0.1),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0, mutation_scale=14))
    ax.text(xb + 0.15, y0 - 1.15 * 1.5 + h / 2,
            "reverse:\nreveal masked tokens\nfrom $p_{\\boldsymbol{\\theta}}(\\mathbf{x}_0^i \\mid \\mathbf{x}_t)$;"
            "\nrevealed stays revealed",
            ha="left", va="center", fontsize=TXT, color=T.ORANGE.dark)
    ax.set_xlim(-4.6, n * (w + gap) + 8.9)
    ax.set_ylim(y0 - 3 * 1.15 - 0.3, y0 + h + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-diffusion-masking")


FIGURES = [
    fig_score_field,
    fig_energy_density,
    fig_noise_ladder,
    fig_paths,
    fig_forward_reverse,
    fig_blueprint,
    fig_unet,
    fig_guidance,
    fig_two_samplers,
    fig_weightings,
    fig_schedules,
    fig_latent_pipeline,
    fig_dit_block,
    fig_flow_map,
    fig_masking,
]


def main():
    only = set(sys.argv[1:])
    for f in FIGURES:
        if not only or f.__name__[4:] in only:
            f()
    for p in fl.WRITTEN:
        print("wrote", os.path.relpath(p))


if __name__ == "__main__":
    main()
