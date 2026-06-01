#!/usr/bin/env python3
"""Generate every illustrative figure for the "Mathematics for Deep Learning ->
Probability and Statistics" appendix chapter in the one shared house style.

This reuses the palette, ``plt.rcParams`` (deterministic SVG hash salt), and the
shared drawing helpers defined in ``tools/gen_mdl_figures.py`` -- the same
generator that owns the Linear Algebra figures -- so every figure across the
appendix reads the same and re-runs are byte-for-byte identical (clean git
diffs).  We import that module rather than editing it.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_probability_figures.py

All figures are written to ``img/mdl-prob-<id>.svg``.  Figures that teach by
being seen with real numbers (the joint/marginal array, the significance tail)
use actual distributions and grids, not sketches.  The script is idempotent:
re-running overwrites cleanly.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # noqa: E402  (shared style + helpers)

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Rectangle  # noqa: E402


# =========================================================================== #
# RANDOM VARIABLES (sec_mdl-random_variables): joint -> marginal              #
# =========================================================================== #

def fig_marginal():
    """A continuous joint density p_{X,Y}(x,y) (Blues filled contours); integrating
    it up a vertical strip at fixed x (over all y) collapses it onto the marginal
    p_X(x), the curve beneath.  Real, normalized mixture-of-Gaussians density, so
    the marginal shown is the true integral."""
    xs = np.linspace(-3.4, 3.8, 260)
    ys = np.linspace(-3.0, 3.6, 260)
    X, Y = np.meshgrid(xs, ys)

    def bump(mx, my, sx, sy, rho, w):
        z = ((X - mx) ** 2 / sx ** 2
             - 2 * rho * (X - mx) * (Y - my) / (sx * sy)
             + (Y - my) ** 2 / sy ** 2)
        return w * np.exp(-z / (2 * (1 - rho ** 2))) / (
            2 * np.pi * sx * sy * np.sqrt(1 - rho ** 2))

    joint = bump(0.4, 0.2, 1.0, 0.9, 0.45, 0.65) + bump(-1.2, 1.3, 0.7, 0.8, -0.2, 0.35)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    joint /= joint.sum() * dx * dy
    marg_x = joint.sum(axis=0) * dy            # integrate over y -> p_X(x)
    x0 = 0.4

    fig, (axj, axm) = plt.subplots(
        2, 1, figsize=(5.6, 5.2), sharex=True,
        gridspec_kw=dict(height_ratios=[4, 1.5], hspace=0.12),
    )

    # --- joint density with the integrated strip highlighted ---
    axj.contourf(X, Y, joint, levels=12, cmap="Blues")
    axj.axvspan(x0 - 0.07, x0 + 0.07, color=ORANGE, alpha=0.30)
    axj.axvline(x0, color=ORANGE, lw=2.0)
    axj.text(x0 + 0.15, ys[-1] - 0.3, r"$\int p\,dy$", color=ORANGE, va="top",
             fontsize=10)
    axj.set_ylabel(r"$y$")
    axj.set_title(r"joint density $p_{X,Y}(x,y)$", fontsize=11)

    # --- marginal density curve p_X(x) = integral over y ---
    axm.plot(xs, marg_x, color=BLUE, lw=2.4)
    axm.fill_between(xs, 0, marg_x, color=BLUE, alpha=0.18, lw=0)
    axm.axvline(x0, color=ORANGE, lw=2.0)
    axm.plot([x0], [np.interp(x0, xs, marg_x)], "o", color=ORANGE, ms=6, zorder=5)
    axm.set_xlabel(r"$x$")
    axm.set_ylabel(r"$p_X(x)$")
    axm.set_ylim(0, marg_x.max() * 1.22)
    axm.set_yticks([])
    axm.spines["top"].set_visible(False)
    axm.spines["right"].set_visible(False)
    axm.spines["left"].set_visible(False)

    fl.save(fig, "mdl-prob-marginal")


# =========================================================================== #
# MAXIMUM LIKELIHOOD (sec_mdl-maximum_likelihood): MAP = NLL + log-prior       #
# =========================================================================== #

def fig_map_prior():
    """MAP estimation as a tug-of-war.  The negative log-likelihood (data term)
    is a bowl whose minimum sits at the MLE; a Gaussian log-prior adds an
    upward parabola centered at the prior mean (0); their sum -- the MAP
    objective -- is a bowl whose minimum is pulled from the MLE toward 0.  All
    three curves are real quadratics so the MAP minimum lands exactly at the
    precision-weighted average the proposition predicts."""
    # NLL ~ (theta - theta_mle)^2 / (2 sigma2); prior penalty ~ theta^2/(2 tau2).
    theta_mle = 2.4
    inv_sig2 = 1.0      # data precision  1/sigma^2
    inv_tau2 = 0.7      # prior precision 1/tau^2  (prior mean 0)
    th = np.linspace(-1.2, 4.0, 500)

    nll = 0.5 * inv_sig2 * (th - theta_mle) ** 2          # data term
    prior = 0.5 * inv_tau2 * th ** 2                       # -log prior (Gaussian, mean 0)
    post = nll + prior                                     # MAP objective (up to const)
    theta_map = (inv_sig2 * theta_mle) / (inv_sig2 + inv_tau2)  # exact minimizer

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(th, nll, color=BLUE, lw=2.0, label=r"NLL  (data term)")
    ax.plot(th, prior, color=ORANGE, lw=2.0, ls="--",
            label=r"$-\log p(\theta)$  (Gaussian prior)")
    ax.plot(th, post, color=GREEN, lw=2.6, label=r"MAP objective (sum)")

    # mark the two minima and the pull between them
    ymle = 0.5 * inv_sig2 * (theta_mle - theta_mle) ** 2
    ymap = 0.5 * inv_sig2 * (theta_map - theta_mle) ** 2 + 0.5 * inv_tau2 * theta_map ** 2
    ax.plot([theta_mle], [ymle], "o", color=BLUE, ms=7, zorder=6)
    ax.plot([theta_map], [ymap], "o", color=GREEN, ms=7, zorder=6)
    ax.axvline(0, color=GRAY, lw=0.8, ls=":")
    ax.annotate("", xy=(theta_map, ymap + 0.35), xytext=(theta_mle, ymle + 0.35),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4))
    ax.text((theta_mle + theta_map) / 2, ymap + 0.75, "shrink toward prior",
            color=GRAY, fontsize=9, ha="center")
    ax.text(theta_mle, ymle - 0.55, r"$\hat\theta_{\mathrm{MLE}}$", color=BLUE,
            ha="center", va="top", fontsize=11)
    ax.text(theta_map, -0.55, r"$\hat\theta_{\mathrm{MAP}}$", color=GREEN,
            ha="center", va="top", fontsize=11)
    ax.text(0, -0.55, r"prior mean", color=GRAY, ha="center", va="top",
            fontsize=9)

    ax.set_xlabel(r"parameter $\theta$")
    ax.set_ylabel("objective")
    ax.set_ylim(-0.2, post.max() * 0.62)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc="upper center", fontsize=9)
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-prob-map-prior")


# =========================================================================== #
# STATISTICS (sec_mdl-statistics)                                             #
# =========================================================================== #

def fig_significance():
    """A standard-normal sampling distribution under H0 with the two-tailed
    rejection region (significance level alpha = 0.05) shaded.  The critical
    values are the real Gaussian quantiles +/- z_{0.975}; the central 95%
    acceptance region is shaded faintly, the 2.5% tails in orange."""
    from math import erf, sqrt

    def Phi(z):  # standard-normal CDF via erf (no scipy dependency)
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def pdf(z):
        return np.exp(-z**2 / 2.0) / np.sqrt(2.0 * np.pi)

    # two-sided alpha = 0.05 -> critical value z* with Phi(z*) = 0.975
    zc = 1.959963984540054  # z_{0.975}
    assert abs(Phi(zc) - 0.975) < 1e-6

    z = np.linspace(-4.0, 4.0, 800)
    y = pdf(z)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    # acceptance region (central 95%) faint blue
    mask_acc = np.abs(z) <= zc
    ax.fill_between(z[mask_acc], y[mask_acc], color=BLUE, alpha=0.15, lw=0)
    # rejection tails (2.5% each) orange
    left = z <= -zc
    right = z >= zc
    ax.fill_between(z[left], y[left], color=ORANGE, alpha=0.55, lw=0)
    ax.fill_between(z[right], y[right], color=ORANGE, alpha=0.55, lw=0)

    ax.plot(z, y, color=BLUE, lw=2.0)
    ax.axvline(0, color=GRAY, lw=0.8, ls="--")

    # critical-value markers and labels
    for s in (-1, 1):
        ax.plot([s * zc, s * zc], [0, pdf(zc)], color=ORANGE, lw=1.4)
    ax.text(0, 0.16, r"accept $H_0$" "\n" r"$1-\alpha = 0.95$",
            ha="center", va="center", color=BLUE, fontsize=10)
    ax.annotate(r"reject  ($\alpha/2$)", xy=(2.55, pdf(2.55)),
                xytext=(3.0, 0.16), color=ORANGE, fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate(r"reject  ($\alpha/2$)", xy=(-2.55, pdf(2.55)),
                xytext=(-3.0, 0.16), color=ORANGE, fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.text(zc, -0.012, r"$+z^\ast$", color=ORANGE, ha="center", va="top",
            fontsize=9.5)
    ax.text(-zc, -0.012, r"$-z^\ast$", color=ORANGE, ha="center", va="top",
            fontsize=9.5)

    ax.set_xlabel("test statistic (under $H_0$)")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.45)
    ax.set_yticks([])
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-prob-significance")


def fig_bias_variance_u_curve():
    """The bias-variance decomposition as a U-curve: as model complexity grows,
    bias^2 falls and variance rises, and their sum -- the MSE / test error --
    is a U with a minimum at the sweet spot.  Curves are real functions of a
    complexity axis so the minimum sits exactly where d/dc(bias^2+var)=0."""
    c = np.linspace(0.4, 5.0, 400)          # model-complexity axis
    bias2 = 1.6 / c**1.3                      # bias^2: high when too simple
    var = 0.045 * c**2                        # variance: grows with complexity
    mse = bias2 + var
    cstar = c[np.argmin(mse)]                 # real sweet-spot complexity

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(c, bias2, color=BLUE, lw=2.0, label=r"bias$^2$")
    ax.plot(c, var, color=ORANGE, lw=2.0, label="variance")
    ax.plot(c, mse, color=GREEN, lw=2.6, label="MSE = test error")

    ax.axvline(cstar, color=GRAY, lw=1.0, ls="--")
    ax.plot([cstar], [mse.min()], "o", color=GREEN, ms=7, zorder=6)
    ax.annotate("sweet spot", xy=(cstar, mse.min()),
                xytext=(cstar + 0.95, mse.min() + 1.2), color=GRAY,
                fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))

    # underfit / overfit regime labels, in their corners away from the curves
    ax.text(0.04, 0.50, "underfit\n(high bias)", color=BLUE, fontsize=9,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(0.97, 0.72, "overfit\n(high variance)", color=ORANGE, fontsize=9,
            ha="right", va="center", transform=ax.transAxes)

    ax.set_xlabel("model complexity")
    ax.set_ylabel("error")
    ax.set_ylim(0, mse[0] * 1.05)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-prob-bias-variance-u-curve")


def fig_sampling_distribution():
    """The sampling distribution of an estimator, drawn for two estimators of
    the same parameter theta.  Each panel histograms the estimate hat-theta over
    many independent datasets: the *center* of the histogram relative to the true
    theta is the bias, and the *spread* is the standard error (sqrt variance).
    Left: a low-bias / high-variance estimator (centered on theta, wide).  Right:
    a high-bias / low-variance estimator (offset from theta, narrow).  These are
    real Gaussian sampling distributions, not sketches, so center and spread are
    exact."""
    theta = 0.0
    z = np.linspace(-3.4, 3.4, 800)

    def gauss(x, mu, sd):
        return np.exp(-((x - mu) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi))

    panels = [
        # (bias, se, title)
        (0.0, 1.0, "low bias, high variance"),
        (1.1, 0.45, "high bias, low variance"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, (bias, se, title) in zip(axes, panels):
        mu = theta + bias
        y = gauss(z, mu, se)
        ax.fill_between(z, y, color=BLUE, alpha=0.18, lw=0)
        ax.plot(z, y, color=BLUE, lw=2.0)
        ymax = gauss(mu, mu, se)

        # true theta (dashed gray) and estimator center (orange)
        ax.axvline(theta, color=GRAY, lw=1.1, ls="--")
        ax.text(theta, ymax * 1.20, r"$\theta$", color=GRAY, ha="center",
                fontsize=11)
        ax.plot([mu, mu], [0, ymax], color=ORANGE, lw=1.6)
        ax.text(mu, ymax * 1.03, r"$\mathbb{E}[\hat\theta]$", color=ORANGE,
                ha="center", va="bottom", fontsize=10)

        # bias = horizontal offset between theta and the center
        if abs(bias) > 1e-6:
            ay = ymax * 0.5
            ax.annotate("", xy=(mu, ay), xytext=(theta, ay),
                        arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
            ax.text((theta + mu) / 2, ay * 1.12, "bias", color=ORANGE,
                    ha="center", va="bottom", fontsize=9.5)

        # spread = standard error, drawn as a +/- 1 SE bar low on the curve
        sy = ymax * 0.13
        ax.annotate("", xy=(mu + se, sy), xytext=(mu - se, sy),
                    arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.3))
        ax.text(mu, -ymax * 0.10, r"spread $=$ std. error",
                color=BLUE, ha="center", va="top", fontsize=9.5)

        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(r"$\hat\theta$")
        ax.set_ylim(-ymax * 0.22, ymax * 1.40)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_aspect("auto")
        fl.clean_axes(ax, equal=False)
    axes[0].set_ylabel("density over datasets")
    fl.save(fig, "mdl-prob-sampling-distribution")


def fig_type_i_ii_matrix():
    """The 2x2 hypothesis-test decision matrix.  Rows: H0 true / H0 false.
    Columns: fail to reject / reject H0.  Cells: correct acceptance, type I
    error (rate alpha), type II error (rate beta), correct rejection (power
    1 - beta).  Diagonal (correct decisions) green, off-diagonal (errors)
    orange -- a pure schematic, no axes."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    # cell geometry: 2x2 unit cells, row 0 at top
    cells = {
        # (col, row): (facecolor, alpha, title, sub)
        (0, 1): (GREEN, 0.16, "correct", r"$1-\alpha$"),         # H0 true, fail to reject
        (1, 1): (ORANGE, 0.22, "type I error", r"$\alpha$  (false positive)"),  # H0 true, reject
        (0, 0): (ORANGE, 0.22, "type II error", r"$\beta$  (false negative)"),  # H0 false, fail to reject
        (1, 0): (GREEN, 0.16, "correct", r"power $1-\beta$"),    # H0 false, reject
    }
    for (cx, cy), (fc, al, title, sub) in cells.items():
        ax.add_patch(Rectangle((cx, cy), 1.0, 1.0, facecolor=fc, alpha=al,
                               edgecolor=GRAY, lw=1.4))
        col = GREEN if fc is GREEN else ORANGE
        ax.text(cx + 0.5, cy + 0.60, title, ha="center", va="center",
                fontsize=11, color=col, weight="bold")
        ax.text(cx + 0.5, cy + 0.34, sub, ha="center", va="center",
                fontsize=10, color=col)

    # column headers (the decision)
    ax.text(0.5, 2.16, "fail to reject $H_0$", ha="center", va="center",
            fontsize=10.5)
    ax.text(1.5, 2.16, "reject $H_0$", ha="center", va="center", fontsize=10.5)
    ax.text(1.0, 2.46, "decision", ha="center", va="center", fontsize=10.5,
            color=GRAY, style="italic")

    # row headers (the truth)
    ax.text(-0.12, 1.5, r"$H_0$ true", ha="right", va="center", fontsize=10.5,
            rotation=90)
    ax.text(-0.12, 0.5, r"$H_0$ false", ha="right", va="center", fontsize=10.5,
            rotation=90)
    ax.text(-0.42, 1.0, "truth", ha="right", va="center", fontsize=10.5,
            color=GRAY, style="italic", rotation=90)

    ax.set_xlim(-0.7, 2.05)
    ax.set_ylim(-0.1, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-prob-type-i-ii-matrix")


def _twobump(x):
    """The two-bump mixture density used by the random-variable figures."""
    g = lambda m: np.exp(-0.5 * (x - m) ** 2) / np.sqrt(2 * np.pi)
    return 0.8 * g(-1.0) + 0.2 * g(3.0)


# --- random variables: density / cdf / Chebyshev / covariance / correlation --

def fig_pdf_area():
    """A density p(x): the whole area is 1; the area over (a,b] is P(a<X<=b)."""
    x = np.linspace(-5, 7, 600); p = _twobump(x); a, b = 0.0, 4.0
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.fill_between(x, 0, p, color=BLUE, alpha=0.12)
    m = (x >= a) & (x <= b)
    ax.fill_between(x[m], 0, p[m], color=BLUE, alpha=0.34)
    ax.plot(x, p, color=BLUE, lw=2.2)
    ax.text(1.9, 0.05, r"$P(a<X\leq b)$", ha="center", fontsize=9.5, color=BLUE)
    ax.text(5.4, 0.17, "total area $=1$", ha="center", fontsize=8.5, color=GRAY)
    ax.set_xticks([a, b]); ax.set_xticklabels(["$a$", "$b$"]); ax.set_yticks([])
    ax.set_xlim(-5, 7); ax.set_ylim(0, None)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-pdf-area")


def fig_pdf_cdf():
    """Density vs c.d.f.: the shaded area P(a<X<=b) under p equals the rise
    F(b)-F(a) of the cumulative distribution function."""
    x = np.linspace(-5, 7, 600); p = _twobump(x)
    F = np.cumsum(p) * (x[1] - x[0]); F /= F[-1]
    a, b = 0.0, 4.0; ia, ib = int(np.searchsorted(x, a)), int(np.searchsorted(x, b))
    fig, (axp, axc) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    m = (x >= a) & (x <= b)
    axp.plot(x, p, color=BLUE, lw=2.2)
    axp.fill_between(x[m], 0, p[m], color=BLUE, alpha=0.32)
    axp.set_title(r"density $p$", fontsize=10.5); axp.set_yticks([])
    axp.set_xticks([a, b]); axp.set_xticklabels(["$a$", "$b$"])
    axc.plot(x, F, color=ORANGE, lw=2.2)
    axc.plot([a, a], [0, F[ia]], ":", color=GRAY, lw=1.0)
    axc.plot([b, b], [0, F[ib]], ":", color=GRAY, lw=1.0)
    axc.annotate("", xy=(b + 0.4, F[ib]), xytext=(b + 0.4, F[ia]),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.3))
    axc.text(b + 0.6, (F[ia] + F[ib]) / 2, r"$F(b)-F(a)$", fontsize=8.5,
             color=BLUE, va="center")
    axc.set_title(r"c.d.f. $F$", fontsize=10.5); axc.set_yticks([0, 1])
    axc.set_xticks([a, b]); axc.set_xticklabels(["$a$", "$b$"])
    for ax in (axp, axc):
        ax.set_xlim(-5, 7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-pdf-cdf")


def fig_chebyshev():
    """Three-atom p.m.f. at {-2,0,2} with masses {p,1-2p,p}: the Chebyshev
    bracket of half-width 4*sqrt(2p) contains the +/-2 atoms (p>1/8), exactly
    touches them (p=1/8, the equality case), or excludes them (p<1/8)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    for ax, p, title in zip(axes, [0.2, 0.125, 0.05],
                            [r"$p>1/8$", r"$p=1/8$ (tight)", r"$p<1/8$"]):
        atoms, heights = [-2, 0, 2], [p, 1 - 2 * p, p]
        ax.vlines(atoms, 0, heights, color=BLUE, lw=3)
        ax.plot(atoms, heights, "o", color=BLUE, ms=5)
        hw = 4 * np.sqrt(2 * p)
        ax.annotate("", xy=(hw, -0.07), xytext=(-hw, -0.07),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
        ax.text(0, -0.15, rf"half-width ${hw:.2f}$", ha="center", va="top",
                color=ORANGE, fontsize=8.5)
        ax.set_title(title, fontsize=10.5); ax.set_xlim(-3, 3); ax.set_ylim(-0.22, 1.0)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-chebyshev")


def fig_covariance():
    """Three (X,Y) clouds with negative / zero / positive covariance."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3))
    for ax, c, title in zip(axes, [-0.9, 0.0, 1.2],
                            [r"cov $<0$", r"cov $=0$", r"cov $>0$"]):
        X = rng.standard_normal(220); Y = c * X + rng.standard_normal(220)
        ax.scatter(X, Y, s=10, color=BLUE, alpha=0.5)
        ax.axhline(0, color=GRAY, lw=0.6); ax.axvline(0, color=GRAY, lw=0.6)
        ax.set_title(title, fontsize=10.5); ax.set_xlim(-3.6, 3.6); ax.set_ylim(-4, 4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-covariance")


def fig_correlation():
    """Three clouds at correlation -0.9 / 0 / 0.9 with matched spread: only the
    tilt changes, in contrast to covariance which also tracks scale."""
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3))
    for ax, r, title in zip(axes, [-0.9, 0.0, 0.9],
                            [r"$\rho=-0.9$", r"$\rho=0$", r"$\rho=0.9$"]):
        X = rng.standard_normal(220)
        Y = r * X + np.sqrt(max(1 - r ** 2, 0.0)) * rng.standard_normal(220)
        ax.scatter(X, Y, s=10, color=ORANGE, alpha=0.5)
        ax.axhline(0, color=GRAY, lw=0.6); ax.axvline(0, color=GRAY, lw=0.6)
        ax.set_title(title, fontsize=10.5); ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-correlation")


# --- distributions: family tree + pmf/pdf galleries + MVN contours -----------

def _box(ax, xy, w, h, label, fc=LIGHT):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=fc, edgecolor="black", lw=1.2, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9,
            zorder=4)


def fig_family_tree():
    """Relationship map of the common distributions inside an exponential-family
    envelope.  Solid arrows *construct* or take limits (Bernoulli -> Binomial ->
    {Poisson, Gaussian}; Bernoulli -> Categorical -> Multinomial); dashed green
    arrows mark the *conjugate prior* of each likelihood family (Beta for
    Bernoulli/Binomial, Gamma for Poisson, Dirichlet for Categorical/Multinomial)."""
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.add_patch(Rectangle((-0.35, -0.35), 8.7, 4.05, facecolor=LIGHT, alpha=0.16,
                           edgecolor=GRAY, lw=1.2, ls="--", zorder=1))
    ax.text(0.05, 3.55, "exponential family", ha="left", va="top", color=GRAY,
            fontsize=9, style="italic")
    w, h = 1.95, 0.66
    boxes = {"Bernoulli": (0.0, 1.5), "Binomial": (3.0, 2.6), "Poisson": (6.3, 2.9),
             "Gaussian": (6.3, 1.5), "Categorical": (3.0, 0.3),
             "Multinomial": (6.3, 0.3)}
    priors = {"Beta": (3.0, 4.4), "Gamma": (6.3, 4.4), "Dirichlet": (6.3, -1.4)}
    centers = {k: (x + w / 2, y + h / 2)
               for k, (x, y) in {**boxes, **priors}.items()}
    arrows = [("Bernoulli", "Binomial", r"sum of $n$"),
              ("Binomial", "Poisson", r"$n\to\infty,\ np\to\lambda$"),
              ("Binomial", "Gaussian", r"CLT"),
              ("Bernoulli", "Categorical", r"$K$ outcomes"),
              ("Categorical", "Multinomial", r"sum of $n$")]
    for s, t, lab in arrows:
        c0, c1 = np.array(centers[s]), np.array(centers[t])
        fl.arrow(ax, c0 + (c1 - c0) * 0.17, c1 - (c1 - c0) * 0.17, color=GRAY,
                 lw=1.4, mut=11)
        mid = (c0 + c1) / 2
        ax.text(mid[0], mid[1] + 0.14, lab, ha="center", va="bottom",
                fontsize=7.0, color=GRAY)
    # dashed conjugate-prior arrows (prior -> its likelihood family)
    for s, t in [("Beta", "Binomial"), ("Gamma", "Poisson"),
                 ("Dirichlet", "Multinomial")]:
        c0, c1 = np.array(centers[s]), np.array(centers[t])
        d = c1 - c0
        ax.annotate("", xy=tuple(c1 - d * 0.30), xytext=tuple(c0 + d * 0.30),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5, ls="--"))
    ax.text(-0.3, -1.15, "dashed: conjugate prior", color=GREEN, fontsize=8.5,
            style="italic", va="center", ha="left")
    for k, (x, y) in boxes.items():
        _box(ax, (x, y), w, h, k, fc=("#dbe6f3" if k == "Gaussian" else LIGHT))
    for k, (x, y) in priors.items():
        _box(ax, (x, y), w, h, k, fc="#e7f3e7")        # light green for priors
    ax.set_xlim(-0.5, 8.6); ax.set_ylim(-1.9, 5.2)
    ax.set_aspect("equal"); ax.axis("off")
    fl.save(fig, "mdl-prob-family-tree")


def fig_discrete_pmfs():
    """Gallery of four discrete p.m.f.s (real masses, stem style)."""
    import math
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.7))
    series = [
        ([0, 1], [0.7, 0.3], r"Bernoulli$(0.3)$"),
        (list(range(1, 7)), [1 / 6] * 6, r"Uniform $\{1,\dots,6\}$"),
        (list(range(0, 11)),
         [math.comb(10, k) * 0.4 ** k * 0.6 ** (10 - k) for k in range(11)],
         r"Binomial$(10,0.4)$"),
        (list(range(0, 15)),
         [math.exp(-4) * 4 ** k / math.factorial(k) for k in range(15)],
         r"Poisson$(4)$"),
    ]
    for ax, (ks, ps, title) in zip(axes, series):
        ax.vlines(ks, 0, ps, color=BLUE, lw=2.2)
        ax.plot(ks, ps, "o", color=BLUE, ms=4)
        ax.set_title(title, fontsize=10); ax.set_ylim(0, None); ax.set_yticks([])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-discrete-pmfs")


def fig_continuous_pdfs():
    """Continuous uniform, standard Gaussian, and a unit-variance Laplace: the
    Laplace has a sharper peak and heavier tails than the Gaussian."""
    x = np.linspace(-5, 5, 600)
    uni = np.where(np.abs(x) <= 2, 0.25, 0.0)
    gauss = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    b = 1 / np.sqrt(2)                       # var = 2 b^2 = 1
    lap = np.exp(-np.abs(x) / b) / (2 * b)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(x, uni, color=GRAY, lw=2.0, label=r"Uniform on $[-2,2]$")
    ax.plot(x, gauss, color=BLUE, lw=2.2, label=r"$\mathcal{N}(0,1)$")
    ax.plot(x, lap, color=ORANGE, lw=2.2, label=r"Laplace, var $1$")
    ax.legend(fontsize=9, loc="upper right"); ax.set_yticks([])
    ax.set_xlim(-5, 5); ax.set_ylim(0, None)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-continuous-pdfs")


def fig_mvn_contours():
    """A 2-D Gaussian sample with elliptical density contours; the ellipse axes
    are the covariance eigenvectors scaled by sqrt(eigenvalue)."""
    rng = np.random.default_rng(0)
    cov = np.array([[2.0, 1.1], [1.1, 1.0]])
    L = np.linalg.cholesky(cov)
    pts = (L @ rng.standard_normal((2, 400))).T
    g = np.linspace(-5, 5, 200); X, Y = np.meshgrid(g, g)
    inv = np.linalg.inv(cov)
    Z = np.exp(-0.5 * (inv[0, 0] * X ** 2 + 2 * inv[0, 1] * X * Y + inv[1, 1] * Y ** 2))
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=BLUE, alpha=0.35, zorder=2)
    ax.contour(X, Y, Z, levels=5, colors=GRAY, linewidths=1.0, zorder=3)
    wv, V = np.linalg.eigh(cov)
    for i in range(2):
        fl.arrow(ax, (0, 0), V[:, i] * np.sqrt(wv[i]) * 2.0, color=ORANGE,
                 lw=2.0, mut=12)
    ax.set_aspect("equal"); ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-mvn-contours")


# =========================================================================== #
# NAIVE BAYES (sec_mdl-naive_bayes): the conditional-independence graph        #
# =========================================================================== #

def fig_naive_independence():
    """The naive-Bayes conditional-independence assumption as a graphical model,
    beside the dependence structure it discards.  Left ("naive Bayes"): the label
    node y fans d directed edges out to the feature nodes x_1..x_d and there are
    NO edges among the x_i -- the visual definition of conditional independence
    given y.  Right ("true model"): the same fan-out, plus a couple of x_i--x_j
    edges that a real model would carry, struck out in orange to dramatize what
    the naive assumption throws away.  A pure schematic (no axes)."""
    from matplotlib.patches import Circle

    def node(ax, xy, label, fc=LIGHT):
        ax.add_patch(Circle(xy, 0.30, facecolor=fc, edgecolor="black", lw=1.4,
                            zorder=4))
        ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=10.5,
                zorder=5)

    def edge_into_node(ax, src, dst, color=GRAY, lw=1.6, r=0.30):
        # arrow from node center `src` to node `dst`, stopping at the rims so the
        # head sits cleanly on the circle (not buried inside it).
        s, d = np.asarray(src, float), np.asarray(dst, float)
        u = (d - s) / np.linalg.norm(d - s)
        fl.arrow(ax, s + r * u, d - r * u, color=color, lw=lw, mut=11)

    # feature-node x-positions and the label-node position (shared by both panels)
    xs = np.array([-2.4, -1.2, 0.0, 1.2, 2.4])
    feat_labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$\cdots$", r"$x_d$"]
    y_top, y_bot = 1.7, -1.1
    ynode = (0.0, y_top)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    for ax, title in ((axL, "naive Bayes (assumed)"),
                      (axR, "true dependence (discarded)")):
        # fan-out: y -> every x_i (skip the ellipsis "node")
        for x, lab in zip(xs, feat_labels):
            if lab == r"$\cdots$":
                ax.text(x, y_bot, lab, ha="center", va="center", fontsize=13,
                        zorder=5)
                continue
            edge_into_node(ax, ynode, (x, y_bot))
        node(ax, ynode, r"$y$", fc="#dbe6f3")
        for x, lab in zip(xs, feat_labels):
            if lab == r"$\cdots$":
                continue
            node(ax, (x, y_bot), lab)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-1.9, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")

    # right panel only: a couple of among-feature edges, struck out in orange to
    # show the dependence naive Bayes refuses to model.  Draw the (curved) edge
    # faint and bowed up between the feature nodes, then a bold orange slash
    # across the arc's apex.
    from matplotlib.patches import FancyArrowPatch
    rad = -0.42                 # arc curvature (bows the edge upward)
    struck = [(0, 1), (2, 4)]   # (x_1,x_2) neighbors and (x_3,x_d)
    for i, j in struck:
        a = np.array([xs[i], y_bot]); b = np.array([xs[j], y_bot])
        # trim the endpoints to the node rims so the edge meets the circles
        u = (b - a) / np.linalg.norm(b - a)
        pa, pb = a + 0.30 * u, b - 0.30 * u
        edge = FancyArrowPatch(
            pa, pb, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
            color=GRAY, lw=1.4, ls=(0, (4, 3)), zorder=2)
        axR.add_patch(edge)
        # apex of an arc3 quadratic Bezier sits at the chord midpoint displaced
        # perpendicular by (rad/2)*|chord|; perp of (dx,dy) is (dy,-dx).
        mid = (pa + pb) / 2
        perp = np.array([pb[1] - pa[1], -(pb[0] - pa[0])])
        apex = mid + 0.5 * rad * perp
        # strike it out: a short bold orange slash centered on the apex
        axR.plot([apex[0] - 0.16, apex[0] + 0.16],
                 [apex[1] - 0.16, apex[1] + 0.16],
                 color=ORANGE, lw=3.0, solid_capstyle="round", zorder=6)
    axR.text(0.0, -1.75, "edges among features struck out", ha="center",
             va="bottom", color=ORANGE, fontsize=8.5)

    fl.save(fig, "mdl-prob-naive-independence")


# =========================================================================== #
# RANDOM VARIABLES: conditioning as slicing                                   #
# =========================================================================== #

def fig_conditional_slice():
    """Conditioning as slicing.  Left: the joint density p(x,y) with a horizontal
    slice at y=y0 highlighted.  Right: that slice, renormalized to unit area, is
    the conditional density p(x|y0).  For an *independent* pair the slice shape
    would be the same at every y0; here the pair is correlated, so it shifts."""
    xs = np.linspace(-3.0, 3.4, 220)
    ys = np.linspace(-3.0, 3.2, 220)
    X, Y = np.meshgrid(xs, ys)
    sx, sy, rho = 1.0, 0.9, 0.55                       # correlated -> dependence
    z = (X ** 2 / sx ** 2 - 2 * rho * X * Y / (sx * sy) + Y ** 2 / sy ** 2)
    joint = np.exp(-z / (2 * (1 - rho ** 2)))
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    joint /= joint.sum() * dx * dy
    y0 = 1.0
    j0 = int(np.argmin(np.abs(ys - y0)))
    sl = joint[j0, :]
    cond = sl / (sl.sum() * dx)                        # renormalize -> p(x|y0)

    fig, (axj, axc) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axj.contourf(X, Y, joint, levels=12, cmap="Blues")
    axj.axhline(y0, color=ORANGE, lw=2.4)
    axj.text(3.2, y0 + 0.12, r"$y=y_0$", color=ORANGE, ha="right", va="bottom",
             fontsize=10)
    axj.set_xlabel(r"$x$"); axj.set_ylabel(r"$y$")
    axj.set_title(r"joint $p(x,y)$ with a slice", fontsize=10.5)

    axc.plot(xs, cond, color=ORANGE, lw=2.6)
    axc.fill_between(xs, 0, cond, color=ORANGE, alpha=0.18, lw=0)
    axc.set_xlabel(r"$x$"); axc.set_ylabel(r"$p(x \mid y_0)$")
    axc.set_title("slice renormalized to unit area", fontsize=10.5)
    axc.set_yticks([])
    axc.spines["top"].set_visible(False); axc.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-conditional-slice")


# =========================================================================== #
# MAXIMUM LIKELIHOOD: NLL = cross-entropy = floor + KL                         #
# =========================================================================== #

def fig_mle_kl():
    """Maximum likelihood as KL minimization.  The per-example cross-entropy
    H(p, q) = H(p) + KL(p || q) splits into an irreducible floor H(p) (the data's
    own entropy) and the excess KL(p || q) that training drives toward zero.  As
    the model q improves across steps, the KL slice shrinks and the cross-entropy
    settles onto the floor."""
    steps = np.arange(5)
    floor = 1.0
    kl = 1.6 * np.exp(-0.8 * steps)
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.bar(steps, [floor] * len(steps), color=BLUE, alpha=0.85,
           label=r"$H(\hat p)$  (irreducible floor)")
    ax.bar(steps, kl, bottom=[floor] * len(steps), color=ORANGE, alpha=0.85,
           label=r"$D_{\mathrm{KL}}(\hat p \,\Vert\, p_\theta)$  (removed by training)")
    ax.axhline(floor, color=GRAY, lw=1.0, ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"cross-entropy  $H(\hat p,\, p_\theta)$")
    ax.set_xticks(steps)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-mle-kl")


# =========================================================================== #
# STATISTICS: the bootstrap                                                   #
# =========================================================================== #

def fig_bootstrap():
    """The bootstrap.  From one observed sample we draw many resamples *with
    replacement*, recompute the statistic on each, and read its spread: the
    histogram of replicates hat-theta* approximates the sampling distribution, and
    its central (1-alpha) percentiles give a confidence interval.  Real resampling
    of a fixed, seeded skewed sample (statistic = the median)."""
    rng = np.random.default_rng(0)
    sample = rng.gamma(2.0, 1.0, size=60)
    reps = np.array([np.median(rng.choice(sample, size=60, replace=True))
                     for _ in range(4000)])
    lo, hi = np.percentile(reps, [2.5, 97.5])

    fig, (axs, axh) = plt.subplots(1, 2, figsize=(8.8, 3.6),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.4]))
    # left: observed sample, then two resamples beneath it
    axs.eventplot([sample], colors=[BLUE], lineoffsets=2.2, linelengths=0.7)
    for k in range(2):
        rs = rng.choice(sample, size=60, replace=True)
        axs.eventplot([rs], colors=[GRAY], lineoffsets=1.2 - 0.8 * k,
                      linelengths=0.6)
    axs.text(np.max(sample) * 0.5, 2.85, "observed sample", color=BLUE,
             fontsize=9, ha="center")
    axs.text(np.max(sample) * 0.5, 1.75, "resamples (with replacement)",
             color=GRAY, fontsize=8.5, ha="center")
    axs.set_ylim(-0.5, 3.2); axs.set_yticks([]); axs.set_xlabel("value")
    for sp in ("top", "right", "left"):
        axs.spines[sp].set_visible(False)

    # right: histogram of bootstrap medians + central-95% percentile band
    axh.hist(reps, bins=40, color=BLUE, alpha=0.7)
    axh.axvspan(lo, hi, color=ORANGE, alpha=0.20)
    for v in (lo, hi):
        axh.axvline(v, color=ORANGE, lw=1.8)
    axh.set_title(r"bootstrap replicates $\hat\theta^{*}$", fontsize=10.5)
    axh.set_xlabel(r"$\hat\theta^{*}$  (median)")
    axh.text((lo + hi) / 2, axh.get_ylim()[1] * 0.93, r"central $95\%$",
             color=ORANGE, ha="center", va="top", fontsize=9)
    axh.set_yticks([])
    axh.spines["top"].set_visible(False); axh.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-bootstrap")


FIGURES = [
    # random variables
    fig_marginal,
    fig_conditional_slice,
    fig_pdf_area,
    fig_pdf_cdf,
    fig_chebyshev,
    fig_covariance,
    fig_correlation,
    # distributions
    fig_family_tree,
    fig_discrete_pmfs,
    fig_continuous_pdfs,
    fig_mvn_contours,
    # maximum likelihood
    fig_map_prior,
    fig_mle_kl,
    # naive bayes
    fig_naive_independence,
    # statistics
    fig_significance,
    fig_bias_variance_u_curve,
    fig_sampling_distribution,
    fig_type_i_ii_matrix,
    fig_bootstrap,
]


def main():
    # write into the shared module's WRITTEN list so verification is uniform
    before = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[before:]

    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert os.path.exists(p), f"missing: {p}"
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            head = fh.read(400)
        assert "<svg" in head, f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):34s} {size:>8,d} bytes")

    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
