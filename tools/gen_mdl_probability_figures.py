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
    # label the strip from clear space to its right, with a leader onto it
    axj.annotate(r"integrate up the strip:  $\int p\,\mathrm{d}y$",
                 xy=(x0 + 0.07, 2.3), xytext=(x0 + 0.55, 2.85),
                 color=ORANGE, va="center", ha="left", fontsize=11,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    axj.set_ylabel(r"$y$", fontsize=12)

    # --- marginal density curve p_X(x) = integral over y ---
    axm.plot(xs, marg_x, color=BLUE, lw=2.4)
    axm.fill_between(xs, 0, marg_x, color=BLUE, alpha=0.18, lw=0)
    axm.axvline(x0, color=ORANGE, lw=2.0)
    axm.plot([x0], [np.interp(x0, xs, marg_x)], "o", color=ORANGE, ms=6, zorder=5)
    axm.set_xlabel(r"$x$", fontsize=12)
    axm.set_ylabel(r"$p_X(x)$", fontsize=12)
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
    # the "pull" arrow sits just above the two minima, MLE -> MAP; label it from
    # the clear pocket just left of and above the MAP minimum (between the two
    # bowls' inner arms and the legend) so the text never crosses the green curve.
    ax.annotate("", xy=(theta_map, ymap + 0.30), xytext=(theta_mle, ymle + 0.30),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
    # higher up the bowl the green arms are farther apart, so the label fits
    # cleanly in that wider gap, centered over the MAP minimum.
    ax.text(theta_map, ymap + 1.02, "shrink toward prior",
            color=GRAY, fontsize=11, ha="center", va="bottom")
    ax.text(theta_mle, ymle - 0.42, r"$\hat\theta_{\mathrm{MLE}}$", color=BLUE,
            ha="center", va="top", fontsize=13)
    ax.text(theta_map, -0.42, r"$\hat\theta_{\mathrm{MAP}}$", color=GREEN,
            ha="center", va="top", fontsize=13)
    ax.text(0, -0.42, r"prior mean", color=GRAY, ha="center", va="top",
            fontsize=11)

    ax.set_xlabel(r"parameter $\theta$", fontsize=13)
    ax.set_ylabel("objective", fontsize=12)
    ax.set_ylim(-0.95, post.max() * 0.62)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc="upper center", fontsize=10.5)
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
            ha="center", va="center", color=BLUE, fontsize=11)
    ax.annotate(r"reject  ($\alpha/2$)", xy=(2.55, pdf(2.55)),
                xytext=(3.0, 0.16), color=ORANGE, fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate(r"reject  ($\alpha/2$)", xy=(-2.55, pdf(2.55)),
                xytext=(-3.0, 0.16), color=ORANGE, fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.text(zc, -0.012, r"$+z^\ast$", color=ORANGE, ha="center", va="top",
            fontsize=11)
    ax.text(-zc, -0.012, r"$-z^\ast$", color=ORANGE, ha="center", va="top",
            fontsize=11)

    # integer ticks, but skip +/-2 (right where +/-z* already labels the axis)
    # so the two labels don't stack on top of each other
    ax.set_xticks([-4, -3, -1, 0, 1, 3, 4])
    ax.set_xlabel("test statistic (under $H_0$)")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.45)
    ax.set_yticks([])
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-prob-significance")


def fig_power_curves():
    """Power of the one-sample two-sided z-test (H0: mu = 0, sigma = 1, at
    alpha = 0.05) as a function of the sample size n, one curve per true
    effect size delta:  power(n; delta) = Phi(delta sqrt(n) - z*) +
    Phi(-delta sqrt(n) - z*) with z* = z_{0.975}.  Every curve starts near
    alpha and climbs to 1; the n needed to hit the conventional target 0.8
    scales like 1/delta^2 -- about 8 samples for delta = 1 but ~8 x 10^4 for
    delta = 0.01."""
    from math import erf, sqrt

    def Phi(z):  # standard-normal CDF via erf (no scipy dependency)
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    zc = 1.959963984540054  # z_{0.975}, two-sided alpha = 0.05
    n = np.logspace(0.0, np.log10(2e5), 600)

    def power(delta):
        return np.array([Phi(delta * sqrt(v) - zc) + Phi(-delta * sqrt(v) - zc)
                         for v in n])

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    # one curve per effect size, a blue family from dark (delta=1) to light
    deltas = [1.0, 0.5, 0.1, 0.01]
    blues = ["#114e7c", BLUE, "#5b9bc9", "#9cc2dd"]
    label_at = [3.0, 17.0, 300.0, 3.0e4]      # n where each curve label sits
    # each curve is still climbing steeply at its own label's n, so all but
    # the widely-spaced delta=1 label need a much bigger vertical clearance;
    # anchoring at the label's *right* edge (ha="right") also keeps the rest
    # of the text over the curve's lower, already-cleared, left side
    label_dy = [0.045, 0.16, 0.16, 0.16]
    for delta, col, ln, dy in zip(deltas, blues, label_at, label_dy):
        pw = power(delta)
        ax.plot(n, pw, color=col, lw=2.2)
        lab = rf"$\delta={delta:g}$"
        ax.text(ln, np.interp(ln, n, pw) + dy, lab, color=col,
                fontsize=11, ha="right", va="bottom")

    # the conventional power target and the test's size alpha
    ax.axhline(0.8, color=GRAY, lw=1.1, ls="--")
    ax.text(1.3, 0.825, "target $0.8$", color=GRAY, fontsize=11, va="bottom")
    ax.axhline(0.05, color=GRAY, lw=1.0, ls=":")
    ax.text(2.1, 0.075, r"$\alpha=0.05$", color=GRAY, fontsize=11, va="bottom")

    # where the extreme curves cross the target (power is increasing in n).
    # n=8's own curve is nearly flat by n=8 (labelled straight above it); the
    # n=8e4 curve is still steeply rising there, so that label is shifted well
    # to its left (in log-n) where the curve is far below the label's height.
    for delta, col, lab, lx, ly in ((1.0, blues[0], r"$n\approx 8$", 1.0, 0.95),
                                    (0.01, blues[3], r"$n\approx 8\times 10^4$",
                                     0.12, 0.71)):
        pw = power(delta)
        nstar = float(np.interp(0.8, pw, n))
        ax.plot([nstar], [0.8], "o", color=col, ms=6, zorder=6)
        ax.text(nstar * lx, ly, lab, color=col, fontsize=11, ha="center")

    ax.set_xscale("log")
    ax.set_xlim(1, 2e5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("power")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-power")


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
                xytext=(cstar + 1.05, mse.min() + 1.25), color=GRAY,
                fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))

    # underfit / overfit regime labels, tucked into the low corners clear of the
    # curves (bias^2 sweeps high on the left, variance rises on the right).
    ax.text(0.035, 0.14, "underfit\n(high bias)", color=BLUE, fontsize=11,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(0.97, 0.70, "overfit\n(high variance)", color=ORANGE, fontsize=11,
            ha="right", va="center", transform=ax.transAxes)

    ax.set_xlabel("model complexity", fontsize=12)
    ax.set_ylabel("error", fontsize=12)
    ax.set_ylim(0, mse[0] * 1.05)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc="upper center", ncol=3, fontsize=10.5)
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

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    for ax, (bias, se, title) in zip(axes, panels):
        mu = theta + bias
        y = gauss(z, mu, se)
        ax.fill_between(z, y, color=BLUE, alpha=0.18, lw=0)
        ax.plot(z, y, color=BLUE, lw=2.0)
        ymax = gauss(mu, mu, se)

        # true theta (dashed gray) and estimator center (orange).  When theta and
        # the center coincide (no bias) the two labels would stack, so offset
        # them to opposite sides of the shared line; otherwise label each in place.
        ax.axvline(theta, color=GRAY, lw=1.1, ls="--")
        ax.plot([mu, mu], [0, ymax], color=ORANGE, lw=1.6)
        if abs(bias) < 1e-6:
            ax.text(theta - 0.18, ymax * 1.12, r"$\theta$", color=GRAY,
                    ha="right", va="bottom", fontsize=13)
            ax.text(mu + 0.18, ymax * 1.12, r"$\mathbb{E}[\hat\theta]$",
                    color=ORANGE, ha="left", va="bottom", fontsize=12)
        else:
            # theta label sits off to the side of its own dashed line, not
            # centered on top of it
            ax.text(theta - 0.15, ymax * 1.12, r"$\theta$", color=GRAY,
                    ha="right", va="bottom", fontsize=13)
            ax.text(mu, ymax * 1.06, r"$\mathbb{E}[\hat\theta]$", color=ORANGE,
                    ha="center", va="bottom", fontsize=12)

        # bias = horizontal offset between theta and the center; the label sits
        # to the left of the arrow's left tip, at the arrow's own height, so it
        # never crosses the arrow shaft.
        if abs(bias) > 1e-6:
            ay = ymax * 0.5
            ax.annotate("", xy=(mu, ay), xytext=(theta, ay),
                        arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
            ax.text(min(theta, mu) - 0.14, ay, "bias", color=ORANGE,
                    ha="right", va="center", fontsize=11)

        # spread = standard error, drawn as a +/- 1 SE bar low on the curve
        sy = ymax * 0.13
        ax.annotate("", xy=(mu + se, sy), xytext=(mu - se, sy),
                    arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.3))
        ax.text(mu, -ymax * 0.10, r"spread $=$ std. error",
                color=BLUE, ha="center", va="top", fontsize=11)

        ax.set_xlabel(r"$\hat\theta$", fontsize=13)
        ax.set_ylim(-ymax * 0.24, ymax * 1.34)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_aspect("auto")
        fl.clean_axes(ax, equal=False)
        # a density is nonnegative: pin the visible x-axis flush to y=0 (not to
        # the bottom of the ylim, which would leave a white gap under the curve).
        # Moving the spine also drags matplotlib's default xlabel placement up
        # to hug it (colliding with the "spread" caption below), so re-pin the
        # label itself, in axes-fraction coordinates, to the bottom margin.
        ax.spines["bottom"].set_position(("data", 0))
        ax.xaxis.set_label_coords(0.5, -0.09)
    axes[0].set_ylabel("density over datasets", fontsize=12)
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
                fontsize=9, color=col)

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
    """A density p(x): the whole area is 1; the area over (a,b] is P(a<X<=b).
    The interval probability is labelled with a leader pointing down into the
    shaded slab (kept off the curve); the total-area note labels the faint
    full-curve fill out in the right tail where it is unambiguous."""
    x = np.linspace(-5, 7, 600); p = _twobump(x); a, b = 0.0, 4.0
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.fill_between(x, 0, p, color=BLUE, alpha=0.12)
    m = (x >= a) & (x <= b)
    ax.fill_between(x[m], 0, p[m], color=BLUE, alpha=0.34)
    ax.plot(x, p, color=BLUE, lw=2.4)

    # P(a<X<=b): label sits in the clear white space above the valley between the
    # two bumps, with a short leader down into the dark slab -- never on the curve.
    ax.annotate(r"$P(a<X\leq b)$", xy=(2.0, 0.055), xytext=(2.0, 0.165),
                ha="center", va="bottom", fontsize=15, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    # total-area note: label the faint full fill in the right tail (clear space).
    ax.annotate("total area $=1$", xy=(5.4, 0.012), xytext=(5.4, 0.115),
                ha="center", va="bottom", fontsize=12.5, color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))

    ax.set_xticks([a, b]); ax.set_xticklabels(["$a$", "$b$"], fontsize=14)
    ax.set_yticks([])
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
    axp.set_yticks([])
    axp.set_xticks([a, b]); axp.set_xticklabels(["$a$", "$b$"])
    axc.plot(x, F, color=ORANGE, lw=2.2)
    axc.plot([a, a], [0, F[ia]], ":", color=GRAY, lw=1.0)
    axc.plot([b, b], [0, F[ib]], ":", color=GRAY, lw=1.0)
    axc.annotate("", xy=(b + 0.4, F[ib]), xytext=(b + 0.4, F[ia]),
                 arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.3))
    axc.text(b + 0.6, (F[ia] + F[ib]) / 2, r"$F(b)-F(a)$", fontsize=11,
             color=BLUE, va="center")
    axc.set_yticks([0, 1])
    axc.set_xticks([a, b]); axc.set_xticklabels(["$a$", "$b$"])
    axp.set_ylim(0, None)
    axc.set_ylim(0, None)
    for ax in (axp, axc):
        ax.set_xlim(-5, 7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        # both curves are nonnegative starting at y=0: pin the x-axis flush
        # there instead of leaving it at the autoscaled ylim floor.
        ax.spines["bottom"].set_position(("data", 0))
    fl.save(fig, "mdl-prob-pdf-cdf")


def fig_inverse_transform():
    """Inverse-transform sampling as a picture.  The c.d.f. of the two-bump
    mixture density (the section's running shape); evenly spread levels U on
    the vertical axis reflect through the curve down to x = F^{-1}(U) on the
    horizontal axis.  Where F is steep (density high) the reflected points
    cluster densely; one level's path is highlighted with arrows.  The curve
    and every reflected point are computed from the real c.d.f."""
    x = np.linspace(-5, 7, 600)
    p = _twobump(x)
    F = np.cumsum(p) * (x[1] - x[0]); F /= F[-1]
    U = np.linspace(0.05, 0.95, 16)                  # evenly spread levels
    xs = np.interp(U, F, x)                          # x = F^{-1}(U)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(x, F, color=ORANGE, lw=2.4, zorder=3)
    for u, xv in zip(U, xs):
        ax.plot([-5, xv], [u, u], color=GRAY, lw=0.7, ls=":", zorder=1)
        ax.plot([xv, xv], [0, u], color=GRAY, lw=0.7, ls=":", zorder=1)
    ax.plot(np.full_like(U, -5), U, "o", color=BLUE, ms=4, zorder=4,
            clip_on=False)
    ax.plot(xs, np.zeros_like(xs), "o", color=BLUE, ms=4, zorder=4,
            clip_on=False)

    # highlight one level's path with arrows: in along u0, then down to x0
    k = 11
    u0, x0 = U[k], xs[k]
    fl.arrow(ax, (-5, u0), (x0, u0), color=BLUE, lw=1.6, mut=12)
    fl.arrow(ax, (x0, u0), (x0, 0.015), color=BLUE, lw=1.6, mut=12)
    ax.text(-4.75, u0 + 0.03, r"$U$", color=BLUE, fontsize=13, ha="left",
            va="bottom")
    ax.text(x0 + 0.15, 0.045, r"$x=F^{-1}(U)$", color=BLUE, fontsize=12,
            ha="left", va="bottom")
    ax.text(5.4, np.interp(5.4, x, F) - 0.06, r"c.d.f. $F$", color=ORANGE,
            fontsize=12, ha="left", va="top")
    # point at the steep stretch where reflected samples pile up; anchored in
    # the clear lower-right pocket so the leader crosses no other label/line
    ax.annotate("steep $F$ = high density:\nsamples cluster here",
                xy=(-1.3, 0.10), xytext=(2.7, 0.62), color=GRAY, fontsize=11,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))

    ax.set_xlim(-5, 7); ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 1])
    ax.set_xticks([])
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel("probability level", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-inverse-transform")


def fig_chebyshev():
    """Three-atom p.m.f. at {-2,0,2} with masses {p,1-2p,p}: the Chebyshev
    event |X-mu| >= alpha*sigma has half-width 4*sqrt(2p).  For p>1/8 the
    half-width exceeds 2, so no atom lies in the event; for p=1/8 the +/-2
    atoms sit exactly at the boundary and carry mass 2p = 1/4, matching the
    bound -- the equality case; for p<1/8 they lie strictly beyond it with
    mass 2p < 1/4.  Atoms inside the event are drawn orange."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    for ax, p, title in zip(axes, [0.2, 0.125, 0.05],
                            [r"$p>1/8$", r"$p=1/8$ (equality)", r"$p<1/8$"]):
        atoms, heights = [-2, 0, 2], [p, 1 - 2 * p, p]
        ax.vlines(atoms, 0, heights, color=BLUE, lw=3, zorder=3)
        # the +/-2 atoms belong to the event {|X| >= half-width} iff p <= 1/8
        edge_col = ORANGE if p <= 0.125 else BLUE
        ax.plot([-2, 2], [p, p], "o", color=edge_col, ms=6, zorder=4)
        ax.plot([0], [1 - 2 * p], "o", color=BLUE, ms=6, zorder=4)
        hw = 4 * np.sqrt(2 * p)
        # the +/-2, 0 tick labels sit just under the axis; drop the arrow (and
        # its caption) further down still so neither touches those digits.
        ax.annotate("", xy=(hw, -0.20), xytext=(-hw, -0.20),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
        ax.text(0, -0.27, rf"half-width ${hw:.2f}$", ha="center", va="top",
                color=ORANGE, fontsize=11)
        ax.set_xlim(-3, 3); ax.set_ylim(-0.38, 1.02)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=11, pad=1.5)
        # pin the x-axis to y=0 so the atoms visibly stand on the baseline
        ax.spines["bottom"].set_position(("data", 0))
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fl.save(fig, "mdl-prob-chebyshev")


def fig_covariance():
    """Covariance carries units.  Top row: three (X,Y) clouds with covariance
    tuned negative / zero / positive, Y measured in dollars.  Bottom row: the
    *identical* draws with Y restated in cents (Y x 100) -- every sample
    covariance is multiplied by 100, while each panel's correlation rho is
    unchanged.  Only the sign of covariance is scale-free."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.2))
    for j, c in enumerate([-0.9, 0.0, 1.2]):
        X = rng.standard_normal(220); Y = c * X + rng.standard_normal(220)
        rho = np.corrcoef(X, Y)[0, 1]          # scale-free: same in both rows
        for i, scale in enumerate([1.0, 100.0]):
            ax = axes[i, j]
            Ys = Y * scale
            cov = np.cov(X, Ys)[0, 1]          # sample covariance (units!)
            ax.scatter(X, Ys, s=10, color=BLUE, alpha=0.5)
            ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
            ax.set_title(rf"cov $\approx {cov:.2f}$", fontsize=11)
            ax.text(0.04, 0.93, rf"$\rho \approx {rho:.2f}$", color=GRAY,
                    fontsize=11, ha="left", va="top", transform=ax.transAxes,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                              pad=1.5), zorder=6)
            ax.set_xlim(-3.6, 3.6)
            ax.set_ylim(-4 * scale, 4 * scale)
            ax.set_yticks([-3 * scale, 0, 3 * scale])
            ax.set_xticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel(r"$Y$ in dollars", fontsize=11)
    axes[1, 0].set_ylabel(r"same data, $Y$ in cents", fontsize=11)
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
        ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-correlation")


# --- distributions: family tree + pmf/pdf galleries + MVN contours -----------

def _box(ax, xy, w, h, label, fc=LIGHT):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=fc, edgecolor="black", lw=1.2, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11,
            zorder=4)


def fig_family_tree():
    """Relationship map of the common distributions inside an exponential-family
    envelope.  Solid arrows *construct* or take limits (Bernoulli -> Binomial ->
    {Poisson, Gaussian}; Poisson -> Exponential as the waiting time between its
    events; Bernoulli -> Categorical -> Multinomial); dashed green arrows mark
    the *conjugate prior* of each likelihood family (Beta for Bernoulli/Binomial,
    Gamma for Poisson, Dirichlet for Categorical/Multinomial)."""
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    w, h = 1.95, 0.66
    boxes = {"Bernoulli": (0.0, 1.5), "Binomial": (3.0, 2.6), "Poisson": (6.3, 2.9),
             "Exponential": (9.3, 2.9),
             "Gaussian": (6.3, 1.5), "Categorical": (3.0, 0.3),
             "Multinomial": (6.3, 0.3)}
    priors = {"Beta": (2.6, 4.4), "Gamma": (6.3, 4.4), "Dirichlet": (6.3, -1.4)}
    centers = {k: (x + w / 2, y + h / 2)
               for k, (x, y) in {**boxes, **priors}.items()}
    arrows = [("Bernoulli", "Binomial", r"sum of $n$"),
              ("Binomial", "Poisson", r"$n\to\infty,\ np\to\lambda$"),
              ("Binomial", "Gaussian", r"CLT"),
              ("Poisson", "Exponential", "waiting time"),
              ("Bernoulli", "Categorical", r"$K$ outcomes"),
              ("Categorical", "Multinomial", r"sum of $n$")]
    # Bernoulli->Binomial and Bernoulli->Categorical both leave from the
    # crowded Bernoulli corner, right where the Beta->Bernoulli conjugate-prior
    # arrow also lands; hand-place these two in the open wedge *between* the
    # two edges (below the former, above the latter) instead of the generic
    # perpendicular offset, which collides with that traffic.
    explicit_pos = {
        ("Bernoulli", "Binomial"): (2.55, 1.98),
        ("Bernoulli", "Categorical"): (3.3, 1.55),
        # float above both box tops: centered on the arrow would bury the
        # trailing lambda behind the Poisson box, so lift it clear.
        ("Binomial", "Poisson"): (5.25, 3.9),
    }
    for s, t, lab in arrows:
        c0, c1 = np.array(centers[s]), np.array(centers[t])
        fl.arrow(ax, c0 + (c1 - c0) * 0.17, c1 - (c1 - c0) * 0.17, color=GRAY,
                 lw=1.4, mut=11)
        mid = (c0 + c1) / 2
        if lab == "waiting time":
            # narrow gap between the Poisson and Exponential boxes: put the
            # label just above the box tops, centered on the visible gap
            ax.text((boxes[s][0] + w + boxes[t][0]) / 2, boxes[t][1] + h + 0.10,
                    lab, ha="center", va="bottom", fontsize=9.5, color="black")
        elif (s, t) in explicit_pos:
            lx, ly = explicit_pos[(s, t)]
            ax.text(lx, ly, lab, ha="center", va="center", fontsize=9.5,
                    color="black")
        else:
            # offset perpendicular to the edge (not just vertically) so the
            # label clears the line along its whole width, whatever the
            # edge's slope; always to the "upper" side of the edge.
            d = c1 - c0
            perp = np.array([-d[1], d[0]])
            perp = perp / np.linalg.norm(perp)
            if perp[1] < 0:
                perp = -perp
            lab_pos = mid + perp * 0.30
            ax.text(lab_pos[0], lab_pos[1], lab, ha="center", va="bottom",
                    fontsize=9.5, color="black")
    # dashed conjugate-prior arrows (prior -> its likelihood families)
    for s, t in [("Beta", "Bernoulli"), ("Beta", "Binomial"),
                 ("Gamma", "Poisson"), ("Dirichlet", "Categorical"),
                 ("Dirichlet", "Multinomial")]:
        c0, c1 = np.array(centers[s]), np.array(centers[t])
        d = c1 - c0
        ax.annotate("", xy=tuple(c1 - d * 0.30), xytext=tuple(c0 + d * 0.30),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5, ls="--"))
    for k, (x, y) in boxes.items():
        _box(ax, (x, y), w, h, k, fc=("#dbe6f3" if k == "Gaussian" else LIGHT))
    for k, (x, y) in priors.items():
        _box(ax, (x, y), w, h, k, fc="#e7f3e7")        # light green for priors
    ax.set_xlim(-0.5, 11.9); ax.set_ylim(-2.1, 5.6)
    ax.set_aspect("equal"); ax.axis("off")
    fl.save(fig, "mdl-prob-family-tree")


def fig_beta_posterior():
    """Bayesian updating as sharpening.  Three panels of the Beta density over a
    coin's bias: the flat Beta(1,1) prior; the Beta(10,5) posterior after
    observing 9 heads in 13 flips; and the Beta(91,41) posterior after 90 heads
    in 130 flips.  The true bias theta* = 0.7 is the dashed line; the exact Beta
    densities (via lgamma, no scipy) visibly pile up on it, their width
    shrinking like 1/sqrt(n)."""
    from math import lgamma

    def beta_pdf(p, a, b):
        logB = lgamma(a) + lgamma(b) - lgamma(a + b)
        return np.exp((a - 1) * np.log(p) + (b - 1) * np.log(1 - p) - logB)

    p = np.linspace(1e-4, 1 - 1e-4, 600)
    theta = 0.7
    panels = [(1, 1, "prior"),
              (10, 5, "9 H in 13 flips"),
              (91, 41, "90 H in 130 flips")]
    ymax = beta_pdf(p, 91, 41).max() * 1.12

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), sharex=True)
    for ax, (a, b, sub) in zip(axes, panels):
        y = beta_pdf(p, a, b)
        ax.plot(p, y, color=BLUE, lw=2.2)
        ax.fill_between(p, 0, y, color=BLUE, alpha=0.15, lw=0)
        ax.axvline(theta, color=ORANGE, lw=1.6, ls="--")
        # the longest caption ("90 H in 130 flips") is wide enough at this
        # bigger font that starting at the old x=0.03 ran it into the
        # theta* dashed line; start it right at the left spine instead.
        ax.text(0.01, ymax * 0.93, sub, color=GRAY, fontsize=13, ha="left",
                va="top")
        ax.set_xlim(0, 1); ax.set_ylim(0, ymax)
        ax.set_xticks([0, theta, 1.0])
        ax.set_xticklabels(["$0$", r"$\theta^{\ast}$", "$1$"], fontsize=13)
        ax.set_yticks([])
        ax.set_xlabel(r"$p$", fontsize=14)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("density", fontsize=13)
    fl.save(fig, "mdl-prob-beta-posterior")


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
        ax.set_ylim(0, None); ax.set_yticks([])
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
    ax.legend(fontsize=11, loc="upper right"); ax.set_yticks([])
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
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-1.9, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")

    # right panel only: a couple of among-feature edges, struck out in orange to
    # show the dependence naive Bayes refuses to model.  Draw the (curved) edge
    # faint and bowed up between the feature nodes, then a bold orange slash
    # across the arc's apex.
    from matplotlib.patches import FancyArrowPatch
    # arc curvature per edge (bows the edge upward); the long (x_3,x_d) arc
    # bows higher (rad=-0.6) so it clears the "..." glyph sitting between them
    struck = [(0, 1, -0.42), (2, 4, -0.6)]   # (x_1,x_2) neighbors, (x_3,x_d)
    for i, j, rad in struck:
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
             va="bottom", color=ORANGE, fontsize=11)

    fl.save(fig, "mdl-prob-naive-independence")


def fig_naive_genvdisc():
    """The generative vs. discriminative routes to a classifier.  Left: a
    generative model (naive Bayes) learns the prior-weighted class-conditional
    densities p(x|y=k) p(y=k) -- here two Gaussians with means -/+1.2, sd 0.8,
    priors 1/2 -- and Bayes' rule flips them into a posterior; the decision
    flips where the weighted curves tie (x* = 0).  Right: a discriminative
    model (softmax regression) never models the inputs at all -- it spends its
    capacity directly on the boundary between the classes."""
    fig, (axg, axd) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # --- left: prior-weighted class-conditional densities on a 1-D axis -----
    x = np.linspace(-3.8, 3.8, 600)
    sd, prior = 0.8, 0.5

    def wdens(mu):
        return prior * np.exp(-((x - mu) ** 2) / (2 * sd ** 2)) / (
            sd * np.sqrt(2 * np.pi))

    y0, y1 = wdens(-1.2), wdens(1.2)
    ymax = y0.max()
    for y, col in ((y0, BLUE), (y1, ORANGE)):
        axg.plot(x, y, color=col, lw=2.2)
        axg.fill_between(x, 0, y, color=col, alpha=0.14, lw=0)
    # the two class labels are long, so anchor each over its own bump and
    # *stagger* them in height (blue higher, orange lower) so they never collide
    # in the middle; each gets a short leader down to its peak.
    axg.annotate(r"$p(x\mid y{=}0)\,p(y{=}0)$", xy=(-1.2, ymax * 1.02),
                 xytext=(-1.9, ymax * 1.34), color=BLUE, ha="center",
                 va="bottom", fontsize=11,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
    axg.annotate(r"$p(x\mid y{=}1)\,p(y{=}1)$", xy=(1.2, ymax * 1.02),
                 xytext=(1.9, ymax * 1.16), color=ORANGE, ha="center",
                 va="bottom", fontsize=11,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    # the tie point of the weighted densities is where the decision flips
    xstar = 0.0
    ytie = np.interp(xstar, x, y0)
    axg.axvline(xstar, color=GRAY, lw=1.2, ls="--")
    axg.annotate("decision\nflips here", xy=(xstar, ytie),
                 xytext=(xstar + 1.0, ymax * 0.16), color=GRAY,
                 fontsize=11, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
    axg.set_title("generative (naive Bayes)", fontsize=12)
    axg.set_xlabel(r"$x$", fontsize=13)
    axg.set_xticks([xstar]); axg.set_xticklabels([r"$x^\ast$"], fontsize=13)
    axg.set_yticks([]); axg.set_xlim(-3.8, 3.8)
    axg.set_ylim(0, ymax * 1.58)
    axg.spines["top"].set_visible(False); axg.spines["right"].set_visible(False)

    # --- right: two labeled clouds and only a boundary between them ---------
    rng = np.random.default_rng(0)
    m0, m1 = np.array([-1.3, -0.9]), np.array([1.3, 0.9])
    c0 = m0 + 0.7 * rng.standard_normal((40, 2))
    c1 = m1 + 0.7 * rng.standard_normal((40, 2))
    # the perpendicular bisector of the class means: m . x = 0, i.e. the line
    # y = -(m_x / m_y) x through the origin (the means are symmetric).
    xx = np.linspace(-3.2, 3.2, 2)
    yy = -(m1[0] / m1[1]) * xx
    axd.fill_between(xx, yy, 3.2, color=ORANGE, alpha=0.07, lw=0)
    axd.fill_between(xx, -3.2, yy, color=BLUE, alpha=0.07, lw=0)
    axd.scatter(c0[:, 0], c0[:, 1], s=12, color=BLUE, alpha=0.45, lw=0)
    axd.scatter(c1[:, 0], c1[:, 1], s=12, color=ORANGE, alpha=0.45, lw=0)
    axd.plot(xx, yy, color="black", lw=2.0, zorder=4)
    axd.text(-1.5, -2.9, "no model of $p(x)$", color=GRAY, fontsize=11,
             ha="center", va="center")
    axd.set_title("discriminative (softmax regression)", fontsize=12)
    axd.set_xlim(-3.2, 3.2); axd.set_ylim(-3.2, 3.2)
    axd.set_xticks([]); axd.set_yticks([])
    axd.spines["top"].set_visible(False); axd.spines["right"].set_visible(False)

    fl.save(fig, "mdl-prob-naive-genvdisc")


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
             fontsize=11)
    axj.set_xlabel(r"$x$"); axj.set_ylabel(r"$y$")

    axc.plot(xs, cond, color=ORANGE, lw=2.6)
    axc.fill_between(xs, 0, cond, color=ORANGE, alpha=0.18, lw=0)
    axc.set_xlabel(r"$x$"); axc.set_ylabel(r"$p(x \mid y_0)$")
    axc.set_yticks([])
    axc.set_ylim(0, None)
    axc.spines["top"].set_visible(False); axc.spines["right"].set_visible(False)
    # the conditional density is nonnegative: pin the x-axis flush to y=0
    axc.spines["bottom"].set_position(("data", 0))
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
    ax.legend(loc="upper right", fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-mle-kl")


def fig_mle_fisher():
    """Fisher information as the curvature of the negative log-likelihood.
    Contours of the Gaussian NLL(mu, sigma) for n = 30 real draws from
    N(1, 1.5^2); the dot marks the MLE (sample mean, sample sd), and the
    overlaid 1-sigma ellipse is the inverse-information covariance
    I(mu, sigma)^{-1}/n = diag(sigma^2/n, sigma^2/(2n)) that asymptotic
    normality predicts: the bowl is twice as curved in sigma as in mu, so the
    ellipse is wider along the mu-axis."""
    from matplotlib.patches import Ellipse

    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.5, 30)
    n = len(x)
    mu_hat, sig_hat = x.mean(), x.std()        # MLE (ddof=0)

    mus = np.linspace(0.1, 2.1, 300)
    sigs = np.linspace(0.85, 2.2, 300)
    M, S = np.meshgrid(mus, sigs)
    nll = n * np.log(S) + ((x[None, None, :] - M[..., None]) ** 2).sum(-1) / (2 * S ** 2)
    levels = nll.min() + np.array([0.5, 2.0, 4.5, 8.0, 12.5])

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.contour(M, S, nll, levels=levels[:1], colors=BLUE, linewidths=1.6)
    ax.contour(M, S, nll, levels=levels[1:], colors=GRAY, linewidths=1.0)
    ax.plot([mu_hat], [sig_hat], "o", color=BLUE, ms=7, zorder=6)

    # 1-sigma ellipse of the asymptotic covariance I^{-1}/n (I is diagonal)
    a = np.sqrt(sig_hat ** 2 / n)              # semi-axis along mu
    b = np.sqrt(sig_hat ** 2 / (2 * n))        # semi-axis along sigma
    ax.add_patch(Ellipse((mu_hat, sig_hat), 2 * a, 2 * b, fill=False,
                         edgecolor=ORANGE, lw=2.2, zorder=5))
    # both labels sit in the clear annular gaps *between* consecutive contour
    # rings (checked against the analytic NLL, not just eyeballed), so neither
    # ever crosses a contour line.  The (mu_hat, sig_hat) label drops well
    # below the ellipse into the band between the first and second gray
    # rings; the information-matrix label sits directly above the ellipse in
    # that same band.
    ax.text(mu_hat, sig_hat - 0.30, r"$(\hat\mu,\hat\sigma)$", color=BLUE,
            ha="center", va="center", fontsize=11)
    ax.annotate(r"$I(\hat\mu,\hat\sigma)^{-1}/n$",
                xy=(mu_hat + a * 0.3, sig_hat + b * 0.95),
                xytext=(mu_hat, sig_hat + 0.62), color=ORANGE,
                fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-mle-fisher")


def fig_elbo():
    """The evidence lower bound.  A one-latent-bit model: z in {0,1} with
    p(z=1) = 1/2 and p(x|z; theta) = N(x; theta + 2z, 1), observing x = 1.5.
    The blue curve is the evidence log p(x; theta); fixing q to the posterior
    at theta^(t) = -1.5, the orange ELBO L(q, theta) touches the evidence at
    theta^(t) and runs below it everywhere else; the gap at any other theta is
    KL(q || p(z|x; theta)).  The M-step climbs the bound to theta^(t+1)."""
    x_obs, th_t = 1.5, -1.5
    th = np.linspace(-2.5, 2.5, 600)

    def log_norm(x, mu):                       # log N(x; mu, 1)
        return -0.5 * (x - mu) ** 2 - 0.5 * np.log(2 * np.pi)

    def log_evidence(theta):                   # log p(x; theta), z marginalized
        l0 = np.log(0.5) + log_norm(x_obs, theta)
        l1 = np.log(0.5) + log_norm(x_obs, theta + 2.0)
        m = np.maximum(l0, l1)
        return m + np.log(np.exp(l0 - m) + np.exp(l1 - m))

    # E-step at theta^(t): q(z) = p(z | x; theta^(t))
    l0t = np.log(0.5) + log_norm(x_obs, th_t)
    l1t = np.log(0.5) + log_norm(x_obs, th_t + 2.0)
    q1 = 1.0 / (1.0 + np.exp(l0t - l1t)); q0 = 1.0 - q1
    entropy = -(q0 * np.log(q0) + q1 * np.log(q1))

    evid = log_evidence(th)
    elbo = (q0 * (np.log(0.5) + log_norm(x_obs, th))
            + q1 * (np.log(0.5) + log_norm(x_obs, th + 2.0)) + entropy)
    th_next = th[np.argmax(elbo)]              # M-step: argmax of the bound

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(th, evid, color=BLUE, lw=2.4)
    ax.plot(th, elbo, color=ORANGE, lw=2.2)
    ax.text(2.45, evid.max() + 0.16, r"$\log p(x;\theta)$", color=BLUE,
            ha="right", va="bottom", fontsize=11)
    ax.text(2.45, elbo[-1] - 0.10, r"ELBO $\mathcal{L}(q,\theta)$",
            color=ORANGE, ha="right", va="top", fontsize=11)

    # the touch point at theta^(t) and the bound's peak at theta^(t+1)
    y_t = float(log_evidence(np.array([th_t]))[0])
    y_next = elbo.max()
    ymin = elbo.min()
    for tv, yv in ((th_t, y_t), (th_next, y_next)):
        ax.plot([tv, tv], [ymin, yv], color=GRAY, lw=1.1, ls="--", zorder=2)
        ax.plot([tv], [yv], "o", color=GRAY, ms=6, zorder=6)
    ax.text(th_t, ymin - 0.10, r"$\theta^{(t)}$", color=GRAY, ha="center",
            va="top", fontsize=11)
    ax.text(th_next, ymin - 0.10, r"$\theta^{(t+1)}$", color=GRAY,
            ha="center", va="top", fontsize=11)

    # the gap between the curves at some other theta is the KL divergence
    th_g = 1.4
    g_lo = float(np.interp(th_g, th, elbo))
    g_hi = float(np.interp(th_g, th, evid))
    ax.annotate("", xy=(th_g, g_hi), xytext=(th_g, g_lo),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6))
    # short label (the caption spells out "gap = KL divergence" in full) placed
    # to the right of the arrow, clear of both curves at every theta it spans
    ax.text(th_g + 0.15, (g_lo + g_hi) / 2, "gap\n(KL)",
            color=GREEN, ha="left", va="center", fontsize=11)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("objective")
    ax.set_yticks([])
    ax.set_ylim(ymin - 0.35, evid.max() + 0.30)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-elbo")


# =========================================================================== #
# STATISTICS: the coverage strip                                              #
# =========================================================================== #

def fig_coverage_strip():
    """The frequentist reading of a confidence interval, drawn literally.  One
    hundred independent datasets of n = 30 draws from N(0,1); for each we build
    the 95% interval mu_hat +/- 1.96 s/sqrt(n) and plot it as a vertical bar
    against the fixed true mean (dashed line).  Real intervals from a seeded
    simulation: the misses (orange, counted in the corner label) are the ~5% the
    guarantee licenses, and nothing about a single bar reveals which kind it is."""
    rng = np.random.default_rng(3)
    mu, n, m = 0.0, 30, 100
    data = rng.normal(mu, 1.0, (m, n))
    mu_hat = data.mean(axis=1)
    se = data.std(axis=1, ddof=1) / np.sqrt(n)
    lo, hi = mu_hat - 1.96 * se, mu_hat + 1.96 * se
    miss = (lo > mu) | (hi < mu)

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for i in range(m):
        col = ORANGE if miss[i] else BLUE
        ax.plot([i, i], [lo[i], hi[i]], color=col,
                lw=1.9 if miss[i] else 1.1,
                alpha=1.0 if miss[i] else 0.55, zorder=4 if miss[i] else 2)
        ax.plot([i], [mu_hat[i]], "o", color=col, ms=2.4,
                zorder=5 if miss[i] else 3)
    ax.axhline(mu, color=GRAY, lw=1.2, ls="--", zorder=1)
    ax.text(m - 0.5, mu + 0.045, r"true $\mu$", color=GRAY, fontsize=11,
            ha="right", va="bottom", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                      pad=1.5))
    ax.text(0.01, 0.04, f"{int(miss.sum())} of {m} intervals miss",
            color=ORANGE, fontsize=11, ha="left", va="bottom",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                      pad=1.5), zorder=6)
    ax.set_xlim(-1.5, m + 0.5)
    ax.set_xlabel("dataset (each yields one interval)", fontsize=11)
    ax.set_ylabel(r"95% CI for $\mu$", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-coverage")


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
    # left: observed sample up top, two resamples beneath -- spaced so each label
    # sits in a clear horizontal band, never on the event strokes.
    cx = np.max(sample) * 0.5
    axs.eventplot([sample], colors=[BLUE], lineoffsets=3.0, linelengths=0.6)
    for k in range(2):
        rs = rng.choice(sample, size=60, replace=True)
        axs.eventplot([rs], colors=[GRAY], lineoffsets=1.5 - 1.0 * k,
                      linelengths=0.6)
    axs.text(cx, 3.62, "observed sample", color=BLUE, fontsize=10.5,
             ha="center", va="bottom")
    axs.text(cx, 2.25, "resamples (with replacement)", color=GRAY,
             fontsize=11, ha="center", va="center")
    axs.set_ylim(-0.4, 4.05); axs.set_yticks([])
    axs.set_xlabel("value", fontsize=12)
    for sp in ("top", "right", "left"):
        axs.spines[sp].set_visible(False)

    # right: histogram of bootstrap medians + central-95% percentile band
    axh.hist(reps, bins=40, color=BLUE, alpha=0.7)
    axh.axvspan(lo, hi, color=ORANGE, alpha=0.20)
    for v in (lo, hi):
        axh.axvline(v, color=ORANGE, lw=1.8)
    axh.set_xlabel(r"$\hat\theta^{*}$  (median)", fontsize=12)
    axh.text((lo + hi) / 2, axh.get_ylim()[1] * 0.98, r"central $95\%$",
             color=ORANGE, ha="center", va="top", fontsize=11, zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                       pad=1.5))
    axh.set_yticks([])
    axh.spines["top"].set_visible(False); axh.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-bootstrap")


FIGURES = [
    # random variables
    fig_marginal,
    fig_conditional_slice,
    fig_pdf_area,
    fig_pdf_cdf,
    fig_inverse_transform,
    fig_chebyshev,
    fig_covariance,
    fig_correlation,
    # distributions
    fig_family_tree,
    fig_beta_posterior,
    fig_discrete_pmfs,
    fig_continuous_pdfs,
    fig_mvn_contours,
    # maximum likelihood
    fig_map_prior,
    fig_mle_kl,
    fig_mle_fisher,
    fig_elbo,
    # naive bayes
    fig_naive_independence,
    fig_naive_genvdisc,
    # statistics
    fig_significance,
    fig_power_curves,
    fig_bias_variance_u_curve,
    fig_sampling_distribution,
    fig_type_i_ii_matrix,
    fig_coverage_strip,
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
