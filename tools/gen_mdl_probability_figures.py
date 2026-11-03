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
    # prior-mean guide: stops above its own label so the text never sits on it
    ax.plot([0, 0], [-0.22, post.max()], color=GRAY, lw=0.8, ls=":")
    # the "pull" arrow runs dot to dot, MLE -> MAP (shrunk so neither end
    # touches a marker); its label sits in the clear wedge to the arrow's
    # right, below the orange prior arm and above the blue NLL floor.
    ax.annotate("", xy=(theta_map, ymap), xytext=(theta_mle, ymle),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5,
                                shrinkA=8, shrinkB=5))
    ax.text(2.2, 0.95, "shrink toward\nprior mean",
            color=fl.INK, fontsize=11, ha="left", va="center")
    ax.text(theta_mle, ymle - 0.42, r"$\hat\theta_{\mathrm{MLE}}$", color=BLUE,
            ha="center", va="top", fontsize=13)
    ax.text(theta_map, -0.42, r"$\hat\theta_{\mathrm{MAP}}$", color=GREEN,
            ha="center", va="top", fontsize=13)
    ax.text(0, -0.42, r"prior mean", color=fl.INK, ha="center", va="top",
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

    # critical-value markers and labels
    for s in (-1, 1):
        ax.plot([s * zc, s * zc], [0, pdf(zc)], color=ORANGE, lw=1.4)
    ax.text(0, 0.16, r"fail to reject $H_0$" "\n" r"$1-\alpha = 0.95$",
            ha="center", va="center", color=BLUE, fontsize=11)
    ax.annotate(r"reject  ($\alpha/2$)", xy=(2.55, pdf(2.55)),
                xytext=(3.0, 0.16), color=ORANGE, fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate(r"reject  ($\alpha/2$)", xy=(-2.55, pdf(2.55)),
                xytext=(-3.0, 0.16), color=ORANGE, fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.text(zc, -0.012, r"$1.96$", color=ORANGE, ha="center", va="top",
            fontsize=11)
    ax.text(-zc, -0.012, r"$-1.96$", color=ORANGE, ha="center", va="top",
            fontsize=11)

    # integer ticks, but skip +/-2 (right where +/-1.96 already labels the axis)
    # so the two labels don't stack on top of each other
    ax.set_xticks([-4, -3, -1, 0, 1, 3, 4])
    ax.set_xlabel("test statistic (under $H_0$)", fontsize=13)
    ax.set_ylabel("density", fontsize=13)
    ax.set_ylim(0, 0.45)
    ax.set_yticks([])
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    fl.save(fig, "mdl-prob-significance")


def fig_power_curves():
    """Power of the one-sample two-sided z-test (H0: mu = 0, sigma = 1, at
    alpha = 0.05) as a function of the sample size n, one curve per true
    effect size delta:  power(n; delta) = Phi(delta sqrt(n) - z*) +
    Phi(-delta sqrt(n) - z*) with z* = z_{0.975}.  Small-effect curves start
    near alpha, and every curve climbs to 1; the n needed to hit the conventional target 0.8
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

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    # one curve per effect size, a blue family from dark (delta=1) to light;
    # the light end is too pale for text, so every label is set in INK beside
    # its own curve
    deltas = [1.0, 0.5, 0.1, 0.01]
    # a dark->light ramp derived from the blue trio (dark, base, then two
    # base->tint blend stops kept dark enough to read as label text)
    def _mix(c1, c2, t):
        a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))
    blues = [_mix(BLUE, fl.INK, 0.45), BLUE, _mix(BLUE, fl.T.BLUE.tint, 0.35),
             _mix(BLUE, fl.T.BLUE.tint, 0.55)]
    # each curve is labelled at power 0.5 (where delta*sqrt(n) = z*, i.e.
    # n = (z*/delta)^2), beside its own curve: delta=1 and the two small effects
    # to the left, delta=0.5 to the right (the delta=1/0.5 gap is too narrow
    # for a label on the left of delta=0.5).
    for delta, col, side in zip(deltas, blues, ("L", "R", "L", "L")):
        pw = power(delta)
        ax.plot(n, pw, color=col, lw=2.2)
        n_half = (zc / delta) ** 2
        if side == "L":
            ax.text(n_half / 1.35, 0.5, rf"$\delta={delta:g}$", color=fl.INK,
                    fontsize=11, ha="right", va="center")
        else:
            # lower down the curve, so it reads apart from the delta=0.1 label
            # that shares this gap
            n_lo = float(np.interp(0.3, pw, n))
            ax.text(n_lo * 1.35, 0.3, rf"$\delta={delta:g}$", color=fl.INK,
                    fontsize=11, ha="left", va="center")

    # the conventional power target and the test's size alpha
    ax.axhline(0.8, color=GRAY, lw=1.1, ls="--")
    # the target label sits mid-plot, where delta=0.1 has saturated and
    # delta=0.01 is still low, so the line is the only thing at that height
    ax.text(1.6e3, 0.825, "target $0.8$", color=fl.INK, fontsize=11, va="bottom")
    ax.axhline(0.05, color=GRAY, lw=1.0, ls=":")
    # far right: every power curve is ~1 there, so the label owns the space
    ax.text(1.5e5, 0.075, r"$\alpha=0.05$", color=fl.INK, fontsize=11,
            va="bottom", ha="right")

    # where the extreme curves cross the target (power is increasing in n).
    # both sit just above the target line, left of their dots, where every
    # curve to the left has either not started (n=8) or already saturated.
    for delta, col, lab, lx, ly, ha, va in (
            (1.0, blues[0], r"$n\approx 8$", 1 / 1.3, 0.825, "right", "bottom"),
            (0.01, blues[3], r"$n\approx 8\times 10^4$", 1 / 1.3, 0.825,
             "right", "bottom")):
        pw = power(delta)
        nstar = float(np.interp(0.8, pw, n))
        ax.plot([nstar], [0.8], "o", color=col, ms=6, zorder=6)
        ax.text(nstar * lx, ly, lab, color=fl.INK, fontsize=11, ha=ha, va=va)

    ax.set_xscale("log")
    ax.set_xlim(1, 2e5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("sample size $n$", fontsize=13)
    ax.set_ylabel("power", fontsize=13)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-power")


def fig_bias_variance_u_curve():
    """Bias^2, variance and MSE of the shrunk mean c * mean as functions of the
    shrinkage factor c in [0, 1], for theta = 1, sigma = 4, n = 30:
    bias^2 = (1 - c)^2 theta^2, variance = c^2 sigma^2 / n = 0.533 c^2.  Their
    sum is minimized at c* = theta^2 / (theta^2 + sigma^2/n) = 0.652, where the
    MSE is 0.348 against 0.533 for the sample mean (c = 1).  Exact curves."""
    theta, sigma, n = 1.0, 4.0, 30
    v = sigma**2 / n
    c = np.linspace(0.0, 1.0, 400)
    bias2 = (1 - c) ** 2 * theta**2
    var = c**2 * v
    mse = bias2 + var
    cstar = theta**2 / (theta**2 + v)
    mstar = (1 - cstar) ** 2 * theta**2 + cstar**2 * v

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(c, bias2, color=BLUE, lw=2.0, label=r"bias$^2$")
    ax.plot(c, var, color=ORANGE, lw=2.0, label="variance")
    ax.plot(c, mse, color=GREEN, lw=2.6, label=r"MSE $=$ bias$^2$ $+$ variance")

    # the minimizer: a dashed drop to the axis and a dot on the MSE curve
    ax.plot([cstar, cstar], [0, mstar], color=GRAY, lw=1.0, ls="--", zorder=1)
    ax.plot([cstar], [mstar], "o", color=GREEN, ms=7, zorder=6)
    # the two ends of the family, labelled in the empty space above each end
    ax.text(0.02, 1.04, "constant $0$\n(all bias)", color=fl.INK, fontsize=12,
            ha="left", va="bottom")
    ax.text(0.98, v + 0.06, r"$\bar x$" "\n(all variance)", color=fl.INK,
            fontsize=12, ha="right", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.32)
    ax.set_xticks([0, cstar, 1])
    ax.set_xticklabels(["$0$", r"$c^\star\approx0.65$", "$1$"])
    ax.set_yticks([0, mstar, v, 1])
    ax.set_yticklabels(["$0$", "$0.35$", "$0.53$", "$1$"])
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r"shrinkage factor $c$", fontsize=13)
    ax.set_ylabel(r"error of $c\,\bar x$", fontsize=13)
    ax.legend(loc="upper center", ncol=1, fontsize=12, frameon=False)
    ax.set_aspect("auto")
    fl.clean_axes(ax, equal=False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(fl.INK)
    fl.save(fig, "mdl-prob-bias-variance-u-curve")


def fig_sampling_distribution():
    """The sampling distributions of the section's two estimators of theta = 1
    from n = 30 draws of N(1, 16): the sample mean (center 1, standard error
    4/sqrt(30) = 0.730) and the shrunk mean 0.65 * mean (center 0.65, i.e.
    bias -0.35, standard error 0.65 * 0.730 = 0.475).  Both are exact Gaussians
    (the mean of Gaussian data is Gaussian), drawn on the same x-window so the
    narrower, offset right panel reads against the wider, centered left one."""
    theta, sigma, n, c = 1.0, 4.0, 30, 0.65
    se_mean = sigma / np.sqrt(n)
    panels = [(theta, se_mean), (c * theta, c * se_mean)]   # (center, se)
    z = np.linspace(-1.4, 3.4, 800)

    def gauss(x, mu, sd):
        return np.exp(-((x - mu) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    # shared vertical scale: guides and label heights use the taller peak
    Y = max(gauss(mu, mu, se) for mu, se in panels)
    for ax, (mu, se) in zip(axes, panels):
        y = gauss(z, mu, se)
        ax.fill_between(z, y, color=BLUE, alpha=0.18, lw=0)
        ax.plot(z, y, color=BLUE, lw=2.0)
        ymax = gauss(mu, mu, se)

        # true theta (dashed gray guide) and the estimator's center (orange)
        ax.plot([theta, theta], [0, Y * 1.3], color=GRAY, lw=1.1, ls="--")
        if abs(mu - theta) < 1e-9:
            ax.plot([mu, mu], [0, ymax], color=ORANGE, lw=1.6)
            ax.text(theta - 0.08, ymax * 1.06, r"$\theta$", color=fl.INK,
                    ha="right", va="bottom", fontsize=14)
            ax.text(mu + 0.08, ymax * 1.06, r"$\mathbb{E}[\hat\theta]$",
                    color=ORANGE, ha="left", va="bottom", fontsize=13)
        else:
            # the center lies LEFT of theta: the bias arrow spans the gap just
            # above the peak; theta is labelled to the right of its guide and
            # the center to the left of its line, both outside the arrow
            ay = ymax * 1.06
            ax.plot([mu, mu], [0, ay], color=ORANGE, lw=1.6)
            ax.annotate("", xy=(mu, ay), xytext=(theta, ay),
                        arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3,
                                        shrinkA=0, shrinkB=0))
            # the arrow is short (0.35), so "bias" sits to its right, past
            # the theta guide, at the arrow's own height
            ax.text(theta + 0.08, ay, "bias", color=ORANGE,
                    ha="left", va="center", fontsize=12)
            ax.text(theta + 0.08, Y * 1.22, r"$\theta$", color=fl.INK,
                    ha="left", va="center", fontsize=14)
            ax.text(mu - 0.08, ay + Y * 0.08, r"$\mathbb{E}[\hat\theta]$",
                    color=ORANGE, ha="right", va="bottom", fontsize=13)

        # spread = standard error, drawn as a +/- 1 se bar low on the curve
        sy = ymax * 0.13
        ax.annotate("", xy=(mu + se, sy), xytext=(mu - se, sy),
                    arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.3))
        ax.text(mu, -Y * 0.05, r"spread $=$ std. error",
                color=BLUE, ha="center", va="top", fontsize=12)

        ax.set_xlabel(r"$\hat\theta$", fontsize=14)
        ax.set_xlim(z[0], z[-1])
        ax.set_ylim(-Y * 0.12, Y * 1.34)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_aspect("auto")
        fl.clean_axes(ax, equal=False)
        # a density is nonnegative: pin the x-axis flush to y=0 and re-pin the
        # axis label to the bottom margin, below the "spread" caption
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["bottom"].set_color(fl.INK)
        ax.spines["left"].set_bounds(0, Y * 1.34)
        ax.spines["left"].set_color(fl.INK)
        ax.xaxis.set_label_coords(0.5, -0.055)
    axes[0].set_ylabel("density over data sets", fontsize=13)
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
        # text in each accent's dark shade: the base colours are too light
        # to read on their own tints
        col = fl.T.GREEN.dark if fc is GREEN else fl.T.ORANGE.dark
        ax.text(cx + 0.5, cy + 0.62, title, ha="center", va="center",
                fontsize=12, color=col, weight="bold")
        ax.text(cx + 0.5, cy + 0.34, sub, ha="center", va="center",
                fontsize=11, color=col)

    # column headers (the decision)
    ax.text(0.5, 2.16, "fail to reject $H_0$", ha="center", va="center",
            fontsize=10.5)
    ax.text(1.5, 2.16, "reject $H_0$", ha="center", va="center", fontsize=10.5)
    ax.text(1.0, 2.46, "decision", ha="center", va="center", fontsize=10.5,
            color=fl.INK, style="italic")

    # row headers (the truth)
    ax.text(-0.12, 1.5, r"$H_0$ true", ha="right", va="center", fontsize=10.5,
            rotation=90)
    ax.text(-0.12, 0.5, r"$H_0$ false", ha="right", va="center", fontsize=10.5,
            rotation=90)
    ax.text(-0.42, 1.0, "truth", ha="right", va="center", fontsize=10.5,
            color=fl.INK, style="italic", rotation=90)

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
    shaded slab (kept off the curve); the total-area note sits inside the
    faint full-curve fill of the big left lobe."""
    x = np.linspace(-5, 7, 600); p = _twobump(x); a, b = 0.0, 4.0
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.fill_between(x, 0, p, color=BLUE, alpha=0.12)
    m = (x >= a) & (x <= b)
    ax.fill_between(x[m], 0, p[m], color=BLUE, alpha=0.34)
    ax.plot(x, p, color=BLUE, lw=2.4)

    # P(a<X<=b): label sits in the clear white space above the valley between the
    # two bumps, with a short leader down into the dark slab -- never on the curve.
    ax.annotate(r"$P(a<X\leq b)$", xy=(2.0, 0.028), xytext=(2.0, 0.165),
                ha="center", va="bottom", fontsize=15, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    # total-area note: set inside the big light-shaded left lobe, which is
    # part of the area it names (an arrow into the thin right tail read as
    # "this tail has area 1")
    ax.text(-1.5, 0.042, "total area $=1$", ha="center", va="center",
            fontsize=12.5, color=fl.INK)

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
    # horizontal guides carry F(a) and F(b) across to the rise arrow, so its
    # two ends visibly sit at the curve's heights over a and b
    axc.plot([a, b + 0.4], [F[ia], F[ia]], ":", color=GRAY, lw=1.0)
    axc.plot([b, b + 0.4], [F[ib], F[ib]], ":", color=GRAY, lw=1.0)
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
    # the x label sits over the dotted guide grid: a white backing keeps the
    # guides from running through the glyphs
    wb = dict(facecolor="white", edgecolor="none", pad=1.0)
    # U reads like a y-tick label: left of the axis, on the highlighted level
    ax.text(-5.3, u0, r"$U$", color=BLUE, fontsize=13, ha="right",
            va="center", clip_on=False)
    ax.text(x0 + 0.15, 0.045, r"$x=F^{-1}(U)$", color=BLUE, fontsize=12,
            ha="left", va="bottom", bbox=wb, zorder=5)
    ax.text(5.4, np.interp(5.4, x, F) - 0.06, r"c.d.f. $F$", color=ORANGE,
            fontsize=12, ha="left", va="top")
    # name the dense cluster with a bracket under the axis rather than a long
    # leader (which had to cross the highlighted path and the guide lines)
    c0, c1, yb = xs[1], xs[10], -0.075
    ax.plot([c0, c0, c1, c1], [yb + 0.025, yb, yb, yb + 0.025], color=GRAY,
            lw=1.1, clip_on=False)
    ax.text((c0 + c1) / 2, yb - 0.02, "dense where $F$ is steep",
            color=fl.INK, fontsize=11, ha="center", va="top")
    ax.set_xlim(-5, 7); ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 1])
    ax.set_xticks([])
    # x names the axis at its right end, leaving the space under the
    # cluster to the bracket label
    ax.text(7, -0.03, r"$x$", fontsize=13, ha="right", va="top")
    ax.set_ylabel("probability level", fontsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-inverse-transform")


def fig_chebyshev():
    """Three-atom p.m.f. at {a-2,a,a+2} with masses {q,1-2q,q}: the Chebyshev
    event |X-a| >= alpha*sigma at alpha=2 has half-width 4*sqrt(2q).  For q>1/8
    the half-width exceeds 2, so no atom lies in the event; for q=1/8 the +/-2
    atoms sit at the boundary and carry mass 2q = 1/4, matching the bound (the
    equality case); for q<1/8 they lie beyond it with mass 2q < 1/4.  Atoms
    inside the event are drawn orange."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4))
    # per-panel titles name the regime (with the q actually drawn); the
    # caption lists the regimes but cannot say which panel is which
    for ax, q, title in zip(axes, [0.2, 0.125, 0.05],
                            [r"$q=0.2>1/8$", r"$q=1/8$", r"$q=0.05<1/8$"]):
        atoms, heights = [-2, 0, 2], [q, 1 - 2 * q, q]
        ax.vlines(atoms, 0, heights, color=BLUE, lw=3, zorder=3)
        # the +/-2 atoms belong to the event {|X-a| >= half-width} iff q <= 1/8
        edge_col = ORANGE if q <= 0.125 else BLUE
        ax.plot([-2, 2], [q, q], "o", color=edge_col, ms=7, zorder=4)
        ax.plot([0], [1 - 2 * q], "o", color=BLUE, ms=7, zorder=4)
        hw = 4 * np.sqrt(2 * q)
        # the a-2, a, a+2 tick labels sit just under the axis; drop the
        # bracket (and its caption) further down still so neither touches them.
        ax.annotate("", xy=(hw, -0.22), xytext=(-hw, -0.22),
                    arrowprops=dict(arrowstyle="|-|", color=ORANGE, lw=1.5,
                                    mutation_scale=6, shrinkA=0, shrinkB=0))
        ax.text(0, -0.30, rf"$\alpha\sigma_X={hw:.2f}$", ha="center",
                va="top", color=fl.T.ORANGE.dark, fontsize=15)
        ax.set_title(title, fontsize=15, pad=4)
        ax.set_xlim(-3, 3); ax.set_ylim(-0.44, 1.0)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([])
        ax.set_xticklabels(["$a-2$", "$a$", "$a+2$"])
        ax.tick_params(axis="x", labelsize=15, pad=1.5)
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
            # braces keep mathtext from spacing the minus as a binary operator
            ax.set_title(rf"cov $\approx {{{cov:.2f}}}$", fontsize=15)
            # rho goes in the corner the cloud leaves empty (top-right for a
            # down-right or round cloud, top-left for an up-right one)
            right = c <= 0
            ax.text(0.96 if right else 0.04, 0.95,
                    rf"$\rho \approx {{{rho:.2f}}}$", color=fl.INK,
                    fontsize=15, ha="right" if right else "left", va="top",
                    transform=ax.transAxes, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0,
                              alpha=0.85))
            ax.set_xlim(-3.6, 3.6)
            ax.set_ylim(-4 * scale, 4 * scale)
            ax.set_yticks([-3 * scale, 0, 3 * scale])
            ax.set_xticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel(r"$Y$ in dollars", fontsize=14)
    axes[1, 0].set_ylabel(r"same data, $Y$ in cents", fontsize=14)
    fl.save(fig, "mdl-prob-covariance")


def fig_correlation():
    """Correlation measures the tightness and sign of a linear relation, not
    its slope.  (a) Y = 0.5 X + noise and (b) Y = 2 X + noise, with the noise
    scaled (sd = slope * sqrt(1/0.81 - 1)) so that both have rho = 0.9;
    (c) Y = X^2 - E[X^2] for X uniform on [-2.5, 2.5] (a sample symmetric
    about 0), a function of X with rho = 0.  Equal windows, so the two slopes are visible; each
    panel's sample rho sits in its empty corner."""
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6))
    n, rho = 220, 0.9
    X = rng.standard_normal(n)
    xi = rng.standard_normal(n)
    u = rng.uniform(0, 2.5, n // 2)
    U = np.concatenate([u, -u])          # symmetric sample: sample rho is 0
    clouds = [
        (X, 0.5 * X + 0.5 * np.sqrt(1 / rho ** 2 - 1) * xi, "(a) slope $0.5$"),
        (X, 2.0 * X + 2.0 * np.sqrt(1 / rho ** 2 - 1) * xi, "(b) slope $2$"),
        (U, U ** 2 - 2.5 ** 2 / 3, "(c) $y=x^2$, centered"),
    ]
    for ax, (x, y, title) in zip(axes, clouds):
        r = round(np.corrcoef(x, y)[0, 1], 2) + 0.0   # + 0.0 drops a -0.0
        ax.scatter(x, y, s=10, color=BLUE, alpha=0.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=15)
        # the up-right clouds leave the top-left corner empty; the parabola
        # leaves the bottom corners empty
        if title.startswith("(c)"):
            pos = dict(x=0.04, y=0.05, ha="left", va="bottom")
        else:
            pos = dict(x=0.04, y=0.95, ha="left", va="top")
        ax.text(pos["x"], pos["y"], rf"$\rho \approx {{{r:.2f}}}$",
                color=fl.INK, fontsize=15, ha=pos["ha"], va=pos["va"],
                transform=ax.transAxes, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0,
                          alpha=0.85))
        ax.set_xlim(-3.6, 3.6); ax.set_ylim(-7, 7)
        ax.set_xticks([]); ax.set_yticks([-6, 0, 6])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-correlation")


# --- distributions: family tree + pmf/pdf galleries + MVN contours -----------

def _box(ax, xy, w, h, label, fc="white", ec=None):
    from matplotlib.patches import FancyBboxPatch
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor=fc, edgecolor=ec or fl.INK, lw=1.2,
                                zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=12,
            color=fl.INK, zorder=4)


def fig_family_tree():
    """How the section's twelve laws grow out of one coin flip.  Filled-head
    arrows construct (a sum, a random sign, a vector of coordinates, a waiting
    time, equal weights); dashed open-head arrows take a limit.  Every arrow is
    clipped to the box outlines, so tails leave and heads land on an edge."""
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    w, h = 2.5, 1.0
    centers = {"Bernoulli": (1.3, 3.2), "Binomial": (5.4, 5.2),
               "Categorical": (5.4, 1.4), "Poisson": (10.0, 6.6),
               "Gaussian": (10.0, 4.0), "Multinomial": (10.0, 2.3),
               "Discrete\nuniform": (10.0, 0.5), "Exponential": (14.6, 6.6),
               "Multivariate\nGaussian": (14.6, 4.0),
               "Continuous\nuniform": (14.6, 0.5), "Gamma": (19.0, 7.4),
               "Laplace": (19.0, 5.8)}

    def edge_pts(s, t, gap=0.08):
        c0, c1 = np.array(centers[s]), np.array(centers[t])
        u = (c1 - c0) / np.linalg.norm(c1 - c0)

        def reach(v):  # distance from a center to its outline along v
            return min((w / 2 + gap) / abs(v[0]) if v[0] else np.inf,
                       (h / 2 + gap) / abs(v[1]) if v[1] else np.inf)
        return c0 + u * reach(u), c1 - u * reach(u)

    def link(p0, p1, limit):
        ax.annotate("", xy=tuple(p1), xytext=tuple(p0),
                    arrowprops=dict(arrowstyle="-|>", edgecolor=fl.INK, lw=1.5,
                                    facecolor="white" if limit else fl.INK,
                                    linestyle=(0, (4, 3)) if limit else "-",
                                    shrinkA=0, shrinkB=0, mutation_scale=15))

    # (source, target, label, label center, is_limit)
    arrows = [
        ("Bernoulli", "Binomial", r"sum of $n$", (2.75, 4.65), False),
        ("Bernoulli", "Categorical", r"$K$ outcomes", (2.6, 1.85), False),
        ("Binomial", "Poisson", "$n\\to\\infty$,\n$np\\to\\lambda$",
         (7.35, 6.65), True),
        ("Binomial", "Gaussian", "standardized,\n$n\\to\\infty$",
         (7.3, 3.85), True),
        ("Poisson", "Exponential", "waiting time", (12.3, 6.95), False),
        ("Exponential", "Gamma", r"sum of $k$", (16.75, 7.55), False),
        ("Exponential", "Laplace", "random sign", (16.45, 5.45), False),
        ("Gaussian", "Multivariate\nGaussian", r"vector, $\boldsymbol{\Sigma}$",
         (12.3, 4.35), False),
        ("Categorical", "Multinomial", r"sum of $n$ draws", (7.3, 2.45), False),
        ("Categorical", "Discrete\nuniform", r"equal $p_k$", (7.55, 0.45),
         False),
        ("Discrete\nuniform", "Continuous\nuniform", "points fill\n$[a,b]$",
         (12.3, 1.3), True),
    ]
    for s, t, lab, (lx, ly), limit in arrows:
        link(*edge_pts(s, t), limit)
        ax.text(lx, ly, lab, ha="center", va="center", fontsize=13,
                color="black", linespacing=1.05)
    for k, (cx, cy) in centers.items():
        _box(ax, (cx - w / 2, cy - h / 2), w, h, "", fc="white")
        ax.text(cx, cy, k, ha="center", va="center", fontsize=14,
                color=fl.INK, zorder=4, linespacing=1.0)
    # legend: name the two arrow styles once, in the empty top-left corner
    for y, limit, lab in ((7.75, False, "construction"), (7.15, True, "limit")):
        link((0.05, y), (1.3, y), limit)
        ax.text(1.5, y, lab, ha="left", va="center", fontsize=13,
                color="black")
    ax.set_xlim(-0.1, 20.35); ax.set_ylim(-0.1, 8.0)
    ax.set_aspect("equal"); ax.axis("off")
    fl.save(fig, "mdl-prob-family-tree")
    import shutil
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-prob-family-tree.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-prob-family-tree.svg"))



def fig_beta_posterior():
    """Bayesian updating of a coin's bias theta.  Three panels of the Beta
    density: the flat Beta(1,1) prior; the Beta(10,5) posterior after 9 heads
    in 13 flips (mean 2/3, sd 0.12); and the Beta(91,41) posterior after 90
    heads in 130 flips (sd 0.04).  The true bias theta* = 0.7 is the dashed
    line; the exact Beta densities (via lgamma, no scipy) pile up on it, their
    width shrinking like 1/sqrt(n)."""
    import shutil
    from math import lgamma

    def beta_pdf(t, a, b):
        logB = lgamma(a) + lgamma(b) - lgamma(a + b)
        return np.exp((a - 1) * np.log(t) + (b - 1) * np.log(1 - t) - logB)

    t = np.linspace(1e-4, 1 - 1e-4, 600)
    theta_star = 0.7
    panels = [(1, 1, "Beta(1,1)\nprior"),
              (10, 5, "Beta(10,5)\n9 H in 13"),
              (91, 41, "Beta(91,41)\n90 H in 130")]
    ymax = beta_pdf(t, 91, 41).max() * 1.12

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), sharex=True)
    for ax, (a, b, sub) in zip(axes, panels):
        y = beta_pdf(t, a, b)
        ax.plot(t, y, color=BLUE, lw=2.2)
        ax.fill_between(t, 0, y, color=BLUE, alpha=0.15, lw=0)
        ax.axvline(theta_star, color=ORANGE, lw=1.6, ls="--")
        # top-left corner: clear of the curves (the Beta(10,5) peak sits at
        # 0.69, next to the dashed line) and of the dashed line at 0.7
        ax.text(0.045, ymax * 0.95, sub, color=fl.INK, fontsize=13, ha="left",
                va="top", linespacing=1.05)
        ax.set_xlim(0, 1); ax.set_ylim(0, ymax)
        ax.set_xticks([0, theta_star, 1.0])
        ax.set_xticklabels(["$0$", r"$\theta^{\ast}$", "$1$"], fontsize=13)
        ax.tick_params(colors="black")
        ax.set_yticks([])
        ax.set_xlabel(r"$\theta$", fontsize=14)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("density", fontsize=13)
    fl.save(fig, "mdl-prob-beta-posterior")
    # the slides read @fig figures from img/auto
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-prob-beta-posterior.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-prob-beta-posterior.svg"))


def fig_discrete_pmfs():
    """The four discrete laws of the section in the order of the text, each
    on the running example: Bernoulli(9/13); Binomial(13, 1/2) with the k = 9
    stem (P = 715/8192) in orange; Poisson(4); and the categorical
    softmax(2, 1, 0.1, -1) against the uniform 1/4."""
    import math
    import shutil
    z = np.array([2.0, 1.0, 0.1, -1.0])
    cat = np.exp(z) / np.exp(z).sum()
    series = [
        ([0, 1], [4 / 13, 9 / 13], r"Bernoulli$(9/13)$"),
        (list(range(14)), [math.comb(13, k) / 2 ** 13 for k in range(14)],
         r"Binomial$(13,\frac{1}{2})$"),
        (list(range(15)),
         [math.exp(-4) * 4 ** k / math.factorial(k) for k in range(15)],
         r"Poisson$(4)$"),
        ([1, 2, 3, 4], list(cat), "categorical"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.9),
                             gridspec_kw=dict(wspace=0.34,
                                              width_ratios=[0.7, 1.3, 1.3, 1]))
    for i, (ax, (ks, ps, title)) in enumerate(zip(axes, series)):
        ax.vlines(ks, 0, ps, color=BLUE, lw=2.4)
        ax.plot(ks, ps, "o", color=BLUE, ms=4.5, clip_on=False)
        few = len(ks) <= 4
        ax.set_xlim(ks[0] - 0.6 if few else ks[0] - 0.8,
                    ks[-1] + 0.6 if few else ks[-1] + 0.8)
        ax.set_xticks(ks if few else [0, 5, 10])
        if few:   # the two-bar-scale panels share 0-0.75
            ax.set_ylim(0, 0.75); ax.set_yticks([0, 0.25, 0.5])
        else:     # binomial and Poisson share 0-0.25
            ax.set_ylim(0, 0.25); ax.set_yticks([0, 0.1, 0.2])
        ax.tick_params(colors="black", labelsize=13)
        ax.set_title(title, fontsize=15)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # the running coin: 9 heads in 13 fair flips
    ax = axes[1]
    p9 = math.comb(13, 9) / 2 ** 13
    ax.vlines([9], 0, [p9], color=ORANGE, lw=2.8)
    ax.plot([9], [p9], "o", color=ORANGE, ms=5.5)
    ax.text(9.45, 0.098, r"$0.087$", fontsize=14,
            color=fl.T.ORANGE.dark, ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    # the categorical against the uniform 1/4
    ax = axes[3]
    ax.axhline(0.25, color=fl.T.FAINT, lw=1.2, ls=(0, (4, 3)))
    ax.text(4.5, 0.30, "uniform", fontsize=13, color="black", ha="right",
            va="bottom")
    ax.set_xlabel("class", fontsize=14, color="black")
    axes[1].set_xlabel("heads", fontsize=14, color="black")
    axes[2].set_xlabel("count", fontsize=14, color="black")
    axes[0].set_xlabel("outcome", fontsize=14, color="black")
    fl.save(fig, "mdl-prob-discrete-pmfs")
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-prob-discrete-pmfs.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-prob-discrete-pmfs.svg"))


def fig_continuous_pdfs():
    """Uniform U(-sqrt3, sqrt3), the standard Gaussian and Laplace(0, 1/sqrt2),
    all of variance one: (a) densities, (b) the same on a logarithmic axis,
    where the Gaussian is a parabola and the Laplace two straight lines."""
    import shutil
    x = np.linspace(-4, 4, 801)
    r3 = np.sqrt(3.0)
    gauss = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    b = 1 / np.sqrt(2)                       # var = 2 b^2 = 1
    lap = np.exp(-np.abs(x) / b) / (2 * b)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4),
                             gridspec_kw=dict(wspace=0.22))
    for ax, logy in zip(axes, (False, True)):
        bottom = 1e-4 if logy else 0.0
        ax.plot([-4, -r3, -r3, r3, r3, 4],
                [bottom, bottom, 1 / (2 * r3), 1 / (2 * r3), bottom, bottom],
                color=GRAY, lw=2.0, label="uniform")
        ax.plot(x, gauss, color=BLUE, lw=2.4, label="Gaussian")
        ax.plot(x, lap, color=ORANGE, lw=2.4,
                label="Laplace")
        ax.set_xlim(-4, 4)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.tick_params(colors="black", labelsize=13)
        ax.set_xlabel("$x$", fontsize=14, color="black")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if logy:
            ax.set_yscale("log"); ax.set_ylim(1e-4, 1.0)
        else:
            ax.set_ylim(0, 0.75); ax.set_yticks([0, 0.25, 0.5, 0.75])
    axes[0].set_ylabel("density", fontsize=14, color="black")
    axes[0].legend(fontsize=13, loc="upper left", handlelength=1.4,
                   frameon=False)
    fl.save(fig, "mdl-prob-continuous-pdfs")
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-prob-continuous-pdfs.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-prob-continuous-pdfs.svg"))


def fig_mvn_contours():
    """The running covariance Sigma = [[2, 1], [1, 2]].  (a) 400 samples, the
    contours at Mahalanobis radius 1, 2, 3, and the eigenvectors v1 along
    (1, 1) and v2 along (1, -1), drawn to the radius-2 contour (lengths
    2 sqrt3 and 2).  (b) The same contours sliced at x2 = 2: the conditional
    density N(1, 3/2) of x1, and the line x1 = x2 / 2 of conditional means."""
    import shutil
    rng = np.random.default_rng(0)
    cov = np.array([[2.0, 1.0], [1.0, 2.0]])
    L = np.linalg.cholesky(cov)
    pts = (L @ rng.standard_normal((2, 400))).T
    gam, V = np.linalg.eigh(cov)                 # ascending: 1, 3
    B = V @ np.diag(np.sqrt(gam))
    th = np.linspace(0, 2 * np.pi, 400)
    circ = np.stack([np.cos(th), np.sin(th)])
    lim = 5.0
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0),
                             gridspec_kw=dict(wspace=0.08))
    for ax in axes:
        fl.axis_cross(ax, (-lim, lim), (-lim, lim), color="black")
        ax.text(lim, -0.12, "$x_1$", fontsize=15, ha="right", va="top",
                color="black")
        ax.text(-0.15, lim, "$x_2$", fontsize=15, ha="right", va="top",
                color="black")
        for r in (1, 2, 3):
            e = r * (B @ circ)
            ax.plot(e[0], e[1], color=GRAY, lw=1.1, zorder=3)
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axis("off")
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=BLUE, alpha=0.30, lw=0,
               zorder=2)
    for v, g, lab in ((np.array([1.0, 1.0]) / np.sqrt(2), 3.0, r"$\mathbf{v}_1$"),
                      (np.array([1.0, -1.0]) / np.sqrt(2), 1.0, r"$\mathbf{v}_2$")):
        tip = 2 * np.sqrt(g) * v
        fl.arrow(ax, (0, 0), tuple(tip), color=ORANGE, lw=2.2, mut=14)
        lp = tip + 0.55 * v
        ax.text(lp[0], lp[1], lab, color=fl.T.ORANGE.dark, fontsize=16,
                ha="center", va="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax = axes[1]
    x2s = np.array([-lim, lim])
    ax.plot(x2s / 2, x2s, color=ORANGE, lw=2.0, zorder=4)
    ax.plot([-lim, lim], [2, 2], color="black", lw=1.0, ls=(0, (4, 3)),
            zorder=4)
    x1 = np.linspace(1 - 3.3, 1 + 3.3, 400)
    dens = np.exp(-(x1 - 1) ** 2 / 3.0) / np.sqrt(3 * np.pi)
    ax.fill_between(x1, 2, 2 + 4.0 * dens, color=fl.T.BLUE.tint, zorder=3,
                    lw=0)
    ax.plot(x1, 2 + 4.0 * dens, color=BLUE, lw=2.0, zorder=5)
    ax.plot([1], [2], "o", color=ORANGE, ms=7, zorder=6)
    ax.text(-lim + 0.05, 2.12, "$x_2=2$", fontsize=14, ha="left", va="bottom",
            color="black")
    ax.text(-0.6, 4.1, r"$x_1\mid x_2=2\sim\mathcal{N}(1,\frac{3}{2})$",
            fontsize=14, ha="right", va="center", color="black",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=7)
    ax.text(-2.0, -4.62, "$x_1=x_2/2$", fontsize=14, ha="left",
            va="center", color=fl.T.ORANGE.dark,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=7)
    fl.save(fig, "mdl-prob-mvn-contours")
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-prob-mvn-contours.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-prob-mvn-contours.svg"))


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
        node(ax, ynode, r"$y$", fc=fl.T.BLUE.tint)
        for x, lab in zip(xs, feat_labels):
            if lab == r"$\cdots$":
                continue
            node(ax, (x, y_bot), lab)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.05, 2.1)
        ax.set_aspect("equal")
        ax.axis("off")

    # right panel only: a couple of among-feature edges -- the conditional
    # dependence that remains in real data (e.g. neighboring pixels) and that a
    # richer model keeps but naive Bayes drops.  Undirected arcs bowed upward
    # between the feature nodes, in orange so they read as the addition.
    from matplotlib.patches import FancyArrowPatch
    # arc curvature per edge (bows the edge upward); the long (x_3,x_d) arc
    # bows higher (rad=-0.6) so it clears the "..." glyph sitting between them
    extra = [(0, 1, -0.42), (2, 4, -0.6)]   # (x_1,x_2) neighbors, (x_3,x_d)
    for i, j, rad in extra:
        a = np.array([xs[i], y_bot]); b = np.array([xs[j], y_bot])
        # trim the endpoints to the node rims so the edge meets the circles
        u = (b - a) / np.linalg.norm(b - a)
        pa, pb = a + 0.30 * u, b - 0.30 * u
        edge = FancyArrowPatch(
            pa, pb, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
            color=ORANGE, lw=2.0, zorder=2)
        axR.add_patch(edge)
    axR.text(0.0, -1.55, "dependence among features", ha="center",
             va="top", color=fl.T.ORANGE.dark, fontsize=11)

    fl.save(fig, "mdl-prob-naive-independence")


def fig_naive_genvdisc():
    """Two routes to a linear classifier on the same 2-D points.  Two classes
    share the tilted covariance Sigma = [[1, .65], [.65, .6]] (correlation .84)
    and have means -/+(1, .2), equal priors.  (a) The naive generative fit
    models each class by an axis-aligned Gaussian (the per-feature variances
    1 and .6, drawn as 1- and 2-sd ellipses) and takes the boundary from Bayes'
    rule: normal (2, .4/.6), population error 19.2%.  (b) The direct fit of
    p(y|x): the population-optimal boundary, normal Sigma^-1 (2, .4), error
    7.2%.  Population values, not fitted ones; the points are a fixed draw."""
    from matplotlib.patches import Ellipse
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.75))
    Sigma = np.array([[1.0, 0.65], [0.65, 0.6]])
    mus = (np.array([-1.0, -0.2]), np.array([1.0, 0.2]))
    rng = np.random.default_rng(3)
    L = np.linalg.cholesky(Sigma)
    clouds = [m + rng.standard_normal((60, 2)) @ L.T for m in mus]
    (x0, x1), (y0, y1) = (-3.7, 3.7), (-2.5, 2.5)
    dmu = mus[1] - mus[0]
    normals = (dmu / np.diag(Sigma), np.linalg.solve(Sigma, dmu))
    titles = ("naive generative fit: error 19%",
              r"direct fit of $p(y\mid\mathbf{x})$: error 7%")
    xx = np.array([x0, x1])
    for ax, nvec, title in zip((axa, axb), normals, titles):
        # boundary n . x = 0 through the midpoint of the means (the origin)
        yy = -nvec[0] / nvec[1] * xx
        side = np.sign(nvec[1])   # +1: class 1 lies above the line
        top, bot = (ORANGE, BLUE) if side > 0 else (BLUE, ORANGE)
        ax.fill_between(xx, yy, y1, color=top, alpha=0.08, lw=0)
        ax.fill_between(xx, y0, yy, color=bot, alpha=0.08, lw=0)
        for c, col in zip(clouds, (BLUE, ORANGE)):
            ax.scatter(c[:, 0], c[:, 1], s=13, color=col, alpha=0.55, lw=0)
        ax.plot(xx, yy, color="black", lw=2.0, zorder=4)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(r"$x_1$", fontsize=14, loc="right", labelpad=2)
        ax.set_ylabel(r"$x_2$", fontsize=14, loc="top", rotation=0, labelpad=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("black")
    # (a) only: the axis-aligned Gaussians that the naive fit uses
    sd = np.sqrt(np.diag(Sigma))
    for m, col in zip(mus, (BLUE, ORANGE)):
        for k in (1, 2):
            axa.add_patch(Ellipse(m, 2 * k * sd[0], 2 * k * sd[1], fill=False,
                                  color=col, lw=1.8 if k == 1 else 1.2, zorder=3))
    for ax in (axa, axb):
        ax.text(-2.9, 1.9, r"$y=0$", color=BLUE, fontsize=14, ha="center",
                va="center")
        ax.text(2.9, -1.9, r"$y=1$", color=ORANGE, fontsize=14, ha="center",
                va="center")
    fl.save(fig, "mdl-prob-naive-genvdisc")


# =========================================================================== #
# RANDOM VARIABLES: conditioning as slicing                                   #
# =========================================================================== #

def fig_conditional_slice():
    """Conditioning as slicing, for a dependent and an independent pair.
    Left column: a joint density with three horizontal slices at y0 = -1, 0, 1.
    Right column: each slice renormalized to unit area, the conditional
    density p(x | y0).  Top: a correlated Gaussian (sx=1, sy=0.9, rho=0.55);
    the three conditionals have the same width but centres at
    rho*sx/sy*y0 = -0.61, 0, 0.61, so the conditional depends on y0.  Bottom:
    the product Gaussian (rho=0); the three conditionals coincide (drawn with
    decreasing widths so every colour stays visible)."""
    xs = np.linspace(-3.0, 3.4, 260)
    ys = np.linspace(-2.8, 2.8, 260)
    X, Y = np.meshgrid(xs, ys)
    dx = xs[1] - xs[0]
    sx, sy = 1.0, 0.9
    y0s, cols, lws = [-1.0, 0.0, 1.0], [BLUE, GREEN, ORANGE], [3.6, 2.6, 1.6]

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2))
    for row, (rho, name) in enumerate([(0.55, "dependent"),
                                       (0.0, "independent")]):
        z = X ** 2 / sx ** 2 - 2 * rho * X * Y / (sx * sy) + Y ** 2 / sy ** 2
        joint = np.exp(-z / (2 * (1 - rho ** 2)))
        axj, axc = axes[row]
        axj.contourf(X, Y, joint, levels=10, cmap="Greys", alpha=0.55)
        for y0, col, lw in zip(y0s, cols, lws):
            axj.axhline(y0, color=col, lw=2.2)
            axj.text(3.3, y0 + 0.08, rf"$y_0={y0:g}$", color=col,
                     ha="right", va="bottom", fontsize=13,
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.6,
                               alpha=0.85))
            j0 = int(np.argmin(np.abs(ys - y0)))
            sl = joint[j0, :]
            cond = sl / (sl.sum() * dx)            # renormalize -> p(x|y0)
            axc.plot(xs, cond, color=col, lw=lw if rho == 0 else 2.4)
        axj.set_xlabel(r"$x$", fontsize=14)
        axj.set_ylabel(r"$y$", fontsize=14)
        axj.set_yticks([-2, 0, 2])
        axj.text(-0.30, 0.5, name, transform=axj.transAxes, rotation=90,
                 ha="center", va="center", fontsize=15, color=fl.INK)
        axc.set_xlabel(r"$x$", fontsize=14)
        axc.set_ylabel(r"$p(x \mid y_0)$", fontsize=14)
        axc.set_yticks([])
        axc.set_ylim(0, 0.55)
        axc.spines["top"].set_visible(False)
        axc.spines["right"].set_visible(False)
        for ax in (axj, axc):
            ax.set_xlim(xs[0], xs[-1])
            ax.set_xticks([-2, 0, 2])
    fig.tight_layout()
    fl.save(fig, "mdl-prob-conditional-slice")


# =========================================================================== #
# MAXIMUM LIKELIHOOD: NLL = cross-entropy = floor + KL                         #
# =========================================================================== #

def fig_mle_kl():
    """The coin's per-flip cross-entropy CE(theta) = -(9/13 log theta + 4/13
    log(1 - theta)) as a function of the model bias.  It never falls below the
    entropy H(9/13) = 0.617 nats of the observed frequencies (dashed floor),
    touches the floor at the MLE 9/13, and its excess over the floor at any
    other theta is D_KL(p_hat || p_theta): 0.076 nats at theta = 1/2 and
    0.164 nats at theta = 0.9 (orange double arrows)."""
    import shutil

    p = 9 / 13
    th = np.linspace(0.2, 0.98, 781)
    ce = -(p * np.log(th) + (1 - p) * np.log(1 - th))
    floor = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(th, ce, color=BLUE, lw=2.4, zorder=4)
    ax.axhline(floor, color=GRAY, lw=1.2, ls="--", zorder=2)
    ax.text(0.21, floor - 0.012, r"$H(\hat p) = 0.617$ nats", color="black",
            fontsize=13, ha="left", va="top")
    ax.plot([p], [floor], "o", color=BLUE, ms=7, zorder=6)
    ax.plot([p, p], [0.55, floor], color=GRAY, lw=1.0, ls=":", zorder=1)
    # the KL excess at two other biases, labelled below the curve on the
    # side with room (the right label is a bare number to stay inside)
    for t0, side, fmt in ((0.5, "left", r"$D_{\mathrm{KL}} = %.3f$"),
                          (0.9, "right", r"$%.3f$")):
        c0 = -(p * np.log(t0) + (1 - p) * np.log(1 - t0))
        ax.annotate("", xy=(t0, c0), xytext=(t0, floor),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.8,
                                    shrinkA=0, shrinkB=0), zorder=5)
        dx = -0.012 if side == "left" else 0.012
        ax.text(t0 + dx, (floor + c0) / 2,
                fmt % (c0 - floor), color="black",
                fontsize=13, ha="right" if side == "left" else "left",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_xlim(0.2, 0.98)
    ax.set_ylim(0.55, 1.25)
    ax.set_xticks([0.2, 0.5, p, 0.9])
    ax.set_xticklabels(["0.2", "0.5", r"$9/13$", "0.9"])
    ax.set_yticks([0.6, 0.8, 1.0, 1.2])
    ax.set_xlabel(r"model bias $\theta$", fontsize=13)
    ax.set_ylabel(r"$\mathrm{CE}(\hat p, p_\theta)$  [nats]", fontsize=13)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
    ax.tick_params(colors="black")
    fl.save(fig, "mdl-prob-mle-kl")
    # the slides read @fig figures from img/auto
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-prob-mle-kl.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-prob-mle-kl.svg"))


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
    # every NLL contour gray; the innermost (NLL min + 1/2) is the one the
    # dashed orange ellipse below nearly traces, so the gray shows through
    # the dash gaps rather than reading as a second, misregistered ring
    ax.contour(M, S, nll, levels=levels, colors=GRAY, linewidths=1.0)
    ax.plot([mu_hat], [sig_hat], "o", color=BLUE, ms=7, zorder=6)

    # 1-sigma ellipse of the asymptotic covariance I^{-1}/n (I is diagonal)
    a = np.sqrt(sig_hat ** 2 / n)              # semi-axis along mu
    b = np.sqrt(sig_hat ** 2 / (2 * n))        # semi-axis along sigma
    ax.add_patch(Ellipse((mu_hat, sig_hat), 2 * a, 2 * b, fill=False,
                         edgecolor=ORANGE, lw=2.2, ls=(0, (4, 3)), zorder=5))
    # the MLE label sits outside the ellipse, to its right at the dot's
    # height, in the band before the next ring; the information-matrix label
    # sits in the band between the next two gray rings above it, on a white
    # backing so no ring runs through it.
    ax.text(mu_hat + a + 0.025, sig_hat, r"$(\hat\mu,\hat\sigma)$", color=BLUE,
            ha="left", va="center", fontsize=11, zorder=7)
    ax.annotate(r"$I(\hat\mu,\hat\sigma)^{-1}/n$",
                xy=(mu_hat + a * 0.3, sig_hat + b * 0.97),
                xytext=(mu_hat, sig_hat + 0.53), color=ORANGE,
                fontsize=11, ha="center", va="center", zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                                shrinkA=2, shrinkB=1))

    ax.set_xlabel(r"$\mu$", fontsize=13)
    ax.set_ylabel(r"$\sigma$", fontsize=13)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0]); ax.set_yticks([1.0, 1.5, 2.0])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-mle-fisher")


def fig_elbo():
    """The evidence lower bound.  A one-latent-bit model with a sign ambiguity:
    z in {0,1} with p(z=1) = 1/2 and p(x|z; theta) = N(x; (2z-1) theta, 1),
    observing x = 2, so the evidence log p(x; theta) is symmetric in theta.
    Fixing q to the posterior at theta^(t) = 0.4 (x = 2), the orange ELBO L(q, theta)
    touches the evidence at theta^(t) and runs strictly below it on both sides
    (for theta < 0 the posterior flips to z = 0, so the gap opens wide); the gap
    at any other theta is KL(q || p(z|x; theta)).  The M-step climbs the bound to
    theta^(t+1) = (q1 - q0) x."""
    x_obs, th_t = 2.0, 0.4
    th = np.linspace(-3.0, 3.6, 700)

    def log_norm(x, mu):                       # log N(x; mu, 1)
        return -0.5 * (x - mu) ** 2 - 0.5 * np.log(2 * np.pi)

    def log_evidence(theta):                   # log p(x; theta), z marginalized
        l0 = np.log(0.5) + log_norm(x_obs, -theta)
        l1 = np.log(0.5) + log_norm(x_obs, theta)
        m = np.maximum(l0, l1)
        return m + np.log(np.exp(l0 - m) + np.exp(l1 - m))

    # E-step at theta^(t): q(z) = p(z | x; theta^(t))
    l0t = np.log(0.5) + log_norm(x_obs, -th_t)
    l1t = np.log(0.5) + log_norm(x_obs, th_t)
    q1 = 1.0 / (1.0 + np.exp(l0t - l1t)); q0 = 1.0 - q1
    entropy = -(q0 * np.log(q0) + q1 * np.log(q1))

    evid = log_evidence(th)
    elbo = (q0 * (np.log(0.5) + log_norm(x_obs, -th))
            + q1 * (np.log(0.5) + log_norm(x_obs, th)) + entropy)
    th_next = th[np.argmax(elbo)]              # M-step: argmax of the bound

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    # the bound falls away fast from its peak; let it leave through the bottom
    # of the window rather than squash the evidence's two bumps flat
    ytop = evid.max() + 0.30
    ybot = evid.min() - 1.6
    ax.plot(th, evid, color=BLUE, lw=2.4)
    ax.plot(th, elbo, color=ORANGE, lw=2.2)
    ax.text(-1.9, evid.max() + 0.08, r"$\log p(x;\theta)$", color=BLUE,
            ha="center", va="bottom", fontsize=11)
    ax.text(-0.95, ybot + 0.45, r"ELBO $\mathcal{L}(q,\theta)$",
            color=ORANGE, ha="right", va="center", fontsize=11)

    # the touch point at theta^(t) and the bound's peak at theta^(t+1)
    y_t = float(log_evidence(np.array([th_t]))[0])
    y_next = elbo.max()
    for tv, yv in ((th_t, y_t), (th_next, y_next)):
        # dashed drop from each iterate down to its own tick on the axis
        ax.plot([tv, tv], [ybot, yv], color=GRAY, lw=1.1, ls="--", zorder=2)
        ax.plot([tv], [yv], "o", color=GRAY, ms=6, zorder=6)
    # the iterates *are* the only meaningful theta values: they become the ticks
    ax.set_xticks([th_t, th_next])
    ax.set_xticklabels([r"$\theta^{(t)}$", r"$\theta^{(t+1)}$"], fontsize=13)

    # the gap between the curves at some other theta is the KL divergence
    th_g = 2.75
    g_lo = float(np.interp(th_g, th, elbo))
    g_hi = float(np.interp(th_g, th, evid))
    ax.annotate("", xy=(th_g, g_hi), xytext=(th_g, g_lo),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6,
                                shrinkA=0, shrinkB=0))
    # short label (the caption spells out "gap = KL divergence" in full) placed
    # to the right of the arrow, clear of both curves at every theta it spans
    ax.text(th_g + 0.15, (g_lo + g_hi) / 2, "gap\n(KL)",
            color=GREEN, ha="left", va="center", fontsize=11)

    ax.set_xlabel(r"$\theta$", fontsize=13)
    ax.set_ylabel("objective", fontsize=13)
    ax.set_yticks([])
    ax.set_xlim(th[0], th[-1])
    ax.set_ylim(ybot, ytop)
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
    # seed 93: exactly 5 misses, and every miss clears mu by a visible margin
    # (an interval ending right on the dashed line would read as a hit)
    rng = np.random.default_rng(93)
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
    # both labels live outside the bars: "true mu" past the last interval at
    # the end of its line, the miss count in a band above the tallest bar
    ax.plot([-1.5, m + 0.5], [mu, mu], color=GRAY, lw=1.2, ls="--", zorder=1)
    ax.text(m + 1.2, mu, r"true $\mu$", color=fl.INK, fontsize=11,
            ha="left", va="center")
    ax.text(0, hi.max() + 0.06, f"{int(miss.sum())} of {m} intervals miss",
            color=ORANGE, fontsize=11, ha="left", va="bottom")
    ax.set_xlim(-1.5, m + 0.5)
    ax.set_ylim(lo.min() - 0.07, hi.max() + 0.2)
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
    # left: observed sample up top, two resamples beneath.  Each value is a
    # tick standing on its row's baseline; a resample's tick is as tall as
    # the number of copies drawn, so "with replacement" is visible (a value
    # drawn twice stands twice as tall, one never drawn is missing).
    cx = np.max(sample) * 0.5
    unit = 0.32

    def strip(vals, base, color):
        u, c = np.unique(vals, return_counts=True)
        axs.vlines(u, base, base + unit * c, color=color, lw=1.6)
        axs.plot([0, sample.max()], [base, base], color=fl.T.HAIRLINE,
                 lw=1.0, zorder=0)

    strip(sample, 4.3, BLUE)
    for k in range(2):
        rs = rng.choice(sample, size=60, replace=True)
        strip(rs, 1.75 - 1.75 * k, GRAY)
    axs.text(cx, 4.78, "observed sample", color=BLUE, fontsize=11,
             ha="center", va="bottom")
    axs.text(cx, 3.6, "resamples (with replacement):\ntick height = copies drawn",
             color=fl.INK, fontsize=11, ha="center", va="center",
             linespacing=1.2)
    axs.set_ylim(-0.25, 5.3); axs.set_yticks([])
    axs.set_xlabel("value", fontsize=12)
    for sp in ("top", "right", "left"):
        axs.spines[sp].set_visible(False)

    # right: histogram of bootstrap medians + central-95% percentile band
    counts, _, _ = axh.hist(reps, bins=40, color=BLUE, alpha=0.7)
    # percentile cut lines plus a span arrow above the histogram (no shaded
    # band: an orange wash under translucent blue bars mixes to gray)
    top = counts.max()
    for v in (lo, hi):
        axh.plot([v, v], [0, top * 1.08], color=ORANGE, lw=1.8)
    axh.annotate("", xy=(hi, top * 1.08), xytext=(lo, top * 1.08),
                 arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4,
                                 shrinkA=0, shrinkB=0))
    axh.text((lo + hi) / 2, top * 1.12, r"central $95\%$",
             color=ORANGE, ha="center", va="bottom", fontsize=11)
    # the observed sample's own median: the estimate the replicates scatter
    # around
    th_obs = float(np.median(sample))
    axh.plot([th_obs, th_obs], [0, top * 0.93], color=fl.INK, lw=1.4,
             ls=(0, (4, 3)), zorder=4)
    # label left of the line, in the gap above the shorter bars there (the
    # tallest bar sits just right of the line)
    axh.text(th_obs - 0.02, top * 0.86, r"$\hat\theta$", color=fl.INK,
             fontsize=13, ha="right", va="center")
    axh.set_ylim(0, top * 1.3)
    axh.set_xlabel(r"$\hat\theta^{*}$  (median)", fontsize=12)
    axh.set_yticks([])
    axh.spines["top"].set_visible(False); axh.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-bootstrap")


# =========================================================================== #
# BAYESIAN COMPUTATION (sec_mdl-bayesian-computation)                         #
# =========================================================================== #

def fig_bayes_importance():
    """Importance sampling depends on proposal coverage.  Same bimodal target
    in both panels; stems show 24 seeded draws from the proposal with height
    equal to the *normalized* weight p/q.  The narrow proposal never reaches
    the right-hand mode: its weights are nearly equal (ESS 23 of 24), yet its
    estimate is wrong.  The covering proposal samples both modes: its weights
    are less equal (ESS 17 of 24), yet its estimate is far better.  ESS
    measures how unequal the weights are, not whether the mass was found."""
    x = np.linspace(-3.2, 4.4, 700)

    def target(t):
        return (0.65 * np.exp(-0.5 * ((t - 0.0) / 0.55) ** 2)
                / (0.55 * np.sqrt(2 * np.pi))
                + 0.35 * np.exp(-0.5 * ((t - 2.3) / 0.45) ** 2)
                / (0.45 * np.sqrt(2 * np.pi)))

    def gauss(t, m, s):
        return np.exp(-0.5 * ((t - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    rng = np.random.default_rng(5)
    draws = rng.standard_normal(24)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=True)
    for ax, (m, s), title in (
            (axa, (-0.1, 0.55), "narrow proposal"),
            (axb, (0.9, 1.45), "covering proposal")):
        theta = m + s * draws
        w = target(theta) / gauss(theta, m, s)
        w = w / w.sum()
        ess = 1.0 / np.sum(w ** 2)
        ax.plot(x, target(x), color=BLUE, lw=2.4, zorder=4)
        ax.plot(x, gauss(x, m, s), color=ORANGE, lw=2.0, ls="--", zorder=4)
        # weights on their own scale so both panels compare directly
        ax.vlines(theta, 0.0, w, color=GREEN, lw=1.6, zorder=3)
        ax.plot(theta, w, "o", color=GREEN, ms=4, zorder=5)
        # one-line functional title: which proposal, and its ESS (the caption
        # carries the interpretation)
        ax.set_title(f"{title},  ESS $\\approx$ {ess:.0f} of 24", fontsize=12)
        ax.set_xlabel(r"$\theta$", fontsize=13)
        ax.set_ylim(0, 0.78)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # beside the right-hand mode, the only curve there (the left mode is
    # crowded by the dashed proposal)
    axa.text(2.85, 0.20, r"target $p$", color=BLUE, ha="left", fontsize=11)
    # proposal label left of its own dashed flank; the coverage note sits
    # centred over the right-hand mode it talks about, with a short leader
    axa.text(-0.9, 0.47, r"proposal $q$", color=ORANGE, ha="right",
             va="center", fontsize=11)
    axa.annotate("no draw ever lands here:\nthe weights cannot know",
                 xy=(2.3, 0.335), xytext=(2.6, 0.45), color=GREEN,
                 ha="center", va="bottom", fontsize=11,
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1,
                                 shrinkA=2, shrinkB=2))
    axb.text(-1.95, 0.13, r"proposal $q$", color=ORANGE, ha="right",
             fontsize=11)
    # the note sits just above the right-hand stems it describes
    axb.text(3.2, 0.12, "both modes\nsampled", color=GREEN, ha="left",
             va="bottom", fontsize=11)
    fl.save(fig, "mdl-prob-bayes-importance")


def fig_bayes_metropolis():
    """What a Metropolis random walk actually does: crawl over the posterior
    contours by local proposals, always accepting uphill moves and sometimes
    accepting downhill ones.  Correlated Gaussian target (contours), 40 seeded
    steps; rejected proposals are gray crosses tied to the state that refused
    them, and the chain repeats that state.  The proposal scale (0.38) and the
    seed are chosen so the steps stay local relative to the contours and every
    proposal, accepted or not, lands inside the window."""
    cov = np.array([[0.65, -0.45], [-0.45, 0.55]])
    prec = np.linalg.inv(cov)

    def logp(t):
        return -0.5 * np.einsum("...i,ij,...j->...", t, prec, t)

    g = np.linspace(-2.6, 2.6, 300)
    GX, GY = np.meshgrid(g, g)
    Z = np.exp(logp(np.stack([GX, GY], axis=-1)))

    rng = np.random.default_rng(221)
    cur = np.array([-2.1, -1.9])
    path = [cur.copy()]
    rejected = []                       # (from-state, rejected proposal)
    for _ in range(40):
        prop = cur + 0.38 * rng.standard_normal(2)
        if np.log(rng.random()) < logp(prop) - logp(cur):
            cur = prop
        else:
            rejected.append((path[-1].copy(), prop.copy()))
        path.append(cur.copy())
    path = np.array(path)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.contour(GX, GY, Z, levels=[0.05, 0.2, 0.45, 0.75],
               colors=[BLUE], linewidths=1.1, alpha=0.85)
    for src, rej in rejected:
        ax.plot([src[0], rej[0]], [src[1], rej[1]], color=GRAY, lw=0.9,
                ls=(0, (1.6, 2.2)), zorder=3)
        ax.plot([rej[0]], [rej[1]], "x", color=GRAY, ms=6, mew=1.4, zorder=3)
    ax.plot(path[:, 0], path[:, 1], color=ORANGE, lw=1.6, zorder=4)
    ax.plot(path[:, 0], path[:, 1], "o", color=ORANGE, ms=3.5, zorder=5)
    ax.plot(path[0, 0], path[0, 1], "s", color="black", ms=7, zorder=6)
    # left of the square: the chain's first steps leave it to the right
    ax.text(path[0, 0] - 0.13, path[0, 1], "start", fontsize=11,
            ha="right", va="center", color="black")
    # key for the crosses: a sample cross beside the text
    ax.plot([1.0], [2.31], "x", color=GRAY, ms=6, mew=1.4)
    ax.text(1.2, 2.52, "rejected proposal:\nchain repeats its state",
            color=fl.INK, fontsize=11, ha="left", va="top",
            linespacing=1.3, transform=ax.transData)
    ax.text(-1.4, 2.15, "posterior contours", color=BLUE, fontsize=11,
            ha="center")
    ax.set_xlabel(r"$\theta_0$", fontsize=13)
    ax.set_ylabel(r"$\theta_1$", fontsize=13)
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-2.55, 2.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-bayes-metropolis")


def fig_bayes_laplace():
    """The Laplace approximation is a quadratic fit to the log posterior at
    its mode.  Left: log density of a Gamma(4, 1) posterior and the tangent
    parabola at the mode — indistinguishable locally.  Right: exponentiating
    both shows what the Gaussian keeps (location, curvature) and what it
    misses (the skew: too much mass left of the mode, some of it below the
    theta = 0 support edge, and too little in the right tail)."""
    x = np.linspace(0.35, 10.5, 800)
    k = 4.0                                     # Gamma(k, 1), mode at k-1
    mode = k - 1.0
    logf = (k - 1) * np.log(x) - x              # up to a constant
    logf -= logf.max()
    curv = (k - 1) / mode ** 2                  # -d^2/dx^2 log f at the mode
    quad = -0.5 * curv * (x - mode) ** 2

    from math import gamma as _gamma
    xd = np.linspace(0.0, 10.5, 900)            # the density is fine at 0
    dens = xd ** (k - 1) * np.exp(-xd) / _gamma(k)
    # the Gaussian has no support constraint: draw it (and the quadratic) on
    # a grid that runs below theta = 0, where the Gamma posterior is zero
    xg = np.linspace(-2.2, 10.5, 950)
    quad_g = -0.5 * curv * (xg - mode) ** 2
    gauss = (np.exp(-0.5 * (xg - mode) ** 2 * curv)
             * np.sqrt(curv / (2 * np.pi)))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    axa.plot(x, logf, color=BLUE, lw=2.4)
    axa.plot(xg, quad_g, color=ORANGE, lw=2.0, ls="--")
    axa.plot([mode], [0.0], "o", color="black", ms=5, zorder=6)
    # the note sits top-right, above where the blue curve has already fallen
    # away, with its leader running above both curves to the mode
    axa.annotate("same value, slope,\nand curvature at the mode",
                 xy=(mode, 0.0), xytext=(4.9, 0.32), fontsize=11,
                 ha="left", va="center", color="black",
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2,
                                 shrinkA=3, shrinkB=5))
    axa.text(1.02, -2.45, r"$\log p(\theta\mid\mathcal{D})$", color=BLUE,
             fontsize=11, ha="left")
    # just left of the falling parabola, well below the blue curve
    axa.text(7.3, -3.6, "quadratic fit", color=ORANGE, fontsize=11,
             ha="right", va="center")
    axa.set_ylim(-4.4, 0.85)
    axa.set_xlabel(r"$\theta$", fontsize=13)
    axa.set_ylabel("log density", fontsize=13)
    axa.set_yticks([])
    # the posterior lives on theta > 0: mark the support edge, and shade the
    # Gaussian mass that falls beyond it
    axb.axvline(0.0, color=fl.INK, lw=1.1, ls=":", zorder=1)
    axb.fill_between(xg, 0, gauss, where=xg <= 0, color=ORANGE, alpha=0.25,
                     lw=0)
    axb.text(-0.2, 0.255, "support\nedge", color=fl.INK, fontsize=11,
             ha="right", va="top")
    axb.plot(xd, dens, color=BLUE, lw=2.4)
    axb.plot(xg, gauss, color=ORANGE, lw=2.0, ls="--")
    axb.text(4.75, 0.208, "true posterior", color=BLUE, fontsize=11,
             ha="left")
    # the Gaussian's label sits in the open top-left corner, above its flank
    axb.text(0.35, 0.212, "Laplace", color=ORANGE,
             fontsize=11, ha="left", va="bottom")
    axb.fill_between(xd, dens, np.interp(xd, xg, gauss), where=xd > 4.9,
                     color=BLUE, alpha=0.15, lw=0)
    # the tail note sits above the curves, with a leader into the shaded band
    axb.annotate("tail mass the\nGaussian drops", xy=(6.1, 0.068),
                 xytext=(7.0, 0.135), color=BLUE, fontsize=11, ha="left",
                 va="bottom",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1,
                                 shrinkA=2, shrinkB=0))
    axb.set_ylim(0, 0.27)
    axb.set_xlabel(r"$\theta$", fontsize=13)
    axb.set_ylabel("density", fontsize=13)
    axb.set_yticks([])
    for ax in (axa, axb):
        ax.set_xlim(-2.3, 10.7)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-bayes-laplace")


def fig_bayes_kl_modes():
    """Which KL you optimize decides how a too-simple family fails.  Bimodal
    target; the reverse-KL Gaussian (what variational inference minimizes)
    locks onto one mode, because putting q-mass where p is tiny is charged
    infinitely; the forward-KL Gaussian (moment matching) covers both modes
    and puts mass in the valley between them.  Both Gaussians found by grid
    minimization of their KL, not by sketching."""
    x = np.linspace(-4.6, 4.6, 1400)
    dx = x[1] - x[0]

    def gauss(t, m, s):
        return np.exp(-0.5 * ((t - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    p = 0.5 * gauss(x, -1.6, 0.5) + 0.5 * gauss(x, 1.6, 0.5)

    # forward KL(p || q): moment matching is the exact optimum
    m_f = np.sum(x * p) * dx
    s_f = np.sqrt(np.sum((x - m_f) ** 2 * p) * dx)
    q_f = gauss(x, m_f, s_f)

    # reverse KL(q || p): coarse grid minimization
    best = (np.inf, 0.0, 1.0)
    for m in np.linspace(-2.4, 2.4, 97):
        for s in np.linspace(0.25, 2.4, 87):
            q = gauss(x, m, s)
            kl = np.sum(q * (np.log(q + 1e-300) - np.log(p + 1e-300))) * dx
            if kl < best[0]:
                best = (kl, m, s)
    q_r = gauss(x, best[1], best[2])

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(x, p, color=BLUE, lw=2.6, zorder=4)
    ax.plot(x, q_r, color=ORANGE, lw=2.2, ls="--", zorder=4)
    ax.plot(x, q_f, color=GREEN, lw=2.2, ls="-.", zorder=4)
    top = max(p.max(), q_r.max(), q_f.max())
    ax.text(-1.6, p.max() + 0.035, r"target $p$", color=BLUE, fontsize=11.5,
            ha="center")
    ax.text(2.65, 0.62, "reverse KL:\nmode-seeking", color=ORANGE,
            fontsize=11, ha="left", va="center")
    ax.text(-3.65, 0.16, "forward KL:\nmass-covering", color=GREEN,
            fontsize=11, ha="center", va="center")
    ax.set_xlabel(r"$\theta$", fontsize=13)
    ax.set_ylabel("density", fontsize=13)
    ax.set_yticks([])
    ax.set_ylim(0, top * 1.12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-prob-bayes-kl-modes")


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
    # bayesian computation
    fig_bayes_importance,
    fig_bayes_metropolis,
    fig_bayes_laplace,
    fig_bayes_kl_modes,
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
