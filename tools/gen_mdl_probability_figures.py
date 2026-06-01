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
    """A joint-probability array p_{X,Y}(x,y) as a Blues heatmap; summing along
    the columns (over y, for fixed x) collapses it onto the marginal p_X(x),
    drawn as the bar strip beneath the grid.  The values are a real, normalized
    joint PMF (a smooth bump) so the column sums shown are the true marginal."""
    # A real joint PMF on a small grid: a Gaussian-ish bump, normalized to 1.
    nx, ny = 10, 8
    xs = np.arange(nx)
    ys = np.arange(ny)
    XX, YY = np.meshgrid(xs, ys)  # YY rows, XX cols
    joint = np.exp(-((XX - 5.5) ** 2) / 7.0 - ((YY - 3.0) ** 2) / 4.5)
    joint += 0.15 * np.exp(-((XX - 2.5) ** 2) / 3.0 - ((YY - 5.0) ** 2) / 2.0)
    joint /= joint.sum()
    marg_x = joint.sum(axis=0)  # sum down each column -> p_X(x)

    fig, (axj, axm) = plt.subplots(
        2, 1, figsize=(5.6, 5.0), sharex=True,
        gridspec_kw=dict(height_ratios=[ny, 3], hspace=0.08),
    )

    # --- joint array ---
    # imshow with origin lower so y increases upward like a probability table.
    axj.imshow(joint, cmap="Blues", origin="lower", aspect="auto",
               extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
               vmin=0, vmax=joint.max())
    # faint cell grid lines so it reads as an "array of probabilities"
    for x in np.arange(-0.5, nx, 1.0):
        axj.axvline(x, color="white", lw=0.8)
    for y in np.arange(-0.5, ny, 1.0):
        axj.axhline(y, color="white", lw=0.8)
    # highlight the column being summed (fixed x) with an outline
    col = 5
    axj.add_patch(Rectangle((col - 0.5, -0.5), 1.0, ny, fill=False,
                            edgecolor=ORANGE, lw=2.2, zorder=5))
    # downward arrows from the highlighted column into the marginal strip
    axj.annotate("", xy=(col, -0.5), xytext=(col, 1.2),
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.0))
    axj.set_ylabel(r"$y$")
    axj.set_yticks(ys)
    axj.set_title(r"joint $p_{X,Y}(x,y)$", fontsize=11)
    axj.spines["top"].set_visible(True)
    axj.spines["right"].set_visible(True)

    # --- marginal strip: bars of p_X(x) = sum over y ---
    axm.bar(xs, marg_x, width=0.86, color=BLUE, edgecolor="white", lw=0.8)
    axm.bar(col, marg_x[col], width=0.86, color=ORANGE, edgecolor="white",
            lw=0.8, zorder=4)
    axm.set_xlabel(r"$x$")
    axm.set_ylabel(r"$p_X(x)$")
    axm.set_xticks(xs)
    axm.set_ylim(0, marg_x.max() * 1.18)
    axm.set_yticks([])
    axm.spines["left"].set_visible(False)

    fl.save(fig, "mdl-prob-marginal")


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


FIGURES = [
    # random variables
    fig_marginal,
    # statistics
    fig_significance,
    fig_bias_variance_u_curve,
    fig_type_i_ii_matrix,
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
