#!/usr/bin/env python3
"""Generate the illustrative figures for the "Mathematics for Deep Learning ->
Information Theory" chapter in the project's one house style.

This reuses the shared style, palette, and helpers defined in
``tools/gen_mdl_figures.py`` (the Linear Algebra generator) so every figure in
the book reads the same; it does NOT edit that file.  All figures here are
written to ``img/mdl-it-<id>.svg`` and self-verified at the end of the run.

Run with the repo's pytorch venv (matplotlib + numpy available):

    .venv-pytorch/bin/python tools/gen_mdl_infotheory_figures.py

The script is idempotent: re-running overwrites byte-for-byte (the shared
``svg.hashsalt`` + ``metadata={'Date': None}`` in ``save()`` keep diffs clean).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_mdl_figures as fl  # shared style, palette, helpers, save()

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

# Two extra hues (matplotlib tab10) for the six-curve generator gallery; the
# shared five-color palette is not large enough there.
RED = "#d62728"
PURPLE = "#9467bd"


# --------------------------------------------------------------------------- #
# Local helper: the two-circle entropy "Venn" used by the MI figures.         #
# --------------------------------------------------------------------------- #

def _entropy_venn(ax, *, cx, r, overlap, fill_overlap=True):
    """Draw the two overlapping entropy circles centred on the x-axis.

    ``cx`` is the centre x-offset of each circle from the origin (the circles
    sit at ``(-cx, 0)`` and ``(+cx, 0)``) and ``r`` is their radius; ``overlap``
    is the centre-to-centre lens width used only for the shaded-lens clip.
    Returns the left/right circle centres.
    """
    left_c = np.array([-cx, 0.0])
    right_c = np.array([cx, 0.0])

    # Faint full discs: H(X) (blue) on the left, H(Y) (green) on the right.
    ax.add_patch(Circle(left_c, r, facecolor=BLUE, alpha=0.18, edgecolor=BLUE,
                        lw=1.6, zorder=1))
    ax.add_patch(Circle(right_c, r, facecolor=GREEN, alpha=0.18, edgecolor=GREEN,
                        lw=1.6, zorder=1))

    if fill_overlap:
        # Shade the lens (the set-intersection) by clipping one disc to the
        # other -- this is the mutual-information region.
        clip = Circle(right_c, r, transform=ax.transData)
        lens = ax.add_patch(Circle(left_c, r, facecolor=ORANGE, alpha=0.45,
                                edgecolor="none", zorder=2))
        lens.set_clip_path(clip)

    return left_c, right_c


# =========================================================================== #
# Mutual information as an entropy area-decomposition (§5.1 / §5.3.1).         #
# =========================================================================== #

def fig_mutual_information():
    """The canonical entropy "Venn": two overlapping discs whose union is the
    joint entropy H(X,Y), whose lens is the mutual information I(X;Y), and whose
    crescents are the conditional entropies H(X|Y) and H(Y|X).

    This regenerates the legacy hand-drawn ``mutual-information.svg`` in the
    house style (output id ``mdl-it-mutual-information``)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    r = 1.55
    cx = 0.78
    left_c, right_c = _entropy_venn(ax, cx=cx, r=r, overlap=2 * (r - cx))

    # Outer "circle name" labels, set above each disc.
    fl.vlabel(ax, (left_c[0] - 0.15, r + 0.28), r"entropy $H(X)$", color=BLUE,
            fontsize=11.5)
    fl.vlabel(ax, (right_c[0] + 0.15, r + 0.28), r"entropy $H(Y)$", color=GREEN,
            fontsize=11.5)

    # Left crescent: conditional entropy H(X|Y).
    fl.vlabel(ax, (-cx - 0.62, 0.42), "conditional", color="black", fontsize=10)
    fl.vlabel(ax, (-cx - 0.62, 0.14), "entropy", color="black", fontsize=10)
    fl.vlabel(ax, (-cx - 0.62, -0.40), r"$H(X\mid Y)$", color=BLUE, fontsize=11)

    # Right crescent: conditional entropy H(Y|X).
    fl.vlabel(ax, (cx + 0.62, 0.42), "conditional", color="black", fontsize=10)
    fl.vlabel(ax, (cx + 0.62, 0.14), "entropy", color="black", fontsize=10)
    fl.vlabel(ax, (cx + 0.62, -0.40), r"$H(Y\mid X)$", color=GREEN, fontsize=11)

    # Centre lens: mutual information I(X;Y).
    fl.vlabel(ax, (0.0, 0.42), "mutual", color="black", fontsize=10)
    fl.vlabel(ax, (0.0, 0.14), "information", color="black", fontsize=10)
    fl.vlabel(ax, (0.0, -0.40), r"$I(X;Y)$", color=ORANGE, fontsize=11)

    # Union = joint entropy, labelled below.
    fl.vlabel(ax, (0.0, -r - 0.30), r"joint entropy $H(X,Y)$", color="black",
            fontsize=11.5)

    fl.clean_axes(ax, lim=((-cx - r - 0.55, cx + r + 0.55), (-r - 0.55, r + 0.55)),
                hide=True)
    fl.save(fig, "mdl-it-mutual-information")


def fig_mi_overlap():
    """Extend the entropy Venn with the additive area accounting made explicit:
    the same two discs, but annotated with the three equivalent expressions for
    I(X;Y) and braces marking H(X) and the joint entropy H(X,Y).

    Output id ``mdl-it-mi-overlap`` (the §5.3.1 "MI as overlap" diagram)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    r = 1.55
    cx = 0.78
    left_c, right_c = _entropy_venn(ax, cx=cx, r=r, overlap=2 * (r - cx))

    # Disc names.
    fl.vlabel(ax, (left_c[0] - 0.15, r + 0.26), r"$H(X)$", color=BLUE,
            fontsize=12)
    fl.vlabel(ax, (right_c[0] + 0.15, r + 0.26), r"$H(Y)$", color=GREEN,
            fontsize=12)

    # Region labels (terse: this panel emphasises the identities).
    fl.vlabel(ax, (-cx - 0.58, -0.05), r"$H(X\mid Y)$", color=BLUE, fontsize=11)
    fl.vlabel(ax, (cx + 0.58, -0.05), r"$H(Y\mid X)$", color=GREEN, fontsize=11)
    fl.vlabel(ax, (0.0, 0.30), r"$I(X;Y)$", color=ORANGE, fontsize=11.5)
    fl.vlabel(ax, (0.0, -0.10), "(overlap)", color=GRAY, fontsize=8.5)

    # Brace spanning the union -> joint entropy H(X,Y).
    y_brace = -r - 0.18
    xL, xR = left_c[0] - r, right_c[0] + r
    ax.annotate("", xy=(xL, y_brace), xytext=(xR, y_brace),
                arrowprops=dict(arrowstyle="|-|", color=GRAY, lw=1.1,
                                shrinkA=0, shrinkB=0))
    fl.vlabel(ax, (0.0, y_brace - 0.30), r"joint entropy $H(X,Y)$",
            color="black", fontsize=11)

    # The three equivalent identities, set to the right of the diagram.
    eqs = [
        r"$I(X;Y) = H(X) - H(X\mid Y)$",
        r"$\;\;\;\;\;\;\;\;\;\; = H(Y) - H(Y\mid X)$",
        r"$\;\;\;\;\;\;\;\;\;\; = H(X) + H(Y) - H(X,Y)$",
    ]
    x_eq = cx + r + 0.35
    for i, s in enumerate(eqs):
        ax.text(x_eq, 0.95 - i * 0.62, s, ha="left", va="center", fontsize=10.5)

    fl.clean_axes(ax, equal=True, hide=True)
    ax.set_xlim(-cx - r - 0.55, x_eq + 4.6)
    ax.set_ylim(y_brace - 0.62, r + 0.55)
    fl.save(fig, "mdl-it-mi-overlap")


# =========================================================================== #
# Variational lower bounds on MI (§5.3.4), after Poole et al. (2019).          #
# =========================================================================== #

def fig_mi_variational_bounds():
    """Schematic of the variational MI lower bounds stacked under the true MI,
    annotated with their bias/variance behaviour as the true MI grows.

    The true MI sweeps upward; every estimator is a *lower* bound, so each curve
    sits at or below the diagonal.  We draw:

      * the diagonal -- the (intractable) true I(X;Y);
      * the log N ceiling -- no sample-based bound certifies more than log N
        (the McAllester-Stratos barrier; InfoNCE saturates here);
      * Barber-Agakov / NWJ -- low variance, but they flatten (bias) as MI grows;
      * Donsker-Varadhan (MINE) -- nearly unbiased but its estimate spreads with
        increasing variance (shaded band) at high MI.

    The numbers are illustrative of the qualitative regimes in Poole et al.
    (2019), not a re-estimation.  Output id ``mdl-it-mi-variational-bounds``."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    I = np.linspace(0.0, 10.0, 400)  # true MI (nats), x-axis

    # log N ceiling for a representative batch (N = 128 -> ~4.85 nats).
    logN = np.log(128.0)

    # True MI: the diagonal y = I.
    ax.plot(I, I, color=GRAY, lw=2.0, label=r"true $I(X;Y)$")

    # InfoNCE / log N ceiling: tracks the diagonal then hard-caps at log N.
    nce = np.minimum(I, logN)
    ax.plot(I, nce, color=BLUE, lw=2.2,
            label=r"InfoNCE $(\leq \log N)$")
    ax.axhline(logN, color=BLUE, lw=0.9, ls=":")
    ax.text(I[-1], logN + 0.06, r"$\log N$", color=BLUE, fontsize=9.5,
            ha="right", va="bottom")

    # Barber-Agakov / NWJ: low variance but increasingly biased downward -- a
    # valid lower bound (always <= the diagonal) whose gap widens with MI.
    nwj = np.minimum(6.2 * (1.0 - np.exp(-I / 4.0)), I)
    ax.plot(I, nwj, color=GREEN, lw=2.2, label="NWJ / Barber-Agakov (low var.)")

    # Donsker-Varadhan (MINE): nearly unbiased mean (hugs the diagonal from
    # below) but variance grows with MI -- shown as a widening band that, being a
    # lower-bound estimator, stays under the true-MI diagonal.
    dv_mean = np.minimum(I - 0.06 * I, I)
    dv_sd = 0.05 + 0.10 * I ** 1.6  # variance balloons at high MI
    ax.fill_between(I, np.maximum(dv_mean - dv_sd, 0.0),
                    np.minimum(dv_mean + dv_sd, I),
                    color=ORANGE, alpha=0.20, lw=0)
    ax.plot(I, dv_mean, color=ORANGE, lw=2.2, label="DV / MINE (low bias, high var.)")

    # Annotate the two regimes.
    ax.annotate("bias\n(flattens)", xy=(8.5, nwj[I.searchsorted(8.5)]),
                xytext=(8.3, 3.1), color=GREEN, fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    ax.annotate("variance\n(spreads)",
                xy=(7.5, dv_mean[I.searchsorted(7.5)] - dv_sd[I.searchsorted(7.5)]),
                xytext=(3.3, 1.0), color=ORANGE, fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))

    ax.set_xlabel(r"true mutual information $I(X;Y)$ (nats)")
    ax.set_ylabel("bound estimate (nats)")
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 10.4)
    ax.set_aspect("auto")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fl.save(fig, "mdl-it-mi-variational-bounds")


# =========================================================================== #
# Entropy and self-information (sec_mdl-information_theory).                  #
# =========================================================================== #

def fig_self_info_curve():
    """Self-information I(x) = -log p(x) in nats as a function of p: zero for
    the certain event, ln 2 at the fair coin, divergent as p -> 0.

    Output id ``mdl-it-self-info-curve`` (replaces the hand-drawn orphan)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    p = np.linspace(0.02, 1.0, 600)
    ax.plot(p, -np.log(p), color=BLUE, lw=2.4)

    # The fair coin: p = 1/2 carries ln 2 ~ 0.693 nats.
    ln2 = np.log(2.0)
    ax.plot([0.5, 0.5], [0, ln2], ls="--", color=GRAY, lw=1.1)
    ax.plot([0, 0.5], [ln2, ln2], ls="--", color=GRAY, lw=1.1)
    ax.plot([0.5], [ln2], "o", color=ORANGE, ms=6, zorder=5)
    ax.text(0.52, ln2 + 0.10, r"fair coin: $\ln 2 \approx 0.693$",
            color=ORANGE, fontsize=9.5, ha="left", va="bottom")

    # The certain event: p = 1 carries zero information.
    ax.plot([1.0], [0.0], "o", color=BLUE, ms=6, zorder=5)
    ax.annotate(r"certain: $I = 0$", xy=(1.0, 0.0), xytext=(0.78, 0.55),
                color=BLUE, fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))

    # Rare-events annotation along the steep part of the curve.
    ax.annotate("rare events carry\nmore information",
                xy=(0.06, -np.log(0.06)), xytext=(0.30, 3.0),
                color=BLUE, fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))

    ax.set_xlabel(r"probability $p(x)$")
    ax.set_ylabel(r"$I(x) = -\log p(x)$ (nats)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 4.1)
    ax.set_xticks([0, 0.5, 1.0])
    fl.save(fig, "mdl-it-self-info-curve")


def fig_bernoulli_entropy():
    """Binary entropy H(p) = -p log p - (1-p) log(1-p) in nats: concave,
    symmetric about p = 1/2 where it peaks at ln 2, zero at the deterministic
    endpoints.

    Output id ``mdl-it-bernoulli-entropy`` (replaces the hand-drawn orphan)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    p = np.linspace(1e-4, 1 - 1e-4, 800)
    H = -p * np.log(p) - (1 - p) * np.log(1 - p)
    ax.plot(p, H, color=BLUE, lw=2.4)

    # Peak at the fair coin: H(1/2) = ln 2 ~ 0.693 nats (= 1 bit).
    ln2 = np.log(2.0)
    ax.plot([0.5, 0.5], [0, ln2], ls="--", color=GRAY, lw=1.1)
    ax.plot([0, 0.5], [ln2, ln2], ls="--", color=GRAY, lw=1.1)
    ax.plot([0.5], [ln2], "o", color=ORANGE, ms=6, zorder=5)
    ax.text(0.5, ln2 + 0.045,
            r"maximum uncertainty: $\ln 2 \approx 0.693$ nats",
            color=ORANGE, fontsize=9.5, ha="center", va="bottom")

    # A biased coin barely surprises: H(0.1) ~ 0.325 nats.
    h01 = -0.1 * np.log(0.1) - 0.9 * np.log(0.9)
    ax.plot([0.1], [h01], "o", color=GREEN, ms=5.5, zorder=5)
    ax.annotate(r"$H(0.1) \approx 0.325$", xy=(0.1, h01),
                xytext=(0.16, 0.13), color=GREEN, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))

    # Deterministic endpoints carry no surprise.
    ax.plot([0, 1], [0, 0], "o", color=BLUE, ms=5.5, zorder=5, lw=0)

    ax.set_xlabel(r"heads probability $p$")
    ax.set_ylabel(r"$H(p)$ (nats)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 0.84)
    ax.set_xticks([0, 0.5, 1.0])
    fl.save(fig, "mdl-it-bernoulli-entropy")


# =========================================================================== #
# Divergences and distances (sec_mdl-divergences-distances).                  #
# =========================================================================== #

def _metric_label(ax, pos, text, color):
    """A region entry that is a genuine metric: boxed (see the legend note)."""
    ax.text(pos[0], pos[1], text, color=color, ha="center", va="center",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor=GRAY, lw=0.9))


def fig_divergence_taxonomy():
    """Three-region 'Venn' of the divergence families: f-divergences, integral
    probability metrics, optimal transport.  TV lives in f-div + IPM,
    Wasserstein-1 in IPM + OT; metrics are boxed.

    Output id ``mdl-it-divergence-taxonomy``."""
    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    r = 1.45
    c_f, c_ipm, c_ot = (-1.9, 0.0), (0.0, 0.0), (1.9, 0.0)
    for c, col in [(c_f, BLUE), (c_ipm, ORANGE), (c_ot, GREEN)]:
        ax.add_patch(Circle(c, r, facecolor=col, alpha=0.13, edgecolor=col,
                            lw=1.8, zorder=1))

    # Family names: outer families labelled above, the middle one below.
    fl.vlabel(ax, (c_f[0] - 0.45, r + 0.30), "f-divergences", color=BLUE,
              fontsize=12)
    fl.vlabel(ax, (c_ipm[0], -r - 0.32), "integral probability metrics",
              color=ORANGE, fontsize=12)
    fl.vlabel(ax, (c_ot[0] + 0.45, r + 0.30), "optimal transport",
              color=GREEN, fontsize=12)

    # f-divergence-only members (density-ratio family).
    for y, s in zip((0.82, 0.41, 0.0, -0.41),
                    ("KL", "reverse KL", r"$\chi^2$", "JS")):
        fl.vlabel(ax, (-2.35, y), s, color=BLUE, fontsize=10.5)
    _metric_label(ax, (-2.35, -0.88), r"$H^2$", BLUE)

    # Intersections: TV (f-div + IPM) and W1 (IPM + OT).
    _metric_label(ax, (-0.95, 0.0), "TV", "black")
    _metric_label(ax, (0.95, 0.0), r"$W_1$", "black")

    # IPM-only and OT-only members.
    _metric_label(ax, (0.0, 0.55), "MMD", ORANGE)
    _metric_label(ax, (2.35, 0.0), r"$W_2$", GREEN)

    # Legend note distinguishing the genuine metrics.
    fl.vlabel(ax, (0.0, -r - 0.78),
              r"boxed = a genuine metric (for $H^2$: its square root $H$)",
              color=GRAY, fontsize=9.5)

    fl.clean_axes(ax, lim=((-3.6, 3.6), (-r - 1.05, r + 0.62)), hide=True)
    fl.save(fig, "mdl-it-divergence-taxonomy")


def fig_f_div_generators():
    """The six generators of the f-divergence gallery on one set of axes: all
    convex, all zero at u = 1, very different penalties away from it.

    Output id ``mdl-it-f-div-generators``."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    u = np.linspace(0.05, 3.0, 800)
    curves = [
        ("Kullback–Leibler (forward)", u * np.log(u), BLUE),
        ("reverse KL", -np.log(u), ORANGE),
        (r"Pearson $\chi^2$", (u - 1) ** 2, GREEN),
        (r"squared Hellinger $H^2$", (np.sqrt(u) - 1) ** 2, RED),
        ("total variation", 0.5 * np.abs(u - 1), PURPLE),
        ("Jensen–Shannon",
         u / 2 * np.log(u) - (u + 1) / 2 * np.log((u + 1) / 2), GRAY),
    ]
    for name, y, col in curves:
        ax.plot(u, y, color=col, lw=2.0, label=name)

    # Every generator touches zero at the no-discrepancy ratio u = 1.
    ax.plot([1.0], [0.0], "o", color="black", ms=6, zorder=6)
    ax.annotate(r"$f(1) = 0$", xy=(1.0, 0.0), xytext=(0.62, -0.47),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1))

    ax.axhline(0.0, color=LIGHT, lw=0.9, zorder=0)
    ax.set_xlabel(r"density ratio $u = p/q$")
    ax.set_ylabel(r"generator $f(u)$")
    ax.set_xlim(0.05, 3.0)
    ax.set_ylim(-0.6, 2.5)
    ax.legend(loc="upper center", ncol=2, fontsize=8.5,
              handlelength=1.5, columnspacing=1.0)
    fl.save(fig, "mdl-it-f-div-generators")


def fig_f_gan_tangent_bound():
    """The KL generator f(u) = u log u as the upper envelope of its tangent
    lines: each tangent of slope t has y-intercept -f*(t) = -e^{t-1}, so one
    tangent is the Fenchel--Young bound and the supremum recovers f.

    Output id ``mdl-it-f-gan-tangent-bound``."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    u = np.linspace(0.02, 3.0, 600)
    ax.plot(u, u * np.log(u), color=BLUE, lw=2.6, zorder=5,
            label=r"$f(u) = u\log u$")

    # Tangent at u0: slope t = log(u0) + 1, y-intercept -u0 = -e^{t-1} = -f*(t).
    line = np.linspace(0.0, 3.0, 2)
    for u0 in (0.4, 0.8, 2.3):
        t = np.log(u0) + 1.0
        ax.plot(line, t * line - u0, color=GRAY, lw=1.1, ls="--", zorder=2)

    u0 = 1.5
    t = np.log(u0) + 1.0
    ax.plot(line, t * line - u0, color=ORANGE, lw=2.0, zorder=4,
            label=r"tangent of slope $t$")
    ax.plot([u0], [u0 * np.log(u0)], "o", color=ORANGE, ms=6, zorder=6)
    ax.text(2.52, t * 2.52 - u0 - 0.33, r"slope $t$", color=ORANGE,
            fontsize=10, ha="center", va="center", rotation=25)

    # The y-intercept of the highlighted tangent is the negative conjugate.
    ax.plot([0.0], [-u0], "o", color=ORANGE, ms=6, zorder=6)
    ax.annotate(r"$-f^*(t) = -e^{\,t-1}$", xy=(0.0, -u0),
                xytext=(0.62, -2.15), color=ORANGE, fontsize=10.5,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    ax.text(0.16, 2.55, "$f$ = sup of its tangents\n(Fenchel–Young)",
            color=BLUE, fontsize=10.5, ha="left", va="center")

    ax.axhline(0.0, color=LIGHT, lw=0.9, zorder=0)
    ax.set_xlabel(r"density ratio $u$")
    ax.set_ylabel(r"$f(u)$")
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(-2.5, 3.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.78), fontsize=9)
    fl.save(fig, "mdl-it-f-gan-tangent-bound")


def fig_tv_area():
    """Two overlapping Gaussian densities with the area between them shaded:
    total variation is half that area, and the optimal distinguishing event
    A* = {p > q} is marked on the x-axis.

    Output id ``mdl-it-tv-area``."""
    fig, ax = plt.subplots(figsize=(6.8, 3.9))

    x = np.linspace(-4.6, 4.2, 1200)
    p = np.exp(-(x + 1.0) ** 2 / 2.0) / np.sqrt(2 * np.pi)
    q = np.exp(-(x - 1.2) ** 2 / (2 * 0.8 ** 2)) / (0.8 * np.sqrt(2 * np.pi))

    ax.fill_between(x, p, q, color=ORANGE, alpha=0.30, lw=0)
    ax.plot(x, p, color=BLUE, lw=2.2, label=r"$p$:  $\mathcal{N}(-1,\, 1)$")
    ax.plot(x, q, color=GREEN, lw=2.2,
            label=r"$q$:  $\mathcal{N}(1.2,\, 0.8^2)$")

    # The optimal event A* = {p > q}: on this window, everything left of the
    # crossing (the second crossing, where p's wider tail wins again, is at
    # x ~ 10, far off-plot and carrying negligible mass).
    cross = x[np.nonzero(np.diff(np.sign(p - q)))[0][0]]
    y_bar = -0.028
    ax.plot([x[0], cross], [y_bar, y_bar], color=BLUE, lw=4.0,
            solid_capstyle="butt")
    ax.plot([cross, cross], [0, max(p[np.searchsorted(x, cross)], 0.0)],
            ls=":", color=GRAY, lw=1.1)
    ax.text((x[0] + cross) / 2, y_bar - 0.016, r"$A^\star = \{p > q\}$",
            color=BLUE, fontsize=10.5, ha="center", va="top")

    ax.annotate(r"$\mathrm{TV}(P, Q) = \frac{1}{2} \times$ shaded area",
                xy=(-1.7, 0.55 * np.interp(-1.7, x, p)), xytext=(-4.4, 0.30),
                fontsize=10.5, ha="left",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("density")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.085, 0.56)
    ax.set_yticks([0, 0.2, 0.4])
    ax.legend(loc="upper right", fontsize=9)
    fl.save(fig, "mdl-it-tv-area")


def _nw_monotone_plan(p, q):
    """The monotone (north-west corner) transport plan, optimal in 1-D."""
    p, q = p.astype(float).copy(), q.astype(float).copy()
    plan = np.zeros((len(p), len(q)))
    i = j = 0
    while i < len(p) and j < len(q):
        m = min(p[i], q[j])
        plan[i, j] = m
        p[i] -= m
        q[j] -= m
        if p[i] <= 1e-12:
            i += 1
        if q[j] <= 1e-12:
            j += 1
    return plan


def fig_ot_transport_plan():
    """Left (two stacked panels): the chapter's six-atom example -- mirrored
    bar charts with the monotone transport plan's arrows, and the two step
    CDFs with the area between them shaded (W1 = 1.7).  Right: point masses at
    separation d -- JS is frozen at log 2 while W1 = d supplies a gradient.

    Output id ``mdl-it-ot-transport-plan``."""
    fig = plt.figure(figsize=(9.6, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0],
                          height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.30)
    axp = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[:, 1])

    atoms = np.arange(6.0)
    p_w = np.array([0.30, 0.20, 0.25, 0.10, 0.10, 0.05])
    q_w = np.array([0.05, 0.10, 0.10, 0.25, 0.20, 0.30])

    # --- (a) mirrored bars + the monotone plan's arrows --------------------
    axp.bar(atoms, p_w, width=0.5, color=BLUE, alpha=0.75, label=r"$P$")
    axp.bar(atoms, -q_w, width=0.5, color=GREEN, alpha=0.75, label=r"$Q$")
    axp.axhline(0.0, color=GRAY, lw=0.9)
    plan = _nw_monotone_plan(p_w, q_w)
    for i in range(6):
        for j in range(6):
            if plan[i, j] > 1e-12 and i != j:
                axp.add_patch(FancyArrowPatch(
                    (atoms[i], 0.012), (atoms[j], -0.012),
                    connectionstyle="arc3,rad=-0.25", arrowstyle="->",
                    mutation_scale=9, color=GRAY, alpha=0.85,
                    lw=0.8 + 9.0 * plan[i, j], zorder=4))
    axp.text(2.0, 0.345, r"plan $\gamma$: move mass $\gamma_{ij}$"
             r" a distance $|x_i - x_j|$", fontsize=9, ha="center", color=GRAY)
    axp.set_ylim(-0.40, 0.46)
    axp.set_yticks([-0.3, 0, 0.3])
    axp.set_yticklabels(["0.3", "0", "0.3"])
    axp.set_xticks(atoms)
    axp.set_ylabel("mass")
    axp.legend(loc="upper right", fontsize=8.5)

    # --- (b) the two step CDFs; area between them = W1 = 1.7 ---------------
    grid = np.concatenate([atoms, [5.6]])
    F_p = np.concatenate([np.cumsum(p_w), [1.0]])
    F_q = np.concatenate([np.cumsum(q_w), [1.0]])
    axc.step(grid, F_p, where="post", color=BLUE, lw=2.0, label=r"$F_P$")
    axc.step(grid, F_q, where="post", color=GREEN, lw=2.0, label=r"$F_Q$")
    axc.fill_between(grid, F_p, F_q, step="post", color=ORANGE, alpha=0.30,
                     lw=0)
    axc.text(2.45, 0.62, r"area $= W_1 = 1.7$", color=ORANGE, fontsize=10.5,
             ha="center")
    axc.set_xticks(atoms)
    axc.set_xlabel(r"$x$")
    axc.set_ylabel("CDF")
    axc.set_ylim(0, 1.06)
    axc.legend(loc="upper left", fontsize=8.5)

    # --- (c) point masses at separation d: JS is blind, W1 sees ------------
    d = np.linspace(0.0, 3.0, 2)
    ln2 = np.log(2.0)
    axr.plot(d, d, color=GREEN, lw=2.2, label=r"$W_1(\delta_0, \delta_d) = d$")
    axr.plot([0.0, 3.0], [ln2, ln2], color=ORANGE, lw=2.2,
             label=r"$\mathrm{JS}(\delta_0, \delta_d)$")
    axr.plot([0.0], [ln2], "o", color=ORANGE, mfc="white", ms=6, zorder=5)
    axr.plot([0.0], [0.0], "o", color=ORANGE, ms=6, zorder=5)
    axr.text(1.5, ln2 + 0.10, r"frozen at $\log 2$: no gradient",
             color=ORANGE, fontsize=9.5, ha="center")
    axr.text(1.78, 2.02, r"gradient $\pm 1$", color=GREEN, fontsize=9.5,
             ha="center", va="center", rotation=42)
    axr.set_xlabel(r"separation $d$")
    axr.set_ylabel("divergence (nats)")
    axr.set_xlim(0, 3.0)
    axr.set_ylim(0, 3.1)
    axr.legend(loc="upper left", fontsize=8.5)
    fl.save(fig, "mdl-it-ot-transport-plan")


def fig_score_field():
    """The score field of a two-component 2-D Gaussian mixture: density
    contours underlaid, arrows showing the gradient of the log-density --
    springs toward each mean, with the valley handing points from one
    attractor to the other.  (Chapter 27's stubs reuse this figure.)

    Output id ``mdl-it-score-field``."""
    fig, ax = plt.subplots(figsize=(6.2, 4.8))

    w = np.array([0.7, 0.3])
    mus = np.array([[-1.5, -1.0], [1.5, 1.0]])

    def comps(X, Y):
        """Component densities w_k N(mu_k, I) on a grid."""
        out = []
        for wk, mk in zip(w, mus):
            out.append(wk * np.exp(-((X - mk[0]) ** 2 + (Y - mk[1]) ** 2) / 2)
                       / (2 * np.pi))
        return np.stack(out)

    # Density contours on a fine grid.
    xs = np.linspace(-4.4, 4.4, 240)
    ys = np.linspace(-3.6, 3.6, 220)
    Xf, Yf = np.meshgrid(xs, ys)
    ax.contourf(Xf, Yf, comps(Xf, Yf).sum(axis=0), levels=11, cmap="Blues")

    # Score s(x) = sum_k r_k(x) * (mu_k - x) on a coarse grid (arrows uphill).
    xc = np.linspace(-4.0, 4.0, 13)
    yc = np.linspace(-3.2, 3.2, 11)
    Xc, Yc = np.meshgrid(xc, yc)
    pk = comps(Xc, Yc)
    r = pk / pk.sum(axis=0)
    U = (r * (mus[:, 0, None, None] - Xc)).sum(axis=0)
    V = (r * (mus[:, 1, None, None] - Yc)).sum(axis=0)
    ax.quiver(Xc, Yc, U, V, color=ORANGE, angles="xy", scale_units="xy",
              scale=3.4, width=0.0042, zorder=4)

    # The two means, the attractors of the field.
    for mk, lab, dx in zip(mus, (r"$\mu_1$", r"$\mu_2$"), (-0.36, 0.36)):
        ax.plot(*mk, "o", color="black", mec="white", mew=1.2, ms=7, zorder=6)
        ax.text(mk[0] + dx, mk[1] + 0.28, lab, fontsize=11, ha="center",
                zorder=6, bbox=dict(boxstyle="round,pad=0.15",
                                    facecolor="white", edgecolor="none",
                                    alpha=0.75))

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    fl.clean_axes(ax, lim=((-4.4, 4.4), (-3.6, 3.6)), hide=False)
    fl.save(fig, "mdl-it-score-field")


# =========================================================================== #
# Mutual information: InfoNCE and the information bottleneck                  #
# (sec_mdl-mutual-information).                                               #
# =========================================================================== #

def fig_infonce_pos_neg():
    """The InfoNCE game.  Left: an anchor x, its positive y+ and N-1 negatives
    in a 2-D embedding sketch, with critic scores f(x, y_j) as rays.  Right:
    the softmax over the N scores, positive bar accented, annotated with the
    bound I >= log N - L_NCE (itself at most log N).

    Output id ``mdl-it-infonce-pos-neg``."""
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(9.2, 3.9), gridspec_kw=dict(width_ratios=[1.2, 1.0]))

    rng = np.random.default_rng(7)
    N = 8

    # --- (a) embedding sketch: anchor, positive, negatives, critic rays ----
    anchor = np.array([-2.1, 0.0])
    pos = np.array([0.7, 0.55])
    neg = np.column_stack([rng.uniform(0.0, 2.4, N - 1),
                           rng.uniform(-1.55, 1.55, N - 1)])
    # Keep the negatives clear of the positive for a readable sketch.
    neg += 0.34 * (neg - pos) / np.maximum(
        np.linalg.norm(neg - pos, axis=1, keepdims=True), 1e-9)

    for yj in neg:
        axl.plot([anchor[0], yj[0]], [anchor[1], yj[1]], color=LIGHT, lw=1.0,
                 zorder=1)
    axl.plot([anchor[0], pos[0]], [anchor[1], pos[1]], color=ORANGE, lw=2.0,
             zorder=2)

    axl.plot(*anchor, "*", color=BLUE, ms=17, zorder=5)
    axl.plot(*pos, "o", color=ORANGE, ms=9, zorder=5)
    axl.plot(pos[0], pos[1], "o", ms=15, mfc="none", mec=ORANGE, mew=1.4,
             zorder=5)
    axl.plot(neg[:, 0], neg[:, 1], "o", color=GRAY, ms=7, zorder=4)

    axl.text(anchor[0], anchor[1] - 0.34, r"anchor $x$", color=BLUE,
             fontsize=10.5, ha="center", va="top")
    axl.text(pos[0] + 0.10, pos[1] + 0.30, r"positive $y^+ \sim p(y \mid x)$",
             color=ORANGE, fontsize=10, ha="center", va="bottom")
    axl.text(1.55, -1.62, r"negatives $y_2, \ldots, y_N \sim p(y)$",
             color=GRAY, fontsize=10, ha="center", va="top")
    # Label a generic critic ray on one of the negatives.
    low = neg[np.argmin(neg[:, 1])]            # the bottom-most negative
    mid = 0.42 * anchor + 0.58 * low
    axl.text(mid[0] - 0.12, mid[1] - 0.26, r"scores $f(x, y_j)$", color=GRAY,
             fontsize=10, ha="center", va="top")
    fl.clean_axes(axl, lim=((-3.0, 3.0), (-2.3, 2.1)), hide=True)

    # --- (b) softmax over the N critic scores ------------------------------
    scores = rng.normal(0.0, 1.0, N)
    scores[0] += 2.6                       # the critic likes the positive
    soft = np.exp(scores - scores.max())
    soft /= soft.sum()
    colors = [ORANGE] + [GRAY] * (N - 1)
    axr.bar(np.arange(N), soft, color=colors, alpha=0.85)
    axr.set_xticks(np.arange(N))
    axr.set_xticklabels([r"$y^+$"] + [rf"$y_{j}$" for j in range(2, N + 1)],
                        fontsize=9)
    axr.set_ylabel("softmax probability")
    axr.set_ylim(0, 1.0)
    axr.text(N / 2 - 0.5, 0.92,
             r"$I(X;Y) \geq \log N - \mathcal{L}_{\mathrm{NCE}}^{(\mathrm{ce})}"
             r" \;(\leq \log N)$",
             fontsize=10.5, ha="center", va="center")
    fl.save(fig, "mdl-it-infonce-pos-neg")


def fig_ib_tradeoff():
    """The information plane of the closed-form Gaussian bottleneck (rho = 0.9):
    the concave frontier of attainable (I(X;Z), I(Y;Z)) pairs, the infeasible
    region above it, the DPI ceiling I(X;Y), and the operating points the
    Lagrange multiplier beta selects -- including total collapse for
    beta <= 1/rho^2.  Matches the chapter's ``mutual-information-ib-plane``
    cell exactly.

    Output id ``mdl-it-ib-tradeoff``."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    rho = 0.9
    sigma = np.logspace(-2, 2, 4001)[::-1]   # decreasing noise -> I_xz rises
    I_xz = 0.5 * np.log1p(sigma ** -2)
    I_yz = -0.5 * np.log(1 - rho ** 2 / (1 + sigma ** 2))
    I_xy = -0.5 * np.log(1 - rho ** 2)       # ~ 0.830 nats

    # Infeasible region: above the frontier (and the frontier's DPI ceiling).
    ax.fill_between(I_xz, I_yz, 1.02, color=LIGHT, alpha=0.45, lw=0)
    ax.text(1.55, 0.945, "infeasible", color=GRAY, fontsize=10.5, ha="center")

    ax.plot(I_xz, I_yz, color=BLUE, lw=2.4, label="IB frontier", zorder=4)
    ax.axhline(I_xy, color=GRAY, lw=1.2, ls="--", zorder=3)
    ax.text(2.95, I_xy + 0.022,
            r"DPI ceiling $I(X;Y) = -\frac{1}{2}\log(1-\rho^2) \approx 0.830$",
            color=GRAY, fontsize=9.5, ha="right", va="bottom")

    # Operating points selected by the multiplier (same grid rule as the
    # notebook cell): argmin of the Lagrangian I(X;Z) - beta I(Y;Z).
    for beta in (1.0, 1.5, 2.0, 4.0, 16.0):
        i = np.argmin(I_xz - beta * I_yz)
        ax.plot(I_xz[i], I_yz[i], "o", color=ORANGE, ms=6.5, zorder=6)
        if I_xz[i] > 1e-3:
            ax.annotate(rf"$\beta = {beta:g}$", (I_xz[i], I_yz[i]),
                        textcoords="offset points", xytext=(9, -7),
                        color=ORANGE, fontsize=9)
    ax.annotate("collapse for $\\beta \\leq 1/\\rho^2 \\approx 1.23$:\n"
                "$I(X;Z) = I(Y;Z) = 0$",
                xy=(0.0, 0.0), xytext=(0.55, 0.16), color=ORANGE, fontsize=9.5,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    ax.set_xlabel(r"$I(X;Z)$ (nats)")
    ax.set_ylabel(r"$I(Y;Z)$ (nats)")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", fontsize=9)
    fl.save(fig, "mdl-it-ib-tradeoff")


FIGURES = [
    # sec_mdl-information_theory
    fig_self_info_curve,         # mdl-it-self-info-curve
    fig_bernoulli_entropy,       # mdl-it-bernoulli-entropy
    # sec_mdl-mutual-information
    fig_mutual_information,      # mdl-it-mutual-information (legacy regenerated)
    fig_mi_overlap,              # mdl-it-mi-overlap
    fig_mi_variational_bounds,   # mdl-it-mi-variational-bounds
    fig_infonce_pos_neg,         # mdl-it-infonce-pos-neg
    fig_ib_tradeoff,             # mdl-it-ib-tradeoff
    # sec_mdl-divergences-distances
    fig_divergence_taxonomy,     # mdl-it-divergence-taxonomy
    fig_f_div_generators,        # mdl-it-f-div-generators
    fig_f_gan_tangent_bound,     # mdl-it-f-gan-tangent-bound
    fig_tv_area,                 # mdl-it-tv-area
    fig_ot_transport_plan,       # mdl-it-ot-transport-plan
    fig_score_field,             # mdl-it-score-field
]


def main():
    # Reset the shared module's WRITTEN list so verification reports only ours.
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()

    print(f"\nWrote {len(fl.WRITTEN)} figures to {fl.IMG_DIR}:")
    for p in fl.WRITTEN:
        size = os.path.getsize(p)
        assert os.path.exists(p), f"missing: {p}"
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            head = fh.read(400)
        assert "<svg" in head, f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):34s} {size:>8,d} bytes")

    print(f"\nAll {len(fl.WRITTEN)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
