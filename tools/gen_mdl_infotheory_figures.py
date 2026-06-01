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


FIGURES = [
    fig_mutual_information,       # mdl-it-mutual-information (legacy regenerated)
    fig_mi_overlap,              # mdl-it-mi-overlap
    fig_mi_variational_bounds,   # mdl-it-mi-variational-bounds
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
