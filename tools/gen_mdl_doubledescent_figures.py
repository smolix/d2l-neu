#!/usr/bin/env python3
"""Generate the double-descent figure for the "Multilayer Perceptrons ->
Generalization in Deep Learning" section (``chapter_multilayer-perceptrons/
generalization-deep.md``) in the one shared house style.

The section is a prose-only conceptual essay; double descent is the single idea
in it that is *picture-first*, so this is the one figure it earns.  The prose
references the generated SVG with no drawing code (like the slide SVGs).

The curve is schematic but *honest*: rather than placing points by hand, the
test error is built from the standard bias/variance decomposition that Belkin et
al. (2019) use to explain the shape.  Bias falls monotonically with capacity; a
classical variance term rises, giving the U on the left; and a near-threshold
variance term spikes where the model has *just barely* enough capacity to
interpolate (a single, high-variance, minimum-norm interpolant is forced) and
then decays through the over-parametrized regime, producing the second descent.
Adding the two regimes gives the non-monotone, two-valley double-descent curve.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` + ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette and the
drawing helpers.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_doubledescent_figures.py

The figure is written to ``img/mdl-mlp-double-descent.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


def fig_double_descent():
    """Test error vs. model capacity: the classical U on the left, a sharp spike
    at the interpolation threshold (capacity = 1, where #params ~ #examples),
    and a second monotone descent into the over-parametrized regime, settling
    below the classical sweet spot.  Training error falls to 0 at the threshold
    and stays there.

    The two error curves are *computed* from a bias/variance decomposition so the
    shape is principled, not freehand:
        test(c) = noise_floor
                + bias(c)                      # falls with capacity (less underfit)
                + var_classical(c)             # rises on the left (classical U)
                + var_interp(c)                # spikes at c=1, decays for c>1
    The classical minimum sits left of the threshold; the over-parametrized tail
    is set to drop just below it, which is the empirically observed ordering.
    """
    thr = 1.0  # interpolation threshold: capacity where #params ~ #examples
    c = np.linspace(0.04, 3.2, 1600)

    # --- training error: monotone decay to 0 at the threshold, then flat at 0 ---
    # smooth approach to zero capacity-by-capacity, clamped to 0 past threshold.
    train = 0.92 * np.exp(-3.1 * c)
    train = np.where(c >= thr, 0.0, train)
    # make it land cleanly at ~0 right at the threshold (no visible step)
    train = np.clip(train - 0.92 * np.exp(-3.1 * thr) * (c / thr) ** 6, 0.0, None)

    # --- test error: bias + classical variance + interpolation-peak variance ---
    noise = 0.045                                   # irreducible (Bayes) error
    bias = 0.62 * np.exp(-2.3 * c)                  # underfitting, falls with capacity
    var_classical = 0.085 * c                       # classical variance, rises with capacity
    # near-threshold variance spike: large where the model can *just* interpolate,
    # decaying on both sides.  A narrow Lorentzian centred at the threshold.
    gamma = 0.085
    var_interp = 0.40 * gamma ** 2 / ((c - thr) ** 2 + gamma ** 2)
    # over-parametrized regime: extra capacity gives the optimizer room to pick a
    # smooth, small-norm interpolant -> a slow further decay below the classical U.
    over = np.where(c > thr, 0.085 * (1.0 - np.exp(-1.6 * (c - thr))), 0.0)

    test = noise + bias + var_classical + var_interp - over
    test = np.clip(test, noise * 0.6, None)

    # locate the classical minimum (left of the threshold) for annotation/leveling
    left = c < thr - 3 * gamma
    i_cmin = np.where(left)[0][np.argmin(test[left])]
    c_classical, e_classical = c[i_cmin], test[i_cmin]
    e_over_tail = test[-1]
    print(f"  classical min  capacity={c_classical:.2f}  test={e_classical:.3f}")
    print(f"  threshold peak  test={test[np.argmin(np.abs(c - thr))]:.3f}")
    print(f"  over-param tail test={e_over_tail:.3f}  "
          f"(below classical min: {e_over_tail < e_classical})")

    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    # regime backdrop: faint shading left vs. right of the threshold
    ax.axvspan(c[0], thr, color=LIGHT, alpha=0.18, lw=0)

    # the interpolation threshold marker
    ax.axvline(thr, color=GRAY, ls="--", lw=1.2, zorder=2)

    # curves
    ax.plot(c, train, color=GRAY, lw=2.2, zorder=3)
    ax.plot(c, test, color=BLUE, lw=2.6, zorder=4)

    ymax = 0.96
    ax.set_ylim(0, ymax)
    ax.set_xlim(c[0], c[-1])

    # the classical sweet spot, and a faint guide line carrying its level
    # rightward so the eye can see the over-parametrized tail dip below it.
    ax.plot([c_classical], [e_classical], "o", color=BLUE, ms=5, zorder=6)
    ax.plot([c_classical, c[-1]], [e_classical, e_classical], ":", color=GRAY,
            lw=1.0, alpha=0.6, zorder=1)
    ax.annotate("classical\nsweet spot", xy=(c_classical, e_classical),
                xytext=(c_classical - 0.34, e_classical - 0.10),
                ha="center", va="top", fontsize=11, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0,
                                connectionstyle="arc3,rad=0.3"))

    # curve labels, placed on uncluttered stretches of each curve.  Wrapped to
    # two short lines (rather than one wide line) so each label's horizontal
    # footprint stays in its own column and never runs into its neighbours.
    xt = 0.62                   # on the flat shelf just left of the sweet spot,
    ax.text(xt, test[np.argmin(np.abs(c - xt))] + 0.11, "test error",
            color=BLUE, fontsize=11, ha="center", va="bottom")
    xg = 0.08                                   # training label high on the gray arm,
    # kept narrow (2 lines) and hugging the left edge, so it stays clear of
    # the "interpolation threshold" / "classical regime" text further right
    ax.text(xg, train[np.argmin(np.abs(c - xg))] + 0.05, "training\nerror",
            color=GRAY, fontsize=11, ha="left", va="bottom")

    # threshold label: right-aligned just LEFT of the dashed marker line (not
    # centred on it) and pushed up near the top spine, clear of the spike
    # (whose peak reaches only ~0.6) and of the regime labels below it
    ax.text(thr - 0.06, ymax * 0.97, "interpolation\nthreshold", color="black",
            fontsize=11, ha="right", va="top")

    # regime labels: each sits in the empty wedge on its side of the threshold,
    # high enough to clear both curves (the low-left corner stays free for the
    # "sweet spot" callout) and low enough to clear the threshold label above.
    # Generic annotations (not tied to a curve's color), so black for contrast.
    ax.text(0.74, ymax * 0.66, "classical\nregime", color="black",
            fontsize=11, ha="center", va="center")
    ax.text((thr + c[-1]) / 2 + 0.20, ymax * 0.66, "over-parametrized\nregime",
            color="black", fontsize=11, ha="center", va="center")

    # axes: schematic, so no numeric ticks; just the conceptual labels
    ax.set_xlabel("model capacity  ($\\#$parameters)", fontsize=11)
    ax.set_ylabel("error", fontsize=11)
    ax.set_xticks([thr])
    ax.set_xticklabels([r"$\#$params $\approx\ \#$examples"], fontsize=11)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fl.save(fig, "mdl-mlp-double-descent")


FIGURES = [fig_double_descent]


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks the Linear Algebra figures, which we don't run here).
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
