#!/usr/bin/env python3
"""Generate the new illustrative figure of "Stochastic Gradient Methods"
(``chapter_mdl-optimization/mdl-stochastic-gradient-methods.md``) in the
chapter's shared house style.

* ``mdl-opt-budget-tradeoff`` -- the two-term bound D^2/(2 eta K) + eta G^2/2
  of SGD on a convex loss against the step eta, for D = G = 1 and the budgets
  K = 10^2 and 10^4, with its two components dashed and the minima
  (eta* = 1/sqrt(K), bound 1/sqrt(K)) joined by their locus of slope one.

The section's other two figures (``mdl-opt-sgd-noise-ball`` and
``mdl-opt-schedule-zoo``) stay in ``gen_mdl_optimization_figures.py``.  The
shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (the fixed ``svg.hashsalt`` and ``metadata={'Date': None}``
in ``save()`` make re-runs byte-for-byte identical).  Run with the repo's
pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_optimization_stochastic_gradient_methods_figures.py

Writes ``img/mdl-opt-budget-tradeoff.svg`` (and the copy in ``img/auto/`` that
the slides read).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY

LBL = 13          # the chapter's in-figure label size
WHITE = dict(facecolor="white", edgecolor="none", pad=1.5)


def _copy_to_auto(name):
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def fig_budget_tradeoff():
    """bound(eta) = 1/(2 eta K) + eta/2 for K = 1e2 (blue) and 1e4 (orange):
    the transient falls with eta, the floor rises with it, and the minimum
    1/sqrt(K) sits at eta* = 1/sqrt(K) (green dots on a dotted line of slope
    one).  Log-log axes; every curve is computed from its formula."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    eta = np.logspace(-4, 0, 400)
    budgets = ((1e2, BLUE, r"$K=10^2$"), (1e4, ORANGE, r"$K=10^4$"))

    # the shared floor eta G^2 / 2 and the two transients D^2 / (2 eta K)
    ax.plot(eta, eta / 2, color=GRAY, lw=1.3, ls=(0, (5, 3)), zorder=2)
    for K, col, _ in budgets:
        ax.plot(eta, 1 / (2 * eta * K), color=col, lw=1.2, ls=(0, (5, 3)),
                alpha=0.8, zorder=2)
        ax.plot(eta, 1 / (2 * eta * K) + eta / 2, color=col, lw=2.2, zorder=3)

    # the minima and their locus eta* = 1/sqrt(K), bound = 1/sqrt(K)
    locus = np.array([5e-3, 0.15])
    ax.plot(locus, locus, color=GRAY, lw=1.0, ls=(0, (1, 2)), zorder=2)
    for K, _, _ in budgets:
        ax.plot(1 / np.sqrt(K), 1 / np.sqrt(K), "o", color=GREEN, ms=7,
                mec="white", mew=1.0, zorder=5)

    # labels, each in the empty region on the outer side of what it names
    ax.text(3e-3, 3.2, r"$K=10^2$", color=BLUE, fontsize=LBL, ha="left",
            va="bottom")
    ax.text(6e-4, 0.12, r"$K=10^4$", color=ORANGE, fontsize=LBL, ha="left",
            va="bottom")
    # left of the rising floor line, below both falling curves
    ax.text(3.3e-3, 2.6e-3, r"floor $\eta G^2/2$", color="black",
            fontsize=LBL, ha="right", va="center")
    # left of the dotted locus, between the two bound curves
    ax.text(0.027, 0.055, r"$\eta^\star=D/(G\sqrt{K})$", color="black",
            fontsize=LBL, ha="right", va="center")
    # lower right, under every line, with leaders to both dashed transients
    ax.text(0.05, 2.0e-3, r"transient $D^2/(2\eta K)$", color="black",
            fontsize=LBL, ha="left", va="center")
    for tip in ((0.028, 1 / (2 * 0.028 * 1e4)), (0.45, 1 / (2 * 0.45 * 1e2))):
        ax.annotate("", xy=tip, xytext=(0.06 if tip[0] < 0.1 else 0.3,
                                         2.5e-3 if tip[0] < 0.1 else 2.6e-3),
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(1e-3, 50.0)
    ax.set_xlabel(r"step $\eta$", color="black")
    ax.set_ylabel("bound", color="black")
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
    ax.tick_params(colors="black", which="both")
    fl.save(fig, "mdl-opt-budget-tradeoff")
    _copy_to_auto("mdl-opt-budget-tradeoff")


FIGURES = [
    fig_budget_tradeoff,
]


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):36s} {size:>8,d} bytes")


if __name__ == "__main__":
    main()
