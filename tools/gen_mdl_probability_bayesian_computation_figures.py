#!/usr/bin/env python3
"""Generate the new illustrative figures of "Bayesian Computation"
(``chapter_mdl-probability-statistics/mdl-bayesian-computation.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (``mdl-prob-bayes-importance``,
``mdl-prob-bayes-metropolis``, ``mdl-prob-bayes-laplace``,
``mdl-prob-bayes-kl-modes``) stay in ``gen_mdl_probability_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_probability_bayesian_computation_figures.py

Writes ``img/mdl-prob-bayes-two-state.svg`` and
``img/mdl-prob-bayes-mean-field.svg`` (and the copies in ``img/auto/`` that
the slides read).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers
from figstyle import tokens as T  # noqa: E402

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, INK = fl.BLUE, fl.ORANGE, fl.GREEN, fl.INK
MUTED = T.MUTED

from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402


def _save_both(fig, name):
    """Write img/<name>.svg and the img/auto copy the slides read."""
    fl.save(fig, name)
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def fig_two_state():
    """The two-state Metropolis chain with target pi = (1/4, 3/4) and a
    proposal that always names the other state.  From state 1 the ratio is 3,
    so the chain always moves; from state 2 it moves with probability 1/3.
    The flow 1 -> 2 is (1/4) * 1 and the flow 2 -> 1 is (3/4) * (1/3): both
    are 1/4, which is detailed balance.  The rejected moves from state 2 are
    the self-loop, carrying (3/4) * (2/3) = 1/2."""
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    r = 0.62
    c1, c2 = np.array([0.0, 0.0]), np.array([3.0, 0.0])
    for c, top, lab in ((c1, "state 1", r"$\pi=1/4$"),
                        (c2, "state 2", r"$\pi=3/4$")):
        ax.add_patch(Circle(c, r, facecolor=T.BLUE.tint, edgecolor=INK,
                            lw=1.6, zorder=3))
        ax.text(c[0], c[1] + 0.13, top, ha="center", va="center",
                fontsize=14, color=INK, zorder=4)
        ax.text(c[0], c[1] - 0.2, lab, ha="center", va="center",
                fontsize=14, color=INK, zorder=4)
    # the two flows: upper arc 1 -> 2, lower arc 2 -> 1, both blue
    for start, end, rad, lab, ty in (
            ((0.45, 0.42), (2.55, 0.42), -0.45, r"$\frac{1}{4}\cdot 1=\frac{1}{4}$",
             1.32),
            ((2.55, -0.42), (0.45, -0.42), -0.45,
             r"$\frac{3}{4}\cdot\frac{1}{3}=\frac{1}{4}$", -1.38)):
        ax.add_patch(FancyArrowPatch(
            start, end, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=18, lw=2.0, color=BLUE,
            zorder=2))
        ax.text(1.5, ty, lab, ha="center", va="center", fontsize=14,
                color=INK)
    # the self-loop on state 2: the rejected proposals, in muted gray
    ax.add_patch(FancyArrowPatch(
        (3.45, 0.42), (3.45, -0.42), connectionstyle="arc3,rad=-1.6",
        arrowstyle="-|>", mutation_scale=16, lw=1.6, color=MUTED, zorder=2))
    ax.text(4.42, 0.0, "stay:\n" + r"$\frac{3}{4}\cdot\frac{2}{3}=\frac{1}{2}$",
            ha="left", va="center", fontsize=14, color=MUTED,
            linespacing=1.4)
    ax.set_xlim(-0.8, 5.7)
    ax.set_ylim(-1.75, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    _save_both(fig, "mdl-prob-bayes-two-state")


def _ellipse(mean, cov, k=2.0, n=400):
    """Points of the k-sd ellipse of a 2-D Gaussian."""
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])
    return mean[:, None] + vecs @ (k * np.sqrt(vals)[:, None] * circle)


def fig_mean_field():
    """Mean-field Gaussians fitted to a correlated Gaussian posterior with
    sds (1, 0.5) and correlation rho = -0.85, all drawn as 2-sd ellipses.
    Reverse KL (what variational inference minimizes) gives the axis-aligned
    Gaussian with the conditional sds, (1, 0.5) * sqrt(1 - rho^2) =
    (0.53, 0.26), which meets the posterior ellipse where it crosses the
    coordinate axes.  Forward KL (moment matching)
    gives the axis-aligned Gaussian with the marginal sds (1, 0.5), which
    covers corners the posterior does not have."""
    sd = np.array([1.0, 0.5])
    rho = -0.85
    cov = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]],
                    [rho * sd[0] * sd[1], sd[1] ** 2]])
    mean = np.zeros(2)
    rev = np.diag((sd * np.sqrt(1 - rho ** 2)) ** 2)
    fwd = np.diag(sd ** 2)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    P = _ellipse(mean, cov)
    R = _ellipse(mean, rev)
    F = _ellipse(mean, fwd)
    ax.plot(F[0], F[1], color=GREEN, lw=2.0, ls="-.", zorder=3)
    ax.plot(P[0], P[1], color=BLUE, lw=2.6, zorder=4)
    ax.plot(R[0], R[1], color=ORANGE, lw=2.2, ls="--", zorder=5)
    ax.plot([0], [0], "o", color=INK, ms=4, zorder=6)
    # coordinate axes in black through the common mean
    ax.axhline(0, color="black", lw=0.9, zorder=1)
    ax.axvline(0, color="black", lw=0.9, zorder=1)
    # labels offset outward, each in the empty region next to its curve
    ax.text(-2.25, 1.27, r"posterior, $\rho=-0.85$", color=BLUE,
            fontsize=13, ha="left", va="center")
    ax.annotate("mean-field, reverse KL:\n" + r"sd $\times\sqrt{1-\rho^2}$",
                xy=(0.675, 0.405), xytext=(1.05, 1.22), color=ORANGE,
                fontsize=13, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1,
                                shrinkA=2, shrinkB=2))
    ax.annotate("diagonal, forward KL:\nmarginal sd",
                xy=(1.98, -0.15), xytext=(2.2, -0.5), color=GREEN,
                fontsize=13, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1,
                                shrinkA=2, shrinkB=2))
    # axis names at the ends of the black coordinate axes
    ax.text(4.05, 0.07, r"$\theta_0$", fontsize=14, ha="right", va="bottom")
    ax.text(0.1, 1.5, r"$\theta_1$", fontsize=14, ha="left", va="top")
    ax.set_xlim(-2.3, 4.1)
    ax.set_ylim(-1.2, 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    _save_both(fig, "mdl-prob-bayes-mean-field")


FIGURES = [fig_two_state, fig_mean_field]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
