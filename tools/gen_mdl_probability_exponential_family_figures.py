#!/usr/bin/env python3
"""Generate the new illustrative figures of "The Exponential Family"
(``chapter_mdl-probability-statistics/mdl-exponential-family.md``) in the
chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's Beta-posterior figure (``mdl-prob-beta-posterior``) stays in
``gen_mdl_probability_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_probability_exponential_family_figures.py

Writes ``img/mdl-prob-maxent-die.svg`` and ``img/mdl-prob-log-partition.svg``
(and the copies in ``img/auto/`` that the slides read).
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
INK = fl.INK
save = fl.save


def _copy_to_auto(name):
    """The slides inline ``@fig:<id>`` from ``img/auto/<id>.svg``."""
    shutil.copyfile(os.path.join(fl.IMG_DIR, f"{name}.svg"),
                    os.path.join(fl.IMG_DIR, "auto", f"{name}.svg"))


def _tilt(eta, k):
    w = np.exp(eta * k)
    return w / w.sum()


def _maxent_tilt(mean, k):
    """Bisection on eta for the tilt p_k ~ exp(eta k) with the given mean
    (the mean increases with eta).  Deterministic: no randomness."""
    lo, hi = -5.0, 5.0
    for _ in range(60):
        eta = (lo + hi) / 2
        if (_tilt(eta, k) * k).sum() < mean:
            lo = eta
        else:
            hi = eta
    return eta, _tilt(eta, k)


def fig_maxent_die():
    """Maximum-entropy distributions of a die whose average roll is fixed at
    3.5, 4.5 and 5.5: the tilts p_k ~ exp(eta k) with eta = 0, 0.371, 1.087
    and entropies H = 1.79, 1.61, 0.95 nats."""
    k = np.arange(1, 7)
    means = [3.5, 4.5, 5.5]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), sharey=True)
    for ax, m in zip(axes, means):
        eta, p = _maxent_tilt(m, k)
        eta = 0.0 if abs(eta) < 1e-9 else eta
        H = -(p * np.log(p)).sum()
        ax.bar(k, p, width=0.7, color=fl.BLUE, alpha=0.85, lw=0)
        # The average roll, marked on the k axis.
        ax.plot([m], [0.0], marker="^", ms=10, color="black", clip_on=False,
                zorder=5)
        ax.set_title(rf"mean ${m}$,  $\eta = {eta:.2f}$,  $H = {H:.2f}$",
                     fontsize=14)
        ax.set_xlim(0.4, 6.6)
        ax.set_ylim(0, 0.72)
        ax.set_xticks(k)
        ax.tick_params(labelsize=13, colors="black")
        ax.set_xlabel(r"$k$", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
    axes[0].set_ylabel(r"$p_k$", fontsize=14)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6])
    fig.tight_layout()
    save(fig, "mdl-prob-maxent-die")
    _copy_to_auto("mdl-prob-maxent-die")


def fig_log_partition():
    """Left: the Bernoulli log-partition A(eta) = log(1 + e^eta) with its
    tangent at eta_hat = log(9/4); the slope is 9/13.  Right: the negative
    log-likelihoods 13 A(eta) - 9 eta (minimum 8.02 at eta_hat) and
    13 A(eta) - 13 eta (13 heads of 13: decreasing, no minimizer)."""
    A = lambda e: np.logaddexp(0.0, e)
    sig = lambda e: 1 / (1 + np.exp(-e))
    eta_hat = np.log(9 / 4)

    left = (-3.0, 4.0)
    right = (-2.0, 6.0)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.0, 3.8),
        gridspec_kw={"width_ratios": [left[1] - left[0], right[1] - right[0]]})

    # (a) A(eta), its asymptotes and the tangent at eta_hat.
    e = np.linspace(*left, 400)
    ax1.plot(e, np.zeros_like(e), color=fl.GRAY, lw=1.0, ls="--")
    ea = np.linspace(-0.3, left[1], 100)
    ax1.plot(ea, ea, color=fl.GRAY, lw=1.0, ls="--")
    ax1.plot(e, A(e), color=fl.BLUE, lw=2.4)
    et = np.linspace(-1.6, 2.6, 50)
    ax1.plot(et, A(eta_hat) + sig(eta_hat) * (et - eta_hat), color=fl.ORANGE,
             lw=2.0)
    ax1.plot([eta_hat], [A(eta_hat)], "o", color="black", ms=6, zorder=5)
    ax1.text(-2.85, 3.0, r"slope $A'(\hat\eta) = 9/13$", color=INK,
             fontsize=14, ha="left", va="center")
    ax1.text(3.7, 0.35, r"$A = 0$", color=INK, fontsize=13, ha="right",
             va="bottom")
    ax1.text(3.0, 2.3, r"$A = \eta$", color=INK, fontsize=13, ha="left",
             va="center")
    ax1.set_xlim(*left)
    ax1.set_ylim(-0.3, 4.3)
    ax1.set_xticks([-2, 0, eta_hat, 2, 4])
    ax1.set_xticklabels(["$-2$", "$0$", r"$\hat\eta$", "$2$", "$4$"])
    ax1.set_xlabel(r"$\eta$", fontsize=14)
    ax1.set_ylabel(r"$A(\eta)$", fontsize=14)

    # (b) The two negative log-likelihoods.
    e = np.linspace(*right, 400)
    ax2.plot(e, 13 * A(e) - 9 * e, color=fl.BLUE, lw=2.4)
    ax2.plot(e, 13 * A(e) - 13 * e, color=fl.ORANGE, lw=2.4)
    ax2.plot([eta_hat], [13 * A(eta_hat) - 9 * eta_hat], "o", color="black",
             ms=6, zorder=5)
    ax2.text(3.0, 18.5, "9 of 13", color=INK, fontsize=14, ha="right",
             va="center")
    ax2.text(2.6, 4.2, "13 of 13: no minimum", color=INK, fontsize=14,
             ha="left", va="center")
    ax2.set_xlim(*right)
    ax2.set_ylim(0, 28)
    ax2.set_xticks([-2, 0, eta_hat, 2, 4, 6])
    ax2.set_xticklabels(["$-2$", "$0$", r"$\hat\eta$", "$2$", "$4$", "$6$"])
    ax2.set_xlabel(r"$\eta$", fontsize=14)
    ax2.set_ylabel("NLL", fontsize=14)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=13, colors="black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
    fig.tight_layout()
    save(fig, "mdl-prob-log-partition")
    _copy_to_auto("mdl-prob-log-partition")


FIGURES = [fig_maxent_die, fig_log_partition]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
