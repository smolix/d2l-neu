#!/usr/bin/env python3
"""Generate the new illustrative figure of "Entropy, Cross-Entropy, and KL
Divergence" (``chapter_mdl-information-theory/mdl-information-theory.md``) in
the chapter's shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's older figures (``mdl-it-self-info-curve``,
``mdl-it-bernoulli-entropy``, ``mdl-it-code-length-bars``) stay in
``gen_mdl_infotheory_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_infotheory_entropy_figures.py

Writes ``img/mdl-it-kl-integrand.svg`` (and the copy in ``img/auto/`` that the
slides read).
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


def _normal_pdf(x, sigma):
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)


def fig_kl_integrand():
    """Where KL(N(0,1) || N(0,4)) = 0.318 and KL(N(0,4) || N(0,1)) = 0.807 come
    from: the integrand p log(p/q) (a) and q log(q/p) (b).  Both change sign
    at |x| = sqrt(8 ln 2 / 3) = 1.36.  In (a) a positive core (+0.424) and
    two slightly negative tails (-0.053 each); in (b) a negative core (-0.240)
    and two heavy positive tails (+0.523 each)."""
    x = np.linspace(-6.5, 6.5, 2601)
    p, q = _normal_pdf(x, 1.0), _normal_pdf(x, 2.0)
    x0 = np.sqrt(8 * np.log(2) / 3)          # sign change of log(p/q)
    panels = [
        (p * np.log(p / q), r"(a) $p\,\log(p/q)$",
         r"$D_{\mathrm{KL}}(P\,\Vert\,Q) = 0.318$",
         [(0.0, 0.085, r"$+0.42$"), (-3.3, -0.085, r"$-0.05$"),
          (3.3, -0.085, r"$-0.05$")]),
        (q * np.log(q / p), r"(b) $q\,\log(q/p)$",
         r"$D_{\mathrm{KL}}(Q\,\Vert\,P) = 0.807$",
         [(0.0, -0.185, r"$-0.24$"), (-3.15, 0.06, r"$+0.52$"),
          (3.15, 0.06, r"$+0.52$")]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), sharey=True)
    for ax, (f, title, total, labels) in zip(axes, panels):
        pos, neg = f >= 0, f < 0
        # Positive contributions in a blue wash, negative ones in orange.
        ax.fill_between(x, 0, f, where=pos, color=fl.BLUE, alpha=0.18, lw=0,
                        interpolate=True)
        ax.fill_between(x, 0, f, where=neg, color=fl.ORANGE, alpha=0.25,
                        lw=0, interpolate=True)
        ax.plot(x, f, color=fl.BLUE, lw=2.2)
        # The zero line is the coordinate axis: black.
        ax.axhline(0.0, color="black", lw=0.9, zorder=1)
        for s in (-x0, x0):                   # short ticks at the crossings
            ax.plot([s, s], [-0.012, 0.012], color="black", lw=1.4)
        for lx, ly, txt in labels:
            ax.text(lx, ly, txt, color=INK, fontsize=14, ha="center",
                    va="center")
        ax.text(0.0, 0.335, total, color=INK, fontsize=14, ha="center",
                va="center")
        ax.set_title(title, fontsize=14)
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-0.25, 0.37)
        ax.set_xticks([-6, -3, 0, 3, 6])
        ax.set_xlabel(r"$x$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("integrand (nats)")
    axes[0].set_yticks([-0.2, 0.0, 0.2])
    fig.tight_layout()
    save(fig, "mdl-it-kl-integrand")
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-it-kl-integrand.svg"),
                    os.path.join(fl.IMG_DIR, "auto", "mdl-it-kl-integrand.svg"))


FIGURES = [fig_kl_integrand]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
