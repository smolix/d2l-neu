#!/usr/bin/env python3
"""Generate the illustrative figure for chapter_transformers/scaling-laws.md
(B7) in the shared house style defined in ``gen_mdl_figures.py``.

Figures:
  mdl-transformers-chinchilla.svg -- contours of the loss
      L(N, D) = E + A/N^alpha + B/D^beta, with the refit constants of
      Besiroglu et al. (2024), over parameters N and training tokens D on
      log-log axes.  Dashed lines are budgets of equal compute C = 6ND; on
      each, the compute-optimal model is where a loss contour touches the
      line, and these points form the compute-optimal path.  Chinchilla
      (70B parameters, 1.4T tokens) lies on the path; Llama 3 8B (15T
      tokens) lies far above it on a budget about a quarter larger.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_transformers_b7.py

Output is written to ``img/`` and copied to ``img/auto/`` (slides).  Byte-idempotent: no timestamps, no
randomness.
"""

from __future__ import annotations

import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
T = fl.T
BLUE, ORANGE, INK = fl.BLUE, fl.ORANGE, fl.INK

# Refit of the Chinchilla law (Besiroglu et al., 2024), rounded as in the text
E, A, B, ALPHA, BETA = 1.82, 482.0, 2085.4, 0.348, 0.366
G = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
XLIM, YLIM = (8.0, 12.0), (9.0, 14.5)          # log10 N, log10 D


def _loss(logN, logD):
    return E + A / 10 ** (ALPHA * logN) + B / 10 ** (BETA * logD)


def _optimum(C):
    N = G * (C / 6) ** (BETA / (ALPHA + BETA))
    return np.log10(N), np.log10(C / (6 * N))


def _angle(ax, slope):
    """Screen angle in degrees of a data-space line of the given slope."""
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0, slope))
    return np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))


def fig_chinchilla() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    x = np.linspace(*XLIM, 400)
    y = np.linspace(*YLIM, 400)
    X, Y = np.meshgrid(x, y)
    levels = [2.0, 2.1, 2.2, 2.4, 2.7]
    ax.contour(X, Y, _loss(X, Y), levels=levels, colors=T.BLUE.base,
                    linewidths=1.3)
    # Label each contour near the right edge, where it runs almost flat
    xls = [11.25, 11.55, 11.55, 11.55, 11.55]
    spots = [(xl, np.log10((B / (L - E - A / 10 ** (ALPHA * xl)))
                           ** (1 / BETA))) for L, xl in zip(levels, xls)]
    for (lx, ly), L in zip(spots, levels):
        ax.text(lx, ly, f"{L:.1f}", fontsize=12, color=T.BLUE.dark,
                ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1.0), zorder=4)

    # Budgets of equal compute, log D = log(C/6) - log N, dashed
    rot = _angle(ax, -1.0)
    C_chin = 5.76e23  # Chinchilla's budget as reported (6ND gives 5.9e23)
    for C, label, lx in ((1e21, r"$C = 10^{21}$", 8.15),
                         (C_chin, r"$C = 5.76 \times 10^{23}$", 8.62)):
        ax.plot(x, np.log10(C / 6) - x, color=T.MUTED, lw=1.1,
                ls=(0, (5, 3)))
        n0, d0 = _optimum(C)
        ax.plot(n0, d0, "o", ms=6, color=T.ORANGE.base, zorder=4)
        ax.text(lx, np.log10(C / 6) - lx + 0.16, label, fontsize=12,
                color=INK, ha="left", va="bottom", rotation=rot,
                rotation_mode="anchor",
                bbox=dict(fc="white", ec="none", pad=0.5))

    # Compute-optimal path through the tangency points
    Cs = np.logspace(19.5, 26.8, 50)
    path = np.array([_optimum(C) for C in Cs])
    ax.plot(path[:, 0], path[:, 1], color=T.ORANGE.base, lw=2.0, zorder=3)
    ax.text(11.0, 12.42, "compute-optimal", fontsize=12,
            color=T.ORANGE.dark, ha="left", va="bottom",
            rotation=_angle(ax, ALPHA / BETA), rotation_mode="anchor")

    # Two models whose budgets differ by about a quarter
    for logN, logD, name, dx, dy, ha in (
            (np.log10(70e9), np.log10(1.4e12), "Chinchilla", 0.12, 0.0,
             "left"),
            (np.log10(8e9), np.log10(15e12), "Llama 3 8B", -0.12, 0.0,
             "right")):
        ax.plot(logN, logD, "s", ms=6, color=INK, zorder=5)
        ax.text(logN + dx, logD + dy, name, fontsize=12, color=INK,
                ha=ha, va="center",
                bbox=dict(fc="white", ec="none", pad=1.0), zorder=6)

    ax.set_xticks(range(8, 13))
    ax.set_xticklabels([rf"$10^{{{k}}}$" for k in range(8, 13)])
    ax.set_yticks(range(9, 15))
    ax.set_yticklabels([rf"$10^{{{k}}}$" for k in range(9, 15)])
    ax.set_xlabel("parameters $N$", fontsize=13, color=INK)
    ax.set_ylabel("training tokens $D$", fontsize=13, color=INK)
    ax.tick_params(colors=INK, labelsize=12)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fl.save(fig, "mdl-transformers-chinchilla")


def _unitless(path):
    """Rewrite matplotlib's pt-sized root width/height as unitless CSS px
    (1 pt = 4/3 px), keeping the viewBox (figure guide: unitless sizes)."""
    with open(path) as fh:
        svg = fh.read()
    head, rest = svg.split("<svg", 1)
    tag, body = rest.split(">", 1)
    tag = re.sub(r'(width|height)="([0-9.]+)pt"',
                 lambda m: f'{m.group(1)}="{float(m.group(2)) * 4 / 3:.2f}"',
                 tag)
    with open(path, "w") as fh:
        fh.write(head + "<svg" + tag + ">" + body)


if __name__ == "__main__":
    fig_chinchilla()
    for p in fl.WRITTEN:
        _unitless(p)
        auto = os.path.join(os.path.dirname(p), "auto", os.path.basename(p))
        shutil.copyfile(p, auto)  # The slides read figures from img/auto
        print("wrote", p, "and", auto)
