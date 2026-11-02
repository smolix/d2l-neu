#!/usr/bin/env python3
"""Generate the illustrative figures for chapter_transformers/moe.md (B6)
in the shared house style defined in ``gen_mdl_figures.py``.

Figures:
  mdl-transformers-moe.svg -- one token's pass through a mixture-of-experts
      layer: a linear router scores all experts, the top-k (here two)
      experts compute, and their outputs are added, each weighted by its
      router probability; the other experts are stored but do no work for
      this token.
  mdl-transformers-moe-fine.svg -- three ways to spend the same expert
      budget (after DeepSeekMoE): 16 experts with top-2; the same
      parameters as 64 quarter-width experts with top-8; one of them made
      a shared expert that every token uses, plus top-7 of 63.  Stored and
      active parameters are equal in all three rows; the number of expert
      combinations a token can choose from is not.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_transformers_b6.py

Output is written to ``img/``.  Byte-idempotent: no timestamps, no unseeded
randomness.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
T = fl.T
BLUE, GREEN, INK = fl.BLUE, fl.GREEN, fl.INK

from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

E = 8                      # experts in the layer figure
SEL = (2, 5)               # the two selected experts (0-based)
PROBS = [0.05, 0.04, 0.24, 0.10, 0.07, 0.31, 0.12, 0.07]  # router output


def _box(ax, cx, cy, w, h, fc, ec=INK, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        fc=fc, ec=ec, lw=lw, linestyle=ls))


def _expert(ax, cx, cy, w, h, selected, label=None):
    """A selected expert is a green FFN block; an idle one is ghosted."""
    if selected:
        _box(ax, cx, cy, w, h, T.GREEN.tint, ec=T.GREEN.base, lw=2.0)
    else:
        _box(ax, cx, cy, w, h, T.PANEL, ec=T.FAINT, lw=1.0, ls=(0, (3, 2)))
    if label:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=13,
                color=INK)


def fig_moe():
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    xs = [0.9 + i * 1.17 for i in range(E)]      # expert centers
    y_tok, y_rt, y_exp, y_sum, y_out = 0.45, 1.85, 4.05, 5.55, 6.35
    x_mid = (xs[0] + xs[-1]) / 2

    # token and router
    _box(ax, x_mid, y_tok, 1.0, 0.62, "white")
    ax.text(x_mid, y_tok, r"$\mathbf{x}$", ha="center", va="center",
            fontsize=15, color=INK)
    ax.text(x_mid + 0.85, y_tok, "token", ha="left", va="center",
            fontsize=13, color=INK)
    _box(ax, x_mid, y_rt, 1.7, 0.62, T.BLUE.tint, ec=BLUE)
    ax.text(x_mid, y_rt, "router", ha="center", va="center", fontsize=14,
            color=INK)
    ax.text(x_mid + 1.15, y_rt,
            r"$\mathbf{p} = \mathrm{softmax}(\mathbf{W}_r \mathbf{x})$",
            ha="left", va="center", fontsize=14, color=INK)
    fl.arrow(ax, (x_mid, y_tok + 0.36), (x_mid, y_rt - 0.36), color=INK,
             lw=1.4, mut=11)

    # experts and the routing fan: heavy green to the top-2, faint elsewhere
    for i, (x, p) in enumerate(zip(xs, PROBS)):
        sel = i in SEL
        _expert(ax, x, y_exp, 0.92, 0.92, sel,
                rf"$\mathrm{{FFN}}_{{{i + 1}}}$")
        if sel:
            fl.arrow(ax, (x_mid, y_rt + 0.36), (x, y_exp - 0.50),
                     color=GREEN, lw=2.2, mut=13)
        else:
            ax.plot([x_mid, x], [y_rt + 0.36, y_exp - 0.52], color=T.FAINT,
                    lw=1.0, alpha=0.25 + 1.6 * p, zorder=1)

    # outputs of the selected experts, weighted and added
    r = 0.24
    ax.add_patch(Circle((x_mid, y_sum), r, fc="white", ec=INK, lw=1.4))
    ax.text(x_mid, y_sum, "+", ha="center", va="center", fontsize=16,
            color=INK)
    for i, side in zip(SEL, (-1, 1)):
        tail = (xs[i], y_exp + 0.50)
        d = np.array([x_mid - xs[i], y_sum - tail[1]])
        tip = np.array([x_mid, y_sum]) - r * d / np.linalg.norm(d)
        fl.arrow(ax, tail, tuple(tip), color=GREEN, lw=2.2, mut=13)
        mid = (0.5 * (tail[0] + tip[0]), 0.5 * (tail[1] + tip[1]))
        ax.text(mid[0] + side * 0.32, mid[1] - 0.12,
                rf"$\times\, p_{{{i + 1}}} = {PROBS[i]:.2f}$",
                ha="left" if side > 0 else "right", va="center",
                fontsize=14, color=INK)
    fl.arrow(ax, (x_mid, y_sum + r), (x_mid, y_out - 0.05), color=INK,
             lw=1.4, mut=11)
    ax.text(x_mid, y_out + 0.02,
            rf"$\mathbf{{y}} = p_{{{SEL[0] + 1}}}\,"
            rf"\mathrm{{FFN}}_{{{SEL[0] + 1}}}(\mathbf{{x}}) + "
            rf"p_{{{SEL[1] + 1}}}\,\mathrm{{FFN}}_{{{SEL[1] + 1}}}"
            rf"(\mathbf{{x}})$",
            ha="center", va="bottom", fontsize=15, color=INK)

    # key, in the empty lower-left corner
    kx, ky = 0.35, 1.2
    _expert(ax, kx + 0.2, ky, 0.4, 0.4, True)
    ax.text(kx + 0.55, ky, "computed (top-2)", ha="left", va="center",
            fontsize=13, color=INK)
    _expert(ax, kx + 0.2, ky - 0.7, 0.4, 0.4, False)
    ax.text(kx + 0.55, ky - 0.7, "stored, idle", ha="left", va="center",
            fontsize=13, color=INK)

    ax.set_xlim(-0.1, xs[-1] + 1.0)
    ax.set_ylim(-0.1, 6.95)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-transformers-moe")


def _row(ax, y, n, chosen, width, h, shared=False):
    """n equal experts side by side over [0, width], the indices in
    `chosen` selected (fixed, irregular positions: routing has no pattern);
    with shared=True one more expert sits apart on the left, drawn solid:
    the shared expert every token uses."""
    gap = 0.05 if n > 20 else 0.14
    x0 = 0.0
    if shared:
        w = (width - 0.35 - n * gap) / (n + 1)
        ax.add_patch(Rectangle((0, y), w, h, fc=T.GREEN.base,
                               ec=T.GREEN.dark, lw=1.2))
        x0 = w + 0.35
    else:
        w = (width - (n - 1) * gap) / n
    for i in range(n):
        x = x0 + i * (w + gap)
        if i in chosen:
            ax.add_patch(Rectangle((x, y), w, h, fc=T.GREEN.tint,
                                   ec=T.GREEN.base, lw=1.6))
        else:
            ax.add_patch(Rectangle((x, y), w, h, fc=T.PANEL, ec=T.FAINT,
                                   lw=0.8, ls=(0, (3, 2))))


def fig_moe_fine():
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    width, h = 12.0, 0.75
    rows = [  # (y, routed experts, selected indices, shared, label)
        (3.95, 16, (3, 12), False,
         r"16 experts, top-2:   $\binom{16}{2} = 120$ choices"),
        (2.05, 64, (2, 9, 11, 24, 33, 39, 50, 61), False,
         r"64 experts of a quarter width, top-8:   "
         r"$\binom{64}{8} \approx 4.4 \times 10^{9}$ choices"),
        (0.15, 63, (6, 14, 15, 28, 40, 47, 57), True,
         r"1 shared expert (solid, every token) + 63 routed, top-7:   "
         r"$\binom{63}{7} \approx 5.5 \times 10^{8}$ choices"),
    ]
    for y, n, chosen, shared, label in rows:
        _row(ax, y, n, set(chosen), width, h, shared)
        ax.text(0, y + h + 0.14, label, ha="left", va="bottom", fontsize=13,
                color=INK)
    ax.set_xlim(-0.1, width + 0.1)
    ax.set_ylim(0.05, 5.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "mdl-transformers-moe-fine")


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
    fig_moe()
    fig_moe_fine()
    for p in fl.WRITTEN:
        _unitless(p)
    for p in fl.WRITTEN:
        print("wrote", p)
