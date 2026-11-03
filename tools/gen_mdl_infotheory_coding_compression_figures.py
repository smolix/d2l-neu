#!/usr/bin/env python3
"""Generate the new illustrative figure of "Codes, Compression, and Perplexity"
(``chapter_mdl-information-theory/mdl-coding-compression.md``) in the chapter's
shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` and ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette.  The
section's Kraft tree (``mdl-it-kraft-tree``) stays in
``gen_mdl_infotheory_figures.py``.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_infotheory_coding_compression_figures.py

Writes ``img/mdl-it-arith-intervals.svg`` (and the copy in ``img/auto/`` that
the slides read).
"""

from __future__ import annotations

import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers
from matplotlib.colors import to_rgba

np, plt = fl.np, fl.plt
BLUE, ORANGE, GRAY, LIGHT, INK = fl.BLUE, fl.ORANGE, fl.GRAY, fl.LIGHT, fl.INK
save = fl.save


def _box(ax, x0, y, w, h, color, alpha, lw=1.4, z=2):
    """Bar with a light wash of ``color`` and a full-strength outline."""
    ax.add_patch(plt.Rectangle((x0, y - h / 2), w, h,
                               facecolor=to_rgba(color, alpha),
                               edgecolor=color, lw=lw, zorder=z))


def _narrow(msg, qa):
    """Interval after each symbol of ``msg`` under the i.i.d. model q(a)=qa."""
    lo, hi, out = 0.0, 1.0, [(0.0, 1.0)]
    for ch in msg:
        w = hi - lo
        if ch == "a":
            hi = lo + w * qa
        else:
            lo = lo + w * qa
        out.append((lo, hi))
    return out


def fig_arith_intervals():
    """Arithmetic coding of ``aaaaaaaaba`` under q(a) = 0.9.

    (a) [0, 1) narrowed by the first three symbols, then the final interval
    [0.3874, 0.4262) of width 0.9^9 * 0.1 = 0.0387 after all ten.
    (b) Zoom: neither 2^-5 cell [0.375, 0.40625), [0.40625, 0.4375) fits
    inside the final interval; the 2^-6 cell [0.390625, 0.40625) does, so the
    codeword is its address 011001.  Every number is computed, not typed."""
    msg, qa = "aaaaaaaaba", 0.9
    steps = _narrow(msg, qa)
    lo, hi = steps[-1]
    L = math.ceil(-math.log2(hi - lo)) + 1          # 6 bits
    m = math.ceil(lo * 2 ** L)                       # 25 -> 011001
    c_lo, c_hi = m / 2 ** L, (m + 1) / 2 ** L
    code = format(m, f"0{L}b")
    assert lo <= c_lo and c_hi <= hi and code == "011001"

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.6, 3.5), gridspec_kw=dict(width_ratios=[1.05, 1.0]))
    FS, BH = 14, 0.46                                 # font size, bar height

    # ---------------- (a) narrowing ---------------------------------------
    rows = [("start", steps[0]), ("after a", steps[1]),
            ("after aa", steps[2]), ("after aaa", steps[3])]
    ys = [4.0, 3.0, 2.0, 1.0]
    for (name, (a0, a1)), y in zip(rows, ys):
        split = a0 + qa * (a1 - a0)
        _box(axa, a0, y, split - a0, BH, BLUE, 0.22)
        _box(axa, split, y, a1 - split, BH, ORANGE, 0.30)
        axa.text((a0 + split) / 2, y, "a", color=INK, fontsize=FS,
                 ha="center", va="center", family="monospace")
        axa.text((split + a1) / 2, y, "b", color=INK, fontsize=FS,
                 ha="center", va="center", family="monospace")
        axa.text(-0.03, y, name, color=INK, fontsize=FS - 1, ha="right",
                 va="center")
        if y > 1.0:                                   # guide to the next bar
            for xg in (a0, split):
                axa.plot([xg, xg], [y - BH / 2, y - 1 + BH / 2], color=GRAY,
                         lw=0.9, ls=(0, (3, 2)))
    axa.text(0.36, 0.42, r"$\vdots$", color=INK, fontsize=FS, ha="center",
             va="center")
    # final row: the whole unit interval with the final sliver
    yf = -0.35
    axa.add_patch(plt.Rectangle((0, yf - BH / 2), 1.0, BH, facecolor="white",
                                edgecolor=GRAY, lw=1.0))
    axa.add_patch(plt.Rectangle((lo, yf - BH / 2), hi - lo, BH,
                                facecolor=BLUE, edgecolor=BLUE, lw=1.0))
    axa.text(-0.03, yf, "after aaaaaaaaba", color=INK, fontsize=FS - 1,
             ha="right", va="center")
    axa.annotate("", xy=(hi + 0.005, yf + BH / 2 + 0.06),
                 xytext=(0.56, yf + BH / 2 + 0.36),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    axa.text(0.57, yf + BH / 2 + 0.36, r"$w = 0.9^9\cdot 0.1 = 0.0387$",
             color=INK, fontsize=FS, ha="left", va="center")
    for xt in (0.0, 0.5, 1.0):
        axa.plot([xt, xt], [yf - BH / 2, yf - BH / 2 - 0.08], color=INK,
                 lw=0.9)
        axa.text(xt, yf - BH / 2 - 0.14, f"{xt:g}", color=INK, fontsize=FS - 1,
                 ha="center", va="top")
    axa.set_xlim(-0.02, 1.02)
    axa.set_ylim(yf - BH / 2 - 0.5, 4.0 + BH / 2 + 0.05)
    axa.axis("off")

    # ---------------- (b) naming the interval ------------------------------
    y_int, y6, y5 = 0.0, 1.0, 2.0
    X0 = 0.3405                                       # left margin for labels
    # the final interval; dashed guides at its ends run behind the opaque
    # cells, so they show in the gaps and never cross a label
    _box(axb, lo, y_int, hi - lo, BH, BLUE, 0.22)
    for xg in (lo, hi):
        axb.plot([xg, xg], [y_int + BH / 2, y5 + BH / 2 + 0.12], color=BLUE,
                 lw=1.0, ls=(0, (4, 3)), zorder=1)
    for xv in (lo, hi):
        axb.text(xv, y_int - BH / 2 - 0.10, f"{xv:.4f}", color=INK,
                 fontsize=FS - 1, ha="center", va="top")
    axb.text((lo + hi) / 2, y_int, "final interval", color=INK,
             fontsize=FS - 1, ha="center", va="center", zorder=3)
    # the 2^-5 cells that meet the interval: both stick out of it
    w5 = 2.0 ** -5
    for k5 in range(int(lo // w5), int(math.ceil(hi / w5))):
        axb.add_patch(plt.Rectangle((k5 * w5, y5 - BH / 2), w5, BH,
                                    facecolor="#E9ECEF", edgecolor=GRAY,
                                    lw=1.4, zorder=2))
        axb.text(k5 * w5 + w5 / 2, y5, format(k5, "05b"), color=INK,
                 fontsize=FS - 1, ha="center", va="center",
                 family="monospace", zorder=3)
    axb.text(X0, y5, "length $2^{-5}$:\nneither fits", color=INK,
             fontsize=FS - 1, ha="left", va="center")
    # the 2^-6 cell that fits
    axb.add_patch(plt.Rectangle((c_lo, y6 - BH / 2), c_hi - c_lo, BH,
                                facecolor="#FBE3CF", edgecolor=ORANGE,
                                lw=1.4, zorder=2))
    axb.text((c_lo + c_hi) / 2, y6, code, color=INK, fontsize=FS - 1,
             ha="center", va="center", family="monospace", zorder=3)
    axb.text(X0, y6, "length $2^{-6}$:\nfits", color=INK, fontsize=FS - 1,
             ha="left", va="center")
    axb.set_xlim(X0 - 0.001, 0.4395)
    axb.set_ylim(y_int - BH / 2 - 0.5, y5 + BH / 2 + 0.2)
    axb.axis("off")

    fig.tight_layout(w_pad=2.5)
    save(fig, "mdl-it-arith-intervals")
    shutil.copyfile(os.path.join(fl.IMG_DIR, "mdl-it-arith-intervals.svg"),
                    os.path.join(fl.IMG_DIR, "auto",
                                 "mdl-it-arith-intervals.svg"))


FIGURES = [fig_arith_intervals]


def main():
    fl.WRITTEN.clear()
    for fn in FIGURES:
        fn()
    for path in fl.WRITTEN:
        print(f"wrote {path} ({os.path.getsize(path):,d} bytes)")


if __name__ == "__main__":
    main()
