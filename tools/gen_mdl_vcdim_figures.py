#!/usr/bin/env python3
"""Generate the illustrative figure(s) for the VC-dimension / shattering
discussion in the "Generalization in Classification" section
(``chapter_linear-classification/generalization-classification.md``) in the one
shared house style defined in ``gen_mdl_figures.py``.

The prose references the generated files with no drawing code (like the slide
SVGs).  The separating lines are *computed* (a hard-margin SVM via a tiny QP-free
support-vector search over point pairs), so the picture is exact, not sketched:
every drawn line genuinely separates the +/- points it is shown with.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_vcdim_figures.py

All figures are written to ``img/mdl-clf-<id>.svg``.
"""

from __future__ import annotations

import os
import sys
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


# --------------------------------------------------------------------------- #
# Geometry helper: a separating line for a +/- labelling, if one exists.       #
# --------------------------------------------------------------------------- #

def _separating_line(P, labels):
    """Return (w, b) of a max-margin separating line w.x + b = 0 for points P
    (rows) with +/-1 ``labels``, or ``None`` if the labelling is not linearly
    separable.

    Max-margin in 2-D with so few points: the optimal separator is governed by
    2 or 3 support points.  We search all candidate normals defined by pairs of
    opposite-labelled points (the boundary is perpendicular to the closest such
    pair) and all 3-point circumscribings, keep separators that classify every
    point correctly, and return the one with the largest geometric margin."""
    P = np.asarray(P, float)
    y = np.asarray(labels, float)
    pos, neg = P[y > 0], P[y < 0]
    if len(pos) == 0 or len(neg) == 0:
        # all one class: a line just outside the cloud, on the side that keeps
        # every point in its (single) half-space.  Normal points "up"; offset so
        # the whole triangle sits on the + side when y>0 (line below it), and on
        # the - side when y<0 (line above it).
        w = np.array([0.0, 1.0])
        s = P @ w
        b = -(s.min() - 0.45) if y[0] > 0 else -(s.max() + 0.45)
        return w, b

    best = None
    best_margin = -1.0
    # Candidate normals: every +/- pair direction, plus a dense angular sweep
    # (the sweep guarantees we find *a* separator whenever one exists; the pair
    # directions make the found one max-margin for the 2-support case).
    dirs = []
    for p in pos:
        for q in neg:
            d = p - q
            n = np.linalg.norm(d)
            if n > 1e-9:
                dirs.append(d / n)
    dirs += [np.array([np.cos(a), np.sin(a)])
             for a in np.linspace(0, np.pi, 360, endpoint=False)]

    for w in dirs:
        s = P @ w
        sp, sn = s[y > 0], s[y < 0]
        # need a gap: max projection of one class < min of the other
        if sp.min() - sn.max() > 1e-9:        # + on the high side
            margin = sp.min() - sn.max()
            b = -0.5 * (sp.min() + sn.max())
            if margin > best_margin:
                best_margin, best = margin, (w.copy(), b)
        elif sn.min() - sp.max() > 1e-9:      # - on the high side
            margin = sn.min() - sp.max()
            b = -0.5 * (sn.min() + sp.max())
            if margin > best_margin:
                best_margin, best = margin, (-w.copy(), -b)
    return best


def _draw_points(ax, P, labels):
    """Plot +/- points: + filled blue circles, - hollow orange squares."""
    P = np.asarray(P, float)
    y = np.asarray(labels)
    for (x, yy), lab in zip(P, y):
        if lab > 0:
            ax.plot(x, yy, "o", color=BLUE, ms=9, zorder=5)
        else:
            ax.plot(x, yy, "s", mfc="white", mec=ORANGE, mew=2.0, ms=9, zorder=5)


def _draw_line(ax, wb, lim, color=GRAY, lw=1.8):
    """Draw the line w.x + b = 0, clipped to the rectangle ``lim`` =
    ((x0,x1),(y0,y1)) so it never bleeds past its own panel."""
    (x0, x1), (y0, y1) = lim
    w, b = wb
    # Intersect the line with the four box edges; keep the points on the border.
    pts = []
    if abs(w[1]) > 1e-9:
        for x in (x0, x1):
            yv = -(b + w[0] * x) / w[1]
            if y0 - 1e-9 <= yv <= y1 + 1e-9:
                pts.append((x, yv))
    if abs(w[0]) > 1e-9:
        for yv in (y0, y1):
            x = -(b + w[1] * yv) / w[0]
            if x0 - 1e-9 <= x <= x1 + 1e-9:
                pts.append((x, yv))
    if len(pts) < 2:
        return
    pts = np.array(pts)
    # take the two extreme intersection points (handles the corner-dupe case)
    i, j = np.argmin(pts[:, 0] + pts[:, 1]), np.argmax(pts[:, 0] + pts[:, 1])
    seg = pts[[i, j]]
    ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, zorder=3,
            solid_capstyle="round")


def fig_shattering():
    """Two panels in the shared house style.

    Left block: all 2^3 = 8 labellings of three points in general position, each
    with a *computed* max-margin separating line, demonstrating that a line in
    the plane shatters 3 points.  Right panel: the XOR labelling of four points,
    for which no separating line exists (we verify none does)."""
    # three points in general position (a triangle, well inside a small box)
    P3 = np.array([[-0.78, -0.55], [0.78, -0.55], [0.0, 0.82]])
    box = ((-1.35, 1.35), (-1.25, 1.35))

    fig = plt.figure(figsize=(11.0, 4.4))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.42, 1.6],
                          wspace=0.12, hspace=0.22)

    # --- left: 2x4 grid of the eight labellings of three points ---
    labelings = list(product([+1, -1], repeat=3))   # 8 of them
    for k, lab in enumerate(labelings):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        wb = _separating_line(P3, lab)
        assert wb is not None, f"3-point labelling {lab} should be separable"
        _draw_line(ax, wb, box)
        _draw_points(ax, P3, lab)
        fl.clean_axes(ax, lim=box, hide=True)
    # no shared heading: the caption already states "all 2^3=8 labelings are
    # separable (left); the XOR labeling of 4 points is not (right)".

    # --- right: four points in XOR position, not shatterable ---
    axr = fig.add_subplot(gs[:, 5])
    P4 = np.array([[-0.85, -0.85], [0.85, 0.85], [-0.85, 0.85], [0.85, -0.85]])
    xor = [+1, +1, -1, -1]            # opposite corners share a label
    assert _separating_line(P4, xor) is None, "XOR must NOT be separable"
    big = ((-1.55, 1.55), (-1.45, 1.55))
    # two candidate lines, each clearly failing to separate the two classes:
    # whichever way you cut, each side carries one + and one -.
    for w, b in [(np.array([0.0, 1.0]), 0.0),
                 (np.array([1.0, 0.0]), 0.0)]:
        _draw_line(axr, (w, b), big, color=LIGHT, lw=1.6)
    _draw_points(axr, P4, xor)
    # no per-panel title: the caption already names this panel ("the XOR
    # labeling of 4 points is not [separable]")
    fl.clean_axes(axr, lim=big, hide=True)

    # small shared legend (drawn once, below the right panel)
    axr.plot([], [], "o", color=BLUE, ms=8, label=r"label $+1$")
    axr.plot([], [], "s", mfc="white", mec=ORANGE, mew=2.0, ms=8,
             label=r"label $-1$")
    axr.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2,
               handletextpad=0.3, columnspacing=1.2, fontsize=11)

    fl.save(fig, "mdl-clf-shattering")


FIGURES = [fig_shattering]


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
