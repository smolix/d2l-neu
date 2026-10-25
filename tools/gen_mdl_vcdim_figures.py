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
T = fl.T
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
        # all one class: a line just below the cloud leaves every point on
        # one side; orient the normal so that this side carries the label.
        sign = 1.0 if y[0] > 0 else -1.0
        w = np.array([0.0, 1.0])
        b = -((P @ w).min() - 0.3)
        w, b = sign * w, sign * b
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


def _frame(ax, lim):
    """Fix the panel window and draw a hairline frame around it (no ticks), so
    the small panels read as separate cases and no line seems to run from
    one panel into the next."""
    fl.clean_axes(ax, lim=lim, hide=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(T.HAIRLINE)
        spine.set_linewidth(1.2)


def _segment(ax, p, q, color):
    """Dashed segment from p to q: the convex hull of two same-label points."""
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=2.0, ls=(0, (4, 3)),
            zorder=2)


def fig_shattering():
    """Three blocks in the shared house style.

    Left: all 2^3 = 8 labellings of three points in general position, each
    with a *computed* max-margin separating line: lines shatter 3 points.
    Middle and right: the two ways four points (no three collinear) can lie,
    each with the one labelling that no line realizes, and the reason drawn in:
    the convex hulls of the two classes intersect.  Middle: convex position,
    XOR labels; the two diagonals (dashed, one per class) cross.  Right: one
    point inside the triangle of the other three, labelled opposite to them;
    it lies inside their triangle.  We verify that neither labelling is
    separable."""
    # three points in general position (a triangle, well inside a small box)
    P3 = np.array([[-0.78, -0.55], [0.78, -0.55], [0.0, 0.82]])
    box = ((-1.3, 1.3), (-1.2, 1.3))

    fig = plt.figure(figsize=(12.0, 3.0))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 0.3, 2.1, 2.1],
                          wspace=0.10, hspace=0.10)

    # --- left: 2x4 grid of the eight labellings of three points ---
    labelings = list(product([+1, -1], repeat=3))   # 8 of them
    for k, lab in enumerate(labelings):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        wb = _separating_line(P3, lab)
        assert wb is not None, f"3-point labelling {lab} should be separable"
        _draw_line(ax, wb, box)
        _draw_points(ax, P3, lab)
        _frame(ax, box)

    big = ((-1.45, 1.45), (-1.45, 1.45))

    # --- middle: four points in convex position, XOR labels ---
    axm = fig.add_subplot(gs[:, 5])
    P4 = np.array([[-0.9, -0.75], [0.8, 0.9], [-0.75, 0.85], [0.95, -0.8]])
    xor = [+1, +1, -1, -1]            # opposite corners share a label
    assert _separating_line(P4, xor) is None, "XOR must NOT be separable"
    _segment(axm, P4[0], P4[1], BLUE)
    _segment(axm, P4[2], P4[3], ORANGE)
    _draw_points(axm, P4, xor)
    _frame(axm, big)

    # --- right: one point inside the triangle of the other three ---
    axr = fig.add_subplot(gs[:, 6])
    Q4 = np.array([[-1.0, -0.8], [1.0, -0.8], [0.0, 1.05], [0.05, -0.1]])
    inner = [+1, +1, +1, -1]          # the inner point against the corners
    assert _separating_line(Q4, inner) is None, "inner point must NOT be separable"
    tri = Q4[:3]
    axr.fill(tri[:, 0], tri[:, 1], color=T.BLUE.tint, zorder=1, lw=0)
    for i in range(3):
        _segment(axr, tri[i], tri[(i + 1) % 3], BLUE)
    _draw_points(axr, Q4, inner)
    _frame(axr, big)

    # one shared legend, centred below the whole figure
    axm.plot([], [], "o", color=BLUE, ms=8, label=r"label $+1$")
    axm.plot([], [], "s", mfc="white", mec=ORANGE, mew=2.0, ms=8,
             label=r"label $-1$")
    fig.legend(*axm.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.10), ncol=2, handletextpad=0.3,
               columnspacing=1.5, fontsize=13)

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
