#!/usr/bin/env python3
"""Generate the illustrative figures for the "Preliminaries" chapter
(``chapter_preliminaries``) that are not promoted slide SVGs, in the one
shared house style.

Currently this is a single figure: the cosine-similarity panel for the
linear-algebra section (two unit vectors at angle theta with their dot
product cos(theta) annotated, for theta in {0, 60, 90, 150} degrees).

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` + ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette and
the drawing helpers (``arrow``/``right_angle``/``clean_axes``/``axis_cross``/
``vlabel``).  The angles and cosines are computed, not sketched.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_prelim_figures.py

All figures are written to ``img/mdl-prelim-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
arrow, vlabel, right_angle = fl.arrow, fl.vlabel, fl.right_angle
clean_axes, axis_cross, save = fl.clean_axes, fl.axis_cross, fl.save

from matplotlib.patches import Arc


# --------------------------------------------------------------------------- #
# LINEAR ALGEBRA (Section 2.3)                                                #
# --------------------------------------------------------------------------- #

def fig_cosine():
    """Four panels: unit vectors u (fixed along the x-axis) and v at angle
    theta in {0, 60, 90, 150} degrees, with the dot product u.v = cos(theta)
    annotated under each panel."""
    thetas_deg = [0, 60, 90, 150]
    fig, axes = plt.subplots(1, len(thetas_deg), figsize=(9.6, 2.9))

    u = np.array([1.0, 0.0])
    for ax, deg in zip(axes, thetas_deg):
        th = np.deg2rad(deg)
        v = np.array([np.cos(th), np.sin(th)])
        cos = float(u @ v)  # computed, not typed in

        # faint upper unit-circle arc for scale (both vectors have length 1)
        circ = np.linspace(np.deg2rad(-12), np.deg2rad(192), 200)
        ax.plot(np.cos(circ), np.sin(circ), color=LIGHT, lw=1.0, zorder=1)

        # the two unit vectors; nudge v slightly upward when theta = 0 so
        # the coinciding arrows stay individually visible
        arrow(ax, (0, 0), u, color=BLUE, lw=2.2)
        if deg == 0:
            eps = 0.055
            arrow(ax, (0, eps), v + (0, eps), color=ORANGE, lw=2.2)
        else:
            arrow(ax, (0, 0), v, color=ORANGE, lw=2.2)
        vlabel(ax, (1.14, -0.16 if deg == 0 else 0.0), r"$\mathbf{u}$",
               color=BLUE, ha="left", fontsize=14)
        voff = v * 1.16
        if deg == 0:
            voff = np.array([1.14, 0.20])
        vlabel(ax, voff, r"$\mathbf{v}$", color=ORANGE,
               ha="left" if v[0] >= 0 else "right", fontsize=14)

        # the angle arc + right-angle marker at 90 degrees
        if deg == 90:
            right_angle(ax, (0, 0), u, v, size=0.16)
        elif deg > 0:
            ax.add_patch(Arc((0, 0), 0.66, 0.66, angle=0, theta1=0,
                             theta2=deg, color=GRAY, lw=1.2))
            # on the true bisector, pushed out well past the arc so the
            # label's clearance to BOTH bounding rays (u and v) stays large
            # even for the narrower 60-degree wedge
            mid = np.deg2rad(deg / 2.0)
            ax.text(0.68 * np.cos(mid), 0.68 * np.sin(mid),
                    rf"${deg}^\circ$", color=GRAY,
                    ha="center", va="center", fontsize=12)

        # annotation: the dot product IS the cosine
        val = f"{cos:.2f}".rstrip("0").rstrip(".")
        if val == "-0":
            val = "0"
        ax.set_title(
            rf"$\mathbf{{u}}^\top \mathbf{{v}} = \cos {deg}^\circ = {val}$",
            fontsize=13, pad=8)

        clean_axes(ax, lim=((-1.35, 1.45), (-0.32, 1.3)), hide=True)

    fig.subplots_adjust(wspace=0.25)
    save(fig, "mdl-prelim-cosine")


FIGURES = [
    fig_cosine,
]


def main():
    # Only verify the figures this script writes (the shared module tracks all
    # figures via the same WRITTEN list).
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
            head = fh.read(400)
        assert "<svg" in head, f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):30s} {size:>8,d} bytes")

    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
