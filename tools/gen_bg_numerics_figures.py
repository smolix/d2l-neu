#!/usr/bin/env python3
"""Generate the illustrative figures for the Builders' Guide v2 chapters that
touch dtypes/mixed precision, residual-branch scaling, and the
straight-through estimator, in the one shared house style defined in
``gen_mdl_figures.py``.

Covers:
  - ``numerics.md``       (:numref:`sec_numerics_v2`)       -- dtype bit
    layouts and the mixed-precision training loop.
  - ``init.md``           (:numref:`sec_init_v2`)           -- residual-stream
    variance growth under GPT-2-style output scaling.
  - ``custom-layers.md``  (:numref:`sec_custom_layers_v2`)  -- the
    straight-through estimator's forward/backward split.

The notebooks / prose reference the generated files with no drawing code
(like the slide SVGs). Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_bg_numerics_figures.py

All figures are written to ``img/bg-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# --------------------------------------------------------------------------- #
# Chapter-local helper: a rounded pipeline box (shared by the amp-loop and     #
# residual-stream figures).                                                   #
# --------------------------------------------------------------------------- #

def box(ax, cx, cy, w, h, title, sub, color, title_fs=12.5, sub_fs=10.5,
        fc_alpha=0.14):
    """Rounded box centred at (cx, cy) with a bold coloured title and an
    optional smaller black second line.  Returns anchor points (l/r/t/b)."""
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8, edgecolor=color, facecolor=color, alpha=fc_alpha))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8, edgecolor=color, facecolor="none"))
    if sub:
        ax.text(cx, cy + 0.19 * h, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=color)
        ax.text(cx, cy - 0.26 * h, sub, ha="center", va="center",
                fontsize=sub_fs, color="black")
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=color)
    return dict(l=x, r=x + w, t=y + h, b=y, cx=cx, cy=cy)


# =========================================================================== #
# 6.5 Numerics: dtype bit layouts + the mixed-precision loop                  #
# =========================================================================== #

def fig_float_formats():
    """The canonical bit-layout figure: one horizontal bar per format (fp32,
    tf32, bf16, fp16, fp8 e4m3), aligned at the sign bit, with the exponent and
    mantissa segments drawn to scale (1 bit = 1 unit of width in every row, so
    the bars' total widths compare directly). Segment/bit counts are labeled
    on every row; the segment identity (sign/exponent/mantissa) is named once,
    on the widest (fp32) row, and carried by color for the rest."""
    formats = [
        ("fp32", 1, 8, 23),
        ("tf32", 1, 8, 10),
        ("bf16", 1, 8, 7),
        ("fp16", 1, 5, 10),
        ("fp8 (e4m3)", 1, 4, 3),
    ]
    h = 1.0     # bar height, in bit-units (1 bit = 1 unit of width)
    pitch = 1.8  # row spacing, roomy enough for the larger row labels
    n = len(formats)

    fig, ax = plt.subplots(figsize=(9.6, 1.0 + pitch * n * 0.62))

    for i, (name, s, e, m) in enumerate(formats):
        y = -(i * pitch)
        total = s + e + m
        # sign cell
        ax.add_patch(Rectangle((0, y), s, h, facecolor=LIGHT, alpha=0.9,
                                edgecolor="black", lw=1.0))
        # exponent segment
        ax.add_patch(Rectangle((s, y), e, h, facecolor=ORANGE, alpha=0.55,
                                edgecolor=ORANGE, lw=1.2))
        ax.text(s + e / 2, y + h / 2, str(e), ha="center", va="center",
                fontsize=14, color="black")
        # mantissa segment
        ax.add_patch(Rectangle((s + e, y), m, h, facecolor=BLUE, alpha=0.35,
                                edgecolor=BLUE, lw=1.2))
        ax.text(s + e + m / 2, y + h / 2, str(m), ha="center", va="center",
                fontsize=14, color="black")
        # row label (format name), black, right of the left margin
        ax.text(-0.6, y + h / 2, name, ha="right", va="center", fontsize=16,
                color="black")
        # total-width readout to the right of the bar
        ax.text(33.0, y + h / 2, f"{total} bits", ha="left", va="center",
                fontsize=13.5, color=GRAY)
        if i == 0:  # direct labels on the widest row double as the key
            ax.text(s / 2, y + h + 0.22, "sign", ha="center", va="bottom",
                    fontsize=14, color="black")
            ax.text(s + e / 2, y + h + 0.22, "exponent", ha="center",
                    va="bottom", fontsize=14, color=ORANGE)
            ax.text(s + e + m / 2, y + h + 0.22, "mantissa", ha="center",
                    va="bottom", fontsize=14, color=BLUE)

    ax.set_xlim(-6.4, 38.5)
    ax.set_ylim(-(n - 1) * pitch - 0.4, h + 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-float-formats")


def fig_amp_loop():
    """Mixed-precision training as a dataflow loop. fp32 master weights (blue)
    are cast to bf16 (orange) for the forward pass and its bf16 activations;
    the loss accumulates back in fp32 (autocast pins reductions there); the
    backward pass produces bf16 gradients; the optimizer step reads those
    gradients but updates the fp32 master copy, closing the cycle. A dashed
    note marks where the fp16 variant needs one extra move, scaling the loss
    up before backward and the gradients back down before the step."""
    fig, ax = plt.subplots(figsize=(11.6, 3.8))

    y = 2.0
    xs = [1.3, 4.1, 6.9, 9.7, 12.5]
    w, h = 2.3, 1.15

    master = box(ax, xs[0], y, w, h, "fp32 master", "weights", BLUE)
    bf16w = box(ax, xs[1], y, w, h, "cast to bf16", "weights", ORANGE)
    fwd = box(ax, xs[2], y, w, h, "forward", "bf16 activations", ORANGE)
    loss = box(ax, xs[3], y, w, h, "loss", "fp32 accumulate", BLUE)
    bwd = box(ax, xs[4], y, w, h, "backward", "bf16 grads", ORANGE)

    for a, b in [(master, bf16w), (bf16w, fwd), (fwd, loss), (loss, bwd)]:
        fl.arrow(ax, (a["r"], y), (b["l"], y), color=GRAY, lw=1.8)

    # the return arc: optimizer step, gradients feed back into the fp32
    # master copy, closing the cycle.  Drawn as an explicit quadratic Bezier
    # (not connectionstyle="arc3") so the apex height is known exactly and the
    # label can sit safely above it with no intersection.
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    p0 = (bwd["cx"], bwd["t"] + 0.05)
    p1 = (master["cx"], master["t"] + 0.05)
    apex_y = 3.55  # the curve's TRUE apex height (solved for below), kept low
    # a quadratic Bezier's midpoint is (p0 + 2*ctrl + p1)/4, so pick the
    # control point that puts the apex exactly at apex_y
    ctrl = ((p0[0] + p1[0]) / 2, 2 * apex_y - (p0[1] + p1[1]) / 2)
    verts = [p0, ctrl, p1]
    path = Path(verts, [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(path, edgecolor=GREEN, facecolor="none", lw=2.0,
                 zorder=5))
    # arrowhead tangent to the Bezier's own end (a point at t=0.92 along the
    # SAME curve, not a straight-line shortcut) so it blends into the arc
    # with no visible fork
    t = 0.92
    p0a, ctrla, p1a = np.asarray(p0), np.asarray(ctrl), np.asarray(p1)
    near_end = (1 - t) ** 2 * p0a + 2 * (1 - t) * t * ctrla + t ** 2 * p1a
    fl.arrow(ax, tuple(near_end), p1, color=GREEN, lw=2.0, mut=15)
    # label sits just above the arc's apex: a small gap, no dead band
    ax.text((p0[0] + p1[0]) / 2, apex_y + 0.14,
            "optimizer step: update fp32 master weights", ha="center",
            va="bottom", fontsize=12.5, color=GREEN)

    # fp16-only note, hung below the backward box (where unscaling happens)
    # with a short dashed leader so it reads as attached, not adrift
    note_x, note_y = bwd["cx"] - 0.5, y - 1.55
    ax.plot([bwd["cx"], note_x], [bwd["b"], note_y + 0.50], ":", color=GRAY,
            lw=1.2, zorder=3)
    ax.text(note_x, note_y,
            "fp16 only: scale the loss up before backward,\n"
            "unscale the gradients back down before the step",
            ha="center", va="center", fontsize=11, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRAY,
                      lw=1.0, ls="--", alpha=0.95))

    ax.set_xlim(0.0, 13.9)
    ax.set_ylim(-0.35, 4.25)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-amp-loop")


# =========================================================================== #
# 6.3 Initialization: residual-stream variance under output scaling          #
# =========================================================================== #

def _residual_stream_panel(ax, scaled, n_blocks=3):
    """One stream with n_blocks additive taps. Unscaled: the stream thickens
    and darkens left to right as variance compounds. Scaled: the stream stays
    a constant width and shade, since each contribution is cut by 1/sqrt(N).

    Drawn LARGE: three taps only, thick strokes, 13-15pt labels, and tight
    limits so the drawing fills the panel at column width."""
    x0, x1 = 0.4, 7.6
    taps = np.linspace(x0 + 1.3, x1 - 0.9, n_blocks)   # 1.7, 4.2, 6.7

    # the stream itself: a sequence of segments whose linewidth (and,
    # unscaled, shade) grows after each tap
    bounds = [x0] + list(taps) + [x1]
    for k in range(len(bounds) - 1):
        if scaled:
            lw, alpha = 8.0, 0.55
        else:
            lw = 8.0 + 5.5 * k
            alpha = 0.32 + 0.16 * k
        ax.plot(bounds[k:k + 2], [0, 0], color=BLUE, lw=lw, alpha=alpha,
                solid_capstyle="butt", zorder=2)

    for k, tx in enumerate(taps):
        by = -1.85
        bw, bh = 1.45, 1.0
        ax.add_patch(FancyBboxPatch((tx - bw / 2, by - bh / 2), bw, bh,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     linewidth=1.5, edgecolor=GRAY, facecolor=LIGHT,
                     alpha=0.6, zorder=3))
        ax.text(tx, by, rf"$f_{k+1}$", ha="center", va="center", fontsize=15,
                color="black", zorder=4)
        fl.arrow(ax, (tx, by + bh / 2), (tx, -0.16), color=GRAY, lw=2.0,
                 mut=13)
        # the join node on the stream
        ax.plot([tx], [0], "o", mfc="white", mec="black", ms=19, mew=1.6,
                zorder=5)
        ax.text(tx, 0.0, r"$\oplus$", ha="center", va="center", fontsize=14,
                color="black", zorder=6)
        if scaled:
            # offset to the side of the up-arrow (not centred on it), so the
            # label never crosses the line it annotates
            ax.text(tx + 0.28, -0.85, r"$\times\,1/\sqrt{N}$",
                    ha="left", va="center", fontsize=13, color=ORANGE,
                    zorder=4)

    label = (r"variance $\sim O(1)$" if scaled else r"variance $\sim N$")
    color = GREEN if scaled else ORANGE
    # verdict label above the (thick) right end of the stream, inside the box
    ax.text(x1 + 0.55, 0.72, label, ha="right", va="center", fontsize=15,
            color=color)
    ax.text(x0 - 0.12, 0.55, r"$\mathbf{x}_0$", ha="right", va="center",
            fontsize=15, color="black")
    fl.arrow(ax, (x1, 0), (x1 + 0.6, 0), color=BLUE, lw=6.0, mut=20,
             alpha=(0.55 if scaled else 0.32 + 0.16 * n_blocks))

    ax.set_xlim(x0 - 1.05, x1 + 0.75)
    ax.set_ylim(-2.5, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_residual_stream():
    """A residual stream with three additive block contributions, unscaled
    (left) versus scaled by 1/sqrt(N) (right). Each block contributes O(1)
    variance to the stream it joins; left uncorrected the stream's variance
    compounds like N, visualized as a thickening, darkening line, while
    scaling each contribution tames it back to O(1), a stream of constant
    width and shade."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.2, 3.4))
    fig.subplots_adjust(wspace=0.06)
    _residual_stream_panel(axa, scaled=False)
    _residual_stream_panel(axb, scaled=True)
    fl.save(fig, "bg-residual-stream")


# =========================================================================== #
# 6.4 Custom layers: the straight-through estimator                          #
# =========================================================================== #

def fig_ste():
    """Straight-through estimator, forward vs backward. Forward: round(x) (the
    staircase, blue) sits close to the identity (gray dashed) it approximates.
    Backward: the true derivative is zero almost everywhere (blue), so no
    gradient would reach x through an honest chain rule; the STE instead
    substitutes a constant surrogate gradient of 1 (orange), the identity's
    own derivative, letting gradient flow as if rounding had not happened."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    xr, yr = (-2.4, 2.4), (-2.4, 2.4)  # round(2.4) = 2, so the staircase never
                                        # clips the frame at the outer steps

    # (a) forward: staircase vs identity
    xs = np.linspace(*xr, 2000)
    axa.plot(xs, xs, "--", color=GRAY, lw=1.6, zorder=2)
    axa.plot(xs, np.round(xs), color=BLUE, lw=2.4, zorder=3)
    fl.axis_cross(axa, xr, yr, color="black")
    axa.text(1.85, 1.30, r"$\mathrm{round}(x)$", color=BLUE, fontsize=13,
             ha="left", va="center")
    # the staircase hugs the identity line everywhere, so no spot directly on
    # either curve is safe; park the label in the empty top-left corner
    # instead, well clear of both the diagonal and the steps
    axa.text(-1.9, 1.75, r"identity", color=GRAY, fontsize=12, ha="center",
             va="center")
    axa.set_title("forward", fontsize=13, color="black")
    fl.clean_axes(axa, lim=(xr, yr), hide=True)

    # (b) backward: true gradient (a.e. 0) vs STE surrogate (== 1).
    # ylim hugs the two lines (labels inside) and equal aspect is OFF so the
    # panel stretches to the same height as (a): balanced pair, no dead bands.
    axb.plot(xr, [0, 0], color=BLUE, lw=2.6, zorder=3)
    axb.plot(xr, [1, 1], color=ORANGE, lw=2.6, zorder=3)
    fl.axis_cross(axb, xr, (-0.32, 1.52), color="black")
    axb.text(0.15, 0.12, r"true gradient $\equiv 0$ a.e.", color=BLUE,
             fontsize=12, ha="left", va="bottom")
    axb.text(0.15, 1.12, r"STE surrogate $\equiv 1$", color=ORANGE,
             fontsize=12, ha="left", va="bottom")
    axb.set_title("backward", fontsize=13, color="black")
    fl.clean_axes(axb, lim=(xr, (-0.45, 1.62)), hide=True, equal=False)

    fl.save(fig, "bg-ste")


if __name__ == "__main__":
    fig_float_formats()
    fig_amp_loop()
    fig_residual_stream()
    fig_ste()
    print(f"wrote {len(fl.WRITTEN)} figures:")
    for p in fl.WRITTEN:
        print(" ", p)
