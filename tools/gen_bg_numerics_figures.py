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
        fc_alpha=0.14, zorder=None):
    """Rounded box centred at (cx, cy) with a bold coloured title and an
    optional smaller black second line.  Returns anchor points (l/r/t/b).
    ``zorder`` (when given) stacks the box and, via an opaque white base,
    fully hides any connector routed behind it."""
    x, y = cx - w / 2, cy - h / 2
    pk = {} if zorder is None else {"zorder": zorder}
    tk = {} if zorder is None else {"zorder": zorder + 1}
    if zorder is not None:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0, edgecolor="none", facecolor="white", **pk))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8, edgecolor=color, facecolor=color, alpha=fc_alpha, **pk))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8, edgecolor=color, facecolor="none", **pk))
    if sub:
        ax.text(cx, cy + 0.19 * h, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=color, **tk)
        ax.text(cx, cy - 0.26 * h, sub, ha="center", va="center",
                fontsize=sub_fs, color="black", **tk)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=color, **tk)
    return dict(l=x, r=x + w, t=y + h, b=y, cx=cx, cy=cy)


# =========================================================================== #
# 6.5 Numerics: dtype bit layouts + the mixed-precision loop                  #
# =========================================================================== #

def fig_float_formats():
    """The canonical bit-layout figure: one horizontal bar per format (fp32,
    bf16, fp16), aligned at the sign bit, with the exponent and
    mantissa segments drawn to scale (1 bit = 1 unit of width in every row, so
    the bars' total widths compare directly). Segment/bit counts are labeled
    on every row; the segment identity (sign/exponent/mantissa) is named once,
    on the widest (fp32) row, and carried by color for the rest."""
    formats = [
        ("fp32", 1, 8, 23),
        ("bf16", 1, 8, 7),
        ("fp16", 1, 5, 10),
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
                fontsize=13.5, color="black")
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
    backward pass produces the gradients; the optimizer step reads those
    gradients but updates the fp32 master copy, closing the cycle. A dashed
    note marks where the fp16 variant needs one extra move, scaling the loss
    up before backward and the gradients back down before the step."""
    fig, ax = plt.subplots(figsize=(11.6, 3.8))

    y = 2.0
    xs = [1.3, 4.1, 6.9, 9.7, 12.5]
    w, h = 2.3, 1.15

    # decreasing z-order left->right so each connector's tail tucks BEHIND its
    # (opaque) source box while its head stays in FRONT of the target box
    zb = [24, 20, 16, 12, 8]
    fs = dict(title_fs=14, sub_fs=13)  # legible at column width
    master = box(ax, xs[0], y, w, h, "fp32 master", "weights", BLUE,
                 zorder=zb[0], **fs)
    bf16w = box(ax, xs[1], y, w, h, "cast to bf16", "weights", ORANGE,
                zorder=zb[1], **fs)
    fwd = box(ax, xs[2], y, w, h, "forward", "bf16 activations", ORANGE,
              zorder=zb[2], **fs)
    loss = box(ax, xs[3], y, w, h, "loss", "in fp32", BLUE, zorder=zb[3],
               **fs)
    # under autocast the parameter gradients are fp32 (the parameters are)
    bwd = box(ax, xs[4], y, w, h, "backward", "gradients", ORANGE,
              zorder=zb[4], **fs)

    for (a, b), za in zip([(master, bf16w), (bf16w, fwd), (fwd, loss),
                           (loss, bwd)], [22, 18, 14, 10]):
        ax.add_patch(FancyArrowPatch(
            (a["r"] - 0.28, y), (b["l"], y), arrowstyle="-|>",
            mutation_scale=15, color=GRAY, lw=1.8, shrinkA=0, shrinkB=0,
            zorder=za))

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
                 zorder=30))
    # arrowhead tangent to the Bezier's own end (a point at t=0.92 along the
    # SAME curve, not a straight-line shortcut) so it blends into the arc
    # with no visible fork; on top so the head lands in front of fp32 master
    t = 0.92
    p0a, ctrla, p1a = np.asarray(p0), np.asarray(ctrl), np.asarray(p1)
    near_end = (1 - t) ** 2 * p0a + 2 * (1 - t) * t * ctrla + t ** 2 * p1a
    ax.add_patch(FancyArrowPatch(
        tuple(near_end), p1, arrowstyle="-|>", mutation_scale=15,
        color=GREEN, lw=2.0, shrinkA=0, shrinkB=0, zorder=30))
    # label sits just above the arc's apex: a small gap, no dead band
    ax.text((p0[0] + p1[0]) / 2, apex_y + 0.14,
            "optimizer step: update fp32 master weights", ha="center",
            va="bottom", fontsize=14, color=GREEN)

    # fp16-only note, hung below the backward box (where unscaling happens)
    # with a short dotted leader so it reads as attached, not adrift.  The
    # note is right-aligned with the backward box (its dashed frame ends where
    # the box does), so it extends leftward under "loss" instead of widening
    # the canvas past the last stage.
    note_y = y - 1.55
    note_r = bwd["r"] + 0.02 - 0.15  # box edge (incl. pad) minus bbox pad
    # straight-down dotted leader from the backward box's centre
    ax.plot([bwd["cx"], bwd["cx"]], [bwd["b"], note_y + 0.45], ":",
            color=GRAY, lw=1.2, zorder=3)
    ax.text(note_r, note_y,
            "fp16 only: scale the loss up before backward,\n"
            "unscale the gradients back down before the step",
            ha="right", va="center", ma="center", fontsize=13, color="black",
            linespacing=0.85,   # matplotlib >= 3.11: pitch = full font box
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=GRAY,
                      lw=1.0, ls="--", alpha=0.95))

    ax.set_xlim(0.0, 13.9)
    ax.set_ylim(-0.35, 4.25)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-amp-loop")


# =========================================================================== #
# 6.3 Initialization: residual-stream variance under output scaling          #
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
    axa.text(1.95, 2.33, r"identity", color=GRAY, fontsize=12, ha="right",
             va="center")
    axa.set_title("forward", fontsize=13, color="black")
    fl.clean_axes(axa, lim=(xr, yr), hide=True)

    # (b) backward: true gradient (a.e. 0) vs STE surrogate (== 1).
    # ylim hugs the two lines (labels inside) and equal aspect is OFF so the
    # panel stretches to the same height as (a): balanced pair, no dead bands.
    # on top of the black x-axis it coincides with, stopping short of the
    # axis arrowhead, so the zero gradient reads as blue
    axb.plot((xr[0], xr[1] - 0.3), [0, 0], color=BLUE, lw=2.6, zorder=6)
    axb.plot((xr[0], xr[1] - 0.3), [1, 1], color=ORANGE, lw=2.6, zorder=3)
    fl.axis_cross(axb, xr, (-0.32, 1.52), color="black")
    axb.text(0.15, 0.12, r"true gradient $\equiv 0$ a.e.", color=BLUE,
             fontsize=12, ha="left", va="bottom")
    axb.text(0.15, 1.12, r"STE surrogate $\equiv 1$", color=ORANGE,
             fontsize=12, ha="left", va="bottom")
    axb.set_title("backward", fontsize=13, color="black")
    fl.clean_axes(axb, lim=(xr, (-0.45, 1.62)), hide=True, equal=False)

    fl.save(fig, "bg-ste")


def fig_update_swamping():
    """Round-to-nearest on two 16-bit grids near a weight of 1, drawn on one
    shared scale (in units of 2^-10).  bf16 keeps 7 mantissa bits, so its grid
    points next to 1 are 2^-7 apart above 1 (and 2^-8 below, since 1 is a
    power of two); the shaded interval holds every value that rounds back to
    1.  The update 1 + 10^-3 (orange) falls inside it and is lost.  On fp16's
    grid (spacing 2^-10 above 1) the same update lands next to the grid point
    1 + 2^-10, so it survives."""
    fig, ax = plt.subplots(figsize=(10.2, 3.5))
    u = lambda x: (x - 1.0) * 1024        # plot coordinate: units of 2^-10
    lo, hi = -5.2, 10.2                   # visible window, in those units
    rows = [  # (name, y, spacing below 1, spacing above 1)
        ("bf16", 1.9, 2.0 ** -8, 2.0 ** -7),
        ("fp16", 0.0, 2.0 ** -11, 2.0 ** -10),
    ]
    step = 1.001
    for name, y, below, above in rows:
        ax.plot([lo, hi], [y, y], color="black", lw=1.2, zorder=2)
        # every representable value inside the window
        grid = [1 - k * below for k in range(1, 200)
                if u(1 - k * below) >= lo] + \
               [1 + k * above for k in range(0, 200)
                if u(1 + k * above) <= hi]
        for g in grid:
            ax.plot([u(g), u(g)], [y - 0.16, y + 0.16], color="black",
                    lw=1.6 if g == 1 else 1.1, zorder=3)
        # the basin of 1: half the spacing on each side
        a, b = u(1 - below / 2), u(1 + above / 2)
        ax.add_patch(Rectangle((a, y - 0.32), b - a, 0.64, facecolor=BLUE,
                               alpha=0.18, edgecolor="none", zorder=1))
        ax.text(lo - 0.4, y, name, ha="right", va="center", fontsize=15,
                color="black")
        # the update w + 1e-3 and where round-to-nearest sends it
        ax.plot([u(step)], [y], "o", ms=8, color=ORANGE, zorder=5)
        dest = min(grid, key=lambda g: abs(g - step))
        if dest == 1.0:
            ax.add_patch(FancyArrowPatch(
                (u(step), y + 0.12), (u(1.0) + 0.05, y + 0.2),
                connectionstyle="arc3,rad=0.9", arrowstyle="-|>",
                mutation_scale=13, color=ORANGE, lw=1.6, zorder=6))
            ax.text(u(step) + 0.35, y + 0.42, r"$1 + 10^{-3}$ rounds to 1",
                    ha="left", va="bottom", fontsize=13, color=ORANGE)
        else:
            ax.text(u(step) + 0.35, y + 0.42,
                    r"$1 + 10^{-3}$ rounds to $1 + 2^{-10}$",
                    ha="left", va="bottom", fontsize=13, color=ORANGE)
    # tick labels for the bf16 grid, which also anchor the fp16 row
    yb = rows[0][1]
    for x, lab in [(1 - 2.0 ** -8, r"$1 - 2^{-8}$"), (1.0, r"$1$"),
                   (1 + 2.0 ** -7, r"$1 + 2^{-7}$")]:
        ax.text(u(x), yb - 0.3, lab, ha="center", va="top", fontsize=13,
                color="black")
    ax.text(u(1.0), -0.3, r"$1$", ha="center", va="top", fontsize=13,
            color="black")
    ax.set_xlim(lo - 1.9, hi + 0.2)
    ax.set_ylim(-0.75, 2.75)
    ax.axis("off")
    fl.save(fig, "bg-update-swamping")


def diamond(ax, cx, cy, w, h, title, color, fs=12.5, zorder=None,
            fill_alpha=0.14):
    """Decision diamond centred at (cx, cy); returns its four tips."""
    from matplotlib.patches import Polygon
    pts = [(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2)]
    zk = {} if zorder is None else {"zorder": zorder}
    ax.add_patch(Polygon(pts, closed=True, facecolor="white",
                         edgecolor="none", **zk))
    ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                         alpha=fill_alpha, edgecolor="none", **zk))
    ax.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor=color,
                         lw=1.8, **zk))
    ax.text(cx, cy, title, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=color,
            **({} if zorder is None else {"zorder": zorder + 1}))
    return dict(l=cx - w / 2, r=cx + w / 2, t=cy + h / 2, b=cy - h / 2,
                cx=cx, cy=cy)


def link(ax, p, q, color=GRAY, rad=0.0, lw=1.8, zorder=40):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=15, color=color, lw=lw,
        shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}",
        zorder=zorder))


def fig_loss_scaling():
    """One step of dynamic loss scaling as a control loop: scale the fp32
    loss by S, backpropagate in fp16, unscale the gradients in fp32, then
    branch on whether every gradient is finite.  Finite: take the optimizer
    step (and double S after a streak of clean steps).  Not finite: skip the
    step and halve S.  Either way the next step starts over."""
    fig, ax = plt.subplots(figsize=(11.4, 3.9))
    y = 2.0
    w, h = 2.35, 1.15
    fs = dict(title_fs=14, sub_fs=13)
    scale = box(ax, 1.3, y, w, h, "scale loss", r"$S \cdot L$", BLUE, **fs)
    bwd = box(ax, 4.05, y, w, h, "backward", "fp16 gradients", ORANGE, **fs)
    unscale = box(ax, 6.8, y, w, h, "unscale", r"divide by $S$ in fp32",
                  BLUE, **fs)
    dec = diamond(ax, 9.35, y, 1.9, 1.5, "finite?", "black", fs=13.5,
                  fill_alpha=0.0)
    step = box(ax, 12.85, y + 1.05, 3.7, h, "optimizer step",
               r"$S \leftarrow 2S$ after a clean streak", GREEN, **fs)
    skip = box(ax, 12.85, y - 1.05, 3.7, h, "skip the step",
               r"$S \leftarrow S/2$", ORANGE, **fs)
    for a, b in [(scale, bwd), (bwd, unscale)]:
        link(ax, (a["r"] + 0.02, y), (b["l"] - 0.04, y))
    link(ax, (unscale["r"] + 0.02, y), (dec["l"] - 0.02, y))
    # the two exits of the decision, each labelled on its outer side
    link(ax, (dec["cx"] + 0.35, dec["t"] - 0.28), (step["l"] - 0.04,
                                                    step["cy"]), color=GREEN)
    link(ax, (dec["cx"] + 0.35, dec["b"] + 0.28), (skip["l"] - 0.04,
                                                    skip["cy"]), color=ORANGE)
    ax.text(dec["cx"] + 0.95, dec["t"] + 0.0, "yes", ha="right",
            va="bottom", fontsize=13, color="black")
    ax.text(dec["cx"] + 0.95, dec["b"] - 0.0, "no", ha="right", va="top",
            fontsize=13, color="black")
    ax.set_xlim(0.0, 14.85)
    ax.set_ylim(0.3, 3.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-loss-scaling")


def fig_res_variance():
    """Mean-square bookkeeping through one residual block whose body is
    Linear(d -> 4d, He) -> ReLU -> Linear(4d -> d, gain g).  The value
    written on each arrow is the mean square of the signal it carries: the He
    layer doubles m_k, the ReLU halves it, the output projection scales it by
    g^2, and the identity path adds m_k back at the sum."""
    fig, ax = plt.subplots(figsize=(11.4, 3.6))
    y = 2.2
    w, h = 2.15, 1.15
    fs = dict(title_fs=13.5, sub_fs=12.5)
    xs = [2.15, 5.4, 8.65]
    l1 = box(ax, xs[0], y, w, h, "linear", r"$d \to 4d$, He", BLUE, **fs)
    relu = box(ax, xs[1], y, w, h, "ReLU", None, ORANGE, **fs)
    l2 = box(ax, xs[2], y, w, h, "linear", r"$4d \to d$, gain $g$", BLUE,
             **fs)
    sx = 11.3  # the sum node
    ax.add_patch(plt.Circle((sx, y), 0.24, facecolor="white",
                            edgecolor="black", lw=1.6, zorder=5))
    ax.text(sx, y, "+", ha="center", va="center", fontsize=16,
            color="black", zorder=6)
    x0 = -0.3
    # main path with the mean square carried by each arrow
    segs = [(x0, l1["l"], r"$m_k$"), (l1["r"], relu["l"], r"$2m_k$"),
            (relu["r"], l2["l"], r"$m_k$"), (l2["r"], sx - 0.26,
                                            r"$g^2 m_k$"),
            (sx + 0.26, sx + 2.0, r"$(1 + g^2)\,m_k$")]
    for a, b, lab in segs:
        link(ax, (a + 0.02, y), (b - 0.04, y), color="black", lw=1.6)
        ax.text((a + b) / 2, y + 0.14, lab, ha="center", va="bottom",
                fontsize=14, color="black")
    ax.text(x0 - 0.1, y, r"$\mathbf{x}_k$", ha="right", va="center",
            fontsize=15, color="black")
    ax.text(sx + 2.1, y, r"$\mathbf{x}_{k+1}$", ha="left", va="center",
            fontsize=15, color="black")
    # identity path under the body, joining at the sum
    yb = y - 1.35
    xb = 0.45
    ax.plot([xb, xb, sx, sx], [y, yb, yb, y - 0.24 - 0.35], color=GRAY,
            lw=1.6, zorder=2)
    link(ax, (sx, y - 0.6), (sx, y - 0.25), color=GRAY, lw=1.6)
    ax.plot([xb], [y], "o", ms=5, color="black", zorder=6)
    ax.text((xb + sx) / 2, yb - 0.12, r"identity: $m_k$, uncorrelated "
            r"with the body output", ha="center", va="top", fontsize=13,
            color="black")
    ax.set_xlim(-1.0, 14.1)
    ax.set_ylim(0.3, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-res-variance")


def fig_zero_out_proj():
    """Zeroing only the output projection of a residual body, step by step,
    against zeroing both layers.  Each row is the body x -> W_in -> h ->
    W_out -> f.  Filled boxes hold nonzero weights, dashed white boxes hold
    zeros; an orange arrow under a box means that layer receives a nonzero
    gradient in that step, a gray one that its gradient is zero."""
    fig, ax = plt.subplots(figsize=(11.4, 6.4))
    w, h = 1.55, 0.9
    xin, xout = 4.75, 8.25
    rows = [
        ("step 0, forward", dict(win=True, wout=False, h=r"$\mathbf{h} \neq 0$",
                                 f=r"$f = 0$", gin=None, gout=None,
                                 note="every block is the identity")),
        ("step 1, backward", dict(win=True, wout=False,
                                  h=r"$\mathbf{h} \neq 0$", f=r"$f = 0$",
                                  gin=False, gout=True,
                                  note=r"$\nabla W_{\mathrm{out}} = "
                                       r"\boldsymbol{\delta}\,\mathbf{h}^{\top}"
                                       r" \neq 0$, $\nabla W_{\mathrm{in}} = 0$")),
        ("step 2 onward", dict(win=True, wout=True, h=r"$\mathbf{h} \neq 0$",
                               f=r"$f \neq 0$", gin=True, gout=True,
                               note="both layers train")),
        ("both zeroed", dict(win=False, wout=False, h=r"$\mathbf{h} = 0$",
                             f=r"$f = 0$", gin=False, gout=False,
                             note="both gradients stay 0 forever")),
    ]
    pitch = 1.75
    for i, (title, r) in enumerate(rows):
        y = -i * pitch - (0.35 if i == 3 else 0.0)
        ax.text(0.0, y, title, ha="left", va="center", fontsize=13.5,
                color="black", fontweight="bold")
        x_in = 3.2
        link(ax, (x_in, y), (xin - w / 2 - 0.04, y), color="black", lw=1.4)
        ax.text(x_in - 0.08, y, r"$\mathbf{x}$", ha="right", va="center",
                fontsize=14, color="black")
        for cx, on, lab in [(xin, r["win"], r"$W_{\mathrm{in}}$"),
                            (xout, r["wout"], r"$W_{\mathrm{out}}$")]:
            if on:
                box(ax, cx, y, w, h, lab, None, BLUE, title_fs=14)
            else:
                ax.add_patch(FancyBboxPatch(
                    (cx - w / 2, y - h / 2), w, h,
                    boxstyle="round,pad=0.02,rounding_size=0.12",
                    linewidth=1.6, edgecolor=GRAY, facecolor="white",
                    ls="--"))
                ax.text(cx, y, lab[:-1] + r" = 0$", ha="center",
                        va="center", fontsize=13.5, color=GRAY)
        link(ax, (xin + w / 2 + 0.02, y), (xout - w / 2 - 0.04, y),
             color="black", lw=1.4)
        ax.text((xin + xout) / 2, y + 0.12, r["h"], ha="center",
                va="bottom", fontsize=13, color="black")
        link(ax, (xout + w / 2 + 0.02, y), (xout + w / 2 + 1.25, y),
             color="black", lw=1.4)
        ax.text(xout + w / 2 + 1.35, y, r["f"], ha="left", va="center",
                fontsize=13.5, color="black")
        # gradient arrows under each layer: right-to-left, as backprop runs
        for cx, g in [(xin, r["gin"]), (xout, r["gout"])]:
            if g is None:
                continue
            yg = y - h / 2 - 0.22
            ax.add_patch(FancyArrowPatch(
                (cx + 0.6, yg), (cx - 0.6, yg), arrowstyle="-|>",
                mutation_scale=12, color=ORANGE if g else GRAY,
                lw=2.0 if g else 1.3, ls="-" if g else "--", shrinkA=0,
                shrinkB=0))
        ax.text(11.6, y, r["note"], ha="left", va="center", fontsize=13,
                color="black")
    ax.plot([0.0, 16.4], [-2.5 * pitch - 0.2] * 2, color=LIGHT, lw=1.2)
    ax.set_xlim(-0.1, 16.5)
    ax.set_ylim(-3 * pitch - 0.35 - 0.85, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-zero-out-proj")


if __name__ == "__main__":
    fig_float_formats()
    fig_amp_loop()
    fig_ste()
    fig_update_swamping()
    fig_loss_scaling()
    fig_res_variance()
    fig_zero_out_proj()
    print(f"wrote {len(fl.WRITTEN)} figures:")
    for p in fl.WRITTEN:
        print(" ", p)
