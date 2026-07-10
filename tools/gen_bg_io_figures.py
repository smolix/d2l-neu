#!/usr/bin/env python3
"""Generate the illustrative figures for the Builders' Guide's two I/O
sections (``saving-loading.md`` and ``reproducibility-inspection.md``) in the
shared house style.

The shared style lives in ``gen_mdl_figures.py``; importing it applies the
``plt.rcParams`` (fixed ``svg.hashsalt`` + ``metadata={'Date': None}`` in
``save()`` make re-runs byte-for-byte identical) and exposes the palette and
the drawing helpers (``arrow``/``clean_axes``/``axis_cross``/``vlabel``). The
notebooks reference the generated files with no drawing code (like the slide
SVGs).

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_bg_io_figures.py

All figures are written to ``img/bg-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyBboxPatch, Rectangle


# --------------------------------------------------------------------------- #
# Small chapter-local helpers for the schematic box-and-arrow diagrams.       #
# --------------------------------------------------------------------------- #

def box(ax, cx, cy, w, h, label, facecolor=LIGHT, edgecolor="black", ls="-",
        lw=1.4, fontsize=12, color="black", rounding=0.06):
    """A labelled rounded box centered at (cx, cy)."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle=f"round,pad=0.0,rounding_size={rounding}",
                 facecolor=facecolor, edgecolor=edgecolor, lw=lw, ls=ls,
                 zorder=3))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize,
            color=color, zorder=4)


def leader(ax, p, q, color=GRAY, lw=0.9, ls="-"):
    """A thin leader line (no arrowhead) connecting a label to what it names."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=2)


# =========================================================================== #
# saving-loading.md :: safetensors section                                   #
# =========================================================================== #

def fig_safetensors_layout():
    """The safetensors file as one horizontal byte-strip: an 8-byte header
    length, then a JSON header naming each tensor's dtype/shape/byte range,
    then the raw tensor bytes packed back to back with no gaps.  Two of the
    raw-tensor sub-spans are labelled with the exact byte ranges quoted in the
    JSON excerpt, tied to their location in the bar by thin leader lines, so
    the figure reads left to right exactly as the file does on disk."""
    fig, ax = plt.subplots(figsize=(11.4, 4.3))

    y0, h = 0.0, 1.0

    # --- section widths: everything abuts (no gap) -- the file itself has no
    # padding between the length prefix, the header, and the tensor bytes.
    w_len = 0.7
    w_hdr = 4.8
    # raw-bytes region subdivided: [other][hidden.weight][output.bias][other],
    # abutting so hidden.weight's span ends exactly where output.bias's begins
    # -- the "back to back, no holes" packing the offsets encode.
    w_other0, w_hw, w_ob, w_other1 = 1.5, 3.0, 0.45, 1.85
    w_raw = w_other0 + w_hw + w_ob + w_other1

    x_len0 = 0.0
    x_hdr0 = x_len0 + w_len
    x_raw0 = x_hdr0 + w_hdr
    x_other0_0 = x_raw0
    x_hw0 = x_other0_0 + w_other0
    x_ob0 = x_hw0 + w_hw
    x_other1_0 = x_ob0 + w_ob
    x_end = x_raw0 + w_raw

    # header-length cell
    ax.add_patch(Rectangle((x_len0, y0), w_len, h, facecolor=GRAY, alpha=0.35,
                           edgecolor="black", lw=1.3, zorder=3))
    # JSON header block
    ax.add_patch(Rectangle((x_hdr0, y0), w_hdr, h, facecolor=ORANGE, alpha=0.15,
                           edgecolor="black", lw=1.3, zorder=3))
    # raw tensor bytes: faint background for "other tensors", then the two
    # highlighted, named sub-spans
    ax.add_patch(Rectangle((x_other0_0, y0), w_other0, h, facecolor=BLUE,
                           alpha=0.10, edgecolor="black", lw=1.3, zorder=3))
    ax.add_patch(Rectangle((x_hw0, y0), w_hw, h, facecolor=BLUE, alpha=0.55,
                           edgecolor="black", lw=1.3, zorder=3))
    ax.add_patch(Rectangle((x_ob0, y0), w_ob, h, facecolor=GREEN, alpha=0.55,
                           edgecolor="black", lw=1.3, zorder=3))
    ax.add_patch(Rectangle((x_other1_0, y0), w_other1, h, facecolor=BLUE,
                           alpha=0.10, edgecolor="black", lw=1.3, zorder=3))

    # "..." markers over the unlabelled other-tensor spans
    ax.text(x_other0_0 + w_other0 / 2, y0 + h / 2, r"$\cdots$", color=GRAY,
            fontsize=13, ha="center", va="center", zorder=4)
    ax.text(x_other1_0 + w_other1 / 2, y0 + h / 2, r"$\cdots$", color=GRAY,
            fontsize=13, ha="center", va="center", zorder=4)

    # header-length label, above, with a leader line down to its cell
    lx = x_len0 + w_len / 2
    ax.text(lx, 2.55, "8 bytes:\nheader length\n(LE u64)", color="black",
            fontsize=11, ha="center", va="bottom")
    leader(ax, (lx, 2.5), (lx, h))

    # JSON header excerpt, rendered as small monospace text inside its box
    hx = x_hdr0 + 0.12
    ax.text(hx, y0 + 0.74, r'"hidden.weight": {"dtype": "F32",', color="black",
            fontsize=8.7, ha="left", va="center", family="monospace", zorder=4)
    ax.text(hx, y0 + 0.50, r'  "shape": [256, 20],', color="black",
            fontsize=8.7, ha="left", va="center", family="monospace", zorder=4)
    ax.text(hx, y0 + 0.26, r'  "data_offsets": [0, 20480]}, ...', color="black",
            fontsize=8.7, ha="left", va="center", family="monospace", zorder=4)
    ax.text(x_hdr0 + w_hdr / 2, 1.55, "JSON header\n(name, dtype, shape, byte offsets)",
            color=ORANGE, fontsize=11, ha="center", va="bottom")
    leader(ax, (x_hdr0 + w_hdr / 2, 1.5), (x_hdr0 + w_hdr / 2, h))

    # labelled sub-spans, each with a leader line down to its slice of the bar
    hw_cx = x_hw0 + w_hw / 2
    ax.text(hw_cx, 2.55, '"hidden.weight"\n[0, 20480)', color=BLUE, fontsize=11,
            ha="center", va="bottom")
    leader(ax, (hw_cx, 2.5), (hw_cx, h))

    ob_cx = x_ob0 + w_ob / 2
    ax.text(ob_cx + 1.55, 3.35, '"output.bias"\n[20480, 20520)', color=GREEN,
            fontsize=11, ha="center", va="bottom")
    leader(ax, (ob_cx + 1.15, 3.28), (ob_cx, h))

    # overall label under the whole raw-bytes region
    ax.text(x_raw0 + w_raw / 2, -0.28, "raw tensor bytes, back to back, no holes",
            color="black", fontsize=12, ha="center", va="top")

    ax.text(x_end, -0.85, "(schematic; widths not to byte scale)", color=GRAY,
            fontsize=9.5, ha="right", va="top", style="italic")

    ax.set_xlim(-0.3, x_end + 0.3)
    ax.set_ylim(-1.25, 3.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-safetensors-layout")


# =========================================================================== #
# saving-loading.md :: Checkpointing a Training Run section                  #
# =========================================================================== #

def fig_checkpoint_contents():
    """One checkpoint file with five labelled compartments, each paired by a
    horizontal arrow to what it restores on resume: model state_dict to
    weights, optimizer state to momentum/moments, RNG state to data order and
    dropout, step to schedule position, config to architecture.  Rows are
    aligned left to right so the five arrows run level and never cross."""
    rows = [
        ("model\nstate_dict", "weights"),
        ("optimizer\nstate", "momentum /\nmoments"),
        ("RNG\nstate", "data order and\ndropout"),
        ("step", "schedule\nposition"),
        ("config", "architecture"),
    ]
    n = len(rows)
    row_h, gap = 0.85, 0.30
    total_h = n * row_h + (n - 1) * gap
    y_top = total_h / 2

    fig, ax = plt.subplots(figsize=(9.4, 5.2))

    w_left, w_right = 2.3, 2.3
    x_left, x_right = -3.1, 3.1

    centers = [y_top - row_h / 2 - i * (row_h + gap) for i in range(n)]

    for cy, (left_lab, right_lab) in zip(centers, rows):
        box(ax, x_left, cy, w_left, row_h, left_lab, facecolor=BLUE, ls="-",
            fontsize=11.5)
        box(ax, x_right, cy, w_right, row_h, right_lab, facecolor=GREEN, ls="-",
            fontsize=11.5)
        fl.arrow(ax, (x_left + w_left / 2 + 0.08, cy),
                 (x_right - w_right / 2 - 0.08, cy), color=GRAY, lw=1.6, mut=13)

    # wrapper box around the checkpoint-file compartments
    pad = 0.28
    wrap = FancyBboxPatch((x_left - w_left / 2 - pad, -total_h / 2 - pad),
                          w_left + 2 * pad, total_h + 2 * pad,
                          boxstyle="round,pad=0.0,rounding_size=0.10",
                          facecolor="none", edgecolor="black", lw=1.6, ls="--",
                          zorder=1)
    ax.add_patch(wrap)

    ax.text(x_left, y_top + pad + 0.42, "checkpoint file", color="black",
            fontsize=14, ha="center", va="bottom", weight="bold")
    ax.text(x_right, y_top + pad + 0.42, "restored on resume", color="black",
            fontsize=14, ha="center", va="bottom", weight="bold")

    ax.set_xlim(x_left - w_left / 2 - pad - 0.3, x_right + w_right / 2 + 0.3)
    ax.set_ylim(-total_h / 2 - pad - 0.3, y_top + pad + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-checkpoint-contents")


# =========================================================================== #
# reproducibility-inspection.md :: Hooks: Looking Inside section             #
# =========================================================================== #

def fig_hooks():
    """The ``__call__`` wrapper as a pipeline: input flows through pre-hooks,
    then ``forward``, then (post-forward) hooks, to the output.  The two hook
    stages are drawn dashed/orange, visually distinct from the solid blue
    ``forward`` box, and a side arrow from the hooks box leads to an observer
    that can capture, check, or modify -- all without editing ``forward``."""
    fig, ax = plt.subplots(figsize=(10.6, 4.6))

    stages = [
        ("input", LIGHT, "-", "black"),
        ("pre-hooks", ORANGE, "--", ORANGE),
        ("forward", BLUE, "-", BLUE),
        ("hooks", ORANGE, "--", ORANGE),
        ("output", LIGHT, "-", "black"),
    ]
    w, h = 1.85, 1.05
    xs = [0.0, 2.35, 4.70, 7.05, 9.40]
    y = 0.0

    for x, (label, fill, ls, edge) in zip(xs, stages):
        alpha_fill = 0.55 if fill is LIGHT else 0.20
        box(ax, x, y, w, h, label, facecolor=fill, edgecolor=edge, ls=ls,
            lw=1.8, fontsize=13)
        ax.patches[-1].set_alpha(alpha_fill)

    for x0, x1 in zip(xs[:-1], xs[1:]):
        fl.arrow(ax, (x0 + w / 2 + 0.06, y), (x1 - w / 2 - 0.06, y),
                 color=GRAY, lw=1.8, mut=14)

    # bracket over pre-hooks / forward / hooks labelled as the __call__ wrapper
    xb0, xb1 = xs[1] - w / 2, xs[3] + w / 2
    yb = y + h / 2 + 0.55
    ax.plot([xb0, xb1], [yb, yb], color="black", lw=1.3, zorder=2)
    ax.plot([xb0, xb0], [yb, y + h / 2 + 0.10], color="black", lw=1.3, zorder=2)
    ax.plot([xb1, xb1], [yb, y + h / 2 + 0.10], color="black", lw=1.3, zorder=2)
    ax.text((xb0 + xb1) / 2, yb + 0.14, r"$\mathtt{\_\_call\_\_}$ wrapper",
            color="black", fontsize=13, ha="center", va="bottom")

    # side arrow from the (post-forward) hooks box down to the observer note
    hx = xs[3]
    fl.arrow(ax, (hx, y - h / 2 - 0.08), (hx, y - h / 2 - 0.85), color=ORANGE,
             lw=1.8, mut=13)
    box(ax, hx, y - h / 2 - 1.55, 3.3, 0.95,
        "observer:\ncapture, check, modify", facecolor="white",
        edgecolor=GRAY, ls="--", lw=1.4, fontsize=11.5, color="black")

    ax.set_xlim(xs[0] - w / 2 - 0.3, xs[-1] + w / 2 + 0.3)
    ax.set_ylim(y - h / 2 - 2.25, yb + 0.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-hooks")


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

FIGURES = [
    fig_safetensors_layout,
    fig_checkpoint_contents,
    fig_hooks,
]


def main():
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
