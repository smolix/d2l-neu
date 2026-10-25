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
    """The safetensors file of the section's MLP as one horizontal byte strip:
    an 8-byte header length, the JSON header, then the raw bytes of the four
    tensors back to back in the order safetensors writes them (sorted by
    name).  The byte offsets under the data region are the real
    ``data_offsets`` of that file (float32, hidden 256 x 20, output 10 x 256),
    and the JSON excerpt quotes the real start of its header.  Widths
    are schematic, not to byte scale."""
    fig, ax = plt.subplots(figsize=(10.0, 3.9))
    T = fl.T
    y0, h = 0.0, 1.0

    # segment widths (schematic); everything abuts: the file has no padding
    w_len, w_hdr = 0.75, 5.0
    tensors = [  # name, width, start offset, end offset, accent
        ("hidden.bias", 0.55, 0, 1024, T.GREEN),
        ("hidden.weight", 3.0, 1024, 21504, T.BLUE),
        ("output.bias", 0.35, 21504, 21544, T.GREEN),
        ("output.weight", 1.9, 21544, 31784, T.BLUE),
    ]
    x_hdr0 = w_len
    x_raw0 = x_hdr0 + w_hdr
    x = x_raw0
    spans = []
    for name, w, a, b, acc in tensors:
        spans.append((name, x, w, a, b, acc))
        x += w
    x_end = x

    def rect(x0, w, face, edge):
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=face, edgecolor=edge,
                               lw=1.3, zorder=3))

    rect(0.0, w_len, T.PANEL, fl.INK)
    rect(x_hdr0, w_hdr, T.ORANGE.tint, fl.INK)
    for name, x0, w, a, b, acc in spans:
        rect(x0, w, acc.tint, fl.INK)

    # labels above the bar
    lx = w_len / 2
    ax.text(lx, 1.55, "8 bytes:\nheader\nlength", color=fl.INK, fontsize=12,
            ha="center", va="bottom")
    leader(ax, (lx, 1.5), (lx, h))
    ax.text(x_hdr0 + w_hdr / 2, 1.55, "JSON header", color=T.ORANGE.dark,
            fontsize=13, ha="center", va="bottom")
    leader(ax, (x_hdr0 + w_hdr / 2, 1.5), (x_hdr0 + w_hdr / 2, h))

    # JSON excerpt inside the header box: the real start of the header, which
    # lists the tensors in the order of their names
    hx = x_hdr0 + 0.12
    for k, line in enumerate(['{"hidden.bias": {"dtype": "F32",',
                              '   "shape": [256],',
                              '   "data_offsets": [0, 1024]},',
                              ' "hidden.weight": {...}, ...}']):
        ax.text(hx, y0 + 0.86 - 0.24 * k, line, color=fl.INK, fontsize=9.5,
                ha="left", va="center", family="monospace", zorder=4)

    # tensor names: inside the wide boxes, above (with a leader) the narrow ones
    for name, x0, w, a, b, acc in spans:
        cx = x0 + w / 2
        if w > 1.5:
            ax.text(cx, y0 + h / 2, name, color=acc.dark, fontsize=12,
                    ha="center", va="center", zorder=4)
    for name, dx, ty in (("hidden.bias", -0.15, 1.55), ("output.bias", 0.55, 1.55)):
        x0, w = [(s[1], s[2]) for s in spans if s[0] == name][0]
        cx = x0 + w / 2
        acc = [s[5] for s in spans if s[0] == name][0]
        ax.text(cx + dx, ty, name, color=acc.dark, fontsize=12, ha="center",
                va="bottom")
        leader(ax, (cx + dx, ty - 0.05), (cx, h))

    # real byte offsets (from the start of the data) under the boundaries
    for x0, off in ((spans[0][1], 0), (spans[1][1], 1024),
                    (spans[2][1], 21504), (x_end, 31784)):
        ax.plot([x0, x0], [y0, y0 - 0.14], color=fl.INK, lw=1.1, zorder=2)
        ax.text(x0, y0 - 0.2, f"{off:,}", color=fl.INK, fontsize=11.5,
                ha="center", va="top")
    ax.text((x_raw0 + x_end) / 2, -0.72,
            "tensor bytes, back to back (offsets in bytes)",
            color=fl.INK, fontsize=12, ha="center", va="top")
    ax.text(0.0, -0.72, "widths not to scale", color=fl.INK,
            fontsize=10.5, ha="left", va="top", style="italic")

    ax.set_xlim(-0.15, x_end + 0.35)
    ax.set_ylim(-1.15, 2.45)
    ax.set_aspect("equal")
    ax.axis("off")
    fl.save(fig, "bg-safetensors-layout")


# =========================================================================== #
# saving-loading.md :: Checkpointing a Training Run section                  #
# =========================================================================== #

def fig_checkpoint_contents():
    """One checkpoint file with five labelled compartments, each paired by a
    horizontal arrow to what it restores on resume: model state to
    weights, optimizer state to Adam's two running averages, RNG state to data order and
    dropout, step to schedule position, config to architecture.  Rows are
    aligned left to right so the five arrows run level and never cross."""
    rows = [
        ("model\nstate", "weights"),
        ("optimizer\nstate", "Adam's two\naverages"),
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
        # trio style: tint fill + base outline + ink text — never a
        # saturated slab with light text (guide §2.3)
        box(ax, x_left, cy, w_left, row_h, left_lab,
            facecolor=fl.T.BLUE.tint, edgecolor=fl.T.BLUE.base, lw=2.0,
            ls="-", fontsize=11.5)
        box(ax, x_right, cy, w_right, row_h, right_lab,
            facecolor=fl.T.GREEN.tint, edgecolor=fl.T.GREEN.base, lw=2.0,
            ls="-", fontsize=11.5)
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

    T = fl.T
    stages = [  # label, fill, line style, edge
        ("input", T.PANEL, "-", fl.INK),
        ("pre-hooks", T.ORANGE.tint, "--", T.ORANGE.base),
        ("forward", T.BLUE.tint, "-", T.BLUE.base),
        ("hooks", T.ORANGE.tint, "--", T.ORANGE.base),
        ("output", T.PANEL, "-", fl.INK),
    ]
    w, h = 1.85, 1.05
    xs = [0.0, 2.35, 4.70, 7.05, 9.40]
    y = 0.0

    for x, (label, fill, ls, edge) in zip(xs, stages):
        box(ax, x, y, w, h, label, facecolor=fill, edgecolor=edge, ls=ls,
            lw=1.8, fontsize=14)

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
        edgecolor=GRAY, ls="--", lw=1.4, fontsize=13, color="black")

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
