#!/usr/bin/env python3
"""Generate the illustrative figures for the Builders' Guide "GPUs, Devices,
and Memory" section (``chapter_builders-guide-v2/gpus-devices-memory.md``) in
the one shared house style defined in ``gen_mdl_figures.py``.

These replace the old hand-drawn ``img/copyto.svg`` and add three new
schematics for the GPU-memory / activation-checkpointing / async-queue
discussion.  The notebook carries no drawing code; the prose references these
generated files directly, exactly like the Mathematics-for-Deep-Learning
figures.

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_bg_memory_figures.py

All figures are written to ``img/bg-<id>.svg``.  The script is idempotent:
re-running overwrites byte-for-byte identical files.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# --------------------------------------------------------------------------- #
# Small chapter-local helpers (rounded device/tensor boxes).                   #
# --------------------------------------------------------------------------- #

def rbox(ax, x0, y0, w, h, color, alpha=0.10, lw=1.8, ls="-"):
    """A rounded rectangle outline + light fill; returns rim anchor points."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=lw, edgecolor=color, facecolor=color, alpha=alpha, ls=ls))
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=lw, edgecolor=color, facecolor="none", ls=ls))
    return dict(l=x0, r=x0 + w, b=y0, t=y0 + h, cx=x0 + w / 2, cy=y0 + h / 2,
                w=w, h=h)


def device_box(ax, x0, y0, w, h, title):
    """A device (GPU) frame: gray box with a black title at the top."""
    d = rbox(ax, x0, y0, w, h, GRAY, alpha=0.06, lw=2.0)
    ax.text(d["cx"], d["t"] - 0.34, title, ha="center", va="center",
            fontsize=14, fontweight="bold", color="black")
    return d


def tensor_box(ax, cx, cy, label, color, w=1.35, h=0.85, dashed=False):
    """A small tensor node: colored rounded box with a math-mode label."""
    b = rbox(ax, cx - w / 2, cy - h / 2, w, h, color, alpha=0.16, lw=1.8,
             ls="--" if dashed else "-")
    ax.text(cx, cy, label, ha="center", va="center", fontsize=13, color=color)
    return b


# =========================================================================== #
# 1. Copying a tensor between two devices, then computing on the result.      #
# =========================================================================== #

def fig_copyto():
    """House-style replacement for the old hand-drawn copyto.svg: X lives on
    GPU 0, Y lives on GPU 1; X.to(gpu(1)) makes a copy Z on GPU 1 (dashed
    outline marks it as a copy, not the original), and Y + Z is then computed
    entirely within GPU 1 -- the picture for "copy explicitly, then add"."""
    fig, ax = plt.subplots(figsize=(9.7, 4.6))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 5.4)
    ax.set_aspect("equal")
    ax.axis("off")

    gpu0 = device_box(ax, 0.4, 0.4, 5.1, 4.4, "GPU 0")
    gpu1 = device_box(ax, 6.9, 0.4, 5.6, 4.4, "GPU 1")

    # X and Z sit high (same height => a level copy arrow); Y sits low and to
    # the left, well clear of that arrow's path; the result sits lower still,
    # so none of the three connecting arrows cross a box that isn't its target.
    X = tensor_box(ax, gpu0["cx"], gpu0["cy"] + 0.90, r"$X$", BLUE)
    Z = tensor_box(ax, gpu1["cx"] + 1.50, gpu1["cy"] + 0.90, r"$Z$", BLUE,
                    dashed=True)
    Y = tensor_box(ax, gpu1["cx"] - 1.50, gpu1["cy"] - 0.30, r"$Y$", ORANGE)
    result = tensor_box(ax, gpu1["cx"], gpu1["b"] + 0.65, r"$Y+Z$", GREEN,
                         w=1.9)

    # the copy: X (GPU 0) -> Z (GPU 1), crossing the device boundary above Y;
    # its label sits centred ABOVE both device boxes, clear of every box
    fl.arrow(ax, (X["r"], X["cy"]), (Z["l"], Z["cy"]), color=BLUE, lw=2.2)
    fl.vlabel(ax, (6.45, 5.05),
              r"$Z = X.\mathrm{to}(\mathrm{gpu}(1))$", color=BLUE,
              fontsize=12.5)

    # the compute, entirely inside GPU 1: Y + Z
    fl.arrow(ax, (Y["cx"], Y["b"]), (result["l"] + 0.30, result["t"]),
             color=ORANGE, lw=1.8)
    fl.arrow(ax, (Z["cx"], Z["b"]), (result["r"] - 0.30, result["t"]),
             color=BLUE, lw=1.8)

    fl.save(fig, "bg-copyto")


# =========================================================================== #
# 2. The caching allocator: nvidia-smi >= reserved >= allocated.              #
# =========================================================================== #

def fig_allocator():
    """Three nested rounded rectangles showing the three numbers PyTorch and
    nvidia-smi report, from the outside in: what nvidia-smi shows (the driver
    allocation), memory_reserved() (PyTorch's cache, a subset), and
    memory_allocated() (live tensors, a further subset).  The gap between the
    middle and inner rectangles is annotated: those are freed blocks the
    allocator kept for reuse instead of returning to the driver."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6.3)
    ax.set_aspect("equal")
    ax.axis("off")

    outer = rbox(ax, 0.4, 0.4, 9.2, 5.5, GRAY, alpha=0.05, lw=2.0)
    middle = rbox(ax, 1.5, 1.1, 7.0, 4.0, BLUE, alpha=0.08, lw=1.8)
    inner = rbox(ax, 2.9, 1.85, 4.2, 2.6, GREEN, alpha=0.14, lw=1.8)

    # one line per ring, each confined to that ring's OWN band (between its
    # box's top edge and the next box in's top edge), so the three titles
    # never share a row
    ax.text(outer["cx"], outer["t"] - 0.40,
            "what nvidia-smi shows (reserved from the driver)",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color="black")

    ax.text(middle["cx"], middle["t"] - 0.35,
            r"memory_reserved(): PyTorch cache", ha="center", va="center",
            fontsize=12, color=BLUE)

    ax.text(inner["cx"], inner["t"] - 0.40,
            r"memory_allocated():", ha="center", va="center",
            fontsize=12, color=GREEN)
    ax.text(inner["cx"], inner["t"] - 0.78,
            "live tensors", ha="center", va="center",
            fontsize=12, color=GREEN)

    # gap annotation: a point strictly between the inner and middle rims,
    # clear of both titles, with a leader line out to the label
    gx, gy = inner["r"] + 0.45, inner["b"] + 0.35
    ax.plot([gx], [gy], "o", color=GRAY, ms=5, zorder=5)
    lx, ly = 11.9, gy
    ax.plot([gx, lx - 0.05], [gy, ly], color=GRAY, lw=1.0)
    ax.text(lx, ly, "freed blocks kept\nfor reuse", ha="left", va="center",
            fontsize=11, color="black")

    fl.save(fig, "bg-allocator")


# =========================================================================== #
# 3. Activation checkpointing: store-all vs. checkpoint-and-recompute.        #
# =========================================================================== #

def fig_activation_checkpoint():
    """Two aligned rows over the same N-layer axis.  (a) standard: every
    activation is stored on the forward pass (filled dots) and read once on
    the backward pass.  (b) checkpointed: only segment-boundary activations
    survive (sparse filled dots); the rest (hollow) are regenerated by a short
    local recompute during backward.  Annotated with the memory and compute
    trade the section states: O(N) vs roughly O(sqrt(N)) memory, 1x vs about
    1.3x compute."""
    n = 9
    xs = 1.2 + 1.15 * np.arange(n)
    x0, x1 = xs[0] - 0.9, xs[-1] + 0.9
    checkpoints = {0, 3, 6}  # segment boundaries: every 3rd activation kept

    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    ax.set_xlim(-2.9, 12.9)
    ax.set_ylim(-0.4, 6.8)
    ax.axis("off")

    # Each row stacks, bottom to top: layer labels, backward pass, dots,
    # (row b only) recompute arcs, forward pass, title -- one clear band per
    # element, with enough vertical margin that neither row's title nor its
    # forward-pass band can reach into the other row's space.
    def row(y, filled_idx, title, mem, comp, recompute=False):
        ax.text(-2.7, y + 1.55, title, ha="left", va="center",
                fontsize=12.5, fontweight="bold", color="black")
        # forward pass (black, left to right) above the dots
        fl.arrow(ax, (x0, y + 0.85), (x1, y + 0.85), color="black", lw=1.6)
        ax.text((x0 + x1) / 2, y + 1.10, "forward", ha="center", va="center",
                fontsize=10.5, color="black")
        # backward pass (blue, right to left) below the dots
        fl.arrow(ax, (x1, y - 0.55), (x0, y - 0.55), color=BLUE, lw=1.6)
        ax.text((x0 + x1) / 2, y - 0.80, "backward", ha="center", va="center",
                fontsize=10.5, color=BLUE)
        # activations along the layer axis
        for i, x in enumerate(xs):
            if i in filled_idx:
                ax.plot(x, y, "o", color=GREEN, ms=11, zorder=5,
                        mec="black", mew=0.8)
            else:
                ax.plot(x, y, "o", color="white", ms=11, zorder=5,
                        mec=GRAY, mew=1.4)
        ax.text(xs[0], y - 1.15, "layer 1", ha="center", va="center",
                fontsize=10, color="black")
        ax.text(xs[-1], y - 1.15, f"layer {n}", ha="center", va="center",
                fontsize=10, color="black")
        # right-margin memory / compute annotation, vertically centred
        # between the forward and backward bands
        ax.text(x1 + 0.55, y + 0.28, mem, ha="left", va="center",
                fontsize=12, color="black")
        ax.text(x1 + 0.55, y - 0.28, comp, ha="left", va="center",
                fontsize=12, color="black")

        if recompute:
            # small recompute arcs just above the dots: each hollow
            # activation is regenerated from its nearest preceding
            # checkpoint, all comfortably below the forward-pass band
            # rad is scaled inversely with the chord length so EVERY arc
            # (one-layer or two-layer hop) peaks at the same modest height,
            # clear of the "recompute" label above
            peak_h = 0.30
            last_ckpt = None
            for i in range(n):
                if i in filled_idx:
                    last_ckpt = i
                    continue
                p0, p1 = xs[last_ckpt], xs[i]
                rad = -peak_h / (0.5 * (p1 - p0))
                arc = FancyArrowPatch((p0, y + 0.10), (p1, y + 0.10),
                                       connectionstyle=f"arc3,rad={rad}",
                                       arrowstyle="-|>", mutation_scale=10,
                                       color=ORANGE, lw=1.3, zorder=4)
                ax.add_patch(arc)
            ax.text((x0 + x1) / 2, y + 0.65, "recompute", ha="center",
                    va="center", fontsize=10.5, color=ORANGE)

    y_a, y_b = 4.65, 1.15
    row(y_a, set(range(n)), "(a) store all",
        r"memory $O(N)$", "compute $1\\times$")
    row(y_b, checkpoints, "(b) checkpoint + recompute",
        r"memory $O(\sqrt{N})$", r"compute $\approx 1.3\times$",
        recompute=True)

    fl.save(fig, "bg-activation-checkpoint")


# =========================================================================== #
# 4. Asynchronous execution: the CPU races ahead, then hits a sync point.     #
# =========================================================================== #

def fig_async_queue():
    """Two timelines, time flowing right.  Python (CPU) issues four kernels
    k1..k4 in quick, evenly-spaced ticks and keeps going; the GPU executes them
    back to back, each one taking longer than the gap between issues, so the
    backlog (the dotted connectors) grows.  Later, `loss.item()` forces a
    synchronization point: the CPU blocks (hatched segment) until the GPU
    drains its queue, marked by the dashed barrier, and only then resumes."""
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.set_xlim(-2.6, 14.6)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    y_cpu, y_gpu = 4.0, 1.3

    ax.text(-2.4, y_cpu, "Python (CPU)", ha="left", va="center",
            fontsize=13, fontweight="bold", color="black")
    ax.text(-2.4, y_gpu, "GPU", ha="left", va="center",
            fontsize=13, fontweight="bold", color="black")

    # issue ticks k1..k4: quick, evenly spaced, and far enough apart that
    # their labels don't crowd each other
    issue_x = [1.0, 1.9, 2.8, 3.7]
    # GPU executes serially: each block starts when both issued AND the GPU
    # is free, and runs longer than the gap between issues (backlog grows)
    starts = [1.0, 3.2, 5.6, 8.2]
    ends = [3.2, 5.6, 8.2, 11.0]

    for k, (ix, s, e) in enumerate(zip(issue_x, starts, ends), start=1):
        ax.plot([ix, ix], [y_cpu, y_cpu + 0.35], color="black", lw=1.6)
        ax.text(ix, y_cpu + 0.55, f"$k_{k}$", ha="center", va="center",
                fontsize=12, color="black")
        rbox(ax, s, y_gpu - 0.30, e - s, 0.60, BLUE, alpha=0.22, lw=1.6)
        ax.text((s + e) / 2, y_gpu, f"$k_{k}$", ha="center", va="center",
                fontsize=11, color="black")
        # dotted connector: issued here, actually runs there (visualizes the
        # growing queue as later kernels start further behind their issue)
        ax.plot([ix, s], [y_cpu - 0.05, y_gpu + 0.30], ":", color=GRAY, lw=1.1)

    # CPU keeps going after issuing all four kernels, well clear of the
    # sync-point marker further on; left-aligned starting well past the last
    # issue tick so it never crowds the k4 label above it
    fl.arrow(ax, (issue_x[-1], y_cpu), (8.0, y_cpu), color="black", lw=1.8)
    ax.text(4.3, y_cpu + 0.32, "Python keeps issuing work",
            ha="left", va="center", fontsize=10.5, color="black")

    # the sync point: loss.item()
    sync_call_x, resume_x = 9.2, 11.0
    ax.plot(sync_call_x, y_cpu, "D", color=ORANGE, ms=8, zorder=5)
    ax.text(sync_call_x, y_cpu + 0.38, r"loss.item()", ha="center",
            va="center", fontsize=11.5, color=ORANGE)

    # blocked: the CPU cannot proceed until the GPU drains the queue
    ax.plot([sync_call_x, resume_x], [y_cpu, y_cpu], color=GRAY, lw=6,
            alpha=0.45, solid_capstyle="butt", zorder=2)
    ax.text((sync_call_x + resume_x) / 2, y_cpu - 0.42, "blocked",
            ha="center", va="center", fontsize=10.5, color="black")

    # dashed barrier: CPU resumes exactly when the last kernel finishes
    ax.plot([resume_x, resume_x], [y_gpu + 0.32, y_cpu], "--", color=GRAY,
            lw=1.4)
    ax.text(resume_x + 0.30, 2.65, "synchronize:\nCPU waits", ha="left",
            va="center", fontsize=11, color="black")

    # CPU resumes
    fl.arrow(ax, (resume_x, y_cpu), (13.6, y_cpu), color="black", lw=1.8)
    ax.text(12.6, y_cpu + 0.32, "resumes", ha="center", va="center",
            fontsize=10.5, color="black")

    # time axis
    fl.arrow(ax, (-2.4, 0.35), (14.2, 0.35), color=GRAY, lw=1.2)
    ax.text(14.2, 0.62, "time", ha="right", va="center", fontsize=11,
            color="black")

    fl.save(fig, "bg-async-queue")


FIGURES = [fig_copyto, fig_allocator, fig_activation_checkpoint,
           fig_async_queue]


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks every other chapter's figures).
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
