#!/usr/bin/env python3
"""Generate the illustrative figures for ``chapter_computational-performance``
(ch. 13) in the one shared house style defined in ``gen_mdl_figures.py``.

The figures follow the approved rebuild proposal
(``reviews/comp-perf-chapter-review-and-proposal-2026-07-20.md`` §6.1).  The
hardware numbers (bandwidth/latency/energy ladders, float formats, spec-table
magnitudes) come from the June-2026-verified ``smolix/mlss-efficiency`` deck
dossiers; the PCIe topology is this repo's own build box (``nvidia-smi topo``).

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_perf_figures.py

All figures are written to ``img/mdl-perf-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
RED = fl.T.RED.base   # loss / gradient role (guide 4.3)
# Coloured *text* uses the accent's dark shade (guide 4.2/7.3: base for
# strokes and fills, dark for labels) so it stays >= 4.5:1 on white.
BLUE_T, ORANGE_T, GREEN_T = fl.T.BLUE.dark, fl.T.ORANGE.dark, fl.T.GREEN.dark
RED_T = fl.T.RED.dark

from matplotlib.patches import FancyBboxPatch, Rectangle


# --------------------------------------------------------------------------- #
# Local helpers                                                               #
# --------------------------------------------------------------------------- #

def _box(ax, cx, cy, w, h, text, color, fontsize=13, ls="-", lw=1.6,
         fc=None, tc="black"):
    """A rounded box (faint fill + coloured edge) with centred text."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        fc=fc if fc is not None else color, ec=color if fc is None else color,
        lw=lw, linestyle=ls, alpha=1.0, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=tc, zorder=4)


def _lane_kernel(ax, x0, w, y, h, color, label=None, fontsize=12, alpha=1.0,
                 ec="none", tc="white"):
    """One solid kernel/segment box on a timeline lane."""
    ax.add_patch(Rectangle((x0, y - h / 2), w, h, fc=color, ec=ec, lw=0.8,
                           alpha=alpha, zorder=3))
    if label:
        ax.text(x0 + w / 2, y, label, ha="center", va="center",
                fontsize=fontsize, color=tc, zorder=4)


# =========================================================================== #
# 13.1  The Performance Model                                                 #
# =========================================================================== #

def fig_roofline():
    """The roofline model, drawn with the build box's own numbers: an RTX 4090
    has 165 TFLOP/s (bf16 tensor cores) and 1.008 TB/s of memory bandwidth, so
    the ridge sits near 164 FLOP/byte.  Log-log; the sloped segment is the
    bandwidth bound, the flat segment the compute roof.  The three points are
    the worked examples of the text: x + 1 in bf16 (1 FLOP per 4 bytes), a
    width-2048 linear layer applied to one token (about 1 FLOP/byte), and the
    same layer applied to a training minibatch of 8,192 tokens (about 910)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    peak = 165.0          # TFLOP/s, bf16 dense
    bw = 1.008            # TB/s -> y[TF] = bw * x[FLOP/byte]
    ridge = peak / bw     # ~164 FLOP/byte

    x = np.logspace(np.log10(0.1), np.log10(16384), 400)
    y = np.minimum(peak, bw * x)
    ax.loglog(x, y, color=BLUE, lw=2.6, zorder=3)

    # Ridge marker (dashed drop from the knee, label to its right).
    ax.plot([ridge], [peak], "o", color=BLUE, ms=7, zorder=4)
    ax.plot([ridge, ridge], [0.05, peak], color=LIGHT, lw=1.2, ls="--",
            zorder=1)
    ax.text(205, 0.065, r"ridge $\approx 164$ FLOP/byte",
            ha="left", va="bottom", fontsize=13, color="black")

    # Slope and roof annotations, offset up-left of (and parallel to) the wall.
    ax.text(0.55, 2.2, "memory bandwidth\n$\\beta$ = 1.0 TB/s",
            ha="left", va="bottom", fontsize=13, color=BLUE_T, rotation=33.0,
            rotation_mode="anchor")
    ax.text(1100, peak * 1.45, "compute roof $P$ = 165 TFLOP/s (bf16)",
            ha="center", va="bottom", fontsize=13, color=BLUE_T)

    # The worked examples of the text.
    ax.plot([0.25], [bw * 0.25], "o", color=ORANGE, ms=8, zorder=5)
    ax.text(0.36, 0.2, "$x + 1$ in bf16 (0.25)",
            ha="left", va="center", fontsize=13, color=ORANGE_T)
    ax.plot([1.0], [bw * 1.0], "o", color=ORANGE, ms=8, zorder=5)
    ax.text(1.45, 0.8, "linear layer, 1 token (1)",
            ha="left", va="center", fontsize=13, color=ORANGE_T)
    ax.plot([910], [peak], "o", color=GREEN, ms=8, zorder=5)
    ax.text(910, peak * 0.5, "linear layer,\n8,192 tokens (910)",
            ha="center", va="top", fontsize=13, color=GREEN_T)

    # Regime labels in the open corners.
    ax.text(30, 1.4, "bandwidth-bound", ha="center", va="center",
            fontsize=14, color="black", style="italic")
    ax.text(2600, 8, "compute-bound", ha="center", va="center",
            fontsize=14, color="black", style="italic")

    ax.set_xlim(0.1, 16384)
    ax.set_ylim(0.05, 500)
    # Plain-number ticks read faster than powers of ten.
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1,000", "10,000"])
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("arithmetic intensity (FLOP/byte)", fontsize=13)
    ax.set_ylabel("attainable TFLOP/s", fontsize=13)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")
    ax.tick_params(colors="black", labelsize=12)
    fl.save(fig, "mdl-perf-roofline")


def fig_async_timeline():
    """Asynchronous dispatch: Python enqueues kernels and returns immediately;
    the GPU works through the queue behind it.  A timer without a barrier
    stops when the *enqueueing* is done; only a barrier waits for the work."""
    fig, ax = plt.subplots(figsize=(8.8, 3.4))

    y_cpu, y_gpu, h = 2.0, 0.8, 0.52
    T = 10.4

    # Lane guides.
    for y, name in ((y_cpu, "Python\n(CPU)"), (y_gpu, "GPU")):
        ax.plot([0.0, T], [y - h / 2 - 0.06, y - h / 2 - 0.06],
                color=LIGHT, lw=0.8, zorder=1)
        ax.text(-0.25, y, name, ha="right", va="center", fontsize=13,
                color="black")

    # CPU enqueues three kernels quickly, then is free.
    launches = [(0.3, r"$k_1$"), (1.1, r"$k_2$"), (1.9, r"$k_3$")]
    starts = [0.9, 3.9, 6.9]  # GPU kernel starts (back to back, long)
    for (lx, lab), gx in zip(launches, starts):
        _lane_kernel(ax, lx, 0.6, y_cpu, h, GRAY, label=lab, fontsize=11)
        ax.annotate("", xy=(gx + 0.12, y_gpu + h / 2 + 0.02),
                    xytext=(lx + 0.32, y_cpu - h / 2 - 0.02),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1,
                                    linestyle="--"))
    for gx, lab in zip(starts, (r"$k_1$", r"$k_2$", r"$k_3$")):
        _lane_kernel(ax, gx, 3.0, y_gpu, h, BLUE, label=lab, fontsize=12)

    # The stop without a barrier vs the stop after a barrier.
    t_naive = launches[-1][0] + 0.75
    t_sync = starts[-1] + 3.0
    ax.axvline(t_naive, color=ORANGE, lw=1.8, ls="--", zorder=2)
    ax.text(t_naive + 0.2, y_cpu + h / 2 + 0.42,
            "timer without a barrier:\nwork only enqueued",
            ha="left", va="bottom", fontsize=13, color=ORANGE_T)
    ax.axvline(t_sync, color=GREEN, lw=1.8, ls="--", zorder=2)
    ax.text(t_sync - 0.15, y_cpu + h / 2 + 0.42,
            "barrier returns here:\nwork done",
            ha="right", va="bottom", fontsize=13, color=GREEN_T)

    ax.annotate("", xy=(T, y_gpu - h / 2 - 0.5), xytext=(0, y_gpu - h / 2 - 0.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(T / 2, y_gpu - h / 2 - 0.62, "time", ha="center", va="top",
            fontsize=13, color="black")

    fl.clean_axes(ax, lim=((-1.6, T + 0.2), (-0.6, 4.1)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-perf-async-timeline")


def fig_regimes():
    """The three-regime diagnostic: where a training step's wall-clock time
    goes.  One bar per regime, decomposed into compute-bound kernels,
    bandwidth-bound kernels, and idle gaps waiting for Python; each regime
    names its standard fix."""
    fig, ax = plt.subplots(figsize=(9.2, 3.1))

    rows = [
        ("compute-bound",   [(0.82, BLUE), (0.13, ORANGE), (0.05, GRAY)],
         "lower precision, more devices"),
        ("bandwidth-bound", [(0.18, BLUE), (0.72, ORANGE), (0.10, GRAY)],
         "fuse kernels: fewer trips to memory"),
        ("overhead-bound",  [(0.10, BLUE), (0.12, ORANGE), (0.78, GRAY)],
         "capture the graph, replay without Python"),
    ]
    h = 0.52
    for i, (name, segs, fix) in enumerate(rows):
        y = 2.0 - i * 1.0
        x0 = 0.0
        for frac, color in segs:
            _lane_kernel(ax, x0, frac * 5.6, y, h, color)
            x0 += frac * 5.6
        ax.text(-0.15, y, name, ha="right", va="center", fontsize=14,
                color="black")
        ax.text(5.75, y, fix, ha="left", va="center", fontsize=13,
                color="black")

    # Legend row (manual, below the bars).
    lx = 0.0
    for color, lab in ((BLUE, "compute-bound kernels"),
                       (ORANGE, "bandwidth-bound kernels"),
                       (GRAY, "idle gaps")):
        _lane_kernel(ax, lx, 0.28, -0.95, 0.28, color)
        ax.text(lx + 0.38, -0.95, lab, ha="left", va="center", fontsize=12,
                color="black")
        lx += 0.38 + 0.14 * len(lab) + 0.5

    fl.clean_axes(ax, lim=((-2.4, 10.6), (-1.3, 2.55)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-perf-regimes")


# =========================================================================== #
# 13.2  Hardware                                                              #
# =========================================================================== #

def _log_ladder(name, rows, xlabel, xlim, ticks, highlight=(),
                figsize=None):
    """A horizontal log-scale bar 'ladder'.  ``rows`` = [(label, value,
    pretty-value)], drawn bottom-to-top in the given order (smallest first
    reads as climbing the ladder).  ``ticks`` = [(value, label)] for the
    x-axis, in readable units rather than powers of ten; a faint gridline
    marks every decade.  ``highlight`` = indices into rows drawn in ORANGE."""
    n = len(rows)
    if figsize is None:
        figsize = (7.8, 0.52 * n + 1.3)
    fig, ax = plt.subplots(figsize=figsize)
    x0 = xlim[0]
    for i, (label, value, pretty) in enumerate(rows):
        color = ORANGE if i in highlight else BLUE
        ax.barh(i, value - x0, left=x0, height=0.62, color=color, zorder=3)
        # Fixed on-screen gap after the bar end (independent of how many
        # decades the axis spans); a white backing keeps gridlines off it.
        ax.annotate(pretty, xy=(value, i), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=13, color="black", zorder=4,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white",
                              ec="none"))
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows], fontsize=13, color="black")
    ax.set_xlabel(xlabel, fontsize=13, color="black")
    decades = 10.0 ** np.arange(np.floor(np.log10(xlim[0])),
                                np.ceil(np.log10(xlim[1])) + 1)
    ax.set_xticks([d for d in decades if xlim[0] <= d <= xlim[1]],
                  minor=True)
    ax.set_xticks([v for v, _ in ticks])
    ax.set_xticklabels([t for _, t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(axis="x", which="both", color=LIGHT, lw=0.7, zorder=0)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(colors="black", labelsize=12)
    ax.tick_params(axis="x", which="minor", bottom=False)
    fl.save(fig, name)


def fig_bandwidth_ladder():
    """Bandwidth of the data paths a training job uses, late 2026 (links per
    direction): every chip boundary costs roughly an order of magnitude.
    The build machine's own two paths, its GDDR6X and its PCIe 4.0 x16
    link, are highlighted.  Sources: NVIDIA RTX 4090 (Ada whitepaper v2.02),
    DGX B200 and NVLink pages (8 TB/s; 1.8 TB/s per GPU = 900 GB/s each
    way), PCI-SIG (PCIe 4.0/6.0 x16 per direction)."""
    rows = [
        (r"NVMe SSD (PCIe 5.0 $\times$4)", 14, "14 GB/s"),
        (r"PCIe 4.0 $\times$16 (RTX 4090)", 32, "32 GB/s"),
        ("800G network card", 100, "100 GB/s"),
        (r"PCIe 6.0 $\times$16", 128, "128 GB/s"),
        ("DDR5 (one CPU socket)", 500, "0.5 TB/s"),
        ("NVLink 5 (per GPU)", 900, "0.9 TB/s"),
        ("GDDR6X (RTX 4090)", 1008, "1.0 TB/s"),
        ("HBM3e (B200)", 8000, "8 TB/s"),
    ]
    _log_ladder("mdl-perf-bandwidth-ladder", rows,
                "bandwidth (log scale; links: one direction)", (5, 40000),
                [(10, "10 GB/s"), (100, "100 GB/s"), (1000, "1 TB/s"),
                 (10000, "10 TB/s")], highlight=(1, 6))


def fig_latency_ladder():
    """Latencies from an L1 hit to a WAN round trip: about eight orders of
    magnitude.  GPU cache and memory latencies are the RTX 4090
    microbenchmarks of Luo et al. (2024) at boost clock; the CUDA launch cost
    is NVIDIA's CUDA Graph guide (1-5 us); the highlighted rung is the host
    cost of one eager framework operation dispatched from Python, the cost
    the overhead regime is made of."""
    rows = [
        ("CPU L1 hit", 1, "1 ns"),
        ("GPU L1 / shared memory", 15, "~15 ns"),
        ("DDR5 access (CPU)", 90, "~90 ns"),
        ("GPU L2 hit", 100, "~100 ns"),
        ("GPU memory access", 250, "200–300 ns"),
        ("CUDA kernel launch", 3e3, "1–5 µs"),
        ("RDMA network round trip", 4e3, "2–5 µs"),
        ("eager operation from Python", 1e4, "5–15 µs"),
        ("NVMe read", 7e4, "50–100 µs"),
        ("WAN round trip", 6e7, "~60 ms"),
    ]
    _log_ladder("mdl-perf-latency-ladder", rows,
                "latency (log scale)", (0.6, 1.6e9),
                [(1, "1 ns"), (1e3, "1 µs"), (1e6, "1 ms"), (1e9, "1 s")],
                highlight=(7,))


def fig_energy_ladder():
    """Energy per operation (Horowitz, ISSCC 2014, 45 nm): arithmetic is
    nearly free; fetching operands from DRAM is the budget -- one 64-bit DRAM
    read costs about 500 fp32 multiplies.  Larger SRAMs cost more per access
    because the bits travel farther."""
    rows = [
        ("int8 add", 0.03, "0.03 pJ"),
        ("int8 multiply", 0.2, "0.2 pJ"),
        ("fp16 multiply", 1.1, "1.1 pJ"),
        ("fp32 multiply", 3.7, "3.7 pJ"),
        ("64-bit SRAM read (8 KB array)", 10, "10 pJ"),
        ("64-bit SRAM read (1 MB array)", 100, "100 pJ"),
        ("64-bit DRAM read", 2000, "1,300–2,600 pJ"),
    ]
    _log_ladder("mdl-perf-energy-ladder", rows,
                "energy per operation (pJ, log scale)", (0.02, 30000),
                [(0.1, "0.1"), (1, "1"), (10, "10"), (100, "100"),
                 (1000, "1,000"), (10000, "10,000")], highlight=(6,))


def fig_memory_hierarchy():
    """The memory hierarchy as a pyramid: each level downward holds orders of
    magnitude more bytes and delivers orders of magnitude less bandwidth."""
    fig, ax = plt.subplots(figsize=(8.8, 4.6))

    levels = [
        ("on-die SRAM (registers, L1/L2)", "~0.1 GB · tens of TB/s", 0.30),
        ("GPU memory (HBM / GDDR)", "24–288 GB · 1–8 TB/s", 0.24),
        ("host DRAM", "0.1–2 TB · ~0.5 TB/s", 0.19),
        ("NVMe SSD", "several TB · ~14 GB/s", 0.14),
        ("network / object store", "unbounded · ~0.1 TB/s per NIC", 0.09),
    ]
    n = len(levels)
    h = 0.82
    top_hw, bot_hw = 2.25, 4.85  # half-widths of the pyramid at top/bottom
    for i, (name, stats, alpha) in enumerate(levels):
        y1 = -i * h          # top edge of this level
        y0 = y1 - h
        f1 = i / n
        f0 = (i + 1) / n
        w1 = top_hw + (bot_hw - top_hw) * f1
        w0 = top_hw + (bot_hw - top_hw) * f0
        ax.add_patch(plt.Polygon(
            [(-w1, y1), (w1, y1), (w0, y0), (-w0, y0)],
            fc=BLUE, ec="white", lw=1.5, alpha=alpha, zorder=2))
        yc = (y0 + y1) / 2
        ax.text(0, yc + 0.13, name, ha="center", va="center", fontsize=13.5,
                color="black", zorder=4)
        ax.text(0, yc - 0.26, stats, ha="center", va="center", fontsize=12,
                color="black", zorder=4)

    # Side arrows: faster toward the cores, bigger away from them.
    ax.annotate("", xy=(-5.3, -0.1), xytext=(-5.3, -n * h + 0.1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.text(-5.5, -n * h / 2, "closer to the cores: faster",
            ha="center", va="center", fontsize=13, color="black", rotation=90)
    ax.annotate("", xy=(5.3, -n * h + 0.1), xytext=(5.3, -0.1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.text(5.5, -n * h / 2, "farther away: bigger",
            ha="center", va="center", fontsize=13, color="black", rotation=-90)

    fl.clean_axes(ax, lim=((-6.05, 6.05), (-n * h - 0.15, 0.15)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-perf-memory-hierarchy")


def fig_shoreline():
    """The shoreline argument: compute units fill the die's *area*, while I/O
    pads live only on its *perimeter*.  Doubling the die side quadruples the
    compute but only doubles the beachfront — bytes-per-FLOP falls with every
    generation."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    def die(cx, cy, side, cells, pads_per_side):
        half = side / 2
        # Interior: compute cells.
        cell = side / cells
        for i in range(cells):
            for j in range(cells):
                ax.add_patch(Rectangle(
                    (cx - half + i * cell + 0.05 * cell,
                     cy - half + j * cell + 0.05 * cell),
                    0.9 * cell, 0.9 * cell, fc=BLUE, alpha=0.30, ec="none",
                    zorder=2))
        # Perimeter: I/O pads.
        pad = side / (2.2 * pads_per_side)
        for k in range(pads_per_side):
            t = -half + (k + 0.5) * side / pads_per_side
            for (px, py) in ((cx + t, cy - half), (cx + t, cy + half),
                             (cx - half, cy + t), (cx + half, cy + t)):
                ax.add_patch(Rectangle((px - pad / 2, py - pad / 2), pad, pad,
                                       fc=ORANGE, ec="none", zorder=4))
        ax.add_patch(Rectangle((cx - half, cy - half), side, side, fc="none",
                               ec="black", lw=1.2, zorder=3))

    die(-2.55, -0.10, 1.7, 4, 4)   # bottom edges aligned at y = -0.95
    die(1.85, 0.75, 3.4, 8, 8)

    ax.text(-2.55, -1.35, "die side $n$", ha="center", va="top", fontsize=14,
            color="black")
    ax.text(-2.55, -1.82, "compute $\\propto n^2$ (16 units)\nI/O $\\propto n$ (16 pads)",
            ha="center", va="top", fontsize=12.5, color="black")
    ax.text(1.85, -1.35, "die side $2n$", ha="center", va="top", fontsize=14,
            color="black")
    ax.text(1.85, -1.82, "compute $\\times 4$ (64 units)\nI/O only $\\times 2$ (32 pads)",
            ha="center", va="top", fontsize=12.5, color="black")

    # Legend, in the empty corner above the small die.
    ax.add_patch(Rectangle((-3.7, 2.25), 0.3, 0.3, fc=BLUE, alpha=0.30,
                           ec="none"))
    ax.text(-3.27, 2.40, "compute (area)", ha="left", va="center",
            fontsize=12.5, color="black")
    ax.add_patch(Rectangle((-3.7, 1.70), 0.3, 0.3, fc=ORANGE, ec="none"))
    ax.text(-3.27, 1.85, "I/O pads (perimeter)", ha="left", va="center",
            fontsize=12.5, color="black")

    fl.clean_axes(ax, lim=((-3.85, 3.75), (-2.85, 2.75)), hide=True)
    fl.save(fig, "mdl-perf-shoreline")


def fig_float_formats():
    """The floating-point format ladder, bit-for-bit to scale: sign, exponent,
    mantissa.  fp32/tf32/bf16 share the 8-bit exponent (same dynamic range);
    every halving of width doubles peak FLOP/s *and* halves the bytes moved."""
    fig, ax = plt.subplots(figsize=(9.4, 4.9))

    formats = [
        ("fp32", 8, 23, "4 B"),
        ("tf32", 8, 10, "(stored in 32 bits)"),
        ("bf16", 8, 7, "2 B"),
        ("fp16", 5, 10, "2 B"),
        ("fp8 (E4M3)", 4, 3, "1 B"),
        ("fp8 (E5M2)", 5, 2, "1 B"),
        ("fp4 (E2M1)", 2, 1, "0.5 B"),
    ]
    bw, bh, dy = 0.24, 0.52, 0.92
    for i, (name, e, m, note) in enumerate(formats):
        y = -i * dy
        ax.text(-0.25, y, name, ha="right", va="center", fontsize=13.5,
                color="black")
        x = 0.0
        for nbits, color in ((1, GRAY), (e, ORANGE), (m, BLUE)):
            for _ in range(nbits):
                ax.add_patch(Rectangle((x, y - bh / 2), bw * 0.92, bh,
                                       fc=color, ec="white", lw=0.6, zorder=3))
                x += bw
        ax.text(x + 0.12, y, note, ha="left", va="center", fontsize=12,
                color="black")

    # Shared-exponent bracket over the three 8-bit-exponent formats.
    xr = (1 + 8) * bw + 0.05
    ax.plot([xr, xr], [-2 * dy - 0.35, 0.55], color="black", lw=1.0, ls="--",
            zorder=2)
    ax.text(xr + 0.12, 0.62,
            "same 8-bit exponent $\\Rightarrow$ same range",
            ha="left", va="bottom", fontsize=12.5, color="black")

    # Legend (bottom right, out of every row's way).
    lx = 4.4
    for color, lab in ((GRAY, "sign"), (ORANGE, "exponent"),
                       (BLUE, "mantissa")):
        ax.add_patch(Rectangle((lx, -5.85), 0.3, 0.3, fc=color, ec="none"))
        ax.text(lx + 0.4, -5.7, lab, ha="left", va="center", fontsize=12.5,
                color="black")
        lx += 0.4 + 0.16 * len(lab) + 0.45

    fl.clean_axes(ax, lim=((-1.8, 9.2), (-6.15, 1.35)), hide=True, equal=False)
    fl.save(fig, "mdl-perf-float-formats")


def fig_pcie_topology():
    """This book's own build box (``nvidia-smi topo``): four RTX 4090s in
    pairs behind two PCIe host bridges, no P2P and no NVLink — every
    GPU-to-GPU byte is staged through host memory."""
    fig, ax = plt.subplots(figsize=(9.0, 4.25))

    _box(ax, 5.0, 4.55, 4.6, 0.95, "CPU + host DRAM", GRAY, fontsize=14,
         fc=fl.T.PANEL)
    for cx, lab in ((2.7, "PCIe host bridge"), (7.3, "PCIe host bridge")):
        _box(ax, cx, 2.95, 2.75, 0.72, lab, GRAY, fontsize=12.5, fc=fl.T.PANEL)
    gpus = [(1.30, "GPU 0"), (3.70, "GPU 1"), (6.30, "GPU 2"), (8.70, "GPU 3")]
    for cx, lab in gpus:
        _box(ax, cx, 1.15, 1.95, 1.0, f"{lab}\nRTX 4090 · 24 GB", BLUE,
             fontsize=12, fc=fl.T.BLUE.tint, tc="black")

    # Gray fabric links for the GPUs off the highlighted path (the orange
    # path below draws the other four links itself, so no gray stub can
    # peek out past its arrowhead).
    ax.plot([3.70, 2.7], [1.67, 2.57], color=GRAY, lw=1.6, zorder=1)
    ax.plot([8.70, 7.3], [1.67, 2.57], color=GRAY, lw=1.6, zorder=1)
    ax.text(1.62, 2.08, r"PCIe 4.0 $\times$16", ha="right", va="center",
            fontsize=12, color="black")

    # The staged path GPU0 -> host -> GPU2 overlays the gray links exactly
    # (the opaque bridge boxes hide the pass-through segments).
    seg = [(1.30, 1.67), (2.7, 2.57), (2.7, 3.33), (4.4, 4.05)]
    seg2 = [(5.6, 4.05), (7.3, 3.33), (7.3, 2.57), (6.30, 1.67)]
    ax.plot([p[0] for p in seg], [p[1] for p in seg], color=ORANGE, lw=2.6,
            zorder=2)
    ax.plot([p[0] for p in seg2[:-1]], [p[1] for p in seg2[:-1]], color=ORANGE,
            lw=2.6, zorder=2)
    # Filled data-flow head whose tip ends exactly on GPU 2's top edge.
    ax.annotate("", xy=seg2[-1], xytext=seg2[-2],
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.6,
                                shrinkA=0, shrinkB=0, mutation_scale=18),
                zorder=2)

    fl.clean_axes(ax, lim=((0.2, 9.8), (0.5, 5.15)), hide=True, equal=False)
    fl.save(fig, "mdl-perf-pcie-topology")


# =========================================================================== #
# 13.3  Compute Graphs and Compilation                                        #
# =========================================================================== #

def fig_compute_graph():
    """The compute graph of a two-layer network with a loss: autograd already
    builds this DAG; eager execution walks it one kernel launch at a time,
    while a compiler sees the whole graph at once."""
    fig, ax = plt.subplots(figsize=(9.6, 3.4))

    OPF, DATF = fl.T.BLUE.tint, fl.T.PANEL
    ops = [(2.0, r"matmul"), (3.35, r"add"), (4.7, r"relu"),
           (6.05, r"matmul"), (7.4, r"add"), (8.75, r"loss")]
    y_main, y_par = 1.55, 0.35
    _box(ax, 0.65, y_main, 0.85, 0.55, r"$\mathbf{X}$", GRAY, fc=DATF,
         fontsize=13)
    for cx, lab in ops[:-1]:
        _box(ax, cx, y_main, 0.95, 0.55, lab, BLUE, fc=OPF, fontsize=12.5)
    _box(ax, ops[-1][0], y_main, 0.95, 0.55, ops[-1][1], RED, fc=fl.T.RED.tint,
         fontsize=12.5)
    params = [(2.0, r"$\mathbf{W}_1$"), (3.35, r"$\mathbf{b}_1$"),
              (6.05, r"$\mathbf{W}_2$"), (7.4, r"$\mathbf{b}_2$"),
              (8.75, r"$\mathbf{y}$")]
    for cx, lab in params:
        _box(ax, cx, y_par, 0.85, 0.5, lab, GRAY, fc=DATF, fontsize=13)
        ax.annotate("", xy=(cx, y_main - 0.30), xytext=(cx, y_par + 0.27),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3))
    # Forward chain arrows.
    chain_x = [0.65] + [c for c, _ in ops]
    for a, b in zip(chain_x[:-1], chain_x[1:]):
        ax.annotate("", xy=(b - 0.50, y_main), xytext=(a + 0.50, y_main),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    # One backward path above: up from the top edge of `loss`, across, and
    # down onto the top edge of X (rounded rectilinear, solid-tip arrow).
    top = y_main + 0.275 + 0.02          # box half-height + FancyBbox pad
    ax.annotate("", xy=(0.65, top), xytext=(ops[-1][0], top),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                                linestyle="--", shrinkA=0, shrinkB=0,
                                mutation_scale=14,
                                connectionstyle="arc,angleA=90,armA=26,"
                                "angleB=90,armB=26,rad=10"))
    ax.text(4.7, y_main + 1.1, "backward pass: the same graph, walked in "
            "reverse", ha="center", va="center", fontsize=13, color=RED_T)
    ax.text(4.7, -0.42, "eager execution: every node is a separate kernel "
            "launch and a round trip to memory",
            ha="center", va="center", fontsize=12.5, color="black")

    fl.clean_axes(ax, lim=((0.0, 9.6), (-0.75, 2.9)), hide=True, equal=False)
    fl.save(fig, "mdl-perf-compute-graph")


def fig_compile_pipelines():
    """Graph capture in one picture: the first call runs the function on
    placeholders that record each tensor operation, a compiler turns the
    recorded graph into kernels, and later calls with the same shapes and
    dtypes reuse them (a note under the kernels).  Two dashed paths mark the
    limits: a branch on a tensor's value cannot be recorded, and a new shape
    or dtype sends the call back to capture."""
    fig, ax = plt.subplots(figsize=(10.4, 3.3))
    bw, bh, y = 1.62, 0.78, 1.30
    stages = ["Python\nfunction", "capture\n(placeholders)", "graph",
              "compiler\n(fuse, schedule)", "compiled\nkernels"]
    xs = np.linspace(1.0, 9.4, len(stages))
    for x, s in zip(xs, stages):
        _box(ax, x, y, bw, bh, s, BLUE, fc=fl.T.BLUE.tint, fontsize=12.5)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(b - bw / 2 - 0.03, y), xytext=(a + bw / 2 + 0.03, y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
    top = bh / 2 + 0.04
    # Above: a branch on a value cannot become part of the graph.
    ax.annotate("", xy=(xs[0], y + top), xytext=(xs[1], y + top),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                linestyle="--", shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=0.5"))
    ax.text(xs[1] + 0.25, y + 0.95,
            "branch on a tensor's value: cannot be recorded",
            ha="left", va="center", fontsize=12.5, color=ORANGE_T)
    # Below: a new input shape or dtype sends the call back to capture; a
    # note under the kernels names the other path, reuse.
    x_from = xs[-1] - 0.45
    ax.annotate("", xy=(xs[1], y - top), xytext=(x_from, y - top),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                linestyle="--", shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=-0.12"))
    ax.text((xs[1] + x_from) / 2, y - 0.98,
            "new shape or dtype: capture again",
            ha="center", va="center", fontsize=12.5, color=ORANGE_T)
    ax.text(xs[-1] + 0.12, y - top - 0.36,
            "same shapes and\ndtypes: reuse",
            ha="center", va="center", fontsize=12.5, color="black",
            linespacing=1.1)
    fl.clean_axes(ax, lim=((0.05, 10.35), (0.10, 2.45)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-perf-compile-pipelines")


def fig_fusion():
    """Memory traffic of the elementwise chain
    0.5 x (1 + tanh(0.8 (x + 0.04 x^3))) on a 64 MB array.  Eager execution
    runs eight kernels that read ten arrays and write eight (18 x 64 MB);
    the fused kernel reads x once and writes the result once (2 x 64 MB)."""
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ops = [("mul", 1), ("pow", 1), ("mul", 1), ("add", 2), ("mul", 1),
           ("tanh", 1), ("add", 1), ("mul", 2)]
    bw, bh = 0.86, 0.56
    xs = 1.55 + 1.08 * np.arange(len(ops))
    READ, WRITE = BLUE, ORANGE

    def hbm(y):
        ax.add_patch(Rectangle((0.95, y - 0.17), 9.0, 0.34, fc=fl.T.PANEL,
                               ec=GRAY, lw=1.0, zorder=2))
        ax.text(0.85, y, "GPU memory", ha="right", va="center",
                fontsize=12.5, color="black")

    def link(x, y0, y1, color):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                    mutation_scale=11, shrinkA=0, shrinkB=0))

    # Eager: one kernel per operation, every result written back.
    yk, ym = 3.95, 2.85
    hbm(ym)
    ax.text(0.85, yk, "eager", ha="right", va="center", fontsize=13,
            color="black")
    for x, (name, nin) in zip(xs, ops):
        _box(ax, x, yk, bw, bh, name, BLUE, fc=fl.T.BLUE.tint, fontsize=12.5)
        reads = [x - 0.24] if nin == 1 else [x - 0.30, x - 0.14]
        for xr in reads:
            link(xr, ym + 0.17, yk - bh / 2 - 0.04, READ)
        link(x + 0.24, yk - bh / 2 - 0.04, ym + 0.17, WRITE)
    ax.text(5.45, ym - 0.52, "8 kernels: 10 reads + 8 writes of 64 MB "
            "= 1.15 GB moved", ha="center", va="center", fontsize=12.5,
            color="black")

    # Fused: one kernel, intermediates stay in registers.
    yk, ym = 1.35, 0.25
    hbm(ym)
    ax.text(0.85, yk, "fused", ha="right", va="center", fontsize=13,
            color="black")
    _box(ax, (xs[0] + xs[-1]) / 2, yk, xs[-1] - xs[0] + bw, bh,
         "mul · pow · mul · add · mul · tanh · add · mul   (in registers)",
         GREEN, fc=fl.T.GREEN.tint, fontsize=12.5)
    link(xs[0] - 0.24, ym + 0.17, yk - bh / 2 - 0.04, READ)
    link(xs[-1] + 0.24, yk - bh / 2 - 0.04, ym + 0.17, WRITE)
    ax.text(5.45, ym - 0.52, "1 kernel: 1 read + 1 write of 64 MB "
            "= 128 MB moved", ha="center", va="center", fontsize=12.5,
            color="black")
    # Key for the arrow colours.
    ax.text(0.85, 3.55, "read", ha="right", va="center", fontsize=12.5,
            color=BLUE_T)
    ax.text(0.85, 3.25, "write", ha="right", va="center", fontsize=12.5,
            color=ORANGE_T)
    fl.clean_axes(ax, lim=((-0.75, 10.05), (-0.45, 4.35)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-perf-fusion")


# =========================================================================== #
# 13.4  Memory and Precision                                                  #
# =========================================================================== #

def fig_memory_anatomy():
    """Where a training step's memory goes, over the step timeline: parameters
    and optimizer states are constant; activations grow through the forward
    pass and shrink through the backward pass as they are consumed; gradients
    accumulate during backward.  Peak memory lands early in backward."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    t = np.linspace(0, 9.5, 400)
    t_f, t_b, t_u = 4.0, 8.0, 9.5   # forward / backward / update boundaries
    params = np.full_like(t, 1.0)
    states = np.full_like(t, 2.0)
    acts = np.where(t < t_f, 2.5 * t / t_f,
                    np.where(t < t_b, 2.5 * (t_b - t) / (t_b - t_f), 0.0))
    grads = np.where(t < t_f, 0.0,
                     np.where(t < t_b, 1.0 * (t - t_f) / (t_b - t_f), 1.0))
    ax.stackplot(t, params, states, grads, acts,
                 colors=[GRAY, LIGHT, ORANGE, BLUE],
                 alpha=0.85, lw=0)

    # Peak marker: max of the total.
    total = params + states + grads + acts
    ti = int(np.argmax(total))
    ax.plot([0, t[ti]], [total[ti], total[ti]], color="black", lw=1.1,
            ls="--", zorder=4)
    ax.plot([t[ti]], [total[ti]], "o", color="black", ms=5, zorder=5)
    # Label to the right of the apex, above the falling activation edge.
    ax.text(t[ti] + 0.25, total[ti] + 0.12,
            "peak: start of the backward pass",
            ha="left", va="bottom", fontsize=13, color="black")

    # Band labels, with the bytes each band holds for N parameters.
    ax.text(2.0, 0.5, "weights: $4N$", ha="center", va="center",
            fontsize=13, color="black")
    ax.text(2.0, 2.0, "Adam's moments $\\mathbf{m}$, $\\mathbf{v}$: $8N$",
            ha="center", va="center", fontsize=13, color="black")
    ax.text(6.6, 3.32, "gradients: $4N$", ha="center", va="center",
            fontsize=13, color="black")
    ax.text(2.7, 3.85, "activations", ha="center", va="center", fontsize=13,
            color="black")

    # Phase separators + labels.
    for x in (t_f, t_b):
        ax.axvline(x, color="white", lw=1.4, zorder=3)
    for x0, x1, lab in ((0, t_f, "forward"), (t_f, t_b, "backward"),
                        (t_b, t_u, "update")):
        ax.text((x0 + x1) / 2, -0.42, lab, ha="center", va="top", fontsize=13,
                color="black")

    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 6.6)
    ax.set_ylabel("memory (bytes, in units of $4N$)", fontsize=12.5)
    ax.set_xticks([])
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(colors="black", labelsize=11)
    fl.save(fig, "mdl-perf-memory-anatomy")


# =========================================================================== #
# 13.5  Multi-GPU from First Principles                                       #
# =========================================================================== #

def fig_splitting():
    """Three ways to split a model across two devices: replicate it and split
    the data; split the layer stack; or split every layer's width."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))

    L, W = 4, 6          # layers x width cells
    c0, c1 = BLUE, GREEN

    def grid(ax, x0, y0, colors, cw=0.42, ch=0.42):
        for i in range(L):          # i = layer row (top row = layer 1)
            for j in range(W):
                ax.add_patch(Rectangle(
                    (x0 + j * cw, y0 - i * ch), cw * 0.9, ch * 0.9,
                    fc=colors(i, j), alpha=0.55, ec="none"))

    # (a) data parallel: full replica per device, batch split.
    axa = axes[0]
    grid(axa, 0.0, 1.6, lambda i, j: c0)
    grid(axa, 3.2, 1.6, lambda i, j: c1)
    # Inline legend: the two colours are the two devices (panels b, c reuse
    # them), so name them once under the replicas.
    for xc, col, b, dev in ((1.25, c0, "batch$_1$", "GPU 0"),
                            (4.45, c1, "batch$_2$", "GPU 1")):
        axa.text(xc, 2.82, b, ha="center", va="bottom", fontsize=12.5,
                 color="black")
        axa.annotate("", xy=(xc, 2.06), xytext=(xc, 2.74),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.6,
                                     shrinkA=0, shrinkB=0))
        axa.text(xc, 0.18, dev, ha="center", va="top", fontsize=12.5,
                 color=BLUE_T if col == c0 else GREEN_T)
    axa.text(2.85, -0.65, "data parallel:\nreplicate the model,\nsplit the batch",
             ha="center", va="top", fontsize=12.5, color="black")
    fl.clean_axes(axa, lim=((-0.4, 6.1), (-2.25, 3.35)), hide=True)

    # (b) pipeline parallel: split the layer stack.
    axb = axes[1]
    grid(axb, 1.4, 1.6, lambda i, j: c0 if i < 2 else c1)
    axb.text(0.95, 1.4, "layers 1–2", ha="right", va="center", fontsize=12,
             color=BLUE_T)
    axb.text(0.95, 0.55, "layers 3–4", ha="right", va="center", fontsize=12,
             color=GREEN_T)
    axb.text(2.65, -0.65, "pipeline parallel:\nsplit the layer stack",
             ha="center", va="top", fontsize=12.5, color="black")
    fl.clean_axes(axb, lim=((-1.7, 4.6), (-2.25, 3.35)), hide=True)

    # (c) tensor parallel: split every layer's width.
    axc = axes[2]
    grid(axc, 1.4, 1.6, lambda i, j: c0 if j < W // 2 else c1)
    axc.annotate("", xy=(2.62, 2.30), xytext=(1.45, 2.30),
                 arrowprops=dict(arrowstyle="-", color=c0, lw=2.4))
    axc.annotate("", xy=(3.90, 2.30), xytext=(2.72, 2.30),
                 arrowprops=dict(arrowstyle="-", color=c1, lw=2.4))
    axc.text(2.65, 2.52, "each layer's units, halved", ha="center",
             va="bottom", fontsize=12, color="black")
    axc.text(2.65, -0.65, "tensor parallel:\nsplit every layer's width",
             ha="center", va="top", fontsize=12.5, color="black")
    fl.clean_axes(axc, lim=((-0.5, 5.8), (-2.25, 3.35)), hide=True)

    fl.save(fig, "mdl-perf-splitting")


def fig_data_parallel():
    """One data-parallel step on two GPUs: split the minibatch, run identical
    forward/backward on each replica, sum the gradients with an allreduce,
    apply the same update everywhere."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    _box(ax, 4.2, 4.35, 3.6, 0.7, "minibatch ($B$ examples)", GRAY,
         fc=fl.T.PANEL, fontsize=13)
    for cx, lab, half in ((2.0, "GPU 0", "$B/2$"), (6.4, "GPU 1", "$B/2$")):
        _box(ax, cx, 2.55, 3.3, 1.5,
             f"{lab}: full parameter copy\nforward + backward on {half}\n"
             r"$\rightarrow$ local gradients", BLUE, fc=fl.T.BLUE.tint,
             fontsize=12, tc="black")
    ax.annotate("", xy=(2.0, 3.34), xytext=(3.5, 4.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(6.4, 3.34), xytext=(4.9, 4.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    _box(ax, 4.2, 0.75, 4.2, 0.7, "allreduce: sum the gradients", ORANGE,
         fc=fl.T.ORANGE.tint, fontsize=13)
    ax.annotate("", xy=(3.2, 1.14), xytext=(2.0, 1.76),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(5.2, 1.14), xytext=(6.4, 1.76),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    # Summed gradients flow back up: each arc swings outside its replica
    # and lands on the box FLANK, incoming at 30 degrees (quadratic Bezier;
    # the control point fixes the terminal tangent).
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import FancyArrowPatch
    for tail, tip, sgn in [((2.1, 0.72), (0.35, 2.55), -1),
                           ((6.3, 0.72), (8.05, 2.55), +1)]:
        ctrl = (tip[0] + sgn * 1.6 * 0.866, tip[1] - 1.6 * 0.5)
        arc = MplPath([tail, ctrl, tip],
                      [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax.add_patch(FancyArrowPatch(path=arc, arrowstyle="->",
                                     mutation_scale=14, ls="--",
                                     color=ORANGE, lw=1.6, fc=ORANGE,
                                     zorder=2))
    ax.text(4.2, 0.0, "every device applies the identical update "
            r"$\rightarrow$ replicas stay in sync",
            ha="center", va="center", fontsize=12.5, color="black")

    fl.clean_axes(ax, lim=((-0.3, 8.7), (-0.4, 4.95)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-perf-data-parallel")


def fig_ring_allreduce():
    """Ring allreduce on four GPUs, computed step by step: reduce-scatter
    accumulates each chunk around the ring, then all-gather distributes the
    finished chunks.  Cell numbers count how many of the four contributions a
    chunk holds; a full chunk (4) is complete."""
    k = 4
    # Simulate: counts[gpu][chunk] = number of contributions summed in.
    # Start: every GPU holds its own full vector, 1 contribution per chunk.
    counts = np.ones((k, k), dtype=int)
    rs_states = [counts.copy()]
    # Reduce-scatter: at step s, GPU g sends chunk (g - s) mod k to GPU g+1.
    work = counts.copy()
    for s in range(k - 1):
        new = work.copy()
        for g in range(k):
            c = (g - s) % k
            new[(g + 1) % k, c] += work[g, c]
        # The sender no longer owns a current copy of that chunk; for the
        # count picture we keep the receiver's accumulation (standard
        # presentation).
        work = new
        rs_states.append(work.copy())
    ag_states = []
    done = work.copy()
    for s in range(k - 1):
        new = done.copy()
        for g in range(k):
            c = (g + 1 - s) % k
            if done[g, c] == k:
                new[(g + 1) % k, c] = k
        done = new
        ag_states.append(done.copy())

    fig, axes = plt.subplots(2, 4, figsize=(10.0, 4.1))
    fig.subplots_adjust(hspace=0.02)

    def draw_state(ax, M, title, prev=None):
        cs = 0.6
        for g in range(k):
            for c in range(k):
                v = M[g, c]
                color = GREEN if v == k else BLUE
                ax.add_patch(Rectangle((c * cs, -g * cs), cs * 0.92,
                                       cs * 0.92, fc=color,
                                       alpha=0.15 + 0.75 * (v / k) * 0.8,
                                       ec="none"))
                # Heavy outline = the chunk this GPU received in this step.
                if prev is not None and v != prev[g, c]:
                    ax.add_patch(Rectangle((c * cs + 0.025, -g * cs + 0.025),
                                           cs * 0.92 - 0.05, cs * 0.92 - 0.05,
                                           fc="none", ec=fl.T.INK, lw=2.0,
                                           zorder=3))
                ax.text(c * cs + cs * 0.46, -g * cs + cs * 0.46, str(v),
                        ha="center", va="center", fontsize=11.5,
                        color="black")
        ax.text(2 * cs - cs / 2, 1.0, title, ha="center", va="center",
                fontsize=12.5, color="black")
        fl.clean_axes(ax, lim=((-1.25, 2.8), (-1.95, 1.35)), hide=True)

    for i, M in enumerate(rs_states):
        draw_state(axes[0, i], M,
                   "start" if i == 0 else f"reduce-scatter, step {i}",
                   prev=None if i == 0 else rs_states[i - 1])
    for g in range(k):
        axes[0, 0].text(-0.18, -g * 0.6 + 0.28, f"GPU {g}", ha="right",
                        va="center", fontsize=12, color="black")
    ag_prev = [rs_states[-1]] + ag_states[:-1]
    for i, M in enumerate(ag_states):
        draw_state(axes[1, i], M, f"all-gather, step {i + 1}",
                   prev=ag_prev[i])
    for g in range(k):
        axes[1, 0].text(-0.18, -g * 0.6 + 0.28, f"GPU {g}", ha="right",
                        va="center", fontsize=12, color="black")
    # Last bottom panel: the key for the outline, the ring itself (who sends
    # to whom), then the byte accounting (inline fractions: stacked ones
    # shrink to illegibility at book width).
    axl = axes[1, 3]
    axl.add_patch(Rectangle((-0.75, 1.12 - 0.165), 0.33, 0.33,
                            fc="white", ec=fl.T.INK, lw=2.0))
    axl.text(-0.3, 1.12, "received this step", ha="left",
             va="center", fontsize=12.5, color="black")
    # The ring: GPU g sends to GPU g+1, clockwise from the top.
    from matplotlib.patches import Circle, FancyArrowPatch
    cx, cy, R, r = -0.16, -0.13, 0.72, 0.22
    ang = [np.pi / 2 - g * np.pi / 2 for g in range(k)]
    for g, a in enumerate(ang):
        x, y = cx + R * np.cos(a), cy + R * np.sin(a)
        axl.add_patch(Circle((x, y), r, fc=fl.T.BLUE.tint, ec=BLUE, lw=1.4,
                             zorder=3))
        axl.text(x, y, str(g), ha="center", va="center", fontsize=12,
                 color="black", zorder=4)
    gap = (r + 0.04) / R            # leave the node circles clear
    for g in range(k):
        a0, a1 = ang[g] - gap, ang[g] - np.pi / 2 + gap
        axl.add_patch(FancyArrowPatch(
            (cx + R * np.cos(a0), cy + R * np.sin(a0)),
            (cx + R * np.cos(a1), cy + R * np.sin(a1)),
            connectionstyle="arc3,rad=-0.22", arrowstyle="-|>",
            mutation_scale=12, color=fl.T.INK, lw=1.3, zorder=2))
    axl.text(cx + R + r + 0.16, cy, "GPU $g$ sends to\nGPU $(g{+}1)$ mod 4",
             ha="left", va="center", fontsize=12, color="black")
    for y, txt in ((-1.31, r"per phase: $(k-1)N/k$ bytes"),
                   (-1.76, r"total $2(k-1)N/k \leq 2N$")):
        axl.text(-0.75, y, txt, ha="left", va="center", fontsize=12.5,
                 color="black")
    fl.clean_axes(axl, lim=((-1.25, 2.8), (-1.95, 1.35)), hide=True)

    fl.save(fig, "mdl-perf-ring-allreduce")


# =========================================================================== #
# 13.6  Multi-GPU in Practice                                                 #
# =========================================================================== #

def fig_ddp_overlap():
    """What DDP buys: gradient bucketing lets allreduce overlap the rest of
    the backward pass instead of waiting for all of it."""
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.70, 3.15],
                                          "hspace": 0.05})

    h = 0.5
    layers = [(0.0, 1.5, "layer 4"), (1.5, 3.0, "layer 3"),
              (3.0, 4.5, "layer 2"), (4.5, 6.0, "layer 1")]

    def lanes(ax, comm, title, total, ybot):
        for (x0, x1, lab) in layers:
            _lane_kernel(ax, x0 + 0.03, (x1 - x0) - 0.06, 1.0, h, BLUE,
                         label=lab, fontsize=12.5)
        for (x0, x1, lab) in comm:
            _lane_kernel(ax, x0 + 0.03, (x1 - x0) - 0.06, 0.0, h, ORANGE,
                         label=lab, fontsize=12.5, tc="black")
        ax.text(-0.25, 1.0, "backward", ha="right", va="center",
                fontsize=12.5, color="black")
        ax.text(-0.25, 0.0, "allreduce", ha="right", va="center",
                fontsize=12.5, color="black")
        ax.text(0.0, 1.78, title, ha="left", va="center", fontsize=13,
                color="black")
        ax.plot([total, total], [-0.45, 1.45], color="black", lw=1.2, ls="--")
        fl.clean_axes(ax, lim=((-1.5, 10.6), (ybot, 2.15)), hide=True,
                      equal=False)

    lanes(axes[0], [(6.0, 10.0, "all gradients at once")],
          "no overlap: wait for the whole backward, then communicate", 10.0,
          -0.55)
    lanes(axes[1], [(3.0, 5.6, "bucket 1"), (6.0, 8.0, "bucket 2")],
          "bucketing: communication overlaps the rest of the backward",
          8.0, -1.0)
    # Faint copy of the upper panel's finish line, so "time saved" spans two
    # visible markers.
    axes[1].plot([10.0, 10.0], [-0.45, 1.45], color=GRAY, lw=1.0,
                 ls=(0, (3, 3)))
    axes[1].annotate("", xy=(10.0, -0.35), xytext=(8.0, -0.35),
                     arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6,
                                     shrinkA=0, shrinkB=0))
    axes[1].text(9.0, -0.62, "time saved", ha="center", va="top",
                 fontsize=12.5, color=GREEN_T)

    fl.save(fig, "mdl-perf-ddp-overlap")


def fig_fsdp_lifecycle():
    """The FSDP lifecycle of one block: parameters live sharded (P/k per
    rank); the full block is materialized by an all-gather only while it
    computes, then freed; gradients leave via reduce-scatter, so each rank
    keeps only its shard."""
    fig, (ax, axm) = plt.subplots(2, 1, figsize=(9.2, 3.3), sharex=True,
                                  gridspec_kw={"height_ratios": [0.6, 1.0],
                                               "hspace": 0.12})

    # Phase widths leave every 12.5-pt label clear padding in its box.
    events = [
        (0.0, 1.55, "all-gather\nweights", ORANGE, "black"),
        (1.55, 3.3, "forward", BLUE, "white"),
        (3.3, 4.1, "free", GRAY, "white"),
        (4.4, 5.95, "all-gather\nweights", ORANGE, "black"),
        (5.95, 7.45, "backward", BLUE, "white"),
        (7.45, 8.25, "free", GRAY, "white"),
        (8.25, 10.5, "reduce-scatter\ngrads", ORANGE, "black"),
    ]
    for x0, x1, lab, color, tc in events:
        _lane_kernel(ax, x0 + 0.03, (x1 - x0) - 0.06, 0.5, 0.62, color,
                     label=lab, fontsize=12.5, tc=tc)
    fl.clean_axes(ax, lim=((-0.3, 10.7), (0.1, 0.9)), hide=True,
                  equal=False)

    # Memory trace of this block's parameter bytes on one rank.
    t = np.linspace(0, 10.5, 500)
    full, shard = 1.0, 1.0 / 4
    mem = np.full_like(t, shard)
    for (g0, g1, f0, f1) in ((0.0, 1.55, 3.3, 4.1), (4.4, 5.95, 7.45, 8.25)):
        rise = (t >= g0) & (t < g1)
        mem[rise] = shard + (full - shard) * (t[rise] - g0) / (g1 - g0)
        mem[(t >= g1) & (t < f0)] = full
        fall = (t >= f0) & (t < f1)
        mem[fall] = full - (full - shard) * (t[fall] - f0) / (f1 - f0)
    axm.plot(t, mem, color=BLUE, lw=2.2, zorder=3)
    # Reference levels stop at the trace's end so their labels sit clear.
    for level, lab in ((shard, r"sharded: $1/k$ of the block"),
                       (full, "gathered: the whole block")):
        axm.plot([-0.3, 10.5], [level, level], color=GRAY, lw=1.0, ls="--",
                 zorder=2)
        axm.text(10.7, level, lab, ha="left", va="center", fontsize=12.5,
                 color="black")
    axm.set_ylim(0, 1.35)
    axm.set_xlim(-0.3, 10.7)
    axm.set_ylabel("this block's\nweights on\none GPU", fontsize=12.5)
    axm.set_xlabel("time", fontsize=12.5, color="black", loc="right")
    axm.set_xticks([])
    axm.set_yticks([])
    for s in ("left", "bottom"):
        axm.spines[s].set_color("black")
    fig.subplots_adjust(right=0.80)
    fl.save(fig, "mdl-perf-fsdp-lifecycle")


# --------------------------------------------------------------------------- #
# Registry + main                                                             #
# --------------------------------------------------------------------------- #

FIGURES = [
    fig_roofline,
    fig_async_timeline,
    fig_regimes,
    fig_bandwidth_ladder,
    fig_latency_ladder,
    fig_energy_ladder,
    fig_memory_hierarchy,
    fig_shoreline,
    fig_float_formats,
    fig_pcie_topology,
    fig_compute_graph,
    fig_compile_pipelines,
    fig_fusion,
    fig_memory_anatomy,
    fig_splitting,
    fig_data_parallel,
    fig_ring_allreduce,
    fig_ddp_overlap,
    fig_fsdp_lifecycle,
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
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):36s} {size:>8,d} bytes")
    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
