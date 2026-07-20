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
    has ~165 TFLOP/s (bf16 tensor cores) and ~1.0 TB/s of memory bandwidth, so
    the ridge sits near 165 FLOP/byte.  Log-log; the sloped segment is the
    bandwidth wall, the flat segment the compute roof."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    peak = 165.0          # TFLOP/s, bf16 dense
    bw = 1.0              # TB/s -> y[TF] = bw * x[FLOP/byte]
    ridge = peak / bw     # 165 FLOP/byte

    x = np.logspace(np.log10(0.25), np.log10(16384), 400)
    y = np.minimum(peak, bw * x)
    ax.loglog(x, y, color=BLUE, lw=2.6, zorder=3)

    # Ridge marker (dashed drop from the knee, label to its right).
    ax.plot([ridge], [peak], "o", color=BLUE, ms=7, zorder=4)
    ax.plot([ridge, ridge], [0.1, peak], color=LIGHT, lw=1.2, ls="--",
            zorder=1)
    ax.text(210, 0.13, r"ridge $\approx 165$ FLOP/byte",
            ha="left", va="bottom", fontsize=13, color="black")

    # Slope and roof annotations, offset up-left of (and parallel to) the wall.
    ax.text(1.15, 7.0, "memory-bandwidth wall\nslope = 1.0 TB/s",
            ha="left", va="bottom", fontsize=13, color=BLUE, rotation=35.6,
            rotation_mode="anchor")
    ax.text(1100, peak * 1.45, "compute roof = 165 TFLOP/s (bf16)",
            ha="center", va="bottom", fontsize=13, color=BLUE)

    # Two workloads at opposite ends.
    ax.plot([1.0], [bw * 1.0], "o", color=ORANGE, ms=8, zorder=5)
    ax.text(1.45, 0.95, "elementwise op\n(intensity $\\approx 1$)",
            ha="left", va="center", fontsize=13, color=ORANGE)
    ax.plot([2048], [peak], "o", color=GREEN, ms=8, zorder=5)
    ax.text(2048, peak * 0.42, "large matmul",
            ha="center", va="top", fontsize=13, color=GREEN)

    # Regime labels in the open corners.
    ax.text(6.0, 0.35, "bandwidth-bound", ha="center", va="center",
            fontsize=14, color="black", style="italic")
    ax.text(2400, 18, "compute-bound", ha="center", va="center",
            fontsize=14, color="black", style="italic")

    ax.set_xlim(0.25, 16384)
    ax.set_ylim(0.1, 500)
    ax.set_xlabel("arithmetic intensity (FLOP/byte)", fontsize=13)
    ax.set_ylabel("attainable TFLOP/s", fontsize=13)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")
    ax.tick_params(colors="black", labelsize=11)
    fl.save(fig, "mdl-perf-roofline")


def fig_async_timeline():
    """Asynchronous dispatch: Python enqueues kernels and returns immediately;
    the GPU works through the queue behind it.  A naive timer stops when the
    *enqueueing* is done; only a synchronization point waits for the work."""
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

    # The naive timer stop vs the synchronized stop.
    t_naive = launches[-1][0] + 0.75
    t_sync = starts[-1] + 3.0
    ax.axvline(t_naive, color=ORANGE, lw=1.8, ls="--", zorder=2)
    ax.text(t_naive + 0.2, y_cpu + h / 2 + 0.42,
            "naive timer stops here:\nwork only enqueued",
            ha="left", va="bottom", fontsize=13, color=ORANGE)
    ax.axvline(t_sync, color=GREEN, lw=1.8, ls="--", zorder=2)
    ax.text(t_sync - 0.15, y_cpu + h / 2 + 0.42,
            "synchronize()\nreturns here",
            ha="right", va="bottom", fontsize=13, color=GREEN)

    ax.annotate("", xy=(T, y_gpu - h / 2 - 0.5), xytext=(0, y_gpu - h / 2 - 0.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(T / 2, y_gpu - h / 2 - 0.62, "time", ha="center", va="top",
            fontsize=13, color="black")

    fl.clean_axes(ax, lim=((-1.6, T + 0.2), (-0.6, 4.1)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-perf-async-timeline")


def fig_regimes():
    """The three-regime diagnostic: where a training step's wall-clock time
    actually goes.  One bar per regime, decomposed into arithmetic (busy tensor
    cores), memory traffic, and idle-waiting-for-Python; each regime names its
    fix and the section that teaches it."""
    fig, ax = plt.subplots(figsize=(9.2, 3.1))

    rows = [
        ("compute-bound",   [(0.82, BLUE), (0.13, ORANGE), (0.05, GRAY)],
         "the good regime — or drop to a cheaper format"),
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
    for color, lab in ((BLUE, "arithmetic"), (ORANGE, "memory traffic"),
                       (GRAY, "idle / launch overhead")):
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

def _log_ladder(name, rows, xlabel, xlim, highlight=None, figsize=None):
    """A horizontal log-scale bar 'ladder'.  ``rows`` = [(label, value,
    pretty-value)], drawn bottom-to-top in the given order (smallest first
    reads as climbing the ladder).  ``highlight`` = index into rows drawn in
    ORANGE."""
    n = len(rows)
    if figsize is None:
        figsize = (7.8, 0.52 * n + 1.3)
    fig, ax = plt.subplots(figsize=figsize)
    x0 = xlim[0]
    for i, (label, value, pretty) in enumerate(rows):
        color = ORANGE if i == highlight else BLUE
        ax.barh(i, value - x0, left=x0, height=0.62, color=color, zorder=3)
        ax.text(value * 1.18, i, pretty, ha="left", va="center",
                fontsize=12.5, color="black", zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows], fontsize=13, color="black")
    ax.set_xlabel(xlabel, fontsize=13, color="black")
    ax.grid(axis="x", color=LIGHT, lw=0.7, zorder=0)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(colors="black", labelsize=11)
    ax.tick_params(axis="x", which="minor", bottom=False)
    fl.save(fig, name)


def fig_bandwidth_ladder():
    """The 2026 bandwidth ladder (June-2026-verified MLSS numbers): every chip
    boundary costs roughly an order of magnitude.  Our own card (GDDR6X) sits
    two rungs below a B200's HBM3e."""
    rows = [
        ("NVMe SSD (seq. read)", 14, "14 GB/s"),
        ("800G NIC", 100, "100 GB/s"),
        (r"PCIe 6.0 $\times$16", 128, "128 GB/s"),
        ("DDR5 socket", 500, "500 GB/s"),
        ("GDDR6X (RTX 4090)", 1008, "1.0 TB/s"),
        ("NVLink 5 (per GPU)", 1800, "1.8 TB/s"),
        ("HBM3e (B200)", 8000, "8.0 TB/s"),
    ]
    _log_ladder("mdl-perf-bandwidth-ladder", rows,
                "bandwidth (GB/s, log scale)", (5, 40000), highlight=6)


def fig_latency_ladder():
    """The 2026 latency ladder, from on-chip SRAM to a WAN round trip — about
    eight orders of magnitude.  The kernel-launch rung is the one this chapter
    keeps meeting: it is why small ops cannot keep a GPU fed."""
    rows = [
        ("SRAM / L1", 1, "1 ns"),
        ("GPU L2", 75, "75 ns"),
        ("DDR5 access", 90, "90 ns"),
        ("HBM access", 300, "300 ns"),
        ("NVLink hop", 1.5e3, r"1.5 $\mu$s"),
        ("RDMA round trip", 5e3, r"5 $\mu$s"),
        ("CUDA kernel launch", 8e3, r"5–15 $\mu$s"),
        ("NVMe read", 5e4, r"50 $\mu$s"),
        ("WAN round trip", 6e7, "60 ms"),
    ]
    _log_ladder("mdl-perf-latency-ladder", rows,
                "latency (ns, log scale)", (0.6, 6e8), highlight=6)


def fig_energy_ladder():
    """Energy per operation (Horowitz ISSCC 2014 / Dally, 45 nm-class
    magnitudes): arithmetic is nearly free; fetching operands from DRAM is the
    budget — one 64-bit DRAM read costs about 500 fp32 multiplies."""
    rows = [
        ("int8 add", 0.03, "0.03 pJ"),
        ("int8 multiply", 0.2, "0.2 pJ"),
        ("fp16 multiply", 1.1, "1.1 pJ"),
        ("fp32 multiply", 3.7, "3.7 pJ"),
        ("SRAM access (8 KB)", 10, "10 pJ"),
        ("on-chip data move", 50, "50 pJ"),
        ("DRAM read (64-bit)", 2000, "~2,000 pJ"),
    ]
    _log_ladder("mdl-perf-energy-ladder", rows,
                "energy per operation (pJ, log scale)", (0.02, 30000),
                highlight=6)


def fig_memory_hierarchy():
    """The memory hierarchy as a pyramid: each level downward holds orders of
    magnitude more bytes and delivers orders of magnitude less bandwidth."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    levels = [
        ("on-die SRAM (registers, L1/L2)", "~0.1–0.24 GB · tens of TB/s", 0.30),
        ("GPU memory (HBM / GDDR)", "24–192 GB · 1–8 TB/s", 0.24),
        ("host DRAM", "0.1–2 TB · ~0.5 TB/s", 0.19),
        ("NVMe SSD", "several TB · ~14 GB/s", 0.14),
        ("network / object store", "unbounded · ~0.1 TB/s per NIC", 0.09),
    ]
    n = len(levels)
    h = 0.82
    top_hw, bot_hw = 1.55, 4.55  # half-widths of the pyramid at top/bottom
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
    ax.annotate("", xy=(-4.95, -0.1), xytext=(-4.95, -n * h + 0.1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.text(-5.15, -n * h / 2, "closer to the cores: faster",
            ha="center", va="center", fontsize=13, color="black", rotation=90)
    ax.annotate("", xy=(4.95, -n * h + 0.1), xytext=(4.95, -0.1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.text(5.15, -n * h / 2, "farther away: bigger",
            ha="center", va="center", fontsize=13, color="black", rotation=-90)

    fl.clean_axes(ax, lim=((-5.7, 5.7), (-n * h - 0.15, 0.15)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-perf-memory-hierarchy")


def fig_shoreline():
    """The shoreline argument: compute units fill the die's *area*, while I/O
    pads live only on its *perimeter*.  Doubling the die side quadruples the
    compute but only doubles the beachfront — bytes-per-FLOP falls with every
    generation."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

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

    # Legend.
    ax.add_patch(Rectangle((3.95, 2.25), 0.3, 0.3, fc=BLUE, alpha=0.30,
                           ec="none"))
    ax.text(4.38, 2.40, "compute (area)", ha="left", va="center", fontsize=12.5,
            color="black")
    ax.add_patch(Rectangle((3.95, 1.70), 0.3, 0.3, fc=ORANGE, ec="none"))
    ax.text(4.38, 1.85, "I/O pads (perimeter)", ha="left", va="center",
            fontsize=12.5, color="black")

    fl.clean_axes(ax, lim=((-3.85, 6.6), (-2.85, 2.75)), hide=True)
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
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    _box(ax, 5.0, 4.55, 4.6, 0.95, "CPU + host DRAM", GRAY, fontsize=14,
         fc="#efefef")
    for cx, lab in ((2.7, "PCIe host bridge"), (7.3, "PCIe host bridge")):
        _box(ax, cx, 2.95, 2.75, 0.72, lab, GRAY, fontsize=12.5, fc="#f7f7f7")
    gpus = [(1.30, "GPU 0"), (3.70, "GPU 1"), (6.30, "GPU 2"), (8.70, "GPU 3")]
    for cx, lab in gpus:
        _box(ax, cx, 1.15, 1.95, 1.0, f"{lab}\nRTX 4090 · 24 GB", BLUE,
             fontsize=12, fc="#e8f1f8", tc="black")

    # Gray fabric links: GPU -> its bridge -> CPU.
    for cx in (1.30, 3.70):
        ax.plot([cx, 2.7], [1.68, 2.57], color=GRAY, lw=1.6, zorder=1)
    for cx in (6.30, 8.70):
        ax.plot([cx, 7.3], [1.68, 2.57], color=GRAY, lw=1.6, zorder=1)
    ax.plot([2.7, 4.4], [3.33, 4.05], color=GRAY, lw=1.6, zorder=1)
    ax.plot([7.3, 5.6], [3.33, 4.05], color=GRAY, lw=1.6, zorder=1)
    ax.text(1.70, 2.32, r"PCIe 4.0 $\times$16", ha="right", va="center",
            fontsize=11.5, color="black", rotation=33)

    # The staged path GPU0 -> host -> GPU2, highlighted alongside the links.
    seg = [(1.18, 1.72), (2.55, 2.60), (2.62, 3.34), (4.32, 4.08)]
    seg2 = [(5.68, 4.08), (7.38, 3.34), (7.45, 2.60), (6.42, 1.94)]
    ax.plot([p[0] for p in seg], [p[1] for p in seg], color=ORANGE, lw=2.6,
            zorder=2)
    ax.plot([p[0] for p in seg2[:-1]], [p[1] for p in seg2[:-1]], color=ORANGE,
            lw=2.6, zorder=2)
    ax.annotate("", xy=seg2[-1], xytext=seg2[-2],
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.6))

    ax.text(5.0, 0.28,
            "no P2P, no NVLink: every GPU$\\to$GPU byte is staged through "
            "host DRAM\n(PCIe-limited: tens of GB/s for a raw copy)",
            ha="center", va="center", fontsize=13, color="black")

    fl.clean_axes(ax, lim=((0.2, 9.8), (-0.25, 5.25)), hide=True, equal=False)
    fl.save(fig, "mdl-perf-pcie-topology")


# =========================================================================== #
# 13.3  Compute Graphs and Compilation                                        #
# =========================================================================== #

def fig_compute_graph():
    """The compute graph of a two-layer network with a loss: autograd already
    builds this DAG; eager execution walks it one kernel launch at a time,
    while a compiler sees the whole graph at once."""
    fig, ax = plt.subplots(figsize=(9.6, 3.4))

    OPF, DATF = "#e8f1f8", "#f3f3f3"
    ops = [(2.0, r"matmul"), (3.35, r"add"), (4.7, r"relu"),
           (6.05, r"matmul"), (7.4, r"add"), (8.75, r"loss")]
    y_main, y_par = 1.55, 0.35
    _box(ax, 0.65, y_main, 0.85, 0.55, r"$\mathbf{X}$", GRAY, fc=DATF,
         fontsize=13)
    for cx, lab in ops[:-1]:
        _box(ax, cx, y_main, 0.95, 0.55, lab, BLUE, fc=OPF, fontsize=12.5)
    _box(ax, ops[-1][0], y_main, 0.95, 0.55, ops[-1][1], GREEN, fc="#e9f4e9",
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
    # One backward arc above.
    ax.annotate("", xy=(0.85, y_main + 0.45), xytext=(8.65, y_main + 0.45),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8,
                                linestyle="--",
                                connectionstyle="arc3,rad=0.12"))
    ax.text(4.7, y_main + 1.15, "backward pass: the same graph, walked in "
            "reverse", ha="center", va="center", fontsize=13, color=ORANGE)
    ax.text(4.7, -0.42, "eager execution: every node is a separate kernel "
            "launch and a round trip to memory",
            ha="center", va="center", fontsize=12.5, color="black")

    fl.clean_axes(ax, lim=((0.0, 9.6), (-0.75, 3.0)), hide=True, equal=False)
    fl.save(fig, "mdl-perf-compute-graph")


def fig_compile_pipelines():
    """The two capture philosophies, side by side: torch.compile captures
    Python bytecode with guards and falls back on a graph break; jax.jit
    traces the function into a jaxpr and recompiles on new shapes."""
    fig, ax = plt.subplots(figsize=(10.2, 3.7))

    def pipeline(y, stages):
        xs = np.linspace(2.15, 9.25, len(stages))
        for x, s in zip(xs, stages):
            _box(ax, x, y, 1.48, 0.66, s, BLUE, fc="#f5f8fb", fontsize=11.5)
        for a, b in zip(xs[:-1], xs[1:]):
            ax.annotate("", xy=(b - 0.77, y), xytext=(a + 0.77, y),
                        arrowprops=dict(arrowstyle="->", color="black",
                                        lw=1.3))
        return xs

    y_pt, y_jx = 2.55, 0.75
    ax.text(0.0, y_pt, "torch.\ncompile", ha="left", va="center",
            fontsize=13, color="black")
    xs_pt = pipeline(y_pt, ["Python\nbytecode", "Dynamo\ncapture, guards",
                            "AOTAutograd\nfwd + bwd", "Inductor",
                            "Triton / C++\nkernels"])
    ax.text(0.0, y_jx, "jax.jit", ha="left", va="center", fontsize=13,
            color="black")
    xs_jx = pipeline(y_jx, ["Python\nfunction", "trace", "jaxpr", "XLA\ncompiler",
                            "fused\nexecutable"])

    # torch: graph break = fall back to Python, keep going.
    ax.annotate("", xy=(xs_pt[0] + 0.1, y_pt + 0.78),
                xytext=(xs_pt[1] - 0.1, y_pt + 0.78),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                linestyle="--",
                                connectionstyle="arc3,rad=0.25"))
    ax.text(xs_pt[1] + 0.35, y_pt + 1.02,
            "graph break: run that bit in Python, resume capture",
            ha="left", va="center", fontsize=12, color=ORANGE)
    # jax: new shape -> retrace.
    ax.annotate("", xy=(xs_jx[1], y_jx - 0.50),
                xytext=(xs_jx[-1], y_jx - 0.50),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                linestyle="--",
                                connectionstyle="arc3,rad=-0.08"))
    ax.text((xs_jx[1] + xs_jx[-1]) / 2, y_jx - 1.12,
            "new input shape or dtype: retrace and recompile",
            ha="center", va="center", fontsize=12, color=ORANGE)

    fl.clean_axes(ax, lim=((-0.2, 10.2), (-0.75, 4.0)), hide=True,
                  equal=False)
    fl.save(fig, "mdl-perf-compile-pipelines")


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
    ax.text(0.15, total[ti] + 0.18,
            "peak: activations full, backward beginning",
            ha="left", va="bottom", fontsize=13, color="black")

    # Band labels.
    ax.text(2.0, 0.5, "parameters", ha="center", va="center", fontsize=13,
            color="black")
    ax.text(2.0, 2.0, "optimizer states ($\\mathbf{m}$, $\\mathbf{v}$)",
            ha="center", va="center", fontsize=13, color="black")
    ax.text(6.6, 3.32, "gradients", ha="center", va="center", fontsize=13,
            color="black")
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
    ax.set_ylabel("memory (multiples of parameter bytes)", fontsize=12.5)
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
    axa.text(1.25, 2.62, "batch$_1$", ha="center", va="center", fontsize=12.5,
             color="black")
    axa.text(4.45, 2.62, "batch$_2$", ha="center", va="center", fontsize=12.5,
             color="black")
    for xc, col in ((1.25, c0), (4.45, c1)):
        axa.annotate("", xy=(xc, 2.10), xytext=(xc, 2.42),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
    axa.text(2.85, -0.15, "data parallel:\nreplicate the model,\nsplit the batch",
             ha="center", va="top", fontsize=12.5, color="black")
    fl.clean_axes(axa, lim=((-0.4, 6.1), (-1.75, 3.0)), hide=True)

    # (b) pipeline parallel: split the layer stack.
    axb = axes[1]
    grid(axb, 1.4, 1.6, lambda i, j: c0 if i < 2 else c1)
    axb.text(0.95, 1.4, "layers 1–2", ha="right", va="center", fontsize=12,
             color=c0)
    axb.text(0.95, 0.55, "layers 3–4", ha="right", va="center", fontsize=12,
             color=c1)
    axb.text(2.65, -0.15, "pipeline parallel:\nsplit the layer stack",
             ha="center", va="top", fontsize=12.5, color="black")
    fl.clean_axes(axb, lim=((-1.7, 4.8), (-1.75, 3.0)), hide=True)

    # (c) tensor parallel: split every layer's width.
    axc = axes[2]
    grid(axc, 1.4, 1.6, lambda i, j: c0 if j < W // 2 else c1)
    axc.annotate("", xy=(2.62, 2.30), xytext=(1.45, 2.30),
                 arrowprops=dict(arrowstyle="-", color=c0, lw=2.4))
    axc.annotate("", xy=(3.90, 2.30), xytext=(2.72, 2.30),
                 arrowprops=dict(arrowstyle="-", color=c1, lw=2.4))
    axc.text(2.65, 2.52, "each layer's units, halved", ha="center",
             va="bottom", fontsize=12, color="black")
    axc.text(2.65, -0.15, "tensor parallel:\nsplit every layer's width",
             ha="center", va="top", fontsize=12.5, color="black")
    fl.clean_axes(axc, lim=((-1.0, 5.5), (-1.75, 3.0)), hide=True)

    fl.save(fig, "mdl-perf-splitting")


def fig_data_parallel():
    """One data-parallel step on two GPUs: split the minibatch, run identical
    forward/backward on each replica, sum the gradients with an allreduce,
    apply the same update everywhere."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    _box(ax, 4.2, 4.35, 3.6, 0.7, "minibatch ($B$ samples)", GRAY,
         fc="#f3f3f3", fontsize=13)
    for cx, lab, half in ((2.0, "GPU 0", "$B/2$"), (6.4, "GPU 1", "$B/2$")):
        _box(ax, cx, 2.55, 3.3, 1.5,
             f"{lab}: full parameter copy\nforward + backward on {half}\n"
             r"$\rightarrow$ local gradients", BLUE, fc="#e8f1f8",
             fontsize=12, tc="black")
    ax.annotate("", xy=(2.0, 3.34), xytext=(3.5, 4.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(6.4, 3.34), xytext=(4.9, 4.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    _box(ax, 4.2, 0.75, 4.2, 0.7, "allreduce: sum the gradients", ORANGE,
         fc="#fdf0e3", fontsize=13)
    ax.annotate("", xy=(3.2, 1.14), xytext=(2.0, 1.76),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.annotate("", xy=(5.2, 1.14), xytext=(6.4, 1.76),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    # Summed gradients flow back up (curved, outside the boxes).
    ax.annotate("", xy=(0.28, 2.55), xytext=(1.95, 0.68),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                linestyle="--",
                                connectionstyle="arc3,rad=-0.4"))
    ax.annotate("", xy=(8.12, 2.55), xytext=(6.45, 0.68),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                linestyle="--",
                                connectionstyle="arc3,rad=0.4"))
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

    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.2))

    def draw_state(ax, M, title):
        cs = 0.6
        for g in range(k):
            for c in range(k):
                v = M[g, c]
                color = GREEN if v == k else BLUE
                ax.add_patch(Rectangle((c * cs, -g * cs), cs * 0.92,
                                       cs * 0.92, fc=color,
                                       alpha=0.15 + 0.75 * (v / k) * 0.8,
                                       ec="none"))
                ax.text(c * cs + cs * 0.46, -g * cs + cs * 0.46, str(v),
                        ha="center", va="center", fontsize=11.5,
                        color="black")
        ax.text(2 * cs - cs / 2, 1.0, title, ha="center", va="center",
                fontsize=12.5, color="black")
        fl.clean_axes(ax, lim=((-1.25, 2.8), (-1.95, 1.35)), hide=True)

    for i, M in enumerate(rs_states):
        draw_state(axes[0, i], M,
                   "start" if i == 0 else f"reduce-scatter, step {i}")
    for g in range(k):
        axes[0, 0].text(-0.18, -g * 0.6 + 0.28, f"GPU {g}", ha="right",
                        va="center", fontsize=11, color="black")
    for i, M in enumerate(ag_states):
        draw_state(axes[1, i], M, f"all-gather, step {i + 1}")
    for g in range(k):
        axes[1, 0].text(-0.18, -g * 0.6 + 0.28, f"GPU {g}", ha="right",
                        va="center", fontsize=11, color="black")
    # Last bottom panel: the accounting.
    axl = axes[1, 3]
    axl.text(0.9, -0.25,
             "per link, per phase:\n$\\frac{k-1}{k}\\,N$ bytes\n\n"
             "total $\\frac{2(k-1)}{k}\\,N$\nindependent of $k$",
             ha="center", va="center", fontsize=13, color="black")
    fl.clean_axes(axl, lim=((-1.25, 2.8), (-1.95, 1.35)), hide=True)

    fl.save(fig, "mdl-perf-ring-allreduce")


# =========================================================================== #
# 13.6  Multi-GPU in Practice                                                 #
# =========================================================================== #

def fig_ddp_overlap():
    """What DDP buys: gradient bucketing lets allreduce overlap the rest of
    the backward pass instead of waiting for all of it."""
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 3.9), sharex=True)

    h = 0.5
    layers = [(0.0, 1.5, r"$\nabla L_4$"), (1.5, 3.0, r"$\nabla L_3$"),
              (3.0, 4.5, r"$\nabla L_2$"), (4.5, 6.0, r"$\nabla L_1$")]

    def lanes(ax, comm, title, total):
        for (x0, x1, lab) in layers:
            _lane_kernel(ax, x0 + 0.03, (x1 - x0) - 0.06, 1.0, h, BLUE,
                         label=lab, fontsize=11.5)
        for (x0, x1, lab) in comm:
            _lane_kernel(ax, x0 + 0.03, (x1 - x0) - 0.06, 0.0, h, ORANGE,
                         label=lab, fontsize=11.5, tc="black")
        ax.text(-0.25, 1.0, "backward\n(compute)", ha="right", va="center",
                fontsize=11.5, color="black")
        ax.text(-0.25, 0.0, "allreduce\n(comm.)", ha="right", va="center",
                fontsize=11.5, color="black")
        ax.text(0.0, 1.78, title, ha="left", va="center", fontsize=12.5,
                color="black")
        ax.plot([total, total], [-0.45, 1.45], color="black", lw=1.2, ls="--")
        fl.clean_axes(ax, lim=((-2.15, 10.6), (-1.0, 2.15)), hide=True,
                      equal=False)

    lanes(axes[0], [(6.0, 10.0, "allreduce: all gradients")],
          "no overlap: wait for the whole backward, then communicate", 10.0)
    lanes(axes[1], [(3.0, 5.6, "bucket 1"), (6.0, 8.0, "bucket 2")],
          "DDP bucketing: communication overlaps the rest of the backward",
          8.0)
    axes[1].annotate("", xy=(10.0, -0.35), xytext=(8.0, -0.35),
                     arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.6))
    axes[1].text(9.0, -0.62, "time saved", ha="center", va="top", fontsize=12,
                 color=GREEN)

    fl.save(fig, "mdl-perf-ddp-overlap")


def fig_fsdp_lifecycle():
    """The FSDP lifecycle of one block: parameters live sharded (P/k per
    rank); the full block is materialized by an all-gather only while it
    computes, then freed; gradients leave via reduce-scatter, so each rank
    keeps only its shard."""
    fig, (ax, axm) = plt.subplots(2, 1, figsize=(9.4, 4.3), sharex=True,
                                  gridspec_kw={"height_ratios": [1.15, 1.0]})

    events = [
        (0.0, 1.4, "all-gather\nparams", ORANGE, "black"),
        (1.4, 3.4, "forward", BLUE, "white"),
        (3.4, 4.2, "free", GRAY, "white"),
        (4.6, 6.0, "all-gather\nparams", ORANGE, "black"),
        (6.0, 7.9, "backward", BLUE, "white"),
        (7.9, 9.75, "reduce-scatter\ngrads", ORANGE, "black"),
        (9.75, 10.4, "free", GRAY, "white"),
    ]
    for x0, x1, lab, color, tc in events:
        _lane_kernel(ax, x0 + 0.03, (x1 - x0) - 0.06, 0.5, 0.62, color,
                     label=lab, fontsize=10.5, tc=tc)
    ax.text(4.4, 1.35, "one block, one step (the other blocks stay sharded "
            "the whole time)", ha="center", va="center", fontsize=12.5,
            color="black")
    fl.clean_axes(ax, lim=((-0.3, 10.7), (-0.1, 1.75)), hide=True,
                  equal=False)

    # Memory trace of this block's parameter bytes on one rank.
    t = np.linspace(0, 10.4, 500)
    full, shard = 1.0, 1.0 / 4
    mem = np.full_like(t, shard)
    for (g0, g1, f0, f1) in ((0.0, 1.4, 3.4, 4.2), (4.6, 6.0, 9.75, 10.4)):
        rise = (t >= g0) & (t < g1)
        mem[rise] = shard + (full - shard) * (t[rise] - g0) / (g1 - g0)
        mem[(t >= g1) & (t < f0)] = full
        fall = (t >= f0) & (t < f1)
        mem[fall] = full - (full - shard) * (t[fall] - f0) / (f1 - f0)
    axm.plot(t, mem, color=BLUE, lw=2.2)
    axm.axhline(shard, color=GRAY, lw=1.0, ls="--")
    axm.text(10.55, shard, r"sharded: $P/k$ per rank", ha="left", va="center",
             fontsize=11.5, color="black")
    axm.text(10.55, full, r"materialized: $P$", ha="left", va="center",
             fontsize=11.5, color="black")
    axm.axhline(full, color=GRAY, lw=1.0, ls="--")
    axm.set_ylim(0, 1.35)
    axm.set_xlim(-0.3, 10.7)
    axm.set_ylabel("this block's\nparam bytes", fontsize=11.5)
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
