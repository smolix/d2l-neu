#!/usr/bin/env python3
"""Generate the conceptual figures for Tools for Deep Learning.

The diagrams share the repository's Raschka-adapted gallery grammar: restrained
blue accent, explicit flow, rounded layers, readable labels, no UI screenshots,
and no decorative topology.  Output is deterministic committed SVG.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from arch_diagrams import (  # noqa: E402
    ACCENT, ACCENT2, ACCENT2_TINT, ACCENT_TINT, CONTAINER_FILL, GRAY_TEXT,
    IMG_DIR, INK, NOVELTY_FILL, SANS, STROKE, Diagram, WRITTEN, save,
)
from figstyle import tokens as _T  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402


WHITE = "#FFFFFF"
PALE = "#F6F8FA"


NOTE_FS = 9.6     # secondary note inside a box / quiet caption
LABEL_FS = 11.5   # default box label


def box(d, x, y, w, h, label, fill=WHITE, edge=INK, color=INK,
        fs=LABEL_FS, bold=False, radius=10, note=None, dashed=False):
    d.ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=STROKE, zorder=3,
        linestyle=(0, (3, 2)) if dashed else "solid"))
    d.ax.text(x, y + (6 if note else 0), label, family=SANS, fontsize=fs,
              fontweight="bold" if bold else "regular", color=color,
              ha="center", va="center", zorder=5)
    if note:
        note_color = "#D9D9D9" if fill == NOVELTY_FILL else GRAY_TEXT
        d.ax.text(x, y - 10, note, family=SANS, fontsize=NOTE_FS,
                  color=note_color, ha="center", va="center", zorder=5)


def dark_box(d, x, y, w, h, label, **kw):
    box(d, x, y, w, h, label, fill=NOVELTY_FILL, edge=NOVELTY_FILL,
        color=WHITE, bold=True, **kw)


def tint_box(d, x, y, w, h, label, accent=ACCENT, tint=ACCENT_TINT, **kw):
    box(d, x, y, w, h, label, fill=tint, edge=accent, color=accent,
        bold=True, **kw)


def _arrow(d, p0, p1, color=INK, dashed=False, lw=STROKE, style="-|>"):
    d.ax.annotate("", xy=p1, xytext=p0,
                  arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                  linestyle=(0, (2, 2)) if dashed else "solid",
                                  mutation_scale=11, shrinkA=0, shrinkB=0),
                  zorder=4)


def h_arrow(d, x0, x1, y, color=INK, dashed=False, lw=STROKE):
    _arrow(d, (x0, y), (x1, y), color, dashed, lw)


def v_arrow(d, x, y0, y1, color=INK, dashed=False, lw=STROKE):
    _arrow(d, (x, y0), (x, y1), color, dashed, lw)


def path(d, pts, color=INK, lw=STROKE):
    xs, ys = zip(*pts)
    d.ax.plot(xs, ys, color=color, lw=lw, solid_joinstyle="miter",
              solid_capstyle="butt", zorder=4)


def caption(d, x, y, text, color=GRAY_TEXT, ha="center", fs=NOTE_FS,
            bold=False, va="center"):
    d.ax.text(x, y, text, family=SANS, fontsize=fs, color=color, ha=ha,
              va=va, fontweight="bold" if bold else "regular",
              linespacing=1.35, zorder=6)


def bracket(d, x0, x1, y, depth=7, color=INK, up=True):
    """Square bracket spanning x0..x1 whose legs point up (or down)."""
    s = depth if up else -depth
    path(d, [(x0, y + s), (x0, y), (x1, y), (x1, y + s)], color=color)


def kernel_state():
    d = Diagram(600, 316)
    # Left: the document as the reader sees it (reading order).
    d.container(22, 64, 306, 296, fill=PALE, edge="#AEB4BA")
    caption(d, 164, 282, "NOTEBOOK DOCUMENT", INK, fs=10.5)
    caption(d, 164, 266, "reading order")
    doc_cells = [(228, "In [2]", "model = MLP()"),
                 (170, "In [3]", "train(model, data)"),
                 (112, "In [1]", "data = load_data()")]
    for y, tag, code in doc_cells:
        box(d, 164, y, 256, 44, "", fill=WHITE)
        d.ax.text(46, y + 12, tag, family=SANS, fontsize=NOTE_FS,
                  color=GRAY_TEXT, ha="left", va="center", zorder=6)
        d.rich(46, y - 8, [(code, INK, False)], 10.5, ha="left", mono=True)
    # Right: what the kernel actually executed (time order).
    d.container(334, 64, 566, 296, fill=ACCENT_TINT, edge=ACCENT)
    caption(d, 450, 282, "KERNEL SESSION", ACCENT, fs=10.5)
    caption(d, 450, 266, "execution order")
    run_steps = [(228, "1", "data = load_data()"),
                 (170, "2", "model = MLP()"),
                 (112, "3", "train(model, data)")]
    for y, num, code in run_steps:
        d.ax.add_patch(Circle((362, y), 10, facecolor=ACCENT,
                              edgecolor="none", zorder=5))
        d.ax.text(362, y, num, family=SANS, fontsize=9.5, color=WHITE,
                  fontweight="bold", ha="center", va="center", zorder=6)
        d.rich(382, y, [(code, INK, False)], 10.5, ha="left", mono=True)
        if y != 112:
            v_arrow(d, 362, y - 12, y - 46, color=ACCENT)
    d.ax.text(542, 112, "✓", family=SANS, fontsize=13, color=ACCENT,
              fontweight="bold", ha="center", va="center", zorder=6)
    # The mapping between the two orders crosses — that crossing is the bug.
    for y_doc, y_run in [(228, 170), (170, 112), (112, 228)]:
        d.ax.plot([294, 350], [y_doc, y_run], color=GRAY_TEXT, lw=1.3,
                  ls=(0, (1.0, 2.6)), solid_capstyle="round", zorder=2)
    box(d, 294, 32, 544, 38,
        "restart + run all replays reading order and exposes the bug:  "
        "NameError: 'data' is not defined",
        fill=ACCENT2_TINT, edge=ACCENT2, color=ACCENT2, bold=True, fs=10.5)
    save(d.fig, "tools-kernel-state")


def remote_layers():
    d = Diagram(556, 200)
    w = 94
    d.container(12, 60, 122, 172, fill=ACCENT_TINT, edge=ACCENT)
    d.container(202, 60, 544, 172, fill=PALE, edge="#AEB4BA")
    caption(d, 67, 186, "LOCAL MACHINE", ACCENT, fs=10.5)
    caption(d, 373, 186, "REMOTE MACHINE / CLOUD", INK, fs=10.5)
    box(d, 67, 116, w, 58, "Editor", note="browser / VS Code")
    remote = [(257, "Notebook server", "JupyterLab"),
              (373, "Python kernel", "environment + state"),
              (489, "Accelerator", "CPU / GPU")]
    for i, (x, label, note) in enumerate(remote):
        if i == 1:
            dark_box(d, x, 116, w, 58, label, note=note)
        else:
            box(d, x, 116, w, 58, label, note=note)
        if i:
            h_arrow(d, remote[i - 1][0] + w / 2, x - w / 2, 116)
    # The SSH tunnel is the one link that crosses the machine boundary.
    h_arrow(d, 67 + w / 2, 257 - w / 2, 116, color=ACCENT, lw=2.2)
    caption(d, 162, 131, "SSH tunnel", ACCENT, fs=10, bold=True)
    d.rich(162, 101, [("localhost:8888", ACCENT, False)], NOTE_FS, mono=True)
    save(d.fig, "tools-remote-layers")


def hosted_lifecycle():
    d = Diagram(580, 260)
    tint_box(d, 115, 178, 188, 70, "Persistent artifacts",
             note="notebook · data · checkpoints")
    dark_box(d, 465, 178, 188, 70, "Ephemeral runtime",
             note="VM · packages · accelerator")
    h_arrow(d, 209, 371, 192)
    caption(d, 290, 205, "attach / restore")
    h_arrow(d, 371, 209, 164, color=ACCENT)
    caption(d, 290, 151, "save selected results", color=ACCENT)
    tint_box(d, 465, 94, 200, 36, "Disconnect, timeout, or reset",
             accent=ACCENT2, tint=ACCENT2_TINT)
    v_arrow(d, 465, 112, 143, color=ACCENT2)
    caption(d, 455, 127, "local disk is lost", color=ACCENT2, ha="right")
    save(d.fig, "tools-hosted-lifecycle")


def notebook_pipeline():
    d = Diagram(520, 200)
    xs, ws = [62, 190, 321, 452], [100, 100, 106, 100]
    specs = [
        ("Book source", "authoritative .md"),
        ("Notebook build", "PT · TF · JAX · NumPy"),
        ("notebooks branch", "revision-pinned .ipynb"),
        ("Provider", "Colab / Kaggle"),
    ]
    for i, (x, (label, note)) in enumerate(zip(xs, specs)):
        if i == 2:
            dark_box(d, x, 122, ws[i], 58, label, note=note)
        else:
            box(d, x, 122, ws[i], 58, label, note=note)
        if i:
            h_arrow(d, xs[i - 1] + ws[i - 1] / 2, x - ws[i] / 2, 122)
    tint_box(d, 321, 42, 150, 44, "manifest.json",
             note="one stable key per book page")
    v_arrow(d, 321, 64, 93, color=ACCENT)
    save(d.fig, "tools-notebook-pipeline")


def cloud_spectrum():
    d = Diagram(620, 230)
    points = [
        (110, "Hyperscaler", "integrated · governed"),
        (310, "GPU specialist", "accelerator-focused"),
        (510, "Marketplace", "variable hosts"),
    ]
    for x, label, note in points:
        d.ax.add_patch(Rectangle((x - 5, 125), 10, 10, facecolor=ACCENT,
                                 edgecolor=ACCENT, zorder=5))
        caption(d, x, 165, label, INK, fs=12, va="baseline")
        caption(d, x, 147, note, va="baseline")
    _arrow(d, (40, 130), (580, 130), color=INK, style="<|-|>", lw=1.6)
    caption(d, 40, 112, "more integration, compliance,\nand managed services",
            ha="left", va="top")
    caption(d, 580, 112, "more price dispersion and\noperational responsibility",
            ha="right", va="top")
    save(d.fig, "tools-cloud-spectrum")


def cloud_lifecycle():
    d = Diagram(540, 200)
    steps = ["Provision", "Connect", "Compute", "Checkpoint", "Delete VM"]
    w = 80
    xs = [52, 160, 268, 376, 484]
    for i, (x, label) in enumerate(zip(xs, steps)):
        if label == "Compute":
            dark_box(d, x, 150, w, 44, label)
        elif label == "Delete VM":
            tint_box(d, x, 150, w, 44, label, accent=ACCENT2,
                     tint=ACCENT2_TINT)
        else:
            box(d, x, 150, w, 44, label)
        if i:
            h_arrow(d, xs[i - 1] + w / 2, x - w / 2, 150)
    tint_box(d, 340, 58, 150, 48, "Persistent object storage",
             note="data · logs · checkpoints")
    v_arrow(d, 376, 128, 82, color=ACCENT)
    # Resume: storage feeds a fresh machine back at the start of the loop.
    path(d, [(265, 58), (52, 58), (52, 112)], color=ACCENT)
    v_arrow(d, 52, 110, 128, color=ACCENT)
    caption(d, 158, 70, "resume on a new VM", color=ACCENT)
    save(d.fig, "tools-cloud-lifecycle")


def _save_plot(fig, name):
    """Like arch_diagrams.save, but keeps axis furniture (tick and axis
    labels), which _content_bbox deliberately crops away for box diagrams."""
    import os
    path = os.path.join(IMG_DIR, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight",
                metadata={"Date": None})
    plt.close(fig)
    WRITTEN.append(path)


def hardware_menu():
    """Capacity × bandwidth scatter: the July 2026 local-hardware menu."""
    fig = plt.figure(figsize=(640 / 72.0, 360 / 72.0))
    ax = fig.add_axes([0.11, 0.14, 0.78, 0.82])
    discrete = [("RTX 5070 Ti", 16, 896, "$0.9k", (-8, -26), "left"),
                ("RTX 3090 used", 24, 936, "$1.2k", (8, 12), "left"),
                ("RTX 5090", 32, 1792, "$3–4k", (0, 30), "center"),
                ("RTX PRO 6000", 96, 1792, "$13k", (12, -6), "left")]
    unified = [("Strix Halo", 128, 256, "$2–2.5k", (-12, -6), "right"),
               ("DGX Spark", 128, 273, "$4.7k", (12, 20), "left"),
               ("M4 Max", 128, 546, "$3.7k", (12, -6), "left"),
               ("M3 Ultra", 512, 819, "$5.3k+", (0, 30), "center")]
    datacenter = [("H100 SXM", 80, 3350, "rented", (12, -6), "left")]
    groups = [(discrete, ACCENT, "discrete GPU — bandwidth first"),
              (unified, ACCENT2, "unified memory — capacity first"),
              (datacenter, NOVELTY_FILL, "datacenter — rent by the hour")]
    for devices, color, _ in groups:
        for name, mem, bw, price, (dx, dy), ha in devices:
            # White rims keep near-coincident points (Spark/Strix) apart.
            ax.scatter([mem], [bw], s=70, color=color, zorder=5,
                       edgecolors=WHITE, linewidths=1.5)
            ax.annotate(name, (mem, bw), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va="center",
                        family=SANS, fontsize=10.5, color=INK, zorder=6)
            ax.annotate(price, (mem, bw), textcoords="offset points",
                        xytext=(dx, dy - 12), ha=ha, va="center",
                        family=SANS, fontsize=9.5, color=GRAY_TEXT, zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(11, 1024)
    ax.set_ylim(190, 5200)
    ax.set_xticks([16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["16", "32", "64", "128", "256", "512"],
                       family=SANS, fontsize=10)
    ax.set_yticks([256, 512, 1024, 2048, 4096])
    ax.set_yticklabels(["256", "512", "1024", "2048", "4096"],
                       family=SANS, fontsize=10)
    ax.minorticks_off()
    ax.set_xlabel("accelerator-visible memory (GB)", family=SANS,
                  fontsize=12, color=INK)
    ax.set_ylabel("memory bandwidth (GB/s)", family=SANS,
                  fontsize=12, color=INK)
    right = ax.secondary_yaxis(
        "right", functions=(lambda v: v / 4.5, lambda v: v * 4.5))
    right.set_yticks([57, 114, 228, 455, 910])
    right.set_yticklabels(["57", "114", "228", "455", "910"],
                          family=SANS, fontsize=10, color=GRAY_TEXT)
    right.minorticks_off()
    right.set_ylabel("≤ decode tok/s (8B model, 4-bit)", family=SANS,
                     fontsize=11, color=GRAY_TEXT)
    right.spines["right"].set_color(GRAY_TEXT)
    right.tick_params(colors=GRAY_TEXT)
    ax.spines["top"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
    ax.tick_params(colors="black")
    ax.grid(True, which="major", color="#DDDDDD", lw=0.7, zorder=0)
    # Legend in the empty upper right (nothing on the menu is both large and
    # fast), on an opaque panel so the grid does not run through its text.
    ax.add_patch(Rectangle((0.565, 0.765), 0.43, 0.215,
                           transform=ax.transAxes, facecolor=WHITE,
                           edgecolor="none", zorder=4))
    for i, (_, color, legend) in enumerate(groups):
        y = 0.935 - 0.068 * i
        ax.scatter([0.59], [y], s=46, color=color, transform=ax.transAxes,
                   zorder=6, clip_on=False)
        ax.text(0.612, y, legend, transform=ax.transAxes, family=SANS,
                fontsize=10.5, color=INK, va="center", zorder=6)
    _save_plot(fig, "tools-hardware-menu")


# Peak-memory terms for a 7B model in BF16 with Adam (GB): the same toy
# numbers the training-systems section uses, so the bars are to scale.
# Gradients take the book's gradient role colour (red).  Each entry carries
# its label colour: white only on the dark fills, INK on the light orange.
MEMORY_TERMS = [("parameters", 14.0, ACCENT, WHITE),
                ("gradients", 14.0, _T.ROLE["grad"].base, WHITE),
                ("optimizer", 56.0, NOVELTY_FILL, WHITE),
                ("activations", 18.0, ACCENT2, INK)]


def _memory_bar(d, x0, y, h, width):
    """Stacked to-scale bar; returns the x boundaries of the segments."""
    scale = width / sum(term[1] for term in MEMORY_TERMS)
    edges, x = [x0], x0
    for label, value, color, text_color in MEMORY_TERMS:
        w = value * scale
        d.ax.add_patch(Rectangle((x, y), w, h, facecolor=color,
                                 edgecolor=WHITE, linewidth=1.5, zorder=3))
        d.ax.text(x + w / 2, y + h / 2, label,
                  family=SANS, fontsize=10.5, color=text_color, ha="center",
                  va="center", zorder=5)
        x += w
        edges.append(x)
    return edges


def memory_fit():
    d = Diagram(610, 200)
    # Device capacity: the dashed frame.  The bar must fit inside with room.
    d.ax.add_patch(FancyBboxPatch(
        (34, 124), 540, 64, boxstyle="round,pad=0,rounding_size=6",
        facecolor="none", edgecolor=INK, linewidth=STROKE,
        linestyle=(0, (3, 2)), zorder=2))
    caption(d, 36, 199, "accelerator memory", INK, ha="left", fs=10.5)
    edges = _memory_bar(d, 40, 130, 52, 456)
    caption(d, (edges[-1] + 568) / 2, 156, "headroom")
    bracket(d, edges[0], edges[-1], 110)
    caption(d, (edges[0] + edges[-1]) / 2, 94, "peak training memory",
            INK, fs=11.5)
    save(d.fig, "tools-memory-fit")


def memory_path():
    d = Diagram(620, 250)
    # Left panel: two separate pools joined by a copy.
    d.container(20, 78, 292, 206, fill=PALE, edge="#AEB4BA")
    caption(d, 156, 220, "DISCRETE GPU", INK, fs=10.5)
    box(d, 86, 142, 108, 54, "CPU RAM", note="large · slower")
    dark_box(d, 226, 142, 108, 54, "GPU VRAM", note="small · fast")
    h_arrow(d, 140, 172, 142)
    caption(d, 156, 102, "every transfer is a PCIe copy")
    # Right panel: one pool, two processors.
    d.container(328, 78, 600, 206, fill=ACCENT_TINT, edge=ACCENT)
    caption(d, 464, 220, "UNIFIED MEMORY", ACCENT, fs=10.5)
    box(d, 404, 174, 96, 34, "CPU")
    dark_box(d, 524, 174, 96, 34, "GPU")
    box(d, 464, 110, 236, 46, "one memory pool",
        note="one address space · shared bandwidth")
    for x in (404, 524):
        path(d, [(x, 157), (x, 133)])
    tint_box(d, 310, 36, 300, 36,
             "capacity and bandwidth are different constraints",
             accent=ACCENT2, tint=ACCENT2_TINT)
    save(d.fig, "tools-memory-path")


def workstation():
    d = Diagram(620, 260)
    d.container(22, 40, 598, 248, fill=CONTAINER_FILL)
    d.stage_label(40, 236, "one-GPU workstation")
    # Data path: storage feeds the CPU, the CPU feeds the GPU.
    row = [(117, "NVMe storage", "datasets · checkpoints"),
           (310, "CPU + system RAM", "feed and preprocess"),
           (503, "GPU + local VRAM", "fit the workload first")]
    for i, (x, label, note) in enumerate(row):
        if i == 2:
            dark_box(d, x, 172, 156, 56, label, note=note)
        else:
            box(d, x, 172, 156, 56, label, note=note)
        if i:
            h_arrow(d, row[i - 1][0] + 78, x - 78, 172)
    # Infrastructure that has to hold up under sustained full load.
    tint_box(d, 179, 88, 280, 48, "Power delivery",
             note="sustained and transient load")
    tint_box(d, 456, 88, 250, 48, "Cooling + acoustics",
             note="airflow at full load")
    save(d.fig, "tools-workstation")


def ecosystem_artifact():
    d = Diagram(620, 280)
    d.container(80, 30, 540, 262, fill=CONTAINER_FILL)
    d.stage_label(100, 250, "versioned model artifact")
    items = [(190, 200, "Configuration", "architecture · shapes"),
             (430, 200, "Weights", "safetensors · shards"),
             (190, 132, "Pre/postprocess", "tokenizer · transforms"),
             (430, 132, "Model card", "license · limits · provenance")]
    for x, y, label, note in items:
        if label == "Weights":
            dark_box(d, x, y, 190, 54, label, note=note)
        else:
            box(d, x, y, 190, 54, label, note=note)
    tint_box(d, 310, 66, 230, 42, "immutable revision",
             note="a commit hash, never main")
    save(d.fig, "tools-ecosystem-artifact")


def training_ladder():
    d = Diagram(620, 250)
    w, h, gap = 108, 50, 34
    levels = [(66, 50, "1 GPU", "baseline"), (66 + w + gap, 96, "DDP", "replicate"),
              (66 + 2 * (w + gap), 142, "FSDP", "shard state"),
              (66 + 3 * (w + gap), 188, "Composed", "TP · PP · CP · EP")]
    for i, (x, y, label, note) in enumerate(levels):
        if label == "FSDP":
            dark_box(d, x, y, w, h, label, note=note)
        else:
            box(d, x, y, w, h, label, note=note)
        if i:
            x0, y0 = levels[i - 1][0] + w / 2, levels[i - 1][1]
            xm = x0 + gap / 2
            path(d, [(x0, y0), (xm, y0), (xm, y)])
            h_arrow(d, xm - STROKE / 2, x - w / 2, y)
    caption(d, 14, 214, "Add a rung only when the simpler one\n"
            "cannot meet memory or throughput.", INK, ha="left",
            va="top", fs=10.5)
    save(d.fig, "tools-training-ladder")


def training_memory():
    d = Diagram(560, 300)
    edges = _memory_bar(d, 40, 140, 46, 420)
    # Sharding attacks the model state (first three terms), from above.
    bracket(d, edges[0], edges[3], 198, up=False)
    v_arrow(d, (edges[0] + edges[3]) / 2, 226, 199, color=ACCENT)
    tint_box(d, (edges[0] + edges[3]) / 2, 244, 150, 40, "shard (FSDP)",
             note="+ all-gather traffic")
    # The other three attack single terms, from below; each note is the price.
    tops = 84
    below = [(250, 250, "offload", "+ host transfers"),
             (370, edges[3] + 20, "checkpoint", "+ recomputation"),
             (490, edges[4] - 16, "accumulate", "+ more sequential steps")]
    for xb, xt, label, note in below:
        tint_box(d, xb, 62, 110, 44, label, note=note)
        if abs(xb - xt) < 1:
            v_arrow(d, xt, tops, 139, color=ACCENT)
        else:
            path(d, [(xb, tops), (xb, 108), (xt, 108)], color=ACCENT)
            v_arrow(d, xt, 108 - STROKE / 2, 139, color=ACCENT)
    save(d.fig, "tools-training-memory")


def prefill_decode():
    d = Diagram(620, 230)
    tint_box(d, 145, 172, 210, 64, "Prefill",
             note="prompt in parallel · compute-bound")
    tint_box(d, 475, 172, 210, 64, "Decode", accent=ACCENT2,
             tint=ACCENT2_TINT, note="one token per step · bandwidth-bound")
    h_arrow(d, 250, 370, 172)
    caption(d, 310, 184, "KV cache")
    # One request on a time axis: a long prefill, then short decode steps.
    y, hb = 58, 22
    d.ax.add_patch(Rectangle((40, y), 210, hb, facecolor=ACCENT,
                             edgecolor="none", zorder=3))
    x, step = 254, 29.0
    while x + 25 <= 562:
        d.ax.add_patch(Rectangle((x, y), 25, hb, facecolor=ACCENT2,
                                 edgecolor="none", zorder=3))
        x += step
    h_arrow(d, 40, 600, y - 10)
    caption(d, 600, y - 22, "time", ha="right")
    bracket(d, 40, 250, y + hb + 8, depth=-6)
    caption(d, 145, y + hb + 22, "TTFT", INK, bold=True, fs=10.5)
    xt = 254 + 3 * step
    bracket(d, xt, xt + 25, y + hb + 8, depth=-6)
    caption(d, xt + 12.5, y + hb + 22, "TPOT", INK, bold=True, fs=10.5)
    save(d.fig, "tools-prefill-decode")


def continuous_batching():
    """The toy scheduler from the text: lengths 5, 2, 7, 3 on two slots."""
    d = Diagram(640, 300)
    cell, x0 = 38, 92
    colors = {1: (ACCENT, WHITE), 2: (_T.TEAL.base, INK),
              3: (NOVELTY_FILL, WHITE), 4: (ACCENT2, INK)}

    def run(slot_y, start, length, req):
        d.ax.add_patch(FancyBboxPatch(
            (x0 + cell * start + 2, slot_y - 13), cell * length - 4, 26,
            boxstyle="round,pad=0,rounding_size=4",
            facecolor=colors[req][0], edgecolor="none", zorder=3))
        d.ax.text(x0 + cell * (start + length / 2), slot_y, f"R{req}",
                  family=SANS, fontsize=10.5, fontweight="bold",
                  color=colors[req][1],
                  ha="center", va="center", zorder=5)

    def pad(slot_y, start, length):
        d.ax.add_patch(FancyBboxPatch(
            (x0 + cell * start + 2, slot_y - 13), cell * length - 4, 26,
            boxstyle="round,pad=0,rounding_size=4", facecolor=PALE,
            edgecolor=GRAY_TEXT, linewidth=1, linestyle=(0, (2, 2)),
            zorder=3))
        d.ax.text(x0 + cell * (start + length / 2), slot_y, "padding",
                  family=SANS, fontsize=NOTE_FS, color=GRAY_TEXT,
                  ha="center", va="center", zorder=5)

    def finish(y_top, y_bot, steps, label):
        x = x0 + cell * steps + 4     # 6 units clear of the bar's end
        d.ax.plot([x, x], [y_bot, y_top], color=INK, lw=STROKE,
                  ls=(0, (3, 2)), zorder=2)
        caption(d, x + 6, (y_top + y_bot) / 2, label, INK, ha="left",
                bold=True, fs=10.5)

    # Static: each batch runs until its longest member finishes.
    caption(d, 14, 282, "STATIC BATCHING", INK, ha="left", fs=10.5)
    s1, s2 = 252, 220
    run(s1, 0, 5, 1), run(s1, 5, 7, 3)
    run(s2, 0, 2, 2), pad(s2, 2, 3), run(s2, 5, 3, 4), pad(s2, 8, 4)
    finish(268, 204, 12, "12 steps")
    # Continuous: a freed slot is refilled at the next step.
    caption(d, 14, 176, "CONTINUOUS BATCHING", ACCENT, ha="left", fs=10.5)
    c1, c2 = 146, 114
    run(c1, 0, 5, 1), run(c1, 5, 3, 4)
    run(c2, 0, 2, 2), run(c2, 2, 7, 3)
    finish(162, 98, 9, "9 steps")
    for y, name in [(s1, "slot 1"), (s2, "slot 2"), (c1, "slot 1"),
                    (c2, "slot 2")]:
        caption(d, x0 - 10, y, name, ha="right")
    for t in range(12):
        caption(d, x0 + cell * (t + 0.5), 80, str(t + 1))
    caption(d, x0 + cell * 6, 62, "decode step")
    save(d.fig, "tools-continuous-batching")


def prefix_cache():
    d = Diagram(620, 250)
    # One shared prefix node, two unique suffixes: a two-leaf prefix tree.
    dark_box(d, 150, 150, 200, 112, "shared system prompt",
             note="KV computed once, cached")
    for y, tag in [(185, "A"), (115, "B")]:
        box(d, 400, y, 150, 44, f"request {tag} suffix")
        h_arrow(d, 250, 325, y)
    caption(d, 150, 78, "reused by every matching request", ACCENT)
    caption(d, 400, 78, "unique work per request")
    tint_box(d, 275, 34, 390, 36,
             "reuse requires identical tokenized prefixes")
    save(d.fig, "tools-prefix-cache")


def main():
    for function in (kernel_state, remote_layers, hosted_lifecycle, notebook_pipeline,
                     cloud_spectrum, cloud_lifecycle, hardware_menu,
                     memory_fit, memory_path, workstation, ecosystem_artifact,
                     training_ladder, training_memory, prefill_decode,
                     continuous_batching, prefix_cache):
        function()
    for output in WRITTEN:
        path = Path(output)
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line.rstrip() for line in lines) + "\n",
                        encoding="utf-8")
    print(f"Generated {len(WRITTEN)} Tools for Deep Learning figures in img/")


if __name__ == "__main__":
    main()
