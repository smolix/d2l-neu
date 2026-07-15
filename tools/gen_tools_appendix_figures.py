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
    INK, NOVELTY_FILL, SANS, STROKE, Diagram, WRITTEN, save,
)
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402


WHITE = "#FFFFFF"
PALE = "#F6F8FA"


def box(d, x, y, w, h, label, fill=WHITE, edge=INK, color=INK,
        fs=11.5, bold=False, radius=10, note=None):
    d.ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=STROKE, zorder=3))
    d.ax.text(x, y + (4 if note else 0), label, family=SANS, fontsize=fs,
              fontweight="bold" if bold else "regular", color=color,
              ha="center", va="center", zorder=5)
    if note:
        note_color = "#D9D9D9" if fill == NOVELTY_FILL else GRAY_TEXT
        d.ax.text(x, y - 11, note, family=SANS, fontsize=8.8,
                  color=note_color, ha="center", va="center", zorder=5)


def h_arrow(d, x0, x1, y, color=INK, dashed=False):
    d.ax.annotate("", xy=(x1, y), xytext=(x0, y),
                  arrowprops=dict(arrowstyle="-|>", color=color, lw=STROKE,
                                  linestyle=(0, (2, 2)) if dashed else "solid",
                                  mutation_scale=11, shrinkA=0, shrinkB=0),
                  zorder=4)


def v_arrow(d, x, y0, y1, color=INK, dashed=False):
    d.ax.annotate("", xy=(x, y1), xytext=(x, y0),
                  arrowprops=dict(arrowstyle="-|>", color=color, lw=STROKE,
                                  linestyle=(0, (2, 2)) if dashed else "solid",
                                  mutation_scale=11, shrinkA=0, shrinkB=0),
                  zorder=4)


def caption(d, x, y, text, color=GRAY_TEXT, ha="center", fs=9.5):
    d.ax.text(x, y, text, family=SANS, fontsize=fs, color=color,
              ha=ha, va="center", zorder=6)


def kernel_state():
    d = Diagram(560, 238)
    box(d, 105, 146, 150, 88, "Notebook document", note="cells · saved outputs")
    box(d, 455, 146, 150, 88, "Live kernel", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True, note="variables · imports · devices")
    h_arrow(d, 180, 380, 158)
    caption(d, 280, 174, "execute selected cells")
    h_arrow(d, 380, 180, 125, color=ACCENT)
    caption(d, 280, 106, "return outputs", color=ACCENT)
    box(d, 280, 42, 192, 36, "Restart + run all", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    v_arrow(d, 280, 60, 97, color=ACCENT)
    caption(d, 280, 18, "the reproducibility check: document → fresh state")
    save(d.fig, "tools-kernel-state")


def remote_layers():
    d = Diagram(600, 260)
    labels = [
        (70, "Editor", "browser / VS Code"), (210, "Notebook server", "JupyterLab"),
        (355, "Python kernel", "environment + state"), (520, "Accelerator", "CPU / GPU"),
    ]
    for i, (x, label, note) in enumerate(labels):
        box(d, x, 132, 118, 58, label, fill=NOVELTY_FILL if i == 2 else WHITE,
            edge=NOVELTY_FILL if i == 2 else INK, color=WHITE if i == 2 else INK,
            bold=i == 2, note=note)
        if i: h_arrow(d, labels[i - 1][0] + 59, x - 59, 132)
    d.container(12, 76, 135, 206, fill=ACCENT_TINT, edge=ACCENT)
    d.container(146, 76, 588, 206, fill=PALE, edge="#AEB4BA")
    caption(d, 73, 222, "LOCAL MACHINE", ACCENT)
    caption(d, 367, 222, "REMOTE MACHINE / CLOUD")
    box(d, 285, 38, 210, 34, "SSH tunnel: localhost:8888", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    v_arrow(d, 180, 76, 55, color=ACCENT)
    save(d.fig, "tools-remote-layers")


def hosted_lifecycle():
    d = Diagram(580, 260)
    box(d, 115, 178, 188, 70, "Persistent artifacts", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True, note="notebook · data · checkpoints")
    box(d, 465, 178, 188, 70, "Ephemeral runtime", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True, note="VM · packages · accelerator")
    h_arrow(d, 209, 371, 188)
    caption(d, 290, 204, "attach / restore")
    h_arrow(d, 371, 209, 162, color=ACCENT)
    caption(d, 290, 143, "save selected results", color=ACCENT)
    box(d, 290, 65, 250, 48, "Disconnect, timeout, or reset", fill=ACCENT2_TINT,
        edge=ACCENT2, color=ACCENT2, bold=True)
    v_arrow(d, 410, 89, 137, color=ACCENT2)
    caption(d, 290, 24, "Treat the runtime as replaceable; make valuable state explicit.")
    save(d.fig, "tools-hosted-lifecycle")


def notebook_pipeline():
    d = Diagram(620, 224)
    xs = [70, 220, 380, 550]
    specs = [
        ("Book source", "authoritative .md"), ("Notebook build", "PT · JAX · NumPy"),
        ("notebooks branch", "revision-pinned .ipynb"), ("Provider", "Colab / Kaggle"),
    ]
    for i, (x, (label, note)) in enumerate(zip(xs, specs)):
        box(d, x, 122, 122, 62, label, fill=NOVELTY_FILL if i == 2 else WHITE,
            edge=NOVELTY_FILL if i == 2 else INK, color=WHITE if i == 2 else INK,
            bold=i == 2, note=note)
        if i: h_arrow(d, xs[i - 1] + 61, x - 61, 122)
    box(d, 380, 38, 182, 36, "manifest.json", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    v_arrow(d, 380, 56, 88, color=ACCENT)
    caption(d, 310, 199, "one stable page key controls every launch action")
    save(d.fig, "tools-notebook-pipeline")


def cloud_spectrum():
    d = Diagram(620, 250)
    d.ax.plot([70, 550], [130, 130], color=INK, lw=2, zorder=2)
    points = [
        (90, "Hyperscaler", "integrated · governed"),
        (270, "GPU specialist", "accelerator-focused"),
        (440, "Marketplace", "variable hosts"),
        (550, "Colocation", "you operate it"),
    ]
    for i, (x, label, note) in enumerate(points):
        d.ax.add_patch(Rectangle((x - 4, 126), 8, 8, facecolor=ACCENT, edgecolor=ACCENT, zorder=5))
        caption(d, x, 166, label, ACCENT if i == 2 else INK, fs=11.5)
        caption(d, x, 148, note, fs=9)
    caption(d, 70, 77, "more abstraction, compliance, and managed services", ha="left")
    caption(d, 550, 77, "more price dispersion and operational responsibility", ha="right")
    h_arrow(d, 70, 550, 99, color=ACCENT)
    box(d, 310, 32, 216, 36, "match trust to the data", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True)
    save(d.fig, "tools-cloud-spectrum")


def cloud_lifecycle():
    d = Diagram(610, 256)
    xs = [72, 210, 350, 492]
    for i, (x, label) in enumerate(zip(xs, ["Provision", "Connect", "Compute", "Checkpoint"])):
        box(d, x, 146, 104, 46, label, fill=NOVELTY_FILL if label == "Compute" else WHITE,
            edge=NOVELTY_FILL if label == "Compute" else INK,
            color=WHITE if label == "Compute" else INK, bold=label == "Compute")
        if i: h_arrow(d, xs[i - 1] + 52, x - 52, 146)
    box(d, 350, 54, 250, 48, "Persistent object storage", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True, note="data · logs · checkpoints")
    v_arrow(d, 492, 122, 78, color=ACCENT)
    h_arrow(d, 225, 492, 54, color=ACCENT)
    caption(d, 120, 54, "resume on a new VM", color=ACCENT)
    h_arrow(d, 225, 124, 54, color=ACCENT)
    box(d, 560, 54, 76, 36, "Delete VM", fill=ACCENT2_TINT, edge=ACCENT2,
        color=ACCENT2, bold=True)
    h_arrow(d, 492, 522, 54, color=ACCENT2)
    save(d.fig, "tools-cloud-lifecycle")


def cost_stack():
    d = Diagram(580, 270)
    components = [("GPU time", 106, ACCENT), ("idle / setup", 58, "#7FAAC9"),
                  ("storage", 48, "#AAB5BF"), ("egress", 34, ACCENT2),
                  ("engineering", 78, NOVELTY_FILL)]
    x = 42
    for label, width, color in components:
        d.ax.add_patch(Rectangle((x, 112), width, 54, facecolor=color,
                                 edgecolor=WHITE, linewidth=1.5))
        d.ax.text(x + width / 2, 139, label, family=SANS, fontsize=9.5,
                  color=WHITE if color != "#AAB5BF" else INK,
                  ha="center", va="center", zorder=5)
        x += width
    caption(d, 290, 194, "invoice price is only one part of experiment cost", fs=12)
    box(d, 290, 55, 238, 40, "cost per completed result", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True)
    v_arrow(d, 290, 112, 75)
    save(d.fig, "tools-cost-stack")


def hardware_decision():
    d = Diagram(620, 330)
    box(d, 310, 286, 210, 42, "What must run locally?", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True)
    boxes = [(125, 202, "Interactive inference", "latency · privacy"),
             (310, 202, "Fine-tuning", "weights + optimizer"),
             (495, 202, "Training", "throughput · scaling")]
    for x, y, label, note in boxes:
        box(d, x, y, 148, 54, label, note=note)
        d.line([(310, 265), (310, 244), (x, 244), (x, 229)])
    outcomes = [(125, "Unified memory", "capacity first"),
                (310, "One discrete GPU", "balanced default"),
                (495, "Rent a cluster", "bursty scale")]
    for x, label, note in outcomes:
        box(d, x, 88, 152, 54, label, fill=ACCENT_TINT, edge=ACCENT,
            color=ACCENT, bold=True, note=note)
        v_arrow(d, x, 175, 115, color=ACCENT)
    caption(d, 310, 28, "memory fit first; then software support, throughput, power, and cost")
    save(d.fig, "tools-hardware-decision")


def memory_fit():
    d = Diagram(610, 260)
    items = [("weights", 128, ACCENT), ("gradients", 94, "#70A7CF"),
             ("optimizer", 146, NOVELTY_FILL), ("activations", 112, ACCENT2)]
    x = 40
    for label, width, color in items:
        d.ax.add_patch(Rectangle((x, 130), width, 52, facecolor=color,
                                 edgecolor=WHITE, linewidth=1.5))
        d.ax.text(x + width / 2, 156, label, family=SANS, fontsize=10.5,
                  color=WHITE, ha="center", va="center")
        x += width
    d.ax.plot([40, 560], [112, 112], color=INK, lw=STROKE)
    d.ax.plot([40, 40], [105, 119], color=INK, lw=STROKE)
    d.ax.plot([560, 560], [105, 119], color=INK, lw=STROKE)
    caption(d, 300, 91, "peak training memory", fs=11.5)
    box(d, 300, 42, 254, 36, "must fit with safety margin", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    caption(d, 300, 218, "parameters alone do not determine whether training fits")
    save(d.fig, "tools-memory-fit")


def memory_path():
    d = Diagram(620, 270)
    # Discrete path
    caption(d, 155, 238, "DISCRETE GPU")
    box(d, 78, 165, 112, 48, "CPU RAM")
    box(d, 235, 165, 112, 48, "GPU VRAM", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True)
    h_arrow(d, 134, 179, 165)
    caption(d, 157, 143, "PCIe copy")
    # Unified path
    caption(d, 465, 238, "UNIFIED MEMORY")
    d.container(355, 102, 575, 205, fill=ACCENT_TINT, edge=ACCENT)
    box(d, 410, 154, 88, 46, "CPU")
    box(d, 520, 154, 88, 46, "GPU", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True)
    caption(d, 465, 115, "one address space · shared bandwidth", color=ACCENT)
    box(d, 310, 47, 280, 38, "capacity and bandwidth are different constraints",
        fill=ACCENT2_TINT, edge=ACCENT2, color=ACCENT2, bold=True)
    save(d.fig, "tools-memory-path")


def workstation():
    d = Diagram(620, 320)
    d.container(72, 48, 548, 278, fill=CONTAINER_FILL)
    d.stage_label(92, 260, "one-GPU workstation")
    box(d, 310, 207, 230, 54, "GPU + local VRAM", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True, note="fit the workload first")
    box(d, 172, 130, 160, 48, "CPU + system RAM", note="feed and preprocess")
    box(d, 448, 130, 160, 48, "NVMe storage", note="cache and checkpoints")
    box(d, 172, 71, 160, 36, "Power delivery", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    box(d, 448, 71, 160, 36, "Cooling + acoustics", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    d.line([(310, 180), (310, 154), (172, 154)])
    d.line([(310, 154), (448, 154)])
    v_arrow(d, 172, 106, 89, color=ACCENT)
    v_arrow(d, 448, 106, 89, color=ACCENT)
    save(d.fig, "tools-workstation")


def ecosystem_artifact():
    d = Diagram(620, 310)
    d.container(80, 40, 540, 274, fill=CONTAINER_FILL)
    d.stage_label(100, 256, "versioned model artifact")
    items = [(190, "Configuration", "architecture · shapes"),
             (430, "Weights", "safetensors · shards"),
             (190, "Pre/postprocess", "tokenizer · transforms"),
             (430, "Model card", "license · limits · provenance")]
    for i, (x, label, note) in enumerate(items):
        box(d, x, 188 if i < 2 else 104, 190, 54, label,
            fill=NOVELTY_FILL if label == "Weights" else WHITE,
            edge=NOVELTY_FILL if label == "Weights" else INK,
            color=WHITE if label == "Weights" else INK,
            bold=label == "Weights", note=note)
    box(d, 310, 53, 228, 34, "immutable revision identifier", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    save(d.fig, "tools-ecosystem-artifact")


def artifact_lifecycle():
    d = Diagram(650, 230)
    xs = [66, 170, 276, 382, 488, 594]
    labels = ["Discover", "Inspect", "Pin", "Cache", "Adapt", "Publish"]
    for i, (x, label) in enumerate(zip(xs, labels)):
        box(d, x, 122, 84, 42, label, fill=NOVELTY_FILL if label == "Pin" else WHITE,
            edge=NOVELTY_FILL if label == "Pin" else INK,
            color=WHITE if label == "Pin" else INK, bold=label == "Pin", fs=10.5)
        if i: h_arrow(d, xs[i - 1] + 42, x - 42, 122)
    box(d, 382, 43, 228, 36, "record revision + license", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    v_arrow(d, 276, 101, 61, color=ACCENT)
    caption(d, 325, 195, "a reproducible workflow preserves identity through every transformation")
    save(d.fig, "tools-artifact-lifecycle")


def training_ladder():
    d = Diagram(620, 315)
    levels = [(78, 64, "1 GPU", "baseline"), (205, 105, "DDP", "replicate"),
              (342, 146, "FSDP", "shard state"), (478, 187, "3-D+", "compose")]
    for i, (x, y, label, note) in enumerate(levels):
        box(d, x, y, 108, 50, label, fill=NOVELTY_FILL if label == "FSDP" else WHITE,
            edge=NOVELTY_FILL if label == "FSDP" else INK,
            color=WHITE if label == "FSDP" else INK, bold=label == "FSDP", note=note)
        if i:
            d.line([(levels[i - 1][0] + 54, levels[i - 1][1]),
                    (x - 72, levels[i - 1][1]), (x - 72, y), (x - 54, y)])
    box(d, 538, 252, 126, 44, "Experts / context", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    v_arrow(d, 478, 212, 230, color=ACCENT)
    caption(d, 305, 291, "add complexity only when the simpler rung cannot meet memory or throughput")
    save(d.fig, "tools-training-ladder")


def training_memory():
    d = Diagram(620, 280)
    labels = [("parameters", 115), ("gradients", 95), ("optimizer", 142), ("activations", 120)]
    x = 45
    colors = [ACCENT, "#75A9CE", NOVELTY_FILL, ACCENT2]
    for (label, width), color in zip(labels, colors):
        d.ax.add_patch(Rectangle((x, 152), width, 52, facecolor=color,
                                 edgecolor=WHITE, linewidth=1.5))
        d.ax.text(x + width / 2, 178, label, family=SANS, fontsize=10,
                  color=WHITE, ha="center", va="center")
        x += width
    techniques = [(105, "shard"), (245, "checkpoint"), (385, "accumulate"), (525, "offload")]
    for x, label in techniques:
        box(d, x, 73, 112, 38, label, fill=ACCENT_TINT, edge=ACCENT,
            color=ACCENT, bold=True)
        v_arrow(d, x, 92, 145, color=ACCENT)
    caption(d, 310, 239, "each technique attacks a different term in peak memory")
    caption(d, 310, 24, "measure the new communication, recomputation, or transfer cost")
    save(d.fig, "tools-training-memory")


def checkpoint_recovery():
    d = Diagram(620, 260)
    box(d, 100, 150, 150, 54, "Training workers", note="distributed state")
    box(d, 310, 150, 150, 54, "Atomic checkpoint", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True, note="model · optimizer · RNG")
    box(d, 520, 150, 150, 54, "Replacement workers", note="possibly new hosts")
    h_arrow(d, 175, 235, 150, color=ACCENT)
    h_arrow(d, 385, 445, 150, color=ACCENT)
    box(d, 310, 56, 218, 40, "preemption / host loss", fill=ACCENT2_TINT,
        edge=ACCENT2, color=ACCENT2, bold=True)
    v_arrow(d, 430, 76, 120, color=ACCENT2)
    caption(d, 310, 222, "recovery is part of the training design, not an afterthought")
    save(d.fig, "tools-checkpoint-recovery")


def serving_stack():
    d = Diagram(620, 315)
    layers = [(270, "Application", "auth · policy · product"),
              (205, "API / router", "backpressure · observability"),
              (140, "Serving engine", "batching · KV cache · scheduling"),
              (75, "Runtime + hardware", "kernels · memory · accelerators")]
    for i, (y, label, note) in enumerate(layers):
        box(d, 310, y, 410 - i * 38, 46, label,
            fill=NOVELTY_FILL if label == "Serving engine" else WHITE,
            edge=NOVELTY_FILL if label == "Serving engine" else INK,
            color=WHITE if label == "Serving engine" else INK,
            bold=label == "Serving engine", note=note)
        if i: v_arrow(d, 310, layers[i - 1][0] - 23, y + 23)
    caption(d, 560, 270, "product layer", ha="right")
    caption(d, 560, 75, "systems layer", ha="right")
    save(d.fig, "tools-serving-stack")


def prefill_decode():
    d = Diagram(620, 250)
    box(d, 145, 145, 210, 72, "Prefill", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True, note="many prompt tokens in parallel")
    box(d, 475, 145, 210, 72, "Decode", fill=WHITE, edge=INK,
        note="one new token per sequence")
    h_arrow(d, 250, 370, 145)
    caption(d, 310, 169, "KV state")
    caption(d, 145, 81, "compute-heavy · drives TTFT", color=ACCENT, fs=11)
    caption(d, 475, 81, "memory-heavy · drives TPOT", color=ACCENT2, fs=11)
    d.ax.plot([40, 580], [49, 49], color=INK, lw=STROKE)
    d.ax.plot([40, 285], [49, 49], color=ACCENT, lw=5)
    d.ax.plot([285, 580], [49, 49], color=ACCENT2, lw=5)
    save(d.fig, "tools-prefill-decode")


def continuous_batching():
    d = Diagram(640, 290)
    rows = [(225, 0, 6), (180, 1, 4), (135, 2, 7), (90, 4, 8)]
    for idx, (y, start, end) in enumerate(rows):
        caption(d, 48, y, f"request {idx + 1}", ha="right")
        for t in range(9):
            fill = ACCENT if start <= t < end else PALE
            edge = WHITE if start <= t < end else "#D8DDE2"
            d.ax.add_patch(Rectangle((70 + 55 * t, y - 14), 48, 28,
                                     facecolor=fill, edgecolor=edge, linewidth=1))
    for t in range(9):
        caption(d, 94 + 55 * t, 52, str(t + 1), fs=9)
    caption(d, 315, 272, "finished requests leave; waiting requests join at the next step")
    box(d, 540, 36, 120, 30, "no padded slots", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True, fs=9.5)
    save(d.fig, "tools-continuous-batching")


def prefix_cache():
    d = Diagram(630, 270)
    box(d, 124, 188, 190, 46, "shared system prompt", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    box(d, 124, 86, 190, 46, "cached prefix KV", fill=NOVELTY_FILL,
        edge=NOVELTY_FILL, color=WHITE, bold=True)
    v_arrow(d, 124, 165, 109, color=ACCENT)
    for i, (x, label) in enumerate([(370, "request A"), (500, "request B")]):
        box(d, x, 188, 104, 40, label)
        d.line([(219, 86), (x, 86), (x, 168)])
        v_arrow(d, x, 168, 148, color=ACCENT)
        box(d, x, 126, 104, 40, "unique suffix", fill=WHITE)
    box(d, 435, 48, 220, 34, "reuse prompt work and memory", fill=ACCENT_TINT,
        edge=ACCENT, color=ACCENT, bold=True)
    caption(d, 315, 243, "prefix caching helps only when tokenized prefixes match")
    save(d.fig, "tools-prefix-cache")


def main():
    for function in (kernel_state, remote_layers, hosted_lifecycle, notebook_pipeline,
                     cloud_spectrum, cloud_lifecycle, cost_stack, hardware_decision,
                     memory_fit, memory_path, workstation, ecosystem_artifact,
                     artifact_lifecycle, training_ladder, training_memory,
                     checkpoint_recovery, serving_stack, prefill_decode,
                     continuous_batching, prefix_cache):
        function()
    for output in WRITTEN:
        path = Path(output)
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line.rstrip() for line in lines) + "\n",
                        encoding="utf-8")
    print("Generated 20 Tools for Deep Learning figures in img/")


if __name__ == "__main__":
    main()
