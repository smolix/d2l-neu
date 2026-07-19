#!/usr/bin/env python3
"""Generate the illustrative figures for the hybrid-architectures section of
``chapter_recurrent-modern`` (12.7 "Hybrid Architectures") in the shared house
style defined in ``gen_mdl_figures.py``.

Two figures:

  * the hybrid cache curve -- decode-time memory against context length for a
    32-layer model at production width: a pure transformer's KV cache grows
    linearly (32 GB at 256K), a pure recurrent stack is flat (64 MB), and a
    4-of-32 hybrid pays exactly the attention fraction of the transformer
    bill.  All three curves are computed from the KV-cache byte formula of
    the transformers chapter plus the Mamba-2 state-size accounting;
  * the hybrid stack schematic -- four shipped answers to "where does the
    attention go": Jamba (attention every 8th layer), Nemotron-H (a strict
    11-layer period, never at the front), Samba (sliding-window attention on
    every second layer, no global attention at all), Zamba2 (two weight-shared
    attention blocks re-entered along a Mamba-2 backbone).  Attention layer
    positions are the config-verified ones from the section's recipe table.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_ssm_hybrids_figures.py

All figures are written to ``img/mdl-modernrnn-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT

from matplotlib.patches import Rectangle


# --------------------------------------------------------------------------- #
# Figure 1: decode-time memory vs context length (the economics figure)       #
# --------------------------------------------------------------------------- #

def fig_hybrid_cache():
    """Three 32-layer stacks at production width, 16-bit.

    Per attention layer, KV bytes = 2 * n_kv * d_head * T * 2  (GQA 8 KV heads
    of dim 128 -> 4 KiB per token per layer).  Per recurrent layer, the
    Mamba-2 state is 2 * d_model * d_state elements (d_model 4096, N=128)
    -> 2 MiB, constant in T.
    """
    n_layers, n_attn_hybrid = 32, 4
    kv_per_tok_layer = 2 * 8 * 128 * 2                  # 4 KiB / token / layer
    state_layer = 2 * 4096 * 128 * 2                    # 2 MiB, flat

    T = np.logspace(np.log10(1024), np.log10(262144), 200)
    GiB = 2.0 ** 30
    transformer = n_layers * kv_per_tok_layer * T / GiB
    hybrid = (n_attn_hybrid * kv_per_tok_layer * T
              + (n_layers - n_attn_hybrid) * state_layer) / GiB
    recurrent = np.full_like(T, n_layers * state_layer / GiB)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.loglog(T, transformer, color=ORANGE, lw=2.4,
              label="pure transformer (32 attention layers)")
    ax.loglog(T, hybrid, color=GREEN, lw=2.4,
              label="hybrid (4 attention + 28 recurrent)")
    ax.loglog(T, recurrent, color=BLUE, lw=2.4,
              label="pure recurrent (32 fixed states)")

    # Right-edge value labels, black, offset clear of the curves.
    for y, txt in [(transformer[-1], "32 GB"), (hybrid[-1], "4 GB"),
                   (recurrent[-1], "64 MB")]:
        ax.annotate(txt, xy=(T[-1], y), xytext=(1.12 * T[-1], y),
                    fontsize=13, color="black", va="center", ha="left")

    ax.set_xlim(1024, 262144 * 3.2)
    ax.set_xticks([1024, 4096, 16384, 65536, 262144])
    ax.set_xticklabels(["1K", "4K", "16K", "64K", "256K"], fontsize=12,
                       color="black")
    yticks = [2 ** -4, 2 ** -1, 2 ** 2, 2 ** 5]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["64 MB", "512 MB", "4 GB", "32 GB"], fontsize=12,
                       color="black")
    ax.set_ylim(2 ** -5, 2 ** 7)
    ax.minorticks_off()
    ax.tick_params(colors="black")
    ax.set_xlabel("context length (tokens)", fontsize=13, color="black")
    ax.set_ylabel("decode-time memory", fontsize=13, color="black")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
    ax.grid(True, which="major", linestyle="--", color=LIGHT, lw=0.8)
    ax.legend(loc="upper left", fontsize=12)
    fl.save(fig, "mdl-modernrnn-hybrid-cache")


# --------------------------------------------------------------------------- #
# Figure 2: where the attention goes (four shipped stacks)                    #
# --------------------------------------------------------------------------- #

# (name, total layers, dict of special positions).  Positions are 1-indexed
# from the input; verified against each model's released config / report.
STACKS = [
    ("Jamba\n32 layers", 32,
     {"attn": [4, 12, 20, 28], "swa": [], "shared": []}),
    ("Nemotron-H\n52 layers", 52,
     {"attn": [8, 19, 30, 41], "swa": [], "shared": []}),
    ("Samba\n64 layers", 64,
     {"attn": [], "swa": list(range(2, 65, 2)), "shared": []}),
    ("Zamba2\n54 layers", 54,
     {"attn": [], "swa": [], "shared": list(range(6, 55, 6))}),
]


def fig_hybrid_stacks():
    """Vertical layer stacks, input at the bottom; attention cells stand out."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    cell_w, gap = 1.3, 2.4                         # column geometry
    unit = 1.0                                     # one layer = one unit tall

    for i, (name, n, kinds) in enumerate(STACKS):
        x0 = i * (cell_w + gap)
        for layer in range(1, n + 1):
            y0 = (layer - 1) * unit
            if layer in kinds["attn"]:
                color, hatch = ORANGE, None
            elif layer in kinds["swa"]:
                color, hatch = GREEN, None
            elif layer in kinds["shared"]:
                color, hatch = ORANGE, "///"
            else:
                color, hatch = BLUE, None
            face = color if hatch is None else "none"
            ax.add_patch(Rectangle(
                (x0, y0), cell_w, 0.86 * unit, facecolor=face,
                edgecolor=color, hatch=hatch, lw=0.9,
                alpha=0.85 if hatch is None else 1.0))
        ax.text(x0 + cell_w / 2, -4.2, name, ha="center", va="top",
                fontsize=13, color="black")

    # Input arrow: every stack reads from the bottom -- "never at the front"
    # is visible as an all-blue base in every column.
    ax.annotate("", xy=(-1.6, 10.0), xytext=(-1.6, -2.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.text(-1.6, -4.2, "input", ha="center", va="top", fontsize=13,
            color="black")

    # Legend built from proxy patches, outside the stacks on the right; the
    # tight bounding box in save() crops the canvas to the content.
    proxies = [
        (Rectangle((0, 0), 1, 1, facecolor=BLUE, alpha=0.85), "recurrent"),
        (Rectangle((0, 0), 1, 1, facecolor=ORANGE, alpha=0.85),
         "full attention"),
        (Rectangle((0, 0), 1, 1, facecolor=GREEN, alpha=0.85),
         "sliding-window attention"),
        (Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=ORANGE,
                   hatch="///"), "shared attention block"),
    ]
    ax.legend([p for p, _ in proxies], [t for _, t in proxies],
              loc="upper left", bbox_to_anchor=(1.01, 0.92), fontsize=12,
              handlelength=1.2, handleheight=1.2)

    ax.set_xlim(-2.4, len(STACKS) * (cell_w + gap) - gap + 0.4)
    ax.set_ylim(-9.5, 65)
    ax.axis("off")
    fl.save(fig, "mdl-modernrnn-hybrid-stacks")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_hybrid_cache,
    fig_hybrid_stacks,
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
