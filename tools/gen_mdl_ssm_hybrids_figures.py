#!/usr/bin/env python3
"""Generate the illustrative figures for the hybrid-architectures section of
``chapter_recurrent-modern`` (12.7 "Hybrid Architectures") in the shared house
style defined in ``gen_mdl_figures.py``.

One figure (the "hybrid stacks" schematic was dropped with the 2026-10
chapter rewrite, which no longer references it; see git history):

  * the hybrid cache curve -- persistent decode state against context length
    for a 32-layer model at production width: a pure transformer's KV cache
    grows linearly (32 GB at 256K), a pure recurrent stack is flat (64 MB),
    and a 4-of-32 hybrid pays exactly the attention fraction of the
    transformer bill.  All three curves are computed from the KV-cache byte
    formula of the transformers chapter plus the Mamba-2 state-size
    accounting.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_ssm_hybrids_figures.py

The figure is written to ``img/mdl-modernrnn-<id>.svg``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
TEAL = fl.T.TEAL.base   # recurrent/state role (guide §4.3), as in ch. 9/10


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
    ax.loglog(T, recurrent, color=TEAL, lw=2.4,
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
    ax.set_ylabel("persistent decode state", fontsize=13, color="black")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
    ax.grid(True, which="major", linestyle="--", color=LIGHT, lw=0.8)
    ax.legend(loc="upper left", fontsize=12)
    fl.save(fig, "mdl-modernrnn-hybrid-cache")


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = [
    fig_hybrid_cache,
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
