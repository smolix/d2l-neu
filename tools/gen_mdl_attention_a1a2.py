#!/usr/bin/env python3
"""Generate the illustrative figures for the "Attention" chapter sections A1
(queries-keys-values.md) and A2 (attention-scoring.md) in the shared house
style defined in ``gen_mdl_figures.py``.

Two figures:

  * ``mdl-attention-kernels`` (A1, sec_attention-pooling) -- the four
    similarity kernels (Gaussian, boxcar, constant, triangular) as a 1x4
    strip; replaces the old in-notebook 4-panel matplotlib cell.
  * ``mdl-attention-alignment`` (A2, From Alignment to Attention) -- a
    schematic soft-alignment matrix between an English source sentence and
    its French translation, in the style of a learned attention map (mostly
    monotone, with the adjective--noun order swap for "black cat" ->
    "chat noir").  House-generated; deliberately NOT the Bahdanau paper's
    figure.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_attention_a1a2.py

All figures are written to ``img/mdl-attention-<id>.svg``.  Deterministic
(no random draws), hence byte-idempotent via the shared ``save()``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


# --------------------------------------------------------------------------- #
# A1: the four similarity kernels.                                            #
# --------------------------------------------------------------------------- #

def fig_kernels():
    """Gaussian, boxcar, constant, and triangular kernels on [-2.5, 2.5]."""
    x = np.linspace(-2.5, 2.5, 501)
    kernels = [
        ("Gaussian", np.exp(-x**2 / 2)),
        ("boxcar", (np.abs(x) <= 1.0).astype(float)),
        ("constant", np.ones_like(x)),
        ("triangular", np.maximum(0.0, 1.0 - np.abs(x))),
    ]
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(9, 2.4))
    for ax, (name, y) in zip(axes, kernels):
        ax.plot(x, y, color=BLUE, lw=2.0)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.06, 1.12)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=10, colors="black")
        ax.set_xlabel(name, fontsize=14, color="black")
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("black")
    axes[0].set_ylabel(r"$\alpha(q, k)$", fontsize=14, color="black")
    fig.subplots_adjust(wspace=0.12)
    fl.save(fig, "mdl-attention-kernels")


# --------------------------------------------------------------------------- #
# A2: schematic soft alignment (English -> French).                           #
# --------------------------------------------------------------------------- #

def fig_alignment():
    """Soft-alignment matrix for "the black cat sat on the mat" -> French.

    Rows are target (French) tokens, columns are source (English) tokens.
    Weights are hand-designed Gaussian bumps around a primary alignment,
    normalized per row: mostly monotone, with the order swap for
    "black cat" -> "chat noir" and two target words ("etait", "assis")
    drawing on the single source word "sat".
    """
    src = ["the", "black", "cat", "sat", "on", "the", "mat"]
    tgt = ["le", "chat", "noir", "était", "assis", "sur", "le", "tapis"]
    primary = [0, 2, 1, 3, 3, 4, 5, 6]  # target row -> source column

    cols = np.arange(len(src))
    W = np.stack([np.exp(-(cols - p) ** 2 / (2 * 0.55**2)) for p in primary])
    W = W / W.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.6, 5.0))
    ax.imshow(W, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(cols)
    ax.set_xticklabels(src, fontsize=13, color="black")
    ax.set_yticks(np.arange(len(tgt)))
    ax.set_yticklabels(tgt, fontsize=13, color="black")
    ax.set_xlabel("source (English)", fontsize=14, color="black")
    ax.set_ylabel("target (French)", fontsize=14, color="black")
    # thin white separators between cells
    ax.set_xticks(cols[:-1] + 0.5, minor=True)
    ax.set_yticks(np.arange(len(tgt) - 1) + 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, colors="black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fl.save(fig, "mdl-attention-alignment")


ALL_FIGURES = [fig_kernels, fig_alignment]


def main():
    for f in ALL_FIGURES:
        f()
    for path in fl.WRITTEN:
        print(f"wrote {os.path.relpath(path, os.path.dirname(fl.IMG_DIR))}")


if __name__ == "__main__":
    main()
