#!/usr/bin/env python3
"""Generate the illustrative figures of ``chapter_lm-hardware`` (Chapter 22,
"Hardware for Training and Inference") in the shared house style.

Twelve figures (chapter spec part 1.11), written to
``img/mdl-lm-hw-<id>.svg``, each included with its ``fig_lm_hw_*`` label:

  chip-budget       22.1  a chip as a budget of die area, power and perimeter
  tile-reuse        22.2  operand reuse of an output tile (the tile intensity)
  flash-tiles       22.2  the tiles of fused attention and the running softmax
  three-workloads   22.3  training, prefill and decode on the roofline
  state-placement   22.4  which bytes of training and serving state live where
  server-graph      22.5  the paths inside a server (GPUs, switches, host, NIC)
  bisection         22.5  the busiest link of an all-reduce and an all-to-all
  two-networks      22.6  the scale-up domain and the scale-out network
  fat-tree          22.6  a two-level fat tree and its bisection
  ring-tree         22.7  ring and tree all-reduce schedules
  protocols         22.7  NCCL's Simple, LL and LL128 protocols
  power-model       22.8  power against clock: static, dynamic, the cap

The box's own topology is the existing ``fig_pcie_topology`` (cited, not
redrawn). Computed plots (ladders, sweeps, heat maps, NCCL curves, the DVFS
fit, the tornado) are drawn by the section's library, not here. This is the
generator's skeleton (infra step); the figures are added with the render and
inspect loop of CLAUDE.md and checked with the figure-style-audit skill.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_mdl_lm_hardware_figures.py

Byte-idempotent: no timestamps, seeded randomness only, fixed svg.hashsalt
(through gen_mdl_figures.save).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # noqa: E402  (applies the shared rcParams)

PREFIX = "mdl-lm-hw"

# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #

FIGURES = []


def main():
    start = len(fl.WRITTEN)
    for fn in FIGURES:
        fn()
    written = fl.WRITTEN[start:]
    print(f"\nWrote {len(written)} figures to {fl.IMG_DIR}:")
    for p in written:
        size = os.path.getsize(p)
        assert size > 0, f"empty: {p}"
        with open(p, "r", encoding="utf-8") as fh:
            assert "<svg" in fh.read(400), f"not valid SVG: {p}"
        print(f"  {os.path.basename(p):40s} {size:>8,d} bytes")
    print(f"\nAll {len(written)} SVGs verified present, non-empty, valid.")


if __name__ == "__main__":
    main()
