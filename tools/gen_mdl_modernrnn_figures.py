#!/usr/bin/env python3
"""Generate the illustrative figures for the "Modern Recurrent Neural
Networks" chapter (``chapter_recurrent-modern``) in the one shared house style
defined in ``gen_mdl_figures.py``.

The notebooks / prose reference the generated files with no drawing code (like
the slide SVGs).  Figures that show a *computed* result (e.g. gradient-norm
decay along a chain, gate-activation traces) should use real numerical
computation so the pictures are exact, not sketches; purely schematic figures
(GRU/LSTM cell wiring, deep/bidirectional/encoder-decoder unrolled-graph
diagrams, ...) use ``set_aspect("equal")`` and the shared drawing helpers
(``arrow``/``vlabel``/``clean_axes``/``axis_cross``/``right_angle``).

Run with the repo's pytorch venv (matplotlib + numpy are available):

    .venv-pytorch/bin/python tools/gen_mdl_modernrnn_figures.py

All figures are written to ``img/mdl-modernrnn-<id>.svg``.

Skeleton: no figures are registered yet.  Add ``fig_...`` functions below (one
per figure, following the pattern in ``gen_mdl_mlp_figures.py`` or
``gen_mdl_calculus_figures.py``), each ending with a call to ``fl.save(fig,
"mdl-modernrnn-<id>")``, then append the function to ``FIGURES`` below.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers

np, plt = fl.np, fl.plt
BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT


# =========================================================================== #
# Figures go here.  One function per figure; end with
# fl.save(fig, "mdl-modernrnn-<id>").
# =========================================================================== #


# =========================================================================== #
# Driver                                                                      #
# =========================================================================== #

FIGURES = []


def main():
    # Verify only the figures THIS script writes (the shared module's WRITTEN
    # list also tracks the Linear Algebra figures, which we don't run here).
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
