#!/usr/bin/env python3
"""Figures for chapter_convolutional-neural-networks (ch7).

    .venv-pytorch/bin/python tools/gen_arch_convnets_figures.py

Both figures here are MECHANICS diagrams: they follow the existing family
look of ``img/conv-pad.svg`` / ``img/conv-multi-in.svg`` (white grids,
light-blue shaded cells, black strokes), via the grid primitives in
``tools/arch_diagrams.py``.  Byte-idempotent.
"""

from arch_diagrams import MechDiagram, save, INK, GRAY_TEXT, SANS


def fig_conv_dilation():
    """3×3 kernel taps at dilation 1 vs dilation 2 on the same input grid."""
    cell = 20.0
    n = 7                      # input grid is 7×7
    W, H = 2 * n * cell + 130, n * cell + 58
    d = MechDiagram(W, H)
    y0 = 12

    from matplotlib.patches import Rectangle

    def footprint(x, y, span):
        """Dashed outline slightly outside the effective footprint so it
        does not coincide with (and vanish into) the cell borders."""
        m = 3.5
        d.ax.add_patch(Rectangle(
            (x - m, y - m), span * cell + 2 * m, span * cell + 2 * m,
            facecolor="none", edgecolor=INK, linewidth=1.3,
            linestyle=(0, (2.4, 2.4)), zorder=6))

    # left: dilation 1 — contiguous 3×3 footprint, centered on the grid
    x0 = 14
    taps1 = {(r, c) for r in (2, 3, 4) for c in (2, 3, 4)}
    d.grid(x0, y0, n, n, cell, shaded=taps1,
           title="Dilation 1: 3×3 footprint")
    footprint(x0 + 2 * cell, y0 + 2 * cell, 3)

    # right: dilation 2 — same nine taps spread over a 5×5 footprint
    x1 = x0 + n * cell + 90
    taps2 = {(r, c) for r in (1, 3, 5) for c in (1, 3, 5)}
    d.grid(x1, y0, n, n, cell, shaded=taps2,
           title="Dilation 2: 5×5 footprint")
    footprint(x1 + cell, y0 + cell, 5)

    # the kernel itself, between the panels
    xk = x0 + n * cell + 45
    d.grid(xk - 1.5 * cell + 2, y0 + (n * cell) / 2 - 1.5 * cell, 3, 3,
           cell, shaded={(r, c) for r in range(3) for c in range(3)},
           title=None)
    d.ax.text(xk + 2, y0 + (n * cell) / 2 + 1.5 * cell + 8, "3×3 kernel",
              fontsize=10.0, color=INK, family=SANS, ha="center",
              va="bottom")

    save(d.fig, "arch-conv-dilation")


def fig_conv_depthwise():
    """Dense conv mixes all channels; depthwise filters each channel
    separately, then a pointwise 1×1 mixes them."""
    cell = 18.0
    off = 7.0
    d = MechDiagram(760, 300)

    def stack_h(n_rows, n):
        return n_rows * cell + 2 * off   # height of a 3-deep stack

    # ---- top row: dense 3×3 convolution ------------------------------------ #
    y_top = 175
    d.grid_stack(20, y_top, 3, 5, 5, cell, off,
                 shaded_front={(r, c) for r in (1, 2, 3) for c in (1, 2, 3)},
                 title="Input (3 channels)")
    d.op(146, y_top + 45, "*")
    d.grid_stack(170, y_top + 18, 3, 3, 3, cell, off,
                 shaded_front={(r, c) for r in range(3) for c in range(3)},
                 shade_backs=True, title="3×3 kernel, all 3 channels")
    d.op(262, y_top + 45, "=")
    d.grid(288, y_top + 9, 3, 3, cell, title="1 output channel")
    d.ax.text(420, y_top + 45,
              "dense: every output channel\nreads every input channel",
              fontsize=10.0, color=GRAY_TEXT, family=SANS, ha="left",
              va="center", style="italic", linespacing=1.4)

    # ---- bottom row: depthwise then pointwise ------------------------------- #
    y_bot = 30
    d.grid_stack(20, y_bot, 3, 5, 5, cell, off,
                 shaded_front={(r, c) for r in (1, 2, 3) for c in (1, 2, 3)},
                 title="Input (3 channels)")
    d.op(146, y_bot + 45, "*")
    # depthwise: one full 3x3 kernel per channel; wider offset so the three
    # kernels read as separate, each fully active
    d.grid_stack(164, y_bot + 8, 3, 3, 3, cell, 13.0,
                 shaded_front={(r, c) for r in range(3) for c in range(3)},
                 shade_backs=True, title="depthwise: 1 kernel\nper channel")
    d.op(262, y_bot + 45, "=")
    d.grid_stack(288, y_bot + 9, 3, 3, 3, cell, off,
                 title="3 channels")
    d.op(392, y_bot + 45, "*")
    # pointwise: a 1x1 kernel spanning all channels
    d.grid_stack(416, y_bot + 31, 3, 1, 1, cell, off,
                 shaded_front={(0, 0)}, shade_backs=True,
                 title="pointwise\n1×1")
    d.op(482, y_bot + 45, "=")
    d.grid(508, y_bot + 18, 3, 3, cell, title="1 output channel")
    d.ax.text(640, y_bot + 45,
              "depthwise separable:\nfilter per channel,\nthen mix with 1×1",
              fontsize=10.0, color=GRAY_TEXT, family=SANS, ha="left",
              va="center", style="italic", linespacing=1.4)

    save(d.fig, "arch-conv-depthwise")


if __name__ == "__main__":
    fig_conv_dilation()
    fig_conv_depthwise()
    from arch_diagrams import WRITTEN
    print("\n".join(WRITTEN))
