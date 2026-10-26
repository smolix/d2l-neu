#!/usr/bin/env python3
"""Figures for chapter_convolutional-neural-networks (ch7).

    .venv-pytorch/bin/python tools/gen_arch_convnets_figures.py

Both figures here are MECHANICS diagrams: they follow the existing family
look of ``img/conv-pad.svg`` / ``img/conv-multi-in.svg`` (white grids,
light-blue shaded cells, black strokes), via the grid primitives in
``tools/arch_diagrams.py``.  Byte-idempotent.
"""

from arch_diagrams import MechDiagram, save, INK, SANS


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


def _star_op(d, x, y, fs=17.0):
    """The cross-correlation operator ⋆ between panels, in Computer Modern
    math like the prose (Source Sans 3 has no ⋆; * would mean convolution)."""
    d.ax.text(x, y, r"$\star$", fontsize=1.25 * fs, color=INK,
              math_fontfamily="cm", ha="center", va="center", zorder=8)


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
    _star_op(d, 146, y_top + 45)
    d.grid_stack(170, y_top + 18, 3, 3, 3, cell, off,
                 shaded_front={(r, c) for r in range(3) for c in range(3)},
                 shade_backs=True, title="3×3 kernel, all 3 channels")
    d.op(262, y_top + 45, "=")
    d.grid(288, y_top + 9, 3, 3, cell, title="1 output channel")

    # ---- bottom row: depthwise then pointwise ------------------------------- #
    y_bot = 30
    d.grid_stack(20, y_bot, 3, 5, 5, cell, off,
                 shaded_front={(r, c) for r in (1, 2, 3) for c in (1, 2, 3)},
                 title="Input (3 channels)")
    _star_op(d, 146, y_bot + 45)
    # depthwise: one full 3x3 kernel per channel; wider offset so the three
    # kernels read as separate, each fully active
    d.grid_stack(164, y_bot + 8, 3, 3, 3, cell, 13.0,
                 shaded_front={(r, c) for r in range(3) for c in range(3)},
                 shade_backs=True, title="depthwise: 1 kernel\nper channel")
    d.op(262, y_bot + 45, "=")
    d.grid_stack(288, y_bot + 9, 3, 3, 3, cell, off,
                 title="3 channels")
    _star_op(d, 392, y_bot + 45)
    # pointwise: a 1x1 kernel spanning all channels
    d.grid_stack(416, y_bot + 31, 3, 1, 1, cell, off,
                 shaded_front={(0, 0)}, shade_backs=True,
                 title="pointwise\n1×1")
    d.op(482, y_bot + 45, "=")
    d.grid(508, y_bot + 18, 3, 3, cell, title="1 output channel")

    save(d.fig, "arch-conv-depthwise")


def fig_conv_rf_towers():
    """sec_padding: receptive field of the top unit of three stacked 1-D
    kernels of size 3, (a) strides 2, 2, 1, (b) stride 1 with dilations
    1, 2, 4: the same 15 input pixels, at a quarter of the positions or at
    all of them.  Cells are positions that exist in each layer; shaded cells
    feed the top unit; lines are kernel taps."""
    from arch_diagrams import FAMILY_BLUE, GRID_FS
    from matplotlib.patches import Rectangle
    cell, n = 10.0, 17      # the 15-pixel field plus one free cell per side
    gap_row = 24.0                     # vertical air between layers
    gap_panel = 26.0
    label_w = 44.0                     # room for the row labels
    pw = n * cell
    W = label_w + 2 * pw + gap_panel + 4
    H = 4 * cell + 3 * gap_row + 30
    d = MechDiagram(W, H)
    ys = [4 + i * (cell + gap_row) for i in range(4)]   # input .. layer 3

    def tower(x0, steps, title):
        """``steps[l]`` = (spacing of layer l+1's positions, tap distance of
        its kernel), both in input pixels."""
        # positions present in each layer, and the ones feeding the top unit
        spacing = [1] + [s for s, _ in steps]
        present = [list(range(0, n)) if sp == 1 else
                   [p for p in range(n) if p % sp == (n // 2) % sp]
                   for sp in spacing]
        feed = [None] * 4
        feed[3] = {n // 2}
        for l in (3, 2, 1):
            tap = steps[l - 1][1]
            feed[l - 1] = {p + o * tap for p in feed[l] for o in (-1, 0, 1)}
        for l in range(4):
            for p in present[l]:
                d.ax.add_patch(Rectangle(
                    (x0 + p * cell, ys[l]), cell, cell, zorder=4,
                    facecolor=FAMILY_BLUE if p in feed[l] else "white",
                    edgecolor=INK, linewidth=1.2))
        for l in (3, 2, 1):
            tap = steps[l - 1][1]
            for p in sorted(feed[l]):
                for o in (-1, 0, 1):
                    q = p + o * tap
                    d.ax.plot([x0 + (p + 0.5) * cell, x0 + (q + 0.5) * cell],
                              [ys[l], ys[l - 1] + cell], color=INK,
                              linewidth=0.8, zorder=3,
                              solid_capstyle="butt")
        d.ax.text(x0 + pw / 2, ys[3] + cell + 9, title, fontsize=GRID_FS,
                  color=INK, family=SANS, ha="center", va="bottom")

    tower(label_w, [(2, 1), (4, 2), (4, 4)], "(a) strides 2, 2, 1")
    tower(label_w + pw + gap_panel, [(1, 1), (1, 2), (1, 4)],
          "(b) dilations 1, 2, 4")
    for l, name in enumerate(("input", "layer 1", "layer 2", "layer 3")):
        d.ax.text(label_w - 6, ys[l] + cell / 2, name, fontsize=11.0,
                  color=INK, family=SANS, ha="right", va="center")
    save(d.fig, "conv-rf-towers")


def fig_conv_groups():
    """sec_channels: which input channels each output channel reads, for 8
    input and 8 output channels: dense, 2 groups, 4 groups, depthwise.  Each
    shaded cell is one k×k spatial kernel."""
    from arch_diagrams import GRID_FS
    cell, c = 11.0, 8
    gap = 30.0
    side = c * cell
    x_lab = 18.0                       # room for the rotated axis label
    W = x_lab + 4 * side + 3 * gap + 4
    H = side + 50
    d = MechDiagram(W, H)
    y0 = 22.0
    for i, g in enumerate((1, 2, 4, 8)):
        x0 = x_lab + i * (side + gap)
        b = c // g
        on = {(r, q) for r in range(c) for q in range(c) if r // b == q // b}
        d.grid(x0, y0, c, c, cell, shaded=on)
        name = {1: "dense", 8: "depthwise"}.get(g, f"{g} groups")
        d.ax.text(x0 + side / 2, y0 + side + 8,
                  f"{name}\n{c * c // g} kernels", fontsize=GRID_FS,
                  color=INK, family=SANS, ha="center", va="bottom",
                  linespacing=1.25)
    d.ax.text(x_lab + side / 2, y0 - 7, "input channel", fontsize=11.0,
              color=INK, family=SANS, ha="center", va="top")
    d.ax.text(x_lab - 7, y0 + side / 2, "output channel", fontsize=11.0,
              color=INK, family=SANS, ha="center", va="bottom",
              rotation=90, rotation_mode="anchor")
    save(d.fig, "conv-groups")


def fig_lenet_vert():
    """The LeNet of sec_lenet as a vertical spine (input at the bottom), in the
    pill style of img/arch-alexnet.svg: an encoder panel and a head panel, with
    the shape of the tensor on each arrow."""
    from arch_diagrams import (Diagram, PILL_H, PILL_STEP, ACCENT_TINT,
                               CONTAINER_FILL, LABEL_BAND)
    encoder = ["5×5 Conv, 6, pad 2, sigmoid", "2×2 AvgPool, stride 2",
               "5×5 Conv, 16, sigmoid", "2×2 AvgPool, stride 2"]
    head = ["Flatten", "Dense 120, sigmoid", "Dense 84, sigmoid", "Dense 10"]
    # shape of each op's output, channels@height×width
    shapes = ["6@28×28", "6@14×14", "16@10×10", "16@5×5",
              "400", "120", "84", "10"]
    gap = 58                      # extra rise between the two panels
    d = Diagram(330, 520)
    x = 120
    pw = max(d.pill_width(s) for s in encoder + head) + 4
    y0 = 70
    ys = [y0 + i * PILL_STEP for i in range(4)]
    ys += [ys[-1] + gap + PILL_STEP + i * PILL_STEP for i in range(4)]
    pad_x, pad_lo = 18, PILL_H / 2 + 14
    pad_hi = PILL_H / 2 + LABEL_BAND - 8
    # one accent per figure: the encoder is the new idea, the head is neutral
    for lo, hi, tint, label in ((0, 3, ACCENT_TINT, "convolutional encoder"),
                                (4, 7, CONTAINER_FILL, "dense head")):
        panel = (x - pw / 2 - pad_x, ys[lo] - pad_lo,
                 x + pw / 2 + pad_x, ys[hi] + pad_hi)
        d.container(*panel, fill=tint)
        d.stage_label(panel[0] + 12, panel[3] - 8, label)
    y_anchor = y0 - pad_lo - 34
    d.anchor(x, y_anchor, "28×28 image")
    d.arrow(x, y_anchor + 12, ys[0] - PILL_H / 2)
    x_note = x + pw / 2 + pad_x + 8
    for i, (op, y) in enumerate(zip(encoder + head, ys)):
        d.pill(x, y, op, w=pw)
        y_next = ys[i + 1] - PILL_H / 2 if i < 7 else ys[-1] + 70
        d.arrow(x, y + PILL_H / 2, y_next)
        if i in (3, 7):   # arrow leaves a panel: note beside its free part
            y_lo = y + pad_hi
            y_hi = ys[i + 1] - pad_lo if i < 7 else y_next
            d.shape_note(x + 8, 0.5 * (y_lo + y_hi), shapes[i])
        else:             # arrow inside a panel: note in the right margin
            d.shape_note(x_note, 0.5 * (y + ys[i + 1]), shapes[i])
    save(d.fig, "lenet-vert")


def fig_why_conv_weight_sharing():
    """sec_why-conv: the weight matrix of a linear layer on a 1-D signal of
    8 samples, (a) fully connected, (b) translation equivariant (Toeplitz:
    weights depend only on the offset), (c) equivariant and local (banded).
    Equal colors are equal weights; white is a zero weight."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    from figstyle import tokens as T
    from arch_diagrams import GRID_FS

    n, cell, gap = 8, 17.0, 46.0
    side = n * cell
    cmap = LinearSegmentedColormap.from_list(
        "weights", ["#CFE3F3", T.BLUE.base, "#123F66"])
    rng = np.random.default_rng(6)            # fixed: byte-idempotent output
    dense = rng.uniform(0.0, 1.0, (n, n))
    by_offset = rng.uniform(0.0, 1.0, 2 * n - 1)   # index = (j - i) + n - 1
    by_offset[n - 2:n + 1] = (0.15, 0.95, 0.55)    # offsets -1, 0, +1
    panels = [
        ("fully connected", f"{n * n} weights",
         lambda i, j: dense[i, j]),
        ("shared", f"{2 * n - 1} weights",
         lambda i, j: by_offset[j - i + n - 1]),
        ("shared and local", "3 weights",
         lambda i, j: by_offset[j - i + n - 1] if abs(j - i) <= 1 else None),
    ]
    x_left = 30.0
    d = MechDiagram(x_left + 3 * side + 2 * gap + 10, side + 100)
    y0 = 56.0
    for p, (title, count, weight) in enumerate(panels):
        x0 = x_left + p * (side + gap)
        for i in range(n):            # row i = output, drawn top to bottom
            for j in range(n):        # column j = input
                v = weight(i, j)
                d.ax.add_patch(Rectangle(
                    (x0 + j * cell, y0 + (n - 1 - i) * cell), cell, cell,
                    facecolor="white" if v is None else cmap(v),
                    edgecolor="white", linewidth=1.0, zorder=3))
        d.ax.add_patch(Rectangle((x0, y0), side, side, facecolor="none",
                                 edgecolor=INK, linewidth=1.4, zorder=5))
        d.ax.text(x0 + side / 2, y0 + side + 9, f"({'abc'[p]}) {title}",
                  fontsize=GRID_FS, color=INK, family=SANS, ha="center",
                  va="bottom")
        d.ax.text(x0 + side / 2, y0 - 30, count, fontsize=GRID_FS,
                  color=INK, family=SANS, ha="center", va="top",
                  fontweight="bold")
    # axis names once, on panel (a): rows are outputs, columns are inputs
    d.ax.text(x_left + side / 2, y0 - 7, "input", fontsize=GRID_FS - 1,
              color=INK, family=SANS, ha="center", va="top")
    d.ax.text(x_left - 8, y0 + side / 2, "output", fontsize=GRID_FS - 1,
              color=INK, family=SANS, ha="center", va="bottom",
              rotation=90, rotation_mode="anchor")
    save(d.fig, "why-conv-weight-sharing")


def fig_conv_receptive_field():
    """sec_conv_layer: receptive field of one unit after three 1-D layers with
    kernel 3 (7 inputs).  Shaded cells are the units the top unit depends on.
    The strided case belongs to sec_padding (fig_conv_rf_towers)."""
    from matplotlib.patches import Rectangle
    from arch_diagrams import GRID_FS, FAMILY_BLUE

    cell, row_gap, k = 16.0, 34.0, 3

    def panel(d, x0, y0, n_in, strides, title):
        # position (in input-cell units) of unit u of each layer: layer 0 is
        # the input; a unit of layer l sits over the center of its window
        rows, step, offset = [], 1, 0.0
        n = n_in
        rows.append([offset + u * step for u in range(n)])
        for s in strides:
            offset += (k - 1) / 2 * step
            step *= s
            n = (n - k) // s + 1
            rows.append([offset + u * step for u in range(n)])
        # receptive field: walk down from the center unit of the top layer
        top = len(rows[-1]) // 2
        active = [set() for _ in rows]
        active[-1] = {top}
        for l in range(len(rows) - 1, 0, -1):
            s = strides[l - 1]
            for u in active[l]:
                active[l - 1].update(range(u * s, u * s + k))
        ys = [y0 + l * row_gap for l in range(len(rows))]
        for l, pos in enumerate(rows):
            for u, p in enumerate(pos):
                if l == len(rows) - 1 and u != top:
                    continue          # show only the one top unit
                d.ax.add_patch(Rectangle(
                    (x0 + p * cell, ys[l]), cell, cell, zorder=4,
                    facecolor=FAMILY_BLUE if u in active[l] else "white",
                    edgecolor=INK, linewidth=1.2))
            if l > 0:                 # connections of the active units
                s = strides[l - 1]
                for u in active[l]:
                    for t in range(u * s, u * s + k):
                        d.ax.plot([x0 + (pos[u] + 0.5) * cell,
                                   x0 + (rows[l - 1][t] + 0.5) * cell],
                                  [ys[l], ys[l - 1] + cell], color=INK,
                                  linewidth=0.8, zorder=3)
        if title:
            d.ax.text(x0 + n_in * cell / 2, ys[-1] + cell + 12, title,
                      fontsize=GRID_FS, color=INK, family=SANS, ha="center",
                      va="bottom")
        return ys, len(active[0])

    # 11 inputs: the top layer has a center unit
    n_in = 11
    d = MechDiagram(n_in * cell + 90, 3 * row_gap + 60)
    x0, y0 = 70.0, 14.0
    ys, r = panel(d, x0, y0, n_in, (1, 1, 1), None)
    assert r == 7
    for l, name in enumerate(("input", "layer 1", "layer 2", "layer 3")):
        d.ax.text(x0 - 10, ys[l] + cell / 2, name, fontsize=GRID_FS - 1,
                  color=INK, family=SANS, ha="right", va="center")
    save(d.fig, "conv-receptive-field")


def fig_lenet():
    """sec_lenet: data flow through LeNet on one 28×28 image, drawn to scale
    (3 pt per pixel): feature-map stacks for the two convolutional blocks,
    bars for the three fully connected layers, a shaded window and its
    output unit for each convolution and pooling step."""
    from matplotlib.patches import Rectangle, Polygon
    from arch_diagrams import FAMILY_BLUE, GRID_FS
    px = 3.0                        # points per pixel
    lw = 1.2
    d = MechDiagram(800, 215)
    y0 = 52.0                       # bottom of every front plane
    gap = 46.0                      # horizontal air between stages

    def plane(x, y, size, z, fill="white"):
        d.ax.add_patch(Rectangle((x, y), size, size, facecolor=fill,
                                 edgecolor=INK, linewidth=lw, zorder=z))

    def stack(x, n, pixels, offset, title):
        """n planes, front plane's lower-left at (x, y0); deeper planes
        offset up-right.  Returns the front plane's (x, y, size)."""
        size = pixels * px
        for i in range(n - 1, -1, -1):
            plane(x + i * offset, y0 + i * offset, size, z=20 - i)
        top = y0 + size + (n - 1) * offset
        d.ax.text(x + (size + (n - 1) * offset) / 2, top + 8, title,
                  fontsize=GRID_FS, color=INK, family=SANS, ha="center",
                  va="bottom")
        return x, y0, size, x + size + (n - 1) * offset

    def window(x, y, size, col, row, k, z=30):
        """Shade a k×k pixel window at pixel (col, row) of a front plane
        (row 0 at the top).  Returns its right-edge midpoint."""
        wx = x + col * px
        wy = y + size - (row + k) * px
        d.ax.add_patch(Rectangle((wx, wy), k * px, k * px,
                                 facecolor=FAMILY_BLUE, edgecolor=INK,
                                 linewidth=lw, zorder=z, alpha=0.95))
        return wx + k * px, wy, wy + k * px

    def cone(src, dst):
        """Lines from the top and bottom right corners of a window to the
        center of the output unit it produces."""
        xr, yb, yt = src
        for yy in (yb, yt):
            d.ax.plot([xr, dst[0]], [yy, dst[1]], color=INK, linewidth=0.9,
                      zorder=40)

    def unit(x, y, size, col, row):
        """Shade one pixel of a front plane; return its center."""
        ux, uy = x + col * px, y + size - (row + 1) * px
        d.ax.add_patch(Rectangle((ux, uy), px, px, facecolor=INK,
                                 edgecolor=INK, linewidth=0.6, zorder=41))
        return ux + px / 2, uy + px / 2

    def op_label(x0, x1, lines):
        d.ax.text((x0 + x1) / 2, y0 - 12, "\n".join(lines),
                  fontsize=GRID_FS - 1, color=INK, family=SANS,
                  ha="center", va="top", linespacing=1.15)

    # input: a 28x28 image with a shirt silhouette
    x = 10.0
    plane(x, y0, 28 * px, z=10)
    shirt = [(9, 4), (12, 4), (14, 6), (16, 4), (19, 4), (25, 9), (22, 13),
             (20, 11), (20, 24), (8, 24), (8, 11), (6, 13), (3, 9)]
    d.ax.add_patch(Polygon([(x + c * px, y0 + (28 - r) * px) for c, r in shirt],
                           closed=True, facecolor="#9a9a9a", edgecolor="none",
                           zorder=11))
    d.ax.text(x + 14 * px, y0 + 28 * px + 8, "28×28 image", fontsize=GRID_FS,
              color=INK, family=SANS, ha="center", va="bottom")
    w_in = window(x, y0, 28 * px, 13, 9, 5)
    x_end = x + 28 * px

    x1 = x_end + gap
    c1 = stack(x1, 6, 28, 6.0, "6@28×28")
    cone(w_in, unit(*c1[:3], 15, 11))
    op_label(x_end, x1, ["5×5 conv,", "sigmoid"])
    w_c1 = window(*c1[:3], 20, 4, 2)

    x2 = c1[3] + gap
    s2 = stack(x2, 6, 14, 6.0, "6@14×14")
    cone(w_c1, unit(*s2[:3], 10, 2))
    op_label(c1[3], x2, ["2×2 avg", "pool"])
    w_s2 = window(*s2[:3], 1, 4, 5)

    x3 = s2[3] + gap
    c3 = stack(x3, 16, 10, 3.5, "16@10×10")
    cone(w_s2, unit(*c3[:3], 1, 4))
    op_label(s2[3], x3, ["5×5 conv,", "sigmoid"])
    w_c3 = window(*c3[:3], 6, 2, 2)

    x4 = c3[3] + gap
    s4 = stack(x4, 16, 5, 3.5, "16@5×5")
    cone(w_c3, unit(*s4[:3], 3, 1))
    op_label(c3[3], x4, ["2×2 avg", "pool"])

    # fully connected layers as bars, heights proportional to their widths
    bar_w = 14.0
    y_mid = y0 + 14 * px
    xs = [s4[3] + gap + 10]
    heights = [120 * 0.85, 84 * 0.85, 20.0]
    for _ in heights[1:]:
        xs.append(xs[-1] + bar_w + gap - 4)
    prev = (s4[0] + s4[2] / 2, s4[1], s4[1] + s4[2])   # flatten from S4
    for i, (xb, h, name) in enumerate(zip(xs, heights, ("120", "84", "10"))):
        yb, yt = y_mid - h / 2, y_mid + h / 2
        d.ax.add_patch(Rectangle((xb, yb), bar_w, h, facecolor="white",
                                 edgecolor=INK, linewidth=lw, zorder=20))
        d.ax.text(xb + bar_w / 2, yt + 8, name, fontsize=GRID_FS, color=INK,
                  family=SANS, ha="center", va="bottom")
        if i == 0:   # from the front plane's lower right, the back's upper right
            starts = ((s4[0] + s4[2], y0), (s4[3], y0 + s4[2] + 15 * 3.5))
        else:
            xp = xs[i - 1] + bar_w
            starts = ((xp, y_mid - heights[i - 1] / 2),
                      (xp, y_mid + heights[i - 1] / 2))
        for (xa, ya), yb2 in zip(starts, (yb, yt)):
            d.ax.plot([xa, xb], [ya, yb2], color=INK, linewidth=0.9,
                      zorder=15)
        x_prev = s4[3] if i == 0 else xs[i - 1] + bar_w
        op_label(x_prev, xb, ["flatten,", "dense,", "sigmoid"] if i == 0 else
                 (["dense,", "sigmoid"] if i == 1 else ["dense"]))
    save(d.fig, "lenet")


if __name__ == "__main__":
    fig_conv_receptive_field()
    fig_why_conv_weight_sharing()
    fig_conv_dilation()
    fig_conv_depthwise()
    fig_conv_rf_towers()
    fig_conv_groups()
    fig_lenet_vert()
    fig_lenet()
    from arch_diagrams import WRITTEN
    print("\n".join(WRITTEN))
