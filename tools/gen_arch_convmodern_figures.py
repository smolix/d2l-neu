#!/usr/bin/env python3
"""Gallery-style architecture figures for chapter_convolutional-modern (ch8).

    .venv-pytorch/bin/python tools/gen_arch_convmodern_figures.py

Writes ``img/arch-<id>.svg`` (byte-idempotent).  Style + primitives:
``tools/arch_diagrams.py`` / docs/convnet-rewrite/figure-style.md.
Layout rules from the pilot review: one spine rhythm (PILL_STEP) everywhere,
LABEL_BAND of headroom under each panel label, axis-aligned leaders only,
repeat braces span their full panel.
"""

from arch_diagrams import SANS as SANS_FONT
from arch_diagrams import (
    Diagram, save,
    ACCENT, ACCENT_TINT, ACCENT2, ACCENT2_TINT, INK, CONTAINER_FILL,
    GRAY_TEXT, PILL_H, PILL_STEP, LABEL_BAND, PLUS_R,
)

LH = 12.5 * 1.25          # callout line height (must match callout())
BLOCK_PAD = 12            # horizontal inner-block padding around its pills
BLOCK_VPAD = 7            # vertical inner-block padding (tighter than horizontal)
BLOCK_GAP = 14            # vertical gap between stacked groups
BLOCK_LABEL_BAND = 12     # extra headroom inside a tinted block for its label


# --------------------------------------------------------------------------- #
# Pilot 1: ResNet-50 bottleneck block vs. ConvNeXt block (two accents).       #
# --------------------------------------------------------------------------- #

def fig_resnet_vs_convnext_block():
    W, H = 720, 590
    d = Diagram(W, H)

    y_anchor = 30                # input-x anchor (shared)
    y0 = 116                     # first pill center (shared baseline)
    OUT_LEN = 32                 # short output arrow beyond the panel top

    def column(x, ops, pw, tint, label, novelty_idx=None, extra_after_plus=None):
        """One block panel: ops on a constant PILL_STEP rhythm, ⊕ one step
        above the last op, optional trailing op, skip routed inside."""
        ys = [y0 + i * PILL_STEP for i in range(len(ops))]
        y_plus = ys[-1] + PILL_STEP
        y_top_el = y_plus + (PILL_STEP if extra_after_plus else 0)
        top_edge = (y_top_el + (PILL_H / 2 if extra_after_plus else PLUS_R)
                    + LABEL_BAND)
        panel = (x - 97, y0 - PILL_H / 2 - 23, x + 97, top_edge)
        y_out = top_edge + OUT_LEN
        d.container(*panel, fill=tint)
        d.stage_label(panel[0] + 14, panel[3] - 9, label)

        d.arrow(x, y_anchor + 14, ys[0] - PILL_H / 2)
        for i, (op, y) in enumerate(zip(ops, ys)):
            if i == novelty_idx:
                pre, kw, post = op
                d.novelty(x, y, pre, kw, post, w=pw + 18)
            else:
                d.pill(x, y, op, w=pw)
            if i < len(ops) - 1:
                d.arrow(x, y + PILL_H / 2, ys[i + 1] - PILL_H / 2)
        d.arrow(x, ys[-1] + PILL_H / 2, y_plus - PLUS_R)
        # skip: branch just below the first pill, route inside the panel
        x_out = x + 76 if novelty_idx is None else x - 76
        d.skip(x, y0 - PILL_H / 2 - 11, y_plus, x_out)
        if extra_after_plus:
            d.arrow(x, y_plus + PLUS_R, y_top_el - PILL_H / 2)
            d.pill(x, y_top_el, extra_after_plus, w=pw)
            d.arrow(x, y_top_el + PILL_H / 2, y_out)
        else:
            d.arrow(x, y_plus + PLUS_R, y_out)
        d.anchor(x, y_anchor, "input x")
        return ys, y_plus, panel

    # ----- left: ResNet-50 bottleneck (accent 2, amber) -------------------- #
    xl, pw_l = 215, 108
    resnet_ops = ["1×1 Conv, 64", "BatchNorm", "ReLU",
                  "3×3 Conv, 64", "BatchNorm", "ReLU",
                  "1×1 Conv, 256", "BatchNorm"]
    ys_l, y_plus_l, panel_l = column(xl, resnet_ops, pw_l, ACCENT2_TINT,
                                     "ResNet-50 block",
                                     extra_after_plus="ReLU")
    d.shape_note(xl - 14, 62, "(256, 56×56)", ha="right")

    # ----- right: ConvNeXt block (accent 1, blue) --------------------------- #
    xr, pw_r = 490, 122
    nxt_ops = [("7×7 ", "Depthwise", " Conv, 96"), "LayerNorm",
               "1×1 Conv, 384", "GELU", "1×1 Conv, 96", "Layer scale γ"]
    ys_r, y_plus_r, panel_r = column(xr, nxt_ops, pw_r, ACCENT_TINT,
                                     "ConvNeXt block", novelty_idx=0)
    d.shape_note(xr + 14, 62, "(96, 56×56)", ha="left")

    # ----- callouts: horizontal leaders at the target pill's own height ----- #
    pl_edge = xl - pw_l / 2 - 4      # left pill edge
    d.callout(8, ys_l[0] + LH / 2, [
        [("Reduce ", INK, 1), ("256→64", ACCENT2, 1), (",", INK, 1)],
        [("process, expand", INK, 1)],
    ], target=(pl_edge, ys_l[0]), leader_from=(118, ys_l[0]), ha="left")
    d.callout(8, ys_l[4] + LH / 2, [
        [("3", ACCENT2, 1), (" norms,", INK, 1)],
        [("3", ACCENT2, 1), (" activations", INK, 1)],
    ], target=(pl_edge, ys_l[4]), leader_from=(88, ys_l[4]), ha="left")

    pr_edge = xr + pw_r / 2 + 4      # right pill edge
    d.callout(W - 8, ys_r[2] + LH / 2, [
        [("Inverted bottleneck:", INK, 1)],
        [("expand ", INK, 1), ("96→384", ACCENT, 1)],
    ], target=(pr_edge, ys_r[2]), leader_from=(W - 140, ys_r[2]), ha="right")
    d.callout(W - 8, ys_r[1] + LH / 2, [
        [("1", ACCENT, 1), (" norm,", INK, 1)],
        [("1", ACCENT, 1), (" activation", INK, 1)],
    ], target=(pr_edge, ys_r[1]), leader_from=(W - 96, ys_r[1]), ha="right")

    save(d.fig, "arch-resnet-vs-convnext-block")


# --------------------------------------------------------------------------- #
# Pilot 2: Inception block (multi-branch layout).                             #
# --------------------------------------------------------------------------- #

def fig_inception_block():
    W, H = 660, 400
    d = Diagram(W, H)

    y_in = 90            # spine entry above the anchor (short input line)
    y_bus_lo = 122       # lower split bus
    y_row1 = 154         # first pill row (one PILL_GAP above the bus + arrow)
    y_row2 = y_row1 + PILL_STEP + 9        # second pill row
    y_bus_hi = y_row2 + PILL_H / 2 + 19    # upper merge bus
    y_cat = y_bus_hi + 34                  # concat novelty box

    xc = W / 2
    bx = [130, 260, 400, 536]   # branch column x positions
    pw = [86, 110, 110, 118]    # per-branch pill widths

    # block panel (the repeated unit); the concat box is centered, so the
    # 'Inception block' label sits in the empty top-left corner at its height
    # rather than in a reserved band above it — keeps the panel compact.
    panel = (54, y_bus_lo - 24, W - 28, y_cat + PILL_H / 2 + 13)
    y_out = panel[3] + 30        # short output arrow beyond the panel top
    d.container(*panel, fill=ACCENT_TINT)
    d.stage_label(panel[0] + 14, y_cat + PILL_H / 2 + 2, "Inception block")
    d.repeat(panel[0] - 13, panel[1], panel[3], "9")

    # input spine joins the lower bus
    d.line([(xc, y_in), (xc, y_bus_lo)])
    d.line([(bx[0], y_bus_lo), (bx[-1], y_bus_lo)])

    # branch 1: single 1×1 conv, centered between the two rows
    y_mid = 0.5 * (y_row1 + y_row2)
    d.arrow(bx[0], y_bus_lo, y_mid - PILL_H / 2)
    d.pill(bx[0], y_mid, "1×1 Conv", w=pw[0])
    d.line([(bx[0], y_mid + PILL_H / 2), (bx[0], y_bus_hi)])

    # branches 2 and 3: 1×1 then wide conv
    for x, w, wide in ((bx[1], pw[1], "3×3 Conv, pad 1"),
                       (bx[2], pw[2], "5×5 Conv, pad 2")):
        d.arrow(x, y_bus_lo, y_row1 - PILL_H / 2)
        d.pill(x, y_row1, "1×1 Conv", w=w)
        d.arrow(x, y_row1 + PILL_H / 2, y_row2 - PILL_H / 2)
        d.pill(x, y_row2, wide, w=w)
        d.line([(x, y_row2 + PILL_H / 2), (x, y_bus_hi)])

    # branch 4: pool then 1×1
    d.arrow(bx[3], y_bus_lo, y_row1 - PILL_H / 2)
    d.pill(bx[3], y_row1, "3×3 MaxPool, pad 1", w=pw[3])
    d.arrow(bx[3], y_row1 + PILL_H / 2, y_row2 - PILL_H / 2)
    d.pill(bx[3], y_row2, "1×1 Conv", w=pw[3])
    d.line([(bx[3], y_row2 + PILL_H / 2), (bx[3], y_bus_hi)])

    # upper merge bus into the concat novelty box
    d.line([(bx[0], y_bus_hi), (bx[-1], y_bus_hi)])
    d.arrow(xc, y_bus_hi, y_cat - PILL_H / 2)
    d.novelty(xc, y_cat, "", "Concatenate", " channels")
    d.arrow(xc, y_cat + PILL_H / 2, y_out)

    d.anchor(xc, y_in - 16, "input x")
    # size labels: outside the panel, offset clear of the spine
    d.shape_note(xc + 14, (panel[1] + y_in - 16) / 2 + 6, "(192, 28×28)")
    d.shape_note(xc + 14, y_out - 6, "(256, 28×28)")

    # callouts at the bottom margin; straight VERTICAL leaders whose feet
    # land on the callout text (leader x sits inside the text's span)
    d.callout(132, 26 + LH, [
        [("1×1 convolutions shrink channels", INK, 1)],
        [("before the costly ", INK, 1), ("3×3", ACCENT, 1),
         (" and ", INK, 1), ("5×5", ACCENT, 1)],
    ], target=(bx[1], y_bus_lo - 6),
        leader_from=(bx[1], 26 + LH + 9), ha="left")
    d.callout(bx[3] - 90, 26 + LH, [
        [("Four branches, four scales,", INK, 1)],
        [("one shared input", INK, 1)],
    ], target=(bx[3] - 28, y_bus_lo - 6),
        leader_from=(bx[3] - 28, 26 + LH + 9), ha="left")

    save(d.fig, "arch-inception-block")


def _blocked_column(d, x, groups, pw, y0, outer_label, anchor_text="input x",
                    novelty_at=None, stage_labels=False):
    """A full-model column: one spine of pills, some grouped into
    accent-tinted inner block containers, all nested in a gray outer network
    container.  ``groups`` = [{'ops': [...], 'tint': bool, 'repeat': str,
    'label': str}].  Arrows run pill-to-pill, crossing block edges (the
    gallery convention).  ``novelty_at`` = (group_idx, op_idx).
    Returns (y_out, outer_panel, layouts)."""
    layouts = []
    y = y0
    for g in groups:
        vpad = BLOCK_VPAD if g.get("tint") else 0
        extra_top = BLOCK_LABEL_BAND if (g.get("tint") and g.get("label")) else 0
        first = y + vpad + PILL_H / 2
        ys = [first + i * PILL_STEP for i in range(len(g["ops"]))]
        top = ys[-1] + PILL_H / 2 + vpad + extra_top
        layouts.append((g, ys, y, top))
        y = top + BLOCK_GAP               # inter-group arrow gap
    inner_top = layouts[-1][3]
    outer = (x - pw / 2 - BLOCK_PAD - 16, y0 - 20,
             x + pw / 2 + BLOCK_PAD + 16, inner_top + LABEL_BAND)
    d.container(*outer, fill=CONTAINER_FILL, zorder=1)
    d.stage_label(outer[0] + 14, outer[3] - 9, outer_label)

    for gi, (g, ys, bot, top) in enumerate(layouts):
        if g.get("tint"):
            d.container(x - pw / 2 - BLOCK_PAD, bot,
                        x + pw / 2 + BLOCK_PAD, top,
                        fill=ACCENT_TINT, round_=12, zorder=2)
            if g.get("repeat"):
                d.repeat(x - pw / 2 - BLOCK_PAD - 12, bot, top, g["repeat"])
            if g.get("label"):
                d.stage_label(x - pw / 2 - BLOCK_PAD + 10, top - 5, g["label"])
        for i, (op, yy) in enumerate(zip(g["ops"], ys)):
            if novelty_at == (gi, i):
                pre, kw, post = op
                d.novelty(x, yy, pre, kw, post, w=pw + 10)
            else:
                d.pill(x, yy, op, w=pw)
        # arrows: within the group, then into the next group's first pill
        for i in range(len(ys) - 1):
            d.arrow(x, ys[i] + PILL_H / 2, ys[i + 1] - PILL_H / 2)
        if gi < len(layouts) - 1:
            d.arrow(x, ys[-1] + PILL_H / 2,
                    layouts[gi + 1][1][0] - PILL_H / 2)

    y_out = outer[3] + 30
    d.arrow(x, layouts[-1][1][-1] + PILL_H / 2, y_out)
    y_anchor = y0 - 20 - 36
    d.arrow(x, y_anchor + 14, layouts[0][1][0] - PILL_H / 2)
    d.anchor(x, y_anchor, anchor_text)
    return y_out, outer, layouts


# --------------------------------------------------------------------------- #
# 7.1: LeNet (ch. 6 version) vs AlexNet (two-column comparison, activations  #
# folded into the layer pills; the story is scale + ReLU + dropout).          #
# --------------------------------------------------------------------------- #

def _plain_column(d, x, ops, pw, tint, label, y0, novelty_idx=None,
                  anchor_text="input x", shape_in=None, shape_notes=()):
    """A straight spine of ops (no skip): panel, pills, arrows, anchor.
    ``shape_notes`` = [(index, text)] drawn beside the arrow ABOVE op index."""
    ys = [y0 + i * PILL_STEP for i in range(len(ops))]
    top_edge = ys[-1] + PILL_H / 2 + LABEL_BAND
    panel = (x - pw / 2 - 22, y0 - PILL_H / 2 - 20, x + pw / 2 + 22, top_edge)
    y_out = top_edge + 30
    d.container(*panel, fill=tint)
    d.stage_label(panel[0] + 14, panel[3] - 9, label)

    y_anchor = y0 - PILL_H / 2 - 20 - 36
    d.arrow(x, y_anchor + 14, ys[0] - PILL_H / 2)
    for i, (op, y) in enumerate(zip(ops, ys)):
        if i == novelty_idx:
            pre, kw, post = op
            d.novelty(x, y, pre, kw, post, w=pw + 14)
        else:
            d.pill(x, y, op, w=pw)
        if i < len(ops) - 1:
            d.arrow(x, y + PILL_H / 2, ys[i + 1] - PILL_H / 2)
    d.arrow(x, ys[-1] + PILL_H / 2, y_out)
    d.anchor(x, y_anchor, anchor_text)
    if shape_in:
        d.shape_note(x + 14, y_anchor + 26, shape_in)
    for idx, text in shape_notes:
        d.shape_note(x + pw / 2 + 8,
                     0.5 * (ys[idx - 1] + ys[idx]) if idx else ys[0],
                     text)
    return ys, panel, y_out


def fig_alexnet():
    # Both columns show the networks as the book builds them for
    # Fashion-MNIST (single-channel input, ten classes): the LeNet of the
    # previous chapter and the AlexNet of sec_alexnet, so every layer and
    # count can be checked against the code.
    W, H = 790, 760
    d = Diagram(W, H)
    y0 = 116

    lenet_ops = ["5×5 Conv, 6, pad 2, sigmoid", "2×2 AvgPool, s2",
                 "5×5 Conv, 16, sigmoid", "2×2 AvgPool, s2",
                 "Flatten",
                 "Dense 120, sigmoid", "Dense 84, sigmoid", "Dense 10"]
    xl, pw_l = 200, 168
    ys_l, panel_l, _ = _plain_column(
        d, xl, lenet_ops, pw_l, ACCENT2_TINT, "LeNet", y0,
        anchor_text="28×28 image")

    alex_ops = [("", "11×11", " Conv, 96, s4, ReLU"),
                "3×3 MaxPool, s2",
                "5×5 Conv, 256, pad 2, ReLU", "3×3 MaxPool, s2",
                "3×3 Conv, 384, pad 1, ReLU", "3×3 Conv, 384, pad 1, ReLU",
                "3×3 Conv, 256, pad 1, ReLU", "3×3 MaxPool, s2",
                "Flatten",
                "Dense 4096, ReLU", "Dropout 0.5",
                "Dense 4096, ReLU", "Dropout 0.5",
                "Dense 10"]
    xr, pw_r = 490, 196
    ys_r, panel_r, _ = _plain_column(
        d, xr, alex_ops, pw_r, ACCENT_TINT, "AlexNet", y0,
        novelty_idx=0, anchor_text="224×224 image")

    # callouts (horizontal leaders at target height)
    pr_edge = xr + pw_r / 2 + 4
    d.callout(W - 8, ys_r[9] + LH / 2, [
        [("A ", INK, 1), ("4096", ACCENT, 1), ("-wide head,", INK, 1)],
        [("kept in check by dropout", INK, 1)],
    ], target=(pr_edge, ys_r[9]), leader_from=(W - 154, ys_r[9]), ha="right")
    d.callout(W - 8, ys_r[4] + LH / 2, [
        [("ReLU everywhere:", INK, 1)],
        [("no saturating sigmoids", INK, 1)],
    ], target=(pr_edge, ys_r[4]), leader_from=(W - 138, ys_r[4]), ha="right")
    # whole-network capacity note in the empty space above the short column
    y_note = panel_l[3] + 96
    d.callout(60, y_note, [
        [("LeNet has ", INK, 1), ("62k", ACCENT2, 1),
         (" parameters;", INK, 1)],
        [("AlexNet has ", INK, 1), ("47M", ACCENT, 1),
         (" parameters", INK, 1)],
    ], ha="left")

    save(d.fig, "arch-alexnet")


# --------------------------------------------------------------------------- #
# 7.2: AlexNet vs VGG-11 — blocks as the unit of design.                      #
# --------------------------------------------------------------------------- #

def fig_vgg():
    # Both columns show the networks for single-channel 224x224 images and
    # ten classes, as sec_vgg builds them; the shared dense head is drawn as
    # one pill so that the comparison stays on the convolutional body.
    W, H = 880, 900
    d = Diagram(W, H)
    y0 = 116
    head = ["Flatten", "Dense 4096, 4096, 10"]

    # left: AlexNet as one unstructured run of ops (no inner blocks)
    alex_ops = ["11×11 Conv, 96, s4, ReLU", "3×3 MaxPool, s2",
                "5×5 Conv, 256, pad 2, ReLU", "3×3 MaxPool, s2",
                "3×3 Conv, 384, pad 1, ReLU", "3×3 Conv, 384, pad 1, ReLU",
                "3×3 Conv, 256, pad 1, ReLU", "3×3 MaxPool, s2"] + head
    xl, pw_l = 190, 196
    _, panel_l, _ = _blocked_column(
        d, xl, [{"ops": alex_ops}], pw_l, y0, "AlexNet",
        anchor_text="224×224 image")

    # right: VGG-11 as five repeated blocks + head
    def vgg_block(c, n):
        return {"ops": [f"3×3 Conv, {c}, pad 1, ReLU"] * n
                + ["2×2 MaxPool, s2"], "tint": True, "repeat": None}

    groups = [vgg_block(64, 1), vgg_block(128, 1), vgg_block(256, 2),
              vgg_block(512, 2), vgg_block(512, 2), {"ops": head}]
    groups[0]["ops"][0] = ("", "3×3", " Conv, 64, pad 1, ReLU")   # novelty
    xr, pw_r = 500, 186
    _, panel_r, lay = _blocked_column(
        d, xr, groups, pw_r, y0, "VGG-11",
        anchor_text="224×224 image", novelty_at=(0, 0))

    # callouts (horizontal, at target height)
    blk3_mid = 0.5 * (lay[2][2] + lay[2][3])
    d.callout(W - 8, blk3_mid + LH / 2, [
        [("The unit of design", INK, 1)],
        [("is the ", INK, 1), ("block", ACCENT, 1),
         (", not the layer", INK, 1)],
    ], target=(xr + pw_r / 2 + BLOCK_PAD + 4, blk3_mid),
        leader_from=(W - 178, blk3_mid), ha="right")
    y_conv2 = lay[3][1][0]
    d.callout(W - 8, y_conv2 + LH / 2, [
        [("Two stacked ", INK, 1), ("3×3", ACCENT, 1), (" convs", INK, 1)],
        [("see 5×5, with fewer weights", INK, 1)],
    ], target=(xr + pw_r / 2 + BLOCK_PAD + 4, y_conv2),
        leader_from=(W - 200, y_conv2), ha="right")
    # depth note above the shorter AlexNet column
    y_note = panel_l[3] + 96
    d.callout(40, y_note, [
        [("VGG-11 has ", INK, 1), ("8", ACCENT, 1),
         (" convolutions,", INK, 1)],
        [("AlexNet ", INK, 1), ("5", ACCENT2, 1)],
    ], ha="left")

    save(d.fig, "arch-vgg")


# --------------------------------------------------------------------------- #
# 7.2: NiN — the head question: dense layers vs 1×1 convs + global pooling.   #
# --------------------------------------------------------------------------- #

def fig_nin():
    W, H = 880, 500
    d = Diagram(W, H)
    y0 = 116

    # left: the VGG-style dense head
    xl, pw_l = 260, 140
    head_ops = ["Flatten", "Dense 4096, ReLU", "Dropout 0.5",
                "Dense 4096, ReLU", "Dropout 0.5", "Dense 10"]
    _, panel_l, lay_l = _blocked_column(
        d, xl, [{"ops": head_ops}], pw_l, y0, "VGG head",
        anchor_text="(512, 7×7) features")

    # right: the end of NiN: dropout, last NiN block, global average pooling
    xr, pw_r = 575, 150
    groups = [{"ops": ["Dropout 0.5"]},
              {"ops": ["3×3 Conv, 10, ReLU", "1×1 Conv, 10, ReLU",
                       "1×1 Conv, 10"],
               "tint": True, "label": "NiN block"},
              {"ops": [("", "Global AvgPool", "")]}]
    _, panel_r, lay_r = _blocked_column(
        d, xr, groups, pw_r, y0, "End of NiN",
        anchor_text="(384, 5×5) features", novelty_at=(2, 0))

    # shape notes in the empty middle gap, fully outside the NiN grey box
    d.shape_note(panel_r[0] - 8,
                 0.5 * (lay_r[1][3] + lay_r[2][1][0] - PILL_H / 2) + 1,
                 "(10, 5×5)", ha="right")
    d.shape_note(xr + 14, panel_r[3] + 14, "10 scores")

    # callouts
    y_dense = lay_l[0][1][1]
    d.callout(32, y_dense + LH, [
        [("Over ", INK, 1), ("90%", ACCENT2, 1)],
        [("of VGG-11's weights", INK, 1)],
        [("live in this head", INK, 1)],
    ], target=(xl - pw_l / 2 - 4, y_dense), leader_from=(148, y_dense),
        ha="left")
    y_gap = lay_r[2][1][0]
    d.callout(W - 48, y_gap + LH / 2, [
        [("No parameters at all:", INK, 1)],
        [("one channel per class", INK, 1)],
    ], target=(xr + (pw_r + 10) / 2 + 4, y_gap), leader_from=(W - 172, y_gap),
        ha="right")

    save(d.fig, "arch-nin")


# --------------------------------------------------------------------------- #
# 8.4: ResNet residual block — identity vs projection shortcut.               #
# --------------------------------------------------------------------------- #

def _residual_column(d, x, ops, pw, tint, label, y0, shortcut_pill=None,
                     top="ReLU"):
    """Residual block column: ops on the spine, skip from below the first
    pill into a ⊕ one step above the last pill; ``shortcut_pill`` (a str, or
    a list of str stacked top to bottom) puts ops on the skip path
    (projection variant).  Returns (ys, y_plus)."""
    ys = [y0 + i * PILL_STEP for i in range(len(ops))]
    y_plus = ys[-1] + PILL_STEP
    y_relu = y_plus + PILL_STEP
    x_out = x + (126 if shortcut_pill else 88)
    panel = (x - pw / 2 - 22, y0 - PILL_H / 2 - 23,
             x_out + (56 if shortcut_pill else 22),
             y_relu + PILL_H / 2 + LABEL_BAND)
    d.container(*panel, fill=tint)
    d.stage_label(panel[0] + 14, panel[3] - 9, label)

    y_anchor = panel[1] - 36
    d.arrow(x, y_anchor + 14, ys[0] - PILL_H / 2)
    for i, (op, y) in enumerate(zip(ops, ys)):
        d.pill(x, y, op, w=pw)
        if i < len(ops) - 1:
            d.arrow(x, y + PILL_H / 2, ys[i + 1] - PILL_H / 2)
    d.arrow(x, ys[-1] + PILL_H / 2, y_plus - PLUS_R)

    y_branch = y0 - PILL_H / 2 - 11
    if shortcut_pill:
        pills = ([shortcut_pill] if isinstance(shortcut_pill, str)
                 else list(shortcut_pill))
        y_mid = 0.5 * (y_branch + y_plus)
        ys_s = [y_mid + (i - (len(pills) - 1) / 2) * PILL_STEP
                for i in range(len(pills))]
        d.line([(x, y_branch), (x_out, y_branch),
                (x_out, ys_s[0] - PILL_H / 2)])
        for i, (op, y) in enumerate(zip(pills, ys_s)):
            d.pill(x_out, y, op, w=None)
            if i < len(pills) - 1:
                d.arrow(x_out, y + PILL_H / 2, ys_s[i + 1] - PILL_H / 2)
        d.line([(x_out, ys_s[-1] + PILL_H / 2), (x_out, y_plus)])
        d.op_circle(x, y_plus)
        d.ax.annotate(
            "", xy=(x + PLUS_R, y_plus), xytext=(x_out, y_plus),
            arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.55",
                            color=INK, lw=1.2, shrinkA=0, shrinkB=0,
                            mutation_scale=11), zorder=3)
    else:
        d.skip(x, y_branch, y_plus, x_out)

    d.arrow(x, y_plus + PLUS_R, y_relu - PILL_H / 2)
    d.pill(x, y_relu, top, w=pw)
    d.arrow(x, y_relu + PILL_H / 2, panel[3] + 30)
    d.anchor(x, y_anchor, "input x")
    return ys, y_plus


def fig_resnet_block():
    W, H = 850, 480
    d = Diagram(W, H)
    y0 = 116
    ops = ["3×3 Conv, BatchNorm", "ReLU", "3×3 Conv, BatchNorm"]

    xl, pw = 210, 138
    ys_l, y_plus_l = _residual_column(
        d, xl, ops, pw, ACCENT_TINT, "Identity\nshortcut", y0)
    xr = 470
    ys_r, y_plus_r = _residual_column(
        d, xr, ops, pw, ACCENT_TINT, "Projection\nshortcut", y0,
        shortcut_pill=["1×1 Conv", "BatchNorm"])

    d.callout(8, y_plus_l + LH, [
        [("The sum is ", INK, 1), ("x + g(x)", ACCENT, 1), (",", INK, 1)],
        [("so g learns", INK, 1)],
        [("a correction", INK, 1)],
    ], target=(xl - PLUS_R - 4, y_plus_l), leader_from=(116, y_plus_l),
        ha="left")
    # The callout points at the 1x1 conv, the lower of the two shortcut pills
    y_conv = 0.5 * ((y0 - PILL_H / 2 - 11) + y_plus_r) - PILL_STEP / 2
    d.callout(W - 2, y_conv + LH, [
        [("A ", INK, 1), ("1×1", ACCENT, 1), (" conv matches", INK, 1)],
        [("the shape when the", INK, 1)],
        [("channels or stride change", INK, 1)],
    ], target=(xr + 126 + 40, y_conv), leader_from=(W - 160, y_conv),
        ha="right")

    save(d.fig, "arch-resnet-block")


# --------------------------------------------------------------------------- #
# 8.4: Regular vs residual block — the residual block learns g(x) = f(x) - x. #
# --------------------------------------------------------------------------- #

def fig_residual_concept():
    W, H = 600, 470
    d = Diagram(W, H)
    y0 = 116
    ops = ["Weight layer", "Activation", "Weight layer"]
    pw = 128
    xl, xr = 110, 370
    ys_r, y_plus = _residual_column(
        d, xr, ops, pw, ACCENT_TINT, "Residual\nblock", y0, top="Activation")
    # Regular block: same rhythm and height, no shortcut, no addition
    y_top = y_plus + PILL_STEP
    panel = (xl - pw / 2 - 22, y0 - PILL_H / 2 - 23, xl + pw / 2 + 22,
             y_top + PILL_H / 2 + LABEL_BAND)
    d.container(*panel, fill=CONTAINER_FILL)
    d.stage_label(panel[0] + 14, panel[3] - 9, "Regular\nblock")
    y_anchor = panel[1] - 36
    d.arrow(xl, y_anchor + 14, ys_r[0] - PILL_H / 2)
    for i, (op, y) in enumerate(zip(ops, ys_r)):
        d.pill(xl, y, op, w=pw)
        if i < len(ops) - 1:
            d.arrow(xl, y + PILL_H / 2, ys_r[i + 1] - PILL_H / 2)
    d.arrow(xl, ys_r[-1] + PILL_H / 2, y_top - PILL_H / 2)
    d.pill(xl, y_top, "Activation", w=pw)
    d.arrow(xl, y_top + PILL_H / 2, panel[3] + 30)
    d.anchor(xl, y_anchor, "input x")

    def math(x, y, text, color=INK):
        d.ax.text(x, y, text, fontsize=12.5, color=color, family=SANS_FONT,
                  style="italic", ha="left", va="center", zorder=8)

    # regular: the stack itself must produce f(x)
    math(xl + 12, y_plus, "f(x)")
    # residual: the stack produces g(x); the sum is f(x) = g(x) + x
    math(xr + 12, 0.5 * (ys_r[-1] + PILL_H / 2 + y_plus - PLUS_R),
         "g(x)", color=ACCENT)
    math(xr + 14, 0.5 * (y_plus + PLUS_R + y_top - PILL_H / 2),
         "f(x) = g(x) + x")
    math(xr + 96, 0.5 * (ys_r[0] + y_plus), "x")

    save(d.fig, "arch-residual-concept")


# --------------------------------------------------------------------------- #
# 8.4: ResNeXt block — grouped bottleneck.                                    #
# --------------------------------------------------------------------------- #

def _note(d, x, y, text, ha="left"):
    """Black italic shape note (annotation text is black, house rule)."""
    from arch_diagrams import NOTE_FS, SANS
    d.ax.text(x, y, text, fontsize=NOTE_FS, color=INK, family=SANS,
              style="italic", ha=ha, va="center", zorder=8)


def fig_resnext_block():
    W, H = 620, 520
    d = Diagram(W, H)
    y0 = 116
    x, pw = 300, 168

    ops = ["1×1 Conv, 128", "BatchNorm, ReLU",
           ("3×3 Conv, 128, ", "32 groups", ""),
           "BatchNorm, ReLU", "1×1 Conv, 256", "BatchNorm"]
    ys = [y0 + i * PILL_STEP for i in range(len(ops))]
    y_plus = ys[-1] + PILL_STEP
    y_relu = y_plus + PILL_STEP
    panel = (x - pw / 2 - 22, y0 - PILL_H / 2 - 23, x + 118,
             y_relu + PILL_H / 2 + LABEL_BAND)
    d.container(*panel, fill=ACCENT_TINT)
    d.stage_label(panel[0] + 14, panel[3] - 9, "ResNeXt block")

    y_anchor = panel[1] - 36
    d.arrow(x, y_anchor + 14, ys[0] - PILL_H / 2)
    for i, (op, y) in enumerate(zip(ops, ys)):
        if i == 2:
            pre, kw, post = op
            d.novelty(x, y, pre, kw, post, w=pw + 22)
        else:
            d.pill(x, y, op, w=pw)
        if i < len(ops) - 1:
            d.arrow(x, y + PILL_H / 2, ys[i + 1] - PILL_H / 2)
    d.arrow(x, ys[-1] + PILL_H / 2, y_plus - PLUS_R)
    d.skip(x, y0 - PILL_H / 2 - 11, y_plus, x + 104)
    d.arrow(x, y_plus + PLUS_R, y_relu - PILL_H / 2)
    d.pill(x, y_relu, "ReLU", w=pw)
    d.arrow(x, y_relu + PILL_H / 2, panel[3] + 30)
    d.anchor(x, y_anchor, "input x")
    _note(d, x - 90, y_anchor + 26, "(256, H×W)", ha="right")

    d.callout(8, ys[2] + LH / 2, [
        [("32", ACCENT, 1), (" groups of 4 channels:", INK, 1)],
        [("1/32", ACCENT, 1), (" of the weights", INK, 1)],
        [("of a dense 3×3 conv", INK, 1)],
    ], target=(x - pw / 2 - 15, ys[2]), leader_from=(150, ys[2]), ha="left")

    save(d.fig, "arch-resnext-block")


# --------------------------------------------------------------------------- #
# 8.4: DenseNet dense block — concatenation instead of addition.              #
# --------------------------------------------------------------------------- #

def fig_densenet_block():
    W, H = 700, 420
    d = Diagram(W, H)
    x = 400
    y0 = 116
    pw = 196

    ys = [y0 + i * PILL_STEP for i in range(4)]     # conv1, cat1, conv2, cat2
    x_l = x - pw / 2 - 58                            # skip rail on the left

    panel = (x_l - 22, y0 - PILL_H / 2 - 23, x + pw / 2 + 22,
             ys[-1] + PILL_H / 2 + LABEL_BAND)
    d.container(*panel, fill=ACCENT_TINT)
    d.stage_label(panel[0] + 14, panel[3] - 9, "Dense block (growth rate 32)")

    y_anchor = panel[1] - 36
    d.arrow(x, y_anchor + 14, ys[0] - PILL_H / 2)
    d.pill(x, ys[0], "BatchNorm, ReLU, 3×3 Conv, 32", w=pw)
    d.arrow(x, ys[0] + PILL_H / 2, ys[1] - PILL_H / 2)
    d.novelty(x, ys[1], "", "Concat", " channels", w=136)
    d.arrow(x, ys[1] + PILL_H / 2, ys[2] - PILL_H / 2)
    d.pill(x, ys[2], "BatchNorm, ReLU, 3×3 Conv, 32", w=pw)
    d.arrow(x, ys[2] + PILL_H / 2, ys[3] - PILL_H / 2)
    d.pill(x, ys[3], "Concat channels", w=136)
    d.arrow(x, ys[3] + PILL_H / 2, panel[3] + 30)
    d.anchor(x, y_anchor, "input x")

    def cat_skip(y_from, y_to, w_cat):
        d.line([(x, y_from), (x_l, y_from), (x_l, y_to)])
        d.ax.annotate(
            "", xy=(x - w_cat / 2, y_to), xytext=(x_l, y_to),
            arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.55",
                            color=INK, lw=1.2, shrinkA=0, shrinkB=0,
                            mutation_scale=11), zorder=3)

    cat_skip(y0 - PILL_H / 2 - 11, ys[1], 136)
    cat_skip(0.5 * (ys[1] + ys[2]) + 1, ys[3], 136)

    # channel growth beside the spine
    _note(d, x + 14, y_anchor + 26, "(64, H×W)")
    _note(d, x + pw / 2 + 30, 0.5 * (ys[1] + ys[2]), "(96, H×W)")
    _note(d, x + 14, panel[3] + 18, "(128, H×W)")

    d.callout(8, ys[1] + LH, [
        [("Inputs are appended,", INK, 1)],
        [("never overwritten: every", INK, 1)],
        [("later layer sees ", INK, 1), ("all", ACCENT, 1),
         (" features", INK, 1)],
    ], target=(x_l - 4, ys[1]), leader_from=(162, ys[1]), ha="left")

    save(d.fig, "arch-densenet-block")


# --------------------------------------------------------------------------- #
# 8.4: ResNet-18 full model (stage containers, shapes as in the book's        #
# 96×96 Fashion-MNIST training run).                                          #
# --------------------------------------------------------------------------- #

def fig_resnet18():
    W, H = 640, 800
    d = Diagram(W, H)
    x, pw = 250, 150
    y0 = 116

    groups = [
        {"ops": ["7×7 Conv, 64, s2", "BatchNorm, ReLU", "3×3 MaxPool, s2"]},
        {"ops": ["Residual block, 64"], "tint": True, "repeat": "2",
         "label": "Stage 1"},
        {"ops": ["Residual block, 128"], "tint": True, "repeat": "2",
         "label": "Stage 2"},
        {"ops": ["Residual block, 256"], "tint": True, "repeat": "2",
         "label": "Stage 3"},
        {"ops": ["Residual block, 512"], "tint": True, "repeat": "2",
         "label": "Stage 4"},
        {"ops": ["Global AvgPool", "Dense 10"]},
    ]
    y_out, outer, lay = _blocked_column(
        d, x, groups, pw, y0, "ResNet-18", anchor_text="96×96 image")

    # shapes at the stage boundaries (right of the inter-group arrows)
    xs = outer[2] + 8               # just outside the grey outer container
    def between(a, b):
        return 0.5 * (lay[a][3] + lay[b][1][0] - PILL_H / 2)
    _note(d, xs, between(0, 1) - 4, "(64, 24×24)")
    _note(d, xs, between(1, 2) - 4, "(64, 24×24)")
    _note(d, xs, between(2, 3) - 4, "(128, 12×12)")
    _note(d, xs, between(3, 4) - 4, "(256, 6×6)")
    _note(d, xs, between(4, 5) - 4, "(512, 3×3)")

    y_s2 = 0.5 * (lay[2][2] + lay[2][3])
    d.callout(W - 8, y_s2 + LH, [
        [("Stages 2–4 open with the", INK, 1)],
        [("projection", ACCENT, 1), (" block, stride 2:", INK, 1)],
        [("resolution ÷2, channels ×2", INK, 1)],
    ], target=(x + pw / 2 + BLOCK_PAD + 4, y_s2),
        leader_from=(W - 178, y_s2), ha="right")

    save(d.fig, "arch-resnet18")


# --------------------------------------------------------------------------- #
# 8.6: ConvNeXt-T full model.                                                 #
# --------------------------------------------------------------------------- #

def fig_convnext():
    W, H = 700, 940
    d = Diagram(W, H)
    x, pw = 260, 168
    y0 = 116

    groups = [
        {"ops": [("", "Patchify", " 4×4 Conv, 96, s4"), "LayerNorm"]},
        {"ops": ["ConvNeXt block, 96"], "tint": True, "repeat": "3",
         "label": "Stage 1"},
        {"ops": ["LN, 2×2 Conv, 192, s2"]},
        {"ops": ["ConvNeXt block, 192"], "tint": True, "repeat": "3",
         "label": "Stage 2"},
        {"ops": ["LN, 2×2 Conv, 384, s2"]},
        {"ops": ["ConvNeXt block, 384"], "tint": True, "repeat": "9",
         "label": "Stage 3"},
        {"ops": ["LN, 2×2 Conv, 768, s2"]},
        {"ops": ["ConvNeXt block, 768"], "tint": True, "repeat": "3",
         "label": "Stage 4"},
        {"ops": ["Global AvgPool", "LayerNorm", "Dense 1000"]},
    ]
    y_out, outer, lay = _blocked_column(
        d, x, groups, pw, y0, "ConvNeXt-T",
        anchor_text="224×224×3 image", novelty_at=(0, 0))

    xs = outer[2] + 8               # just outside the grey outer container
    def between(a, b):
        return 0.5 * (lay[a][3] + lay[b][1][0] - PILL_H / 2)
    d.shape_note(xs, between(0, 1) - 4, "(96, 56×56)")
    d.shape_note(xs, between(2, 3) - 4, "(192, 28×28)")
    d.shape_note(xs, between(4, 5) - 4, "(384, 14×14)")
    d.shape_note(xs, between(6, 7) - 4, "(768, 7×7)")

    y_s3 = 0.5 * (lay[5][2] + lay[5][3])
    d.callout(W - 8, y_s3 + LH / 2, [
        [("Stage ratio ", INK, 1), ("3:3:9:3", ACCENT, 1), (":", INK, 1)],
        [("deepest where maps are small", INK, 1)],
    ], target=(x + pw / 2 + BLOCK_PAD + 4, y_s3),
        leader_from=(W - 200, y_s3), ha="right")
    y_ds = lay[2][1][0]
    d.callout(W - 8, y_ds + LH / 2, [
        [("No pooling anywhere:", INK, 1)],
        [("a strided conv downsamples", INK, 1)],
    ], target=(x + pw / 2 + 4, y_ds), leader_from=(W - 186, y_ds),
        ha="right")

    save(d.fig, "arch-convnext")


# --------------------------------------------------------------------------- #
# 8.7: depthwise-separable block.                                             #
# --------------------------------------------------------------------------- #

def fig_dws_block():
    import matplotlib.pyplot as plt
    from arch_diagrams import SANS
    # Subscripts as mathtext set in the book font (it has no subscript i/o).
    with plt.rc_context({"mathtext.fontset": "custom", "mathtext.rm": SANS,
                         "mathtext.it": f"{SANS}:italic",
                         "mathtext.bf": f"{SANS}:bold"}):
        _fig_dws_block()


def _fig_dws_block():
    W, H = 600, 440
    d = Diagram(W, H)
    x, pw = 250, 172
    y0 = 116

    ops = [("3×3 ", "Depthwise", r" Conv, $c_\mathrm{i}$"), "BatchNorm, ReLU",
           r"1×1 Conv, $c_\mathrm{o}$", "BatchNorm, ReLU"]
    ys = [y0 + i * PILL_STEP for i in range(len(ops))]
    panel = (x - pw / 2 - 22, y0 - PILL_H / 2 - 23, x + pw / 2 + 22,
             ys[-1] + PILL_H / 2 + LABEL_BAND)
    d.container(*panel, fill=ACCENT_TINT)
    d.stage_label(panel[0] + 14, panel[3] - 9, "Separable block")

    y_anchor = panel[1] - 36
    d.arrow(x, y_anchor + 14, ys[0] - PILL_H / 2)
    for i, (op, y) in enumerate(zip(ops, ys)):
        if i == 0:
            pre, kw, post = op
            d.novelty(x, y, pre, kw, post, w=pw + 14)
        else:
            d.pill(x, y, op, w=pw)
        if i < len(ops) - 1:
            d.arrow(x, y + PILL_H / 2, ys[i + 1] - PILL_H / 2)
    d.arrow(x, ys[-1] + PILL_H / 2, panel[3] + 30)
    d.anchor(x, y_anchor, "input x")
    d.shape_note(x - 96, y_anchor + 26, r"($c_\mathrm{i}$, h×w)", ha="right")
    d.shape_note(x + 14, panel[3] + 18, r"($c_\mathrm{o}$, h×w)")

    d.callout(W - 8, ys[0] + LH / 2, [
        [("Spatial filtering,", INK, 1)],
        [("one channel at a time", INK, 1)],
    ], target=(x + (pw + 14) / 2 + 4, ys[0]), leader_from=(W - 156, ys[0]),
        ha="right")
    d.callout(W - 8, ys[2] + LH / 2, [
        [("Channel mixing,", INK, 1)],
        [("one position at a time", INK, 1)],
    ], target=(x + pw / 2 + 4, ys[2]), leader_from=(W - 158, ys[2]),
        ha="right")
    y_mid = 0.5 * (ys[1] + ys[2])  # no leader: the note covers the whole block
    d.callout(8, y_mid + LH / 2, [
        [("Together: ", INK, 1), (r"1/$c_\mathrm{o}$ + 1/9", ACCENT, 1)],
        [("of a dense 3×3 conv", INK, 1)],
    ], ha="left")

    save(d.fig, "arch-dws-block")


# --------------------------------------------------------------------------- #
# 8.7: RepVGG structural re-parameterization (train-time vs inference).       #
# --------------------------------------------------------------------------- #

def fig_repvgg_reparam():
    W, H = 760, 480
    d = Diagram(W, H)
    y0 = 116

    # ----- left: train-time three-branch block ----------------------------- #
    xl = 240
    bx = [110, 240, 370]
    pw_b = 116
    y_row = y0 + PILL_STEP / 2          # branch pills (single row, centered)
    y_plus = y0 + 2 * PILL_STEP         # ⊕ just above the merge bus (compact)
    y_relu = y_plus + PILL_STEP
    y_bus_lo = y0 - PILL_H / 2 - 11

    panel = (bx[0] - pw_b / 2 - 18, y_bus_lo - 13, bx[2] + pw_b / 2 + 18,
             y_relu + PILL_H / 2 + LABEL_BAND)
    d.container(*panel, fill=ACCENT_TINT)
    d.stage_label(panel[0] + 14, panel[3] - 9, "Train time")

    y_anchor = panel[1] - 36
    d.line([(xl, y_anchor + 14, ), (xl, y_bus_lo)])
    d.line([(bx[0], y_bus_lo), (bx[2], y_bus_lo)])
    for xb, lab in zip(bx, ["3×3 Conv + BN", "1×1 Conv + BN",
                            "BatchNorm (identity)"]):
        d.arrow(xb, y_bus_lo, y_row - PILL_H / 2)
        d.pill(xb, y_row, lab, w=pw_b)
        d.line([(xb, y_row + PILL_H / 2), (xb, y_plus - 34)])
    d.line([(bx[0], y_plus - 34), (bx[2], y_plus - 34)])
    d.arrow(xl, y_plus - 34, y_plus - PLUS_R)
    d.op_circle(xl, y_plus)
    d.arrow(xl, y_plus + PLUS_R, y_relu - PILL_H / 2)
    d.pill(xl, y_relu, "ReLU", w=pw_b)
    d.arrow(xl, y_relu + PILL_H / 2, panel[3] + 30)
    d.anchor(xl, y_anchor, "input x")

    # ----- fuse arrow ------------------------------------------------------- #
    xm = 490
    d.ax.annotate(
        "", xy=(xm + 55, y_plus - 20), xytext=(xm - 35, y_plus - 20),
        arrowprops=dict(arrowstyle="-|>,head_width=0.42,head_length=0.7",
                        color=INK, lw=1.8, shrinkA=0, shrinkB=0,
                        mutation_scale=13), zorder=6)
    d.rich(xm + 10, y_plus + 2, [("fuse", INK, True)], 12.5, ha="center")
    d.rich(xm + 10, y_plus - 40, [("weight algebra only,", GRAY_TEXT, False)],
           10, ha="center", italic=True)
    d.rich(xm + 10, y_plus - 54, [("identical outputs", GRAY_TEXT, False)],
           10, ha="center", italic=True)

    # ----- right: inference-time single conv (same rhythm/baseline) --------- #
    xr = 650
    panel_r = (xr - pw_b / 2 - 22, y_bus_lo - 13, xr + pw_b / 2 + 22,
               y_relu + PILL_H / 2 + LABEL_BAND)
    d.container(*panel_r, fill=ACCENT_TINT)
    d.stage_label(panel_r[0] + 14, panel_r[3] - 9, "Inference")
    d.arrow(xr, y_anchor + 14, y_row + PILL_STEP - PILL_H / 2)
    d.novelty(xr, y_row + PILL_STEP, "", "3×3 Conv", " (fused)", w=pw_b + 8)
    d.arrow(xr, y_row + PILL_STEP + PILL_H / 2, y_relu - PILL_H / 2)
    d.pill(xr, y_relu, "ReLU", w=pw_b)
    d.arrow(xr, y_relu + PILL_H / 2, panel_r[3] + 30)
    d.anchor(xr, y_anchor, "input x")

    save(d.fig, "arch-repvgg-reparam")


# --------------------------------------------------------------------------- #
# 7.5: RepVGG kernel algebra -- three folded kernels add to one (mechanics     #
# grid family; one output channel o and the input channel i = o).             #
# --------------------------------------------------------------------------- #

def fig_repvgg_kernels():
    from arch_diagrams import MechDiagram
    cell, gap = 26.0, 19.0
    W3 = [[1, 0, 2], [0, 1, 1], [2, 1, 0]]       # the 3x3 branch, folded
    W1 = [[0, 0, 0], [0, 3, 0], [0, 0, 0]]       # the 1x1 branch, padded
    I = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]        # the identity, i = o
    fused = [[W3[r][c] + W1[r][c] + I[r][c] for c in range(3)]
             for r in range(3)]
    panels = [(W3, "3×3 branch"), (W1, "1×1 branch, padded"),
              (I, "identity"), (fused, "fused")]
    d = MechDiagram(4 * 3 * cell + 6 * gap + 40, 3 * cell + 52)
    y, ty = 20, 20 + 3 * cell + 10
    x = 20
    for k, (K, title) in enumerate(panels):
        values = {(r, c): str(K[r][c]) for r in range(3) for c in range(3)}
        shaded = {(r, c) for r in range(3) for c in range(3) if K[r][c]}
        w, _ = d.grid(x, y, 3, 3, cell, shaded=shaded, values=values,
                      title=title, title_y=ty)
        x += w + gap
        if k < len(panels) - 1:
            d.op(x, y + 1.5 * cell, "=" if k == 2 else "+")
            x += gap
    save(d.fig, "arch-repvgg-kernels")


# --------------------------------------------------------------------------- #
# 8.8: AnyNet design space (symbolic stem/body/head).                         #
# --------------------------------------------------------------------------- #

def fig_anynet():
    W, H = 590, 760
    d = Diagram(W, H)
    x, pw = 230, 190
    y0 = 116

    groups = [
        {"ops": ["3×3 Conv, c₀, s2"]},
        {"ops": ["ResNeXt block, c₁, g₁, k₁"], "tint": True, "repeat": "d₁",
         "label": "Stage 1"},
        {"ops": ["ResNeXt block, c₂, g₂, k₂"], "tint": True, "repeat": "d₂",
         "label": "Stage 2"},
        {"ops": ["ResNeXt block, c₃, g₃, k₃"], "tint": True, "repeat": "d₃",
         "label": "Stage 3"},
        {"ops": ["ResNeXt block, c₄, g₄, k₄"], "tint": True, "repeat": "d₄",
         "label": "Stage 4"},
        {"ops": ["Global AvgPool", "Dense n"]},
    ]
    y_out, outer, lay = _blocked_column(
        d, x, groups, pw, y0, "AnyNet design space",
        anchor_text="r×r image")

    xs = outer[2] + 8               # just outside the grey outer container
    def between(a, b):
        return 0.5 * (lay[a][3] + lay[b][1][0] - PILL_H / 2)
    d.shape_note(xs, between(0, 1) - 4, "(c₀, r/2 × r/2)")
    d.shape_note(xs, between(1, 2) - 4, "(c₁, r/4 × r/4)")
    d.shape_note(xs, between(2, 3) - 4, "(c₂, r/8 × r/8)")
    d.shape_note(xs, between(3, 4) - 4, "(c₃, r/16 × r/16)")
    d.shape_note(xs, between(4, 5) - 4, "(c₄, r/32 × r/32)")

    y_s2 = 0.5 * (lay[2][2] + lay[2][3])
    d.callout(W - 8, y_s2 + 1.5 * LH, [
        [("Per stage: depth ", INK, 1), ("d", ACCENT, 1), (",", INK, 1)],
        [("width ", INK, 1), ("c", ACCENT, 1), (",", INK, 1)],
        [("group width ", INK, 1), ("g", ACCENT, 1), (",", INK, 1)],
        [("bottleneck ratio ", INK, 1), ("k", ACCENT, 1)],
    ], target=(x + pw / 2 + BLOCK_PAD + 4, y_s2),
        leader_from=(W - 152, y_s2), ha="right")

    save(d.fig, "arch-anynet")


# --------------------------------------------------------------------------- #
# 8.3: Normalization axes — which numbers share one mean and variance.        #
# --------------------------------------------------------------------------- #

def fig_norm_axes():
    """Five (N, C, H·W) cubes; the shaded cells share one mean/variance."""
    from matplotlib.patches import Polygon
    from arch_diagrams import FAMILY_BLUE
    from figstyle import tokens as T

    NC, NH, NN = 6, 5, 4         # channels (across), positions (up), examples (depth)
    cell = 13.0                  # front-face cell size
    dx, dy = 8.5, 6.5            # oblique offset per example (depth)
    gap = 44.0
    cube_w = NC * cell + NN * dx
    W = 5 * cube_w + 4 * gap + 40
    H = 200
    d = Diagram(W, H)
    grid_c, edge_c = T.FAINT, INK

    def quad(pts, shaded):
        d.ax.add_patch(Polygon(pts, closed=True,
                               facecolor=FAMILY_BLUE if shaded else "white",
                               edgecolor=grid_c, lw=0.6, zorder=3))

    panels = [
        ("Batch norm", lambda n, c, h: c == 2),
        ("Layer norm", lambda n, c, h: n == 0),
        ("Group norm", lambda n, c, h: n == 0 and c in (2, 3)),
        ("Instance norm", lambda n, c, h: n == 0 and c == 2),
        ("Layer norm per position", lambda n, c, h: n == 0 and h == 2),
    ]
    y0 = 52.0                    # bottom of the front face
    for k, (title, sel) in enumerate(panels):
        x0 = 30 + k * (cube_w + gap)

        def P(c, h, n):          # front-bottom-left corner of cell (c, h, n)
            return (x0 + c * cell + n * dx, y0 + h * cell + n * dy)

        # top face (h = NH), cells (c, n), back to front
        for n in reversed(range(NN)):
            for c in range(NC):
                quad([P(c, NH, n), P(c + 1, NH, n), P(c + 1, NH, n + 1),
                      P(c, NH, n + 1)], sel(n, c, NH - 1))
        # right face (c = NC), cells (n, h)
        for n in reversed(range(NN)):
            for h in range(NH):
                quad([P(NC, h, n), P(NC, h, n + 1), P(NC, h + 1, n + 1),
                      P(NC, h + 1, n)], sel(n, NC - 1, h))
        # front face (n = 0), cells (c, h)
        for c in range(NC):
            for h in range(NH):
                quad([P(c, h, 0), P(c + 1, h, 0), P(c + 1, h + 1, 0),
                      P(c, h + 1, 0)], sel(0, c, h))
        # outline of the cube
        outline = [P(0, 0, 0), P(NC, 0, 0), P(NC, 0, NN), P(NC, NH, NN),
                   P(0, NH, NN), P(0, NH, 0), P(0, 0, 0)]
        d.ax.plot(*zip(*outline), color=edge_c, lw=1.1, zorder=5)
        d.ax.plot(*zip(P(0, NH, 0), P(NC, NH, 0), P(NC, 0, 0)),
                  color=edge_c, lw=1.1, zorder=5)
        d.ax.plot(*zip(P(NC, NH, 0), P(NC, NH, NN)), color=edge_c, lw=1.1,
                  zorder=5)
        # axis labels
        fs = 12.5
        d.ax.text(P(NC / 2, 0, 0)[0], y0 - 12, "C", fontsize=fs,
                  color=INK, family="Source Sans 3", style="italic",
                  ha="center", va="center", zorder=8)
        d.ax.text(x0 - 8, y0 + NH * cell / 2, "H·W", fontsize=fs,
                  color=INK, family="Source Sans 3", style="italic",
                  ha="right", va="center", zorder=8)
        xn, yn = P(NC, 0, NN / 2)
        d.ax.text(xn + 7, yn - 5, "N", fontsize=fs, color=INK,
                  family="Source Sans 3", style="italic", ha="left",
                  va="center", zorder=8)
        # panel title below
        d.ax.text(x0 + cube_w / 2 - 4, y0 - 36, title, fontsize=13,
                  color=INK, family="Source Sans 3", ha="center",
                  va="center", zorder=8)

    save(d.fig, "arch-norm-axes")


FIGURES = {
    "arch-resnet-vs-convnext-block": fig_resnet_vs_convnext_block,
    "arch-inception-block": fig_inception_block,
    "arch-alexnet": fig_alexnet,
    "arch-vgg": fig_vgg,
    "arch-nin": fig_nin,
    "arch-resnet-block": fig_resnet_block,
    "arch-resnext-block": fig_resnext_block,
    "arch-densenet-block": fig_densenet_block,
    "arch-resnet18": fig_resnet18,
    "arch-convnext": fig_convnext,
    "arch-dws-block": fig_dws_block,
    "arch-repvgg-reparam": fig_repvgg_reparam,
    "arch-repvgg-kernels": fig_repvgg_kernels,
    "arch-anynet": fig_anynet,
    "arch-norm-axes": fig_norm_axes,
    "arch-residual-concept": fig_residual_concept,
}


if __name__ == "__main__":
    import sys
    # Optional figure ids on the command line regenerate only those figures.
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
    from arch_diagrams import WRITTEN
    print("\n".join(WRITTEN))
