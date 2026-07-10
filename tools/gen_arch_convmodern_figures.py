#!/usr/bin/env python3
"""Gallery-style architecture figures for chapter_convolutional-modern (ch8).

    .venv-pytorch/bin/python tools/gen_arch_convmodern_figures.py

Writes ``img/arch-<id>.svg`` (byte-idempotent).  Style + primitives:
``tools/arch_diagrams.py`` / docs/convnet-rewrite/figure-style.md.
Layout rules from the pilot review: one spine rhythm (PILL_STEP) everywhere,
LABEL_BAND of headroom under each panel label, axis-aligned leaders only,
repeat braces span their full panel.
"""

from arch_diagrams import (
    Diagram, save,
    ACCENT, ACCENT_TINT, ACCENT2, ACCENT2_TINT, INK,
    PILL_H, PILL_STEP, LABEL_BAND, PLUS_R,
)

LH = 12.5 * 1.25          # callout line height (must match callout())


# --------------------------------------------------------------------------- #
# Pilot 1: ResNet-50 bottleneck block vs. ConvNeXt block (two accents).       #
# --------------------------------------------------------------------------- #

def fig_resnet_vs_convnext_block():
    W, H = 800, 620
    d = Diagram(W, H)

    y_in = 56                    # spine entry above the anchor (shared)
    y0 = 116                     # first pill center (shared baseline)
    y_out = H - 14               # output arrow tips (shared)

    def column(x, ops, pw, tint, label, novelty_idx=None, extra_after_plus=None):
        """One block panel: ops on a constant PILL_STEP rhythm, ⊕ one step
        above the last op, optional trailing op, skip routed inside."""
        ys = [y0 + i * PILL_STEP for i in range(len(ops))]
        y_plus = ys[-1] + PILL_STEP
        y_top_el = y_plus + (PILL_STEP if extra_after_plus else 0)
        top_edge = (y_top_el + (PILL_H / 2 if extra_after_plus else PLUS_R)
                    + LABEL_BAND)
        panel = (x - 97, y0 - PILL_H / 2 - 23, x + 97, top_edge)
        d.container(*panel, fill=tint)
        d.stage_label(panel[0] + 14, panel[3] - 9, label)

        d.arrow(x, y_in - 26, ys[0] - PILL_H / 2)
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
        d.anchor(x, y_in - 38, "input x")
        return ys, y_plus, panel

    # ----- left: ResNet-50 bottleneck (accent 2, amber) -------------------- #
    xl, pw_l = 215, 108
    resnet_ops = ["1×1 Conv, 64", "BatchNorm", "ReLU",
                  "3×3 Conv, 64", "BatchNorm", "ReLU",
                  "1×1 Conv, 256", "BatchNorm"]
    ys_l, y_plus_l, panel_l = column(xl, resnet_ops, pw_l, ACCENT2_TINT,
                                     "ResNet-50 bottleneck block",
                                     extra_after_plus="ReLU")
    d.shape_note(xl - 66, y_in - 14, "(256, 56×56)", ha="right")

    # ----- right: ConvNeXt block (accent 1, blue) --------------------------- #
    xr, pw_r = 570, 122
    nxt_ops = [("7×7 ", "Depthwise", " Conv, 96"), "LayerNorm",
               "1×1 Conv, 384", "GELU", "1×1 Conv, 96"]
    ys_r, y_plus_r, panel_r = column(xr, nxt_ops, pw_r, ACCENT_TINT,
                                     "ConvNeXt block", novelty_idx=0)
    d.shape_note(xr + 66, y_in - 14, "(96, 56×56)", ha="left")

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
    W, H = 660, 420
    d = Diagram(W, H)

    y_in = 58            # spine entry above the anchor
    y_bus_lo = 120       # lower split bus
    y_row1 = 152         # first pill row (one PILL_GAP above the bus + arrow)
    y_row2 = y_row1 + PILL_STEP + 9        # second pill row
    y_bus_hi = y_row2 + PILL_H / 2 + 19    # upper merge bus
    y_cat = y_bus_hi + 34                  # concat novelty box
    y_out = 396          # output arrow tip (outside the panel)

    xc = W / 2
    bx = [130, 260, 400, 536]   # branch column x positions
    pw = [86, 110, 110, 118]    # per-branch pill widths

    # block panel (the repeated unit) with LABEL_BAND above the concat box
    panel = (54, y_bus_lo - 24, W - 28, y_cat + PILL_H / 2 + LABEL_BAND)
    d.container(*panel, fill=ACCENT_TINT)
    d.stage_label(panel[0] + 14, panel[3] - 9, "Inception block")
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

    d.anchor(xc, y_in - 14, "input x")
    d.shape_note(xc + 12, (y_in + y_bus_lo) / 2 + 4, "(192, 28×28)")
    d.shape_note(xc + 12, y_out - 8, "(256, 28×28)")

    # callouts at the bottom margin; straight VERTICAL leaders up to targets
    d.callout(20, 30 + LH, [
        [("1×1 convolutions shrink channels", INK, 1)],
        [("before the costly ", INK, 1), ("3×3", ACCENT, 1),
         (" and ", INK, 1), ("5×5", ACCENT, 1)],
    ], target=(bx[1], y_row1 - PILL_H / 2 - 3),
        leader_from=(bx[1], 30 + LH + 10), ha="left")
    d.callout(W - 8, 30 + LH, [
        [("Four branches, four scales,", INK, 1)],
        [("one shared input", INK, 1)],
    ], target=(bx[3] - 28, y_bus_lo - 2),
        leader_from=(bx[3] - 28, 30 + LH + 10), ha="right")

    save(d.fig, "arch-inception-block")


if __name__ == "__main__":
    fig_resnet_vs_convnext_block()
    fig_inception_block()
    from arch_diagrams import WRITTEN
    print("\n".join(WRITTEN))
