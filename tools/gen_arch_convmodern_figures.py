#!/usr/bin/env python3
"""Gallery-style architecture figures for chapter_convolutional-modern (ch8).

    .venv-pytorch/bin/python tools/gen_arch_convmodern_figures.py

Writes ``img/arch-<id>.svg`` (byte-idempotent).  Style + primitives:
``tools/arch_diagrams.py`` / docs/convnet-rewrite/figure-style.md.
"""

from arch_diagrams import (
    Diagram, save, spread,
    ACCENT, ACCENT_TINT, ACCENT2, ACCENT2_TINT, INK, PILL_H, PLUS_R,
)


# --------------------------------------------------------------------------- #
# Pilot 1: ResNet-50 bottleneck block vs. ConvNeXt block (two accents).       #
# --------------------------------------------------------------------------- #

def fig_resnet_vs_convnext_block():
    W, H = 780, 560
    d = Diagram(W, H)

    y_in = 56          # spine entry above the anchor (shared baseline)
    y_first, y_last = 116, 404   # first / last pill centers on the spine
    y_plus = 444       # ⊕ height (shared)
    y_relu2 = 478      # ResNet's post-addition ReLU
    y_out = H - 14     # output arrow tips (shared)

    # ----- left panel: ResNet-50 bottleneck (accent 2, amber) -------------- #
    xl = 200           # spine x
    pw = 108           # shared pill width within the panel
    resnet_ops = ["1×1 Conv, 64", "BatchNorm", "ReLU",
                  "3×3 Conv, 64", "BatchNorm", "ReLU",
                  "1×1 Conv, 256", "BatchNorm"]
    ys = spread(y_first, y_last, len(resnet_ops))

    panel_l = (xl - 92, y_first - 36, xl + 92, y_relu2 + 32)
    d.container(*panel_l, fill=ACCENT2_TINT)
    d.stage_label(panel_l[0] + 14, panel_l[3] - 9, "ResNet-50 bottleneck block")

    d.arrow(xl, y_in - 26, ys[0] - PILL_H / 2)
    for i, (op, y) in enumerate(zip(resnet_ops, ys)):
        d.pill(xl, y, op, w=pw)
        if i < len(resnet_ops) - 1:
            d.arrow(xl, y + PILL_H / 2, ys[i + 1] - PILL_H / 2)
    d.arrow(xl, ys[-1] + PILL_H / 2, y_plus - PLUS_R)
    # skip routed on the inner (right) side, branching inside the panel
    d.skip(xl, y_first - 24, y_plus, xl + 76)
    d.arrow(xl, y_plus + PLUS_R, y_relu2 - PILL_H / 2)
    d.pill(xl, y_relu2, "ReLU", w=pw)
    d.arrow(xl, y_relu2 + PILL_H / 2, y_out)

    d.shape_note(xl - 66, y_in - 14, "(256, 56×56)", ha="right")
    d.anchor(xl, y_in - 38, "input x")

    # ----- right panel: ConvNeXt block (accent 1, blue) -------------------- #
    xr = 545
    pwr = 122
    nxt_ops = ["7×7 DWConv, 96", "LayerNorm", "1×1 Conv, 384",
               "GELU", "1×1 Conv, 96"]
    ysr = spread(y_first, y_last, len(nxt_ops))

    panel_r = (xr - 97, y_first - 36, xr + 97, y_plus + 34)
    d.container(*panel_r, fill=ACCENT_TINT)
    d.stage_label(panel_r[0] + 14, panel_r[3] - 9, "ConvNeXt block")

    d.arrow(xr, y_in - 26, ysr[0] - PILL_H / 2)
    for i, (op, y) in enumerate(zip(nxt_ops, ysr)):
        if i == 0:
            d.novelty(xr, y, "7×7 ", "Depthwise", " Conv, 96", w=pwr + 18)
        else:
            d.pill(xr, y, op, w=pwr)
        if i < len(nxt_ops) - 1:
            d.arrow(xr, y + PILL_H / 2, ysr[i + 1] - PILL_H / 2)
    d.arrow(xr, ysr[-1] + PILL_H / 2, y_plus - PLUS_R)
    # skip routed on the inner (left) side, branching inside the panel
    d.skip(xr, y_first - 24, y_plus, xr - 76)
    d.arrow(xr, y_plus + PLUS_R, y_out)

    d.shape_note(xr + 66, y_in - 14, "(96, 56×56)", ha="left")
    d.anchor(xr, y_in - 38, "input x")

    # ----- callouts (leaders never cross their own text) -------------------- #
    d.callout(8, 158, [
        [("Reduce ", INK, 1), ("256→64", ACCENT2, 1), (",", INK, 1)],
        [("process, expand", INK, 1)],
    ], target=(xl - pw / 2 - 4, ys[0] + 4), leader_from=(96, 136), ha="left")
    d.callout(8, 300, [
        [("3", ACCENT2, 1), (" norms,", INK, 1)],
        [("3", ACCENT2, 1), (" activations", INK, 1)],
    ], target=(xl - pw / 2 - 4, ys[4]), leader_from=(82, 288), ha="left")

    d.callout(W - 8, 288, [
        [("Inverted bottleneck:", INK, 1)],
        [("expand ", INK, 1), ("96→384", ACCENT, 1)],
    ], target=(xr + pwr / 2 + 4, ysr[2]), leader_from=(W - 116, 264),
        ha="right")
    d.callout(W - 8, 168, [
        [("1", ACCENT, 1), (" norm,", INK, 1)],
        [("1", ACCENT, 1), (" activation", INK, 1)],
    ], target=(xr + pwr / 2 + 4, ysr[1]), leader_from=(W - 86, 158),
        ha="right")

    save(d.fig, "arch-resnet-vs-convnext-block")


# --------------------------------------------------------------------------- #
# Pilot 2: Inception block (multi-branch layout).                             #
# --------------------------------------------------------------------------- #

def fig_inception_block():
    W, H = 620, 440
    d = Diagram(W, H)

    y_in = 26           # spine entry above the anchor
    y_bus_lo = 88       # lower split bus
    y_row1 = 132        # first pill row
    y_row2 = 222        # second pill row
    y_bus_hi = 272      # upper merge bus
    y_cat = 320         # concat novelty box
    y_out = 404         # output arrow tip (outside the panel)

    xc = W / 2
    bx = [110, 240, 380, 516]   # branch column x positions
    pw = [86, 110, 110, 118]    # per-branch pill widths

    # block panel (the repeated unit)
    panel = (34, y_bus_lo - 26, W - 28, y_cat + 40)
    d.container(*panel, fill=ACCENT_TINT)
    d.stage_label(panel[0] + 14, panel[3] - 9, "Inception block")
    d.repeat(panel[0], panel[1], "9")

    # input spine joins the lower bus (heads only where a line enters a box)
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
    d.shape_note(xc + 12, (y_in + y_bus_lo) / 2, "(192, 28×28)")
    d.shape_note(xc + 12, y_out - 10, "(256, 28×28)")

    # callouts
    d.callout(30, H - 10, [
        [("1×1 convolutions shrink channels", INK, 1)],
        [("before the costly ", INK, 1), ("3×3", ACCENT, 1),
         (" and ", INK, 1), ("5×5", ACCENT, 1)],
    ], target=(bx[1] - pw[1] / 2 - 3, y_row1 + 4),
        leader_from=(172, H - 46), ha="left")
    d.callout(W - 8, 46, [
        [("Four branches, four scales,", INK, 1)],
        [("one shared input", INK, 1)],
    ], target=(bx[3] - 24, y_bus_lo - 2),
        leader_from=(W - 130, 56), ha="right")

    save(d.fig, "arch-inception-block")


if __name__ == "__main__":
    fig_resnet_vs_convnext_block()
    fig_inception_block()
    from arch_diagrams import WRITTEN
    print("\n".join(WRITTEN))
