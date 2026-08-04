#!/usr/bin/env python3
"""Chapter 5 (multilayer perceptrons) legacy-figure remakes — chapter-by-
chapter review loop (docs/figure-style-guide.md §9).  UNDER REVIEW.

Writes -> img/:
  mlp.svg        the pilot remake, promoted to its home chapter
  dropout2.svg   before/after dropout, in the same network-graph family

The chapter's eight mdl-mlp-* figures are owned by gen_mdl_mlp_figures.py,
gen_mdl_backprop_figures.py, gen_mdl_xor_figures.py,
gen_mdl_doubledescent_figures.py and gen_mdl_kfold_figures.py, all of
which inherit the token layer through gen_mdl_figures.py.  Kaggle
screenshots and house-pricing photos are out of scope.

Run:  python3 tools/figstyle/remake_ch05.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from figstyle import tokens as T
from figstyle.svg import Figure, sub, var

IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "img")

R = 21.0  # node radius shared by every figure in this network-graph family


def _edges(f, srcs, dsts, ys, yd):
    for xa in srcs:
        for xb in dsts:
            ang = math.atan2(yd - ys, xb - xa)
            f.arrow(xa + R * math.cos(ang), ys + R * math.sin(ang),
                    xb - (R + 2) * math.cos(ang),
                    yd - (R + 2) * math.sin(ang),
                    stroke=T.FAINT, sw=T.SW_HAIR)


def _node(f, x, y, letter, idx, accent=None):
    if accent:
        f.circle(x, y, R, fill=accent.tint, stroke=accent.base, sw=T.SW_BOX)
    else:
        f.circle(x, y, R, fill=T.PAPER, stroke=T.INK, sw=T.SW_BOX)
    f.text(x, y, [var(letter), sub(idx)], size=T.FS_LABEL)


# --------------------------------------------------------------------------- #
# mlp — the approved pilot, now owned by its home chapter.                    #
# --------------------------------------------------------------------------- #

def fig_mlp():
    f = Figure()
    ROWY = {"out": 0.0, "hid": 140.0, "in": 280.0}
    XC = 300.0

    def xs(n, gap=100.0):
        return [XC + (i - (n - 1) / 2) * gap for i in range(n)]

    nodes = {
        "in": (xs(4), "x", None),
        "hid": (xs(5), "h", T.BLUE),
        "out": (xs(3), "o", T.GREEN),
    }
    _edges(f, nodes["in"][0], nodes["hid"][0], ROWY["in"], ROWY["hid"])
    _edges(f, nodes["hid"][0], nodes["out"][0], ROWY["hid"], ROWY["out"])
    for row, (positions, letter, accent) in nodes.items():
        for i, x in enumerate(positions, 1):
            _node(f, x, ROWY[row], letter, str(i), accent)

    lx = min(nodes["in"][0]) - 175
    for row, lab in (("out", "output layer"), ("hid", "hidden layer"),
                     ("in", "input layer")):
        f.text(lx, ROWY[row], lab.upper(), size=T.FS_TINY, color=T.MUTED,
               anchor="start", tracking=T.LETTERSPACE_CAPS)

    f.save("mlp", out_dir=IMG,
           desc="A multilayer perceptron with one hidden layer.")


# --------------------------------------------------------------------------- #
# dropout2 — POINT: dropout removes hidden units (h2, h5 here), and with      #
# them every edge in or out; the surviving network is thinner.                #
# --------------------------------------------------------------------------- #

def fig_dropout2():
    f = Figure()
    Y = {"out": 0.0, "hid": 130.0, "in": 260.0}

    def panel(ox, dropped, title):
        xin = [ox + i * 95 for i in range(4)]
        xhid = [ox - 47 + i * 95 for i in range(5)]
        xout = [ox + 47 + i * 95 for i in range(3)]
        keep = [x for i, x in enumerate(xhid) if i not in dropped]
        _edges(f, xin, keep, Y["in"], Y["hid"])
        _edges(f, keep, xout, Y["hid"], Y["out"])
        for i, x in enumerate(xin, 1):
            _node(f, x, Y["in"], "x", str(i))
        for i, x in enumerate(xhid, 1):
            if i - 1 in dropped:
                # a dropped unit: ghosted ring, no label, no edges
                f.circle(x, Y["hid"], R, fill=T.PAPER, stroke=T.HAIRLINE,
                         sw=T.SW_BOX)
                d = R * 0.62
                f.line(x - d, Y["hid"] - d, x + d, Y["hid"] + d,
                       stroke=T.RED.base, sw=T.SW_BOX)
                f.line(x - d, Y["hid"] + d, x + d, Y["hid"] - d,
                       stroke=T.RED.base, sw=T.SW_BOX)
            else:
                _node(f, x, Y["hid"], "h", str(i), T.BLUE)
        for i, x in enumerate(xout, 1):
            _node(f, x, Y["out"], "o", str(i), T.GREEN)
        f.text(ox + 142, Y["out"] - 60, title, size=T.FS_TITLE, color=T.INK)

    panel(0.0, [], "before dropout")
    panel(560.0, [1, 4], "after dropout")
    f.save("dropout2", out_dir=IMG,
           desc="Dropout removes hidden units and all of their edges.")


def main():
    fig_mlp()
    fig_dropout2()
    print("wrote chapter-5 remakes to", IMG)


if __name__ == "__main__":
    main()
