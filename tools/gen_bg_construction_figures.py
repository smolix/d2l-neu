#!/usr/bin/env python3
"""Figures for two Computation sections: ``model-construction.md`` (the module
tree of a config-built model, with registry names) and ``custom-layers.md``
(what RMSNorm does to a vector).

The tree is a concept diagram drawn with the deterministic SVG composer
(``figstyle.svg``); the RMSNorm figure is a geometric picture drawn with the
shared matplotlib house style of ``gen_mdl_figures.py``.  The notebooks
include the SVGs with no drawing code.

Run with the repo's pytorch venv:

    .venv-pytorch/bin/python tools/gen_bg_construction_figures.py

Writes ``img/bg-module-names.svg`` and ``img/bg-rmsnorm.svg``.  Byte-idempotent.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_mdl_figures as fl  # importing applies the shared style + helpers
from figstyle import tokens as T
from figstyle.svg import Figure, Span

np, plt = fl.np, fl.plt


# =========================================================================== #
# model-construction.md: the module tree of build(MLPConfig()) with the       #
# registry name of every child on the edge that leads to it.  The point: a    #
# parameter's name is the path of registry names from the root to it.        #
# =========================================================================== #

ROW = 92.0          # vertical distance between tree levels
FS_NODE = T.FS_SMALL
FS_EDGE = T.FS_SMALL


def _node(f, cx, cy, label, *, on_path=False, leaf=False, stacked=False):
    """A module (rounded block) or a parameter (leaf tile); returns its rect."""
    accent = T.BLUE if on_path else None
    if stacked:   # three identical children drawn as a small stack of cards
        for k in (2, 1):
            x, y, w, h = _measure(cx, cy, label)
            f.rect(x + 5 * k, y + 5 * k, w, h, fill=T.PAPER, stroke=T.FAINT,
                   sw=T.SW_HAIR, r=T.R_BLOCK)
    spans = [Span(label, "m" if leaf else "r")]
    return f.block(cx, cy, spans, accent=accent, size=FS_NODE)


def _measure(cx, cy, label):
    from figstyle.textmetrics import measure
    w = measure(label, FS_NODE).width + 2 * T.PAD_BLOCK_X
    h = measure("Hg", FS_NODE).height + 2 * T.PAD_BLOCK_Y
    return cx - w / 2, cy - h / 2, w, h


def _edges(f, parent, children, *, path_child=None):
    """Elbow connectors from a parent rect to child rects, each labelled with
    its registry name just right of the child's vertical drop."""
    px, py, pw, ph = parent
    x0, y_par = px + pw / 2, py + ph
    tops = [(c[0] + c[2] / 2, c[1]) for c, _ in children]
    y_mid = y_par + 0.42 * (tops[0][1] - y_par)
    xs = [x for x, _ in tops]
    f.line(x0, y_par, x0, y_mid, stroke=T.MUTED, sw=T.SW_HAIR)
    f.line(min(xs + [x0]), y_mid, max(xs + [x0]), y_mid,
           stroke=T.MUTED, sw=T.SW_HAIR)
    for (cx, ctop), (_, name) in zip(tops, children):
        f.line(cx, y_mid, cx, ctop, stroke=T.MUTED, sw=T.SW_HAIR)
        f.text(cx + 6, 0.5 * (y_mid + ctop) + 1, [Span(name, "m")],
               size=FS_EDGE, color=T.INK, anchor="start")
    if path_child is not None:   # redraw the highlighted route on top
        (cx, ctop) = tops[path_child]
        f.polyline([(x0, y_par), (x0, y_mid), (cx, y_mid), (cx, ctop)],
                   stroke=T.BLUE.base, sw=T.SW_HEAVY)


def fig_module_names():
    f = Figure()
    y0, y1, y2, y3, y4 = 0.0, ROW, 2 * ROW, 3 * ROW, 4 * ROW

    root = _node(f, 300, y0, "Sequential", on_path=True)
    lin0 = _node(f, 60, y1, "Linear")
    res1 = _node(f, 250, y1, "ResidualBlock", on_path=True)
    res2 = _node(f, 440, y1, "ResidualBlock", stacked=True)
    lin5 = _node(f, 590, y1, "Linear")
    _edges(f, root, [(lin0, "0"), (res1, "1"), (res2, "2, 3, 4"),
                     (lin5, "5")], path_child=1)

    body = _node(f, 250, y2, "Sequential", on_path=True)
    _edges(f, res1, [(body, "body")], path_child=0)

    l0 = _node(f, 150, y3, "Linear", on_path=True)
    relu = _node(f, 250, y3, "ReLU")
    l2 = _node(f, 350, y3, "Linear")
    _edges(f, body, [(l0, "0"), (relu, "1"), (l2, "2")], path_child=0)

    w = _node(f, 100, y4, "256 × 256", leaf=True, on_path=True)
    b = _node(f, 215, y4, "256", leaf=True)
    _edges(f, l0, [(w, "weight"), (b, "bias")], path_child=0)

    # the name of the highlighted parameter = its path of registry names
    f.text(268, y4, [Span("name: ", "r", color=T.INK),
                     Span("1.body.0.weight", "m", color=T.BLUE.dark)],
           size=T.FS_LABEL, anchor="start")

    f.save("bg-module-names", out_dir=fl.IMG_DIR,
           desc="Module tree of a residual MLP built from a config: a "
                "Sequential root with children named 0 to 5; child 1, a "
                "ResidualBlock, holds a Sequential named body whose child 0 "
                "is a Linear layer with a weight and a bias. The highlighted "
                "path spells the parameter name 1.body.0.weight.")
    fl.WRITTEN.append(os.path.join(fl.IMG_DIR, "bg-module-names.svg"))


# =========================================================================== #
# custom-layers.md: RMSNorm in two dimensions.  (a) With gain 1, every input  #
# moves along its ray from the origin onto the circle of radius sqrt(d): the  #
# direction is kept and the length discarded.  (b) The gain g = (1.5, 0.5)    #
# stretches that circle into an ellipse with semi-axes 1.5 sqrt 2, 0.5 sqrt 2.#
# =========================================================================== #

def _rmsnorm(x, g=(1.0, 1.0)):
    x = np.asarray(x, float)
    return np.asarray(g) * x / np.sqrt((x ** 2).mean(-1, keepdims=True))


def fig_rmsnorm():
    # two inputs on one ray (radii 0.6 and 2.9) land on the same output
    ang = np.radians([30, 30, 118, 215, 297])
    rad = np.array([0.6, 2.9, 2.8, 0.8, 2.7])
    inputs = rad[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
    g = np.array([1.5, 0.5])
    t = np.linspace(0, 2 * np.pi, 400)
    circ = np.sqrt(2) * np.stack([np.cos(t), np.sin(t)], 1)
    lim = ((-3.1, 3.1), (-3.1, 3.1))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.0, 4.4))

    # (a) normalization: inputs (blue) move along their rays to the circle
    out = _rmsnorm(inputs)
    axa.plot(circ[:, 0], circ[:, 1], color=fl.ORANGE, lw=2.0, zorder=2)
    for x, y in zip(inputs, out):
        axa.plot([0, max(np.linalg.norm(x), np.linalg.norm(y)) * x[0] /
                  np.linalg.norm(x)],
                 [0, max(np.linalg.norm(x), np.linalg.norm(y)) * x[1] /
                  np.linalg.norm(x)],
                 color=fl.LIGHT, lw=1.2, ls="--", zorder=1)
        fl.arrow(axa, x, y, color=fl.GRAY, lw=1.3, mut=11)
    axa.scatter(*inputs.T, s=42, color=fl.BLUE, zorder=4)
    axa.scatter(*out.T, s=42, color=fl.ORANGE, zorder=5)
    fl.axis_cross(axa, *lim, color="black")
    axa.text(-1.55, 2.47, r"inputs $\mathbf{x}$", color=T.BLUE.dark,
             fontsize=15, ha="right", va="center")
    axa.text(1.55, -1.2, r"radius $\sqrt{2}$", color=T.ORANGE.dark,
             fontsize=15, ha="left", va="top")
    fl.clean_axes(axa, lim=lim, hide=True)

    # (b) gain: the circle becomes an ellipse with semi-axes g_i sqrt(2)
    ell = circ * g
    axb.plot(circ[:, 0], circ[:, 1], color=fl.ORANGE, lw=1.4, ls="--",
             zorder=2)
    axb.plot(ell[:, 0], ell[:, 1], color=fl.GREEN, lw=2.2, zorder=3)
    outg = _rmsnorm(inputs, g)
    for y, z in zip(out, outg):
        fl.arrow(axb, y, z, color=fl.GRAY, lw=1.3, mut=11)
    axb.scatter(*out.T, s=42, color=fl.ORANGE, zorder=4)
    axb.scatter(*outg.T, s=42, color=fl.GREEN, zorder=5)
    fl.axis_cross(axb, *lim, color="black")
    axb.text(-2.95, -2.35, r"$\mathbf{g} = (1.5,\ 0.5)$",
             color=T.GREEN.dark, fontsize=15, ha="left", va="top")
    fl.clean_axes(axb, lim=lim, hide=True)

    fig.subplots_adjust(wspace=0.08)
    fl.save(fig, "bg-rmsnorm")


FIGURES = [fig_module_names, fig_rmsnorm]


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
        print(f"  {os.path.basename(p):32s} {size:>8,d} bytes")


if __name__ == "__main__":
    main()
