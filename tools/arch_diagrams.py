#!/usr/bin/env python3
"""Primitive library for network *architecture / dataflow* diagrams
("gallery style", docs/convnet-rewrite/figure-style.md).

Visual grammar: vertical spine bottom->top, white pills with near-stadium
corners, nested rounded containers (network -> block), one accent color per
figure (a second only for design-vs-design comparisons), exactly one dark
"novelty" box per figure, rectilinear skip connections into a ⊕ on the spine,
dotted leaders to bold margin callouts, gray italic ``(c, r×r)`` shape notes.

Mechanics diagrams (kernel grids, padding, dilation) do NOT use this style;
they follow the plain white-grid family of ``img/conv-pad.svg``.

Byte-idempotency: fixed ``svg.hashsalt``, no timestamps, fonts embedded as
paths (the book's own Source Sans 3 / Inconsolata from ``static/fonts``), so
macOS and the GPU box produce identical bytes.  Never add unseeded randomness.

All geometry is laid out in *points* (1 unit = 1 pt); text extents are
measured with ``TextPath`` so layout is renderer-independent.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("svg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath

# --------------------------------------------------------------------------- #
# Pinned style constants — never override per figure.                         #
# --------------------------------------------------------------------------- #

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(_ROOT, "img")
_FONT_DIR = os.path.join(_ROOT, "static", "fonts")

for _f in ("SourceSans3-Regular.ttf", "SourceSans3-Bold.ttf",
           "SourceSans3-Italic.ttf", "Inconsolata-Regular.ttf"):
    fontManager.addfont(os.path.join(_FONT_DIR, _f))

SANS = "Source Sans 3"
MONO = "Inconsolata"

INK = "#000000"                 # strokes, arrows, pill borders, labels
STROKE = 1.2                    # pt — pill borders, arrows, skip lines

# Accent 1 (default): d2l blue.
ACCENT = "#0B6BB2"              # saturated tone: numbers, repeat, keywords
ACCENT_TINT = "#CDE8FA"         # repeated-block panel fill (blue on white)
# Accent 2 (comparisons only): warm amber.
ACCENT2 = "#B45309"
ACCENT2_TINT = "#FBE8D3"

NOVELTY_FILL = "#3B3B3B"        # near-black novelty box, white text
CONTAINER_FILL = "#E4E4E4"      # outermost network container
INSET_FILL = "#ECECEC"          # dashed inset panels
GRAY_TEXT = "#6E6E6E"           # shape notes, stage labels, input anchor

PILL_H = 26.0                   # pt — pill height (single-line)
PILL_GAP = 15.0                 # pt — default vertical gap between spine ops
PILL_STEP = PILL_H + PILL_GAP   # pt — default spine rhythm (center to center)
LABEL_BAND = 34.0               # pt — panel headroom above the top op for the
                                #      stage label (keeps label off the pills)
PILL_FS = 11.5                  # pill labels
CALLOUT_FS = 12.5               # margin callouts (bold)
NOTE_FS = 10.0                  # shape notes / stage labels
ANCHOR_FS = 11.0                # monospace input anchor
PILL_PAD_X = 13.0               # horizontal text padding inside a pill
PILL_ROUND = 0.45 * PILL_H      # corner rounding (near-stadium)
CONTAINER_ROUND = 18.0
PLUS_R = 7.5                    # radius of ⊕ / ⊗ circles
LEADER = dict(color=INK, lw=1.7, ls=(0, (0.9, 2.4)),
              solid_capstyle="round", dash_capstyle="round")

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "svg.fonttype": "path",
    "svg.hashsalt": "arch-diagrams",
})

WRITTEN: list[str] = []


def save(fig, name: str) -> None:
    """Write ``img/<name>.svg`` byte-idempotently and close the figure."""
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, f"{name}.svg")
    fig.savefig(path, format="svg", bbox_inches="tight",
                metadata={"Date": None})
    plt.close(fig)
    WRITTEN.append(path)


# --------------------------------------------------------------------------- #
# Text measurement (renderer-independent).                                    #
# --------------------------------------------------------------------------- #

def _prop(bold=False, italic=False, mono=False):
    return FontProperties(family=MONO if mono else SANS,
                          weight="bold" if bold else "regular",
                          style="italic" if italic else "normal")


def text_width(s: str, fs: float, bold=False, italic=False, mono=False) -> float:
    if not s:
        return 0.0
    # Sentinel glyphs on both sides so leading/trailing spaces (which have no
    # ink and are dropped by TextPath extents) still count toward the advance.
    prop = _prop(bold, italic, mono)
    w_wrap = TextPath((0, 0), f"H{s}H", size=fs, prop=prop).get_extents().width
    w_sent = TextPath((0, 0), "HH", size=fs, prop=prop).get_extents().width
    return float(w_wrap - w_sent)


# --------------------------------------------------------------------------- #
# The canvas.                                                                 #
# --------------------------------------------------------------------------- #

class Diagram:
    """A fixed-size canvas addressed in points, y increasing upward."""

    def __init__(self, width: float, height: float):
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width / 72.0, height / 72.0))
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.axis("off")
        self.ax = ax

    # -- rich text: a list of (text, color, bold) segments on one line ------ #
    def rich(self, x, y, segments, fs, ha="center", va="center",
             italic=False, mono=False, zorder=10):
        """Draw mixed-color text.  ``segments`` = [(str, color, bold), ...].
        Returns total width."""
        widths = [text_width(s, fs, bold=b, italic=italic, mono=mono)
                  for s, _c, b in segments]
        total = sum(widths)
        if ha == "center":
            cur = x - total / 2.0
        elif ha == "right":
            cur = x - total
        else:
            cur = x
        for (s, color, bold), w in zip(segments, widths):
            self.ax.text(cur, y, s, fontsize=fs, color=color,
                         family=MONO if mono else SANS,
                         fontweight="bold" if bold else "regular",
                         style="italic" if italic else "normal",
                         ha="left", va=va, zorder=zorder)
            cur += w
        return total

    def _label_segments(self, label, color):
        """Normalize a pill label to segments: str -> single black segment."""
        if isinstance(label, str):
            return [(label, color, False)]
        return label

    # -- pills --------------------------------------------------------------#
    def pill_width(self, label, fs=PILL_FS):
        segs = self._label_segments(label, INK)
        w = sum(text_width(s, fs, bold=b) for s, _c, b in segs)
        return w + 2 * PILL_PAD_X

    def pill(self, x, y, label, w=None, fs=PILL_FS, zorder=6):
        """White pill centered at (x, y).  Returns its half-height."""
        segs = self._label_segments(label, INK)
        if w is None:
            w = self.pill_width(segs, fs)
        self.ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - PILL_H / 2), w, PILL_H,
            boxstyle=f"round,pad=0,rounding_size={PILL_ROUND}",
            facecolor="white", edgecolor=INK, linewidth=STROKE,
            zorder=zorder - 1))
        self.rich(x, y, segs, fs, zorder=zorder)
        return PILL_H / 2

    def novelty(self, x, y, pre, keyword, post="", w=None, fs=PILL_FS,
                accent=ACCENT, zorder=6):
        """The one new-op box: near-black fill, white text, accent keyword."""
        segs = [(pre, "white", False), (keyword, accent, True),
                (post, "white", False)]
        segs = [s for s in segs if s[0]]
        if w is None:
            w = sum(text_width(s, fs, bold=b) for s, _c, b in segs) \
                + 2 * PILL_PAD_X
        self.ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - PILL_H / 2), w, PILL_H,
            boxstyle=f"round,pad=0,rounding_size={PILL_ROUND}",
            facecolor=NOVELTY_FILL, edgecolor=NOVELTY_FILL,
            linewidth=STROKE, zorder=zorder - 1))
        self.rich(x, y, segs, fs, zorder=zorder)
        return PILL_H / 2

    # -- containers ----------------------------------------------------------#
    def container(self, x0, y0, x1, y1, fill=CONTAINER_FILL, dashed=False,
                  round_=CONTAINER_ROUND, edge=None, zorder=1):
        style = dict(facecolor=fill, linewidth=STROKE if (dashed or edge) else 0,
                     edgecolor=edge or (INK if dashed else "none"),
                     zorder=zorder)
        if dashed:
            style["linestyle"] = (0, (1.4, 2.2))
            style["linewidth"] = 1.6
        self.ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={round_}", **style))

    def stage_label(self, x, y, text, ha="left"):
        """Gray small-caps container/stage label (top-left convention)."""
        self.ax.text(x, y, text.upper(), fontsize=NOTE_FS, color=GRAY_TEXT,
                     family=SANS, ha=ha, va="top", zorder=8)

    def repeat(self, x, y0, y1, n_text, accent=ACCENT, depth=9.0):
        """Repeat multiplier beside a block panel: a curly brace spanning the
        panel's FULL height (y0..y1), cusp at x pointing left, with ``n ×`` to
        its left at mid-height."""
        ym = 0.5 * (y0 + y1)
        xa = x + depth               # arm x (brace opens toward the panel)
        verts = [(xa, y1),
                 (x + depth * 0.15, y1), (xa, ym), (x, ym),
                 (xa, ym), (x + depth * 0.15, y0), (xa, y0)]
        codes = [MplPath.MOVETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        self.ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none",
                                    edgecolor=accent, linewidth=1.8,
                                    capstyle="round", zorder=8))
        self.rich(x - 6, ym, [(n_text + " ", accent, True), ("×", accent, True)],
                  13, ha="right", va="center", zorder=8)

    # -- spine plumbing -------------------------------------------------------#
    def arrow(self, x, y0, y1, zorder=4):
        """Vertical spine arrow from y0 up to y1 (head at y1)."""
        self.ax.annotate(
            "", xy=(x, y1), xytext=(x, y0),
            arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.55",
                            color=INK, lw=STROKE, shrinkA=0, shrinkB=0,
                            mutation_scale=11),
            zorder=zorder)

    def line(self, pts, zorder=4):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=INK, lw=STROKE, solid_joinstyle="miter",
                     solid_capstyle="butt", zorder=zorder)

    def op_circle(self, x, y, op="+", zorder=6):
        """⊕ (op='+') or ⊗ (op='x') on the spine."""
        self.ax.add_patch(Circle((x, y), PLUS_R, facecolor="white",
                                 edgecolor=INK, linewidth=STROKE,
                                 zorder=zorder))
        r = PLUS_R * 0.52
        if op == "+":
            self.ax.plot([x - r, x + r], [y, y], color=INK, lw=STROKE,
                         zorder=zorder + 1)
            self.ax.plot([x, x], [y - r, y + r], color=INK, lw=STROKE,
                         zorder=zorder + 1)
        else:
            d = r * 0.78
            self.ax.plot([x - d, x + d], [y - d, y + d], color=INK,
                         lw=STROKE, zorder=zorder + 1)
            self.ax.plot([x - d, x + d], [y + d, y - d], color=INK,
                         lw=STROKE, zorder=zorder + 1)

    def skip(self, x_spine, y_from, y_to, x_out, op="+", zorder=3):
        """Rectilinear skip: leave the spine at (x_spine, y_from), route via
        x_out, enter the ⊕ at (x_spine, y_to) from the side.  Draws the ⊕."""
        self.op_circle(x_spine, y_to, op=op)
        sgn = 1 if x_out > x_spine else -1
        self.line([(x_spine, y_from), (x_out, y_from), (x_out, y_to)],
                  zorder=zorder)
        # horizontal entry into the circle, with an arrow head
        self.ax.annotate(
            "", xy=(x_spine + sgn * PLUS_R, y_to), xytext=(x_out, y_to),
            arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.55",
                            color=INK, lw=STROKE, shrinkA=0, shrinkB=0,
                            mutation_scale=11),
            zorder=zorder)

    # -- annotation layer -----------------------------------------------------#
    def leader(self, p0, p1, zorder=5):
        """Dotted leader line between two points.  Keep leaders axis-aligned
        (horizontal or vertical) — pass points sharing an x or a y."""
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], zorder=zorder, **LEADER)

    def callout(self, x, y, lines, target=None, ha="left", fs=CALLOUT_FS,
                leader_from=None, zorder=9):
        """Bold black margin callout; numbers in accent.
        ``lines`` = list of segment-lists (one per line, top to bottom).
        ``target`` = point the dotted leader points at;
        ``leader_from`` overrides the leader's origin on the text side."""
        lh = fs * 1.25
        y0 = y
        for i, segs in enumerate(lines):
            segs = [(s, c, True) for s, c, _b in segs]   # callouts are bold
            self.rich(x, y0 - i * lh, segs, fs, ha=ha, va="center",
                      zorder=zorder)
        if target is not None:
            src = leader_from if leader_from is not None else (x, y)
            self.leader(src, target)

    def shape_note(self, x, y, text, ha="left"):
        """Gray italic ``(c, r×r)`` note beside a spine arrow."""
        self.ax.text(x, y, text, fontsize=NOTE_FS, color=GRAY_TEXT,
                     family=SANS, style="italic", ha=ha, va="center",
                     zorder=8)

    def anchor(self, x, y, text):
        """Monospace gray input anchor at the bottom of the spine."""
        self.ax.text(x, y, text, fontsize=ANCHOR_FS, color="#3d3d3d",
                     family=MONO, ha="center", va="center", zorder=8)


# --------------------------------------------------------------------------- #
# Grid primitives for MECHANICS diagrams (the conv-pad.svg family: white     #
# grids, light-blue shaded cells, thin black strokes, black labels).  These  #
# deliberately do NOT use the gallery accent colors.                         #
# --------------------------------------------------------------------------- #

FAMILY_BLUE = "#B2D9FF"     # the family's shaded-cell blue (from conv-pad.svg)
GRID_LW = 1.2
GRID_FS = 11.0              # in-cell values and grid titles


class MechDiagram(Diagram):
    """Canvas for mechanics figures; adds the grid family's primitives."""

    def grid(self, x, y, rows, cols, cell=20.0, shaded=(), dashed=False,
             values=None, title=None, zorder=4):
        """Family grid with lower-left corner at (x, y).  ``shaded`` holds
        (row, col) pairs, row 0 at the TOP (reading order).  ``values`` maps
        (row, col) -> str.  Returns (width, height)."""
        from matplotlib.patches import Rectangle
        shaded = set(shaded)
        style = dict(edgecolor=INK, linewidth=GRID_LW)
        if dashed:
            style["linestyle"] = (0, (2.0, 2.0))
        for r in range(rows):
            for c in range(cols):
                cx, cy = x + c * cell, y + (rows - 1 - r) * cell
                self.ax.add_patch(Rectangle(
                    (cx, cy), cell, cell, zorder=zorder,
                    facecolor=FAMILY_BLUE if (r, c) in shaded else "white",
                    **style))
                if values and (r, c) in values:
                    self.ax.text(cx + cell / 2, cy + cell / 2, values[(r, c)],
                                 fontsize=GRID_FS, color=INK, family=SANS,
                                 ha="center", va="center", zorder=zorder + 1)
        if title:
            self.ax.text(x + cols * cell / 2, y + rows * cell + 10, title,
                         fontsize=GRID_FS, color=INK, family=SANS,
                         ha="center", va="bottom", zorder=zorder + 1)
        return cols * cell, rows * cell

    def grid_stack(self, x, y, n, rows, cols, cell=20.0, offset=7.0,
                   shaded_front=(), shade_backs=False, title=None, zorder=4):
        """A stack of ``n`` channel grids, drawn back-to-front with the front
        grid's lower-left at (x, y); deeper channels offset up-right (the
        conv-multi-in.svg convention).  ``shaded_front`` shades cells of the
        FRONT grid only; ``shade_backs`` shades every cell of the back grids
        (for kernels that are active through the whole depth).
        Returns (total_width, total_height)."""
        allc = {(r, c) for r in range(rows) for c in range(cols)}
        for i in range(n - 1, 0, -1):
            self.grid(x + i * offset, y + i * offset, rows, cols, cell,
                      shaded=allc if shade_backs else (),
                      zorder=zorder - 2 * i)
        self.grid(x, y, rows, cols, cell, shaded=shaded_front, zorder=zorder)
        tw = cols * cell + (n - 1) * offset
        th = rows * cell + (n - 1) * offset
        if title:
            self.ax.text(x + tw / 2, y + th + 10, title,
                         fontsize=GRID_FS, color=INK, family=SANS,
                         ha="center", va="bottom", zorder=zorder + 1)
        return tw, th

    def op(self, x, y, symbol, fs=17.0):
        """Family operator (*, =, +) between panels, centered at (x, y)."""
        self.ax.text(x, y, symbol, fontsize=fs, color=INK, family=SANS,
                     fontweight="bold", ha="center", va="center", zorder=8)


# --------------------------------------------------------------------------- #
# Spine layout helper: evenly spaced node y-positions.                        #
# --------------------------------------------------------------------------- #

def spread(y_bottom, y_top, n):
    """n evenly spaced y positions from bottom to top (inclusive)."""
    if n == 1:
        return [0.5 * (y_bottom + y_top)]
    step = (y_top - y_bottom) / (n - 1)
    return [y_bottom + i * step for i in range(n)]
