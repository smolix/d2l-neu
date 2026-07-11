#!/usr/bin/env python3
"""Extra primitives for the Builders' Guide (chapter 6) *systems* diagrams,
drawn in the same gallery style as the network-architecture figures
(``tools/arch_diagrams.py`` / docs/convnet-rewrite/figure-style.md).

Chapter 6's figures are not all pill-on-a-spine dataflow: some are timelines
(async queue, activation checkpointing), byte strips (safetensors, the memory
ledger), nested containers (the caching allocator), bit-field bars (float
formats) or two-panel schematics (residual stream, the STE).  They all share
the gallery *visual grammar* though: grayscale everything, ONE blue accent used
sparingly, at most one near-black "novelty" box, white rounded pills with thin
black borders, Source Sans 3 / Inconsolata type, dotted-leader callouts.

This module builds on ``arch_diagrams`` and adds the few primitives that grammar
needs beyond the vertical-spine set: horizontal arrows, two-line boxes, plain
rounded rectangles that report their anchors, timeline dots, and byte-bar
segments.  Byte-idempotency and font embedding come for free from
``arch_diagrams`` (same ``save``, same pinned rcParams).
"""

from __future__ import annotations

from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from arch_diagrams import (  # noqa: F401  (re-exported for the generators)
    Diagram, MechDiagram, save, spread, text_width,
    INK, STROKE, ACCENT, ACCENT_TINT, ACCENT2, ACCENT2_TINT, NOVELTY_FILL,
    CONTAINER_FILL, INSET_FILL, GRAY_TEXT, SANS, MONO,
    PILL_H, PILL_GAP, PILL_STEP, PILL_FS, CALLOUT_FS, NOTE_FS, ANCHOR_FS,
    PILL_PAD_X, PILL_ROUND, PLUS_R, LEADER,
)

# A light neutral fill for "secondary" chart segments (kept distinct from the
# accent tint and from pure white); the gallery's container gray.
NEUTRAL_FILL = CONTAINER_FILL          # "#E4E4E4"
DARK_FILL = "#4A4A4A"                   # solid dark segment (byte bars)

_ARROW = "-|>,head_width=0.32,head_length=0.55"


# --------------------------------------------------------------------------- #
# Arrows and connectors (the spine ``arrow`` in arch_diagrams is vertical      #
# only; systems diagrams also flow left->right and branch sideways).          #
# --------------------------------------------------------------------------- #

def harrow(d, x0, x1, y, color=INK, dashed=False, lw=STROKE, zorder=4):
    """Horizontal arrow from x0 to x1 at height y (head at x1)."""
    d.ax.annotate("", xy=(x1, y), xytext=(x0, y),
                  arrowprops=dict(arrowstyle=_ARROW, color=color, lw=lw,
                                  shrinkA=0, shrinkB=0, mutation_scale=11,
                                  linestyle="--" if dashed else "-"),
                  zorder=zorder)


def arrow_to(d, p0, p1, color=INK, lw=STROKE, zorder=4):
    """Straight arrow between two arbitrary points (head at p1)."""
    d.ax.annotate("", xy=p1, xytext=p0,
                  arrowprops=dict(arrowstyle=_ARROW, color=color, lw=lw,
                                  shrinkA=0, shrinkB=0, mutation_scale=11),
                  zorder=zorder)


def connector(d, pts, color=INK, dashed=False, lw=STROKE, zorder=3):
    """Rectilinear polyline (no head).  Pass axis-aligned segments."""
    xs, ys = zip(*pts)
    d.ax.plot(xs, ys, color=color, lw=lw,
              linestyle=(0, (1.4, 2.2)) if dashed else "-",
              solid_joinstyle="miter", solid_capstyle="round", zorder=zorder)


# --------------------------------------------------------------------------- #
# Text.                                                                        #
# --------------------------------------------------------------------------- #

def text(d, x, y, s, fs=PILL_FS, color=INK, ha="center", va="center",
         bold=False, italic=False, mono=False, zorder=10):
    """A single styled text run in the gallery type family."""
    d.ax.text(x, y, s, fontsize=fs, color=color,
              family=MONO if mono else SANS,
              fontweight="bold" if bold else "regular",
              style="italic" if italic else "normal",
              ha=ha, va=va, zorder=zorder)


# --------------------------------------------------------------------------- #
# Boxes.  ``pill``/``novelty`` in arch_diagrams are single-line and self-size; #
# systems diagrams also need fixed-size boxes and a bold-title + sub-line box. #
# --------------------------------------------------------------------------- #

def _anchors(cx, cy, w, h):
    return dict(cx=cx, cy=cy, w=w, h=h, l=cx - w / 2, r=cx + w / 2,
                t=cy + h / 2, b=cy - h / 2,
                top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                left=(cx - w / 2, cy), right=(cx + w / 2, cy))


def rrect(d, cx, cy, w, h, fill="white", edge=INK, dashed=False,
          lw=STROKE, round_=None, zorder=5):
    """A rounded rectangle centered at (cx, cy).  Returns anchor points."""
    if round_ is None:
        round_ = min(0.42 * h, 12.0)
    style = dict(facecolor=fill, edgecolor=edge, linewidth=lw)
    if dashed:
        style["linestyle"] = (0, (2.4, 2.2))
    d.ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={round_}", zorder=zorder, **style))
    return _anchors(cx, cy, w, h)


def box(d, cx, cy, w, h, top, sub=None, fill="white", edge=INK,
        tint=False, novelty=False, keyword=None, dashed=False,
        top_fs=PILL_FS, sub_fs=NOTE_FS, top_color=None, sub_color=GRAY_TEXT,
        top_mono=False, sub_mono=False, round_=None, zorder=5):
    """A rounded box with a bold title and an optional smaller second line.

    ``tint`` fills with the accent tint (the repeated / injected unit);
    ``novelty`` fills near-black with white text and ``keyword`` in accent
    (the one new op).  Returns anchor points."""
    if novelty:
        fill, edge, top_color = NOVELTY_FILL, NOVELTY_FILL, "white"
        sub_color = "white"
    elif tint:
        fill = ACCENT_TINT
    if top_color is None:
        top_color = INK
    a = rrect(d, cx, cy, w, h, fill=fill, edge=edge, dashed=dashed,
              round_=round_, zorder=zorder)
    if sub:
        y_top = cy + h * 0.16
        y_sub = cy - h * 0.24
    else:
        y_top = cy
    if novelty and keyword:
        # split the title around the accent keyword
        pre, _, post = top.partition(keyword)
        segs = [(pre, "white", True), (keyword, ACCENT, True), (post, "white", True)]
        segs = [s for s in segs if s[0]]
        d.rich(cx, y_top, segs, top_fs, zorder=zorder + 5)
    else:
        text(d, cx, y_top, top, fs=top_fs, color=top_color, bold=True,
             mono=top_mono, zorder=zorder + 5)
    if sub:
        text(d, cx, y_sub, sub, fs=sub_fs, color=sub_color, mono=sub_mono,
             zorder=zorder + 5)
    return a


# --------------------------------------------------------------------------- #
# ⊕ / branch dot / timeline dot.                                              #
# --------------------------------------------------------------------------- #

def branch_dot(d, x, y, r=3.4, zorder=6):
    """A small solid black branch point (a path splits here)."""
    d.ax.add_patch(Circle((x, y), r, facecolor=INK, edgecolor=INK, zorder=zorder))


def state_dot(d, x, y, filled=True, r=6.5, zorder=6):
    """A stored-state marker: accent-filled if kept, hollow if recomputed."""
    if filled:
        d.ax.add_patch(Circle((x, y), r, facecolor=ACCENT, edgecolor=INK,
                              linewidth=1.0, zorder=zorder))
    else:
        d.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=GRAY_TEXT,
                              linewidth=1.3, zorder=zorder))


# --------------------------------------------------------------------------- #
# Byte / bit bar segments (the ledger, float formats, safetensors strip).     #
# --------------------------------------------------------------------------- #

def bar_seg(d, x0, y0, w, h, fill, edge="white", lw=1.4, zorder=3):
    """A rectangular bar segment (sharp corners, abutting neighbours)."""
    d.ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fill, edgecolor=edge,
                             linewidth=lw, zorder=zorder))
    return _anchors(x0 + w / 2, y0 + h / 2, w, h)
