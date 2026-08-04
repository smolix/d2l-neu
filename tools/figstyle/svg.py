"""Minimal deterministic SVG composer for d2l diagram figures.

Why not matplotlib for block diagrams: matplotlib excels at data-driven
mathematical figures (keep using it there, via figstyle.mpl), but block/flow
diagrams built from FancyBboxPatch fight the Axes model — canvas cropping,
text measurement, and arrow joins all need workarounds.  This composer draws
exactly what you say in figure coordinates (y-down, like SVG), measures text
with the real bundled fonts (figstyle.textmetrics), and crops the canvas to
content automatically.

House rules baked in (docs/figure-style-guide.md):
  * text is OUTLINED to paths by default — identical rendering in browsers,
    librsvg (the PDF pipeline), and print, with no font installation anywhere
  * deterministic output: fixed number formatting, no ids, no timestamps —
    re-running a generator yields byte-identical SVGs (clean git diffs)
  * flat fills, no gradients, no shadows
  * data-flow arrows get small filled triangular heads; annotation leaders
    are dotted with light chevron heads (figstyle.tokens)

Coordinates are SVG user units == CSS px at 1:1 display.  Draw with y
increasing DOWNWARD (SVG convention).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from . import tokens as T
from .textmetrics import TextOutliner, measure

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(REPO_ROOT, "img")


def _f(v: float) -> str:
    """Fixed, locale-free float formatting (deterministic output)."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


@dataclass
class _BBox:
    x0: float = math.inf
    y0: float = math.inf
    x1: float = -math.inf
    y1: float = -math.inf

    def add(self, x0, y0, x1, y1, pad=0.0):
        self.x0 = min(self.x0, x0 - pad)
        self.y0 = min(self.y0, y0 - pad)
        self.x1 = max(self.x1, x1 + pad)
        self.y1 = max(self.y1, y1 + pad)


# Style variants for rich text runs.
_OUTLINERS: dict[tuple[str, str], TextOutliner] = {}


def _outliner(family: str, style: str) -> TextOutliner:
    key = (family, style)
    if key not in _OUTLINERS:
        _OUTLINERS[key] = TextOutliner(family, style)
    return _OUTLINERS[key]


@dataclass
class Span:
    """One run of styled text inside a label.

    kind: 'r' regular, 'b' bold, 'i' italic (math-ish variables), 'm' mono.
    script: 0 normal, -1 subscript, +1 superscript.
    color: overrides the label color for this run (e.g. an accent number).
    """

    text: str
    kind: str = "r"
    script: int = 0
    color: str | None = None


def sub(text: str, kind: str = "i") -> Span:
    return Span(text, kind, script=-1)


def var(text: str) -> Span:
    """A math-ish variable, set in italic (Distill-style sans math)."""
    return Span(text, "i")


_KIND2FONT = {
    "r": ("sans", "regular"),
    "b": ("sans", "bold"),
    "i": ("sans", "italic"),
    "m": ("mono", "regular"),
}
_SCRIPT_SCALE = 0.72
_SUB_SHIFT = 0.18    # fraction of font size, downward
_SUP_SHIFT = -0.42


class Figure:
    """An SVG scene with automatic tight cropping.

    >>> f = Figure()
    >>> f.block(0, 0, "multi-head attention", role="attention")
    >>> f.save("demo")            # -> img/demo.svg   (doctest: +SKIP)
    """

    def __init__(self, pad: float = T.PAD_CANVAS):
        self._parts: list[str] = []
        self._bb = _BBox()
        self.pad = pad

    # ------------------------------------------------------------- low level
    def raw(self, element: str, x0, y0, x1, y1, pad: float = 0.0):
        """Append a raw SVG element, declaring its extent."""
        self._parts.append(element)
        self._bb.add(x0, y0, x1, y1, pad=pad)

    def rect(self, x, y, w, h, *, fill=T.PAPER, stroke=T.INK, sw=T.SW_BOX,
             r=0.0, dash: str | None = None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        rr = f' rx="{_f(r)}"' if r else ""
        s = (f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}"'
             f'{rr} fill="{fill}" stroke="{stroke or "none"}"'
             f' stroke-width="{_f(sw)}"{d}/>')
        self.raw(s, x, y, x + w, y + h, )
        return (x, y, w, h)

    def circle(self, cx, cy, r, *, fill=T.PAPER, stroke=T.INK, sw=T.SW_BOX):
        s = (f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="{fill}"'
             f' stroke="{stroke or "none"}" stroke-width="{_f(sw)}"/>')
        self.raw(s, cx - r, cy - r, cx + r, cy + r)

    def line(self, x1, y1, x2, y2, *, stroke=T.INK, sw=T.SW_LINE,
             dash: str | None = None, cap="butt"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        c = f' stroke-linecap="{cap}"' if cap != "butt" else ""
        s = (f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}"'
             f' stroke="{stroke}" stroke-width="{_f(sw)}"{d}{c}/>')
        self.raw(s, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), pad=sw)

    def polyline(self, pts, *, stroke=T.INK, sw=T.SW_LINE,
                 dash: str | None = None):
        p = " ".join(f"{_f(x)},{_f(y)}" for x, y in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s = (f'<polyline points="{p}" fill="none" stroke="{stroke}"'
             f' stroke-width="{_f(sw)}" stroke-linejoin="round"{d}/>')
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        self.raw(s, min(xs), min(ys), max(xs), max(ys), pad=sw)

    def path(self, d: str, x0, y0, x1, y1, *, fill="none", stroke=T.INK,
             sw=T.SW_LINE, dash: str | None = None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        s = (f'<path d="{d}" fill="{fill}" stroke="{stroke or "none"}"'
             f' stroke-width="{_f(sw)}"{dd}/>')
        self.raw(s, x0, y0, x1, y1, pad=sw)

    # ----------------------------------------------------------------- text
    def text(self, x, y, content, *, size=T.FS_LABEL, color=T.INK,
             anchor="middle", va="middle", tracking=0.0):
        """Typeset a label (string or list of Spans), outlined to paths.

        anchor: 'start' | 'middle' | 'end'   (horizontal alignment at x)
        va:     'baseline' | 'middle' | 'top' (vertical alignment at y)
        Returns the text bbox (x0, y0, x1, y1).
        """
        spans = [Span(content)] if isinstance(content, str) else list(content)

        widths = []
        for sp in spans:
            fam, sty = _KIND2FONT[sp.kind]
            sz = size * (_SCRIPT_SCALE if sp.script else 1.0)
            w = measure(sp.text, sz, fam, sty).width
            w += tracking * sz * max(0, len(sp.text) - 1)
            widths.append(w)
        total = sum(widths)

        ext = measure("Hg", size)  # vertical metrics of the base size
        if anchor == "middle":
            cx = x - total / 2
        elif anchor == "end":
            cx = x - total
        else:
            cx = x
        if va == "middle":
            base = y + ext.cap_height / 2
        elif va == "top":
            base = y + ext.cap_height
        else:
            base = y

        for sp, w in zip(spans, widths):
            fam, sty = _KIND2FONT[sp.kind]
            sz = size * (_SCRIPT_SCALE if sp.script else 1.0)
            by = base + (_SUB_SHIFT if sp.script < 0 else
                         _SUP_SHIFT if sp.script > 0 else 0.0) * size
            if tracking:
                px = cx
                parts = []
                for ch in sp.text:
                    parts.append(_outliner(fam, sty).path_data(ch, sz, px, by))
                    px += measure(ch, sz, fam, sty).width + tracking * sz
                d = " ".join(p for p in parts if p)
            else:
                d = _outliner(fam, sty).path_data(sp.text, sz, cx, by)
            if d:
                self._parts.append(
                    f'<path d="{d}" fill="{sp.color or color}"/>')
            cx += w

        x0 = {"middle": x - total / 2, "end": x - total}.get(anchor, x)
        y0, y1 = base - ext.ascent, base + ext.descent
        self._bb.add(x0, y0, x0 + total, y1)
        return (x0, y0, x0 + total, y1)

    # --------------------------------------------------------------- arrows
    def _head(self, x, y, angle, *, color, length=T.ARROW_L, width=T.ARROW_W):
        """Filled triangular data-flow arrowhead with tip at (x, y)."""
        ca, sa = math.cos(angle), math.sin(angle)
        bx, by = x - length * ca, y - length * sa
        px, py = -sa * width / 2, ca * width / 2
        pts = f"{_f(x)},{_f(y)} {_f(bx + px)},{_f(by + py)} {_f(bx - px)},{_f(by - py)}"
        self.raw(f'<polygon points="{pts}" fill="{color}"/>',
                 min(x, bx) - width, min(y, by) - width,
                 max(x, bx) + width, max(y, by) + width)

    def arrow(self, x1, y1, x2, y2, *, stroke=T.INK, sw=T.SW_LINE,
              dash: str | None = None):
        """Straight data-flow arrow; shaft stops at the head base."""
        ang = math.atan2(y2 - y1, x2 - x1)
        hl = T.ARROW_L * (sw / T.SW_LINE) ** 0.5
        hw = T.ARROW_W * (sw / T.SW_LINE) ** 0.5
        bx, by = x2 - 0.9 * hl * math.cos(ang), y2 - 0.9 * hl * math.sin(ang)
        self.line(x1, y1, bx, by, stroke=stroke, sw=sw, dash=dash)
        self._head(x2, y2, ang, color=stroke, length=hl, width=hw)

    def ortho_arrow(self, pts, *, stroke=T.INK, sw=T.SW_LINE,
                    dash: str | None = None):
        """Rectilinear data-flow arrow through ``pts`` (>= 2 points)."""
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        ang = math.atan2(yb - ya, xb - xa)
        hl = T.ARROW_L * (sw / T.SW_LINE) ** 0.5
        hw = T.ARROW_W * (sw / T.SW_LINE) ** 0.5
        shaft = list(pts[:-1]) + [(xb - 0.9 * hl * math.cos(ang),
                                   yb - 0.9 * hl * math.sin(ang))]
        self.polyline(shaft, stroke=stroke, sw=sw, dash=dash)
        self._head(xb, yb, ang, color=stroke, length=hl, width=hw)

    def leader(self, x1, y1, x2, y2, *, stroke=T.MUTED, sw=T.SW_HAIR):
        """Dotted annotation leader with a light open chevron at the far end."""
        self.line(x1, y1, x2, y2, stroke=stroke, sw=sw, dash=T.DOT_LEADER,
                  cap="round")
        ang = math.atan2(y2 - y1, x2 - x1)
        L = T.CHEVRON_L
        for da in (math.radians(152), math.radians(-152)):
            self.line(x2, y2,
                      x2 + L * math.cos(ang + da), y2 + L * math.sin(ang + da),
                      stroke=stroke, sw=sw, cap="round")

    # ------------------------------------------------------------ mid level
    def block(self, cx, cy, label, *, role: str | None = None,
              accent: T.Accent | None = None, size=T.FS_LABEL,
              min_w: float = 0.0, h: float | None = None, dash=None,
              mono=False, novelty=False):
        """Auto-sized rounded block centered at (cx, cy); returns (x,y,w,h).

        role/accent pick the tint fill + base outline; default is a white
        block with ink outline (the neutral op).  novelty=True draws the
        approved near-black novelty box with white text.
        """
        a = accent or (T.ROLE[role] if role else None)
        spans = [Span(label, "m" if mono else "r")] if isinstance(label, str) else label
        widths = []
        for sp in spans:
            fam, sty = _KIND2FONT[sp.kind]
            sz = size * (_SCRIPT_SCALE if sp.script else 1.0)
            widths.append(measure(sp.text, sz, fam, sty).width)
        tw = sum(widths)
        w = max(min_w, tw + 2 * T.PAD_BLOCK_X)
        bh = h or (measure("Hg", size).height + 2 * T.PAD_BLOCK_Y)
        x, y = cx - w / 2, cy - bh / 2
        if novelty:
            fill, edge, txt = T.NOVELTY_FILL, "none", T.PAPER
        elif a:
            fill, edge, txt = a.tint, a.base, T.INK
        else:
            fill, edge, txt = T.PAPER, T.INK, T.INK
        self.rect(x, y, w, bh, fill=fill, stroke=edge, sw=T.SW_BOX,
                  r=T.R_BLOCK, dash=dash)
        self.text(cx, cy, spans, size=size, color=txt)
        return (x, y, w, bh)

    def pill_op(self, cx, cy, glyph="+", *, r=12.0, stroke=T.INK):
        """Elementwise-op circle on a spine: ⊕ (add) or ⊗ (product)."""
        self.circle(cx, cy, r, fill=T.PAPER, stroke=stroke, sw=T.SW_BOX)
        if glyph == "+":
            self.line(cx - r * 0.52, cy, cx + r * 0.52, cy, stroke=stroke, sw=T.SW_BOX)
            self.line(cx, cy - r * 0.52, cx, cy + r * 0.52, stroke=stroke, sw=T.SW_BOX)
        elif glyph == "x":
            d = r * 0.38
            self.line(cx - d, cy - d, cx + d, cy + d, stroke=stroke, sw=T.SW_BOX)
            self.line(cx - d, cy + d, cx + d, cy - d, stroke=stroke, sw=T.SW_BOX)
        return r

    def panel(self, x, y, w, h, *, label: str | None = None, fill=T.PANEL,
              stroke="none", dash=None, label_color=T.MUTED):
        """Grouping panel; optional letterspaced ALL-CAPS label at top-left."""
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=T.SW_HAIR,
                  r=T.R_PANEL, dash=dash)
        if label:
            self.text(x + T.PAD_PANEL, y + T.PAD_PANEL, label.upper(),
                      size=T.FS_TINY, color=label_color, anchor="start",
                      va="middle", tracking=T.LETTERSPACE_CAPS)

    # ----------------------------------------------------------------- save
    def render(self, *, desc: str | None = None) -> str:
        bb = self._bb
        x0, y0 = bb.x0 - self.pad, bb.y0 - self.pad
        w, h = (bb.x1 - bb.x0) + 2 * self.pad, (bb.y1 - bb.y0) + 2 * self.pad
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(w)}"'
                f' height="{_f(h)}" viewBox="{_f(x0)} {_f(y0)} {_f(w)} {_f(h)}">')
        dsc = f"<desc>{desc}</desc>" if desc else ""
        return head + dsc + "".join(self._parts) + "</svg>"

    def save(self, name: str, *, desc: str | None = None,
             out_dir: str = IMG_DIR) -> str:
        svg = self.render(desc=desc)
        assert "<svg" in svg and len(svg) > 100, "empty figure?"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.svg")
        with open(path, "w") as fh:
            fh.write(svg)
        return path
