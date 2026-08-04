"""Text measurement and outlining against the repo's bundled fonts.

The figure toolkit sizes boxes, pads canvases, and places labels from *real*
font metrics rather than guesses, using the same Source Sans 3 / JetBrains
Mono files the site serves (static/fonts/).  Two services:

* :func:`measure` — exact advance width + vertical metrics for a string at a
  given size (kerning via GPOS pair positioning is deliberately ignored; for
  UI-length strings the error is < 1%, and we always pad boxes anyway).
* :class:`TextOutliner` — converts a string to a single SVG ``<path>`` using
  the font's actual glyph outlines.  Outlined text renders identically in
  every browser, in librsvg (the PDF pipeline's rasterizer), and in print,
  with no webfont or @font-face dependency.

Both are pure fontTools; no matplotlib required.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT_DIR = os.path.join(REPO_ROOT, "static", "fonts")

# Logical name -> bundled file.  These are the fonts the site itself uses,
# so figures typeset in them look native on the page.
FONT_FILES = {
    ("sans", "regular"): "SourceSans3-Regular.ttf",
    ("sans", "bold"): "SourceSans3-Bold.ttf",
    ("sans", "italic"): "SourceSans3-Italic.ttf",
    ("serif", "regular"): "SourceSerif4-Regular.ttf",
    ("serif", "italic"): "SourceSerif4-Italic.ttf",
    ("serif", "bold"): "SourceSerif4-Bold.ttf",
    ("mono", "regular"): "Inconsolata-Regular.ttf",
    ("mono", "bold"): "Inconsolata-Bold.ttf",
}


@dataclass(frozen=True)
class TextExtent:
    """Extent of a single-line string at a given font size (all in px)."""

    width: float
    ascent: float      # baseline -> top of tallest possible glyph (hhea)
    descent: float     # baseline -> bottom (positive number)
    cap_height: float  # baseline -> top of capitals ("H")
    x_height: float    # baseline -> top of lowercase ("x")

    @property
    def height(self) -> float:
        return self.ascent + self.descent


@functools.lru_cache(maxsize=None)
def _font(family: str = "sans", style: str = "regular") -> TTFont:
    try:
        fname = FONT_FILES[(family, style)]
    except KeyError:
        raise ValueError(f"no bundled font for ({family!r}, {style!r})") from None
    return TTFont(os.path.join(FONT_DIR, fname))


@functools.lru_cache(maxsize=None)
def _metrics(family: str, style: str):
    f = _font(family, style)
    upm = f["head"].unitsPerEm
    os2 = f["OS/2"]
    return {
        "upm": upm,
        "cmap": f.getBestCmap(),
        "hmtx": f["hmtx"],
        "ascent": f["hhea"].ascender / upm,
        "descent": -f["hhea"].descender / upm,
        "cap_height": getattr(os2, "sCapHeight", int(0.66 * upm)) / upm,
        "x_height": getattr(os2, "sxHeight", int(0.48 * upm)) / upm,
    }


def measure(text: str, size: float, family: str = "sans", style: str = "regular") -> TextExtent:
    """Measure a single-line string at ``size`` px."""
    m = _metrics(family, style)
    cmap, hmtx, upm = m["cmap"], m["hmtx"], m["upm"]
    units = 0
    for ch in text:
        gname = cmap.get(ord(ch))
        if gname is None:
            gname = cmap.get(ord(" "), ".notdef")
        units += hmtx[gname][0]
    return TextExtent(
        width=units / upm * size,
        ascent=m["ascent"] * size,
        descent=m["descent"] * size,
        cap_height=m["cap_height"] * size,
        x_height=m["x_height"] * size,
    )


class TextOutliner:
    """Convert strings to SVG path data using real glyph outlines.

    >>> out = TextOutliner()
    >>> d = out.path_data("norm", size=15, x=10, y=40)   # y = baseline
    >>> f'<path d="{d}" fill="#3A4049"/>'  # doctest: +SKIP
    """

    def __init__(self, family: str = "sans", style: str = "regular"):
        self.family = family
        self.style = style
        f = _font(family, style)
        self._glyphset = f.getGlyphSet()
        self._m = _metrics(family, style)

    def path_data(self, text: str, size: float, x: float = 0.0, y: float = 0.0) -> str:
        """SVG path ``d`` for ``text`` with baseline at ``(x, y)`` (y-down)."""
        m = self._m
        scale = size / m["upm"]
        cmap, hmtx = m["cmap"], m["hmtx"]
        parts: list[str] = []
        pen_x = x
        for ch in text:
            gname = cmap.get(ord(ch))
            if gname is None:
                gname = cmap.get(ord(" "), ".notdef")
            glyph = self._glyphset[gname]
            spen = SVGPathPen(self._glyphset, ntos=lambda v: f"{v:.2f}")
            # Font units are y-up; SVG is y-down: flip y and drop the baseline at y.
            tpen = TransformPen(spen, (scale, 0, 0, -scale, pen_x, y))
            glyph.draw(tpen)
            d = spen.getCommands()
            if d:
                parts.append(d)
            pen_x += hmtx[gname][0] * scale
        return " ".join(parts)
