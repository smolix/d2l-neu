#!/usr/bin/env python3
"""Outline the live ``<text>`` of standalone SVGs to ``<path>`` elements.

The JS diagram pipeline (diagrams/*.mjs -> node diagrams/render.mjs) emits
live ``<text>`` in 'Source Sans 3' / 'JetBrains Mono' plus an ``@import`` of
Google Fonts.  Neither the PDF machine (rsvg-convert falls back to whatever
fontconfig finds) nor an offline viewer has those fonts, so the committed
figures must not depend on them (docs/figure-style-guide.md §5.1: text in
committed SVGs is outlined to paths).  This post-processor rewrites every
``<text>`` into glyph outlines from the repo's bundled fonts
(figstyle.textmetrics.TextOutliner over static/fonts/):

* sans families (Source Sans 3, system-ui, Arial, sans-serif, ...)
  -> Source Sans 3; monospace families (JetBrains Mono, Inconsolata,
  monospace, ...) -> Inconsolata; serif families -> Source Serif 4.
  JetBrains Mono, which the engine names but the repo does not bundle, is
  drawn in Inconsolata scaled to JetBrains Mono's cap height (x1.17, the
  CSS font-size-adjust idea), so mono labels keep their designed size.
* font-weight >= 600 -> Bold, else Regular (CSS matching against the
  bundled 400/700 faces); font-style italic/oblique -> the Italic face;
  bold italic -> the Italic face emboldened by a 0.05 em stroke in its own
  colour (synthetic bold, as browsers do).
* A character missing from the chosen face falls back, in order, to the
  other bundled families and then to matplotlib's pinned DejaVu / STIX
  fonts (matplotlib==3.10.8 in pyproject.toml, so the outlines are stable).
  A character no font covers is an error, never a silent blank.
* Layout follows the SVG text model: ``<tspan>`` nesting with inherited
  presentation attributes and ``style=""`` declarations, x/y/dx/dy (lists
  included; an absolute x or y starts a new text chunk), text-anchor
  start/middle/end per chunk, baseline-shift (sub, super, lengths,
  percentages), font-size changes (px, pt, em, %), letter-spacing,
  xml:space (default whitespace collapsing or preserve), per-run fill /
  fill-opacity / stroke paint, and the ``<text>``'s opacity, transform,
  id and class (moved to the wrapping ``<g>``).
* dominant-baseline is honoured even though librsvg ignores it: central
  centres capitals on y (half the face's cap height, the same rule as
  diagrams/engine.mjs baselineY), middle half the x-height, hanging the cap
  height, text-before-edge / text-after-edge the ascent / descent.
* Accessibility: each outlined label becomes ``<g aria-label="...">`` with
  the original string (sub/superscript runs marked ``_`` / ``^``), so the
  figure is not mute when inlined (slides).
* Once no ``<text>`` remains, an ``@import``-only font ``<style>`` block is
  removed.

Kerning and ligatures are not applied (as in figstyle.textmetrics; < 1% of
width for label-length strings).  Unsupported text features (textPath,
per-glyph rotate, textLength, vertical writing-mode) raise an error.

Deterministic and byte-idempotent: everything outside the ``<text>``
elements is left byte-for-byte unchanged, coordinates are printed with a
fixed precision, no ids are generated, and a file without ``<text>`` (an
already outlined one, or a matplotlib figure) is left untouched.

Usage:  python tools/figstyle/outline_svg.py FILE.svg [FILE2.svg ...]
(run with .venv-pytorch/bin/python: it needs fontTools.)
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

_TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from figstyle import textmetrics as TM  # noqa: E402

XML_NS = "{http://www.w3.org/XML/1998/namespace}"

# --------------------------------------------------------------------------- #
# Fonts                                                                        #
# --------------------------------------------------------------------------- #

_SANS = {"source sans 3", "source sans pro", "system-ui", "-apple-system",
         "blinkmacsystemfont", "helvetica neue", "helvetica", "arial",
         "segoe ui", "noto sans", "dejavu sans", "liberation sans",
         "sans-serif", "ui-sans-serif"}
_MONO = {"jetbrains mono", "inconsolata", "menlo", "monaco", "consolas",
         "courier", "courier new", "dejavu sans mono", "sf mono",
         "liberation mono", "monospace", "ui-monospace"}
_SERIF = {"source serif 4", "source serif pro", "georgia", "times",
          "times new roman", "dejavu serif", "liberation serif", "serif",
          "ui-serif"}

# font-size-adjust: a requested family that is NOT bundled is drawn in the
# bundled face of its class scaled to the requested family's cap height, so
# a figure laid out for that family keeps its type size.  JetBrains Mono
# (the engine's FM; caps 730/1000) -> Inconsolata (caps 623/1000) at x1.17:
# digits as tall as designed, advance 0.586 em vs. 0.6 em designed.
_DESIGN_CAP_HEIGHT = {"jetbrains mono": 0.730}

# Bold italic is not bundled: the regular-weight italic is emboldened with a
# stroke in its own fill colour, as browsers synthesize it.  0.05 em total
# (Source Sans 3 stems: Regular 0.083 em, Bold 0.147 em).
FAUX_BOLD_EM = 0.05

# matplotlib's bundled fallback faces, per (class, style).
_MPL_FALLBACK = {
    ("sans", "regular"): "DejaVuSans.ttf",
    ("sans", "bold"): "DejaVuSans-Bold.ttf",
    ("sans", "italic"): "DejaVuSans-Oblique.ttf",
    ("mono", "regular"): "DejaVuSansMono.ttf",
    ("mono", "bold"): "DejaVuSansMono-Bold.ttf",
    ("mono", "italic"): "DejaVuSansMono-Oblique.ttf",
    ("serif", "regular"): "DejaVuSerif.ttf",
    ("serif", "bold"): "DejaVuSerif-Bold.ttf",
    ("serif", "italic"): "DejaVuSerif-Italic.ttf",
    ("stix", "regular"): "STIXGeneral.ttf",
    ("stix", "bold"): "STIXGeneralBol.ttf",
    ("stix", "italic"): "STIXGeneralItalic.ttf",
}


def _mpl_font_dir() -> str | None:
    spec = importlib.util.find_spec("matplotlib")
    if spec is None or not spec.submodule_search_locations:
        return None
    d = os.path.join(list(spec.submodule_search_locations)[0],
                     "mpl-data", "fonts", "ttf")
    return d if os.path.isdir(d) else None


_FALLBACKS_REGISTERED = False


def _register_fallbacks() -> None:
    """Expose matplotlib's DejaVu/STIX faces to textmetrics as
    ('mpl-<class>', style) -- FONT_FILES joins its values onto FONT_DIR, and
    an absolute path survives the join."""
    global _FALLBACKS_REGISTERED
    if _FALLBACKS_REGISTERED:
        return
    _FALLBACKS_REGISTERED = True
    d = _mpl_font_dir()
    if d is None:
        return
    for (cls, sty), fname in _MPL_FALLBACK.items():
        path = os.path.join(d, fname)
        if os.path.exists(path):
            TM.FONT_FILES.setdefault((f"mpl-{cls}", sty), path)


_CMAPS: dict[tuple[str, str], dict] = {}
_OUTLINERS: dict[tuple[str, str], TM.TextOutliner] = {}


def _cmap(face):
    if face not in _CMAPS:
        _CMAPS[face] = TM._font(*face).getBestCmap()
    return _CMAPS[face]


def _outliner(face) -> TM.TextOutliner:
    if face not in _OUTLINERS:
        _OUTLINERS[face] = TM.TextOutliner(*face)
    return _OUTLINERS[face]


def _font_class(family_list: str, warn) -> tuple[str, float]:
    """(bundled class, font-size scale) for a CSS font-family list."""
    for name in _split_families(family_list):
        n = name.lower()
        cls = ("mono" if n in _MONO else "sans" if n in _SANS
               else "serif" if n in _SERIF else None)
        if cls is None:
            continue
        scale = 1.0
        if n in _DESIGN_CAP_HEIGHT:
            scale = _DESIGN_CAP_HEIGHT[n] / TM.measure("", 1.0, cls).cap_height
        return cls, scale
    warn(f"unrecognised font-family {family_list!r}; using the bundled sans")
    return "sans", 1.0


def _split_families(s: str) -> list[str]:
    out, cur, q = [], "", None
    for ch in s:
        if q:
            if ch == q:
                q = None
            else:
                cur += ch
        elif ch in "'\"":
            q = ch
        elif ch == ",":
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _style_name(weight: int, italic: bool) -> str:
    if italic:
        return "italic"
    return "bold" if weight >= 600 else "regular"


def _face_chain(cls: str, weight: int, italic: bool, warn) -> list[tuple[str, str]]:
    """Primary bundled face for (class, weight, italic), then fallbacks."""
    sty = _style_name(weight, italic)
    if cls == "mono" and italic:
        warn("no bundled italic monospace face; using upright Inconsolata")
        sty = "bold" if weight >= 600 else "regular"
    chain = [(cls, sty)]
    for other in ("sans", "serif", "mono"):
        if other != cls:
            s2 = sty if (other, sty) in TM.FONT_FILES else "regular"
            chain.append((other, s2))
    _register_fallbacks()
    for other in dict.fromkeys((cls, "sans", "mono", "stix")):
        for s2 in (sty, "regular"):
            if (f"mpl-{other}", s2) in TM.FONT_FILES:
                chain.append((f"mpl-{other}", s2))
                break
    return chain


def _pick_face(ch: str, chain):
    for face in chain:
        if ord(ch) in _cmap(face):
            return face
    return None


# --------------------------------------------------------------------------- #
# Style resolution                                                             #
# --------------------------------------------------------------------------- #

_INHERITED = ("font-family", "font-size", "font-weight", "font-style",
              "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width",
              "stroke-opacity", "stroke-linejoin", "stroke-linecap",
              "paint-order", "text-anchor", "letter-spacing", "xml:space",
              "dominant-baseline", "visibility")
_PAINT = ("fill", "fill-opacity", "fill-rule", "stroke", "stroke-width",
          "stroke-opacity", "stroke-linejoin", "stroke-linecap", "paint-order")
# attributes whose effect we cannot reproduce: refuse rather than mis-draw
_FATAL = {"rotate", "textLength", "lengthAdjust", "writing-mode",
          "glyph-orientation-vertical"}
# attributes that are harmless to drop on outlining
_IGNORED = {"x", "y", "dx", "dy", "style", "class", "id", "transform",
            "opacity", "baseline-shift", "font-variant", "font-stretch",
            "text-rendering", "font-kerning", "font-feature-settings",
            "role", "aria-label", "aria-hidden", "word-spacing",
            "text-decoration", "direction", "unicode-bidi"}


def _attrs(el) -> dict[str, str]:
    d = {}
    for k, v in el.attrib.items():
        if k.startswith(XML_NS):
            k = "xml:" + k[len(XML_NS):]
        elif k.startswith("{"):
            continue
        d[k] = v.strip()
    st = d.pop("style", None)
    if st:
        for part in st.split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                d[k.strip().lower()] = v.strip()
    return d


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _length(v: str, fs: float, pct_of: float | None = None) -> float:
    m = re.fullmatch(rf"({_NUM})\s*(px|pt|em|ex|%)?", v.strip())
    if not m:
        raise ValueError(f"unsupported length {v!r}")
    x, unit = float(m.group(1)), m.group(2)
    if unit in (None, "px"):
        return x
    if unit == "pt":
        return x * 4.0 / 3.0
    if unit == "em":
        return x * fs
    if unit == "ex":
        return x * fs * 0.5
    return x / 100.0 * (fs if pct_of is None else pct_of)


def _numbers(v: str | None) -> list[float]:
    if not v:
        return []
    return [float(t) for t in re.findall(_NUM, v)]


def _weight(v: str, parent: int) -> int:
    v = v.lower()
    if v == "normal":
        return 400
    if v == "bold":
        return 700
    if v == "bolder":
        return 400 if parent < 350 else 700 if parent < 550 else 900
    if v == "lighter":
        return 100 if parent < 550 else 400 if parent < 750 else 700
    return int(float(v))


def _compute(el, parent: dict) -> dict:
    """Computed style of `el` from its own attributes and `parent`'s."""
    a = _attrs(el)
    c = dict(parent)
    for k in _INHERITED:
        if k in a and a[k] != "inherit":
            c[k] = a[k]
    pfs = parent["_fs"]
    c["_fs"] = _length(c["font-size"], pfs, pfs) if "font-size" in a else pfs
    c["_fw"] = _weight(a["font-weight"], parent["_fw"]) if "font-weight" in a else parent["_fw"]
    return c


_ROOT_STYLE = {"font-family": "sans-serif", "font-size": "16", "_fs": 16.0,
               "_fw": 400, "font-style": "normal", "fill": "#000000",
               "text-anchor": "start", "letter-spacing": "normal",
               "xml:space": "default", "dominant-baseline": "auto"}


# --------------------------------------------------------------------------- #
# Layout                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class _Ch:
    ch: str
    st: dict
    shift: float          # accumulated baseline-shift, y-down (px)
    x: float | None = None
    y: float | None = None
    dx: float = 0.0
    dy: float = 0.0
    face: tuple = ()       # face the glyph is drawn from
    primary: tuple = ()    # face the style asks for (sets the baseline)
    size: float = 0.0      # outlined font size (font-size x size adjust)
    faux_bold: bool = False
    adv: float = 0.0
    gx: float = 0.0       # glyph origin (baseline, before anchoring)
    gy: float = 0.0
    rel: float = 0.0      # vertical offset from the chunk baseline (for aria)


@dataclass
class _Ctx:
    path: str
    warnings: list = field(default_factory=list)

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _baseline_shift(v: str | None, fs: float, parent_fs: float) -> float:
    """baseline-shift as a y-DOWN offset.  sub/super follow librsvg and
    Inkscape (-0.2 em / +0.4 em of the element's own font size)."""
    if not v or v == "baseline":
        return 0.0
    if v == "sub":
        return 0.2 * fs
    if v == "super":
        return -0.4 * fs
    return -_length(v, fs, parent_fs)


def _collect(text_el, base_style, ctx):
    """Flatten <text> into addressable characters with per-element ranges."""
    chars: list[_Ch] = []
    ranges: list[tuple] = []      # (element attrs, start, end) in pre-order
    state = {"prev_space": True}

    def add(s: str, st: dict, shift: float):
        preserve = st.get("xml:space") == "preserve"
        for ch in s:
            if ch in "\n\r\t":
                ch = " "
            if not preserve and ch == " ":
                if state["prev_space"]:
                    continue
            state["prev_space"] = ch == " " and not preserve
            chars.append(_Ch(ch, st, shift))

    def walk(el, st, shift):
        a = _attrs(el)
        for k in a:
            if k in _FATAL:
                raise ValueError(f"<{_local(el.tag)} {k}=...> is not supported")
            if k not in _IGNORED and k not in _INHERITED:
                ctx.warn(f"ignored attribute {k!r} on <{_local(el.tag)}>")
        if "opacity" in a and el is not text_el:
            ctx.warn("opacity on <tspan> is ignored (not rendered by SVG 1.1)")
        idx = len(ranges)
        ranges.append((a, len(chars), None))
        if el.text:
            add(el.text, st, shift)
        for child in el:
            tag = _local(child.tag)
            if tag == "tspan":
                cst = _compute(child, st)
                cshift = shift + _baseline_shift(
                    _attrs(child).get("baseline-shift"), cst["_fs"], st["_fs"])
                walk(child, cst, cshift)
            elif tag in ("title", "desc", "metadata"):
                pass
            else:
                raise ValueError(f"<{tag}> inside <text> is not supported")
            if child.tail:
                add(child.tail, st, shift)
        ranges[idx] = (a, ranges[idx][1], len(chars))

    walk(text_el, base_style, _baseline_shift(
        _attrs(text_el).get("baseline-shift"), base_style["_fs"], base_style["_fs"]))
    # strip trailing collapsible spaces
    while chars and chars[-1].ch == " " and chars[-1].st.get("xml:space") != "preserve":
        chars.pop()
    n = len(chars)
    # position attributes: ancestors first, descendants override (pre-order)
    for a, s, e in ranges:
        e = min(e, n)
        for key in ("x", "y", "dx", "dy"):
            vals = _numbers(a.get(key))
            for i, v in enumerate(vals[: max(0, e - s)]):
                setattr(chars[s + i], key, v)
    return chars


def _db_shift(mode: str, face, fs: float) -> float:
    """dominant-baseline -> y-down shift of the alphabetic baseline."""
    if mode in ("auto", "alphabetic", "", None):
        return 0.0
    ext = TM.measure("", fs, *face)
    if mode == "central":
        return ext.cap_height / 2
    if mode == "middle":
        return ext.x_height / 2
    if mode == "hanging":
        return ext.cap_height
    if mode in ("text-before-edge", "text-top"):
        return ext.ascent
    if mode in ("text-after-edge", "text-bottom", "ideographic"):
        return -ext.descent
    if mode == "mathematical":
        return ext.ascent / 2
    raise ValueError(f"unsupported dominant-baseline {mode!r}")


def _layout(chars: list[_Ch], root_style: dict, ctx) -> None:
    db_mode = root_style.get("dominant-baseline", "auto")
    for c in chars:
        st = c.st
        cls, scale = _font_class(st["font-family"], ctx.warn)
        italic = st.get("font-style", "normal") in ("italic", "oblique")
        chain = _face_chain(cls, st["_fw"], italic, ctx.warn)
        c.size = st["_fs"] * scale
        c.faux_bold = italic and st["_fw"] >= 600 and chain[0][1] == "italic"
        face = _pick_face(c.ch, chain)
        if face is None:
            raise ValueError(f"no bundled or fallback font has {c.ch!r} "
                             f"(U+{ord(c.ch):04X})")
        if face != chain[0] and c.ch != " ":
            ctx.warn(f"{c.ch!r} (U+{ord(c.ch):04X}) not in {chain[0]}; "
                     f"drawn from {face}")
        c.face, c.primary = face, chain[0]
        ls = st.get("letter-spacing", "normal")
        spacing = 0.0 if ls == "normal" else _length(ls, st["_fs"])
        c.adv = TM.measure(c.ch, c.size, *face).width + spacing

    pen_x = pen_y = 0.0
    chunks: list[int] = []
    chunk_base = 0.0
    for i, c in enumerate(chars):
        new = i == 0
        if c.x is not None:
            pen_x, new = c.x, True
        if c.y is not None:
            pen_y, new = c.y, True
        if new:
            chunks.append(i)
            chunk_base = pen_y
        pen_x += c.dx
        pen_y += c.dy
        c.rel = pen_y + c.shift - chunk_base
        c.gx = pen_x
        c.gy = pen_y + c.shift + _db_shift(db_mode, c.primary, c.size)
        pen_x += c.adv

    # text-anchor per chunk (SVG 2 "apply anchoring")
    bounds = chunks + [len(chars)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if s == e:
            continue
        anchor = chars[s].st.get("text-anchor", "start")
        lo = min(min(c.gx, c.gx + c.adv) for c in chars[s:e])
        hi = max(max(c.gx, c.gx + c.adv) for c in chars[s:e])
        x0 = chars[s].gx
        shift = {"start": x0 - lo, "middle": x0 - (lo + hi) / 2,
                 "end": x0 - hi}.get(anchor)
        if shift is None:
            raise ValueError(f"unsupported text-anchor {anchor!r}")
        for c in chars[s:e]:
            c.gx += shift


# --------------------------------------------------------------------------- #
# Emission                                                                     #
# --------------------------------------------------------------------------- #

def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _aria(chars: list[_Ch], fs: float) -> str:
    out, level = [], 0
    for c in chars:
        lv = -1 if c.rel > 0.1 * fs else 1 if c.rel < -0.1 * fs else 0
        if lv != level and lv != 0:
            out.append("_" if lv < 0 else "^")
        level = lv
        out.append(c.ch)
    return "".join(out).strip()


def _outline_text(text_el, parents, ctx) -> str:
    chain = []
    p = parents.get(text_el)
    while p is not None:
        chain.append(p)
        p = parents.get(p)
    st = dict(_ROOT_STYLE)
    for anc in reversed(chain):
        st = _compute(anc, st)
    st = _compute(text_el, st)

    chars = _collect(text_el, st, ctx)
    a = _attrs(text_el)
    if st.get("visibility") in ("hidden", "collapse") or a.get("display") == "none":
        return ""
    _layout(chars, st, ctx)

    paths, cur_paint, cur_d = [], None, []

    def flush():
        if cur_d:
            paint = "".join(f' {k}="{_esc(v)}"' for k, v in cur_paint)
            paths.append(f'<path d="{" ".join(cur_d)}"{paint}/>')

    for c in chars:
        if c.ch == " ":
            continue
        d = _outliner(c.face).path_data(c.ch, c.size, c.gx, c.gy)
        if not d:
            continue
        paint = tuple((k, c.st[k]) for k in _PAINT if k in c.st)
        if c.faux_bold and "stroke" not in c.st:
            paint += (("stroke", c.st.get("fill", "#000000")),
                      ("stroke-width", f"{FAUX_BOLD_EM * c.size:.2f}"),
                      ("stroke-linejoin", "round"))
        if paint != cur_paint:
            flush()
            cur_paint, cur_d = paint, []
        cur_d.append(d)
    flush()

    g = [f'aria-label="{_esc(_aria(chars, st["_fs"]))}"']
    for k in ("id", "class", "transform", "opacity"):
        if k in a:
            g.append(f'{k}="{_esc(a[k])}"')
    return f'<g {" ".join(g)}>{"".join(paths)}</g>'


_TEXT_RE = re.compile(r"<text\b[^>]*?(?:/>|>.*?</text\s*>)", re.S)
_FONT_STYLE_RE = re.compile(
    r"<style\b[^>]*>\s*(?:<!\[CDATA\[)?\s*"
    r"(?:@import\s+url\(\s*['\"]?[^)'\"]*font[^)'\"]*['\"]?\s*\)\s*;?\s*)+"
    r"(?:\]\]>)?\s*</style>", re.S | re.I)


def outline_string(svg: str, path: str = "<svg>") -> tuple[str, int, list[str]]:
    """Return (new_svg, number of <text> outlined, warnings)."""
    ctx = _Ctx(path)
    spans = list(_TEXT_RE.finditer(svg))
    if not spans:
        new = svg if "<text" in svg else _FONT_STYLE_RE.sub("", svg)
        return new, 0, ctx.warnings
    root = ET.fromstring(svg.encode("utf-8"))
    parents = {ch: p for p in root.iter() for ch in p}
    texts = [el for el in root.iter() if _local(el.tag) == "text"]
    if len(texts) != len(spans):
        raise ValueError(f"{path}: found {len(spans)} <text> spans but "
                         f"{len(texts)} <text> elements; refusing to splice")
    out, last = [], 0
    for m, el in zip(spans, texts):
        out.append(svg[last:m.start()])
        try:
            out.append(_outline_text(el, parents, ctx))
        except ValueError as e:
            snippet = m.group(0)[:160]
            raise ValueError(f"{path}: {e}\n  in: {snippet}") from None
        last = m.end()
    out.append(svg[last:])
    new = "".join(out)
    if not _TEXT_RE.search(new):
        new = _FONT_STYLE_RE.sub("", new)
    return new, len(spans), ctx.warnings


def outline(path: str) -> str:
    """Outline `path` in place; return a one-line report."""
    with open(path, encoding="utf-8") as fh:
        svg = fh.read()
    new, n, warnings = outline_string(svg, path)
    for w in warnings:
        print(f"  warning: {path}: {w}", file=sys.stderr)
    if new == svg:
        return f"{path}: no live text"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
    return f"{path}: outlined {n} <text>"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[0])
        print("usage: outline_svg.py FILE.svg [FILE2.svg ...]")
        return 2
    rc = 0
    for p in argv:
        try:
            print(outline(p))
        except (ValueError, ET.ParseError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
