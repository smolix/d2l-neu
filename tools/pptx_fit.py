"""Template and font fitting for the pptx slide export (slides_to_pptx.py).

Pandoc's pptx writer leaves every text box at the template's default sizes
(33 pt titles, 24 pt body, code inheriting the body size), so most D2L slides
overflow. Two steps fix that:

* `build_reference` writes a reference .pptx: Calibri/Consolas theme fonts,
  D2L accent colors, a taller content area and smaller default sizes, so
  slides an instructor adds by hand also look right.
* `fit` sets each text box's font size so its content fits the box. It lays
  the text out with real font metrics (Carlito is metric-compatible with
  Calibri; Liberation Mono is slightly wider than Consolas, so code is
  estimated conservatively), wraps words greedily, and picks the largest
  size from a ladder that fits. Sizes go into the box's list style, so
  PowerPoint, Keynote, Google Slides and LibreOffice all honour them.
"""
import io
import math
import re
import subprocess
import zipfile
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

EMU_PER_PT = 12700
FONT_DIRS = ['/usr/share/fonts/truetype/crosextra',
             '/usr/share/fonts/truetype/liberation']

# Geometry of the reference template, in EMU (slide: 9144000 x 5143500).
MARGIN_X = 365760                     # 0.4 in
TITLE = (MARGIN_X, 182880, 8412480, 640080)       # y 0.2 in, h 0.7 in
BODY = (MARGIN_X, 868680, 8412480, 3840480)       # y 0.95 in, h 4.2 in
GAP = 182880                          # 0.2 in between two columns
COL_W = (BODY[2] - GAP) // 2
LEFT = (BODY[0], BODY[1], COL_W, BODY[3])
RIGHT = (BODY[0] + COL_W + GAP, BODY[1], COL_W, BODY[3])

# Web-deck palette (_d2l-slides.scss).
INK, INK2, INK3 = '15181C', '3A4049', '6A717B'
BLUE, BLUE_DARK, BLUE_LIGHT = '2196F3', '1976D2', 'BBDEFB'
KICKER = '\u27e6K\u27e7'             # marks a kicker paragraph for fit()
RULE_Y = 832104                       # blue rule under the title (0.91 in)
FOOT_Y, FOOT_H = 4800600, 228600      # footer strip (y 5.25 in)
LOGO_H = 256032                       # 0.28 in
LOGO_AR = 2884 / 712                  # static/logo-with-text.png

TITLE_MAX, TITLE_MIN = 24, 16
BODY_MAX, BODY_MIN = 18, 8
CODE_RATIO = 0.78                     # code size / body size, as on the web
LINE = 1.25                           # line height / font size
SPACE_BEFORE = 0.3                    # paragraph gap / font size
INSET_X, INSET_Y = 7.2, 3.6           # pt, PowerPoint's default text insets
BULLET_INDENT = 0.9                   # bullet hanging indent / font size
TABLE_MAX = 16                        # largest table text size, pt
STACK_GAP = 91440                     # 0.1 in between stacked blocks
PICTURE_SHARE = 0.5                   # pictures keep at least this much of
                                      # a column's height (or their own)
CELL_X, CELL_Y = 7.2, 3.6             # pt, PowerPoint's default cell margins


@lru_cache(None)
def _font(name, size):
    for d in FONT_DIRS:
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size=size * 10)
    raise FileNotFoundError(name)


def text_width(s, size, bold=False):
    f = _font('Carlito-Bold.ttf' if bold else 'Carlito-Regular.ttf', size)
    return f.getlength(s) / 10


def mono_width(s, size):
    return len(s.expandtabs(4)) * 0.6 * size


def wrap_lines(words, width, size):
    """Greedy word wrap of (word, bold) pairs; returns the line count."""
    lines, cur = 1, 0.0
    space = text_width(' ', size)
    for w, bold in words:
        ww = text_width(w, size, bold)
        if cur and cur + space + ww > width:
            lines += 1
            cur = ww
        else:
            cur += (space if cur else 0) + ww
        while cur > width:          # a single word longer than the line
            lines += 1
            cur -= width
    return lines


# ── template ─────────────────────────────────────────────────────────────

def _xfrm(geom):
    x, y, w, h = geom
    return (f'<a:xfrm><a:off x="{x}" y="{y}" /><a:ext cx="{w}" cy="{h}" />'
            f'</a:xfrm>')


def _set_geom(sp, geom):
    if re.search(r'<a:xfrm>.*?</a:xfrm>', sp, re.S):
        return re.sub(r'<a:xfrm>.*?</a:xfrm>', _xfrm(geom), sp, count=1,
                      flags=re.S)
    return re.sub(r'<p:spPr\s*/>', f'<p:spPr>{_xfrm(geom)}</p:spPr>', sp,
                  count=1)


def _ph(sp):
    m = re.search(r'<p:ph\b([^>]*)/?>', sp)
    if not m:
        return None, None
    a = m.group(1)
    t = re.search(r'type="(\w+)"', a)
    i = re.search(r'idx="(\d+)"', a)
    return (t.group(1) if t else 'body'), (int(i.group(1)) if i else 0)


def _fill(rgb):
    return f'<a:solidFill><a:srgbClr val="{rgb}"/></a:solidFill>'


def _rule(sid):
    x, w = MARGIN_X, BODY[2]
    return (f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="Title rule"/>'
            f'<p:cNvCxnSpPr/><p:nvPr userDrawn="1"/></p:nvCxnSpPr><p:spPr>'
            f'<a:xfrm><a:off x="{x}" y="{RULE_Y}"/><a:ext cx="{w}" cy="0"/>'
            f'</a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom>'
            f'<a:ln w="22225">{_fill(BLUE_LIGHT)}</a:ln></p:spPr></p:cxnSp>')


def _textbox(sid, name, geom, text, sz, rgb, algn='l', bold=False):
    x, y, w, h = geom
    b = ' b="1"' if bold else ''
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{name}"/>'
            f'<p:cNvSpPr txBox="1"/><p:nvPr userDrawn="1"/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" '
            f'cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/>'
            f'</a:prstGeom><a:noFill/></p:spPr><p:txBody>'
            f'<a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" '
            f'anchor="ctr"><a:noAutofit/></a:bodyPr><a:lstStyle/><a:p>'
            f'<a:pPr algn="{algn}"/><a:r><a:rPr lang="en-US" sz="{sz}"{b} '
            f'dirty="0">{_fill(rgb)}<a:latin typeface="+mn-lt"/></a:rPr>'
            f'<a:t>{_esc(text)}</a:t></a:r></a:p></p:txBody></p:sp>')


def _pic(sid, rid, geom):
    x, y, w, h = geom
    return (f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="D2L logo" '
            f'descr="Dive into Deep Learning"/><p:cNvPicPr><a:picLocks '
            f'noChangeAspect="1"/></p:cNvPicPr><p:nvPr userDrawn="1"/>'
            f'</p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/><a:stretch>'
            f'<a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm>'
            f'<a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')


def _esc(t):
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _add_rel(rels, rid, target):
    rel = (f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org'
           f'/officeDocument/2006/relationships/image" Target="{target}"/>')
    return rels.replace('</Relationships>', rel + '</Relationships>')


def _append_shapes(xml, shapes):
    return xml.replace('</p:spTree>', ''.join(shapes) + '</p:spTree>', 1)


def _restyle_master(xml):
    def sp(m):
        s = m.group(0)
        typ, _ = _ph(s)
        if typ == 'title':
            return _set_geom(s, TITLE)
        if typ == 'body':
            return _set_geom(s, BODY)
        return s
    xml = re.sub(r'<p:sp>.*?</p:sp>', sp, xml, flags=re.S)
    title = re.search(r'<p:titleStyle>.*?</p:titleStyle>', xml, re.S).group(0)
    new = re.sub(r'sz="\d+"', 'sz="2400" b="1"', title)
    new = new.replace('algn="ctr"', 'algn="l"', 1)
    new = new.replace('<a:schemeClr val="tx1"/>', f'<a:srgbClr val="{INK}"/>')
    xml = xml.replace(title, new)
    # Title sits on the rule: anchor the master title box to its bottom.
    xml = xml.replace('rtlCol="0" anchor="ctr">', 'rtlCol="0" anchor="b">', 1)
    body = re.search(r'<p:bodyStyle>.*?</p:bodyStyle>', xml, re.S).group(0)
    sizes = iter([1800, 1600, 1400] + [1400] * 6)
    new = re.sub(r'(<a:lvl\dpPr[^>]*>.*?<a:defRPr[^>]*?)sz="\d+"',
                 lambda m: m.group(1) + f'sz="{next(sizes)}"', body,
                 flags=re.S)
    new = re.sub(r'<a:spcPct val="20000" ?/>', '<a:spcPct val="30000"/>', new)
    new = new.replace('<a:schemeClr val="tx1"/>', f'<a:srgbClr val="{INK2}"/>')
    new = re.sub(r'(<a:buFont[^>]*/>)', rf'<a:buClr><a:srgbClr val="{BLUE}"/>'
                 r'</a:buClr>\1', new)
    return xml.replace(body, new)


def _restyle_layout(xml):
    name = re.search(r'<p:cSld name="([^"]*)"', xml).group(1)

    def sp(m):
        s = m.group(0)
        typ, idx = _ph(s)
        if typ == 'title' and '<a:xfrm>' in s:
            return _set_geom(s, TITLE)
        if name == 'Two Content' and typ == 'body':
            return _set_geom(s, LEFT if idx == 1 else RIGHT)
        if name == 'Title and Content' and typ == 'body' and '<a:xfrm>' in s:
            return _set_geom(s, BODY)
        if name == 'Comparison' and typ == 'body':
            col = LEFT if idx in (1, 2) else RIGHT
            head = 457200                 # 0.5 in for the column heading
            return _set_geom(s, (col[0], col[1], col[2], head)
                             if idx in (1, 3) else
                             (col[0], col[1] + head, col[2], col[3] - head))
        if name == 'Content with Caption' and typ == 'body':
            cap = BODY[2] // 3
            return _set_geom(s, (BODY[0], BODY[1], cap - GAP // 2, BODY[3])
                             if idx == 2 else
                             (BODY[0] + cap + GAP // 2, BODY[1],
                              BODY[2] - cap - GAP // 2, BODY[3]))
        return s
    return re.sub(r'<p:sp>.*?</p:sp>', sp, xml, flags=re.S)


def _restyle_theme(xml):
    xml = re.sub(r'(<a:(?:major|minor)Font>\s*<a:latin typeface=")[^"]*"',
                 r'\1Calibri"', xml)
    for slot, rgb in (('accent1', '1976D2'), ('accent2', 'FF5722'),
                      ('dk2', '15181C'), ('hlink', '1976D2')):
        xml = re.sub(rf'<a:{slot}>.*?</a:{slot}>',
                     f'<a:{slot}><a:srgbClr val="{rgb}"/></a:{slot}>', xml,
                     flags=re.S)
    return xml


@lru_cache(None)
def _default_reference(quarto: str) -> bytes:
    return subprocess.run(
        [quarto, 'pandoc', '--print-default-data-file', 'reference.pptx'],
        check=True, capture_output=True).stdout


def build_reference(dst: Path, quarto: Path, footer: str = '',
                    cover: str = '', logo: Path = None):
    """Write the reference deck; `footer` goes bottom left on every slide,
    the logo bottom right, and `cover` under the title on the title slide."""
    default = _default_reference(str(quarto))
    src = zipfile.ZipFile(io.BytesIO(default))
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            data = src.read(item.filename)
            n = item.filename
            if n == 'ppt/theme/theme1.xml':
                data = _restyle_theme(data.decode()).encode()
            elif n == 'ppt/slideMasters/slideMaster1.xml':
                x = _restyle_master(data.decode())
                lw = int(LOGO_H * LOGO_AR)
                shapes = [_rule(90),
                          _textbox(91, 'Footer', (MARGIN_X, FOOT_Y, 5486400,
                                                  FOOT_H), footer, 1000, INK3)]
                if logo:
                    shapes.append(_pic(92, 'rIdD2lLogo', (
                        MARGIN_X + BODY[2] - lw,
                        FOOT_Y + (FOOT_H - LOGO_H) // 2, lw, LOGO_H)))
                data = _append_shapes(x, shapes).encode()
            elif n == 'ppt/slideMasters/_rels/slideMaster1.xml.rels' and logo:
                data = _add_rel(data.decode(), 'rIdD2lLogo',
                                '../media/d2l-logo.png').encode()
            elif n == 'ppt/slideLayouts/_rels/slideLayout1.xml.rels' and logo:
                data = _add_rel(data.decode(), 'rIdD2lLogo',
                                '../media/d2l-logo.png').encode()
            elif n == 'ppt/slideLayouts/slideLayout1.xml':
                x = _restyle_layout(data.decode())
                # The title slide hides the master's rule, footer and small
                # logo; it shows a large logo and the section reference.
                x = x.replace('<p:sldLayout ', '<p:sldLayout showMasterSp="0" ', 1)
                x = x.replace('<a:lstStyle/>', '<a:lstStyle><a:lvl1pPr algn="ctr">'
                              '</a:lvl1pPr></a:lstStyle>', 1)
                shapes = []
                if logo:
                    h = 640080
                    w = int(h * LOGO_AR)
                    shapes.append(_pic(92, 'rIdD2lLogo',
                                       ((9144000 - w) // 2, 594360, w, h)))
                shapes.append(_textbox(93, 'Section', (MARGIN_X, 4434840,
                                       BODY[2], 320040), cover, 1400, BLUE,
                                       algn='ctr', bold=True))
                data = _append_shapes(x, shapes).encode()
            elif re.match(r'ppt/slideLayouts/slideLayout\d+\.xml$', n):
                data = _restyle_layout(data.decode()).encode()
            elif n == '[Content_Types].xml' and logo and 'Extension="png"' not in data.decode():
                data = data.decode().replace(
                    '<Default Extension="xml"',
                    '<Default Extension="png" ContentType="image/png"/>'
                    '<Default Extension="xml"', 1).encode()
            out.writestr(item, data)
        if logo:
            out.write(logo, 'ppt/media/d2l-logo.png')


# ── fitting ──────────────────────────────────────────────────────────────

def _geom_map(xml):
    """Placeholder geometry of a layout or master, keyed by (type, idx)."""
    out = {}
    for s in re.findall(r'<p:sp>.*?</p:sp>', xml, re.S):
        typ, idx = _ph(s)
        g = re.search(r'<a:off x="(\d+)" y="(\d+)" ?/><a:ext cx="(\d+)" '
                      r'cy="(\d+)" ?/>', s)
        if typ and g:
            out[(typ, idx)] = tuple(int(v) for v in g.groups())
    return out


def _paragraphs(body):
    """Each paragraph as (kind, payload, bulleted)."""
    out = []
    for p in re.findall(r'<a:p>.*?</a:p>', body, re.S):
        bulleted = '<a:buNone' not in p
        if '<m:oMathPara' in p:
            tall = bool(re.search(r'<m:(f|rad|nary|sSubSup)\b', p))
            out.append(('display', tall, False))
            continue
        if 'typeface="Consolas"' in p and '<a:br' in p or (
                'typeface="Consolas"' in p and
                not re.search(r'<a:r><a:rPr(?![^>]*Consolas)[^/]*/>', p)):
            lines = []
            for chunk in re.split(r'<a:br\s*/>', p):
                lines.append(''.join(re.findall(r'<a:t>([^<]*)</a:t>',
                                                chunk)))
            out.append(('code', lines, False))
            continue
        words = []
        for run in re.findall(r'<a:r>.*?</a:r>|<a14:m>.*?</a14:m>', p, re.S):
            bold = 'b="1"' in run
            txt = ''.join(re.findall(r'<(?:a|m):t[^>]*>([^<]*)</(?:a|m):t>',
                                     run))
            txt = (txt.replace('&amp;', '&').replace('&lt;', '<')
                   .replace('&gt;', '>'))
            words += [(w, bold) for w in txt.split()]
        out.append(('text', words, bulleted))
    return out


def body_height(paras, size, width):
    code = round(size * CODE_RATIO * 2) / 2
    h = 0.0
    for i, (kind, payload, bulleted) in enumerate(paras):
        s = code if kind == 'code' else size
        if i:
            h += SPACE_BEFORE * s
        if kind == 'display':
            h += (2.4 if payload else 1.4) * size
        elif kind == 'code':
            n = sum(max(1, math.ceil(mono_width(l, code) / width))
                    for l in payload)
            h += n * LINE * code
        else:
            w = width - (BULLET_INDENT * size if bulleted else 0)
            h += wrap_lines(payload, w, size) * LINE * size
    return h


def _ladder(hi, lo):
    s = hi
    while s >= lo:
        yield s
        s -= 0.5


def fit_size(paras, w_pt, h_pt):
    width = w_pt - 2 * INSET_X
    height = (h_pt - 2 * INSET_Y) * 0.97
    for s in _ladder(BODY_MAX, BODY_MIN):
        if body_height(paras, s, width) <= height:
            return s
    return BODY_MIN


def fit_title(text, w_pt, h_pt):
    width = w_pt - 2 * INSET_X
    words = [(w, False) for w in text.split()]
    for s in _ladder(TITLE_MAX, TITLE_MIN):
        n = wrap_lines(words, width, s)
        if n * LINE * s <= h_pt - 2 * INSET_Y:
            return s
    return TITLE_MIN


def _list_style(size):
    code = round(size * CODE_RATIO * 2) / 2
    lvls = ''.join(
        f'<a:lvl{k}pPr marL="{int((k - 1) * 0.9 * size * EMU_PER_PT + 0.9 * size * EMU_PER_PT)}" '
        f'indent="{-int(0.9 * size * EMU_PER_PT)}">'
        f'<a:spcBef><a:spcPts val="{int(SPACE_BEFORE * size * 100)}" /></a:spcBef>'
        f'<a:defRPr sz="{int(size * 100)}" /></a:lvl{k}pPr>'
        for k in (1, 2, 3))
    return f'<a:lstStyle>{lvls}</a:lstStyle>', int(code * 100)


def _apply(sp, size, title=False):
    if title:
        lst = (f'<a:lstStyle><a:lvl1pPr><a:defRPr sz="{int(size * 100)}" />'
               f'</a:lvl1pPr></a:lstStyle>')
        code_sz = None
    else:
        lst, code_sz = _list_style(size)
    sp = re.sub(r'<a:lstStyle\s*/>|<a:lstStyle>.*?</a:lstStyle>', lst, sp,
                count=1, flags=re.S)
    sp = re.sub(r'<a:bodyPr\s*/>', '<a:bodyPr><a:normAutofit /></a:bodyPr>',
                sp, count=1)
    if KICKER in sp:
        ksz = int(max(9, round(size * 0.62)) * 100)

        def kick(m):
            r = m.group(0)
            if KICKER not in r:
                return r
            r = r.replace(KICKER, '')
            r = re.sub(r'<a:rPr\b[^>]*?(/>|>.*?</a:rPr>)',
                       f'<a:rPr lang="en-US" sz="{ksz}" b="1" cap="all" '
                       f'spc="120">{_fill(BLUE)}</a:rPr>', r, count=1,
                       flags=re.S)
            return r
        sp = re.sub(r'<a:r>.*?</a:r>', kick, sp, flags=re.S)
    if code_sz:
        def code_run(m):
            r = m.group(0)
            if 'typeface="Consolas"' not in r:
                return r
            r = re.sub(r'\ssz="\d+"', '', r)
            return r.replace('<a:rPr', f'<a:rPr sz="{code_sz}"', 1)
        sp = re.sub(r'<a:rPr\b[^>]*?(?:/>|>.*?</a:rPr>)', code_run, sp,
                    flags=re.S)
    return sp


def _shape_geom(s, geo):
    """A shape's box: its own xfrm, else its placeholder's on the layout."""
    own = re.search(r'<(?:a|p):xfrm>\s*<a:off x="(-?\d+)" y="(-?\d+)" ?/>'
                    r'\s*<a:ext cx="(\d+)" cy="(\d+)"', s)
    if own:
        return tuple(int(v) for v in own.groups())
    typ, idx = _ph(s)
    return (geo.get((typ, idx)) or geo.get((typ, 0)) or
            geo.get(('body', idx)) or BODY)


def _text_height(paras, size, w):
    """Box height (EMU) that `fit_size` fills at `size` for width `w`."""
    h = body_height(paras, size, w / EMU_PER_PT - 2 * INSET_X)
    return int((h / 0.97 + 2 * INSET_Y + 1) * EMU_PER_PT)


def _cells(frame):
    """A table's rows as lists of cell txBody strings."""
    return [re.findall(r'<a:txBody>.*?</a:txBody>', row, re.S)
            for row in re.findall(r'<a:tr\b.*?</a:tr>', frame, re.S)]


def _col_widths(frame, w):
    """Column widths for a table `w` wide.

    Each column gets its longest word plus a share of the rest in proportion
    to how much text it holds, so a column of short labels stays narrow and
    a column of sentences wraps less. Sized at 10 pt; widths scale with size.
    """
    rows = _cells(frame)
    ncol = max(len(r) for r in rows)
    longest, chars = [0.0] * ncol, [0.0] * ncol
    for row in rows:
        for j, cell in enumerate(row):
            for kind, payload, _ in _paragraphs(cell):
                words = ([(ln, False) for ln in payload] if kind == 'code'
                         else payload if kind == 'text' else [])
                for word, bold in words:
                    longest[j] = max(longest[j], text_width(word, 10, bold))
                chars[j] += sum(len(wd) + 1 for wd, _ in words)
    total = w / EMU_PER_PT
    need = [lw + 2 * CELL_X + 2 for lw in longest]
    if sum(need) >= total:
        return [int(w * n / sum(need)) for n in need]
    spare = total - sum(need)
    weight = [c ** 0.5 for c in chars]
    weight = [wt / sum(weight) if sum(weight) else 1 / ncol for wt in weight]
    return [int((n + spare * wt) * EMU_PER_PT) for n, wt in zip(need, weight)]


def _table_height(frame, size, widths):
    h = 0.0
    for row in _cells(frame):
        h += max((body_height(_paragraphs(c), size,
                              wd / EMU_PER_PT - 2 * CELL_X) + 2 * CELL_Y
                  for c, wd in zip(row, widths)), default=0)
    return int((h + 2) * EMU_PER_PT)


def _size_table(frame, size, widths):
    """Set a table's text size and column widths."""
    code = round(size * CODE_RATIO * 2) / 2
    lst = (f'<a:lstStyle><a:lvl1pPr><a:defRPr sz="{int(size * 100)}" />'
           '</a:lvl1pPr></a:lstStyle>')
    frame = re.sub(r'(<a:tc>\s*<a:txBody>\s*<a:bodyPr\s*/>\s*)'
                   r'(?:<a:lstStyle\s*/>|<a:lstStyle>.*?</a:lstStyle>)',
                   lambda m: m.group(1) + lst, frame, flags=re.S)
    cols = iter(widths)
    frame = re.sub(r'<a:gridCol w="\d+"', lambda m: f'<a:gridCol w="{next(cols)}"',
                   frame)

    def code_run(m):
        r = m.group(0)
        if 'typeface="Consolas"' not in r:
            return r
        return re.sub(r'\ssz="\d+"', '', r).replace(
            '<a:rPr', f'<a:rPr sz="{int(code * 100)}"', 1)
    return re.sub(r'<a:rPr\b[^>]*?(?:/>|>.*?</a:rPr>)', code_run, frame,
                  flags=re.S)


def _place_frame(frame, geom):
    x, y, w, h = geom
    return re.sub(r'<p:xfrm>.*?</p:xfrm>',
                  f'<p:xfrm><a:off x="{x}" y="{y}" /><a:ext cx="{w}" '
                  f'cy="{h}" /></p:xfrm>', frame, count=1, flags=re.S)


def _stack(blocks, region):
    """Lay a column's blocks out top to bottom; returns {start: new xml}.

    `blocks` are (kind, start, xml, payload) in reading order: text
    (payload: paragraphs), table, or picture (payload: height/width). Text
    and tables share one size, the largest that leaves the pictures
    PICTURE_SHARE of the column (or their full-width height, if less);
    the pictures scale into what remains.
    """
    x, top, w, h = region
    pics = [min(w * b[3], h) for b in blocks if b[0] == 'pic']
    full = sum(pics)
    gaps = STACK_GAP * (len(blocks) - 1)
    tables = {b[1]: _col_widths(b[2], w) for b in blocks if b[0] == 'table'}

    def heights(size):
        return [_text_height(b[3], size, w) if b[0] == 'text' else
                _table_height(b[2], min(size, TABLE_MAX), tables[b[1]])
                if b[0] == 'table' else 0 for b in blocks]
    for size in _ladder(BODY_MAX, BODY_MIN):
        hs = heights(size)
        rest = h - gaps - sum(hs)
        if rest >= min(full, PICTURE_SHARE * h):
            break
    scale = min(1.0, max(rest, h // 4) / full) if full else 0
    y = top + (int(max(0, rest - full * scale)) // 2 if not sum(hs) else 0)
    out = {}
    for b, bh in zip(blocks, hs):
        kind, start, s, payload = b
        if kind == 'pic':
            ph = int(min(w * payload, h) * scale)
            pw = int(ph / payload)
            out[start] = _set_geom(s, (x + (w - pw) // 2, y, pw, ph))
            bh = ph
        elif kind == 'table':
            out[start] = _place_frame(
                _size_table(s, min(size, TABLE_MAX), tables[start]),
                (x, y, w, bh))
        else:
            out[start] = _set_geom(s, (x, y, w, bh))
        y += bh + STACK_GAP
    return out


def _restack(xml, geo, layout):
    """Stack text, tables and pictures in each column of a slide.

    Pandoc places a slide's pictures and tables with fixed layouts: text
    above a picture in a column gets a 0.5 in heading strip, text after it
    lands on the picture, and a table after text goes into "Content with
    Caption", which overlaps the title. Columns holding a picture or a
    table are laid out again here.
    """
    shapes = []
    for m in re.finditer(r'<p:(sp|pic|graphicFrame)>.*?</p:\1>', xml, re.S):
        s = m.group(0)
        typ, idx = _ph(s)
        if typ in ('title', 'ctrTitle', 'dt', 'ftr', 'sldNum'):
            continue
        g = _shape_geom(s, geo)
        if m.group(1) == 'pic':
            shapes.append(('pic', m.start(), s, g[3] / g[2], g))
        elif m.group(1) == 'graphicFrame':
            if '<a:tbl>' in s:
                shapes.append(('table', m.start(), s, None, g))
        elif '<p:txBody>' in s and (re.search(r'<a:t>[^<]*\S', s) or
                                    '<a14:m>' in s):
            body = re.search(r'<p:txBody>.*?</p:txBody>', s, re.S).group(0)
            shapes.append(('text', m.start(), s, _paragraphs(body), g))
    if not any(b[0] != 'text' for b in shapes):
        return xml
    two = layout in ('Two Content', 'Comparison')
    mid = BODY[0] + BODY[2] // 2
    columns = {}
    for b in shapes:
        col = (RIGHT if b[4][0] + b[4][2] // 2 > mid else LEFT) if two else BODY
        columns.setdefault(col, []).append(b[:4])
    new = {}
    for col, blocks in columns.items():
        # A text-only column keeps its box, except under Comparison, whose
        # heading strip is too short for a column of text.
        if layout == 'Comparison' or any(b[0] != 'text' for b in blocks):
            new.update(_stack(blocks, col))
    return re.sub(r'<p:(sp|pic|graphicFrame)>.*?</p:\1>',
                  lambda m: new.get(m.start(), m.group(0)), xml, flags=re.S)


MC_NS = 'http://schemas.openxmlformats.org/markup-compatibility/2006'
A14_NS = 'http://schemas.microsoft.com/office/drawing/2010/main'


def _wrap_table_math(xml):
    """Declare the math namespace for tables, as pandoc does for text boxes.

    Pandoc writes an equation as <a14:m> and wraps a text box holding one in
    mc:AlternateContent, whose mc:Choice declares the a14 prefix. A table
    (p:graphicFrame) with math in a cell gets no wrapper, so the slide's XML
    has an unbound prefix and PowerPoint reports the deck as damaged.
    """
    def frame(m):
        s = m.group(0)
        if '<a14:m>' not in s:
            return s
        return (f'<mc:AlternateContent xmlns:mc="{MC_NS}">'
                f'<mc:Choice xmlns:a14="{A14_NS}" Requires="a14">{s}'
                '</mc:Choice></mc:AlternateContent>')
    return re.sub(r'(?<!Requires="a14">)<p:graphicFrame>.*?</p:graphicFrame>',
                  frame, xml, flags=re.S)


def _number_shapes(xml):
    """Give every shape on a slide its own id (1 is the slide's root group).

    Pandoc numbers some shapes 0 or 1, clashing with the root group; nothing
    in these decks refers to shape ids, so they are renumbered in order.
    """
    n = iter(range(1, 1 << 30))
    return re.sub(r'(<p:cNvPr\b[^>]*?\bid=")\d+"',
                  lambda m: f'{m.group(1)}{next(n)}"', xml)


def fit(pptx: Path, logo: Path = None):
    """Rewrite every slide's text sizes in place; returns per-slide sizes."""
    src = zipfile.ZipFile(pptx)
    files = {i.filename: src.read(i.filename) for i in src.infolist()}
    infos = {i.filename: i for i in src.infolist()}
    src.close()
    master = _geom_map(files['ppt/slideMasters/slideMaster1.xml'].decode())
    report = []
    for name in sorted(files, key=lambda n: (len(n), n)):
        if not re.match(r'ppt/slides/slide\d+\.xml$', name):
            continue
        rels = files[name.replace('slides/', 'slides/_rels/') + '.rels']
        layout = re.search(rb'slideLayouts/(slideLayout\d+\.xml)', rels)
        geo = dict(master)
        geo.update(_geom_map(files['ppt/slideLayouts/' +
                                   layout.group(1).decode()].decode()))
        xml = files[name].decode()
        lname = re.search(r'<p:cSld name="([^"]*)"', files[
            'ppt/slideLayouts/' + layout.group(1).decode()].decode())
        xml = _restack(xml, geo, lname.group(1) if lname else '')
        sizes = []

        def sp(m):
            s = m.group(0)
            if '<p:txBody>' not in s:
                return s
            typ, idx = _ph(s)
            own = re.search(r'<p:spPr>\s*<a:xfrm>\s*<a:off x="(\d+)" '
                            r'y="(\d+)" ?/>\s*<a:ext cx="(\d+)" cy="(\d+)"', s)
            g = (tuple(int(v) for v in own.groups()) if own else
                 geo.get((typ, idx)) or geo.get((typ, 0)) or
                 geo.get(('body', idx)) or BODY)
            w, h = g[2] / EMU_PER_PT, g[3] / EMU_PER_PT
            body = re.search(r'<p:txBody>.*?</p:txBody>', s, re.S).group(0)
            if typ in ('title', 'ctrTitle'):
                text = ' '.join(re.findall(r'<a:t>([^<]*)</a:t>', body))
                size = fit_title(text, w, h)
                sizes.append(('title', size))
                return _apply(s, size, title=True)
            if typ in ('dt', 'ftr', 'sldNum', 'subTitle'):
                return s
            size = fit_size(_paragraphs(body), w, h)
            sizes.append(('body', size))
            return _apply(s, size)

        xml = re.sub(r'<p:sp>.*?</p:sp>', sp, xml, flags=re.S)
        files[name] = _number_shapes(_wrap_table_math(xml)).encode()
        report.append((name, sizes))
    # Pandoc keeps the reference deck's master and layouts but not their
    # media, so the logo they point at has to be put back.
    refs = any(b'd2l-logo.png' in d for n, d in files.items()
               if n.endswith('.rels'))
    with zipfile.ZipFile(pptx, 'w', zipfile.ZIP_DEFLATED) as out:
        for n, data in files.items():
            if n == '[Content_Types].xml' and b'Extension="png"' not in data:
                data = data.replace(b'<Default Extension="xml"',
                                    b'<Default Extension="png" ContentType='
                                    b'"image/png"/><Default Extension="xml"', 1)
            out.writestr(infos[n], data)
        if refs and 'ppt/media/d2l-logo.png' not in files:
            if logo is None:
                raise ValueError('deck references the logo; pass logo=')
            out.write(logo, 'ppt/media/d2l-logo.png')
    return report
