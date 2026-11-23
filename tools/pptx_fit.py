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
        files[name] = xml.encode()
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
