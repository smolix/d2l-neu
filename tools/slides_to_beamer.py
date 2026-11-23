#!/usr/bin/env python3
"""Convert injected slide decks (.qmd) to Beamer PDFs.

The sibling of slides_to_pptx.py, for lecturers who teach from LaTeX. It
reads the same injected deck (gen_slides.py + inject_outputs.py, no
execution) and writes pandoc markdown for pandoc's beamer writer:

* diagram SVGs and SVG figures become vector PDFs (rsvg-convert -f pdf);
  rasters are capped and re-encoded as for PowerPoint;
* `.cols` / `.col` become beamer columns with the web deck's widths
  (`.fig` 44%, `.fig.big` 54%, `.narrow` 36%, the rest share what is left);
* the kicker becomes \\dtlkicker, code cells pandoc-highlighted blocks and
  injected outputs a framed verbatim box (tools/beamer/outputs.lua).

The look (tools/beamer/d2l-beamer.tex) follows the PowerPoint template:
16:9, Source Sans 3 with Latin Modern math, the web deck's colours, a
blue rule under each frame title, "§N.M Title · d2l.ai" bottom left and
the D2L logo bottom right.

Fitting. Each frame records the vertical position of its first and last
line (zref-savepos) and the top of the footline. A frame that runs into
the footline gets a smaller body size (code at 0.78x, images capped in
proportion), found by a few xelatex passes of a per-frame bisection
between 11.3 pt (the PowerPoint 18 pt) and 5 pt. Long code lines cap the
size up front from the column width (Inconsolata is 0.5 em per glyph),
but not below 8 pt: past that, long lines wrap.

Usage:
  python3 tools/slides_to_beamer.py _slides/pytorch/chapter_x/f.qmd out.pdf --bundle out-latex.zip
  python3 tools/slides_to_beamer.py --tree _slides/pytorch --workers 8
(`make slides-beamer` runs the second form for every framework: it writes
<deck>.pdf and <deck>-latex.zip, the self-contained LaTeX source, beside
each deck.)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slides_to_pptx import (REPO, QUARTO, LOGO, MAX_PX,  # noqa: E402
                            shrink_raster, resolve_image, deck_meta)

THEME = REPO / 'tools/beamer/d2l-beamer.tex'
LUA = REPO / 'tools/beamer/outputs.lua'
FONTS = REPO / 'static/fonts'      # found by file name via OSFONTDIR

BASE, MIN = 11.3, 5.0          # body size range, pt
CODE = 0.78                    # code size / body size
STEP = 0.25                    # size resolution, pt
TEXTWIDTH = 418.8              # pt: 160 mm page minus 2 x 0.64 cm margins
GLYPH = 0.5                    # Inconsolata advance, em
COL_GAP = 0.04                 # share of the width between two columns
WIDTHS = {'big': 0.54, 'fig': 0.44, 'narrow': 0.36}
MAX_PASSES = 7
CODE_FLOOR = 8.0               # below this body size, long code lines wrap


def svg_to_pdf(svg: Path, pdf: Path) -> Path:
    subprocess.run(['rsvg-convert', '-f', 'pdf', str(svg), '-o', str(pdf)],
                   check=True)
    return pdf


def tex_escape(s: str) -> str:
    """Escape LaTeX specials outside $...$ (for the kicker)."""
    parts = re.split(r'(\$[^$]*\$)', s)
    for i in range(0, len(parts), 2):
        p = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', parts[i])
        p = re.sub(r'([&%#_{}])', r'\\\1', p)
        parts[i] = p.replace('~', r'\textasciitilde{}').replace('^', r'\^{}')
    return ''.join(parts)


ONE_ARG = ('mathcal', 'mathbb', 'mathbf', 'mathrm', 'mathsf', 'mathit',
           'mathfrak', 'mathscr', 'boldsymbol', 'bm', 'textrm', 'text', 'textbf',
           'textit', 'operatorname', 'hat', 'tilde', 'bar', 'vec', 'dot',
           'widehat', 'widetilde', 'overline', 'sqrt')
TWO_ARG = ('frac', 'dfrac', 'tfrac', 'binom')


def _group_end(s: str, i: int) -> int:
    """Index just past the brace group starting at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def brace_scripts(md: str) -> str:
    """Brace a command used as a bare sub/superscript: x_\\mathcal{B} ->
    x_{\\mathcal{B}}. KaTeX accepts the bare form; unicode-math does not."""
    pat = re.compile(r'([_^])\s*\\(' + '|'.join(ONE_ARG + TWO_ARG) + r')(?![A-Za-z])\s*(?=\{)')
    out, pos = [], 0
    for m in pat.finditer(md):
        if m.start() < pos:
            continue
        end = _group_end(md, m.end())
        if end < 0:
            continue
        if m.group(2) in TWO_ARG and end < len(md) and md[end] == '{':
            end2 = _group_end(md, end)
            end = end2 if end2 > 0 else end
        out.append(md[pos:m.start()] + m.group(1) + '{' +
                   md[m.start() + 1:end].lstrip() + '}')
        pos = end
    out.append(md[pos:])
    return ''.join(out)


def col_width(classes: str) -> float:
    if 'big' in classes:
        return WIDTHS['big']
    if 'fig' in classes:
        return WIDTHS['fig']
    if 'narrow' in classes:
        return WIDTHS['narrow']
    return 0.0


def set_column_widths(md: str) -> str:
    """Rewrite .cols/.col divs as pandoc columns with explicit widths."""
    lines = md.split('\n')
    stack, fence = [], False
    for i, line in enumerate(lines):
        if line.startswith('```'):
            fence = not fence
            continue
        if fence or not line.startswith(':::'):
            continue
        m = re.match(r'^:{3,}\s*\{([^}]*)\}', line)
        if m:
            cls = m.group(1)
            if re.search(r'(^|\s)\.cols?\b|(^|\s)\.columns?\b', cls):
                is_group = re.search(r'\.(cols|columns)\b', cls)
                if is_group:
                    align = ' align=center' if '.vc' in cls else ''
                    lines[i] = f'::: {{.columns{align}}}'
                    stack.append(('cols', []))
                else:
                    stack.append(('col', None))
                    if stack[-2:-1] and stack[-2][0] == 'cols':
                        stack[-2][1].append((i, cls))
            else:
                stack.append(('other', None))
            continue
        if re.match(r'^:{3,}\s*$', line) and stack:
            kind, cols = stack.pop()
            if kind == 'cols' and cols:
                fixed = [col_width(c) for _, c in cols]
                free = [w for w in fixed if not w]
                room = 1 - COL_GAP * (len(cols) - 1) - sum(fixed)
                share = room / len(free) if free else 0
                for (j, _), w in zip(cols, fixed):
                    lines[j] = f'::: {{.column width="{round((w or share) * 100)}%"}}'
    return '\n'.join(lines)


IMG_LINE = re.compile(r'^!\[[^\]]*\]\([^)]+\)\s*$', re.M)


def images_beside_text(slide: str) -> str:
    """Text left, a slide's lone images right (as in the PowerPoint)."""
    if not slide.startswith('## ') or '{.cols' in slide:
        return slide
    head, _, rest = slide.partition('\n')
    images = IMG_LINE.findall(rest)
    text = IMG_LINE.sub('', rest).strip()
    kicker = re.match(r'(\\dtlkicker\{.*\}\n)', text)
    lead = ''
    if kicker:
        lead, text = kicker.group(1) + '\n', text[kicker.end():].strip()
    if not images or not text:
        return slide
    return (f'{head}\n\n{lead}::: {{.cols .vc}}\n::: {{.col}}\n{text}\n:::\n'
            f'::: {{.col .fig}}\n' + '\n\n'.join(i.strip() for i in images) +
            '\n:::\n:::\n\n')


def convert(qmd: Path, work: Path):
    """Return (markdown, subtitle) for pandoc's beamer writer."""
    text = qmd.read_text()
    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    title = re.search(r'^title:\s*"?(.*?)"?\s*$', m.group(1), re.M).group(1)
    body = text[m.end():]

    subtitle = ''
    cover = re.search(r'::: \{\.cover\}\n(.*?)\n:::\n', body, re.S)
    if cover:
        sub = re.search(r'\*\*(.+?)\*\*', cover.group(1))
        subtitle = sub.group(1) if sub else ''
        body = body.replace(cover.group(0), '')
    body = re.sub(r'(?m)^# .*\n', '', body)

    counter = [0]
    figs = work / 'figures'
    figs.mkdir(exist_ok=True)
    scratch = work / 'svg'
    scratch.mkdir(exist_ok=True)

    def dgm(match):
        counter[0] += 1
        svg = scratch / f'dgm-{counter[0]:02d}.svg'
        svg.write_text(match.group(1))
        pdf = svg_to_pdf(svg, figs / f'dgm-{counter[0]:02d}.pdf')
        return f'![](figures/{pdf.name})'

    body = re.sub(r'```\{=html\}\n<div class="dgm">\s*(<svg.*?</svg>)\s*</div>\n```',
                  dgm, body, flags=re.S)

    def image(match):
        if match.group(2).startswith('figures/'):     # a diagram, done above
            return match.group(0)
        path = resolve_image(match.group(2), qmd)
        counter[0] += 1
        if path.suffix == '.svg':
            path = svg_to_pdf(path, figs / f'img-{counter[0]:02d}.pdf')
        elif path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            path = shrink_raster(path, figs / f'img-{counter[0]:02d}.png')
        else:
            dst = figs / f'img-{counter[0]:02d}{path.suffix}'
            shutil.copy(path, dst)
            path = dst
        return f'![](figures/{path.name})'

    body = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)(\{[^}]*\})?', image, body)

    def cell(match):
        lines = match.group(1).split('\n')
        if any(re.match(r'#\|\s*echo:\s*false', l) for l in lines):
            return ''                   # output-only cell (`@!id` on the web)
        lines = [l for l in lines if not l.startswith('#|')]
        return '```python\n' + '\n'.join(lines).strip('\n') + '\n```'

    body = re.sub(r'```\{python\}\n(.*?)\n```', cell, body, flags=re.S)
    body = body.replace('<!-- d2l:output -->\n', '')
    body = re.sub(r'::: \{\.cell-output[^}]*\}\n(.*?)\n:::\n?',
                  lambda m: m.group(1) + '\n', body, flags=re.S)
    body = re.sub(r'::: \{\.d2l-note[^}]*\}\n(.*?)\n:::',
                  lambda m: m.group(1), body, flags=re.S)
    body = body.replace('<br>', ' ')
    body = re.sub(r'^\[([^\]]*)\]\{\.kicker\}$',
                  lambda m: '\\dtlkicker{' + tex_escape(m.group(1)) + '}',
                  body, flags=re.M)

    frames = re.split(r'(?m)^(?=## )', body)
    out, n = [], 0
    for f in frames:
        if not f.startswith('## '):
            out.append(f)
            continue
        n += 1
        f = images_beside_text(f)
        head, _, rest = f.partition('\n')
        head = re.sub(r'\s*\{[^}]*\}\s*$', '', head) + ' {.fragile}'
        out.append(f'{head}\n\n```{{=latex}}\n\\dtlstart{{{n}}}\n```\n\n'
                   f'{rest.strip()}\n\n```{{=latex}}\n\\dtlend{{{n}}}\n```\n\n')
    md = set_column_widths(''.join(out))
    parts = re.split(r'(```.*?```)', md, flags=re.S)     # leave code alone
    md = ''.join(p if p.startswith('```') else brace_scripts(p) for p in parts)
    def yq(v):                       # YAML single-quoted: no escapes
        return "'" + v.replace("'", "''") + "'"
    meta = f'---\ntitle: {yq(title)}\n'
    if subtitle:
        meta += f'subtitle: {yq(subtitle)}\n'
    return meta + '---\n' + md, n


def code_caps(md: str) -> dict:
    """Largest body size per frame at which every code line fits its column."""
    caps, frame, widths, fence, longest = {}, 0, [1.0], None, 0
    framed = False
    for line in md.split('\n'):
        if fence is None and line.startswith('## '):
            frame += 1
            widths = [1.0]
            continue
        if line.startswith('```'):
            if fence is None:
                lang = line[3:].strip()
                if lang.startswith('{='):
                    fence = 'raw'
                else:
                    fence, framed, longest = 'code', not lang, 0
            else:
                if fence == 'code' and longest:
                    room = widths[-1] * TEXTWIDTH - (7 if framed else 2)
                    cap = room / (CODE * GLYPH * longest)
                    caps[frame] = min(caps.get(frame, BASE), max(cap, CODE_FLOOR))
                fence = None
            continue
        if fence == 'code':
            longest = max(longest, len(line.rstrip()))
            continue
        if fence:
            continue
        m = re.match(r'^::: \{\.column width="(\d+)%"\}', line)
        if m:
            widths.append(int(m.group(1)) / 100)
        elif line.startswith('::: {'):
            widths.append(widths[-1])
        elif re.match(r'^:::\s*$', line) and len(widths) > 1:
            widths.pop()
    return caps


IMG_SHARE = 0.8      # images start at this share of the body height


def img_cap(avail: float, size: float) -> float:
    """Image height cap (pt): a share of the body, shrinking with the font."""
    return IMG_SHARE * avail * size / BASE


def snap(x: float) -> float:
    return max(MIN, min(BASE, int(x / STEP) * STEP))


def read_positions(aux: Path) -> dict:
    pos = {}
    for m in re.finditer(r'\\zref@newlabel\{(d2l[a-z]*-?\d*)\}\{.*?\\posy\{(-?\d+)\}',
                         aux.read_text(errors='replace')):
        pos[m.group(1)] = int(m.group(2)) / 65536.0      # sp -> pt
    return pos


README = """{title} ({sec}), Dive into Deep Learning, https://d2l.ai
Beamer source of the slide deck; builds with XeLaTeX (TeX Live 2022 or later):

    xelatex {stem}.tex

Files: {stem}.tex (the deck), d2l-beamer.tex (the D2L theme), sizes.tex
(per-frame font sizes fitted so no frame overflows), figures/, d2l-logo.png.
Fonts: Source Sans 3 and Inconsolata if installed (free from Google Fonts),
else TeX Live's Source Sans Pro and Inconsolata (zi4), else the defaults.
To reuse a frame in your own Beamer deck, copy it together with the
\\input{{d2l-beamer}} preamble lines; \\dtlstart/\\dtlend only set its size.
"""


def build_pdf(qmd: Path, pdf: Path, bundle: Path = None, keep: Path = None,
              verbose=False):
    """Write the deck's PDF and (optionally) a zip of its LaTeX source."""
    qmd, pdf = qmd.resolve(), pdf.resolve()
    title, sec = deck_meta(qmd)
    stem = qmd.stem
    env = dict(os.environ, OSFONTDIR=str(FONTS))
    with tempfile.TemporaryDirectory(prefix='d2l-beamer-') as tmp:
        work = Path(tmp)
        md, nframes = convert(qmd, work)
        (work / 'deck.md').write_text(md)
        shutil.copy(THEME, work / 'd2l-beamer.tex')
        shutil.copy(LOGO, work / 'd2l-logo.png')
        footer = f'{sec} {title}  ·  d2l.ai'.strip()
        (work / 'header.tex').write_text(
            '\\def\\dtllogo{d2l-logo.png}\n'
            f'\\def\\dtlfooter{{{footer}}}\n'
            f'\\def\\dtlcover{{Dive into Deep Learning{"  ·  " + sec if sec else ""}}}\n'
            '\\input{d2l-beamer}\n\\InputIfFileExists{sizes}{}{}\n')
        tex_name = f'{stem}.tex'
        r = subprocess.run(
            [str(QUARTO), 'pandoc', 'deck.md', '-f', 'markdown', '-t', 'beamer',
             '-s', '-V', 'aspectratio=169', '-V', 'classoption=t',
             '--pdf-engine=xelatex', '--lua-filter', str(LUA),
             '-H', 'header.tex', '-o', tex_name],
            cwd=work, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f'pandoc failed for {qmd}:\n{r.stderr}')
        tex = (work / tex_name).read_text()
        tex = tex.replace(r'\frame{\titlepage}',
                          '\\begin{frame}[plain]\n\\titlepage\n\\end{frame}')
        (work / tex_name).write_text(tex)

        caps = code_caps(md)
        size = {k: snap(caps.get(k, BASE)) for k in range(1, nframes + 1)}
        fit, fail = {}, {}
        avail0 = {}          # body height of each frame, from the first pass
        imgmax = {}
        final = False
        for npass in range(1, MAX_PASSES + 2):
            (work / 'sizes.tex').write_text(
                '% Per-frame body size (pt) and image height cap (pt), fitted by\n'
                '% d2l tools/slides_to_beamer.py so that no frame overflows.\n' +
                ''.join(f'\\expandafter\\def\\csname d2lsz@{k}\\endcsname{{{v:g}}}\n'
                        for k, v in size.items()) +
                ''.join(f'\\expandafter\\def\\csname d2lim@{k}\\endcsname{{{v:.1f}}}\n'
                        for k, v in imgmax.items()))
            r = subprocess.run(['xelatex', '-interaction=nonstopmode',
                                '-halt-on-error', tex_name],
                               cwd=work, capture_output=True, text=True, env=env)
            logf = work / f'{stem}.log'
            if r.returncode:
                log = logf.read_text(errors='replace') if logf.exists() else r.stdout
                err = log[log.find('\n!'):][:1500] if '\n!' in log else log[-1500:]
                raise RuntimeError(f'xelatex failed for {qmd}:\n{err}')
            pos = read_positions(work / f'{stem}.aux')
            limit = pos['d2lfoot'] + 0.448 / 2.54 * 72.27 + 3
            changed = False
            for k in range(1, nframes + 1):
                top, bot = pos[f'd2ls-{k}'], pos[f'd2le-{k}']
                avail, used = top - limit, top - bot
                s = size[k]
                avail0.setdefault(k, avail)
                if used <= avail:
                    fit[k] = max(fit.get(k, 0), s)
                else:
                    fail[k] = min(fail.get(k, 99), s)
                lo, hi = fit.get(k), fail.get(k)
                if hi is None:                      # fits at its ceiling
                    nxt = s
                elif lo is None:                    # still overflows
                    nxt = snap(min(s * (avail / used) ** 0.6, hi - STEP))
                    if s <= MIN:
                        nxt = MIN
                elif hi - lo > STEP:
                    nxt = snap((lo + hi) / 2)
                else:
                    nxt = lo
                if npass >= MAX_PASSES:             # out of passes: settle on
                    nxt = lo if lo else nxt         # the largest size that fit
                if nxt != s:
                    size[k], changed = nxt, True
                imgmax[k] = img_cap(avail0[k], size[k])
            if verbose:
                print(f'pass {npass}: ' + ' '.join(
                    f'{k}:{size[k]:g}' for k in size), file=sys.stderr)
            if not changed or final:
                break
            final = npass >= MAX_PASSES             # one last compile
        over = [k for k in range(1, nframes + 1)
                if pos[f'd2ls-{k}'] - pos[f'd2le-{k}'] > pos[f'd2ls-{k}'] - limit]
        log = (work / f'{stem}.log').read_text(errors='replace')
        missing = sorted(set(re.findall(r'Missing character: There is no (.) ', log)))
        pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(work / f'{stem}.pdf', pdf)
        if bundle:
            (work / 'README.txt').write_text(
                README.format(title=title, sec=sec or 'd2l.ai', stem=stem))
            files = [tex_name, 'd2l-beamer.tex', 'sizes.tex', 'd2l-logo.png',
                     'README.txt'] + sorted(
                f'figures/{f.name}' for f in (work / 'figures').iterdir())
            tmpzip = work / 'bundle.zip'
            with zipfile.ZipFile(tmpzip, 'w', zipfile.ZIP_DEFLATED) as z:
                for f in files:                       # fixed dates: reproducible
                    info = zipfile.ZipInfo(f'{stem}/{f}', (1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    z.writestr(info, (work / f).read_bytes())
            shutil.copy(tmpzip, bundle)
        if keep:
            keep.mkdir(parents=True, exist_ok=True)
            for f in ('deck.md', tex_name, 'sizes.tex', f'{stem}.log'):
                shutil.copy(work / f, keep / f)
    return {'passes': npass, 'sizes': size, 'overflow': over,
            'missing_glyphs': missing}


TOOLS = [Path(__file__).resolve(), THEME, LUA, LOGO,
         Path(__file__).resolve().parent / 'slides_to_pptx.py']


def outputs_for(qmd: Path):
    return qmd.with_suffix('.pdf'), qmd.with_name(qmd.stem + '-latex.zip')


def _stale(qmd: Path) -> bool:
    outs = outputs_for(qmd)
    if not all(o.exists() for o in outs):
        return True
    t = min(o.stat().st_mtime for o in outs)
    return any(p.stat().st_mtime > t for p in [qmd, *TOOLS])


def _one(qmd):
    try:
        rep = build_pdf(qmd, *outputs_for(qmd))
        return qmd, None, rep['overflow']
    except Exception as e:  # report every failure, build the rest
        return qmd, str(e), []


def build_tree(tree: Path, workers: int, force: bool = False) -> int:
    """Build <deck>.pdf and <deck>-latex.zip next to every chapter_*/<deck>.qmd."""
    from concurrent.futures import ProcessPoolExecutor
    decks = sorted(tree.glob('chapter_*/*.qmd'))
    todo = [q for q in decks if force or _stale(q)]
    print(f'{tree}: {len(todo)} of {len(decks)} decks to convert to Beamer')
    failed, overflowing = [], []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for n, (qmd, err, over) in enumerate(pool.map(_one, todo), 1):
            if err:
                failed.append(qmd)
                print(f'  FAILED {qmd}: {err}', file=sys.stderr)
            elif over:
                overflowing.append(qmd)
                print(f'  still overflowing: {qmd} frames {over}')
            if n % 25 == 0 or n == len(todo):
                print(f'  {n}/{len(todo)} done')
    print(f'beamer: {len(todo) - len(failed)} built, {len(failed)} failed, '
          f'{len(overflowing)} with a frame still overflowing')
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('qmd', type=Path, nargs='?', help='one deck; omit with --tree')
    ap.add_argument('pdf', type=Path, nargs='?')
    ap.add_argument('--bundle', type=Path, help='also write the LaTeX source zip')
    ap.add_argument('--tree', type=Path,
                    help='convert every deck under this framework dir '
                         '(e.g. _slides/pytorch), writing .pdf and -latex.zip')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--keep', type=Path, help='copy the .md/.tex/log here')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    if a.tree:
        sys.exit(build_tree(a.tree, a.workers, a.force))
    if not (a.qmd and a.pdf):
        ap.error('give QMD and PDF, or --tree DIR')
    rep = build_pdf(a.qmd, a.pdf, a.bundle, a.keep, a.verbose)
    print(rep)
    sys.exit(1 if rep['overflow'] else 0)


if __name__ == '__main__':
    main()
