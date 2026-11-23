#!/usr/bin/env python3
"""Convert injected slide decks (.qmd) to editable PowerPoint (.pptx).

Input is a deck that gen_slides.py generated and inject_outputs.py filled
with outputs from the store (no execution). The script rewrites it into
plain pandoc markdown that pandoc's pptx writer understands, then calls
pandoc:

* inline diagram SVGs (```{=html} <div class="dgm"><svg>…) and every SVG
  image become PNGs (Google Slides and older PowerPoint cannot import SVG);
* `.cols` / `.col` become pandoc's `.columns` / `.column` (Two Content layout);
  a slide's kicker line moves into its first column, because pandoc puts
  any text outside a columns div on a slide of its own;
* `{python}` cells become plain ```python blocks without their `#|` options,
  and injected outputs become plain code blocks;
* reveal-only markup (cover div, fragments, nostretch) is dropped;
* LaTeX math stays math: pandoc writes it as native, editable equations.

Font sizes, template, logo and footer: see pptx_fit.py.

Usage:
  python3 tools/slides_to_pptx.py _slides/pytorch/chapter_x/f.qmd out.pptx
  python3 tools/slides_to_pptx.py --tree _slides/pytorch --workers 8
(`make slides-pptx` runs the second form for every framework.)
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pptx_fit  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


MAX_PX = 1800      # longest side of a rasterized figure (~2x a full-width slide)


def svg_to_png(svg: Path, png: Path, zoom: float = 3.0) -> Path:
    """Rasterize an SVG for PowerPoint, capped at MAX_PX on its longest side."""
    subprocess.run(['rsvg-convert', '-z', str(zoom), str(svg), '-o', str(png)],
                   check=True)
    return shrink_raster(png, png)


def shrink_raster(src: Path, dst: Path) -> Path:
    """Cap a raster at MAX_PX and re-encode it compactly: flat-colour
    figures (plots, diagrams) as 256-colour PNG, photographs as JPEG.
    Returns the written path (its suffix may change to .jpg)."""
    from PIL import Image
    with Image.open(src) as im:
        im.load()
    if max(im.size) > MAX_PX:
        im.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)
    rgba = im.convert('RGBA')
    probe = rgba.copy()
    probe.thumbnail((256, 256))
    colors = probe.getcolors(1 << 16)
    if colors is None or len(colors) > 6000:          # photograph-like
        white = Image.new('RGB', rgba.size, 'white')
        white.paste(rgba, mask=rgba.getchannel('A'))
        dst = dst.with_suffix('.jpg')
        white.save(dst, quality=88, optimize=True)
    else:
        rgba.quantize(256, method=Image.Quantize.FASTOCTREE).save(
            dst.with_suffix('.png'), optimize=True)
        dst = dst.with_suffix('.png')
    return dst


def resolve_image(src: str, qmd: Path) -> Path:
    p = (qmd.parent / src).resolve()
    if p.exists():
        return p
    alt = REPO / 'img' / Path(src).name
    if alt.exists():
        return alt
    raise FileNotFoundError(f'{qmd}: image {src} not found')


IMG_LINE = re.compile(r'^!\[[^\]]*\]\([^)]+\)\s*$', re.M)


def images_beside_text(slide: str) -> str:
    """Put a slide's images in a right-hand column next to its other content.

    Pandoc's pptx writer gives an image that shares a slide with text the
    "Content with Caption" layout and moves whatever follows the image onto
    an untitled slide of its own. Two columns keep the slide whole.
    """
    if not slide.startswith('## ') or '::: {.columns}' in slide:
        return slide
    head, _, rest = slide.partition('\n')
    images = IMG_LINE.findall(rest)
    text = IMG_LINE.sub('', rest).strip()
    if not images or not text:
        return slide
    return (f'{head}\n\n:::: {{.columns}}\n::: {{.column}}\n{text}\n:::\n'
            f'::: {{.column}}\n' + '\n\n'.join(i.strip() for i in images) +
            '\n:::\n::::\n\n')


def fold_into_columns(slide: str) -> str:
    """Move text outside a slide's columns div into its first/last column.

    Pandoc's pptx writer gives a columns div a slide of its own: text before
    it stays on the titled slide and the columns go to a new, untitled one.
    The web decks often put a kicker or a lead-in line above the columns,
    so fold that text into the first column (and trailing text into the
    last) to keep one slide per source slide.
    """
    if not slide.startswith('## ') or '::: {.columns}' not in slide:
        return slide
    lines = slide.split('\n')
    head, rest = lines[0], lines[1:]
    start = rest.index('::: {.columns}')
    depth, end = 0, None
    for i in range(start, len(rest)):
        if rest[i].startswith(':::'):
            depth += 1 if rest[i].strip() != ':::' else -1
            if depth == 0:
                end = i
                break
    if end is None:
        return slide
    pre = '\n'.join(rest[:start]).strip()
    post = '\n'.join(rest[end + 1:]).strip()
    cols = rest[start:end + 1]
    if pre:
        first = cols.index('::: {.column}')
        cols[first + 1:first + 1] = [pre, '']
    if post:
        last = len(cols) - 2            # the closing ::: of the last column
        cols[last:last] = ['', post]
    return '\n'.join([head, ''] + cols) + '\n\n'


def convert(qmd: Path, work: Path) -> str:
    text = qmd.read_text()
    # Title from the YAML header; drop the reveal format block.
    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    title = re.search(r'^title:\s*"?(.*?)"?\s*$', m.group(1), re.M).group(1)
    body = text[m.end():]

    counter = [0]

    def dgm(match):
        counter[0] += 1
        svg = work / f'dgm-{counter[0]:02d}.svg'
        svg.write_text(match.group(1))
        png = svg_to_png(svg, svg.with_suffix('.png'))
        return f'![]({png})'

    body = re.sub(r'```\{=html\}\n<div class="dgm">\s*(<svg.*?</svg>)\s*</div>\n```',
                  dgm, body, flags=re.S)

    def image(match):
        src = match.group(2)
        path = resolve_image(src, qmd)
        counter[0] += 1
        if path.suffix == '.svg':
            path = svg_to_png(path, work / f'img-{counter[0]:02d}.png')
        elif path.suffix.lower() in ('.png', '.jpg', '.jpeg') and path.exists():
            path = shrink_raster(path, work / f'img-{counter[0]:02d}.png')
        return f'![{match.group(1)}]({path})'

    body = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)(\{[^}]*\})?', image, body)

    # Code cells: drop the {python} engine marker and the #| options.
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

    # Reveal-only wrappers.
    body = re.sub(r'::: \{\.cover\}\n.*?\n:::\n', '', body, flags=re.S)
    # The deck's `# Title` section slide repeats the title slide.
    body = re.sub(r'(?m)^# .*\n', '', body)
    body = re.sub(r'::: \{\.d2l-note[^}]*\}\n(.*?)\n:::',
                  lambda m: m.group(1), body, flags=re.S)

    # Columns, with the kicker moved into the first column.
    slides = re.split(r'(?m)^(?=## )', body)
    out = []
    for s in slides:
        kicker = re.search(r'^\[([^\]]*)\]\{\.kicker\}\n', s, re.M)
        if kicker:
            s = s.replace(kicker.group(0),
                          f'{pptx_fit.KICKER}{kicker.group(1)}\n', 1)
        s = re.sub(r'::: \{\.cols[^}]*\}', '::: {.columns}', s)
        s = re.sub(r'::: \{\.col(?: [^}]*)?\}', '::: {.column}', s)
        s = fold_into_columns(s)
        s = images_beside_text(s)
        out.append(s)
    body = ''.join(out)
    return f'---\ntitle: "{title}"\n---\n{body}'


LOGO = REPO / 'static/logo-with-text.png'
QUARTO = REPO / '.venv-build/bin/quarto'
# A deck is rebuilt when its .qmd or any of these is newer than the .pptx.
TOOLS = [Path(__file__).resolve(), Path(pptx_fit.__file__).resolve(), LOGO]


def deck_meta(qmd: Path):
    """(title, section) of a deck, section like '§5.4' or ''."""
    text = qmd.read_text()
    title = re.search(r'^title:\s*"?(.*?)"?\s*$', text, re.M).group(1)
    sec = re.search(r'Dive into Deep Learning · (§[\d.]+)', text)
    return title, (sec.group(1) if sec else '')


def build_pptx(qmd: Path, pptx: Path, reference_doc: Path = None,
               keep_md: Path = None, verbose: bool = False):
    qmd, pptx = qmd.resolve(), pptx.resolve()
    with tempfile.TemporaryDirectory(prefix='d2l-pptx-') as tmp:
        work = Path(tmp)
        md = convert(qmd, work)
        md_path = work / 'deck.md'
        md_path.write_text(md)
        if keep_md:
            keep_md.write_text(md)
        ref = reference_doc
        if ref is None:
            title, sec = deck_meta(qmd)
            ref = work / 'd2l-reference.pptx'
            pptx_fit.build_reference(
                ref, QUARTO, footer=f'{sec} {title}  ·  d2l.ai'.strip(),
                cover=f'Dive into Deep Learning  ·  {sec}' if sec else '',
                logo=LOGO)
        tmp_out = work / 'deck.pptx'
        cmd = [str(QUARTO), 'pandoc', str(md_path),
               '-f', 'markdown', '-t', 'pptx', '--slide-level=2',
               '-V', 'monofont=Consolas', '--reference-doc', str(ref),
               '-o', str(tmp_out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f'pandoc failed for {qmd}:\n{r.stderr}')
        report = pptx_fit.fit(tmp_out, logo=LOGO)
        pptx.parent.mkdir(parents=True, exist_ok=True)
        pptx.write_bytes(tmp_out.read_bytes())
    if verbose:
        for name, sizes in report:
            print(name.split('/')[-1], sizes)
    return report


def _stale(qmd: Path, pptx: Path) -> bool:
    if not pptx.exists():
        return True
    t = pptx.stat().st_mtime
    return any(p.stat().st_mtime > t for p in [qmd, *TOOLS])


def _one(job):
    qmd, pptx = job
    try:
        build_pptx(qmd, pptx)
        return qmd, None
    except Exception as e:  # report every failure, build the rest
        return qmd, str(e)


def build_tree(tree: Path, workers: int, force: bool = False) -> int:
    """Build <deck>.pptx next to every chapter_*/<deck>.qmd under `tree`."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(q, q.with_suffix('.pptx'))
            for q in sorted(tree.glob('chapter_*/*.qmd'))]
    todo = [j for j in jobs if force or _stale(*j)]
    print(f'{tree}: {len(todo)} of {len(jobs)} decks to convert to pptx')
    failed = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for n, (qmd, err) in enumerate(pool.map(_one, todo), 1):
            if err:
                failed.append(qmd)
                print(f'  FAILED {qmd}: {err}', file=sys.stderr)
            elif n % 25 == 0 or n == len(todo):
                print(f'  {n}/{len(todo)} done')
    print(f'pptx: {len(todo) - len(failed)} built, {len(failed)} failed')
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('qmd', type=Path, nargs='?',
                    help='one deck; omit with --tree')
    ap.add_argument('pptx', type=Path, nargs='?')
    ap.add_argument('--tree', type=Path,
                    help='convert every deck under this framework dir '
                         '(e.g. _slides/pytorch), writing .pptx beside each')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--reference-doc', type=Path)
    ap.add_argument('--keep-md', type=Path)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    if args.tree:
        sys.exit(build_tree(args.tree, args.workers, args.force))
    if not (args.qmd and args.pptx):
        ap.error('give QMD and PPTX, or --tree DIR')
    build_pptx(args.qmd, args.pptx, args.reference_doc, args.keep_md,
               args.verbose)


if __name__ == '__main__':
    main()
