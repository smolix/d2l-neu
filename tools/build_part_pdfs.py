#!/usr/bin/env python3
"""Build one PDF per book part (Basics, Advanced, ...) from the full book.

The per-part PDFs are cut out of the finished full-book LaTeX, so they share
every byte of its content with the full PDF and need no second pipeline:

  1. `make pdf-<fw>` renders the full book: gen_pdf.py -> quarto --to latex ->
     fix_latex.py -> XeLaTeX x3. That leaves, in _pdf/<fw>/,
     `Dive-into-Deep-Learning.tex` (with a `%D2L-SRC <file.md>` marker before
     each source file's top-level heading, put there by fix_latex.py) and
     `Dive-into-Deep-Learning.aux` (every label's final number).
  2. For each part this script writes `_pdf/<fw>/part-<slug>.tex`:
       - the full book's preamble (7 x 10 in, 10pt: gen_pdf.py's
         PDF_GEOMETRY / PDF_FONTSIZE, shared by every PDF) plus pdfpages;
       - the cover (the print-ready trim PDF from the cover zip, or the PNG);
       - a table of contents, which LaTeX builds from this document only, so
         it lists just the part;
       - the part's slice of the body, from its part page up to the next
         file that is not in the part. fix_latex.py already put an explicit
         \\setcounter{chapter}{N-1} before every chapter, and LaTeX resets
         section, equation, figure and table counters per chapter, so every
         number in the slice is the full book's (and the HTML's) number;
       - references cited in the slice only (the citeproc bibliography,
         filtered by the `\\citeproc{ref-KEY}` keys the slice uses).
     A reference to a label outside the part (\\ref{X} or \\hyperref[X]{..})
     is replaced by plain text: the label's number from the full book's .aux,
     or the link text. Such references therefore read exactly as in the full
     book, with no link and no "??".
  3. XeLaTeX x3 in _pdf/<fw>/ (where chapter_*/, img/ and static/fonts/ live)
     with -jobname part-<slug>, then the PDF moves to
     _pdf/parts/Dive-into-Deep-Learning-<slug>-<fw>.pdf.
  4. Checks: every label defined in the part must carry the same number as
     in the full book; no undefined references or citations in the log; no
     "??" in the text (pdftotext). A failed check fails the build.

Usage:
    python3 tools/build_part_pdfs.py [--framework pytorch] [--parts basics advanced]

See docs/build-system.md (§6.6a).
"""
import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from fix_latex import SRC_MARKER  # noqa: E402
from gen_pdf import PDF_FONTSIZE, PDF_GEOMETRY  # noqa: E402
from d2l_preprocess import CHAPTER_NUMBERING  # noqa: E402

# slug -> (part page in _quarto.yml, cover name stem)
PARTS = {
    'basics': ('chapter_basics/index.md', 'Basics'),
    'advanced': ('chapter_advanced/index.md', 'Advanced'),
    'language-models': ('chapter_language-models/index.md', 'Language-Models'),
    'mathematics-for-deep-learning': (
        'chapter_mathematics-for-deep-learning/index.md',
        'Mathematics-for-Deep-Learning'),
}

COVER_ZIP = ROOT / 'static/cover/d2l-part-covers-and-spines-pdf.zip'
COVER_DIR = ROOT / 'static/cover'

BOOK = 'Dive-into-Deep-Learning'


def part_members(part_md):
    """Source .md files of a part, part page first, in _quarto.yml order."""
    cfg = yaml.safe_load((ROOT / '_quarto.yml').read_text(encoding='utf-8'))
    part_qmd = part_md[:-3] + '.qmd'
    for item in cfg['book']['chapters']:
        if isinstance(item, dict) and item.get('part') == part_qmd:
            return [part_md] + [c[:-4] + '.md' for c in item['chapters']]
    sys.exit(f'ERROR: part {part_qmd} not found in _quarto.yml')


def read_aux_labels(aux):
    """label -> number string from a LaTeX .aux file (\\newlabel{X}{{num}...})."""
    labels = {}
    for m in re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^{}]*)\}',
                         aux.read_text(encoding='utf-8', errors='replace')):
        labels[m.group(1)] = m.group(2)
    return labels


def split_book(tex):
    """(preamble, body, bibliography entries) of the fix_latex'd full book."""
    begin = re.search(r'(?m)^\\begin\{document\}[ \t]*$', tex)
    if begin is None:
        sys.exit('ERROR: no \\begin{document} in the full-book .tex')
    preamble = tex[:begin.start()]
    refs = re.search(r'\\begin\{CSLReferences\}\{[^}]*\}\{[^}]*\}\n(.*?)'
                     r'\\end\{CSLReferences\}', tex, re.S)
    if refs is None:
        sys.exit('ERROR: no CSLReferences bibliography in the full-book .tex')
    entries = []
    for chunk in re.split(r'(?=\\bibitem\[\\citeproctext\]\{)', refs.group(1)):
        m = re.match(r'\\bibitem\[\\citeproctext\]\{([^}]*)\}', chunk)
        if m:
            entries.append((m.group(1), chunk.rstrip() + '\n'))
    return preamble, tex[begin.end():], entries


def slice_part(body, members):
    """The body text from the part page up to the first non-member file."""
    markers = [(m.start(), m.group(1)) for m in
               re.finditer(r'(?m)^' + re.escape(SRC_MARKER) + r'(\S+)$', body)]
    if not markers:
        sys.exit(f'ERROR: no "{SRC_MARKER}" markers in the full-book .tex; '
                 'rebuild it with the current tools/fix_latex.py')
    order = [rel for _, rel in markers]
    # PDF-excluded files (website-only pages) have no marker.
    expect = [rel for rel in members if rel in set(order)]
    if members[0] not in order:
        sys.exit(f'ERROR: part page {members[0]} has no marker in the .tex')
    i0 = order.index(members[0])
    if order[i0:i0 + len(expect)] != expect:
        sys.exit(f'ERROR: the files of part {members[0]} are not contiguous '
                 'in the full-book .tex, or not in _quarto.yml order')
    start = markers[i0][0]
    i1 = i0 + len(expect)
    if i1 < len(markers):
        end = markers[i1][0]
    else:  # the last part: stop at the References chapter
        end = body.index('\\chapter*{References}')
    text = body[start:end]
    # The full book switches to arabic page numbers at Preliminaries; a part
    # PDF does so at its own start (\mainmatter).
    text = text.replace('\\pagenumbering{arabic}\\setcounter{page}{1}\n', '')
    # Drop a trailing \bookmarksetup that belonged to the next file.
    text = re.sub(r'\\bookmarksetup\{startatroot\}\s*$', '', text)
    return text


def _brace_arg(text, i):
    """End index (exclusive) of the {...} group starting at text[i] == '{'."""
    depth, j = 0, i
    while j < len(text):
        c = text[j]
        if c == '\\':          # skip the escaped character (\{, \}, \\)
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise ValueError('unbalanced braces')


def externalize_refs(text, full_labels):
    """Plain-text every reference to a label the part does not define."""
    local = set(re.findall(r'\\label\{([^}]+)\}', text))
    missing, n_ref = set(), 0

    def ref(m):
        nonlocal n_ref
        label = m.group(1)
        if label in local:
            return m.group(0)
        if label in full_labels:
            n_ref += 1
            return full_labels[label]
        missing.add(label)
        return m.group(0)

    text = re.sub(r'\\ref\{([^}]+)\}', ref, text)

    out, pos, n_link = [], 0, 0
    for m in re.finditer(r'\\hyperref\[([^\]]+)\]\{', text):
        if m.start() < pos or m.group(1) in local:
            continue
        end = _brace_arg(text, m.end() - 1)
        out.append(text[pos:m.start()])
        out.append(text[m.end():end - 1])     # the link text
        pos = end
        n_link += 1
    out.append(text[pos:])
    return ''.join(out), local, n_ref, n_link, missing


def cover_file(stem, work):
    """The part's cover: the print-ready trim PDF, else the 300 dpi PNG."""
    name = f'd2l-part-{stem}-cover-trim-7x10in.pdf'
    if COVER_ZIP.exists():
        with zipfile.ZipFile(COVER_ZIP) as z:
            if name in z.namelist():
                dst = work / 'covers' / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(name))
                return dst
    # Some PNGs carry a download suffix such as " (1)".
    pngs = sorted(COVER_DIR.glob(f'd2l-part-{stem}-7x10in-300dpi*.png'))
    if pngs:
        dst = work / 'covers' / f'd2l-part-{stem}.png'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pngs[0], dst)
        return dst
    sys.exit(f'ERROR: no cover for part {stem} in {COVER_DIR}')


def part_preamble(preamble, title):
    """The full book's preamble, unchanged except for the title and pdfpages.

    Page and type come from gen_pdf.py (PDF_GEOMETRY, PDF_FONTSIZE), which
    sets them for the full books; a part inherits them. Refuse a full-book
    .tex built with another geometry (a stale _pdf/<fw>/ tree)."""
    geometry = ','.join(PDF_GEOMETRY)
    if f'\\usepackage[{geometry}]{{geometry}}' not in preamble or \
            not re.search(r'\\documentclass\[\s*' + PDF_FONTSIZE + ',', preamble):
        sys.exit('ERROR: the full-book .tex was not built with gen_pdf.py\'s '
                 f'{PDF_FONTSIZE} / {geometry} page; rebuild it (make pdf-<fw>)')
    pre = preamble.replace('pdftitle={Dive into Deep Learning}',
                           f'pdftitle={{Dive into Deep Learning: {title}}}')
    pre = pre.replace('\\title{Dive into Deep Learning}',
                      f'\\title{{Dive into Deep Learning: {title}}}')
    return pre + '\\usepackage{pdfpages}\n'


def bibliography(entries, text, book_tex):
    cited = set()
    for m in re.finditer(r'\\(?:citeproc|cite)\{([^}]*)\}', text):
        cited.update(k.strip() for k in m.group(1).split(','))
    kept = [chunk for key, chunk in entries if key in cited]
    if not kept:   # a part that cites nothing gets no empty References page
        return '', 0, cited
    head = re.search(r'\\chapter\*\{References\}.*?\\begin\{CSLReferences\}'
                     r'\{[^}]*\}\{[^}]*\}\n', book_tex, re.S).group(0)
    tex = ('\\bookmarksetup{startatroot}\n\n' + head + '\n'.join(kept)
           + '\n\\end{CSLReferences}\n')
    return tex, len(kept), cited - {k for k, _ in entries}


def build_part(slug, fw, pdf_dir, out_dir, book_tex, full_labels):
    part_md, stem = PARTS[slug]
    members = part_members(part_md)
    title = (ROOT / part_md).read_text(encoding='utf-8').split('\n')[0]
    title = title.lstrip('# ').strip()

    preamble, body, entries = split_book(book_tex)
    text = slice_part(body, members)
    text, local, n_ref, n_link, missing = externalize_refs(text, full_labels)
    if missing:
        sys.exit(f'ERROR [{slug}]: references to labels that the full book '
                 f'does not define either: {sorted(missing)[:20]}')
    refs_tex, n_bib, uncited = bibliography(entries, text, book_tex)
    cover = cover_file(stem, pdf_dir)

    doc = (part_preamble(preamble, title)
           + '\\begin{document}\n'
           + '\\frontmatter\n'
           + f'\\includepdf[pages=1]{{{cover.relative_to(pdf_dir)}}}\n'
           # the back of the cover: blank, without a running head
           + '\\null\\thispagestyle{empty}\\newpage\n'
           + '\\renewcommand*\\contentsname{Table of contents}\n'
           + '{\n\\setcounter{tocdepth}{2}\n\\tableofcontents\n}\n'
           + '\\mainmatter\n'
           + '\\bookmarksetup{startatroot}\n\n'
           + text + '\n' + refs_tex
           + '\n\\backmatter\n\n\\end{document}\n')
    job = f'part-{slug}'
    (pdf_dir / f'{job}.tex').write_text(doc, encoding='utf-8')
    print(f'[{slug}] {len(members)} files, {len(local)} labels, '
          f'{n_ref} outside refs as text, {n_link} outside links as text, '
          f'{n_bib} references')

    for f in pdf_dir.glob(f'{job}.*'):
        if f.suffix in ('.pdf', '.aux', '.toc', '.out', '.log'):
            f.unlink()
    for _ in range(3):
        subprocess.run(['xelatex', '-interaction=nonstopmode',
                        f'-jobname={job}', f'{job}.tex'], cwd=pdf_dir,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = pdf_dir / f'{job}.pdf'
    if not pdf.exists():
        sys.exit(f'ERROR [{slug}]: xelatex produced no PDF; see '
                 f'{pdf_dir / (job + ".log")}')
    out = out_dir / f'{BOOK}-{slug}-{fw}.pdf'
    shutil.move(pdf, out)

    # ── Checks ──
    errors = []
    part_labels = read_aux_labels(pdf_dir / f'{job}.aux')
    # An unnumbered \chapter* (the part page) has no number of its own; its
    # label inherits whatever counter came before, so it is not compared.
    starred = set(re.findall(
        r'\\chapter\*\{[^}]*\}\\addcontentsline\{toc\}\{chapter\}'
        r'\{[^}]*\}\\label\{([^}]+)\}', text))
    checked = [lab for lab in local - starred
               if lab in part_labels and lab in full_labels]
    wrong = [(lab, part_labels[lab], full_labels[lab]) for lab in checked
             if part_labels[lab] != full_labels[lab]]
    n_checked = len(checked)
    print(f'[{slug}] {n_checked} labels checked against the full book, '
          f'{len(wrong)} differ')
    if wrong:
        errors.append(f'{len(wrong)} labels numbered differently from the '
                      f'full book, e.g. {wrong[:5]}')
    # Chapter numbers must be the HTML edition's (CHAPTER_NUMBERING), the
    # math part, Tools and the Attic included: no appendix lettering.
    want = [CHAPTER_NUMBERING[rel][0] for rel in members
            if CHAPTER_NUMBERING.get(rel) and len(CHAPTER_NUMBERING[rel]) == 1]
    got = [int(n) + 1 for n in
           re.findall(r'\\setcounter\{chapter\}\{(\d+)\}', text)]
    if got != want or '\\appendix' in text:
        errors.append(f'chapter numbers {got} differ from CHAPTER_NUMBERING '
                      f'{want}, or the slice holds \\appendix')
    log = (pdf_dir / f'{job}.log').read_text(encoding='utf-8',
                                             errors='replace')
    undef = sorted(set(re.findall(
        r"(?:Reference|Citation) `([^']+)' on page \d+ undefined", log)))
    if undef:
        errors.append(f'undefined references/citations: {undef[:10]}')
    pages = first_ch = None
    if shutil.which('pdftotext') and shutil.which('pdfinfo'):
        info = subprocess.run(['pdfinfo', str(out)], capture_output=True,
                              text=True).stdout
        pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
        txt = subprocess.run(['pdftotext', str(out), '-'],
                             capture_output=True, text=True).stdout
        # LaTeX prints an unresolved \ref as "??"; the book also has literal
        # ones (IPython's `obj??`), so only an excess over those is an error.
        n_qq, n_lit = txt.count('??'), doc.count('??')
        if n_qq > n_lit:
            errors.append(f'{n_qq - n_lit} unresolved "??" in the text '
                          f'({n_qq} found, {n_lit} literal in the source)')
    m = re.search(r'\\setcounter\{chapter\}\{(\d+)\}\n\\chapter\{', text)
    if m:
        n = int(m.group(1)) + 1
        first_ch = str(n)
    print(f'[{slug}] -> {out.relative_to(ROOT)}: {pages} pages, first '
          f'chapter {first_ch}, {n_bib} references'
          + (f', {len(uncited)} cited keys missing from the bibliography'
             if uncited else ''))
    if errors:
        for e in errors:
            print(f'ERROR [{slug}]: {e}')
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--framework', default='pytorch')
    ap.add_argument('--parts', nargs='*', default=list(PARTS),
                    choices=list(PARTS))
    ap.add_argument('--out-dir', type=Path, default=ROOT / '_pdf/parts')
    args = ap.parse_args()

    pdf_dir = ROOT / '_pdf' / args.framework
    tex, aux = pdf_dir / f'{BOOK}.tex', pdf_dir / f'{BOOK}.aux'
    for f in (tex, aux):
        if not f.exists():
            sys.exit(f'ERROR: {f} missing; build the full book first '
                     f'(make pdf-{args.framework})')
    book_tex = tex.read_text(encoding='utf-8')
    full_labels = read_aux_labels(aux)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    for slug in args.parts:   # sequential on purpose: one xelatex at a time
        ok &= build_part(slug, args.framework, pdf_dir, args.out_dir,
                         book_tex, full_labels)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
