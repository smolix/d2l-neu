#!/usr/bin/env python3
"""Build slide-deck integration artifacts:

  * `_slides/index.html`        — the Slides landing page: every deck, grouped
                                  by part and chapter and numbered as in the
                                  book. It is set inside the book's own page
                                  frame (navbar, sidebar, footer), taken from a
                                  rendered part page in `_book/`, so it looks
                                  like any other book page. Linked from the
                                  book navbar (`Slides`).
  * `_slides/manifest.json`     — `{"chapter_X/file": ["pytorch", "jax"], ...}`
                                  so external consumers can introspect which
                                  decks exist.
  * `_d2l-slides-data.html`     — `<script>window.D2L_SLIDES_MANIFEST=…;
                                  window.D2L_SLIDES_EXPORTS=…;</script>`
                                  included into every book page via the
                                  `include-after-body:` Quarto hook so the
                                  right-sidebar TOC slides-button knows which
                                  framework variants exist for the page.

Walks `_slides/<fw>/chapter_*/*.html` for each framework. If `_slides/` is
missing or empty, emits a stub data file with an empty manifest so the
book still builds cleanly without slides present.

The html recipe runs this script twice: before `quarto render` (data include,
manifest, and a landing page in the frame of the previous build), and from
`tools/integrate_slides.sh` with `--landing-only --out
_book/slides/index.html`, once the freshly rendered `_book/` can supply the
frame.

Usage:
    python tools/build_slides_index.py [--slides-dir _slides]
                                       [--source .]
                                       [--data-file _d2l-slides-data.html]
                                       [--landing-only] [--out PATH]
                                       [--frame _book/<part>/index.html]
"""

import argparse
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import CHAPTER_NUMBERING
import i18n


# Downloadable versions of each deck, written beside <deck>.html:
# PowerPoint (tools/slides_to_pptx.py), Beamer PDF and its LaTeX source
# (tools/slides_to_beamer.py).
EXPORTS = {'pdf': '.pdf', 'pptx': '.pptx', 'tex': '-latex.zip'}

FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'mxnet']


def extract_h1(md_path: Path) -> str:
    """Return the first level-1 heading text from a source .md file."""
    if not md_path.exists():
        return md_path.stem.replace('-', ' ').title()
    for line in md_path.read_text(encoding='utf-8').split('\n'):
        s = line.strip()
        if s.startswith('# ') and not s.startswith('## '):
            heading = re.sub(r'\s*\{#[^}]+\}', '', s[2:]).strip()
            heading = re.sub(r':label:`[^`]+`', '', heading).strip()
            return heading
    return md_path.stem.replace('-', ' ').title()


def book_parts(source: Path) -> dict:
    """Map each chapter directory to the title of its book part, read from
    the `- part: <dir>/index.qmd` entries of `_quarto.yml` (the sidebar's
    order and grouping)."""
    parts, part = {}, None
    quarto = source / '_quarto.yml'
    if not quarto.exists():
        return parts
    for line in quarto.read_text(encoding='utf-8').split('\n'):
        m = re.match(r'\s*- part:\s*(\S+)/index\.qmd\s*$', line)
        if m:
            part = extract_h1(source / m.group(1) / 'index.md')
            continue
        m = re.match(r'\s*- (chapter_[^/\s]+)/[^/\s]+\.qmd\s*$', line)
        if m and part:
            parts.setdefault(m.group(1), part)
    return parts


def number(rel: str) -> str:
    """The book's number of a page: '8.1' for a section, '8' for a chapter,
    '' for an unnumbered page."""
    num = CHAPTER_NUMBERING.get(rel)
    return '.'.join(str(n) for n in num) if num else ''


def deck_entry(slides_dir: Path, chap: str, stem: str, title: str,
               num: str):
    """A deck's frameworks and downloads, or None if no framework has it."""
    available = [fw for fw in FRAMEWORKS
                 if (slides_dir / fw / chap / f'{stem}.html').exists()]
    if not available:
        return None
    entry = {'stem': stem, 'title': title, 'num': num,
             'frameworks': available}
    for kind, suffix in EXPORTS.items():
        fws = [fw for fw in available
               if (slides_dir / fw / chap / f'{stem}{suffix}').exists()]
        if fws:
            entry[kind] = fws
    return entry


def build_index(slides_dir: Path, source: Path):
    """Return ({'parts': [...], 'exports': {...}}, manifest).

    Parts hold chapters in book order; a chapter has its section decks and,
    if its index page has one, an overview deck. The manifest (and the
    exports map) list section decks only, as the in-book Slides button
    expects."""
    part_of = book_parts(source)
    chapters, order = {}, []
    manifest = {}
    exports = {kind: {} for kind in EXPORTS}

    for rel in CHAPTER_NUMBERING:
        rel_path = Path(rel)
        chap = rel_path.parent.name
        if chap not in chapters:
            order.append(chap)
            chapters[chap] = {
                'dir': chap,
                'title': extract_h1(source / chap / 'index.md'),
                'num': number(f'{chap}/index.md'),
                'part': part_of.get(chap, ''),
                'overview': deck_entry(
                    slides_dir, chap, 'index',
                    extract_h1(source / chap / 'index.md'),
                    number(f'{chap}/index.md')),
                'decks': [],
            }
        if rel_path.stem == 'index':
            continue
        entry = deck_entry(slides_dir, chap, rel_path.stem,
                           extract_h1(source / rel), number(rel))
        if entry is None:
            continue
        chapters[chap]['decks'].append(entry)
        manifest[f'{chap}/{rel_path.stem}'] = entry['frameworks']
        for kind in EXPORTS:
            if kind in entry:
                exports[kind][f'{chap}/{rel_path.stem}'] = entry[kind]

    parts = []
    for chap in order:
        c = chapters[chap]
        if not c['decks']:
            continue
        if not parts or parts[-1]['title'] != c['part']:
            parts.append({'title': c['part'], 'chapters': []})
        parts[-1]['chapters'].append(c)
    return {'parts': parts, 'exports': exports}, manifest


# ──────────────────────────────────────────────────────────
# Landing page
# ──────────────────────────────────────────────────────────

def html_escape(s: str) -> str:
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;')
             .replace('"', '&quot;'))


def slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')


FW_SHORT = {'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow', 'jax': 'JAX',
            'mxnet': 'MXNet'}

def download_kinds():
    """(kind, label, icon, attributes) of each download, in the UI language."""
    t, h = i18n.t, i18n.html_text
    return (
        ('pdf', h(t('slides_index.pdf')), 'bi-file-earmark-pdf',
         f'title="{h(t("slides_index.pdf_title"))}"'),
        ('pptx', h(t('slides_index.pptx')), 'bi-file-earmark-slides',
         f'download title="{h(t("slides_index.pptx_title"))}"'),
        ('tex', h(t('slides_index.latex')), 'bi-file-earmark-code',
         f'download title="{h(t("slides_index.latex_title"))}"'),
    )


def fw_phrase(fws: list) -> str:
    """'All four frameworks', 'PyTorch and JAX only', 'PyTorch only'."""
    if len(fws) == len(FRAMEWORKS):
        return i18n.t('slides_index.all_frameworks')
    names = [FW_SHORT[f] for f in fws]
    head = i18n.t('slides_index.list_sep').join(names[:-1])
    listed = (i18n.t('slides_index.list_and', head=head, last=names[-1])
              if head else names[-1])
    return i18n.t('slides_index.only', list=listed)


def render_deck(chap_dir: str, deck: dict, overview: bool = False) -> str:
    """One row: number, title (opens the deck), the framework the deck opens
    in when the chosen one lacks it, downloads, and a link to the section."""
    stem = deck['stem']
    downloads = ''.join(
        f'<a class="d2l-dl" data-kind="{kind}" '
        f'data-fws="{",".join(deck.get(kind, []))}" {attrs}>'
        f'<i class="bi {icon}" aria-hidden="true"></i>{label}</a>'
        for kind, label, icon, attrs in download_kinds())
    page = 'index' if overview else stem
    title = (html_escape(i18n.t('slides_index.chapter_overview')) if overview
             else html_escape(deck['title']))
    read = i18n.t('slides_index.read_chapter' if overview
                  else 'slides_index.read_section')
    return (
        f'<li class="d2l-deck{" d2l-overview" if overview else ""}" '
        f'data-dir="{chap_dir}" data-stem="{stem}" '
        f'data-fws="{",".join(deck["frameworks"])}">'
        f'<a class="d2l-open" href="pytorch/{chap_dir}/{stem}.html">'
        f'<span class="d2l-num">{deck["num"]}</span>'
        f'<span class="d2l-title">{title}</span>'
        f'<span class="d2l-fallback"></span></a>'
        f'<span class="d2l-actions">{downloads}'
        f'<a class="d2l-book" href="../{chap_dir}/{page}.html" '
        f'title="{html_escape(read)}"><i class="bi bi-book" '
        f'aria-hidden="true"></i></a>'
        f'</span></li>')


def render_content(idx: dict) -> tuple:
    """Return (main column HTML, margin table of contents HTML)."""
    n_decks = sum(len(c['decks']) + bool(c['overview'])
                  for p in idx['parts'] for c in p['chapters'])
    tabs = ''.join(
        f'<li class="nav-item" role="presentation"><a class="nav-link" '
        f'href="#" role="tab" data-fw="{fw}">{FW_SHORT[fw]}</a></li>'
        for fw in FRAMEWORKS)
    t = i18n.t
    head = [
        '<header id="title-block-header" class="quarto-title-block default">'
        f'<div class="quarto-title"><h1 class="title">'
        f'{html_escape(t("slides_index.title"))}</h1></div>'
        '</header>',
        '<div class="d2l-slides-index">',
        f'<p class="d2l-lead">{t("slides_index.lead", n=n_decks)}</p>',
        '<div class="d2l-fw-bar"><span class="d2l-fw-label">'
        f'{html_escape(t("slides_index.framework"))}</span>'
        f'<ul class="nav nav-tabs" role="tablist">{tabs}</ul></div>',
    ]
    toc = ['<nav id="TOC" role="doc-toc">',
           f'<h2 id="toc-title">{html_escape(t("slides_index.parts"))}</h2>',
           '<ul>']
    cards, sections = [], []
    for part in idx['parts']:
        pid = 'part-' + slug(part['title'] or 'front-matter')
        fws = sorted({fw for c in part['chapters'] for d in c['decks']
                      for fw in d['frameworks']}, key=FRAMEWORKS.index)
        nums = [int(c['num']) for c in part['chapters']
                if c['num'].isdigit()]
        span = ('' if not nums else t('slides_index.chapter', n=nums[0])
                if len(nums) == 1 else
                t('slides_index.chapters', first=nums[0], last=nums[-1]))
        count = sum(len(c['decks']) + bool(c['overview'])
                    for c in part['chapters'])
        decks = i18n.tn('slides_index.decks', count)
        fw_words = fw_phrase(fws)
        summary = ' &middot; '.join(x for x in (span, decks, fw_words) if x)
        missing = [FW_SHORT[f] for f in FRAMEWORKS if f not in fws]
        toc.append(f'<li><a href="#{pid}" class="nav-link" '
                   f'data-scroll-target="#{pid}">'
                   f'{html_escape(part["title"])}</a></li>')
        cards.append(f'<a class="d2l-card" href="#{pid}">'
                     f'<span class="d2l-card-title">'
                     f'{html_escape(part["title"])}</span>'
                     f'<span class="d2l-card-meta">'
                     f'{" &middot; ".join(x for x in (span, decks) if x)}'
                     f'</span><span class="d2l-card-fws">{fw_words}</span>'
                     f'</a>')
        sections.append(f'<section id="{pid}" class="level2 d2l-part" '
                   f'data-fws="{",".join(fws)}">')
        sections.append(f'<h2 class="anchored">{html_escape(part["title"])}</h2>')
        sections.append(f'<p class="d2l-part-note">{summary}</p>')
        if missing:
            sections.append(
                '<p class="d2l-part-fallback" hidden>' +
                t('slides_index.part_fallback',
                  fw='<span class="d2l-fw-name"></span>',
                  first=FW_SHORT[fws[0]]) + '</p>')
        for chap in part['chapters']:
            cid = 'chap-' + slug(chap['dir'].replace('chapter_', ''))
            num = (f'<span class="header-section-number">{chap["num"]}'
                   f'</span> ' if chap['num'] else '')
            sections.append(f'<section id="{cid}" class="level3 d2l-chapter">')
            sections.append(f'<h3 class="anchored">{num}'
                       f'{html_escape(chap["title"])}</h3>')
            sections.append('<ol class="d2l-decks">')
            if chap['overview']:
                sections.append(render_deck(chap['dir'], chap['overview'], True))
            for deck in chap['decks']:
                sections.append(render_deck(chap['dir'], deck))
            sections.append('</ol></section>')
        sections.append('</section>')
    toc.append('</ul></nav>')
    page = head + ['<nav class="d2l-cards" aria-label="'
                   f'{html_escape(t("slides_index.parts"))}">'
                   + ''.join(cards) + '</nav>'] + sections + ['</div>']
    return '\n'.join(page), '\n'.join(toc)


STYLE = """<style>
.d2l-slides-index { --d2l-blue: #2196F3; --d2l-blue-dark: #1565C0;
  --d2l-rule: #e7e9ec; --d2l-hover: #f2f8fe; --d2l-muted: #6c757d; }
.d2l-slides-index .d2l-lead { font-size: 1.1rem; line-height: 1.6;
  margin: 0.5rem 0 1.25rem; }
.d2l-fw-bar { display: flex; align-items: flex-end; gap: 1rem;
  border-bottom: 1px solid #dee2e6; margin: 0 0 1.5rem; padding-top: .25rem; }
.d2l-fw-bar .nav-tabs { border-bottom: 0; flex-wrap: nowrap;
  overflow-x: auto; }
.d2l-fw-bar .nav-link { white-space: nowrap; }
.d2l-fw-label { color: var(--d2l-muted); font-size: .85rem;
  padding-bottom: .55rem; }
.d2l-cards { display: grid; gap: .75rem; margin: 0 0 2rem;
  grid-template-columns: repeat(auto-fill, minmax(165px, 1fr)); }
.d2l-card { display: flex; flex-direction: column; gap: .1rem;
  padding: .8rem .9rem .75rem; border: 1px solid var(--d2l-rule);
  border-top: 3px solid var(--d2l-blue); border-radius: 4px;
  color: inherit; text-decoration: none; background: #fff;
  transition: border-color .12s, box-shadow .12s, transform .12s; }
.d2l-card:hover { border-color: #bbdefb; border-top-color: var(--d2l-blue-dark);
  box-shadow: 0 4px 14px rgba(13, 71, 161, .08); transform: translateY(-1px); }
.d2l-card-title { font-weight: 600; font-size: 1.02rem; line-height: 1.25;
  margin-bottom: .3rem; }
.d2l-card-meta { color: var(--d2l-muted); font-size: .85rem; }
.d2l-card-fws { color: var(--d2l-blue-dark); font-size: .78rem;
  margin-top: .35rem; }
.d2l-part { margin-top: 2.5rem; scroll-margin-top: 90px; }
.d2l-chapter { scroll-margin-top: 90px; }
.d2l-part-fallback { font-size: .9rem; margin: -.4rem 0 1rem;
  padding: .45rem .75rem; border-left: 3px solid var(--d2l-blue);
  background: var(--d2l-hover); }
.d2l-part > h2 { margin-bottom: .35rem; }
.d2l-part-note { color: var(--d2l-muted); font-size: .9rem;
  margin: 0 0 1rem; }
.d2l-chapter > h3 { margin: 1.6rem 0 .4rem; }
ol.d2l-decks { list-style: none; margin: 0; padding: 0;
  border-top: 1px solid var(--d2l-rule); }
.d2l-deck { display: flex; align-items: center; gap: .75rem;
  border-bottom: 1px solid var(--d2l-rule); margin: 0; }
.d2l-deck:hover { background: var(--d2l-hover); }
.d2l-open { flex: 1 1 auto; display: flex; align-items: baseline;
  gap: .6rem; min-width: 0; padding: .5rem .5rem;
  color: inherit; text-decoration: none; }
.d2l-open:hover .d2l-title { color: var(--d2l-blue-dark);
  text-decoration: underline; text-underline-offset: 3px; }
.d2l-num { flex: 0 0 2.6rem; color: var(--d2l-muted);
  font-variant-numeric: tabular-nums; }
.d2l-title { min-width: 0; }
.d2l-overview .d2l-title { font-style: italic; }
.d2l-fallback { color: var(--d2l-muted); font-size: .8rem;
  white-space: nowrap; }
.d2l-fallback:empty { display: none; }
.d2l-actions { flex: 0 0 auto; display: flex; align-items: center;
  gap: .9rem; padding-right: .5rem; font-size: .82rem; }
.d2l-actions a { color: var(--d2l-muted); text-decoration: none;
  white-space: nowrap; }
.d2l-actions a[href]:hover { color: var(--d2l-blue-dark); }
.d2l-actions .bi { margin-right: .2rem; }
.d2l-dl.d2l-hidden { visibility: hidden; }
.d2l-book .bi { margin: 0; font-size: .95rem; }
@media (max-width: 767.98px) {
  .d2l-deck { flex-wrap: wrap; gap: 0; }
  .d2l-open { flex-basis: 100%; padding-bottom: .1rem; }
  .d2l-actions { padding: 0 .5rem .45rem 3.7rem; }
  .d2l-fw-label { display: none; }
}
</style>"""


SCRIPT = """<script data-cfasync="false">
(function() {
  var KEY = 'quarto-persistent-tabsets-data';
  var NAMES = {pytorch: 'PyTorch', tensorflow: 'TensorFlow', jax: 'JAX',
               mxnet: 'MXNet'};
  var SUFFIX = {pdf: '.pdf', pptx: '.pptx', tex: '-latex.zip'};
  function stored() {
    try {
      var name = JSON.parse(localStorage.getItem(KEY) || '{}').framework;
      for (var fw in NAMES) if (NAMES[fw] === name) return fw;
    } catch (e) {}
    return null;
  }
  function store(fw) {
    try {
      var data = JSON.parse(localStorage.getItem(KEY) || '{}');
      data.framework = NAMES[fw];
      localStorage.setItem(KEY, JSON.stringify(data));
    } catch (e) {}
  }
  function apply(fw) {
    document.querySelectorAll('.d2l-fw-bar .nav-link').forEach(function(a) {
      var on = a.dataset.fw === fw;
      a.classList.toggle('active', on);
      a.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    document.querySelectorAll('.d2l-part').forEach(function(part) {
      var note = part.querySelector('.d2l-part-fallback');
      if (!note) return;
      note.hidden = part.dataset.fws.split(',').indexOf(fw) >= 0;
      note.querySelector('.d2l-fw-name').textContent = NAMES[fw];
    });
    document.querySelectorAll('.d2l-deck').forEach(function(row) {
      var fws = row.dataset.fws.split(',');
      var use = fws.indexOf(fw) >= 0 ? fw : fws[0];
      var base = use + '/' + row.dataset.dir + '/' + row.dataset.stem;
      row.querySelector('.d2l-open').href = base + '.html';
      var part = row.closest('.d2l-part');
      var partHas = part.dataset.fws.split(',').indexOf(fw) >= 0;
      row.querySelector('.d2l-fallback').textContent =
        use === fw || !partHas ? '' : NAMES[use];
      row.querySelectorAll('.d2l-dl').forEach(function(a) {
        var ok = a.dataset.fws.split(',').indexOf(use) >= 0;
        a.classList.toggle('d2l-hidden', !ok);
        if (ok) a.href = base + SUFFIX[a.dataset.kind];
        else a.removeAttribute('href');
      });
    });
  }
  document.querySelectorAll('.d2l-fw-bar .nav-link').forEach(function(a) {
    a.addEventListener('click', function(e) {
      e.preventDefault(); store(a.dataset.fw); apply(a.dataset.fw);
    });
  });
  apply(stored() || 'pytorch');
})();
</script>"""


def _replace_between(html: str, start: str, end: str, new: str) -> str:
    """Replace html[i:j] where i is the end of `start` and j the start of the
    next `end` after it; raise if either marker is missing."""
    i = html.index(start) + len(start)
    j = html.index(end, i)
    return html[:i] + new + html[j:]


def frame_page(frame: str, main: str, toc: str) -> str:
    """Put the landing page into a rendered book page of the same depth."""
    html = re.sub(r'<title>[^<]*</title>',
                  lambda m: '<title>' + html_escape(i18n.t(
                      'slides_index.page_title', book=i18n.t('book.title')))
                  + '</title>', frame, count=1)
    # Main column; the part page's title block is replaced too.
    html = _replace_between(html, '<main class="content" '
                            'id="quarto-document-content">', '</main>',
                            '\n' + main + '\n')
    # Margin column: a table of contents of the parts (Quarto's scrollspy
    # follows #TOC links).
    html = re.sub(r'(<div id="quarto-margin-sidebar"[^>]*>).*?(</div>)',
                  lambda m: m.group(1) + toc + m.group(2), html, count=1,
                  flags=re.S)
    # Nothing in the sidebar is the current page; collapse its parts.
    side_a = html.index('<nav id="quarto-sidebar"')
    side_b = html.index('<div id="quarto-sidebar-glass"', side_a)
    side = html[side_a:side_b]
    side = re.sub(r'(class="[^"]*sidebar-link) active', r'\1', side)
    side = re.sub(r'(class="collapse list-unstyled sidebar-section '
                  r'depth\d+) show"', r'\1"', side)
    side = side.replace('aria-expanded="true"', 'aria-expanded="false"')
    html = html[:side_a] + side + html[side_b:]
    # Page-specific data the landing page does not need (large).
    html = re.sub(r'<script data-cfasync="false">window\.D2L_NOTEBOOKS_'
                  r'MANIFEST=.*?</script>', '', html, count=1, flags=re.S)
    return html.replace('</head>', STYLE + '\n</head>', 1).replace(
        '</body>', SCRIPT + '\n</body>', 1)


def standalone_page(main: str, bootstrap_css: str) -> str:
    """Fallback when no rendered book page exists to supply the frame."""
    return (
        f'<!DOCTYPE html>\n<html lang="{html_escape(i18n.setting("lang"))}"'
        f'{" dir=" + chr(34) + "rtl" + chr(34) if i18n.setting("dir") == "rtl" else ""}>'
        '<head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>' + html_escape(i18n.t('slides_index.page_title',
                                       book=i18n.t('book.title'))) +
        '</title>'
        '<link rel="stylesheet" href="../site_libs/bootstrap/'
        'bootstrap-icons.css">'
        f'<link rel="stylesheet" href="{bootstrap_css}">'
        '<link rel="stylesheet" href="../_d2l-style.css">'
        f'{STYLE}</head><body class="quarto-light">'
        '<main class="content container" style="max-width:900px;'
        f'padding:2rem 1rem">{main}</main>{SCRIPT}</body></html>\n')


def find_frame(source: Path, book: Path):
    """The first rendered part page: one directory deep, like slides/."""
    quarto = source / '_quarto.yml'
    if quarto.exists():
        for m in re.finditer(r'- part:\s*(\S+)/index\.qmd',
                             quarto.read_text(encoding='utf-8')):
            page = book / m.group(1) / 'index.html'
            if page.exists():
                return page
    return None


def bootstrap_href(book: Path) -> str:
    # Quarto names the Bootstrap bundle by content hash; take the newest.
    bs_dir = book / 'site_libs/bootstrap'
    candidates = sorted(bs_dir.glob('bootstrap-*.min.css'),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True) if bs_dir.exists() else []
    return (f'../site_libs/bootstrap/{candidates[0].name}' if candidates
            else '../site_libs/bootstrap/bootstrap.min.css')


def write_landing(idx: dict, out: Path, source: Path, book: Path,
                  frame: Path = None):
    main, toc = render_content(idx)
    frame = frame or find_frame(source, book)
    if frame is not None and frame.exists():
        html = frame_page(frame.read_text(encoding='utf-8'), main, toc)
        how = f'in the frame of {frame}'
    else:
        html = standalone_page(main, bootstrap_href(book))
        how = 'standalone (no rendered book page for the frame)'
    out.write_text(html, encoding='utf-8')
    n_chap = sum(len(p['chapters']) for p in idx['parts'])
    print(f'wrote {out} — {len(idx["parts"])} parts, {n_chap} chapters, '
          f'{how}')


def write_data_include(data_file: Path, manifest: dict, exports: dict = None):
    """Write `_d2l-slides-data.html` for Quarto's `include-after-body`.

    Defines `window.D2L_SLIDES_MANIFEST` so the in-book TOC slides
    block can determine which framework variants exist for the page
    being viewed, and `window.D2L_SLIDES_EXPORTS`, {kind: {page: [fws]}}
    for kind in pdf / pptx / tex, the frameworks whose deck also has
    that downloadable version.
    """
    payload = json.dumps(manifest, ensure_ascii=False, separators=(',', ':'))
    exports_payload = json.dumps(exports or {}, ensure_ascii=False,
                                 separators=(',', ':'))
    content = (
        '<script>\n'
        '// Auto-generated by tools/build_slides_index.py — do not edit.\n'
        f'window.D2L_SLIDES_MANIFEST = {payload};\n'
        f'window.D2L_SLIDES_EXPORTS = {exports_payload};\n'
        '</script>\n'
    )
    data_file.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slides-dir', type=Path, default=Path('_slides'))
    parser.add_argument('--source', type=Path, default=Path('.'))
    parser.add_argument('--book', type=Path, default=Path('_book'))
    parser.add_argument('--data-file', type=Path,
                        default=Path('_d2l-slides-data.html'),
                        help='Where to write the JS manifest include')
    parser.add_argument('--landing-only', action='store_true',
                        help='Write only the landing page (to --out)')
    parser.add_argument('--out', type=Path, default=None,
                        help='Landing page path (default: '
                             '<slides-dir>/index.html)')
    parser.add_argument('--frame', type=Path, default=None,
                        help='Rendered book page whose frame the landing '
                             'page uses (default: the first part page)')
    args = parser.parse_args()

    if args.slides_dir.exists():
        idx, manifest = build_index(args.slides_dir, args.source)
    else:
        idx, manifest = {'parts': [], 'exports': {}}, {}
    out = args.out or args.slides_dir / 'index.html'

    if not args.landing_only:
        # Always emit the data include — even if empty — so the book can
        # build cleanly without slides being present.
        exports = idx.get('exports', {})
        write_data_include(args.data_file, manifest, exports)
        print(f'wrote {args.data_file} — {len(manifest)} pages have slides; '
              'with ' + ', '.join(f'{k}: {len(v)}'
                                  for k, v in exports.items()))

    if not idx['parts']:
        print(f'no decks under {args.slides_dir}; skipped landing page')
        return
    if not args.landing_only:
        # A manifest.json next to the landing page for any external
        # consumer (e.g. CI sanity checks, tooling).
        (args.slides_dir / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8')
    write_landing(idx, out, args.source, args.book, args.frame)


if __name__ == '__main__':
    main()
