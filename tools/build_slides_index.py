#!/usr/bin/env python3
"""Build slide-deck integration artifacts:

  * `_slides/index.html`        — landing page listing every deck, with a
                                  navbar matching the book's. Linked from
                                  the book navbar (`Slides`).
  * `_slides/manifest.json`     — `{"chapter_X/file": ["pytorch", "jax"], ...}`
                                  so external consumers can introspect which
                                  decks exist.
  * `_d2l-slides-data.html`     — `<script>window.D2L_SLIDES_MANIFEST=…;</script>`
                                  included into every book page via the
                                  `include-after-body:` Quarto hook so the
                                  right-sidebar TOC slides-button knows which
                                  framework variants exist for the page.

Walks `_slides/<fw>/chapter_*/*.html` for each framework. If `_slides/` is
missing or empty, emits a stub data file with an empty manifest so the
book still builds cleanly without slides present.

Usage:
    python tools/build_slides_index.py [--slides-dir _slides]
                                       [--source .]
                                       [--data-file _d2l-slides-data.html]
"""

import argparse
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import CHAPTER_NUMBERING, FRAMEWORK_DISPLAY


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


def build_index(slides_dir: Path, source: Path):
    chapters = {}
    chapter_order = []
    chapter_titles = {}
    manifest = {}

    for rel, _ in CHAPTER_NUMBERING.items():
        rel_path = Path(rel)
        chap = rel_path.parent.name
        if chap not in chapters:
            chapters[chap] = []
            chapter_order.append(chap)
            idx_md = source / chap / 'index.md'
            if idx_md.exists():
                chapter_titles[chap] = extract_h1(idx_md)
            else:
                chapter_titles[chap] = (chap.replace('chapter_', '')
                                            .replace('-', ' ').title())
        if rel_path.stem == 'index':
            continue
        stem = rel_path.stem
        available = []
        for fw in FRAMEWORKS:
            html = slides_dir / fw / chap / f'{stem}.html'
            if html.exists():
                available.append(fw)
        if not available:
            continue
        title = extract_h1(source / rel)
        chapters[chap].append({
            'stem': stem,
            'title': title,
            'frameworks': available,
        })
        manifest[f'{chap}/{stem}'] = available

    chapter_order = [c for c in chapter_order if chapters[c]]

    return {
        'chapters': [
            {'dir': chap,
             'title': chapter_titles[chap],
             'decks': chapters[chap]}
            for chap in chapter_order
        ],
    }, manifest


# ──────────────────────────────────────────────────────────
# Landing page
# ──────────────────────────────────────────────────────────

LANDING_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Slides — Dive into Deep Learning</title>
<style>
:root {
  --blue: #2196F3;
  --blue-dark: #1976D2;
  --blue-darker: #0D47A1;
  --blue-light: #BBDEFB;
  --bg: #fafbfc;
  --surface: #ffffff;
  --border: #e0e0e0;
  --text: #1a1a1a;
  --text-dim: #666;
  --text-muted: #999;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: 'Source Serif 4', Georgia, serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}

/* Navbar — matches the book's Quarto navbar (cosmo + custom blue). */
.d2l-navbar {
  background: var(--blue);
  color: white;
  padding: 0.5rem 1rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.08);
  font-family: 'Source Sans 3', system-ui, -apple-system, sans-serif;
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  min-height: 3.5rem;
}

.d2l-navbar-brand {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  color: white;
  text-decoration: none;
  font-weight: 500;
  font-size: 1.1rem;
  margin: 0 auto;
}
.d2l-navbar-brand:hover { color: white; opacity: 0.92; }
.d2l-navbar-brand .navbar-subtitle {
  font-weight: 300;
  opacity: 0.85;
  font-size: 0.95rem;
}

.d2l-navbar-nav {
  display: flex;
  gap: 0.25rem;
  align-items: center;
  list-style: none;
  margin: 0 0 0 auto;
  padding: 0;
}

.d2l-navbar-nav a {
  color: white;
  text-decoration: none;
  padding: 0.4rem 0.75rem;
  border-radius: 3px;
  font-size: 0.95rem;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  opacity: 0.92;
}
.d2l-navbar-nav a:hover {
  background: rgba(255,255,255,0.15);
  opacity: 1;
}

.d2l-navbar-nav a.active {
  background: rgba(255,255,255,0.18);
  font-weight: 500;
  opacity: 1;
}

/* Sub-bar holding the framework picker */
.d2l-subbar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0.5rem 1rem;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  font-family: 'Source Sans 3', system-ui, sans-serif;
  font-size: 0.9rem;
  gap: 0.5rem;
  color: var(--text-dim);
}

.d2l-subbar select {
  font: inherit;
  padding: 0.25rem 0.5rem;
  border: 1px solid var(--border);
  border-radius: 3px;
  background: white;
  color: var(--text);
  cursor: pointer;
}

main {
  max-width: 950px;
  margin: 1.75rem auto;
  padding: 0 1.5rem;
}

.intro {
  background: var(--surface);
  border-left: 4px solid var(--blue);
  padding: 1rem 1.25rem;
  margin-bottom: 2rem;
  border-radius: 0 4px 4px 0;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  font-size: 0.95rem;
}
.intro p { margin: 0.4rem 0; color: var(--text-dim); }
.intro p:first-child { margin-top: 0; }
.intro p:last-child { margin-bottom: 0; }
.intro a { color: var(--blue-dark); }
.intro kbd {
  background: #eee;
  border: 1px solid #ccc;
  border-radius: 3px;
  padding: 0 0.3rem;
  font-family: Inconsolata, monospace;
  font-size: 0.85em;
}

.chapter { margin-bottom: 2.25rem; }

.chapter h2 {
  font-family: 'Source Sans 3', system-ui, sans-serif;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--blue-dark);
  margin: 0 0 0.5rem 0;
  padding-bottom: 0.35rem;
  border-bottom: 2px solid var(--blue-light);
}

.deck-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.2rem;
}

.deck {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  border-radius: 3px;
  background: var(--surface);
  border: 1px solid transparent;
  transition: background 0.1s, border-color 0.1s;
}
.deck:hover {
  background: white;
  border-color: var(--blue-light);
}
.deck.unavailable { background: transparent; }

.deck-main {
  flex-grow: 1;
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  padding: 0.5rem 0.75rem;
  text-decoration: none;
  color: var(--text);
  border-radius: 3px;
}
.deck-main:hover { color: var(--blue-darker); }
.deck.unavailable .deck-main {
  color: var(--text-muted);
  pointer-events: none;
}
.deck.unavailable .deck-title {
  text-decoration: line-through;
  text-decoration-color: var(--text-muted);
  text-decoration-thickness: 1px;
}

.deck-num {
  font-family: 'Source Sans 3', system-ui, sans-serif;
  font-size: 0.85rem;
  color: var(--text-muted);
  min-width: 2.5rem;
  flex-shrink: 0;
}
.deck-title { flex-grow: 1; }

.fw-badges {
  display: flex;
  gap: 0.2rem;
  font-family: 'Source Sans 3', system-ui, sans-serif;
  font-size: 0.7rem;
  padding: 0.5rem 0.75rem 0.5rem 0;
  flex-shrink: 0;
}
.fw-badges .badge {
  display: inline-block;
  padding: 0.1rem 0.45rem;
  border-radius: 2px;
  background: var(--blue-light);
  color: var(--blue-dark);
  font-weight: 500;
  letter-spacing: 0.02em;
  text-decoration: none;
  cursor: pointer;
  transition: background 0.1s, color 0.1s, transform 0.1s;
  outline-offset: 1px;
}
.fw-badges a.badge:hover {
  background: var(--blue-dark);
  color: white;
  transform: translateY(-1px);
}
.fw-badges .badge.absent {
  background: #f0f0f0;
  color: #bbb;
  cursor: default;
}

footer {
  text-align: center;
  margin: 3rem auto 1.5rem;
  font-family: 'Source Sans 3', system-ui, sans-serif;
  font-size: 0.85rem;
  color: var(--text-muted);
}

@media (max-width: 720px) {
  .d2l-navbar { gap: 0.5rem; }
  .d2l-navbar-brand { margin: 0; flex-grow: 1; }
  .d2l-navbar-nav { width: 100%; justify-content: center; flex-wrap: wrap; }
  main { padding: 0 1rem; }
  .deck-main { padding: 0.55rem 0.6rem; }
}
</style>
</head>
<body>
<nav class="d2l-navbar">
  <a class="d2l-navbar-brand" href="../index.html">
    <span>Dive into Deep Learning</span>
    <span class="navbar-subtitle">— Slides</span>
  </a>
  <ul class="d2l-navbar-nav">
    <li><a href="../index.html">Book</a></li>
    <li><a href="./index.html" class="active">Slides</a></li>
    <li><a href="https://courses.d2l.ai">Courses</a></li>
    <li><a href="https://github.com/d2l-ai/d2l-en" title="GitHub">GitHub</a></li>
    <li><a href="https://discuss.d2l.ai">Discuss</a></li>
    <li><a href="https://d2l.ai/d2l-en.pdf">PDF</a></li>
  </ul>
</nav>

<div class="d2l-subbar">
  <label for="fw">Framework:</label>
  <select id="fw">
    <option value="pytorch">PyTorch</option>
    <option value="tensorflow">TensorFlow</option>
    <option value="jax">JAX</option>
    <option value="mxnet">MXNet</option>
  </select>
</div>

<main>
  <section class="intro">
    <p><kbd>→</kbd>/<kbd>←</kbd> to navigate slides,
       <kbd>?</kbd> for help, <kbd>S</kbd> for speaker notes.
       Click a row to open in the selected framework, or click a
       badge (PT&nbsp;/&nbsp;TF&nbsp;/&nbsp;JAX&nbsp;/&nbsp;MX) to
       open that framework's variant.</p>
  </section>

  __CONTENT__
</main>

<footer>
  Generated from <code>_slides/&lt;fw&gt;/</code>.
</footer>

<script>
const DATA = __DATA__;

const fwSelect = document.getElementById('fw');

function persist(fw) {
  try {
    const data = JSON.parse(localStorage.getItem('quarto-persistent-tabsets-data') || '{}');
    const display = {pytorch:'PyTorch', tensorflow:'TensorFlow', jax:'JAX', mxnet:'MXNet'}[fw];
    if (display) {
      data['framework'] = display;
      localStorage.setItem('quarto-persistent-tabsets-data', JSON.stringify(data));
    }
  } catch (_) {}
}

function restore() {
  try {
    const data = JSON.parse(localStorage.getItem('quarto-persistent-tabsets-data') || '{}');
    const display = data['framework'];
    return {PyTorch:'pytorch', TensorFlow:'tensorflow', JAX:'jax', MXNet:'mxnet'}[display] || null;
  } catch (_) { return null; }
}

function applyFramework(fw) {
  document.querySelectorAll('.deck').forEach(deck => {
    const fws = deck.dataset.fws.split(',');
    const main = deck.querySelector('.deck-main');
    if (fws.includes(fw)) {
      deck.classList.remove('unavailable');
      main.href = `${fw}/${deck.dataset.dir}/${deck.dataset.stem}.html`;
    } else {
      deck.classList.add('unavailable');
      main.removeAttribute('href');
    }
  });
  document.querySelectorAll('.fw-badges .badge').forEach(b => {
    if (b.dataset.fw === fw) b.style.outline = '2px solid var(--blue-dark)';
    else b.style.outline = '';
  });
  persist(fw);
}

const initial = restore() || 'pytorch';
fwSelect.value = initial;
applyFramework(initial);
fwSelect.addEventListener('change', e => applyFramework(e.target.value));
</script>
</body>
</html>
"""


def render_html(idx: dict) -> str:
    parts = []
    short_names = {'pytorch': 'PT', 'tensorflow': 'TF',
                   'jax': 'JAX', 'mxnet': 'MX'}
    for chap in idx['chapters']:
        parts.append('  <section class="chapter">')
        parts.append(f'    <h2>{html_escape(chap["title"])}</h2>')
        parts.append('    <ul class="deck-list">')
        for i, deck in enumerate(chap['decks'], start=1):
            fws_attr = ','.join(deck['frameworks'])
            stem = deck['stem']
            chap_dir = chap['dir']
            badges = []
            for fw in FRAMEWORKS:
                short = short_names[fw]
                title = FRAMEWORK_DISPLAY.get(fw, fw)
                if fw in deck['frameworks']:
                    href = f'{fw}/{chap_dir}/{stem}.html'
                    badges.append(
                        f'<a class="badge" data-fw="{fw}" '
                        f'href="{href}" title="Open {title} variant">'
                        f'{short}</a>')
                else:
                    badges.append(
                        f'<span class="badge absent" data-fw="{fw}" '
                        f'title="No {title} deck for this section">'
                        f'{short}</span>')
            parts.append(
                f'      <li class="deck" '
                f'data-dir="{chap_dir}" data-stem="{stem}" '
                f'data-fws="{fws_attr}">'
                f'<a class="deck-main" href="#">'
                f'<span class="deck-num">{i}.</span>'
                f'<span class="deck-title">{html_escape(deck["title"])}</span>'
                f'</a>'
                f'<span class="fw-badges">{"".join(badges)}</span>'
                f'</li>')
        parts.append('    </ul>')
        parts.append('  </section>')
    return '\n'.join(parts)


def html_escape(s: str) -> str:
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;')
             .replace('"', '&quot;'))


def write_data_include(data_file: Path, manifest: dict):
    """Write `_d2l-slides-data.html` for Quarto's `include-after-body`.

    Defines `window.D2L_SLIDES_MANIFEST` so the in-book TOC slides
    button can determine which framework variants exist for the
    page being viewed.
    """
    payload = json.dumps(manifest, ensure_ascii=False, separators=(',', ':'))
    content = (
        '<script>\n'
        '// Auto-generated by tools/build_slides_index.py — do not edit.\n'
        f'window.D2L_SLIDES_MANIFEST = {payload};\n'
        '</script>\n'
    )
    data_file.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slides-dir', type=Path, default=Path('_slides'))
    parser.add_argument('--source', type=Path, default=Path('.'))
    parser.add_argument('--data-file', type=Path,
                        default=Path('_d2l-slides-data.html'),
                        help='Where to write the JS manifest include')
    args = parser.parse_args()

    if args.slides_dir.exists():
        idx, manifest = build_index(args.slides_dir, args.source)
    else:
        idx, manifest = {'chapters': []}, {}

    # Always emit the data include — even if empty — so the book can
    # build cleanly without slides being present.
    write_data_include(args.data_file, manifest)
    n_pages = len(manifest)
    print(f'wrote {args.data_file} — {n_pages} pages have slides')

    if not idx['chapters']:
        print(f'no decks under {args.slides_dir}; skipped landing page')
        return

    body = render_html(idx)
    html = (LANDING_TEMPLATE
            .replace('__CONTENT__', body)
            .replace('__DATA__', json.dumps(idx, ensure_ascii=False)))
    out = args.slides_dir / 'index.html'
    # Also emit a manifest.json next to the landing page for any
    # external consumer (e.g. CI sanity checks, tooling).
    (args.slides_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    out.write_text(html, encoding='utf-8')
    n_decks = sum(len(c['decks']) for c in idx['chapters'])
    print(f'wrote {out} — {len(idx["chapters"])} chapters, {n_decks} decks')


if __name__ == '__main__':
    main()
