#!/usr/bin/env python3
"""Generate `_slides/index.html` — a landing page for the slide decks.

Walks `_slides/<fw>/chapter_*/*.html` for each framework, builds a
chapter-grouped list of decks with framework variants, and writes a
self-contained HTML page styled to match the book theme. A framework
selector at the top toggles all links to point at the chosen
framework's deck.

Usage:
    python tools/build_slides_index.py [--slides-dir _slides]
                                       [--source .]
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


def chapter_label(rel: str, numbering) -> str:
    """e.g. 'chapter_linear-regression' → '3 Linear Regression'."""
    nums = numbering
    label = rel.replace('chapter_', '').replace('-', ' ').title()
    if nums is None:
        return label
    return f'{".".join(str(n) for n in nums)} {label}'


def build_index(slides_dir: Path, source: Path):
    # Group source files by chapter dir, in CHAPTER_NUMBERING order.
    chapters = {}
    chapter_order = []
    chapter_titles = {}
    for rel, nums in CHAPTER_NUMBERING.items():
        rel_path = Path(rel)
        chap = rel_path.parent.name
        if chap not in chapters:
            chapters[chap] = []
            chapter_order.append(chap)
            # Use the chapter's index.md H1 if present, else humanize the dir
            idx_md = source / chap / 'index.md'
            if idx_md.exists():
                chapter_titles[chap] = extract_h1(idx_md)
            else:
                chapter_titles[chap] = chap.replace('chapter_', '').replace('-', ' ').title()
        # Skip the index entry itself in the per-section list
        if rel_path.stem == 'index':
            continue
        # Determine which frameworks have this deck
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

    # Drop chapters that have no decks (e.g. preface)
    chapter_order = [c for c in chapter_order if chapters[c]]

    return {
        'chapters': [
            {
                'dir': chap,
                'title': chapter_titles[chap],
                'decks': chapters[chap],
            }
            for chap in chapter_order
        ],
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dive into Deep Learning — Slides</title>
<style>
:root {
  --blue: #2196F3;
  --blue-dark: #1976D2;
  --blue-light: #BBDEFB;
  --orange: #FF5722;
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

header {
  position: sticky;
  top: 0;
  z-index: 10;
  background: var(--blue);
  color: white;
  padding: 0.75rem 1.25rem;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  display: flex;
  align-items: center;
  gap: 1.5rem;
  flex-wrap: wrap;
}

header h1 {
  margin: 0;
  font-family: 'Source Sans 3', system-ui, sans-serif;
  font-size: 1.25rem;
  font-weight: 500;
  flex-grow: 1;
}

header h1 .subtitle {
  font-weight: 300;
  opacity: 0.85;
}

.fw-picker {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-family: 'Source Sans 3', system-ui, sans-serif;
}

.fw-picker label {
  font-size: 0.9rem;
  opacity: 0.9;
}

.fw-picker select {
  font: inherit;
  font-size: 0.95rem;
  padding: 0.3rem 0.6rem;
  border: none;
  border-radius: 3px;
  background: white;
  color: var(--text);
  cursor: pointer;
}

main {
  max-width: 950px;
  margin: 2rem auto;
  padding: 0 1.5rem;
}

.intro {
  background: var(--surface);
  border-left: 4px solid var(--blue);
  padding: 1.25rem 1.5rem;
  margin-bottom: 2rem;
  border-radius: 0 4px 4px 0;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

.intro p { margin: 0.5rem 0; color: var(--text-dim); }
.intro p:first-child { margin-top: 0; }
.intro p:last-child { margin-bottom: 0; }

.intro a { color: var(--blue-dark); }

.chapter {
  margin-bottom: 2.25rem;
}

.chapter h2 {
  font-family: 'Source Sans 3', system-ui, sans-serif;
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--blue-dark);
  margin: 0 0 0.75rem 0;
  padding-bottom: 0.4rem;
  border-bottom: 2px solid var(--blue-light);
}

.deck-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.25rem;
}

.deck {
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  padding: 0;
  border-radius: 3px;
  background: var(--surface);
  border: 1px solid transparent;
  transition: background 0.1s, border-color 0.1s;
}

.deck:hover {
  background: white;
  border-color: var(--blue-light);
}

.deck.unavailable {
  background: transparent;
}

.deck-main {
  flex-grow: 1;
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  padding: 0.55rem 0.75rem;
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
  padding: 0.55rem 0.75rem 0.55rem 0;
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
  header { gap: 0.75rem; }
  header h1 { font-size: 1.05rem; flex-basis: 100%; }
  main { padding: 0 1rem; }
  .deck { padding: 0.6rem; }
}
</style>
</head>
<body>
<header>
  <h1>Dive into Deep Learning <span class="subtitle">— Slides</span></h1>
  <div class="fw-picker">
    <label for="fw">Framework:</label>
    <select id="fw">
      <option value="pytorch">PyTorch</option>
      <option value="tensorflow">TensorFlow</option>
      <option value="jax">JAX</option>
      <option value="mxnet">MXNet</option>
    </select>
  </div>
</header>

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
  Source on the <code>slides</code> branch.
</footer>

<script>
const DATA = __DATA__;

const fwSelect = document.getElementById('fw');

function persist(fw) {
  try { localStorage.setItem('d2l-slides-fw', fw); } catch (_) {}
}
function restore() {
  try { return localStorage.getItem('d2l-slides-fw'); } catch (_) { return null; }
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slides-dir', type=Path, default=Path('_slides'))
    parser.add_argument('--source', type=Path, default=Path('.'))
    args = parser.parse_args()

    idx = build_index(args.slides_dir, args.source)
    body = render_html(idx)
    html = (HTML_TEMPLATE
            .replace('__CONTENT__', body)
            .replace('__DATA__', json.dumps(idx, ensure_ascii=False)))
    out = args.slides_dir / 'index.html'
    out.write_text(html, encoding='utf-8')
    n_decks = sum(len(c['decks']) for c in idx['chapters'])
    print(f'wrote {out} — {len(idx["chapters"])} chapters, {n_decks} decks')


if __name__ == '__main__':
    main()
