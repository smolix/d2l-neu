#!/usr/bin/env python3
"""UI string catalog and language switch of the book build.

Every piece of text the build emits outside the chapter sources (navbar,
injected HTML/JS, slide index and templates, PDF front matter, notebook
downloads) lives in `i18n/en.yml`, together with the per-language settings
(Quarto `lang`, text direction, fonts, hosts, forum). A localized build selects
another catalog with two environment variables:

    D2L_LANG=<code>          ISO 639-1 code (default: en)
    D2L_LOCALE_FILE=<path>   YAML file with the schema of i18n/en.yml; its keys
                             override English key by key, a missing key stays
                             English. Without it, i18n/<code>.yml is used if it
                             exists.

Python emitters `import i18n` and call `i18n.t(key, **fill)`. The HTML/JS
includes are templates: `{{t:key}}` (HTML-escaped) and `{{tjs:key}}` (escaped
for a JavaScript string literal) placeholders, rendered into `_i18n/` by

    python3 tools/i18n.py render

which also localizes `_quarto.yml` in place (navbar labels, `lang`, `dir`, the
language menu's fallback links, Quarto language overrides). With English
selected every rendered file is byte-identical to its template's English text
and `_quarto.yml` is the committed one, so an English build is unchanged.

Other commands:

    python3 tools/i18n.py check FILE    validate a locale file against en.yml
    python3 tools/i18n.py show [KEY]    print the effective catalog (or one key)

See docs/build-guide.md §4.14 for the localized-build recipe.
"""
from __future__ import annotations

import argparse
import functools
import html
import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / 'i18n' / 'en.yml'
OUT_DIR = '_i18n'

# HTML includes rendered from templates into _i18n/ (same file names).
TEMPLATES = ('_d2l-favicon.html', '_d2l-tabs.html', '_d2l-notebooks.html',
             '_d2l-mobilenav.html', '_d2l-languages.html',
             '_d2l-slides-overlay.html')

# Keys whose values are not strings (settings); every other key is a UI string.
SETTINGS = ('code', 'lang', 'dir', 'site.', 'forum.', 'fonts.', 'pdf.lang',
            'quarto.')

_PLACEHOLDER = re.compile(r'\{([a-z_]+)\}')
_TEMPLATE = re.compile(r'\{\{(t|tjs):([A-Za-z0-9_.\-]+)\}\}')


class CatalogError(KeyError):
    pass


def _flatten(d, prefix=''):
    """Nested mappings → dotted keys; lists and scalars are values."""
    out = {}
    for k, v in (d or {}).items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def _load_yaml(path):
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f'i18n: {path}: expected a mapping at top level')
    return _flatten(data)


def is_setting(key: str) -> bool:
    return any(key == s or (s.endswith('.') and key.startswith(s))
               for s in SETTINGS)


@functools.lru_cache(maxsize=None)
def english() -> dict:
    return _load_yaml(CATALOG)


def locale_path():
    """The locale file in use, or None for English."""
    path = os.environ.get('D2L_LOCALE_FILE')
    if path:
        p = Path(path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise SystemExit(f'i18n: D2L_LOCALE_FILE={path} does not exist')
        return p
    code = os.environ.get('D2L_LANG') or 'en'
    cand = REPO / 'i18n' / f'{code}.yml'
    return cand if code != 'en' and cand.exists() else None


@functools.lru_cache(maxsize=None)
def catalog() -> dict:
    """The effective catalog: English overridden by the locale file."""
    base = english()
    cat = dict(base)
    path = locale_path()
    over = _load_yaml(path) if path else {}
    for k in sorted(over):
        if k not in base and not k.startswith('quarto.'):
            print(f'i18n: {path}: unknown key {k!r} (ignored)', file=sys.stderr)
    cat.update({k: v for k, v in over.items()
                if k in base or k.startswith('quarto.')})
    env_code = os.environ.get('D2L_LANG')
    if env_code:
        if 'code' in over and over['code'] != env_code:
            print(f'i18n: D2L_LANG={env_code} but {path} has code '
                  f'{over["code"]!r}; using {env_code}', file=sys.stderr)
        cat['code'] = env_code
    if 'lang' not in over and cat['code'] != base['code']:
        cat['lang'] = cat['code']
    return cat


def reset():
    """Forget cached catalogs (tests change the environment)."""
    english.cache_clear()
    catalog.cache_clear()


def code() -> str:
    return catalog()['code']


def is_english() -> bool:
    """True when every key has its English value."""
    return catalog() == english()


def setting(key: str):
    cat = catalog()
    if key not in cat:
        raise CatalogError(f'i18n: no key {key!r} in {CATALOG}')
    return cat[key]


def fill(text: str, **kw) -> str:
    """Replace each {name} for a given name; other braces stay as they are."""
    return _PLACEHOLDER.sub(
        lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0),
        text)


def t(key: str, **kw) -> str:
    """The UI string `key`, with its {name} placeholders filled from kw."""
    value = setting(key)
    if not isinstance(value, str):
        raise CatalogError(f'i18n: {key!r} is not a string')
    return fill(value, **kw)


def tn(key: str, n: int, **kw) -> str:
    """A count: `key.one` for n == 1, else `key.other`; {n} is filled."""
    return t(f'{key}.{"one" if n == 1 else "other"}', n=n, **kw)


def en(key: str, **kw) -> str:
    """The English value of `key` (for recognising English source text)."""
    return fill(english()[key], **kw)


def variants(key: str) -> tuple:
    """The English and the localized value of `key` (one if they agree).
    For detecting structural text in sources that may be in either."""
    a, b = en(key), t(key)
    return (a,) if a == b else (a, b)


# ── Quarto's own language strings ─────────────────────────────────────────

_QUARTO_EN = {
    'crossref-ch-prefix': 'Chapter', 'crossref-sec-prefix': 'Section',
    'crossref-fig-title': 'Figure', 'crossref-fig-prefix': 'Figure',
    'crossref-tbl-title': 'Table', 'crossref-tbl-prefix': 'Table',
    'crossref-lst-title': 'Listing', 'crossref-lst-prefix': 'Listing',
    'crossref-eq-prefix': 'Equation', 'section-title-references': 'References',
}


def _quarto_language_dir():
    for d in sorted(REPO.glob('.venv-build/lib/python*/site-packages/'
                              'quarto_cli/share/language')):
        return d
    return None


@functools.lru_cache(maxsize=None)
def _quarto_language() -> dict:
    d = _quarto_language_dir()
    terms = dict(_QUARTO_EN)
    if d is None:
        return terms
    base = d / '_language.yml'
    if base.exists():
        terms.update(yaml.safe_load(base.read_text(encoding='utf-8')) or {})
    lang = catalog()['lang']
    for name in (lang, lang.split('-')[0]):
        f = d / f'_language-{name}.yml'
        if name != 'en' and f.exists():
            terms.update(yaml.safe_load(f.read_text(encoding='utf-8')) or {})
            break
    return terms


def quarto_term(key: str) -> str:
    """The term Quarto renders for `key` (e.g. crossref-ch-prefix) in the
    selected language: the locale file's `quarto.<key>` override, else
    Quarto's language file for `lang`, else English."""
    over = catalog().get(f'quarto.{key}')
    if over is not None:
        return over
    return str(_quarto_language().get(key, _QUARTO_EN.get(key, key)))


def quarto_overrides() -> dict:
    """{quarto language key: value} the locale file sets."""
    return {k[len('quarto.'):]: v for k, v in catalog().items()
            if k.startswith('quarto.')}


# ── Escaping and templates ────────────────────────────────────────────────

def html_text(s: str) -> str:
    """Escape for HTML text and double-quoted attributes."""
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;').replace('"', '&quot;'))


def js_text(s: str) -> str:
    """Escape for the inside of a '…' or "…" JavaScript string literal."""
    return (s.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
             .replace('\n', '\\n').replace('</', '<\\/'))


def render_template(text: str, name: str = '<template>') -> str:
    def sub(m):
        kind, key = m.group(1), m.group(2)
        try:
            value = t(key)
        except KeyError as e:
            raise SystemExit(f'i18n: {name}: {e}')
        return html_text(value) if kind == 't' else js_text(value)
    return _TEMPLATE.sub(sub, text)


def _head_extras() -> str:
    """Extra <head> markup for a locale: fonts and text direction."""
    cat, base = catalog(), english()
    out = []
    google = cat.get('fonts.html.google') or []
    if google:
        fams = '&'.join('family=' + g.replace(' ', '+') for g in google)
        out.append('<link rel="stylesheet" href="https://fonts.googleapis.com/'
                   f'css2?{fams}&display=swap">')
    css = []
    if cat['fonts.html.sans'] != base['fonts.html.sans']:
        css.append(f'body, .d2l-landing {{ font-family: {cat["fonts.html.sans"]}; }}')
    if cat['fonts.html.mono'] != base['fonts.html.mono']:
        css.append(f'code, pre, kbd, samp {{ font-family: {cat["fonts.html.mono"]}; }}')
    if css:
        out.append('<style>\n' + '\n'.join(css) + '\n</style>')
    return ''.join(x + '\n' for x in out)


def not_found_page() -> str:
    """The site's 404.qmd: says the page is missing, then goes home. The
    language menu links to the same path in another language, which lands
    here when that language lacks the page (DESIGN §8 of d2l-localization)."""
    title = json.dumps(t('notfound.title'), ensure_ascii=False)
    return (
        '---\n'
        '# Generated by tools/i18n.py render from i18n/en.yml (gitignored).\n'
        f'title: {title}\n'
        'number-sections: false\n'
        'toc: false\n'
        '---\n\n'
        f'{t("notfound.text")}\n\n'
        f'[{t("notfound.link")}](/)\n\n'
        '```{=html}\n'
        '<script>setTimeout(function() { location.replace("/"); }, 3000);'
        '</script>\n'
        '```\n')


def _write_if_changed(dst: Path, text: str) -> bool:
    if dst.exists() and dst.read_text(encoding='utf-8') == text:
        return False
    dst.write_text(text, encoding='utf-8')
    return True


def render_includes(root: Path = REPO, verbose: bool = True) -> list:
    """Render the HTML templates into root/_i18n/, the effective catalog into
    root/_i18n/catalog.json and the 404 page into root/404.qmd; write only
    changed files. Returns the paths written."""
    out_dir = root / OUT_DIR
    out_dir.mkdir(exist_ok=True)
    outputs = {}
    for name in TEMPLATES:
        src = root / name
        text = render_template(src.read_text(encoding='utf-8'), name)
        if name == '_d2l-favicon.html':
            text += _head_extras()
        outputs[out_dir / name] = text
    outputs[out_dir / 'catalog.json'] = json.dumps(
        catalog(), ensure_ascii=False, indent=1, sort_keys=True) + '\n'
    outputs[root / '404.qmd'] = not_found_page()
    written = [p for p, text in outputs.items() if _write_if_changed(p, text)]
    if verbose:
        print(f'i18n: {code()} — rendered {len(TEMPLATES)} includes, '
              f'catalog.json and 404.qmd ({len(written)} changed)')
    return written


# ── _quarto.yml ───────────────────────────────────────────────────────────
#
# The navbar labels are edited in place: each catalog key names one scalar
# in the YAML (found by the href or menu it belongs to), and only scalars
# whose value differs are rewritten, so the English catalog leaves the file
# untouched and a later English run restores the English labels. Settings
# that English does not set (lang, dir, Quarto language overrides) live in a
# delimited block at the end of the file, removed for English.

_BLOCK_START = '# >>> i18n: generated by tools/i18n.py render; English removes it\n'
_BLOCK_END = '# <<< i18n\n'


def _node_get(node, key):
    if isinstance(node, yaml.MappingNode):
        for k, v in node.value:
            if k.value == key:
                return v
    return None


def _menu_hrefs(item):
    menu = _node_get(item, 'menu')
    if not isinstance(menu, yaml.SequenceNode):
        return []
    return [(_node_get(m, 'href').value if _node_get(m, 'href') else None, m)
            for m in menu.value]


def _quarto_targets(root_node):
    """[(catalog key, ScalarNode)] for the localizable _quarto.yml scalars."""
    out = []
    book = _node_get(root_node, 'book')
    if book is None:
        return out
    title = _node_get(book, 'title')
    if title is not None:
        out.append(('book.title', title))
    right = _node_get(_node_get(book, 'navbar'), 'right')
    by_href = {'slides/index.html': 'nav.slides',
               'https://courses.d2l.ai': 'nav.courses',
               'https://github.com/smolix/d2l-neu': 'nav.github',
               'https://d2l.discourse.group': 'nav.discuss'}
    parts = {'basics': 'nav.part_basics', 'advanced': 'nav.part_advanced',
             'language-models': 'nav.part_language_models',
             'mathematics-for-deep-learning': 'nav.part_mathematics'}
    for item in (right.value if isinstance(right, yaml.SequenceNode) else []):
        href = _node_get(item, 'href')
        text = _node_get(item, 'text')
        if href is not None and href.value in by_href and text is not None:
            out.append((by_href[href.value], text))
            continue
        hrefs = _menu_hrefs(item)
        if any(h and h.startswith('/pdf/') for h, _ in hrefs):
            if text is not None:
                out.append(('nav.pdf', text))
            after_rule = False
            for h, m in hrefs:
                mt = _node_get(m, 'text')
                if mt is None:
                    continue
                if mt.value == '---':
                    after_rule = True
                elif h is None and after_rule:
                    out.append(('nav.pdf_parts', mt))
                elif h:
                    pm = re.match(r'/pdf/parts/Dive-into-Deep-Learning-(.+)-'
                                  r'pytorch\.pdf$', h)
                    if pm and pm.group(1) in parts:
                        out.append((parts[pm.group(1)], mt))
        elif any(h and h.startswith('/notebooks/') for h, _ in hrefs):
            if text is not None:
                out.append(('nav.notebooks', text))
        elif _node_get(item, 'icon') is not None and \
                _node_get(item, 'icon').value == 'translate':
            aria = _node_get(item, 'aria-label')
            if aria is not None:
                out.append(('nav.language', aria))
    return out


def _language_menu_hrefs(root_node):
    """[(code, href ScalarNode)] of the language menu's current languages."""
    out = []
    right = _node_get(_node_get(_node_get(root_node, 'book'), 'navbar'),
                      'right')
    for item in (right.value if isinstance(right, yaml.SequenceNode) else []):
        icon = _node_get(item, 'icon')
        if icon is None or icon.value != 'translate':
            continue
        for h, m in _menu_hrefs(item):
            text = _node_get(m, 'text')
            if text is None or text.value == '---':
                break           # old editions follow the rule
            href = _node_get(m, 'href')
            mm = re.match(r'([a-z]{2,3}) · ', text.value)
            if mm and href is not None:
                out.append((mm.group(1), href))
    return out


def _yaml_scalar(value: str, node) -> str:
    """YAML text for `value`, plain when the node was plain and it is safe."""
    if node.style is None and value == value.strip() and '\n' not in value:
        try:
            if yaml.safe_load(value) == value and ' #' not in value:
                return value
        except yaml.YAMLError:
            pass
    return json.dumps(value, ensure_ascii=False)


def _language_href(lang_code: str, own: str) -> str:
    if lang_code == own:
        return '/'
    host = english()['site.hosts'][0] if lang_code == 'en' else \
        f'{lang_code}.d2l.ai'
    return f'https://{host}'


def localize_quarto_text(text: str) -> str:
    """Return _quarto.yml text localized for the selected catalog."""
    start = text.find(_BLOCK_START)
    if start >= 0:
        end = text.index(_BLOCK_END, start) + len(_BLOCK_END)
        text = text[:start].rstrip('\n') + '\n' + text[end:].lstrip('\n')
    root_node = yaml.compose(text)
    edits = []
    for key, node in _quarto_targets(root_node):
        value = t(key)
        if node.value != value:
            edits.append((node, _yaml_scalar(value, node)))
    own = code()
    for lang_code, node in _language_menu_hrefs(root_node):
        value = _language_href(lang_code, own)
        if lang_code == 'en' and own == 'en':
            value = '/'
        if node.value != value:
            edits.append((node, _yaml_scalar(value, node)))
    for node, new in sorted(edits, key=lambda e: -e[0].start_mark.index):
        text = (text[:node.start_mark.index] + new +
                text[node.end_mark.index:])
    cat, base = catalog(), english()
    block = []
    if cat['lang'] != base['lang']:
        block.append(f'lang: {json.dumps(cat["lang"])}')
    if cat['dir'] != base['dir']:
        block.append(f'dir: {json.dumps(cat["dir"])}')
    overrides = quarto_overrides()
    if overrides:
        block.append('language:')
        block += [f'  {k}: {json.dumps(v, ensure_ascii=False)}'
                  for k, v in sorted(overrides.items())]
    if block:
        text = (text.rstrip('\n') + '\n\n' + _BLOCK_START +
                ''.join(b + '\n' for b in block) + _BLOCK_END)
    return text


def localize_quarto(root: Path = REPO, verbose: bool = True) -> bool:
    path = root / '_quarto.yml'
    old = path.read_text(encoding='utf-8')
    new = localize_quarto_text(old)
    if new != old:
        path.write_text(new, encoding='utf-8')
    if verbose:
        print(f'i18n: {code()} — _quarto.yml '
              + ('unchanged' if new == old else
                 'restored to English' if is_english() else 'localized'))
    return new != old


# ── Helpers for other emitters ────────────────────────────────────────────

def quarto_lang_yaml(indent: str = '') -> str:
    """`lang:`/`dir:` lines for a generated Quarto document or project, or ''
    for English (so English output stays byte-identical)."""
    cat, base = catalog(), english()
    lines = []
    if cat['lang'] != base['lang']:
        lines.append(f'{indent}lang: {json.dumps(cat["lang"])}')
    if cat['dir'] != base['dir']:
        lines.append(f'{indent}dir: {json.dumps(cat["dir"])}')
    return ''.join(x + '\n' for x in lines)


def latex_cjk_preamble() -> str:
    """xeCJK setup for the CJK fonts of the locale, or '' (English)."""
    cat = catalog()
    main = cat.get('fonts.pdf.cjk')
    if not main:
        return ''
    lines = ['\\usepackage{xeCJK}', f'\\setCJKmainfont{{{main}}}']
    if cat.get('fonts.pdf.cjk_sans'):
        lines.append(f'\\setCJKsansfont{{{cat["fonts.pdf.cjk_sans"]}}}')
    if cat.get('fonts.pdf.cjk_mono'):
        lines.append(f'\\setCJKmonofont{{{cat["fonts.pdf.cjk_mono"]}}}')
    return '\n'.join(lines) + '\n'


# ── Locale-file check ─────────────────────────────────────────────────────

def check_file(path: Path) -> list:
    """Problems of a locale file: unknown keys, wrong types, placeholders
    that differ from English. [] when it is clean."""
    base = english()
    over = _load_yaml(path)
    problems = []
    for k, v in sorted(over.items()):
        if k.startswith('quarto.'):
            continue
        if k not in base:
            problems.append(f'unknown key {k!r}')
            continue
        b = base[k]
        if is_setting(k):
            if b is not None and v is not None and type(v) is not type(b):
                problems.append(f'{k}: expected {type(b).__name__}, '
                                f'got {type(v).__name__}')
            continue
        if not isinstance(v, str):
            problems.append(f'{k}: expected a string')
            continue
        want = set(_PLACEHOLDER.findall(b))
        have = set(_PLACEHOLDER.findall(v))
        if want != have:
            problems.append(f'{k}: placeholders {sorted(have)} != English '
                            f'{sorted(want)}')
    if 'dir' in over and over['dir'] not in ('ltr', 'rtl'):
        problems.append("dir: must be 'ltr' or 'rtl'")
    if 'forum.mode' in over and over['forum.mode'] not in ('english', 'own'):
        problems.append("forum.mode: must be 'english' or 'own'")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('render', help='render _i18n/ includes and localize '
                       '_quarto.yml for D2L_LANG / D2L_LOCALE_FILE')
    r.add_argument('--root', type=Path, default=REPO)
    c = sub.add_parser('check', help='validate a locale file')
    c.add_argument('file', type=Path)
    s = sub.add_parser('show', help='print the effective catalog')
    s.add_argument('key', nargs='?')
    args = ap.parse_args(argv)

    if args.cmd == 'render':
        render_includes(args.root)
        localize_quarto(args.root)
    elif args.cmd == 'check':
        problems = check_file(args.file)
        for p in problems:
            print(f'{args.file}: {p}')
        if problems:
            sys.exit(1)
        print(f'{args.file}: OK')
    elif args.cmd == 'show':
        cat = catalog()
        if args.key:
            print(json.dumps(cat[args.key], ensure_ascii=False, indent=2))
        else:
            print(yaml.safe_dump(cat, allow_unicode=True, sort_keys=True,
                                 width=100))


if __name__ == '__main__':
    main()
