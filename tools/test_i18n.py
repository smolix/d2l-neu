#!/usr/bin/env python3
"""Tests of the UI string catalog (tools/i18n.py, i18n/en.yml).

    python3 tools/test_i18n.py

English must leave every output as it was: the rendered includes carry no
placeholder, _quarto.yml is untouched, and the English literals kept outside
the catalog agree with it. A pseudo-locale must reach every emitter, and a
later English run must restore _quarto.yml byte for byte.
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import i18n  # noqa: E402

PSEUDO = """\
lang: ja
nav.slides: スライド
nav.pdf_parts: 部 (PyTorch)
book.title: ディープラーニング入門
book.exercises_heading: 練習問題
book.discussions_link: ディスカッション
share.label: 共有
toc_slides.label: "スライド · {fw}"
slides.code_label: 入力
fonts.html.google: ["Noto Sans JP:wght@400;700"]
fonts.html.sans: "'Source Sans 3', 'Noto Sans JP', sans-serif"
fonts.pdf.cjk: Noto Serif CJK JP
quarto.crossref-ch-prefix: 章
"""


class Env:
    """Set D2L_* variables for a block, then restore them."""

    def __init__(self, **env):
        self.env = env

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in ('D2L_LANG', 'D2L_LOCALE_FILE')}
        for k in self.old:
            os.environ.pop(k, None)
        os.environ.update(self.env)
        i18n.reset()
        i18n._quarto_language.cache_clear()

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        i18n.reset()
        i18n._quarto_language.cache_clear()


def pseudo_file(tmp):
    p = Path(tmp) / 'xx.yml'
    p.write_text(PSEUDO, encoding='utf-8')
    return p


class EnglishIsIdentity(unittest.TestCase):
    def test_templates_render_without_placeholders(self):
        with Env():
            for name in i18n.TEMPLATES:
                out = i18n.render_template((REPO / name).read_text(), name)
                self.assertNotIn('{{t', out, name)

    def test_quarto_yml_untouched(self):
        with Env():
            text = (REPO / '_quarto.yml').read_text(encoding='utf-8')
            self.assertEqual(i18n.localize_quarto_text(text), text)
            self.assertTrue(i18n.is_english())

    def test_every_navbar_key_found(self):
        import yaml
        root = yaml.compose((REPO / '_quarto.yml').read_text(encoding='utf-8'))
        keys = {k for k, _ in i18n._quarto_targets(root)}
        nav = {k for k in i18n.english() if k.startswith('nav.')}
        self.assertEqual(nav | {'book.title'}, keys)

    def test_english_literals_outside_catalog_agree(self):
        # _d2l-slides.scss draws the English In/Out tags itself.
        scss = (REPO / '_d2l-slides.scss').read_text(encoding='utf-8')
        for key in ('slides.code_label', 'slides.output_label'):
            self.assertIn(f'content: "{i18n.en(key)}"', scss)

    def test_every_template_key_exists(self):
        for name in i18n.TEMPLATES:
            for _, key in i18n._TEMPLATE.findall((REPO / name).read_text()):
                self.assertIn(key, i18n.english(), f'{name}: {key}')

    def test_english_catalog_checks_clean(self):
        self.assertEqual(i18n.check_file(i18n.CATALOG), [])


class PseudoLocale(unittest.TestCase):
    def test_fill_and_plural(self):
        with Env():
            self.assertEqual(i18n.t('toc_slides.label', fw='JAX'), 'Slides · JAX')
            self.assertEqual(i18n.tn('slides_index.decks', 1), '1 deck')
            self.assertEqual(i18n.tn('slides_index.decks', 3), '3 decks')
            # braces that are not placeholders stay
            self.assertIn('\\input{d2l-beamer}', i18n.t('slides.beamer_readme'))

    def test_override_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Env(D2L_LANG='xx', D2L_LOCALE_FILE=str(pseudo_file(tmp))):
                self.assertEqual(i18n.code(), 'xx')
                self.assertEqual(i18n.setting('lang'), 'ja')
                self.assertEqual(i18n.t('nav.slides'), 'スライド')
                self.assertEqual(i18n.t('nav.courses'), 'Courses')  # fallback
                self.assertEqual(i18n.variants('book.exercises_heading'),
                                 ('Exercises', '練習問題'))
                self.assertEqual(i18n.quarto_term('crossref-ch-prefix'), '章')
                self.assertFalse(i18n.is_english())
                self.assertIn('xeCJK', i18n.latex_cjk_preamble())
                self.assertEqual(i18n.quarto_lang_yaml(), 'lang: "ja"\n')

    def test_emphasis_setting(self):
        with Env():
            self.assertEqual(i18n.emphasis(), 'italic')
            self.assertEqual(i18n.latex_locale_preamble(), '')
            self.assertNotIn('font-weight: bold', i18n._head_extras())
        with tempfile.TemporaryDirectory() as tmp:
            bold = Path(tmp) / 'bold.yml'
            bold.write_text(PSEUDO + 'fonts.emphasis: bold\n', encoding='utf-8')
            with Env(D2L_LANG='xx', D2L_LOCALE_FILE=str(bold)):
                self.assertIn('em { font-style: normal; font-weight: bold; }',
                              i18n._head_extras())
                pre = i18n.latex_locale_preamble()
                self.assertIn('xeCJK', pre)
                self.assertIn('\\renewcommand{\\emph}[1]{\\textbf{#1}}', pre)
            bad = Path(tmp) / 'bad.yml'
            bad.write_text(PSEUDO + 'fonts.emphasis: dots\n', encoding='utf-8')
            with Env(D2L_LANG='xx', D2L_LOCALE_FILE=str(bad)):
                with self.assertRaises(SystemExit):
                    i18n.emphasis()

    def test_quarto_yml_round_trip(self):
        text = (REPO / '_quarto.yml').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as tmp:
            with Env(D2L_LANG='ja', D2L_LOCALE_FILE=str(pseudo_file(tmp))):
                loc = i18n.localize_quarto_text(text)
        import yaml
        cfg = yaml.safe_load(loc)
        self.assertEqual(cfg['lang'], 'ja')
        self.assertEqual(cfg['language']['crossref-ch-prefix'], '章')
        self.assertEqual(cfg['book']['title'], 'ディープラーニング入門')
        right = cfg['book']['navbar']['right']
        self.assertEqual(right[0]['text'], 'スライド')
        menu = right[-1]['menu']
        current = menu[:[m['text'] for m in menu].index('---')]
        langs = {m['text'][:2]: m['href'] for m in current}
        self.assertEqual(langs['ja'], '/')
        self.assertEqual(langs['en'], 'https://d2l.ai')
        with Env():
            self.assertEqual(i18n.localize_quarto_text(loc), text)

    def test_render_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in i18n.TEMPLATES:
                (root / name).write_text((REPO / name).read_text())
            with Env(D2L_LANG='ja', D2L_LOCALE_FILE=str(pseudo_file(tmp))):
                i18n.render_includes(root, verbose=False)
            tabs = (root / '_i18n' / '_d2l-tabs.html').read_text()
            self.assertIn("'共有'", tabs)
            self.assertIn("'スライド · {fw}'", tabs)
            head = (root / '_i18n' / '_d2l-favicon.html').read_text()
            self.assertIn('Noto+Sans+JP', head)
            self.assertIn('Page not found', (root / '404.qmd').read_text())
            langs = (root / '_i18n' / '_d2l-languages.html').read_text()
            self.assertIn("var CURRENT = 'ja';", langs)

    def test_detectors_accept_translated_headings(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, 'D2L_LANG': 'ja',
                   'D2L_LOCALE_FILE': str(pseudo_file(tmp))}
            code = (
                'import sys; sys.path.insert(0, %r)\n'
                'import d2l_preprocess as d, exercise_refs as e\n'
                'lines = ["# T", "", "## 練習問題", "", "1. a", "   :exlabel:`ex_a`",'
                ' "2. b", "", "[ディスカッション](https://d2l.discourse.group/t/1)"]\n'
                'print(d._exercise_body(lines), e.labels_in("\\n".join(lines)))\n'
                % str(TOOLS))
            r = subprocess.run([sys.executable, '-c', code], env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "(3, 8) {'ex_a': {1}}")

    def test_check_file_reports_problems(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.yml'
            p.write_text('nav.slide: x\ntoc_slides.label: "Slides"\ndir: up\n',
                         encoding='utf-8')
            problems = i18n.check_file(p)
            self.assertTrue(any('unknown key' in x for x in problems))
            self.assertTrue(any('placeholders' in x for x in problems))
            self.assertTrue(any(x.startswith('dir:') for x in problems))
            self.assertEqual(i18n.check_file(pseudo_file(tmp)), [])

    def test_gen_pdf_localizes_front_matter(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, 'D2L_LANG': 'ja',
                   'D2L_LOCALE_FILE': str(pseudo_file(tmp))}
            code = ('import sys; from pathlib import Path; sys.path.insert(0, %r)\n'
                    'import gen_pdf\n'
                    'gen_pdf._write_quarto_yml(Path(%r), "jax", [])\n'
                    'print(gen_pdf.strip_discussions("A\\n\\n## ディスカッション\\n\\nx\\n"))\n'
                    % (str(TOOLS), tmp))
            r = subprocess.run([sys.executable, '-c', code], env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), 'A')
            yml = (Path(tmp) / '_quarto.yml').read_text(encoding='utf-8')
            self.assertIn('title: "ディープラーニング入門"', yml)
            self.assertIn('static/d2l-i18n-preamble.tex', yml)
            self.assertTrue(re.search(r'mainfont: "SourceSerif4"', yml))


if __name__ == '__main__':
    unittest.main(verbosity=1)
