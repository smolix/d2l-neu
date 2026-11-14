#!/usr/bin/env python3
"""CPU tests for the book corpus (chapter 19, sec_lm_retrieval):
`d2l.clean_d2l_markdown` rules 1-8, the chunkers, `tools/build_r3.py` on the
fixture tree `tools/lm/fixtures/encoders/book/` (two builds byte-identical,
`--check`, inclusion and exclusion, splits, chunk invariants) and the
`d2l.book_corpus` loader's version check. Tests that tokenize need the
Qwen3-0.6B tokenizer in the Hugging Face cache (HF_HOME) and skip without it.
"""
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile

from d2l import torch as d2l

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import build_r3  # noqa: E402

BOOK = os.path.join(TOOLS, 'lm', 'fixtures', 'encoders', 'book')


def read(path):
    with open(os.path.join(BOOK, path), encoding='utf-8') as f:
        return f.read()


def tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            d2l.R3_TOKENIZER, revision=d2l.HF_REVISIONS[d2l.R3_TOKENIZER],
            local_files_only=True)
    except Exception as e:  # noqa: BLE001 (not cached, offline)
        raise unittest.SkipTest(f'{d2l.R3_TOKENIZER} tokenizer: {e}')


class Cleaning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.titles = d2l._r3_label_titles(
            {p: read(p + '.md') for p in ('chapter_alpha/index',
                                          'chapter_alpha/one',
                                          'chapter_alpha/two')})
        cls.text, cls.meta = d2l.clean_d2l_markdown(
            read('chapter_alpha/one.md'), label_titles=cls.titles)

    def test_rule1_slides_cut(self):
        self.assertNotIn('Slides are cut', self.text)
        two, _ = d2l.clean_d2l_markdown(read('chapter_alpha/two.md'))
        self.assertNotIn('cover', two)
        self.assertNotIn('Text after the cover', two)
        self.assertIn('### A Short Subsection\n', two)  # no {.unnumbered}
        self.assertIn('Too short, and cited. The sentence', two)

    def test_rule2_tabs(self):
        self.assertIn('In PyTorch the map is one multiplication.', self.text)
        self.assertNotIn('In JAX', self.text)
        self.assertNotIn(':begin_tab:', self.text)
        self.assertNotIn('one-alpha-map-jax',
                         [c['id'] for c in self.meta['cell_ids']])
        two, meta = d2l.clean_d2l_markdown(read('chapter_alpha/two.md'))
        self.assertEqual([c['id'] for c in meta['cell_ids']], ['two-measure'])

    def test_rule3_cells(self):
        ids = [c['id'] for c in self.meta['cell_ids']]
        self.assertEqual(ids, ['one-alpha-map'])  # imports-only cells dropped
        self.assertIn('```python\n#@save\ndef alpha_map', self.text)
        self.assertNotIn('%%tab', self.text)
        for c in self.meta['cell_ids']:
            cell = self.text[c['char_start']:c['char_end']]
            self.assertTrue(cell.startswith('```python\n'))
            self.assertTrue(cell.endswith('```'))
        index, meta = d2l.clean_d2l_markdown(read('chapter_alpha/index.md'))
        self.assertNotIn('d2lbook', index)  # notebook boilerplate dropped
        self.assertNotIn('```toc', index)
        self.assertNotIn('maxdepth', index)
        text, _ = d2l.clean_d2l_markdown('# T\n\n```bash\nls -l\n```\n')
        self.assertIn('```bash\nls -l\n```', text)

    def test_rule4_references(self):
        t = self.text
        self.assertNotIn(':cite', t)
        self.assertIn('sends $x$ to $\\alpha x$.', t)
        self.assertIn('(Equation)', t)
        self.assertIn('Exercise', d2l.clean_d2l_markdown(
            read('chapter_alpha/one.md'), keep_exercises=True)[0])
        self.assertIn('Table "Alpha values and their names" and Figure', t)
        self.assertIn('Table: Alpha values and their names.\n', t)
        self.assertIn('exercise, or Exercise.', t)
        self.assertIn('Figure "The alpha map scales every input"', t)
        self.assertIn('Figure: The alpha map scales every input. A second '
                      'sentence.', t)
        self.assertIn('see also\nanother section', t)
        self.assertEqual(self.meta['unresolved_refs'], ['sec_missing'])
        for directive in (':label:', ':eqlabel:', ':width:', ':numref:'):
            self.assertNotIn(directive, t)
        index, meta = d2l.clean_d2l_markdown(read('chapter_alpha/index.md'),
                                             label_titles=self.titles)
        self.assertIn('Section "Defining Alpha"', index)
        self.assertIn('Section "Measuring Alpha"', index)
        self.assertEqual(meta['unresolved_refs'], ['chap_zeta'])
        self.assertEqual((meta['title'], meta['label']),
                         ('Alpha Methods', 'chap_alpha'))
        self.assertEqual(self.titles['chap_alpha'][:2],
                         ('Chapter', 'Alpha Methods'))

    def test_rule5_markup(self):
        t = self.text
        self.assertNotIn('<!--', t)
        self.assertNotIn('A comment', t)
        self.assertNotIn(':::', t)
        self.assertIn('A fenced div keeps its content', t)
        self.assertNotIn('Discussions', t)
        self.assertNotIn('\n\n\n', t)

    def test_rule6_math(self):
        self.assertIn('$$\nf(x) = \\alpha x,\n$$', self.text)
        sparse = d2l._r3_strip_math(self.text)
        self.assertNotIn('\\alpha', sparse.split('```')[0])
        self.assertIn('alpha * x', sparse)  # code keeps its text

    def test_rule7_exercises(self):
        self.assertTrue(self.meta['has_exercises'])
        self.assertNotIn('## Exercises', self.text)
        kept, _ = d2l.clean_d2l_markdown(read('chapter_alpha/one.md'),
                                         keep_exercises=True)
        self.assertIn('Prove that the alpha map is linear', kept)

    def test_rule8_errors(self):
        with self.assertRaises(ValueError):
            d2l.clean_d2l_markdown('# T\n\n```python\nx = 1\n')
        with self.assertRaises(ValueError):
            d2l.clean_d2l_markdown('# T\n\nSee :foo:`bar`.\n')
        with self.assertRaises(ValueError):
            d2l.clean_d2l_markdown('# T\n\n:begin_tab:`pytorch`\nx\n')

    def test_headings(self):
        hs = self.meta['headings']
        self.assertEqual([(h['level'], h['title'], h['label']) for h in hs], [
            (1, 'Defining Alpha', 'sec_alpha_one'),
            (2, 'Computing the Map', 'subsec_alpha_compute')])
        for h in hs:
            self.assertTrue(self.text[h['char_start']:].startswith(
                '#' * h['level'] + ' ' + h['title']))

    def test_quarto_and_splits(self):
        files = d2l._r3_quarto_files(read('_quarto.yml'))
        self.assertEqual(files[0], ('chapter_basics/index', None))
        self.assertIn(('chapter_alpha/one', 'chapter_basics/index'), files)
        ids = [f'chapter_x/f{i}' for i in range(20)] + ['chapter_y/a']
        split = d2l._r3_split_docs(ids)
        counts = {s: sum(split[d] == s for d in ids[:20])
                  for s in ('train', 'dev', 'test')}
        self.assertEqual(counts, dict(train=14, dev=3, test=3))
        self.assertEqual(split['chapter_y/a'], 'train')


class Building(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer()  # skip without the tokenizer
        cls.tmp = tempfile.mkdtemp()
        out = os.path.join(cls.tmp, 'a')
        assert build_r3.main(['--root', BOOK, '--worktree', '--out', out,
                              '--zip']) == 0
        cls.zip = os.path.join(out, build_r3.ZIP_NAME)
        cls.manifest, cls.docs, cls.chunks = d2l._r3_read_zip(cls.zip)

    def test_two_builds_identical(self):
        out = os.path.join(self.tmp, 'b')
        build_r3.main(['--root', BOOK, '--worktree', '--out', out, '--zip'])
        with open(self.zip, 'rb') as f, open(
                os.path.join(out, build_r3.ZIP_NAME), 'rb') as g:
            self.assertEqual(f.read(), g.read())
        self.assertEqual(build_r3.main(
            ['--root', BOOK, '--worktree', '--out', out,
             '--check', self.zip]), 0)

    def test_included_and_excluded(self):
        self.assertEqual(sorted(d['doc_id'] for d in self.docs), [
            'chapter_alpha/index', 'chapter_alpha/one', 'chapter_alpha/two',
            'chapter_beta/index'])
        excluded = {(e['directory'], e['reason'])
                    for e in self.manifest['excluded']}
        self.assertIn(('chapter_lm-zeta', 'Language Models part '
                       '(self-reference)'), excluded)
        self.assertIn(('chapter_basics', 'part placeholder'), excluded)
        self.assertEqual(set(self.manifest['doc_split'].values()), {'train'})
        self.assertEqual(self.manifest['corpus_version'], 'r3.1')
        self.assertEqual(self.manifest['unresolved_refs']['count'], 2)

    def test_members_hashed(self):
        with zipfile.ZipFile(self.zip) as z:
            for name in build_r3.MEMBERS:
                data = gzip.decompress(z.read(name))
                self.assertEqual(hashlib.sha256(data).hexdigest(),
                                 self.manifest['members'][name])

    def test_chunk_invariants(self):
        docs = {d['doc_id']: d for d in self.docs}
        for tag, chunks in self.chunks.items():
            ks = {}
            for c in chunks:
                text = docs[c['doc_id']]['text']
                self.assertEqual(text[c['char_start']:c['char_end']],
                                 c['text'])
                self.assertLessEqual(c['n_tokens'],
                                     512 if tag == 'sec512' else 256)
                k = ks.setdefault(c['doc_id'], 0)
                self.assertEqual(c['id'], f'{tag}:{c["doc_id"]}#{k:03d}')
                ks[c['doc_id']] += 1
        sec = {c['id']: c for c in self.chunks['sec512']}
        one = [c for c in sec.values() if c['doc_id'] == 'chapter_alpha/one']
        self.assertTrue(any('one-alpha-map' in c['cell_ids'] for c in one))
        self.assertTrue(any(c['section_label'] == 'subsec_alpha_compute'
                            for c in one))

    def test_fixed_windows_on_characters(self):
        tok = tokenizer()
        text = ('The expectation 𝔼[x] of αβγ and 深度学习 ' * 12).strip()
        doc = dict(doc_id='d', text=text, headings=[], cell_ids=[])
        for overlap in (0, 3):
            ws = d2l.chunk_fixed(doc, tok, size=7, overlap=overlap, tag='t')
            self.assertEqual(ws[0]['char_start'], 0)
            self.assertEqual(ws[-1]['char_end'], len(text))
            for w, nxt in zip(ws, ws[1:]):
                if overlap == 0:
                    self.assertEqual(w['char_end'], nxt['char_start'])
                self.assertLess(w['char_start'], nxt['char_start'])
            for w in ws:
                self.assertEqual(text[w['char_start']:w['char_end']],
                                 w['text'])
                self.assertEqual(w['n_tokens'], len(tok(
                    w['text'], add_special_tokens=False)['input_ids']))
                self.assertLessEqual(w['n_tokens'], 9)

    def test_book_corpus_loader(self):
        root = os.path.join(self.tmp, 'root')
        os.makedirs(os.path.join(root, 'lm-book-r3-v1'))
        with open(self.zip, 'rb') as f:
            data = f.read()
        path = os.path.join(root, 'lm-book-r3-v1', 'lm-book-r3-v1.zip')
        with open(path, 'wb') as f:
            f.write(data)
        saved = d2l.DATA_HUB.get('lm-book-r3-v1')
        d2l.DATA_HUB['lm-book-r3-v1'] = (
            'https://example.invalid/lm-book-r3-v1.zip',
            hashlib.sha1(data).hexdigest())
        try:
            corpus = d2l.book_corpus(root=root)
            self.assertEqual(len(corpus), len(self.chunks['sec512']))
            self.assertEqual(corpus.split_of('chapter_beta/index'), 'train')
            c = corpus.chunks[0]
            self.assertEqual(corpus[c['id']], c)
            self.assertTrue(corpus.texts('display')[0].startswith(
                corpus.header(c) + '\n'))
            fixed = d2l.book_corpus('fixed', overlap=64, root=root)
            self.assertEqual(fixed.tag, 'fix256o64')
            with self.assertRaises(ValueError):
                d2l.book_corpus('fixed', size=128, root=root)
            # a zip of another corpus version is refused
            buf = io.BytesIO()
            with zipfile.ZipFile(self.zip) as z, zipfile.ZipFile(
                    buf, 'w') as w:
                for name in z.namelist():
                    blob = z.read(name)
                    if name == 'manifest.json':
                        m = json.loads(blob)
                        m['corpus_version'] = 'r3.0'
                        blob = json.dumps(m).encode()
                    w.writestr(name, blob)
            with open(path, 'wb') as f:
                f.write(buf.getvalue())
            d2l.DATA_HUB['lm-book-r3-v1'] = (
                d2l.DATA_HUB['lm-book-r3-v1'][0],
                hashlib.sha1(buf.getvalue()).hexdigest())
            with self.assertRaises(ValueError):
                d2l.book_corpus(root=root)
        finally:
            if saved is None:
                del d2l.DATA_HUB['lm-book-r3-v1']
            else:
                d2l.DATA_HUB['lm-book-r3-v1'] = saved


if __name__ == '__main__':
    unittest.main()
