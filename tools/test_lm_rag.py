#!/usr/bin/env python3
"""CPU tests for sec_lm_rag (chapter_lm-encoders/rag.md), the shared part
written before the section: `gold_chunk_ids` on a hand-made three-document
corpus under two chunkings, with and without the alternative spans; the
`book_qa` reader on a fixture question-set zip (splits, filters, the
stratified `subset`, `stats`, the notice on a second `test` load)."""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile

from d2l import torch as d2l

TEXT = {'chapter_a/one': 'A' * 100 + 'B' * 100,
        'chapter_a/two': 'C' * 120,
        'chapter_b/three': 'D' * 80}


def corpus(tag, windows):
    docs = [dict(doc_id=d, text=t, title=d, chapter_title=d, headings=[])
            for d, t in TEXT.items()]
    chunks = []
    for d, t in TEXT.items():
        for k, (a, b) in enumerate(windows(len(t))):
            chunks.append(dict(id=f'{tag}:{d}#{k:03d}', doc_id=d,
                               text=t[a:b], heading_path=[d],
                               char_start=a, char_end=b))
    split = {'chapter_a/one': 'dev', 'chapter_a/two': 'dev',
             'chapter_b/three': 'train'}
    return d2l.BookCorpus(dict(corpus_version='r3.1', doc_split=split),
                          docs, chunks, tag)


def halves(n):
    return [(0, n // 2), (n // 2, n)]


def overlapping(n, size=60, step=40):
    return [(a, min(a + size, n)) for a in range(0, n - size + step, step)]


ITEM = dict(qid='dev-0000', split='dev', type='multihop', answerable=True,
            gold_spans=[dict(doc_id='chapter_a/one', char_start=110,
                             char_end=150),
                        dict(doc_id='chapter_a/two', char_start=0,
                             char_end=50)],
            alt_spans=[dict(doc_id='chapter_a/one', char_start=0,
                            char_end=100)])


class Gold(unittest.TestCase):
    def test_two_chunkings(self):
        c = corpus('sec512', halves)
        self.assertEqual(d2l.gold_chunk_ids(ITEM, c), {
            'sec512:chapter_a/one#001', 'sec512:chapter_a/two#000',
            'sec512:chapter_a/one#000'})
        self.assertEqual(d2l.gold_chunk_ids(ITEM, c, include_alt=False), {
            'sec512:chapter_a/one#001', 'sec512:chapter_a/two#000'})
        f = corpus('fix256o64', overlapping)
        # one: windows [0,60) [40,100) [80,140) [120,180) [160,200);
        # span [110,150) covers 30 of 60 chars of [80,140) and [120,180)
        self.assertEqual(d2l.gold_chunk_ids(ITEM, f, include_alt=False), {
            'fix256o64:chapter_a/one#002', 'fix256o64:chapter_a/one#003',
            'fix256o64:chapter_a/two#000'})  # [40,100) shares 10 of 50
        self.assertEqual(d2l.gold_chunk_ids(ITEM, f, min_overlap=0.9,
                                            include_alt=False),
                         {'fix256o64:chapter_a/two#000'})
        none = dict(ITEM, gold_spans=[], alt_spans=[])
        self.assertEqual(d2l.gold_chunk_ids(none, c), set())


def qa_zip(path):
    items = {s: [dict(qid=f'{s}-{i:04d}', split=s,
                      type=['single', 'single', 'code', 'unanswerable'][i % 4],
                      answerable=i % 4 != 3, gold_spans=[], alt_spans=[],
                      flags=['general_knowledge'] if i % 2 else [])
                 for i in range(40)] for s in ('train', 'dev', 'test')}
    with zipfile.ZipFile(path, 'w') as z:
        for s, rows in items.items():
            z.writestr(f'lm-book-qa-v1/qa_{s}.jsonl',
                       ''.join(json.dumps(r) + '\n' for r in rows))
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


class Reader(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, 'lm-book-qa-v1'))
        sha1 = qa_zip(os.path.join(self.root, 'lm-book-qa-v1',
                                   'lm-book-qa-v1.zip'))
        self.saved = d2l.DATA_HUB.get('lm-book-qa-v1')
        d2l.DATA_HUB['lm-book-qa-v1'] = (
            'https://example.invalid/lm-book-qa-v1.zip', sha1)

    def tearDown(self):
        if self.saved is None:
            del d2l.DATA_HUB['lm-book-qa-v1']
        else:
            d2l.DATA_HUB['lm-book-qa-v1'] = self.saved

    def test_reader(self):
        qa = d2l.book_qa('dev', root=self.root)
        self.assertEqual((len(qa), qa.split), (40, 'dev'))
        self.assertEqual(len(d2l.book_qa('dev', answerable=True,
                                         root=self.root)), 30)
        self.assertEqual(len(d2l.book_qa('dev', types=['code'],
                                         root=self.root)), 10)
        sub = d2l.book_qa('dev', n=8, seed=0, root=self.root)
        types = [q['type'] for q in sub]
        self.assertEqual((types.count('single'), types.count('code'),
                          types.count('unanswerable')), (4, 2, 2))
        self.assertEqual([q['qid'] for q in sub], sorted(q['qid']
                                                         for q in sub))
        self.assertEqual([q['qid'] for q in sub], [q['qid'] for q in
                         d2l.book_qa('dev', n=8, seed=0, root=self.root)])
        st = qa.stats()
        self.assertEqual(st.loc['total', 'items'], 40)
        self.assertEqual(st.loc['total', 'book_specific'], 20)
        with self.assertRaises(ValueError):
            d2l.book_qa('val', root=self.root)

    def test_second_test_load_notice(self):
        d2l._BOOK_QA_LOADS['test'] = 0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            d2l.book_qa('test', root=self.root)
        self.assertEqual(out.getvalue(), '')
        with contextlib.redirect_stdout(out):
            d2l.book_qa('test', root=self.root)
        self.assertIn('loaded a second time', out.getvalue())


if __name__ == '__main__':
    unittest.main()
