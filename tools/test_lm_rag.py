#!/usr/bin/env python3
"""CPU tests for sec_lm_rag (chapter_lm-encoders/rag.md), the shared part
written before the section: `gold_chunk_ids` on a hand-made three-document
corpus under two chunkings, with and without the alternative spans; the
`book_qa` reader on a fixture question-set zip (splits, filters, the
stratified `subset`, `stats`, the notice on a second `test` load); the prompt, citation and
judge helpers (`rag_prompt`, `parse_citations`, `_rag_parse_json`,
`rag_judge` with a stub judge); the pure functions of
tools/build_book_qa.py (evidence spans, lexical filter, phrase blacklist,
identifier flag, kappa, deterministic zip)."""
import collections
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
            z.writestr(f'lm-book-qa-v2/qa_{s}.jsonl',
                       ''.join(json.dumps(r) + '\n' for r in rows))
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


class Reader(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, 'lm-book-qa-v2'))
        sha1 = qa_zip(os.path.join(self.root, 'lm-book-qa-v2',
                                   'lm-book-qa-v2.zip'))
        self.saved = d2l.DATA_HUB.get('lm-book-qa-v2')
        d2l.DATA_HUB['lm-book-qa-v2'] = (
            'https://example.invalid/lm-book-qa-v2.zip', sha1)

    def tearDown(self):
        if self.saved is None:
            del d2l.DATA_HUB['lm-book-qa-v2']
        else:
            d2l.DATA_HUB['lm-book-qa-v2'] = self.saved

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


class Prompts(unittest.TestCase):
    def test_rag_prompt(self):
        hits = [dict(heading_path=['Ch', 'Sec'], text='alpha'),
                dict(heading_path=['Ch'], text='beta')]
        m = d2l.rag_prompt('Why?', hits)
        self.assertIn('[1] Ch > Sec\nalpha', m[1]['content'])
        self.assertIn('[2] Ch\nbeta', m[1]['content'])
        self.assertTrue(m[0]['content'].endswith(d2l.ABSTAIN))
        closed = d2l.rag_prompt('Why?', None)
        self.assertNotIn('Passages', closed[1]['content'])
        self.assertNotIn('cite', closed[0]['content'])
        self.assertTrue(closed[0]['content'].endswith(d2l.ABSTAIN))

    def test_parse_citations(self):
        self.assertEqual(d2l.parse_citations('a [1] b [1][3] c [2, 5]', 4),
                         [1, 3, 2])
        self.assertEqual(d2l.parse_citations('none here', 4), [])
        self.assertEqual(d2l.parse_citations('[0] [9]', 4), [])

    def test_parse_json(self):
        keys = ['correct', 'faithful']
        self.assertEqual(d2l._rag_parse_json(
            'x {"correct": true, "faithful": "no"} y', keys),
            {'correct': True, 'faithful': False})
        self.assertIsNone(d2l._rag_parse_json('{"correct": true}', keys))
        self.assertIsNone(d2l._rag_parse_json('not json', keys))
        self.assertEqual(d2l._rag_parse_json(
            '{"a": 1} {"correct": false, "faithful": true}', keys),
            {'correct': False, 'faithful': True})

    def test_abstained(self):
        self.assertTrue(d2l._rag_abstained(' ' + d2l.ABSTAIN.upper()))
        self.assertTrue(d2l._rag_abstained(d2l.ABSTAIN[:-1]))
        self.assertFalse(d2l._rag_abstained('It is 4. ' + d2l.ABSTAIN))


class StubJudge(unittest.TestCase):
    """rag_judge with d2l.generate replaced: abstentions by string match,
    closed book judged for 'correct' only, one retry on malformed JSON."""
    def test_verdicts(self):
        replies = iter(['{"correct": true}', 'garbage',
                        '{"correct": false, "faithful": true, '
                        '"supported": false}'])
        retry = iter(['{"correct": true, "faithful": true, '
                      '"supported": true}'])
        calls = []
        def fake(model, tok, prompts, **kw):
            calls.append(prompts)
            src = replies if len(calls) == 1 else retry
            return [d2l.Generation(next(src), [], None, 'eos')
                    for _ in prompts]
        q = dict(question='Q', reference='R', key_facts=['f'],
                 answerable=True)
        u = dict(question='Q', reference='none', key_facts=[],
                 answerable=False)
        old, d2l.generate = d2l.generate, fake
        try:
            v = d2l.rag_judge(None, None, [q, q, q, u, q],
                              [dict(answer='A'), dict(answer='A [1]',
                                                      cited_ids=[1]),
                               dict(answer='B'), dict(answer=d2l.ABSTAIN),
                               dict(answer=d2l.ABSTAIN)],
                              [[], ['p'], ['p'], ['p'], []])
        finally:
            d2l.generate = old
        self.assertEqual(v[0], dict(correct=True, faithful=None,
                                    supported=None))
        self.assertEqual(v[1], dict(correct=True, faithful=True,
                                    supported=True))  # after the retry
        self.assertEqual(v[2], dict(correct=False, faithful=True,
                                    supported=False))
        self.assertEqual(v[3], dict(correct=True, faithful=True,
                                    supported=None))
        self.assertEqual(v[4], dict(correct=False, faithful=True,
                                    supported=None))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]), 3)  # abstentions never reach it
        self.assertEqual(calls[1][0][-1]['content'],
                         d2l.RAG_JUDGE_V1['retry'])


class StubEndpointJudge(unittest.TestCase):
    """rag_judge with a d2l.ChatAPI-like judge (an object with achat):
    thinking on by default, greedy requests with max_tokens 3,000 and
    seed 0, the verdict read after '</think>', a reply cut off at the
    token limit and a failed request treated as malformed (one retry)."""
    def test_endpoint(self):
        R = collections.namedtuple('R', 'text finish_reason')
        script = {
            'A': [R('<think>long</think>{"correct": true}', 'stop')],
            'B': [R('', 'length'), R('{"correct": false}', 'stop')],
            'C': [RuntimeError('503'), RuntimeError('503')]}
        sent = []

        class Stub:
            async def achat(self, messages, **params):
                sent.append((messages, params))
                answer = messages[1]['content'].split(
                    'Answer to grade: ')[1].split('\n')[0]
                r = script[answer].pop(0)
                if isinstance(r, BaseException):
                    raise r
                return r
        q = dict(question='Q', reference='R', key_facts=['f'],
                 answerable=True)
        v = d2l.rag_judge(Stub(), None, [q, q, q],
                          [dict(answer=a) for a in 'ABC'], [[], [], []],
                          batch_size=2)
        self.assertEqual([x['correct'] for x in v], [True, False, None])
        self.assertEqual(len(sent), 5)  # three, then B and C again
        self.assertEqual(sent[0][1], dict(max_tokens=3000, temperature=0,
                                          seed=0))
        retry = [m for m, _ in sent[3:]]
        self.assertEqual(retry[0][-2], {'role': 'assistant', 'content': ''})
        self.assertEqual(retry[0][-1]['content'], d2l.RAG_JUDGE_V1['retry'])

    def test_model_thinking(self):
        """A Transformers judge with thinking=True: enable_thinking is
        passed on, the verdict follows '</think>', a reply cut off while
        reasoning counts as malformed."""
        seen = []
        def fake(model, tok, prompts, **kw):
            seen.append(kw)
            return [d2l.Generation('x</think>{"correct": true}', [], None,
                                   'eos'),
                    d2l.Generation('reasoning {"correct": false', [], None,
                                   'length')][:len(prompts)]
        q = dict(question='Q', reference='R', key_facts=['f'],
                 answerable=True)
        old, d2l.generate = d2l.generate, fake
        try:
            v = d2l.rag_judge(None, None, [q, q], [dict(answer='A'),
                                                   dict(answer='B')],
                              [[], []], thinking=True)
        finally:
            d2l.generate = old
        self.assertEqual(v[0]['correct'], True)
        self.assertEqual(seen[0]['enable_thinking'], True)
        self.assertEqual(seen[0]['max_new_tokens'], 3000)
        self.assertEqual(len(seen), 2)  # the cut-off reply is retried


class BuildBookQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'build_book_qa.py')
        spec = importlib.util.spec_from_file_location('build_book_qa', path)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_evidence(self):
        f = self.m.find_evidence
        text = 'The block adds its\ninput to the output.'
        self.assertEqual(f(text, 'adds its input to'), (10, 27))
        self.assertEqual(f(text, 'The block adds'), (0, 14))
        self.assertIsNone(f(text, 'removes its input'))
        self.assertIsNone(f(text, 'adds'))  # too short to anchor

    def test_lexical(self):
        anchor = ('We note that each convolution followed by batch '
                  'normalization drops the `conv_bias` term.')
        jac, shared = self.m.lexical_check(
            'Why does each convolution followed by batch normalization '
            'lose a term?', anchor)
        self.assertEqual(shared, ['convolution followed by batch '
                                  'normalization', 'each convolution '
                                  'followed by batch'])
        jac, shared = self.m.lexical_check(
            'Which term does `conv_bias` drop?', anchor)
        self.assertEqual(shared, [])
        self.assertLess(jac, 0.5)

    def test_phrases_and_identifiers(self):
        bad = self.m.CONTEXT_PHRASES.search
        self.assertTrue(bad('What does the passage say about Adam?'))
        self.assertTrue(bad('What does this section define?'))
        self.assertTrue(bad('Which value is shown above?'))
        self.assertFalse(bad('What learning rate does the book use for '
                             'AdamW in its transformer example?'))
        ident = self.m.identifier_flag
        for q in ('What does `foo` return?', 'Why use batch_size 64?',
                  'What does d2l.Trainer do?', 'When is nn.Linear lazy?',
                  'What does fit() call?', 'Why is LayerNorm used?'):
            self.assertTrue(ident(dict(question=q, reference='')), q)
        self.assertFalse(ident(dict(question='Why, e.g., use dropout?',
                                    reference='It regularizes.')))

    def test_kappa(self):
        k = self.m.kappa
        self.assertAlmostEqual(k([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(k([1, 0, 1, 0], [1, 1, 0, 0]), 0.0)
        self.assertIsNone(k([1, 1], [1, 1]))

    def test_zip_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, 'a.zip'), os.path.join(d, 'b.zip')
            self.m.write_zip(a, {'y': b'2', 'x': b'1'})
            self.m.write_zip(b, {'x': b'1', 'y': b'2'})
            with open(a, 'rb') as f, open(b, 'rb') as g:
                self.assertEqual(f.read(), g.read())


if __name__ == '__main__':
    unittest.main()
