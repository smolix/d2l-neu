#!/usr/bin/env python3
"""CPU tests for sec_lm_rag (chapter_lm-encoders/rag.md), the shared part
written before the section: `gold_chunk_ids` on a hand-made three-document
corpus under two chunkings, with and without the alternative spans; the
`book_qa` reader on a fixture question-set zip (splits, filters, the
stratified `subset`, `stats`, the notice on a second `test` load); the prompt, citation and
judge helpers (`rag_prompt`, `parse_citations`, `_rag_parse_json`,
`rag_judge` with a stub judge); the run plumbing that sec_lm_rag and
sec_lm_rag_fixes share (the configuration tables, `_rag_rerun`,
`_rag_run_path`, `_rag_settings`, the record readers, `_rag_run_config`
with a stub pipeline, `_rag_boot`, `_rag_scores`, `_rag_ready`); the pure
functions of
tools/build_book_qa.py (evidence spans, lexical filter, phrase blacklist,
identifier flag, kappa, deterministic zip)."""
import collections
import contextlib
import hashlib
import io
import json
import os
import tempfile
import types
import unittest
import unittest.mock
import zipfile

import numpy as np
import pandas as pd
import torch

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
        # an item without alt_spans (synthesized by another chapter)
        bare = {k: v for k, v in ITEM.items() if k != 'alt_spans'}
        self.assertEqual(d2l.gold_chunk_ids(bare, c), {
            'sec512:chapter_a/one#001', 'sec512:chapter_a/two#000'})


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
        # with an explanation before or after (Qwen3.5-9B's habit)
        self.assertTrue(d2l._rag_abstained(
            'The passages do not say.\n\n**' + d2l.ABSTAIN + '**'))
        self.assertTrue(d2l._rag_abstained(
            d2l.ABSTAIN + ' The text mentions X [1] but not Y.'))
        self.assertTrue(d2l._rag_abstained('Therefore, i cannot answer this '
                                           'from the  provided passages'))
        self.assertFalse(d2l._rag_abstained('It is 4 [1].'))
        self.assertFalse(d2l._rag_abstained(
            'I can answer this from the provided passages: 4 [1].'))


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
    token limit treated as malformed (one retry with the reply shown), a
    failed request asked again unchanged, and a request that fails twice
    raised, never stored as a verdict."""
    def test_endpoint(self):
        R = collections.namedtuple('R', 'text finish_reason')
        script = {
            'A': [R('<think>long</think>{"correct": true}', 'stop')],
            'B': [R('', 'length'), R('{"correct": false}', 'stop')],
            'C': [RuntimeError('503'), R('{"correct": true}', 'stop')],
            'D': [RuntimeError('503'), ConnectionError('refused')],
            'E': [R('{"correct": true}', 'stop')]}
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
        self.assertEqual([x['correct'] for x in v], [True, False, True])
        self.assertEqual(len(sent), 5)  # three, then B and C again
        self.assertEqual(sent[0][1], dict(max_tokens=3000, temperature=0,
                                          seed=0))
        retry = [m for m, _ in sent[3:]]
        self.assertEqual(retry[0][-2], {'role': 'assistant', 'content': ''})
        self.assertEqual(retry[0][-1]['content'], d2l.RAG_JUDGE_V1['retry'])
        self.assertEqual(retry[1], sent[2][0])  # C: the same request again
        with self.assertRaisesRegex(RuntimeError, '1 judge requests failed '
                                    "twice, first: ConnectionError"):
            d2l.rag_judge(Stub(), None, [q, q], [dict(answer='E'),
                                                 dict(answer='D')],
                          [[], []])

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


def tokened(tag, windows):
    """corpus() with n_tokens (a tenth of the characters) on each chunk."""
    c = corpus(tag, windows)
    for ch in c.chunks:
        ch['n_tokens'] = (ch['char_end'] - ch['char_start']) // 10
    return c


class StageDefs(unittest.TestCase):
    def test_cov_at_budget(self):
        c = tokened('fix', overlapping)  # 60-character windows, 6 tokens
        q = dict(ITEM, alt_spans=[])
        one = [i for i in c.ids if i.startswith('fix:chapter_a/one')]
        # the span [110, 150) of chapter_a/one: window 2 covers [80, 140)
        self.assertTrue(d2l.cov_at_budget(one[2:], q, c, 6))
        self.assertFalse(d2l.cov_at_budget(one[2:], q, c, 5))  # none fits
        self.assertFalse(d2l.cov_at_budget(one[:2], q, c, 100))
        self.assertTrue(d2l.cov_at_budget(one[:2], ITEM, c, 100))  # alt
        un = dict(ITEM, gold_spans=[], alt_spans=[], answerable=False)
        self.assertFalse(d2l.cov_at_budget(c.ids, un, c, 10**6))

    def test_assemble_context(self):
        hits = [dict(id=f'c{i}', n_tokens=n) for i, n in
                enumerate([300, 500, 400, 200, 100])]
        out, used = d2l.assemble_context(hits, 1000)
        self.assertEqual(([h['id'] for h in out], used), (['c0', 'c1'], 800))
        out, used = d2l.assemble_context(hits, 10**4, k=3)
        self.assertEqual([h['id'] for h in out], ['c0', 'c1', 'c2'])
        for order, ids in (('reverse', ['c4', 'c3', 'c2', 'c1', 'c0']),
                           ('edges', ['c0', 'c2', 'c4', 'c3', 'c1'])):
            out, used = d2l.assemble_context(hits, 10**4, order)
            self.assertEqual(([h['id'] for h in out], used), (ids, 1500))
        V = torch.eye(5)
        V[2] = V[0]  # c2 duplicates c0
        out, used = d2l.assemble_context(hits, 10**4, embed=lambda h: V[
            [int(x['id'][1:]) for x in h]])
        self.assertEqual(([h['id'] for h in out], used),
                         (['c0', 'c1', 'c3', 'c4'], 1100))
        for budget in range(0, 1600, 50):  # never over the budget
            self.assertLessEqual(d2l.assemble_context(hits, budget)[1], budget)
        with self.assertRaises(ValueError):
            d2l.assemble_context(hits, order='middle')


class Metrics(unittest.TestCase):
    def setUp(self):
        self.c = tokened('t', halves)  # relevant to ITEM: one#000, one#001
        self.q = dict(ITEM, alt_spans=[])  # (alt in one#000), two#000

    def test_citation_metrics(self):
        r = [dict(context_ids=['t:chapter_a/one#001', 't:chapter_b/three#000',
                               't:chapter_a/two#000'], cited_ids=[1, 2]),
             dict(context_ids=['t:chapter_b/three#000'], cited_ids=[]),
             dict(context_ids=['t:chapter_a/two#000'], cited_ids=[1, 1, 7])]
        m = d2l.citation_metrics(r, [self.q] * 3, self.c)
        self.assertEqual(m.precision.tolist()[0], 0.5)
        self.assertEqual(m.recall.tolist()[0], 0.5)
        self.assertTrue(np.isnan(m.precision[1]) and np.isnan(m.recall[1]))
        self.assertEqual((m.precision[2], m.recall[2], m.n_cited[2]),
                         (1.0, 1.0, 2))

    def test_classify_failure(self):
        rel, other = 't:chapter_a/one#001', 't:chapter_b/three#000'
        un = dict(self.q, answerable=False, gold_spans=[])
        cases = [  # (item, ranked, context, answer, correct, faithful)
            (self.q, [rel], [rel], 'A [1].', True, True, 'correct'),
            (un, [other], [other], d2l.ABSTAIN, True, True, 'correct'),
            (un, [other], [other], 'B [1].', False, None, 'false_answer'),
            (self.q, [other], [other], 'C', False, True, 'not_retrieved'),
            (self.q, [other, rel], [other], 'C', False, True, 'ranked_low'),
            (self.q, [rel], [rel], d2l.ABSTAIN, False, True,
             'false_abstain'),
            (self.q, [rel], [rel], 'D [1].', False, True, 'ignored'),
            (self.q, [rel], [rel], 'D [1].', np.False_, False, 'unsupported'),
            # a verdict the class needs is missing: no class at all
            (self.q, [rel], [rel], 'D [1].', None, None, 'unjudged'),
            (self.q, [other], [other], 'C', None, None, 'unjudged'),
            (self.q, [rel], [rel], 'D [1].', False, None, 'unjudged')]
        for q, ranked, ctx, answer, correct, faithful, want in cases:
            r = dict(ranked_ids=ranked, context_ids=ctx, answer=answer)
            v = dict(correct=correct, faithful=faithful, supported=None)
            self.assertEqual(d2l.classify_failure(r, q, v, self.c), want)
        r = dict(ranked_ids=[other] * 50 + [rel], context_ids=[other],
                 answer='E')  # the relevant chunk below the depth
        self.assertEqual(d2l.classify_failure(r, self.q, dict(
            correct=False, faithful=None), self.c, depth=50), 'not_retrieved')


class PromptVariants(unittest.TestCase):
    def test_fix_prompts(self):
        hits = [dict(heading_path=['Ch'], text='alpha')]
        fixed = d2l.rag_prompt('Q?', hits)
        self.assertEqual(d2l._rag_fix_prompt('Q?', hits, 'fixed'), fixed)
        quote = d2l._rag_fix_prompt('Q?', hits, 'quote')
        self.assertEqual(quote[0], fixed[0])
        self.assertIn('"Quotes:"', quote[1]['content'])
        after = d2l._rag_fix_prompt('Q?', hits, 'rule_after')
        self.assertNotIn(d2l.ABSTAIN, after[0]['content'])
        self.assertTrue(after[1]['content'].endswith(d2l.ABSTAIN))
        self.assertTrue(after[0]['content'].endswith('[2].'))
        with self.assertRaises(ValueError):
            d2l._rag_fix_prompt('Q?', hits, 'other')
        self.assertEqual(d2l._rag_quoted_answer(
            'Quotes: x [1]\nAnswer: It is y [1].'), 'It is y [1].')
        self.assertEqual(d2l._rag_quoted_answer(' plain '), 'plain')

    def test_records(self):
        d = tempfile.mkdtemp()
        a, b = os.path.join(d, 'a.jsonl'), os.path.join(d, 'b.jsonl')
        d2l.write_jsonl(a, [dict(config='x', qid='q1', v=1)])
        d2l.write_jsonl(b, [dict(config='y', qid='q1', v=2)])
        df = d2l._rag_records([a, None, b, os.path.join(d, 'missing')])
        self.assertEqual(len(df), 2)
        self.assertEqual(list(d2l._rag_records([]).columns), ['config', 'qid'])
        with self.assertRaises(ValueError):
            d2l._rag_records([a, a])


class JudgeInEventLoop(unittest.TestCase):
    """Regression: a notebook's kernel runs an event loop, and rag_judge
    with an endpoint judge used asyncio.run, which refuses to start
    inside one."""
    def test_running_loop(self):
        import asyncio

        class Stub:
            async def achat(self, messages, **params):
                return collections.namedtuple('R', 'text finish_reason')(
                    '{"correct": true}', 'stop')
        q = dict(question='Q', reference='R', key_facts=['f'],
                 answerable=True)

        async def main():
            return d2l.rag_judge(Stub(), None, q, dict(answer='A'), [])
        self.assertEqual(asyncio.run(main())['correct'], True)

    def test_one_loop_for_every_batch(self):
        """Regression: each batch on a fresh event loop left the client's
        HTTP connections on closed loops ('Event loop is closed' noise)."""
        fake = d2l.FakeChatServer(0.0, 4)
        api = d2l.ChatAPI('http://127.0.0.1/v1', 'judge', mode='live',
                          http_client=fake.transport())
        q = dict(question='Q', reference='R', key_facts=['f'],
                 answerable=True)
        for _ in range(2):
            d2l.rag_judge(api, None, [q, q], [dict(answer='A'),
                                              dict(answer='B')], [[], []])
        self.assertGreaterEqual(fake.n_requests, 4)
        loops = list(api._sdks)
        self.assertEqual(len(loops), 1)
        self.assertFalse(loops[0].is_closed())


class StubRetriever:
    """A retriever over a test corpus: a ranking per query by hashing,
    every call logged."""
    def __init__(self, c, log):
        self.corpus, self.log, self.rrf_k = c, log, 60
        self.dense, self.reranker = 'stub-embedder', 'stub-reranker'

    def search_batch(self, queries, k=10, mode='hybrid', rerank_k=None):
        self.log.append(('search', mode, tuple(queries)))
        out = []
        for q in queries:
            ids = sorted(self.corpus.ids, key=lambda i: hashlib.sha1(
                (q + i).encode()).hexdigest())[:k]
            out.append([d2l.Hit(i, 1.0, n, '', [], {})
                        for n, i in enumerate(ids, 1)])
        return out

    def release(self):
        self.log.append(('release',))

    def chunk_vectors(self):
        return torch.eye(len(self.corpus))

    def _reranker(self):
        return None, None


class Pipeline(unittest.TestCase):
    """RAGPipeline with a stub retriever and a stub generator: the phase
    order, prepare + answer(contexts=) equal to answer(), the query paths
    of rewrite, hyde and decompose, reranking and abstention."""
    REPLIES = {20: 'rewritten query', 120: 'a hypothetical passage',
               60: 'first part?\nsecond part?\nthird part?'}

    def pipe(self, **kw):
        self.log = []
        c = tokened('t', overlapping)
        p = d2l.RAGPipeline(StubRetriever(c, self.log), None, None,
                            budget_tokens=20, **kw)

        def gen(messages, max_new_tokens, batch_size):
            self.log.append(('generate', max_new_tokens))
            return [(self.REPLIES.get(max_new_tokens, 'It is [1] and [2].'),
                     100, 7, 0.0) for _ in messages]
        p._generate = gen
        return p

    def test_phases_and_contexts(self):
        p = self.pipe(rewrite='generator')
        qs = [dict(qid='dev-1', question='What is A?'), 'What is B?']
        res = p.answer(qs)
        kinds = [e[0] for e in self.log]
        self.assertEqual(kinds, ['generate', 'search', 'release', 'generate'])
        self.assertEqual(self.log[0], ('generate', 20))
        self.assertEqual(self.log[1][2], ('rewritten query',) * 2)
        self.assertEqual((res[0].qid, res[1].qid), ('dev-1', None))
        self.assertEqual(res[0].query_used, 'rewritten query')
        self.assertEqual(res[0].cited_ids, [1, 2])
        self.assertEqual(len(res[0].ranked_ids), len(p.retriever.corpus))
        tokens = sum(p.retriever.corpus[i]['n_tokens']
                     for i in res[0].context_ids)
        self.assertLessEqual(tokens, 20)
        again = p.answer(qs, contexts=p.prepare(qs))
        strip = lambda r: r._replace(latency_s=0)
        self.assertEqual([strip(r) for r in res], [strip(r) for r in again])
        closed = p.answer(['Q?'], contexts=[dict(context=None)])[0]
        self.assertEqual((closed.context_ids, closed.cited_ids), ([], []))

    def test_query_paths(self):
        p = self.pipe(hyde=True)
        p.prepare(['Q?'])
        self.assertIn(('search', 'bm25', ('Q?',)), self.log)
        self.assertIn(('search', 'dense', ('a hypothetical passage',)),
                      self.log)
        p = self.pipe(decompose=True)
        p.prepare(['Q?'])
        self.assertIn(('search', 'hybrid', ('first part?', 'second part?')),
                      self.log)
        p = self.pipe(rewrite='fuse')
        out = p.prepare(['Q?'])
        self.assertIn(('search', 'hybrid', ('rewritten query',)), self.log)
        self.assertEqual(out[0]['query_used'], 'rewritten query')
        p = self.pipe()  # no extra ranking: the retriever's own fusion
        p.prepare(['Q?'])
        self.assertEqual(self.log, [('search', 'hybrid', ('Q?',))])
        with self.assertRaises(ValueError):
            self.pipe(rewrite='always')

    def test_rerank_and_abstain(self):
        old = d2l.rerank_yes_no
        d2l.rerank_yes_no = lambda m, t, q, docs, instruction=None: (
            np.linspace(0.9, 0.1, len(docs))[::-1])  # the last is best
        try:
            p = self.pipe(rerank_k=3, k=1)
            out = p.prepare(['Q?'])[0]
            first = p.retriever.search_batch(['Q?'], k=100)[0]
            self.assertEqual(out['context'][0]['id'], first[2].chunk_id)
            self.assertAlmostEqual(out['top_score'], 0.9)
            p = self.pipe(abstain_below=0.95)
            prep = p.prepare(['Q?'])
            self.assertTrue(prep[0]['abstain'])
            res = p.answer(['Q?'], contexts=prep)[0]
            self.assertTrue(res.abstained)
            self.assertNotIn(('generate', 300), self.log)
        finally:
            d2l.rerank_yes_no = old


    def test_verify(self):
        """verify: a sentence stays when P(yes) > P(no) for its cited
        passages; one without a citation goes; an emptied answer abstains;
        an abstention is not checked."""
        seen = []

        def lp(model, tok, prompts, labels, **kw):
            seen.extend(prompts)
            return torch.tensor([[-0.1, -2.0] if 'Keep' in m[0]['content']
                                 else [-2.0, -0.1] for m in prompts])
        old = d2l.label_logprobs
        d2l.label_logprobs = lp
        try:
            p = self.pipe(verify=True)
            ctx = [dict(text='alpha'), dict(text='beta')]
            prep = {0: dict(context=ctx), 1: dict(context=ctx),
                    2: dict(context=ctx)}
            out = p._verified({0: 'Keep this [1]. Drop that [2]. No cite.',
                               1: 'Drop it [1].', 2: d2l.ABSTAIN}, prep, 8)
        finally:
            d2l.label_logprobs = old
        self.assertEqual(out, {0: 'Keep this [1].', 1: d2l.ABSTAIN,
                               2: d2l.ABSTAIN})
        self.assertEqual(len(seen), 3)  # two sentences of 0, one of 1
        self.assertIn('[1] alpha', seen[0][0]['content'])
        self.assertNotIn('beta', seen[0][0]['content'])


@unittest.skipUnless(os.environ.get('D2L_TEST_NET') == '1',
                     'network/model test: set D2L_TEST_NET=1')
class PipelineGeneration(unittest.TestCase):
    """The plan's oracle: the pipeline's batched Transformers generation
    (left padding, logits of the last position at the prefill) gives the
    same tokens as d2l.generate's own loop on Qwen3-0.6B (CPU, float32)."""
    def test_against_generate(self):
        model, tok = d2l.load_hf('Qwen/Qwen3-0.6B', dtype=torch.float32,
                                 device_map='cpu')
        p = d2l.RAGPipeline(StubRetriever(tokened('t', halves), []), model,
                            tok)
        msgs = [d2l.rag_prompt(q, None) for q in (
            'What is a tensor?', 'Why does dropout help a network '
            'generalize, and when does it hurt?', 'Define softmax.',
            'How does the learning rate of minibatch stochastic gradient '
            'descent interact with the batch size in practice?')]
        ours = p._generate(msgs, 32, 4)
        ref = d2l.generate(model, tok, msgs, max_new_tokens=32, chat=True,
                           enable_thinking=False, batch_size=4)
        self.assertEqual([o[0] for o in ours], [g.text.strip() for g in ref])


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


class RunTables(unittest.TestCase):
    """The configuration tables and run plumbing that sec_lm_rag and
    sec_lm_rag_fixes share (rag-lib.md): the tables, the run order of
    D2L_RERUN, where a run file is read from, the settings JSON, the
    answer key, and the readers of answer and judgment records."""
    GEN_REV = {d2l.RAG_GENERATOR: 'c' * 40}  # the notebooks pin it

    def test_tables(self):
        C = d2l.RAG_CONFIGS
        self.assertIs(C['base'], d2l.RAG_BASE)
        self.assertEqual(len(C), 10 + len(d2l.RAG_FIXES))
        self.assertTrue(set(d2l.RAG_STAGES + d2l.RAG_CONTROLS) <= set(C))
        for c, fix in d2l.RAG_FIXES.items():
            self.assertIs(C[f'fix-{c}'], fix)
        self.assertEqual(set(d2l.RAG_FIXES), {
            'not_retrieved', 'ranked_low', 'ignored', 'unsupported',
            'false_answer', 'false_abstain'})
        for name, c in C.items():  # every entry has the baseline's keys
            self.assertTrue(set(d2l.RAG_BASE) <= set(c), name)
        self.assertEqual(C['closed']['context'], 'none')
        self.assertEqual(C['fixed-over']['over'], 255)

    def test_rerun_order(self):
        def rerun(value):
            with unittest.mock.patch.dict(os.environ, {'D2L_RERUN': value}):
                return d2l._rag_rerun()
        self.assertEqual(rerun('0'), set())
        self.assertEqual(rerun('rag-judge-cache'),
                         {'rag-judge-cache', 'rag-judge'})
        self.assertEqual(rerun('rag-cards,rag-gen-stages'),
                         {'rag-cards', 'rag-gen-stages'})
        for bad in ('rag-judge,rag-gen-stages', 'rag-judge-cache,rag-gen-fix',
                    'rag-gen-fix,rag-gen-base', '1'):
            with self.assertRaises(AssertionError):
                rerun(bad)

    def test_run_path(self):
        self.assertEqual(d2l._rag_run_path('rag-cards', {'rag-cards'}),
                         os.path.join('..', 'data', 'lm', 'rerun',
                                      'lm-rag-cards.jsonl'))
        with unittest.mock.patch.dict(d2l.DATA_HUB, {}, clear=True):
            self.assertIsNone(d2l._rag_run_path('rag-cards', set()))

    def test_settings(self):
        with unittest.mock.patch.dict(d2l.HF_REVISIONS, self.GEN_REV):
            s = json.loads(d2l._rag_settings('fixed-over'))
            self.assertEqual(d2l._rag_settings('base'),
                             d2l._rag_settings('base'))
            self.assertNotEqual(d2l._rag_settings('base'),
                                d2l._rag_settings('reverse'))
        self.assertEqual(s['generator'], f'{d2l.RAG_GENERATOR}@' + 'c' * 40)
        self.assertEqual((s['over'], s['qa'], s['corpus']),
                         (255, 'lm-book-qa-v2', d2l.R3_VERSION))
        self.assertTrue(s['reranker'].startswith(d2l.RAG_RERANKER + '@'))
        self.assertEqual(list(s), sorted(s))  # sorted keys: one string

    def test_answer_key(self):
        r = dict(answer='It is A [1].', context_ids=['a', 'b'],
                 cited_ids=[1])
        k = d2l._rag_answer_key(r)
        self.assertRegex(k, '^[0-9a-f]{16}$')
        self.assertEqual(k, d2l._rag_answer_key(dict(
            r, cited_ids=np.array([1]), context_ids=('a', 'b'))))
        for change in (dict(answer='It is B [1].'), dict(cited_ids=[2]),
                       dict(context_ids=['b', 'a'])):
            self.assertNotEqual(k, d2l._rag_answer_key(dict(r, **change)))

    def test_load_records(self):
        with tempfile.TemporaryDirectory() as d, \
                unittest.mock.patch.dict(d2l.HF_REVISIONS, self.GEN_REV):
            os.makedirs(os.path.join(d, 'nb'))
            rerun = os.path.join(d, 'data', 'lm', 'rerun')
            os.makedirs(rerun)
            cwd = os.getcwd()
            os.chdir(os.path.join(d, 'nb'))  # notebooks run one level down
            try:
                rec = lambda c, q, a: dict(
                    config=c, qid=q, split='dev', answer=a, context_ids=['x'],
                    cited_ids=[1], settings=d2l._rag_settings(c))
                d2l.write_jsonl(os.path.join(rerun, 'lm-rag-gen-base.jsonl'),
                                [rec('base', 'q1', 'A [1].'),
                                 rec('base', 'q2', d2l.ABSTAIN)])
                d2l.write_jsonl(os.path.join(
                    rerun, 'lm-rag-gen-stages.jsonl'),
                    [rec('reverse', 'q1', 'B [1].')])
                names = {'rag-gen-base', 'rag-gen-stages', 'rag-gen-fix'}
                g = d2l._rag_load_gens(names)
                self.assertEqual(len(g), 3)
                self.assertEqual(g.abstained.tolist(), [False, True, False])
                self.assertEqual(g.answer_sha[0], d2l._rag_answer_key(
                    g.to_dict('records')[0]))
                d2l.write_jsonl(  # a record made with other settings
                    os.path.join(rerun, 'lm-rag-gen-fix.jsonl'),
                    [dict(rec('reverse', 'q2', 'C'),
                          settings=d2l._rag_settings('base'))])
                with self.assertRaises(AssertionError):
                    d2l._rag_load_gens(names)
                path = os.path.join(rerun, 'judge.jsonl')
                d2l.write_jsonl(path, [
                    dict(config='base', qid='q1', concurrency=8,
                         correct=True, faithful=None, supported=False),
                    dict(config='base', qid='q2', concurrency=8,
                         correct=float('nan'), faithful=1, supported=0,
                         max_new_tokens=6000, backend='local')])
                j = d2l._rag_load_judgments(path)
            finally:
                os.chdir(cwd)
        self.assertEqual(j.correct.tolist(), [True, None])
        self.assertEqual(j.faithful.tolist(), [None, True])
        self.assertEqual(j.supported.tolist(), [False, False])
        self.assertEqual(j.max_new_tokens.tolist(), [
            d2l.RAG_JUDGE_LOCAL['max_new_tokens'], 6000])
        self.assertEqual(j.backend.tolist()[1], 'local')


class RunConfig(unittest.TestCase):
    """_rag_run_config with a stub pipeline and generator: the arms'
    contexts (closed: none; oracle: the gold chunks of answerable items),
    other batches in item order again, budgets only for a 'base' budget,
    and the generator freed before retrieval when it may not stay."""
    def setUp(self):
        self.c = tokened('t', halves)
        self.log = []
        log = self.log

        class Pipe:
            def __init__(self, retriever, model, tokenizer, **kw):
                log.append(('init', kw))
                self.retriever = retriever

            def prepare(self, items, budgets=None):
                log.append(('prepare', [q['qid'] for q in items], budgets))
                return [dict(context=[dict(id='retrieved')])
                        for _ in items]

            def answer(self, items, batch_size, contexts):
                log.append(('answer', [q['qid'] for q in items],
                            batch_size))
                R = collections.namedtuple('R', 'qid question answer ctx')
                return [R(q['qid'], q['question'], 'A', [
                    h['id'] for h in p['context'] or []])
                    for q, p in zip(items, contexts)]

        self.patches = [
            unittest.mock.patch.object(d2l, 'RAGPipeline', Pipe),
            unittest.mock.patch.object(d2l, '_rag_generator',
                                       lambda: ('model', 'tok')),
            unittest.mock.patch.object(d2l, '_rag_free_generator',
                                       lambda: log.append(('free',))),
            unittest.mock.patch.object(d2l, '_rag_settings',
                                       lambda name: f'settings of {name}'),
            unittest.mock.patch.object(d2l, 'assemble_context',
                                       lambda hits, budget, k=None: (
                                           list(hits), 0))]
        for p in self.patches:
            p.start()
        un = dict(ITEM, qid='dev-0001', answerable=False, gold_spans=[])
        self.items = [dict(ITEM, question=f'Q{i}?', qid=f'dev-000{i}')
                      for i in (2, 3, 4)] + [dict(un, question='U?')]
        self.ret = {'sections': types.SimpleNamespace(corpus=self.c),
                    'fixed': types.SimpleNamespace(corpus=self.c)}

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_arms(self):
        run = lambda name, **kw: d2l._rag_run_config(
            name, self.items, self.ret, True, **kw)
        ids = [q['qid'] for q in self.items]
        r = run('base')
        self.assertEqual([x['qid'] for x in r], ids)
        self.assertEqual(r[0]['settings'], 'settings of base')
        self.assertNotIn('question', r[0])
        self.assertEqual(r[0]['ctx'], ['retrieved'])
        init = [e for e in self.log if e[0] == 'init'][0][1]
        self.assertEqual((init['k'], init['budget_tokens'], init['depth']),
                         (5, 3000, 50))
        closed = run('closed')
        self.assertEqual({tuple(x['ctx']) for x in closed}, {()})
        oracle = run('oracle')  # the gold chunks of both spans
        gold = [i for s in ITEM['gold_spans'] for i in
                self.c.span_to_chunk_ids(s['doc_id'], s['char_start'],
                                         s['char_end'])]
        self.assertEqual(oracle[0]['ctx'], list(dict.fromkeys(gold)))
        self.assertEqual(oracle[-1]['ctx'], ['retrieved'])  # unanswerable
        self.log.clear()
        rep = run('base-repeat')  # other batches, records in item order
        self.assertEqual([x['qid'] for x in rep], ids)
        asked = [e for e in self.log if e[0] == 'answer'][0][1]
        self.assertEqual(sorted(asked), sorted(ids))
        self.assertNotEqual(asked, ids)
        with self.assertRaises(AssertionError):  # 'base' budget: needs them
            run('fixed')
        self.log.clear()
        run('fixed', budgets={q: 10 for q in ids})
        self.assertEqual(self.log[1], ('prepare', ids, [10] * 4))
        with self.assertRaises(AssertionError):
            run('base', budgets={q: 10 for q in ids})

    def test_generator_memory(self):
        d2l._rag_run_config('base', self.items, self.ret, False)
        self.assertEqual(self.log[0], ('free',))  # off the card first
        self.log.clear()
        d2l._rag_run_config('base', self.items, self.ret, True)
        self.assertNotIn(('free',), self.log)
        with self.assertRaises(AssertionError):  # the rewriter needs it
            d2l._rag_run_config('rewrite-rerank', self.items, self.ret,
                                False)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), unittest.mock.patch.object(
                d2l, 'lm_peak', lambda: 'peak'):
            recs = d2l._rag_run_configs(['base', 'closed'], self.items,
                                        self.ret, True)
        self.assertEqual(len(recs), 8)
        self.assertRegex(out.getvalue(), r'^base: 0\.0 min, peak\n'
                                         r'closed: 0\.0 min, peak\n$')


class Scores(unittest.TestCase):
    """The scores of the RAG sections: the documents of the items, the
    bootstrap strings (* below 100 questions, + below 30 documents, with a
    sign-flip p for a paired difference), per-question scores of the judged
    answers, and the readiness check."""
    def test_clusters(self):
        c = tokened('t', halves)
        items = [ITEM, dict(ITEM, qid='dev-0001', gold_spans=[],
                            anchor_ids=['t:chapter_b/three#001'])]
        self.assertEqual(d2l._rag_clusters(items, c), {
            'dev-0000': 'chapter_a/one', 'dev-0001': 'chapter_b/three'})

    def test_boot(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 200).astype(float)
        many = np.arange(200) % 40  # 40 documents
        self.assertRegex(d2l._rag_boot(a, cluster=many),
                         r'^0\.\d{3} \[0\.\d{3}, 0\.\d{3}\]$')
        self.assertTrue(d2l._rag_boot(a[:50], cluster=many[:50])
                        .endswith('*'))
        few = np.arange(200) % 10  # 10 documents: questions resampled
        self.assertTrue(d2l._rag_boot(a, cluster=few).endswith('+'))
        b = rng.integers(0, 2, 200).astype(float)
        self.assertRegex(d2l._rag_boot(a, b, few),
                         r'^[+-]0\.\d{3} .*\+ \(sign-flip p [01]\.\d{3}\)$')
        self.assertEqual(d2l._rag_boot([1.0, np.nan], cluster=[0, 1]), 'n/a')
        r, by_doc, x, y, cl = d2l._rag_ci([1, np.nan, 0, 1], [0, 1, np.nan,
                                                              1], [0, 1, 2, 3])
        self.assertEqual((x.tolist(), y.tolist(), cl.tolist(), by_doc),
                         ([1.0, 1.0], [0.0, 1.0], [0, 3], False))
        p = d2l._rag_signflip_p(np.array([1.0, 1.0, 1.0]),
                                np.array([0, 1, 2]))
        self.assertAlmostEqual(p, 0.25, delta=0.02)  # 2 of 8 sign patterns

    def tables(self):
        c = tokened('t', halves)
        rel = 't:chapter_a/one#001'  # relevant to ITEM
        other = 't:chapter_b/three#000'
        un = dict(ITEM, qid='dev-0003', answerable=False, gold_spans=[],
                  flags=[], type='unanswerable')
        items = {q['qid']: q for q in [
            dict(ITEM, qid='dev-0001', flags=[]),
            dict(ITEM, qid='dev-0002', flags=['general_knowledge']), un]}
        gens = pd.DataFrame([
            dict(config='base', split='dev', qid='dev-0001',
                 answer='A [1].', context_ids=[rel], ranked_ids=[rel],
                 cited_ids=[1], abstained=False, n_prompt_tokens=10,
                 n_new_tokens=3, latency_s=0.1),
            dict(config='base', split='dev', qid='dev-0002', answer='B',
                 context_ids=[other], ranked_ids=[other, rel], cited_ids=[],
                 abstained=False, n_prompt_tokens=10, n_new_tokens=3,
                 latency_s=0.1),
            dict(config='base', split='dev', qid='dev-0003',
                 answer=d2l.ABSTAIN, context_ids=[other], ranked_ids=[other],
                 cited_ids=[], abstained=True, n_prompt_tokens=10,
                 n_new_tokens=3, latency_s=0.1)])
        gens['answer_sha'] = [d2l._rag_answer_key(r)
                              for r in gens.to_dict('records')]
        jud = pd.DataFrame([
            dict(config='base', qid='dev-0001', concurrency=8, correct=True,
                 faithful=True, supported=True),
            dict(config='base', qid='dev-0002', concurrency=8, correct=None,
                 faithful=None, supported=None),  # unjudged
            dict(config='base', qid='dev-0003', concurrency=8, correct=True,
                 faithful=True, supported=None),
            dict(config='base', qid='dev-0001', concurrency=16,
                 correct=False, faithful=False, supported=False)])
        jud['answer_sha'] = [dict(zip(gens.qid, gens.answer_sha))[q]
                             for q in jud.qid]
        ret = {'sections': types.SimpleNamespace(corpus=c)}
        cluster = {q: 'doc-' + q for q in items}
        return gens, jud, ret, items, cluster

    def test_scores(self):
        gens, jud, ret, items, cluster = self.tables()
        s = d2l._rag_scores('base', 'dev', gens, jud, ret, items, cluster, 8)
        self.assertEqual(s.qid.tolist(), ['dev-0001', 'dev-0002', 'dev-0003'])
        self.assertEqual(s.accuracy.tolist()[0], 1.0)  # concurrency 8 only
        self.assertTrue(np.isnan(s.accuracy[1]))  # unjudged: left out
        self.assertEqual(s.hit.tolist()[:2], [1.0, 0.0])
        self.assertTrue(np.isnan(s.hit[2]))  # unanswerable
        self.assertEqual(s['recall@5'].tolist()[:2], [1.0, 1.0])
        self.assertTrue(np.isnan(s.faithful[2]))  # an abstention
        self.assertEqual(s.book_specific.tolist(), [True, False, True])
        self.assertEqual(s.cite_precision.tolist()[0], 1.0)
        self.assertEqual(s.cluster.tolist()[0], 'doc-dev-0001')
        self.assertEqual(s.verdicts[0], dict(correct=True, faithful=True,
                                             supported=True))
        a = s.assign(accuracy=[1.0, 1.0, 1.0])  # paired: rows 1 and 3
        self.assertRegex(d2l._rag_paired(a, s, 'accuracy'),
                         r'^\+0\.000 .*\+ \(sign-flip p 1\.000\)$')
        stale = jud.assign(answer_sha=['0' * 16] + list(jud.answer_sha[1:]))
        self.assertEqual(d2l._rag_scores(  # a verdict on another answer
            'base', 'dev', gens, stale, ret, items, cluster).qid.tolist(),
            ['dev-0002', 'dev-0003'])

    def test_ready(self):
        gens, jud, ret, items, cluster = self.tables()
        self.assertTrue(d2l._rag_ready(['base'], 'dev', gens, jud, set()))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertFalse(d2l._rag_ready(['closed'], 'dev', gens, jud,
                                            set()))  # no answers: no error
            self.assertFalse(d2l._rag_ready(['base'], 'dev', gens,
                                            jud.iloc[1:], {'rag-judge'}))
            with self.assertRaises(AssertionError):  # registered, unjudged
                d2l._rag_ready(['base'], 'dev', gens, jud.iloc[1:], set())
        self.assertIn("'closed': 'no answers'", out.getvalue())
        self.assertIn("'base': '1 of 3 answers lack a verdict'",
                      out.getvalue())


if __name__ == '__main__':
    unittest.main()
