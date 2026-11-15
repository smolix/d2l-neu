#!/usr/bin/env python3
"""CPU tests for sec_lm_retrieval (chapter_lm-encoders/retrieval.md):
`bm25_tokenize` cases; `BM25Index` against `bm25s` (method 'lucene', the same
tokens) on a 300-document fixture (rank-1 ids, scores to 1e-4 relative);
`rrf_fuse` by hand; `kmeans` (inertia never rises with more iterations,
seeded runs repeat, each row's label is its nearest centroid); `IVFIndex`
probing every cell equals exact search; `ProductQuantizer` (table distances
equal the distances to the reconstructions, the hand case of the text, codes
equal Faiss `IndexPQ`'s given the same codebooks); `Hit`;
`BookRetriever(dense=None)` on a hand-made corpus, whose `bm25_params=None`
resolves to the stored defaults. Network-gated
(D2L_TEST_NET=1 and the model in the HF cache): `rerank_yes_no` against the
Qwen3-Reranker-0.6B card's reference code on 8 pairs (CPU float32, P(yes) to
1e-4)."""
import os
import unittest

import numpy as np
import torch
from d2l import torch as d2l

NET = os.environ.get('D2L_TEST_NET') == '1'


def fixture_docs(n_docs=300, seed=0):
    """Documents of Zipf-distributed words, and 200 queries."""
    rng = np.random.default_rng(seed)
    vocab = [f'w{i}' for i in range(2000)]
    p = 1 / np.arange(1, 2001) ** 1.1
    p /= p.sum()
    docs = [list(rng.choice(vocab, size=rng.integers(5, 200), p=p))
            for _ in range(n_docs)]
    queries = [list(rng.choice(vocab, size=rng.integers(1, 6), p=p ** 0.5
                               / (p ** 0.5).sum())) for _ in range(200)]
    return docs, queries


def toy_corpus():
    """A three-document BookCorpus, one chunk per paragraph."""
    texts = {'chapter_a/one': 'The key-value cache stores keys and values '
             'per token.\n\nAttention reads the cache.',
             'chapter_a/two': 'BM25 scores documents by term frequency.\n\n'
             'The d2l.book_corpus function loads the corpus.',
             'chapter_b/three': 'Gradient descent follows the negative '
             'gradient.'}
    docs, chunks = [], []
    for doc_id, text in texts.items():
        docs.append(dict(doc_id=doc_id, text=text, title=doc_id,
                         chapter_title=doc_id.split('/')[0], headings=[]))
        start = 0
        for k, para in enumerate(text.split('\n\n')):
            a = text.index(para, start)
            chunks.append(dict(id=f'sec512:{doc_id}#{k:03d}', doc_id=doc_id,
                               text=para, heading_path=[doc_id],
                               char_start=a, char_end=a + len(para)))
            start = a + len(para)
    manifest = dict(corpus_version='r3.1', doc_split={
        d: 'train' for d in texts})
    return d2l.BookCorpus(manifest, docs, chunks, 'sec512')


class Tokenize(unittest.TestCase):
    def test_cases(self):
        f = d2l.bm25_tokenize
        self.assertEqual(f('Hello, World!'), ['hello', 'world'])
        self.assertEqual(f('d2l.book_corpus'),
                         ['d2l.book_corpus', 'd2l', 'book', 'corpus'])
        self.assertEqual(f('BookRetriever'),
                         ['bookretriever', 'book', 'retriever'])
        self.assertEqual(f('HTTPServer'), ['httpserver', 'http', 'server'])
        self.assertEqual(f('top_k=5 and 3.14'),
                         ['top_k', 'top', 'k', '5', 'and', '3.14'])
        self.assertEqual(f('d2l.book_corpus', split_identifiers=False),
                         ['d2l.book_corpus'])
        self.assertEqual(f('end of sentence.'), ['end', 'of', 'sentence'])


class BM25(unittest.TestCase):
    def test_against_bm25s(self):
        try:
            import bm25s
        except ImportError as e:
            self.skipTest(f'bm25s not installed: {e}')
        docs, queries = fixture_docs()
        ours = d2l.BM25Index(docs, k1=1.2, b=0.75)
        ref = bm25s.BM25(method='lucene', k1=1.2, b=0.75)
        vocab = {t: i for i, t in enumerate(sorted({t for d in docs
                                                    for t in d}))}
        ref.index(bm25s.tokenization.Tokenized(
            ids=[[vocab[t] for t in d] for d in docs], vocab=vocab),
            show_progress=False)
        agree, worst = 0, 0.0
        queries = [q for q in queries if any(t in vocab for t in q)]
        for q in queries:
            s = ours.scores(q)
            r = ref.get_scores([vocab[t] for t in q if t in vocab])
            worst = max(worst, np.max(np.abs(s - r) / np.maximum(
                np.abs(r), 1e-6)))
            agree += int(np.argmax(s) == np.argmax(r))
        self.assertGreaterEqual(agree / len(queries), 0.99)
        self.assertLess(worst, 1e-4)

    def test_search_and_formula(self):
        docs = [['a', 'b', 'a'], ['b', 'c'], ['c', 'c', 'c', 'd']]
        idx = d2l.BM25Index(docs, k1=1.2, b=0.75)
        avg = np.mean([3, 2, 4])
        idf = lambda df: np.log(1 + (3 - df + 0.5) / (df + 0.5))
        want = idf(1) * 2 / (2 + 1.2 * (0.25 + 0.75 * 3 / avg))
        self.assertAlmostEqual(idx.scores(['a'])[0], want, places=12)
        ids, s = idx.search(['c'], 2)
        self.assertEqual(list(ids), [2, 1])
        self.assertTrue(np.all(idx.scores(['zzz']) == 0))
        self.assertAlmostEqual(idx.scores(['c', 'c'])[2],
                               2 * idx.scores(['c'])[2])


class Fusion(unittest.TestCase):
    def test_rrf_by_hand(self):
        got = dict(d2l.rrf_fuse([['a', 'b', 'c'], ['b', 'c', 'd'],
                                 ['c', 'a']], k=60))
        want = {'a': 1 / 61 + 1 / 62, 'b': 1 / 62 + 1 / 61,
                'c': 1 / 63 + 1 / 62 + 1 / 61, 'd': 1 / 63}
        self.assertEqual(set(got), set(want))
        for i in want:
            self.assertAlmostEqual(got[i], want[i], places=15)
        order = [i for i, _ in d2l.rrf_fuse([['a', 'b', 'c'],
                                             ['b', 'c', 'd'], ['c', 'a']])]
        self.assertEqual(order, ['c', 'a', 'b', 'd'])  # tie: first seen
        w = dict(d2l.rrf_fuse([['a'], ['b']], k=1, weights=[2, 1]))
        self.assertEqual(w, {'a': 1.0, 'b': 0.5})


def blobs(n=2000, d=16, k=8, seed=0):
    """n points around k random centres in d dimensions, float32."""
    rng = np.random.default_rng(seed)
    centres = 3 * rng.standard_normal((k, d))
    return torch.tensor(centres[rng.integers(k, size=n)]
                        + rng.standard_normal((n, d)), dtype=torch.float32)


class KMeans(unittest.TestCase):
    def test_inertia_seed_and_labels(self):
        X = blobs()
        inertia = []
        for n_iter in (0, 1, 2, 5, 10):
            C, a = d2l.kmeans(X, 8, n_iter=n_iter, seed=0)
            inertia.append(((X - C[a]) ** 2).sum().item())
        for before, after in zip(inertia, inertia[1:]):
            self.assertLessEqual(after, before * (1 + 1e-6))
        C1, a1 = d2l.kmeans(X, 8, n_iter=10, seed=0)
        self.assertTrue(torch.equal(C, C1) and torch.equal(a, a1))
        self.assertTrue(torch.equal(torch.cdist(X, C).argmin(1), a))
        C2, _ = d2l.kmeans(X, 8, n_iter=10, seed=1)
        self.assertEqual(C2.shape, (8, 16))

    def test_ivf_all_cells_is_exact(self):
        X, Q = blobs(), blobs(50, seed=1)
        ivf = d2l.IVFIndex(16, nprobe=16)
        ivf.train(X)
        ivf.add(X)
        D, I = ivf.search(Q, 5)
        exact = ((Q[:, None] - X) ** 2).sum(-1).topk(5, largest=False)
        self.assertTrue(torch.equal(I, exact.indices))
        torch.testing.assert_close(D, exact.values, rtol=1e-5, atol=1e-5)
        ivf.nprobe = 1  # fewer cells: ids still valid, some neighbours lost
        _, I1 = ivf.search(Q, 5)
        self.assertTrue(bool(((I1 >= -1) & (I1 < len(X))).all()))


class PQ(unittest.TestCase):
    def test_adc_equals_reconstruction_distance(self):
        X, Y = blobs(3000, 32), blobs(10, 32, seed=2)
        pq = d2l.ProductQuantizer(4, nbits=8).fit(X)
        codes = pq.encode(X)
        self.assertEqual((codes.dtype, tuple(codes.shape)),
                         (torch.uint8, (3000, 4)))
        Xh = pq.decode(codes)
        for y in Y:
            table = pq.adc_table(y)
            self.assertEqual(tuple(table.shape), (4, 256))
            adc = table[torch.arange(4), codes.long()].sum(1).double()
            want = ((y.double() - Xh.double()) ** 2).sum(1)
            self.assertLess(((adc - want).abs() / want).max().item(), 1e-5)
            D, I = pq.adc_search(y, codes, 5)
            self.assertTrue(torch.equal(I, adc.topk(5, largest=False).indices))
        D, I = pq.adc_search(Y, codes, 5)
        self.assertEqual(tuple(I.shape), (10, 5))
        try:
            import faiss
        except ImportError as e:
            self.skipTest(f'faiss not installed: {e}')
        index = faiss.IndexPQ(32, 4, 8)
        faiss.copy_array_to_vector(pq.codebooks.numpy().ravel(),
                                   index.pq.centroids)
        index.is_trained = True
        same = (index.sa_encode(X.numpy()) == codes.numpy()).all(1).mean()
        self.assertGreaterEqual(same, 0.999)

    def test_hand_case(self):
        pq = d2l.ProductQuantizer(2, nbits=1)
        pq.codebooks = torch.tensor([[[1., 0.], [0., 1.]],
                                     [[1., 1.], [-1., 0.]]])
        code = pq.encode(torch.tensor([[0.9, 0.2, -0.8, 0.1]]))
        self.assertEqual(code[0].tolist(), [0, 1])
        self.assertEqual(pq.decode(code)[0].tolist(), [1., 0., -1., 0.])
        table = pq.adc_table(torch.tensor([1., 1., 0., 0.]))
        self.assertEqual(table.tolist(), [[1., 1.], [2., 1.]])
        with self.assertRaises(ValueError):
            d2l.ProductQuantizer(3).encode(torch.zeros(2, 4))


class Retriever(unittest.TestCase):
    def test_hit_fields(self):
        self.assertEqual(d2l.Hit._fields, ('chunk_id', 'score', 'rank',
                                           'text', 'heading_path', 'scores'))

    def test_bm25_only(self):
        corpus = toy_corpus()
        r = d2l.BookRetriever(corpus, dense=None)
        self.assertEqual(r.bm25_params, d2l.BookRetriever.BM25_PARAMS)
        hits = r.search('What does the key-value cache store?', k=2)
        self.assertEqual(hits[0].chunk_id, 'sec512:chapter_a/one#000')
        self.assertEqual([h.rank for h in hits], [1, 2])
        self.assertIn('bm25', hits[0].scores)
        self.assertEqual(hits[0].score, hits[0].scores['bm25'])
        hits = r.search('book_corpus function', k=1)
        self.assertEqual(hits[0].chunk_id, 'sec512:chapter_a/two#001')
        batch = r.search_batch(['gradient', 'attention'], k=1)
        self.assertEqual([b[0].chunk_id for b in batch],
                         ['sec512:chapter_b/three#000',
                          'sec512:chapter_a/one#001'])
        with self.assertRaises(ValueError):
            r.search('x', mode='dense')
        with self.assertRaises(ValueError):
            r.search('x', rerank_k=5)
        r2 = d2l.BookRetriever(corpus, dense=None, bm25_params=dict(
            k1=0.9, b=0.4, split_identifiers=False))
        self.assertEqual(r2.bm25.k1, 0.9)

    def test_iterate_corpus(self):
        corpus = toy_corpus()
        self.assertEqual([c['id'] for c in corpus], corpus.ids)
        self.assertEqual(len(list(corpus)), len(corpus))
        self.assertIs(corpus[corpus.ids[1]], list(corpus)[1])

    def test_span_to_chunk_ids(self):
        corpus = toy_corpus()
        c = corpus['sec512:chapter_a/one#000']
        self.assertEqual(corpus.span_to_chunk_ids(
            'chapter_a/one', c['char_start'] + 4, c['char_end'] - 4),
            [c['id']])
        n = corpus.docs['chapter_a/one']['text']
        self.assertEqual(len(corpus.span_to_chunk_ids(
            'chapter_a/one', 0, len(n))), 2)


@unittest.skipUnless(NET, 'network/model test: set D2L_TEST_NET=1')
class Reranker(unittest.TestCase):
    def test_against_card(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = 'Qwen/Qwen3-Reranker-0.6B'
        rev = d2l.HF_ENCODER_REVISIONS[mid]
        tok = AutoTokenizer.from_pretrained(mid, revision=rev,
                                            padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(
            mid, revision=rev, dtype=torch.float32).eval()
        task = ('Given a web search query, retrieve relevant passages that '
                'answer the query')
        queries = ['What is the capital of China?', 'Explain gravity'] * 4
        docs = ['The capital of China is Beijing.',
                'Gravity is a force that attracts two bodies towards each '
                'other.', 'Paris is in France.', 'Bananas are yellow.',
                'Beijing hosted the 2008 Olympics.', 'Newton described it.',
                'China is a country in Asia.', 'Apples fall from trees.']
        prefix = ('<|im_start|>system\nJudge whether the Document meets the '
                  'requirements based on the Query and the Instruct '
                  'provided. Note that the answer can only be "yes" or '
                  '"no".<|im_end|>\n<|im_start|>user\n')
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        want = []
        for q, d in zip(queries, docs):  # the card's code, one pair a time
            ids = enc(prefix) + enc(f'<Instruct>: {task}\n<Query>: {q}\n'
                                    f'<Document>: {d}') + enc(suffix)
            with torch.no_grad():
                z = model(torch.tensor([ids])).logits[0, -1]
            yes, no = tok.convert_tokens_to_ids('yes'), \
                tok.convert_tokens_to_ids('no')
            want.append(torch.softmax(z[[no, yes]], 0)[1].item())
        got = [d2l.rerank_yes_no(model, tok, q, [d])[0]
               for q, d in zip(queries, docs)]
        np.testing.assert_allclose(got, want, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
