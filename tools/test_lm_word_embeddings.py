#!/usr/bin/env python3
"""CPU tests for sec_lm_word_embeddings (chapter_lm-encoders/word-embeddings.md):
`cooccurrence_counts` against a double loop, `ppmi` against the dense formula,
`sgns_loss` against masked binary cross-entropy, the shifted-PMI optimum of
free SGNS scores on a toy count matrix, `spearman_eval` against
scipy.stats.spearmanr, `nearest_neighbors` (vocabulary and tokenizer paths),
and `TokenEmbedding` on the fixture `vec.txt` (three words), loaded through a
stubbed `download_extract`. Until the retiring `similarity-analogy.md` is
deleted, the library keeps that section's `TokenEmbedding`, which is the same
code; the test covers whichever is in force."""
import math
import os
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

GLOVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lm',
                     'fixtures', 'encoders', 'glove')


def loop_counts(corpus, V, window, weighting):
    """Reference: every ordered pair within the window, inside a sentence."""
    X = np.zeros((V, V))
    sentences, cur = [], []
    for t in corpus:
        if t < 0:
            sentences.append(cur)
            cur = []
        else:
            cur.append(t)
    sentences.append(cur)
    for s in sentences:
        for i, w in enumerate(s):
            for j, o in enumerate(s):
                d = abs(i - j)
                if 0 < d <= window:
                    X[w, o] += 1.0 if weighting == 'uniform' else 1.0 / d
    return X


class CooccurrenceTest(unittest.TestCase):
    def test_against_loop(self):
        rng = np.random.default_rng(0)
        corpus = rng.integers(0, 7, 50)
        corpus[[9, 10, 23, 41]] = -1  # breaks, two of them adjacent
        for weighting in ('uniform', 'harmonic'):
            for window in (1, 3, 5):
                X = d2l.cooccurrence_counts(corpus, 7, window, weighting)
                ref = loop_counts(corpus, 7, window, weighting)
                np.testing.assert_allclose(X.toarray(), ref, rtol=0,
                                           atol=1e-12)
                np.testing.assert_array_equal(X.toarray(), X.toarray().T)
                self.assertEqual(X.dtype, np.float64)

    def test_no_pair_crosses_a_break(self):
        X = d2l.cooccurrence_counts(np.array([0, 1, -1, 2, 3]), 4, 5)
        self.assertEqual(X[1, 2], 0)
        self.assertEqual(X[0, 3], 0)
        self.assertEqual(X.sum(), 4)  # (0,1), (1,0), (2,3), (3,2)

    def test_bad_weighting(self):
        with self.assertRaises(ValueError):
            d2l.cooccurrence_counts(np.array([0, 1]), 2, 1, 'linear')


class PPMITest(unittest.TestCase):
    def test_against_dense(self):
        import scipy.sparse
        rng = np.random.default_rng(1)
        X = rng.poisson(1.5, (6, 6)).astype(float)
        X = X + X.T
        X[5] = X[:, 5] = 0  # a word that never occurs
        for alpha in (1.0, 0.75, 0.0):
            for shift in (0.0, math.log(5)):
                M = d2l.ppmi(scipy.sparse.csr_matrix(X), alpha, shift)
                total = X.sum()
                n_o = X.sum(0)
                p_o = np.where(n_o > 0, n_o ** alpha, 0)
                p_o = p_o / p_o.sum()
                with np.errstate(divide='ignore', invalid='ignore'):
                    pmi = np.log(X / total / np.outer(X.sum(1) / total, p_o))
                ref = np.maximum(np.nan_to_num(pmi - shift, nan=0.0,
                                               neginf=0.0), 0)
                np.testing.assert_allclose(M.toarray(), ref, rtol=0,
                                           atol=1e-12)
                self.assertTrue(np.all(M.data > 0))


class SGNSLossTest(unittest.TestCase):
    def test_masked_bce(self):
        torch.manual_seed(0)
        v, u = nn.Embedding(9, 4), nn.Embedding(9, 4)
        centers = torch.randint(0, 9, (5,))
        ctx = torch.randint(0, 9, (5, 6))
        labels = torch.tensor([[1, 1, 0, 0, 0, 0]] * 5)
        masks = (torch.rand(5, 6) > 0.3).float()
        masks[:, 0] = 1
        got = d2l.sgns_loss(v, u, centers, ctx, labels, masks)
        scores = (v(centers)[:, None, :] * u(ctx)).sum(-1)
        bce = F.binary_cross_entropy_with_logits(scores, labels.float(),
                                                 reduction='none')
        ref = (bce * masks).sum() / masks.sum()
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)
        # a masked slot does not change the loss, whatever word it holds
        ctx2 = ctx.clone()
        ctx2[masks == 0] = 0
        torch.testing.assert_close(
            d2l.sgns_loss(v, u, centers, ctx2, labels, masks), got)


class ShiftedPMITest(unittest.TestCase):
    """At the optimum of free scores, s*(w, o) = pmi(w, o) - log K +
    log(p(o) / P(o)) (Levy and Goldberg 2014), checked by L-BFGS."""

    def test_toy(self):
        X = np.array([[4., 4., 7., 2.], [4., 2., 5., 13.], [7., 5., 2., 3.],
                      [2., 13., 3., 2.]])
        n_w, n_o, total = X.sum(1), X.sum(0), X.sum()
        pmi = np.log(X * total / np.outer(n_w, n_o))
        for K in (1, 5, 15):
            for alpha in (1.0, 0.75):
                P = n_o ** alpha / (n_o ** alpha).sum()
                Xt, Pt = torch.tensor(X), torch.tensor(P)
                s = torch.zeros_like(Xt, requires_grad=True)
                opt = torch.optim.LBFGS(
                    [s], max_iter=500, tolerance_grad=1e-12,
                    tolerance_change=1e-16, line_search_fn='strong_wolfe')

                def closure():
                    opt.zero_grad()
                    loss = (Xt * F.softplus(-s) + K * Xt.sum(1, keepdim=True)
                            * Pt * F.softplus(s)).sum()
                    loss.backward()
                    return loss
                opt.step(closure)
                want = pmi - math.log(K) + np.log(n_o / total / P)
                np.testing.assert_allclose(s.detach().numpy(), want,
                                           atol=1e-6, rtol=0)


class SpearmanTest(unittest.TestCase):
    def setUp(self):
        self.vocab = d2l.Vocab('a b c d e f g h'.split())
        rng = np.random.default_rng(2)
        words = 'a b c d e f g h zz'.split()  # zz is not covered
        w1, w2 = rng.choice(words, 40), rng.choice(words, 40)
        self.pairs = pd.DataFrame(dict(word1=w1, word2=w2,
                                       score=rng.random(40)))
        self.E = torch.randn(len(self.vocab), 3,
                             generator=torch.Generator().manual_seed(0))

    def test_against_scipy(self):
        from scipy.stats import spearmanr
        r = d2l.spearman_eval(self.E, self.vocab, self.pairs)
        idx = self.vocab.token_to_idx
        p = self.pairs[[a in idx and b in idx for a, b in
                        zip(self.pairs.word1, self.pairs.word2)]]
        cos = F.cosine_similarity(self.E[[idx[w] for w in p.word1]],
                                  self.E[[idx[w] for w in p.word2]]).double()
        self.assertEqual(r.n_covered, len(p))
        self.assertEqual(r.n_total, 40)
        self.assertLess(r.n_covered, 40)
        np.testing.assert_allclose(r.cosines, cos.numpy(), atol=1e-6)
        self.assertAlmostEqual(r.rho, spearmanr(r.cosines, p.score)[0],
                               delta=1e-12)
        self.assertLess(r.ci[0], r.rho)
        self.assertLess(r.rho, r.ci[1])
        again = d2l.spearman_eval(self.E, self.vocab, self.pairs)
        self.assertEqual(r.ci, again.ci)  # seeded

    def test_no_bootstrap(self):
        r = d2l.spearman_eval(self.E, self.vocab, self.pairs, n_boot=0)
        self.assertTrue(all(math.isnan(c) for c in r.ci))


class StubTokenizer:
    """One token per entry of a fixed list ('|' separates tokens)."""
    def __init__(self, words):
        self.words = words

    def encode(self, text, add_special_tokens=False):
        return [self.words.index(w) for w in text.split('|')]

    def decode(self, ids):
        return ''.join(self.words[i] for i in ids)

    def __len__(self):
        return len(self.words)


class NearestNeighborsTest(unittest.TestCase):
    def test_vocab(self):
        vocab = d2l.Vocab('x y z w'.split())
        E = torch.zeros(len(vocab), 2)
        E[0] = torch.tensor([1.0, 0.0])  # <unk> sits on top of x: skipped
        for t, v in zip('xyzw', ([1, 0], [1, 0.1], [0, 1], [-1, 0])):
            E[vocab.token_to_idx[t]] = torch.tensor(v, dtype=torch.float32)
        got = d2l.nearest_neighbors(E, vocab, 'x', k=2)
        self.assertEqual([t for t, _ in got], ['y', 'z'])
        self.assertAlmostEqual(got[1][1], 0.0, places=6)
        got = d2l.nearest_neighbors(E, vocab, 'x', k=1, exclude_self=False)
        self.assertEqual(got[0][0], 'x')

    def test_tokenizer(self):
        tok = StubTokenizer([' a', ' b', ' c'])
        E = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0],
                          [1.0, 0.0]])  # row 3: padding beyond the vocabulary
        got = d2l.nearest_neighbors(E, tok, ' a', k=2)
        self.assertEqual([t for t, _ in got], [' b', ' c'])
        with self.assertRaises(ValueError):
            d2l.nearest_neighbors(E, tok, ' a| b')


class TokenEmbeddingTest(unittest.TestCase):
    def test_fixture(self):
        saved = d2l.download_extract
        d2l.download_extract = lambda name, folder=None: GLOVE
        try:
            emb = d2l.TokenEmbedding('glove.fixture')
        finally:
            d2l.download_extract = saved
        self.assertEqual(len(emb), 4)
        self.assertEqual(emb.idx_to_token, ['<unk>', 'the', 'cat', 'dog'])
        self.assertEqual(emb.token_to_idx['cat'], 2)
        self.assertEqual(emb.unknown_idx, 0)
        torch.testing.assert_close(
            emb[['cat', 'nope']],
            torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]]))


if __name__ == '__main__':
    unittest.main()
