#!/usr/bin/env python3
"""CPU tests for sec_lm_text_embeddings (chapter_lm-encoders/text-embeddings.md),
the shared part: `pool_hidden` against per-example loops with left and right
padding; `retrieval_metrics` against `pytrec_eval` (success@k for recall@k,
reciprocal rank of the run cut at 10, ndcg_cut_10) on a toy run; `embed_texts`
on a stub encoder (order restored, padding side, truncation count);
`info_nce_loss` against `F.cross_entropy` (with hard negatives, symmetric) and
against sentence-transformers' `MultipleNegativesRankingLoss`; its gradient
(P - I) / (tau B) in float64; `anisotropy` against the mean-vector identity and
its sampled estimate; `align_uniform` on four points by hand;
`truncate_embeddings`; `binarize` + `hamming_scores` against sign dot products;
`no_duplicate_batches` (no repeated text in 1,000 batches, every pair once);
`train_embedder` on a tiny random ModernBERT (loss falls, seeds reproduce, the
Matryoshka and hard-negative paths run).
Network-gated (D2L_TEST_NET=1 and the model in the HF cache): `embed_texts`
with Qwen3-Embedding-0.6B against `SentenceTransformer.encode` (cosine
>= 0.999 per text)."""
import contextlib
import io
import os
import unittest

import numpy as np
import torch
from d2l import torch as d2l

NET = os.environ.get('D2L_TEST_NET') == '1'


class Pooling(unittest.TestCase):
    def test_against_loops(self):
        torch.manual_seed(0)
        h = torch.randn(4, 7, 5)
        lens = [7, 3, 5, 1]
        for side in ('right', 'left'):
            mask = torch.zeros(4, 7, dtype=torch.long)
            for i, n in enumerate(lens):
                if side == 'right':
                    mask[i, :n] = 1
                else:
                    mask[i, 7 - n:] = 1
            for mode in ('cls', 'mean', 'last'):
                got = d2l.pool_hidden(h, mask, mode)
                for i in range(4):
                    pos = mask[i].nonzero().flatten()
                    want = {'cls': h[i, pos[0]], 'last': h[i, pos[-1]],
                            'mean': h[i, pos].mean(0)}[mode]
                    torch.testing.assert_close(got[i], want, atol=1e-6,
                                               rtol=0)


class Metrics(unittest.TestCase):
    def test_against_pytrec_eval(self):
        try:
            import pytrec_eval
        except ImportError as e:
            self.skipTest(f'pytrec_eval not installed: {e}')
        rng = np.random.default_rng(0)
        ranked, gold = [], []
        for q in range(30):
            ranked.append([f'd{i}' for i in rng.permutation(40)[:25]])
            gold.append({f'd{i}' for i in rng.choice(40, size=rng.integers(
                1, 4), replace=False)})
        ours = d2l.retrieval_metrics(ranked, gold, ks=(1, 5, 10))
        qrel = {f'q{q}': {d: 1 for d in g} for q, g in enumerate(gold)}
        run = {f'q{q}': {d: float(100 - r) for r, d in enumerate(lst)}
               for q, lst in enumerate(ranked)}
        cut = {q: {d: s for d, s in docs.items() if s > 90}
               for q, docs in run.items()}  # the first ten
        ev = pytrec_eval.RelevanceEvaluator(
            qrel, {'success.1,5,10', 'ndcg_cut.10'})
        rr = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'}).evaluate(cut)
        res = ev.evaluate(run)
        for q in range(30):
            r = res[f'q{q}']
            for k in (1, 5, 10):
                self.assertAlmostEqual(ours[f'recall@{k}'][q],
                                       r[f'success_{k}'], places=9)
            self.assertAlmostEqual(ours['ndcg@10'][q], r['ndcg_cut_10'],
                                   places=9)
            self.assertAlmostEqual(ours['mrr@10'][q],
                                   rr[f'q{q}']['recip_rank'], places=9)

    def test_empty_gold_is_nan(self):
        m = d2l.retrieval_metrics([['a']], [set()], ks=(1,))
        self.assertTrue(np.isnan(m['recall@1'][0]))


class StubEncoder(torch.nn.Module):
    """last_hidden_state[b, t] = (token id, position, 1) as floats."""
    def __init__(self):
        super().__init__()
        self.config = type('C', (), {'hidden_size': 3})()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape
        pos = torch.arange(T).expand(B, T)
        h = torch.stack([input_ids.float(), pos.float(),
                         torch.ones(B, T)], -1)
        return type('O', (), {'last_hidden_state': h})()


class StubTokenizer:
    padding_side, pad_token_id = 'right', 0

    def __call__(self, texts, truncation=False, max_length=None):
        ids = [[len(w) for w in t.split()] for t in texts]
        if truncation:
            ids = [r[:max_length] for r in ids]
        return {'input_ids': ids}


class Embed(unittest.TestCase):
    def test_order_padding_and_truncation(self):
        texts = ['aa b', 'ccc dd eeee fffff', 'g']
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            Z = d2l.embed_texts(StubEncoder(), StubTokenizer(), texts,
                                pooling='last', batch_size=2, max_len=3,
                                normalize=False, padding_side='left')
        self.assertIn('1 of 3 texts truncated to 3 tokens', out.getvalue())
        self.assertEqual(Z[:, 0].tolist(), [1.0, 4.0, 1.0])  # last token
        Z = d2l.embed_texts(StubEncoder(), StubTokenizer(), texts,
                            pooling='mean', batch_size=1, normalize=False)
        self.assertAlmostEqual(Z[1, 0].item(), 3.5)
        self.assertEqual(Z.dtype, torch.float32)


def unit(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(
        torch.randn(*shape, generator=g, dtype=dtype), dim=-1)


class InfoNCE(unittest.TestCase):
    def test_against_cross_entropy(self):
        q, d, h = unit(8, 5, seed=0), unit(8, 5, seed=1), unit(8, 5, seed=2)
        F = torch.nn.functional
        y = torch.arange(8)
        for tau in (0.05, 0.5):
            loss, acc = d2l.info_nce_loss(q, d, tau)
            want = F.cross_entropy(q @ d.T / tau, y)
            self.assertAlmostEqual(loss.item(), want.item(), delta=1e-6)
            self.assertAlmostEqual(acc.item(), ((q @ d.T).argmax(1) == y)
                                   .float().mean().item())
            loss, _ = d2l.info_nce_loss(q, d, tau, hard_neg=h)
            want = F.cross_entropy(q @ torch.cat([d, h]).T / tau, y)
            self.assertAlmostEqual(loss.item(), want.item(), delta=1e-6)
            loss, _ = d2l.info_nce_loss(q, d, tau, symmetric=True)
            want = (F.cross_entropy(q @ d.T / tau, y)
                    + F.cross_entropy(d @ q.T / tau, y)) / 2
            self.assertAlmostEqual(loss.item(), want.item(), delta=1e-6)

    def test_against_sentence_transformers(self):
        try:
            from sentence_transformers.sentence_transformer.losses import (
                MultipleNegativesRankingLoss)
        except ImportError as e:
            self.skipTest(f'sentence-transformers not installed: {e}')
        q, d, h = unit(16, 32, seed=3), unit(16, 32, seed=4), unit(
            16, 32, seed=5)
        mnrl = MultipleNegativesRankingLoss(None, scale=1 / 0.05)
        for extra in ([], [h]):
            want = mnrl.compute_loss_from_embeddings([q, d, *extra], None)
            got, _ = d2l.info_nce_loss(q, d, 0.05, *extra)
            self.assertAlmostEqual(got.item(), want.item(), delta=1e-5)

    def test_gradient(self):
        tau = 0.05
        q, d = (unit(32, 16, seed=s, dtype=torch.float64).requires_grad_()
                for s in (6, 7))
        d2l.info_nce_loss(q, d, tau)[0].backward()
        P = torch.softmax(q @ d.T / tau, dim=1).detach()
        G = (P - torch.eye(32, dtype=torch.float64)) / (tau * 32)
        self.assertLess((q.grad - G @ d.detach()).abs().max().item(), 1e-10)
        self.assertLess((d.grad - G.T @ q.detach()).abs().max().item(), 1e-10)


class Geometry(unittest.TestCase):
    def test_anisotropy_identity(self):
        Z = unit(300, 8, seed=8, dtype=torch.float64) + 0.5
        Zn = torch.nn.functional.normalize(Z, dim=-1)
        n = len(Zn)
        want = ((Zn.sum(0) ** 2).sum() - n) / (n * (n - 1))
        self.assertAlmostEqual(d2l.anisotropy(Z, n_pairs=None), want.item(),
                               places=12)
        self.assertAlmostEqual(d2l.anisotropy(Z, n_pairs=10**6),
                               want.item(), places=12)  # covers all pairs
        self.assertAlmostEqual(d2l.anisotropy(Z, n_pairs=20_000),
                               want.item(), delta=0.01)

    def test_align_uniform_four_points(self):
        a = np.radians([0, 90, 180, 270])
        Z = torch.tensor(np.stack([np.cos(a), np.sin(a)], 1))
        align, unif = d2l.align_uniform(Z[[0, 2]], Z[[1, 3]], t=2.0)
        self.assertAlmostEqual(align, 2.0, places=12)  # adjacent: |z-z'|^2=2
        want = np.log((4 * np.exp(-4) + 2 * np.exp(-8)) / 6)
        self.assertAlmostEqual(unif, want, places=12)
        # Zu: uniformity over its own rows only; alignment unchanged
        Zu = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        align_u, unif_u = d2l.align_uniform(Z[[0, 2]], Z[[1, 3]], Zu=3 * Zu)
        self.assertAlmostEqual(align_u, align, places=12)
        want_u = np.log((2 * np.exp(-4) + np.exp(-8)) / 3)
        self.assertAlmostEqual(unif_u, want_u, places=12)

    def test_truncate(self):
        Z = unit(10, 12, seed=9)
        T = d2l.truncate_embeddings(Z, 4)
        self.assertEqual(T.shape, (10, 4))
        torch.testing.assert_close(T.norm(dim=-1), torch.ones(10))
        torch.testing.assert_close(T * Z[:, :4].norm(dim=-1, keepdim=True),
                                   Z[:, :4])

    def test_binary_hamming(self):
        Q, D = unit(37, 64, seed=10), unit(53, 64, seed=11)
        Qb, Db = d2l.binarize(Q), d2l.binarize(D)
        self.assertEqual((Qb.dtype, Qb.shape), (np.uint8, (37, 8)))
        sq, sd = np.sign(Q.numpy()), np.sign(D.numpy())
        want = (64 + sq @ sd.T) / 2  # equal signs
        np.testing.assert_array_equal(d2l.hamming_scores(Qb, Db), want)
        self.assertEqual(d2l.hamming_scores(Qb, Db).dtype, np.int32)


class Sampler(unittest.TestCase):
    def test_no_duplicates(self):
        rng = np.random.default_rng(0)
        n = 64 * 1000
        qs = [f'q{i}' for i in rng.integers(20_000, size=n)]
        ps = [f'p{i}' for i in rng.integers(48_000, size=n)]  # repeats
        negs = [f'p{i}' for i in rng.integers(30_000, size=n)]
        for negatives in (None, negs):
            batches = d2l.no_duplicate_batches(qs, ps, 64, seed=1,
                                               negatives=negatives)
            self.assertGreaterEqual(len(batches), 1000)
            self.assertEqual(sorted(i for b in batches for i in b),
                             list(range(n)))
            for b in batches:
                self.assertEqual(len({qs[i] for i in b}), len(b))
                passages = [ps[i] for i in b] + (
                    [] if negatives is None else [negs[i] for i in b])
                self.assertEqual(len(set(passages)), len(passages))
        self.assertEqual(d2l.no_duplicate_batches(qs, ps, 64, seed=1),
                         d2l.no_duplicate_batches(qs, ps, 64, seed=1))


class CharTokenizer:
    padding_side, pad_token_id = 'right', 0

    def __call__(self, texts, truncation=False, max_length=None):
        ids = [[1] + [2 + ord(c) % 60 for c in t] for t in texts]
        if truncation:
            ids = [r[:max_length] for r in ids]
        return {'input_ids': ids}


class TrainEmbedder(unittest.TestCase):
    def tiny(self):
        from transformers import ModernBertConfig, ModernBertModel
        torch.manual_seed(0)
        return ModernBertModel(ModernBertConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=2, pad_token_id=0,
            bos_token_id=1, eos_token_id=1, cls_token_id=1, sep_token_id=1,
            global_attn_every_n_layers=1, max_position_embeddings=128,
            attn_implementation='sdpa'))

    def pairs(self):
        rng = np.random.default_rng(0)
        words = ['alpha', 'beta', 'gamma', 'delta', 'kappa', 'omega']
        out = []
        for _ in range(96):
            w = rng.choice(words, size=3)
            out.append((' '.join(w), ' '.join(w[::-1]) + ' text'))
        return out

    def test_loss_falls_and_seeds_reproduce(self):
        pairs = self.pairs()
        runs = []
        for _ in range(2):
            model = self.tiny()
            runs.append(d2l.train_embedder(
                model, CharTokenizer(), pairs, batch_size=16, lr=3e-3,
                epochs=4, tau=0.1, max_len_q=24, max_len_d=32))
        r = runs[0]
        self.assertEqual(r['steps'], len(r['loss']))
        self.assertLess(np.mean(r['loss'][-4:]), np.mean(r['loss'][:4]))
        np.testing.assert_allclose(runs[0]['loss'], runs[1]['loss'],
                                   rtol=1e-5)
        self.assertFalse(model.training)

    def test_hard_negatives_and_matryoshka(self):
        pairs = self.pairs()
        negs = [d for _, d in pairs[1:] + pairs[:1]]
        r = d2l.train_embedder(self.tiny(), CharTokenizer(), pairs,
                               batch_size=8, lr=1e-3, steps=3,
                               hard_negs=negs, mrl_dims=(32, 16, 8))
        self.assertEqual(r['steps'], 3)
        self.assertTrue(all(np.isfinite(r['loss'])))


@unittest.skipUnless(NET, 'network/model test: set D2L_TEST_NET=1')
class AgainstSentenceTransformers(unittest.TestCase):
    def test_qwen3_embedding(self):
        from sentence_transformers import SentenceTransformer
        mid = 'Qwen/Qwen3-Embedding-0.6B'
        rev = d2l.HF_ENCODER_REVISIONS[mid]
        texts = ['The key-value cache stores keys and values.', 'BM25',
                 'Gradient descent follows the negative gradient of the '
                 'loss, one step at a time.', 'α β γ 深度学习']
        st = SentenceTransformer(mid, revision=rev, device='cpu',
                                 model_kwargs={'dtype': torch.float32},
                                 processor_kwargs={'padding_side': 'left'})
        want = st.encode(texts, normalize_embeddings=True,
                         convert_to_tensor=True).float()
        model, tok = d2l.load_encoder(mid, dtype=torch.float32, device='cpu',
                                      padding_side='left')
        got = d2l.embed_texts(model, tok, texts, pooling='last',
                              batch_size=2)
        self.assertGreaterEqual((got * want).sum(-1).min().item(), 0.999)


if __name__ == '__main__':
    unittest.main()
