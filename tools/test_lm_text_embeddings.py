#!/usr/bin/env python3
"""CPU tests for sec_lm_text_embeddings (chapter_lm-encoders/text-embeddings.md),
the shared part: `pool_hidden` against per-example loops with left and right
padding; `retrieval_metrics` against `pytrec_eval` (success@k for recall@k,
reciprocal rank of the run cut at 10, ndcg_cut_10) on a toy run; `embed_texts`
on a stub encoder (order restored, padding side, truncation count).
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
