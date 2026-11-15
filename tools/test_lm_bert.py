#!/usr/bin/env python3
"""CPU tests for sec_lm_bert (chapter_lm-encoders/bert.md): `mlm_corrupt` on a
million tokens (selected share 0.15, split 0.8/0.1/0.1, special tokens never
selected, labels -100 elsewhere); `PooledClassifier` against Transformers'
`ModernBertForSequenceClassification` on a tiny random ModernBERT after the
weight rename (mean and first-token pooling, padded batch, fp32 1e-5);
`align_token_labels` (one label per word, in order, for both schemes);
`span_f1` against `seqeval` on 200 sentences with injected errors (overall
and per type, 1e-9); `cost_per_example`; `finetune_classifier` lowers the loss
of the tiny model in 20 steps, for a sentence head and a token head, and
returns a prediction per example or per scored token."""
import copy
import unittest

import numpy as np
import torch
from d2l import torch as d2l


def tiny_config(**kw):
    import transformers
    return transformers.ModernBertConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=64, global_attn_every_n_layers=2,
        local_attention=8, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        cls_token_id=1, sep_token_id=2, **kw)


class MlmCorrupt(unittest.TestCase):
    def test_shares(self):
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(5, 1000, (10_000, 100), generator=g)
        ids[:, 0], ids[:, -1] = 1, 2
        X, labels = d2l.mlm_corrupt(ids, 3, 1000, [0, 1, 2, 3, 4],
                                    generator=g)
        special = ids < 5
        sel = labels != -100
        self.assertFalse((sel & special).any())
        self.assertTrue(torch.equal(labels[sel], ids[sel]))
        self.assertTrue(torch.equal(X[~sel], ids[~sel]))
        self.assertAlmostEqual(sel[~special].float().mean().item(), 0.15,
                               delta=1e-3)
        masked, kept = X[sel] == 3, X[sel] == ids[sel]
        for share, want in ((masked, 0.8), (kept, 0.1),
                            (~masked & ~kept, 0.1)):
            self.assertAlmostEqual(share.float().mean().item(), want,
                                   delta=2e-3)


class Pooled(unittest.TestCase):
    def test_equals_transformers_head(self):
        import transformers
        for pooling in ('mean', 'cls'):
            torch.manual_seed(0)
            ref = transformers.ModernBertForSequenceClassification(
                tiny_config(num_labels=3, classifier_pooling=pooling,
                            attn_implementation='sdpa')).eval()
            clf = d2l.PooledClassifier(copy.deepcopy(ref.model), 3, pooling)
            clf.load_state_dict({k.replace('model.', 'encoder.', 1): v
                                 for k, v in ref.state_dict().items()})
            clf.eval()
            lens = [7, 3, 12, 1]
            X = torch.zeros(4, 12, dtype=torch.long)
            M = torch.zeros(4, 12, dtype=torch.long)
            for i, n in enumerate(lens):
                X[i, :n] = torch.randint(5, 64, (n,))
                M[i, :n] = 1
            with torch.no_grad():
                want = ref(input_ids=X, attention_mask=M).logits
                got = clf(X, M)
            torch.testing.assert_close(got, want, atol=1e-5, rtol=0)

    def test_head_follows_encoder(self):
        import transformers
        for device, dtype in (('cpu', torch.bfloat16),
                              ('meta', torch.float32)):
            enc = transformers.ModernBertModel(tiny_config()).to(device, dtype)
            clf = d2l.PooledClassifier(enc, 2)
            for p in clf.parameters():
                self.assertEqual((p.device.type, p.dtype), (device, dtype))

    def test_bad_pooling(self):
        import transformers
        enc = transformers.ModernBertModel(tiny_config())
        with self.assertRaises(ValueError):
            d2l.PooledClassifier(enc, 2, pooling='max')

    def test_cls_needs_right_padding(self):
        import transformers
        clf = d2l.PooledClassifier(transformers.ModernBertModel(tiny_config(
            attn_implementation='sdpa')), 2, pooling='cls').eval()
        X = torch.randint(5, 64, (2, 6))
        M = torch.tensor([[1] * 6, [0, 0, 1, 1, 1, 1]])  # left padded
        with torch.no_grad(), self.assertRaises(ValueError):
            clf(X, M)
        with torch.no_grad():  # right padded: fine
            self.assertEqual(clf(X, M.flip(-1)).shape, (2, 2))


class Align(unittest.TestCase):
    def test_one_label_per_word(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            n = int(rng.integers(1, 15))
            labels = rng.integers(0, 9, n).tolist()
            word_ids = [None] + [w for w in range(n)
                                 for _ in range(rng.integers(1, 4))] + [None]
            first = d2l.align_token_labels(word_ids, labels)
            self.assertEqual(len(first), len(word_ids))
            self.assertEqual([l for l in first if l != -100], labels)
            starts = [i for i, w in enumerate(word_ids) if w is not None and
                      (i == 0 or word_ids[i - 1] != w)]
            self.assertEqual([i for i, l in enumerate(first) if l != -100],
                             starts)
            every = d2l.align_token_labels(word_ids, labels, scheme='all')
            self.assertEqual(every, [-100 if w is None else labels[w]
                                     for w in word_ids])
        with self.assertRaises(ValueError):
            d2l.align_token_labels([0], [1], scheme='last')


class SpanF1(unittest.TestCase):
    def test_against_seqeval(self):
        try:
            from seqeval.metrics import classification_report, f1_score
            from seqeval.metrics import precision_score, recall_score
        except ImportError as e:
            self.skipTest(f'seqeval not installed: {e}')
        id2label = {0: 'O', 1: 'B-ORG', 2: 'B-MISC', 3: 'B-PER', 4: 'I-PER',
                    5: 'B-LOC', 6: 'I-ORG', 7: 'I-MISC', 8: 'I-LOC'}
        inside = {1: 6, 2: 7, 3: 4, 5: 8}
        rng = np.random.default_rng(0)
        gold, pred = [], []
        for _ in range(200):
            seq, i, n = [], 0, int(rng.integers(1, 20))
            while i < n:  # valid IOB2: entities of length 1-3 between Os
                if rng.random() < 0.4:
                    b = int(rng.choice(list(inside)))
                    k = int(rng.integers(1, 4))
                    seq += [b] + [inside[b]] * (k - 1)
                    i += k
                else:
                    seq.append(0)
                    i += 1
            gold.append(seq)
            p = list(seq)  # errors: any tag anywhere, including bad I- starts
            for j in range(len(p)):
                if rng.random() < 0.15:
                    p[j] = int(rng.integers(0, 9))
            pred.append(p)
        ours = d2l.span_f1(pred, gold, id2label)
        names = lambda seqs: [[id2label[t] for t in s] for s in seqs]
        g, p = names(gold), names(pred)
        self.assertAlmostEqual(ours['f1'], f1_score(g, p), delta=1e-9)
        self.assertAlmostEqual(ours['precision'], precision_score(g, p),
                               delta=1e-9)
        self.assertAlmostEqual(ours['recall'], recall_score(g, p),
                               delta=1e-9)
        report = classification_report(g, p, output_dict=True,
                                       zero_division=0)
        for kind, v in ours['per_type'].items():
            for key in ('precision', 'recall', 'f1'):
                self.assertAlmostEqual(
                    v[key], report[kind][key if key != 'f1' else 'f1-score'],
                    delta=1e-9)

    def test_hand_example(self):
        id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC'}
        gold = [[1, 2, 0, 3]]
        pred = [[1, 2, 0, 1]]  # PER right, LOC predicted as PER
        r = d2l.span_f1(pred, gold, id2label)
        self.assertEqual((r['precision'], r['recall'], r['f1']),
                         (0.5, 0.5, 0.5))
        self.assertEqual(r['per_type']['LOC']['recall'], 0.0)
        r = d2l.span_f1([[0, 2, 2, 0]], [[0, 1, 2, 0]], id2label)
        self.assertEqual(r['f1'], 1.0)  # I- after O starts an entity


class Cost(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(d2l.cost_per_example(100, 7), 1400)
        self.assertEqual(d2l.cost_per_example(100, 7, n_embed=40), 840)


class Finetune(unittest.TestCase):
    def data(self, n, token=False, seed=0):
        """Label: does token 7 occur (sentence), or is this token 7 (tags)."""
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            x = rng.integers(5, 20, int(rng.integers(3, 10))).tolist()
            if token:
                y = [int(t == 7) for t in x]
                y[0] = -100  # an unscored token
                rows.append((x, y))
            else:
                rows.append((x, int(7 in x)))
        return rows

    def test_sentence_head(self):
        import transformers
        torch.manual_seed(0)
        model = d2l.PooledClassifier(transformers.ModernBertModel(
            tiny_config(attn_implementation='sdpa')), 2)
        train, ev = self.data(160), self.data(30, seed=1)
        r = d2l.finetune_classifier(model, train, ev, 3e-3, 1, 8, seed=0,
                                    device='cpu')
        self.assertEqual(len(r['loss']), 20)
        self.assertLess(np.mean(r['loss'][-5:]), np.mean(r['loss'][:5]))
        self.assertEqual(len(r['eval_pred']), 30)
        self.assertTrue(all(isinstance(p, int) for p in r['eval_pred']))
        acc = np.mean(np.array(r['eval_pred']) == [y for _, y in ev])
        self.assertAlmostEqual(r['eval_acc'], acc, delta=1e-12)
        self.assertTrue(np.isnan(r['peak_gib']))

    def test_token_head_and_seed(self):
        import transformers
        runs = []
        for _ in range(2):
            torch.manual_seed(0)
            model = transformers.ModernBertForTokenClassification(
                tiny_config(num_labels=2, attn_implementation='sdpa'))
            train, ev = self.data(160, True), self.data(20, True, seed=1)
            runs.append(d2l.finetune_classifier(model, train, ev, 3e-3, 1, 8,
                                                seed=0, device='cpu'))
        r = runs[0]
        self.assertLess(np.mean(r['loss'][-5:]), np.mean(r['loss'][:5]))
        self.assertEqual([len(p) for p in r['eval_pred']],
                         [len(x) - 1 for x, _ in ev])
        self.assertEqual(r['loss'], runs[1]['loss'])  # seed: same run


if __name__ == '__main__':
    unittest.main()
