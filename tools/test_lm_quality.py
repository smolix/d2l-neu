#!/usr/bin/env python3
"""CPU tests for sec_lm_quality_filtering (chapter_lm-data/quality-filtering.md
and the quality cells of crawl-lib.md): one fixture per reason code fires
exactly that code; unspaced scripts under the word rules; the agreement and
interval helpers; `fit_quality_head` and `HashedBagClassifier` on toy data;
`audit_removals` coverage on synthetic groups; the F1 identity and the
label-noise ceiling against simulation; `edu_label_scores` batching on a tiny
random model (caps, a prompt alone in its batch, input order, equality with
batch size 1); the stream-half rule.

Gated (no network, the local HF cache only): Qwen3.5-4B's tokenizer maps the
digits after both score prefixes to single tokens; the released annotation
prompts are EDU_RUBRIC with the extract filled in; the annotation sample's
val and test rows lie in the released classifier's hold-out
(lm-edu-holdout-v1, when downloaded)."""
import glob
import math
import os
import unittest

import numpy as np
import pandas as pd
import torch

from d2l import torch as d2l

HF = os.environ.get('HF_HOME', '/home/smola/.cache/hf-lm')
QWEN_TOK = glob.glob(f'{HF}/hub/models--Qwen--Qwen3.5-4B/snapshots/*/'
                     'tokenizer_config.json')
ANNOT = sorted(glob.glob(
    f'{HF}/hub/datasets--HuggingFaceFW--fineweb-edu-llama3-annotations/'
    'snapshots/*/data/train-00000-of-00008.parquet'))


class RulesTest(unittest.TestCase):
    def test_fixtures(self):
        for code, text in d2l.QUALITY_FIXTURES.items():
            want = [] if code == 'keep' else [code]
            self.assertEqual(d2l.heuristic_filters(text), want, code)
        self.assertEqual(set(d2l.QUALITY_FIXTURES) - {'keep'},
                         set(d2l.GOPHER_CFG))

    def test_order_and_cfg(self):
        text = d2l.QUALITY_FIXTURES['gopher_words'] + '\n\n*\n\n*\n\n*'
        codes = d2l.heuristic_filters(text)
        self.assertEqual(codes, [c for c in d2l.GOPHER_CFG if c in codes])
        only = {'gopher_words': d2l.GOPHER_CFG['gopher_words']}
        self.assertEqual(d2l.heuristic_filters(text, only), ['gopher_words'])
        loose = dict(only, gopher_words=dict(min=1, max=10**6))
        self.assertEqual(d2l.heuristic_filters(text, loose), [])

    def test_unspaced_scripts(self):
        zh = ('河流发源于北部的山区，向南流经山谷约四十公里。几百年来，沿岸的农民'
              '一直用河水灌溉小麦和大麦。春天积雪融化，河水淹没低洼的草地，'
              '所以古老的村庄都建在高处。') * 12
        words = [w for w in d2l._rule_words(zh) if not d2l._rule_is_symbol(w)]
        self.assertEqual(len(words), 7 * 12)  # one "word" per clause
        self.assertGreater(np.mean([len(w) for w in words]), 9)
        self.assertIn('gopher_stop_words', d2l.heuristic_filters(zh))
        self.assertIn('gopher_words', d2l.heuristic_filters(zh[:300]))
        de = ('Der Fluss entspringt in den Hügeln im Norden und fließt etwa '
              'vierzig Kilometer nach Süden durch das Tal. Seit Jahrhunderten '
              'bewässern die Bauern an seinen Ufern ihre Felder mit Weizen '
              'und Gerste. Im Frühjahr schmilzt der Schnee und der Fluss '
              'überschwemmt die tiefen Wiesen. Deshalb stehen die alten '
              'Dörfer auf höherem Grund. Heute hält ein kleiner Damm einen '
              'Teil des Wassers zurück.')
        self.assertNotIn('gopher_words', d2l.heuristic_filters(de))
        self.assertIn('gopher_stop_words', d2l.heuristic_filters(de))

    def test_rule_stats(self):
        self.assertEqual(d2l._rule_dups(['a', 'bb', 'a', 'bb', 'a']), (3, 4))
        w = 'a b a b a b'.split()
        self.assertEqual(d2l._rule_top_ngram(w, 2), 3 * 3)  # 'a b' x 3
        self.assertEqual(d2l._rule_dup_ngram('x y z x y z'.split(), 3), 3)
        self.assertEqual(d2l._rule_words("don't stop... now!"),
                         ["don't", 'stop', '...', 'now', '!'])
        c = d2l.GOPHER_CFG['c4_quality']
        ok = '\n'.join(f'This is sentence number {k} here.' for k in range(5))
        self.assertFalse(d2l._rule_c4(ok, c))
        self.assertTrue(d2l._rule_c4(ok + '\nvar x = {a: 1} in code', c))
        self.assertFalse(d2l._rule_c4(ok + '\nx {', c))  # under 3 words


class StatsTest(unittest.TestCase):
    def test_kappas(self):
        from sklearn.metrics import cohen_kappa_score
        rng = np.random.default_rng(0)
        a = rng.integers(0, 6, 500)
        b = np.clip(a + rng.integers(-1, 2, 500), 0, 5)
        self.assertAlmostEqual(d2l.cohen_kappa(a, b),
                               cohen_kappa_score(a, b), places=10)
        self.assertAlmostEqual(d2l.quadratic_kappa(a, b), cohen_kappa_score(
            a, b, weights='quadratic', labels=range(6)), places=10)
        self.assertTrue(math.isnan(d2l.cohen_kappa([1, 1], [1, 1])))

    def test_wilson_f1_auc(self):
        lo, hi = d2l.wilson_ci(81, 263)  # Wilson's interval, z = 1.96
        self.assertAlmostEqual(lo, 0.2553, places=4)
        self.assertAlmostEqual(hi, 0.3662, places=4)
        from sklearn.metrics import f1_score, roc_auc_score
        rng = np.random.default_rng(1)
        y = rng.random(400) < 0.2
        s = y + rng.normal(0, 1, 400)
        s[:20] = 0.5  # ties
        self.assertAlmostEqual(d2l.binary_f1(y, s > 0.6),
                               f1_score(y, s > 0.6), places=12)
        self.assertAlmostEqual(d2l.rank_auc(y, s), roc_auc_score(y, s),
                               places=12)
        self.assertEqual(d2l.binary_f1([0, 0], [0, 0]), 0.0)

    def test_f1_identity_and_ceiling(self):
        rng = np.random.default_rng(0)
        y = rng.random(2000) < 0.1
        s = y + rng.normal(0, 0.7, 2000)
        for t in np.quantile(s, np.linspace(0.05, 0.95, 19)):
            pred = s >= t
            pi, rho, tpr = y.mean(), pred.mean(), pred[y].mean()
            self.assertAlmostEqual(d2l.binary_f1(y, pred),
                                   2 * pi * tpr / (pi + rho), places=12)
        pi, n = 0.0926, 10**6
        for eta, f1 in ((0.02, 0.901), (0.05, 0.779), (0.10, 0.625)):
            p = 1 - eta
            r = pi * (1 - eta) / (pi * (1 - eta) + (1 - pi) * eta)
            self.assertAlmostEqual(2 * p * r / (p + r), f1, places=3)
            truth = rng.random(n) < pi
            noisy = truth ^ (rng.random(n) < eta)
            prec, rec = noisy[truth].mean(), truth[noisy].mean()
            self.assertLess(abs(prec - p),
                            3 * math.sqrt(p * (1 - p) / truth.sum()))
            self.assertLess(abs(rec - r),
                            3 * math.sqrt(r * (1 - r) / noisy.sum()))

    def test_edu_batches(self):
        lengths = [5, 30, 7, 7, 100, 12, 3, 30, 1]
        b = d2l._edu_batches(lengths, batch_size=3, max_batch_tokens=40)
        self.assertEqual(sorted(i for x in b for i in x),
                         list(range(len(lengths))))
        self.assertIn([4], b)  # over the cap: alone
        for x in b:
            self.assertLessEqual(len(x), 3)
            self.assertEqual(lengths[x[0]], max(lengths[i] for i in x))
            if len(x) > 1:
                self.assertLessEqual(lengths[x[0]] * len(x), 40)


class HeadsTest(unittest.TestCase):
    def test_fit_quality_head(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 16)).astype(np.float32)
        w = rng.normal(size=16)
        y = (X @ w > 0).astype(np.float32)
        head, losses = d2l.fit_quality_head(X, y, 'logistic', epochs=300,
                                            lr=0.05)
        with torch.no_grad():
            acc = ((head(torch.tensor(X))[:, 0] > 0).numpy() == y).mean()
        self.assertGreaterEqual(acc, 0.99)
        yr = (X @ w + 2.0).astype(np.float32)
        _, losses = d2l.fit_quality_head(X, yr, 'regression', epochs=100,
                                         lr=1e-3)
        self.assertTrue(np.all(np.diff(losses) <= 1e-9))
        with self.assertRaises(ValueError):
            d2l.fit_quality_head(X, y, 'hinge')

    def test_hashed_bag_overfits(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        vocab = [f'w{k}' for k in range(200)]
        texts = [' '.join(rng.choice(vocab, 30)) for _ in range(64)]
        y = torch.tensor(rng.random(64) < 0.5, dtype=torch.float32)
        bag = d2l.HashedBagClassifier(num_buckets=2**16, dim=16)
        ids, offsets = bag.featurize(texts)
        self.assertEqual(len(offsets), 64)
        self.assertEqual(len(ids), 64 * (30 + 29))
        opt = torch.optim.SGD(bag.parameters(), lr=0.5)
        for _ in range(300):  # summed loss: per-document SGD steps
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                bag(ids, offsets)[:, 0], y, reduction='sum')
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = ((bag(ids, offsets)[:, 0] > 0).float() == y).float().mean()
        self.assertEqual(acc.item(), 1.0)
        self.assertEqual(bag.hash_ids('A b'), bag.hash_ids('a B'))


class AuditTest(unittest.TestCase):
    def test_coverage(self):
        rng = np.random.default_rng(0)
        cover, cover_d = 0, 0
        for rep in range(200):
            g = np.repeat(['a', 'b'], 1000)
            removed = np.r_[rng.random(1000) < 0.1, rng.random(1000) < 0.3]
            df = pd.DataFrame(dict(g=g, keep=~removed))
            t = d2l.audit_removals(df, 'keep', 'g', ref='a', n_boot=1000,
                                   seed=rep).set_index('group')
            cover += t.lo['a'] <= 0.1 <= t.hi['a']
            cover_d += t.diff_lo['b'] <= 0.2 <= t.diff_hi['b']
        self.assertLess(abs(cover / 200 - 0.95), 0.04)
        self.assertLess(abs(cover_d / 200 - 0.95), 0.04)

    def test_clusters_and_too_few(self):
        df = pd.DataFrame(dict(g=['x'] * 50 + ['y'] * 200,
                               dom=[f'd{k % 5}' for k in range(250)],
                               keep=[k % 3 > 0 for k in range(250)]))
        t = d2l.audit_removals(df, 'keep', 'g', cluster='dom').set_index(
            'group')
        self.assertTrue(t.too_few['x'])  # under 100 rows
        self.assertTrue(t.too_few['y'])  # 5 clusters
        self.assertEqual(t.n_clusters['y'], 5)
        self.assertNotIn('diff', t)

    def test_stream_half(self):
        urls = [f'https://{s}.example{k}.co.uk/p{j}' for k in range(50)
                for s in ('www', 'blog') for j in range(3)]
        half = {}
        for u in urls:
            d = d2l.registered_domain(u)
            half.setdefault(d, set()).add(d2l._stream_half(d))
        self.assertEqual(len(half), 50)
        self.assertTrue(all(len(v) == 1 for v in half.values()))
        self.assertEqual({next(iter(v)) for v in half.values()}, {'A', 'B'})
        self.assertEqual(d2l._cctld('https://a.b.example.co.uk/x'), '.uk')
        self.assertEqual(d2l._cctld('http://example.com'), '.com')


def tiny_lm():
    import transformers
    from transformers import PreTrainedTokenizerFast
    texts = ['the quick brown fox jumps over the lazy dog 0 1 2 3 4 5'] * 20
    raw = d2l.train_hf_bpe(texts, 280, ['<|endoftext|>', '<|pad|>'])
    tok = PreTrainedTokenizerFast(tokenizer_object=raw,
                                  eos_token='<|endoftext|>',
                                  pad_token='<|pad|>')
    tok.chat_template = (
        "{% for m in messages %}<|user|> {{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|> {% endif %}")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=512)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    model.config._attn_implementation = 'eager'
    return model, tok


class LabelScoresTest(unittest.TestCase):
    def test_batching_equals_single(self):
        model, tok = tiny_lm()
        rng = np.random.default_rng(0)
        words = 'the quick brown fox jumps over lazy dog'.split()
        prompts = [' '.join(rng.choice(words, n)) for n in
                   (3, 40, 7, 60, 12, 5, 33, 2)]
        one = d2l.edu_label_scores(model, tok, prompts, batch_size=1,
                                   prefix='score: ')
        many = d2l.edu_label_scores(model, tok, prompts, batch_size=3,
                                    max_batch_tokens=100, prefix='score: ')
        np.testing.assert_allclose(many, one, atol=1e-5)
        self.assertEqual(one.shape, (8, 6))
        self.assertTrue(np.all(one.sum(1) <= 1 + 1e-9))
        rev = d2l.edu_label_scores(model, tok, prompts[::-1], batch_size=4,
                                   prefix='score: ')
        np.testing.assert_allclose(rev[::-1], one, atol=1e-5)

    @unittest.skipUnless(QWEN_TOK, 'Qwen3.5-4B tokenizer not in the cache')
    def test_qwen_digit_tokens(self):
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(
            os.path.dirname(QWEN_TOK[0]))
        chat = tok.apply_chat_template(
            [{'role': 'user', 'content': 'Rate this.'}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        for prefix in ('Educational score: ', 'Educational score:  '):
            base = tok.encode(chat + prefix + '0', add_special_tokens=False)
            for d in range(6):  # edu_label_scores's cut: the same tokens
                ids = tok.encode(chat + prefix + str(d),
                                 add_special_tokens=False)
                digit = tok.encode(str(d), add_special_tokens=False)
                self.assertEqual(len(digit), 1)
                self.assertEqual(ids, base[:-1] + digit, (prefix, d))
        # on its own, a prefix ending in two spaces merges them
        self.assertNotEqual(
            tok.encode('Educational score:  ', add_special_tokens=False),
            tok.encode('Educational score:  0', add_special_tokens=False)[:-1])

    @unittest.skipUnless(ANNOT, 'the annotation files are not in the cache')
    def test_rubric_matches_released_prompts(self):
        import pyarrow.parquet as pq
        pre, suf = d2l.EDU_RUBRIC.split('{extract}')
        t = pq.read_table(ANNOT[0], columns=['text', 'prompt']).slice(0, 300)
        for text, prompt in zip(*(t.column(c).to_pylist()
                                  for c in ('text', 'prompt'))):
            self.assertTrue(prompt.startswith(pre) and prompt.endswith(suf))
            extract = prompt[len(pre):len(prompt) - len(suf)]
            self.assertTrue(text.startswith(extract.rstrip('.')[:1400]))


HOLDOUT = os.path.join('..', 'data', 'lm', 'lm-edu-holdout-v1',
                       'lm-edu-holdout-v1.json.gz')
ALL_ANNOT = glob.glob(
    f'{HF}/hub/datasets--HuggingFaceFW--fineweb-edu-llama3-annotations/'
    'snapshots/*/data/train-0000[0-7]-of-00008.parquet')


class AnnotSampleTest(unittest.TestCase):
    @unittest.skipUnless(len(ALL_ANNOT) == 8 and os.path.exists(HOLDOUT),
                         'annotation files or lm-edu-holdout-v1 not local')
    def test_val_test_in_released_holdout(self):
        import gzip
        import json
        with gzip.open(HOLDOUT, 'rt') as f:
            hold = json.load(f)
        self.assertEqual((hold['holdout_size'], hold['n_rows']),
                         (46_867, 46_744))
        df = d2l._edu_annot_sample()
        self.assertEqual(df.split.tolist(), ['train'] * 18_000 +
                         ['val'] * 2_000 + ['test'] * 2_000)
        self.assertTrue(df.row.is_unique)
        p = np.random.default_rng(0).permutation(d2l._EDU_ANNOT_ROWS)
        np.testing.assert_array_equal(df.row[:18_000], p[:18_000])
        h = np.isin(df.row, hold['rows'])
        self.assertTrue(h[18_000:].all())  # out-of-sample for the released
        self.assertLess(h[:18_000].mean(), 0.11)  # train: about 10%


if __name__ == '__main__':
    unittest.main()
