#!/usr/bin/env python3
"""CPU tests for sec_lm_long_context (chapter_lm-pretraining/long-context.md
and long-context-lib.md): `rope_inv_freq` checks (a)-(g) of the blueprint
(scale 1 gives the base frequencies; interpolation is division by s;
NTK-aware: first pair unchanged, last pair divided by s, the overhang of
0.43 at m = 23 on 8 of the 9 slow pairs; YaRN pair counts 9/12/11 at 4K and
14/12/6 at 1K; `yarn_mscale(4)`; `rope_mscale` scales the attention logits
by m^2 with the softmax scale untouched; gated: Transformers' YaRN
initialization with truncate=False to 1e-6); `make_recall_task` invariants
(gold once, distinct candidates from the haystack, unique keys, length
within one line, depth within one line) and the oracle copier at 100%;
`_recall_ll` against `score_choices(norm='sum')` on 1K-token items of a
tiny random model (fp32, 1e-4); `nll_by_position` (buckets average to the
total); `long_context_gain` on a synthetic repeat (the copy oracle gains
about ln V, a unigram model nothing); `sliding_nlls` against an explicit
short context; `repeat_gaps` on a hand example; the effective-context rule
on a synthetic grid; the staged stream (pieces attend only to themselves,
equal to forwards of the pieces alone); the arm configurations; a toy T1
campaign through `lc_arms` end to end (job folders, stages, periodic
evaluations, the merge, the stored result) and a resumed arm equal to an
uninterrupted one; the page's CPU cells; the page's later cells on a
pending result; the regeneration jobs against the page's literals.
"""
import contextlib
import io
import json
import math
import os
import re
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from d2l import torch as d2l

torch.set_num_threads(min(8, torch.get_num_threads()))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER = os.path.join(ROOT, 'chapter_lm-pretraining')
PAGE = os.path.join(CHAPTER, 'long-context.md')
V = 32768  # the reference tokenizer's vocabulary
TOY = d2l.Arch(L=2, d=32, H=4, H_kv=2, d_h=8, f=64, vocab=V, T=64,
               rope_base=1e4, qk_norm=False, tie=True)


def page_cells():
    with open(PAGE) as f:
        src = f.read()
    return dict(re.findall(r'```\{\.python \.input #([^}]*)\}\n'
                           r'%%tab pytorch\n(.*?)```', src, re.S))


def run_cell(name, ns):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(page_cells()[name], ns)
    return out.getvalue()


def page_ns(**kw):
    return dict(math=math, np=np, pd=pd, torch=torch, d2l=d2l, **kw)


def toy_model(seed=0, max_len=64):
    torch.manual_seed(seed)
    return d2l.PackedGPT.from_arch(TOY._replace(T=max_len)).eval()


def r2_tok():
    try:
        with d2l._quiet():
            return d2l.r2_tokenizer()
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f'reference tokenizer unavailable: {e}')


@contextlib.contextmanager
def in_chapter():
    old = os.getcwd()
    os.chdir(CHAPTER)
    try:
        yield
    finally:
        os.chdir(old)


class RopeTest(unittest.TestCase):
    d, L = 64, 4096

    def test_scale_one(self):  # (a)
        theta = d2l.rope_inv_freq(self.d)
        base = 1.0 / 1e4 ** (torch.arange(0, self.d, 2, dtype=torch.float32)
                             / self.d)
        self.assertTrue(torch.equal(theta, base))
        for k in ('pi', 'ntk', 'yarn'):
            self.assertLessEqual(float((d2l.rope_inv_freq(
                self.d, scheme=k, s=1) - theta).abs().max()), 1e-7)
        with self.assertRaises(ValueError):
            d2l.rope_inv_freq(self.d, scheme='linear')

    def test_pi_and_ntk(self):  # (b), (c), (f)
        theta = d2l.rope_inv_freq(self.d)
        self.assertTrue(torch.equal(d2l.rope_inv_freq(self.d, scheme='pi',
                                                      s=4), theta / 4))
        ntk = d2l.rope_inv_freq(self.d, scheme='ntk', s=4)
        self.assertAlmostEqual(float(ntk[0]), 1.0, places=7)
        self.assertAlmostEqual(float(ntk[-1] / (theta[-1] / 4)), 1.0,
                               places=5)
        r = self.L * theta / (2 * math.pi)  # rotations within 4K
        self.assertEqual(int((r >= 1).sum()), 23)
        over = (4 * ntk / theta - 1)[r < 1]  # phase overhang, slow pairs
        self.assertEqual(int((over > 1e-5).sum()), 8)
        self.assertAlmostEqual(float(over[0]), 0.43, places=2)
        self.assertEqual(int(torch.argmax(over)), 0)

    def test_yarn_counts(self):  # (d)
        theta = d2l.rope_inv_freq(self.d)
        for L, counts in ((4096, (9, 12, 11)), (1024, (14, 12, 6))):
            for s in (2, 4, 8):
                ratio = d2l.rope_inv_freq(self.d, scheme='yarn', s=s,
                                          L=L) / theta
                got = (int((ratio == 1 / s).sum()),
                       int(((ratio > 1 / s) & (ratio < 1)).sum()),
                       int((ratio == 1).sum()))
                self.assertEqual(got, counts, (L, s))

    def test_mscale_value(self):  # (e)
        self.assertAlmostEqual(d2l.yarn_mscale(4), 1.1386, places=4)
        self.assertEqual(d2l.yarn_mscale(1), 1.0)
        self.assertEqual(d2l.yarn_mscale(0.5), 1.0)

    def test_mscale_once(self):  # (e): logits by m^2, softmax scale kept
        m = d2l.yarn_mscale(4)
        a, b = toy_model(), toy_model()
        a.set_rope(None, m)
        with torch.no_grad():
            for blk in b.blks:
                blk.attention.W_q.weight.mul_(m)
                blk.attention.W_k.weight.mul_(m)
        X = torch.randint(1, V, (2, 48))
        with torch.no_grad():
            torch.testing.assert_close(a(X), b(X), rtol=1e-5, atol=1e-5)
        pos = torch.arange(10)
        q, k = torch.randn(10, 8), torch.randn(10, 8)
        rot = lambda x, ms: d2l.rope(x, tables=d2l._rope_tables(
            pos, 8, 1e4, mscale=ms))
        ratio = (rot(q, m) @ rot(k, m).T) / (rot(q, 1.0) @ rot(k, 1.0).T)
        torch.testing.assert_close(ratio, torch.full_like(ratio, m * m),
                                   rtol=1e-4, atol=1e-4)

    def test_transformers(self):  # (g), gated
        try:
            hf, factor = d2l.transformers_yarn(self.d, s=4, L=self.L)
            rounded, _ = d2l.transformers_yarn(self.d, s=4, L=self.L,
                                               truncate=True)
        except Exception as e:  # noqa: BLE001
            self.skipTest(f'transformers YaRN unavailable: {e}')
        yarn = d2l.rope_inv_freq(self.d, scheme='yarn', s=4, L=self.L)
        self.assertLessEqual(float((yarn - hf).abs().max()), 1e-6)
        self.assertGreater(float((yarn - rounded).abs().max()), 1e-4)
        self.assertAlmostEqual(factor, d2l.yarn_mscale(4), places=6)


def copier(prompt, choices):
    key = prompt.rsplit('The value of ', 1)[1].removesuffix(' is')
    value = prompt.split(f'The value of {key} is ', 1)[1].split('.', 1)[0]
    return choices.index(' ' + value)


class RecallTaskTest(unittest.TestCase):
    def test_invariants(self):
        tok = r2_tok()
        for length in (1024, 4096):
            for depth in (0.1, 0.5, 0.9):
                t = d2l.make_recall_task(tok, length, depth, n_items=5,
                                         seed=1)
                again = d2l.make_recall_task(tok, length, depth, n_items=5,
                                             seed=1)
                self.assertEqual(t, again)
                for p, cs, g, da in zip(t['prompts'], t['choices'],
                                        t['gold'], t['depth_actual']):
                    lines = p.split('\n')[1:-1]
                    keys = [x[len('The value of '):x.index(' is ')]
                            for x in lines]
                    values = [' ' + x.rsplit(' ', 1)[1][:-1] for x in lines]
                    self.assertEqual(len(set(keys)), len(keys))
                    self.assertEqual(len(set(cs)), 8)
                    self.assertTrue(set(cs) <= set(values))
                    self.assertEqual(sum(v == cs[g] for v in values), 1)
                    self.assertEqual(copier(p, cs), g)
                    n = len(tok(p + cs[g])['input_ids'])
                    line = max(len(x) for x in d2l._lm_encode(
                        tok, [x + '\n' for x in lines]))
                    self.assertLessEqual(n, length)
                    self.assertGreater(n, length - 3 * line)
                    self.assertLessEqual(abs(da - depth), 1.5 / len(lines)
                                         + 0.01)
        t = d2l.make_recall_task(tok, 1024, 0.5, n_pairs=10, n_items=2)
        self.assertEqual(t['prompts'][0].count(' is '), 10)

    def test_oracle_lengths(self):
        tok = r2_tok()
        for length in (2048, 8192):
            t = d2l.make_recall_task(tok, length, 0.3, n_items=10)
            self.assertEqual([copier(p, c) for p, c in zip(t['prompts'],
                                                           t['choices'])],
                             t['gold'])


class ScoreRecallTest(unittest.TestCase):
    def test_equals_score_choices(self):
        tok = r2_tok()
        model = toy_model(max_len=1024).float()
        task = d2l.make_recall_task(tok, 1024, 0.5, n_items=3)
        ll = d2l._recall_ll(model, tok, task, chunk=300)
        ref = d2l.score_choices(model, tok, task['prompts'], task['choices'],
                                norm='sum')
        torch.testing.assert_close(ll, ref.double(), rtol=0, atol=1e-4)
        ok = d2l.score_recall(model, tok, task)
        np.testing.assert_array_equal(ok, ll.argmax(1).numpy() ==
                                      np.array(task['gold']))


class MeasureTest(unittest.TestCase):
    def test_buckets_average(self):
        model = toy_model(max_len=256)
        torch.manual_seed(1)
        W = torch.randint(1, V, (4, 257))
        parts = d2l.nll_by_position(model, W, range(0, 257, 64))
        whole = d2l.nll_by_position(model, W, [0, 256])
        self.assertLess(abs(parts['nll'].mean() - whole['nll'][0]), 1e-6)
        self.assertEqual(list(parts['bucket']), ['0-64', '64-128',
                                                 '128-192', '192-256'])
        self.assertTrue((parts['lo'] <= parts['nll']).all()
                        and (parts['nll'] <= parts['hi']).all())

    def test_gain_oracle(self):
        Vs, T, short = 2048, 1024, 128
        torch.manual_seed(0)
        x = torch.stack([torch.randperm(Vs)[:T // 2] for _ in range(4)])
        rep = torch.cat([x, x, x[:, :1]], 1)
        g = {}
        for name, m in (('oracle', d2l.CopyOracle(Vs)),
                        ('unigram', d2l.UnigramModel(Vs))):
            g[name] = d2l.long_context_gain(m, rep, short_len=short,
                                            long_len=T, min_gap=short,
                                            rare_ids=torch.arange(Vs),
                                            edges=[short, T])
        # one target per row (the second copy's first) has no cue: gain 0
        self.assertAlmostEqual(g['oracle']['gain'][0], math.log(Vs) * (
            T // 2) / (T // 2 + 1), places=3)
        self.assertEqual(g['unigram']['gain'][0], 0.0)
        self.assertEqual(int(g['oracle']['n'][0]), 4 * (T // 2 + 1))

    def test_sliding(self):
        model = toy_model(max_len=128)
        torch.manual_seed(2)
        W = torch.randint(1, V, (2, 129))
        short = d2l.sliding_nlls(model, W, 32)
        full = d2l.window_nlls(model, W)
        torch.testing.assert_close(short[:, :32], full[:, :32], rtol=1e-5,
                                   atol=1e-5)
        for t in (40, 47, 100, 127):  # context: from (t // 16 - 1) * 16
            a = (t // 16 - 1) * 16
            ref = d2l.window_nlls(model, W[:, a:t + 2])[:, -1]
            torch.testing.assert_close(short[:, t], ref, rtol=1e-5,
                                       atol=1e-5)

    def test_repeat_gaps(self):
        W = torch.tensor([[5, 7, 5, 9, 7, 5, 3]])
        gaps = d2l.repeat_gaps(W, [5, 7, 9])
        self.assertEqual(gaps.tolist(), [[0, 2, 0, 3, 3, 0]])
        self.assertEqual(d2l.repeat_gaps(W, [7]).tolist(),
                         [[0, 0, 0, 3, 0, 0]])

    def test_window_ci(self):
        mean, lo, hi = d2l.window_ci(np.arange(10.0), np.ones(10))
        self.assertEqual(mean, 4.5)
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        self.assertTrue(all(math.isnan(v) for v in d2l.window_ci(
            [0.0], [0.0])))
        self.assertEqual(d2l.bucket_name(0, 4096), '0-4K')
        self.assertEqual(d2l.bucket_name(1536, 3072), '1.5-3K')
        self.assertEqual(d2l.bucket_name(256, 512), '256-512')


class EffectiveContextTest(unittest.TestCase):
    def test_rule(self):
        rows = []
        for L, acc in ((1024, 1.0), (2048, 0.8), (3072, 0.52),
                       (4096, 0.6)):
            for d in d2l.LC_DEPTHS:
                rows += [('A', 0, L, d, i, i < acc * 100)
                         for i in range(100)]
        recall = pd.DataFrame(rows, columns=d2l._LC_COLUMNS['recall'])
        gain = pd.DataFrame([
            ('A', d2l.bucket_name(L // 2, L), True, 0.1, lo, 0.2, 10)
            for L, lo in ((1024, 0.05), (2048, 0.05), (3072, 0.05),
                          (4096, -0.01))], columns=d2l._LC_COLUMNS['gain'])
        # 3K: Wilson lower bound of 260/500 is below 0.5; 4K: gain bound < 0
        self.assertEqual(d2l.effective_context(recall, gain)['A'], 2048)
        gain.loc[3, 'lo'] = 0.01
        self.assertEqual(d2l.effective_context(recall, gain)['A'], 4096)
        self.assertEqual(d2l.effective_context(recall, gain,
                                               threshold=0.9)['A'], 1024)
        recall['correct'] = False
        self.assertEqual(d2l.effective_context(recall, gain)['A'], 0)


class StagedStreamTest(unittest.TestCase):
    def test_pieces(self):
        class Mix:
            rank, world, step = 0, 1, 0

            def batch(self, step):
                g = torch.Generator().manual_seed(step)
                X = torch.randint(1, 50, (3, 32), generator=g)
                X[0, 5] = X[1, 20] = X[2, 16] = 0  # EOS
                Y = torch.cat([X[:, 1:], X[:, :1]], 1)
                doc, pos = d2l.doc_ids_from_eos(X.int(), 0)
                return dict(input_ids=X, labels=Y, doc_ids=doc,
                            pos_ids=pos)

        cfg = dict(stages=[dict(T=16, s=2, steps=2), dict(T=32, s=4,
                                                          steps=2)])
        st = d2l._StagedStream(Mix(), cfg, 0)
        for step in range(4):
            b, ref = st.batch(step), Mix().batch(step)
            torch.testing.assert_close(b['labels'], ref['labels'])
            if step >= 2:
                torch.testing.assert_close(b['doc_ids'], ref['doc_ids'])
                continue
            doc, pos = b['doc_ids'], b['pos_ids']
            self.assertTrue((doc[:, 16:].amin(1) > doc[:, :16].amax(1))
                            .all())
            self.assertTrue((pos[:, 16] == 0).all())
            model = toy_model(max_len=32)
            with torch.no_grad():
                whole = model(b['input_ids'], d2l.PackCtx(doc, pos))
                for k in (0, 1):
                    Xk = b['input_ids'][:, 16 * k:16 * (k + 1)]
                    alone = model(Xk, d2l.PackCtx(*d2l.doc_ids_from_eos(
                        Xk.int(), 0)))
                    torch.testing.assert_close(
                        whole[:, 16 * k:16 * (k + 1)], alone, rtol=1e-4,
                        atol=1e-4)
        self.assertEqual(d2l._lc_stage_of(cfg, 1), 0)
        self.assertEqual(d2l._lc_stage_of(cfg, 2), 1)
        self.assertEqual(d2l._lc_stage_of(cfg, 9), 1)


class ConfigTest(unittest.TestCase):
    def test_t2(self):
        cfgs = d2l._lc_configs(**d2l._lc_t2_call())
        self.assertEqual(sorted(cfgs), [('PI', 0), ('YaRN', 0),
                                        ('YaRN-prog', 0)])
        self.assertEqual(cfgs['PI', 0]['steps'], 381)
        self.assertEqual([st['steps'] for st in cfgs['YaRN-prog', 0][
            'stages']], [153, 229])
        self.assertEqual(cfgs['YaRN', 0]['decay'], round(0.15 * 381))
        self.assertEqual(cfgs['PI', 0]['global_batch'], 16)
        self.assertEqual(cfgs['PI', 0]['trained_len'], 4096)
        self.assertTrue(d2l._lc_job(cfgs['YaRN-prog', 0]).endswith(
            os.path.join('lc', 't2-yarn-prog-s0')))
        with self.assertRaises(ValueError):
            d2l._lc_configs(**dict(d2l._lc_t2_call(), arms={
                'PI': dict(scheme='pi', stages=((4096, 4, 1.0),))}))

    def test_t1(self):
        cfgs = d2l._lc_configs(**dict(d2l._lc_t1_call(), peak_lr=1e-4,
                                      warmup=20, decay_frac=0.15,
                                      long_share=0.5))
        self.assertEqual(len(cfgs), 7)
        self.assertEqual(cfgs['PI', 2]['steps'], 244)
        self.assertEqual([st['steps'] for st in cfgs['YaRN-prog', 0][
            'stages']], [98, 146])
        self.assertEqual(cfgs['PI', 0]['micro_batch'], d2l._LC_MICRO['t1'])

    def test_set_scheme(self):
        m = toy_model()
        d2l._lc_set_scheme(m, 'yarn', 4, 64)
        self.assertEqual(m.max_len, 256)
        self.assertAlmostEqual(m.rope_mscale, d2l.yarn_mscale(4))
        self.assertEqual(m.config()['rope_scheme'], 'yarn')
        d2l._lc_set_scheme(m, 'yarn', 2, 64, mscale=1.0)
        self.assertEqual((m.max_len, m.rope_mscale), (128, 1.0))
        d2l._lc_set_scheme(m, 'none', 4, 64)
        self.assertIsNone(m.rope_inv_freq)


class IdTokenizer:
    """A tiktoken-like encoding whose texts are space-separated ids."""
    n_vocab = V
    special_tokens_set = {'<|endoftext|>'}

    def encode_single_token(self, s):
        return 0

    def encode_ordinary(self, t):
        return [int(x) for x in t.split()]

    def encode_ordinary_batch(self, ts):
        return [self.encode_ordinary(t) for t in ts]


def fixture_shards(out, n_docs, lo, hi, seed):
    rng = np.random.default_rng(seed)
    docs = [(f'd{i}', ' '.join(map(str, rng.integers(32, V, rng.integers(
        lo, hi + 1)))), {'part': 'pg19'}) for i in range(n_docs)]
    d2l.write_shards(docs, IdTokenizer(), out, 'train',
                     shard_tokens=10 ** 9, eos_id=0)
    return d2l.TokenShards(out)


TIERS = {t: dict(trained=256, lengths=(256, 512, 768, 1024), books=6,
                 n_items=2, zs_items=2, eval_books=3, eval_items=2)
         for t in ('t1', 't2')}
ARMS = {'PI': dict(scheme='pi', stages=((1024, 4, 1.0),), seeds=(0, 1)),
        'YaRN': dict(scheme='yarn', stages=((1024, 4, 1.0),), seeds=(0, 1)),
        'YaRN-prog': dict(scheme='yarn', stages=((512, 2, 0.5),
                                                 (1024, 4, 0.5)))}


class ArmsEndToEndTest(unittest.TestCase):
    """A toy T1 campaign through lc_arms with D2L_RERUN=lc-t1."""
    @classmethod
    def setUpClass(cls):
        cls.tok = r2_tok()
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.tmp.name
        cls.shards = dict(
            long=fixture_shards(os.path.join(t, 'long'), 100, 300, 2400, 0),
            short=fixture_shards(os.path.join(t, 'short'), 400, 5, 300, 1))
        torch.manual_seed(3)
        cls.books = torch.randint(32, V, (6, 1025))
        cls.base = os.path.join(t, 'base.safetensors')
        d2l._save_bf16(cls.base, toy_model(max_len=256), 0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def patches(self, root, rerun='lc-t1'):
        items = pd.DataFrame(dict(task=['t1', 't1', 't2'],
                                  item_id=['a', 'b', 'c'],
                                  score=[1.0, 0.0, 1.0]))
        val = torch.randint(32, V, (4, 257))
        zs = dict(base='r2', schemes=d2l._LC_SCHEMES,
                  lengths=(1024, 2048, 4096))
        return [
            mock.patch.object(torch.cuda, 'is_available', lambda: False),
            mock.patch.object(d2l, '_job_root', lambda: root),
            mock.patch.object(d2l, '_rerun_dir',
                              lambda: os.path.join(root, 'rerun')),
            mock.patch.object(d2l, '_lc_base_path', lambda base: self.base),
            mock.patch.object(d2l, 'long_data', lambda tier: d2l.LcData(
                None, self.shards, '')),
            mock.patch.object(d2l, 'test_books',
                              lambda n, T: self.books[:n, :T + 1]),
            mock.patch.object(d2l, '_val_windows', lambda T: (val, 0)),
            mock.patch.object(d2l, '_rare_ids', lambda: torch.arange(V)),
            mock.patch.object(d2l, '_cpt_suite', lambda *a, **k: 'suite'),
            mock.patch.object(d2l, '_suite_items',
                              lambda *a, **k: items.copy()),
            mock.patch.object(d2l, '_lc_zeroshot_call', lambda: zs),
            mock.patch.object(d2l, 'r2_tokenizer', lambda: self.tok),
            mock.patch.dict(d2l._LC_TIERS, TIERS),
            mock.patch.dict(d2l._LC_MICRO, {'t1': 2}),
            mock.patch.object(d2l, '_LC_EVAL_EVERY', 4),
            mock.patch.object(d2l, '_LC_STATE_EVERY', 3),
            mock.patch.dict(os.environ, {'D2L_RERUN': rerun})]

    def test_campaign(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.ExitStack() as stack, \
                contextlib.redirect_stdout(io.StringIO()):
            for p in self.patches(root):
                stack.enter_context(p)
            runs = d2l.lc_arms(ARMS, base='t1', tokens=8 * 4096,
                               seq_len=1024, batch_tokens=4096)
            self.assertTrue(os.path.exists(os.path.join(
                root, 'lc', 't1-yarn-prog-s0', 'DONE')))
            self.assertFalse(os.path.exists(os.path.join(
                root, 'lc', 't1-pi-s1', 'state')))
            stored = d2l._read_result(os.path.join(
                root, 'rerun', d2l._next_key('lm-lc-t1-v1') + '.json.gz'),
                'json-table')
        t = runs.table
        self.assertEqual(list(t['arm']), ['base', 'PI', 'YaRN', 'YaRN-prog'])
        self.assertEqual(list(t.columns), [
            'arm', 'val loss', 'Δ val', 'NLL 768-1024', 'recall 256',
            'recall 1K', 'min', 'max', 'effective'])
        self.assertEqual(t['Δ val'].iloc[0], 0.0)
        logs = runs.logs
        prog = logs[logs['arm'] == 'YaRN-prog']
        self.assertEqual(list(prog['step']), list(range(1, 9)))
        self.assertEqual(list(prog['stage']), [0] * 4 + [1] * 4)
        tl = runs.timeline
        self.assertEqual(sorted(tl[(tl['arm'] == 'PI') & (tl['seed'] == 1)]
                                ['step']), [0, 4, 8])
        self.assertEqual(set(runs.recall['model']),
                         {'base', 'PI', 'YaRN', 'YaRN-prog'})
        self.assertEqual(sorted(runs.recall['length'].unique()),
                         [256, 512, 768, 1024])
        self.assertEqual(len(runs.recall), 6 * 4 * 5 * 2)
        self.assertEqual(set(runs.gain['rule']), {True, False})
        self.assertIn('PI, zero-shot', set(runs.position['model']))
        self.assertEqual(list(runs.zeroshot['scheme']), list(d2l._LC_SCHEMES))
        self.assertEqual(list(runs.short['arm']), ['PI', 'YaRN', 'YaRN-prog'])
        self.assertEqual(runs.short['Δ suite'].iloc[0], 0.0)
        d = d2l.scheme_difference(runs, 'YaRN', 'PI')
        self.assertEqual(d['method'], 't interval, 2 seeds')
        self.assertEqual(list(stored.table['arm']), list(t['arm']))
        self.assertEqual(stored.table.attrs['config']['tokens'], 32768)
        self.assertEqual(stored.zs_config['lengths'], [256, 512, 1024])

    def test_resume_equal(self):
        cfg = dict(d2l._lc_configs(ARMS, 't1', 12 * 2048, 1024, 2048, 1e-3,
                                   2, 0.25, 0.5)['YaRN-prog', 0],
                   micro_batch=2, evaluate=False, trained_len=256)
        logs, ckpts = [], []
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(d2l, '_LC_STATE_EVERY', 3):
            for name, stops in (('straight', (None,)),
                                ('resumed', (4, None))):
                job = os.path.join(root, name)
                for stop in stops:
                    model = d2l._load_bf16(self.base, torch.float32).train()
                    log = d2l._lc_train(model, d2l._lc_stream(self.shards,
                                                              cfg), cfg, job,
                                        stop=stop)
                logs.append(log.set_index('step')['loss'])
                ckpts.append(d2l._load_bf16(os.path.join(
                    job, 'ckpt', 's12.safetensors')))
        self.assertEqual(len(logs[0]), 12)
        np.testing.assert_allclose(logs[0].loc[4:].to_numpy(),
                                   logs[1].loc[4:].to_numpy(), rtol=1e-6)
        self.assertEqual(ckpts[1].rope_scheme, 'yarn')
        self.assertEqual(ckpts[1].max_len, 1024)
        self.assertAlmostEqual(ckpts[1].rope_mscale, d2l.yarn_mscale(4))


class PageCellsTest(unittest.TestCase):
    """The page's CPU cells, run from the page."""
    def test_scheme_checks(self):
        ns = page_ns()
        exec(page_cells()['long-context-rope-inv-freq'], ns)
        with mock.patch.object(d2l, 'rope_inv_freq', ns['rope_inv_freq']):
            out = run_cell('long-context-scheme-checks', ns)
        lines = out.splitlines()
        self.assertLessEqual(float(lines[0].rsplit(' ', 1)[1]), 1e-7)
        self.assertEqual(lines[1], 'PI equals θ/s: True')
        self.assertEqual(lines[2], 'YaRN pairs: 9 interpolated, 12 blended, '
                                   '11 untouched')
        self.assertEqual(lines[3], 'yarn_mscale(4) = 1.1386')
        self.assertTrue(lines[4].startswith('vs Transformers'))
        self.assertLessEqual(max(map(len, lines)), 80)

    def test_attention_share(self):
        out = run_cell('long-context-attention-share', page_ns())
        lines = out.splitlines()
        self.assertEqual(lines[0], 'F: 1.51 / 2.19 / 3.55 GFLOP per token at '
                         '4K / 8K / 16K (attention share 0.45 / 0.62 / 0.77)')
        self.assertIn('-15%', lines[1])
        self.assertEqual(lines[2], 'FlopCounterMode at 16K: ratio 1.000')
        self.assertIn('×2.35', lines[3])

    def test_measure_checks(self):
        torch.manual_seed(0)
        ns = page_ns(model=toy_model(max_len=256), s=4, L=64,
                     books=torch.randint(32, V, (4, 257)))
        out = run_cell('long-context-measure-checks', ns).splitlines()
        self.assertLess(float(out[0].rsplit(' ', 1)[1]), 1e-6)
        self.assertTrue(out[1].startswith('long-context gain: oracle +7.6'))
        self.assertTrue(out[1].endswith('unigram 0.000'))

    def test_task_checks(self):
        ns = page_ns(tok=r2_tok())
        out = run_cell('long-context-task-checks', ns).splitlines()
        self.assertRegex(out[0], r'^item: 16,3\d\d tokens, 1,\d{3} pairs, '
                                 r'query depth 0\.50, 8 candidates$')
        self.assertEqual(out[1], 'oracle copier: 100%, 100%, 100% at 4K, '
                                 '8K, 16K')


class PendingPageTest(unittest.TestCase):
    """The page's later cells run on pending results (empty tables)."""
    def test_cells(self):
        names = {'lc-t1', 'lc-zeroshot', 'lc-t2-evals'}
        hub = {k: v for k, v in d2l.DATA_HUB.items()
               if k not in ('lm-lc-t1-v1', 'lm-lc-evals-v1')}
        ns = page_ns()
        with mock.patch.object(d2l, 'PRETRAIN_PENDING', frozenset(names)), \
                mock.patch.dict(os.environ, {'D2L_RERUN': ''}), \
                mock.patch.dict(d2l.DATA_HUB, hub, clear=True):
            out = run_cell('long-context-zeroshot', ns)
            self.assertEqual(out.splitlines()[0], '[lc-zeroshot pending]')
            run_cell('long-context-schedule', ns)
            for cell in ('arms', 'position', 'recall-grid', 'gain',
                         'first-steps', 'short-cost', 't1'):
                run_cell(f'long-context-{cell}', ns)
            out = run_cell('long-context-scheme-difference', ns)
        self.assertEqual(out.splitlines(), ['T1: pending', 'T2: pending'])
        self.assertIn('pending', ns['lc'].label)

    def test_fallback(self):
        """While the T2 arms are pending, the T1 result under its label."""
        t1 = d2l.LcRuns(*[pd.DataFrame()] * 13, None, 'smoke')
        with mock.patch.object(d2l, 'load_artifact', lambda name: t1):
            res = d2l._pending_result('lc-t2-evals', 'lc-t1')
            zs = d2l._lc_zs_as_tuple(d2l._pending_result('lc-zeroshot',
                                                         'lc-t1'))
        self.assertEqual(res.label, 'T2 run pending; showing the T1 '
                                    'configuration (smoke)')
        self.assertEqual(zs.label, res.label)


class JobsTest(unittest.TestCase):
    """The regeneration jobs call what the page calls."""
    def literal(self, name):
        code = '\n'.join(page_cells().values())
        m = re.search(rf'^{name} = (\{{.*?\}})$', code, re.S | re.M)
        return eval(re.sub(r'#.*', '', m.group(1)), {'dict': dict})

    def test_calls(self):
        self.assertEqual(self.literal('ARMS'), d2l._lc_t2_call()['arms'])
        self.assertEqual(self.literal('ARMS_T1'), d2l._lc_t1_call()['arms'])
        self.assertEqual(self.literal('SCHEMES'),
                         d2l._lc_zeroshot_call()['schemes'])
        cells = page_cells()
        self.assertIn("d2l.lc_zeroshot('r2', schemes=SCHEMES, "
                      "lengths=(4096, 8192, 16_384))",
                      cells['long-context-zeroshot'])
        self.assertIn("d2l.lc_arms(ARMS, base='r2', tokens=TOKENS, "
                      "seq_len=16_384,\n                 batch_tokens=262_144,"
                      " peak_lr=1e-4, warmup=20,\n                 "
                      "decay_frac=0.15, long_share=0.5)",
                      cells['long-context-arms'])
        self.assertIn('TOKENS = 100_000_000', cells['long-context-schedule'])
        self.assertIn("d2l.lc_arms(ARMS_T1, base='t1', tokens=16_000_000, "
                      "seq_len=4096,\n                 batch_tokens=65_536)",
                      cells['long-context-t1'])
        for k in ('pi', 'yarn', 'prog', 'evals'):
            self.assertEqual(d2l.PRETRAIN_JOBS[f'lc-t2-{k}'][0], 'lc_arms')
            self.assertEqual(d2l.PRETRAIN_JOBS[f'lc-t2-{k}'][1](),
                             d2l._lc_t2_call())
        self.assertEqual(d2l.PRETRAIN_JOBS['lc-t1'][1](), d2l._lc_t1_call())
        self.assertEqual(d2l.PRETRAIN_JOBS['lc-zeroshot'][1](),
                         d2l._lc_zeroshot_call())
        self.assertEqual(set(d2l._LC_T2_NAMES.values()) | {
            'lc-t2-evals', 'lc-t1', 'lc-zeroshot'},
            {n for n in d2l.PRETRAIN_RERUN_NAMES if n.startswith('lc-')})


class DataTest(unittest.TestCase):
    """The builder's local shards, when present (no download)."""
    def test_local(self):
        with in_chapter():
            if not all(os.path.exists(os.path.join(p, 'manifest.json'))
                       for p in d2l._LC_LOCAL.values()):
                self.skipTest('local lc shards not built')
            books = d2l.test_books(4, 16_384)
            self.assertEqual(tuple(books.shape), (4, 16_385))
            self.assertFalse((books == 0).any())
            with self.assertRaises(ValueError):
                d2l.test_books(4, 16_385)
            long = d2l._lc_shards('lc-long', 8_000_000)
            self.assertEqual(long.n_shards, 1)
            meta = long.meta(0)
            self.assertEqual(set(meta['part']), {'pg19'})


if __name__ == '__main__':
    unittest.main()
