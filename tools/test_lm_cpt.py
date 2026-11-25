#!/usr/bin/env python3
"""CPU tests for sec_lm_continued_pretraining (chapter_lm-pretraining/
continued-pretraining.md and continued-pretraining-lib.md):
`mix_streams` (counts sum to the batch, the realized share within three
binomial standard errors, the cursor table against a running sum, the
windows each stream gives in its own order, one stream alone equal to a
PackedStream within an epoch, resume through state_dict exact and a
changed mixture refused, ranks 1/2/4 the same multiset); the page's
quadratic cell (frontier identity and the 2-D points, run from the page);
`forgetting_gain` (the page's planted check, identical arms give zero,
paired resamples), `frontier_excess` (and its note when an arm is
missing or the line's forgetting is not resolved from 0) and
`replay_fractions`; `rewarm_bump` (from each arm's own step 0, transient
or not); `book_split`
(deterministic, disjoint, every fifth section held out); `cpt_data('t2')`
in a normal run opens no shard; the arm schedule, configurations and
evaluation grid; `_doc_nll` against a direct loss; a toy two-arm run of
`cpt_arms` end to end (job folders, equal-token and suite files, the
merge, the stored json-table read back, the step-0 row), a job folder of
another arm refused, and a resumed arm equal to an uninterrupted one; the
page's T2 cells on a pending result and the fallback's own label kept; the
SmolLM3 merge (token counts, byte-weighted bits per byte) and
`stage_deltas` with the page's cell; the regeneration jobs against the
page's literals.
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
PAGE = os.path.join(ROOT, 'chapter_lm-pretraining',
                    'continued-pretraining.md')
TOY = d2l.Arch(L=2, d=32, H=4, H_kv=2, d_h=8, f=64, vocab=512, T=32,
               rope_base=1e4, qk_norm=False, tie=True)


class IdTokenizer:
    """A tiktoken-like encoding whose texts are space-separated ids."""
    n_vocab = 512
    special_tokens_set = {'<|endoftext|>'}

    def encode_single_token(self, s):
        return 0

    def encode_ordinary(self, t):
        return [int(x) for x in t.split()]

    def encode_ordinary_batch(self, ts):
        return [self.encode_ordinary(t) for t in ts]


def fixture_shards(out, n_docs=300, seed=0, lo=5, hi=120):
    rng = np.random.default_rng(seed)
    docs = [(f'd{i}', ' '.join(map(str, rng.integers(32, 512, rng.integers(
        lo, hi + 1))))) for i in range(n_docs)]
    d2l.write_shards(docs, IdTokenizer(), out, 'train', shard_tokens=10 ** 9,
                     eos_id=0)
    return d2l.TokenShards(out)


def page_cells():
    src = open(PAGE).read()
    return dict(re.findall(r'```\{\.python \.input #([^}]*)\}\n'
                           r'%%tab pytorch\n(.*?)```', src, re.S))


def run_cell(name, ns):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(page_cells()[name], ns)
    return out.getvalue()


def page_ns():
    return dict(math=math, np=np, pd=pd, torch=torch, d2l=d2l)


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.shards = {k: fixture_shards(os.path.join(cls.tmp.name, k),
                                        seed=i)
                      for i, k in enumerate(('pes2o', 'book', 'replay',
                                             'val_domain', 'val_general',
                                             'val_book'))}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def streams(self, T=16, B=8, seeds=(0, 1)):
        return [d2l.PackedStream(self.shards[k], T, B, seed=s)
                for k, s in zip(('pes2o', 'replay'), seeds)]


class MixTest(Fixtures):
    def test_counts_and_share(self):
        mix = d2l.mix_streams(self.streams(B=128), (0.75, 0.25), 128)
        counts = np.stack([mix.counts(s) for s in range(10_000)])
        self.assertTrue((counts.sum(1) == 128).all())
        share = counts[:, 1].sum() / counts.sum()
        self.assertLess(abs(share - 0.25),
                        3 * math.sqrt(0.25 * 0.75 / counts.sum()))

    def test_cursor_table(self):
        mix = d2l.mix_streams(self.streams(), (0.5, 0.3, 0.2)[:2], 8, seed=3)
        running = np.zeros(2, dtype=np.int64)
        for s in range(300):
            np.testing.assert_array_equal(mix.cursors(s), running)
            running += mix.counts(s)

    def test_windows_in_stream_order(self):
        streams = self.streams()
        mix = d2l.mix_streams(streams, (0.7, 0.3), 8, seed=1)
        S = 40  # beyond one epoch of the smaller stream's windows? any S
        got = np.concatenate([np.concatenate([
            b['input_ids'].numpy(), b['labels'].numpy()[:, -1:]], 1)
            for b in (mix.batch(s) for s in range(S))])
        want = np.concatenate([d2l._stream_windows(st, 0, int(c))
                               for st, c in zip(streams, mix.cursors(S))])
        key = lambda a: sorted(map(tuple, a[:, :-1]))
        self.assertEqual(key(got), key(want))

    def test_single_stream_is_packed_stream(self):
        st = self.streams()[0]
        mix = d2l.mix_streams([d2l.PackedStream(st.shards, 16, 8)], (1.0,), 8)
        for s in range(st.steps_per_epoch):
            a = mix.batch(s)['input_ids'].numpy()
            b = st.batch(s)['input_ids'].numpy()
            self.assertEqual(sorted(map(tuple, a)), sorted(map(tuple, b)))

    def test_resume_exact(self):
        mix = d2l.mix_streams(self.streams(), (0.75, 0.25), 8, seed=2)
        it = iter(mix)
        for _ in range(37):
            next(it)
        self.assertEqual(mix.step, 37)
        resumed = d2l.mix_streams(self.streams(), (0.75, 0.25), 8, seed=5)
        resumed.load_state_dict(mix.state_dict())
        a, b = iter(mix), iter(resumed)
        for _ in range(20):
            x, y = next(a), next(b)
            for k in x:
                self.assertTrue(torch.equal(x[k], y[k]), k)
        other = d2l.mix_streams(self.streams(), (0.5, 0.5), 8)
        with self.assertRaises(ValueError):
            other.load_state_dict(mix.state_dict())

    def test_ranks_same_multiset(self):
        full = d2l.mix_streams(self.streams(), (0.6, 0.4), 8, seed=4)
        for world in (2, 4):
            parts = [d2l.mix_streams(self.streams(), (0.6, 0.4), 8, seed=4,
                                     rank=r, world=world) for r in range(world)]
            for s in (0, 5, 17):
                want = sorted(map(tuple, full.batch(s)['input_ids'].numpy()))
                got = sorted(map(tuple, np.concatenate([
                    p.batch(s)['input_ids'].numpy() for p in parts])))
                self.assertEqual(got, want)

    def test_zero_weight_stream(self):
        mix = d2l.mix_streams(self.streams(), (1.0, 0.0), 8)
        self.assertTrue((mix.cursors(50)[1] == 0))
        self.assertEqual(mix.batch(3)['input_ids'].shape, (8, 16))
        self.assertEqual(d2l._mix_tokens(mix, 10, (0,)), 10 * 8 * 16)


class QuadraticTest(unittest.TestCase):
    def test_page_cell(self):
        ns = page_ns()
        with mock.patch.object(d2l, 'plot'):
            out = run_cell('continued-pretraining-quadratic', ns)
        lines = out.splitlines()
        self.assertEqual(len(lines), 6)
        self.assertRegex(lines[0], r'max error \d\.\de[-+]\d\d$')
        row = dict(zip(['f', 'replay', 'es'], lines[4].split()))
        self.assertEqual((row['f'], row['replay'], row['es']),
                         ('0.2', '0.659', '0.358'))


def synthetic_docs(n=300, seed=0, gain=0.05, forget=0.02, noise=0.05):
    rng = np.random.default_rng(seed)
    n_tok = rng.integers(100, 3000, 2 * n)
    per = 2.5 + 0.3 * rng.standard_normal(2 * n)
    base = pd.DataFrame(dict(set=np.repeat(['domain', 'general'], n),
                             doc_id=np.tile(np.arange(n), 2), n_tokens=n_tok,
                             sum_nll=per * n_tok))
    dom = (base['set'] == 'domain').to_numpy()
    shift = np.where(dom, -gain, forget)
    arm = base.assign(arm='X', sum_nll=(per + shift + noise *
                                        rng.standard_normal(2 * n)) * n_tok)
    return base, arm


class ForgettingGainTest(unittest.TestCase):
    def test_page_check(self):
        out = run_cell('continued-pretraining-estimator-check', page_ns())
        self.assertRegex(out, r'^planted 0\.050: estimated 0\.0\d\d '
                              r'\[0\.0\d\d, 0\.0\d\d\]\n$')

    def test_identical_arm(self):
        base, _ = synthetic_docs()
        fg = d2l.forgetting_gain(base, base.assign(arm='same'))
        self.assertEqual(fg[['gain', 'gain_lo', 'gain_hi', 'forget',
                             'forget_lo', 'forget_hi']].abs().max().max(), 0)

    def test_two_arms_and_frontier(self):
        base, a = synthetic_docs(gain=0.10, forget=0.04, noise=0.01)
        _, c = synthetic_docs(gain=0.05, forget=0.04, noise=0.01)
        _, b = synthetic_docs(gain=0.08, forget=0.01, noise=0.01)
        docs = pd.concat([a.assign(arm='A'), b.assign(arm='B'),
                          c.assign(arm='C')])
        fg = d2l.forgetting_gain(base, docs)
        self.assertEqual(list(fg['arm']), ['A', 'B', 'C'])
        r = fg.set_index('arm')
        for arm, g, f in (('A', .10, .04), ('B', .08, .01), ('C', .05, .04)):
            self.assertAlmostEqual(r.gain[arm], g, delta=0.005)
            self.assertAlmostEqual(r.forget[arm], f, delta=0.005)
            self.assertLess(r.gain_lo[arm], r.gain[arm])
        ex = d2l.frontier_excess(fg, 'B', 'C')
        self.assertAlmostEqual(ex.value, 0.08 - 0.05 * 0.01 / 0.04,
                               delta=0.01)
        self.assertGreater(ex.lo, 0)
        self.assertEqual(ex.note, '')
        miss = d2l.frontier_excess(fg, 'B', 'Z')
        self.assertTrue(math.isnan(miss.value))
        self.assertEqual(miss.note, 'needs arm Z (T2 only)')
        kept, forgot = d2l.replay_fractions(fg, 'B', of='A')
        self.assertAlmostEqual(kept.value, 0.8, delta=0.06)
        self.assertAlmostEqual(forgot.value, 0.25, delta=0.15)
        self.assertLess(kept.lo, kept.value)
        self.assertGreater(kept.hi, kept.value)
        self.assertIn('needs arm', d2l.replay_fractions(fg, 'B', 'Z')[0].note)

    def test_frontier_unresolved(self):
        """C's forgetting interval reaches 0: no line, a note instead."""
        base, a = synthetic_docs(gain=0.10, forget=0.04, noise=0.01)
        _, c = synthetic_docs(gain=0.01, forget=0.0, noise=0.0)
        _, b = synthetic_docs(gain=0.08, forget=0.01, noise=0.01)
        fg = d2l.forgetting_gain(base, pd.concat([
            a.assign(arm='A'), b.assign(arm='B'), c.assign(arm='C')]))
        self.assertLessEqual(fg.set_index('arm').forget_lo['C'], 0)
        ex = d2l.frontier_excess(fg, 'B', 'C')
        self.assertTrue(math.isnan(ex.value))
        self.assertIn('not resolved', ex.note)

    def test_empty(self):
        cols = d2l._CPT_COLUMNS
        fg = d2l.forgetting_gain(pd.DataFrame(columns=cols['base_docs']),
                                 pd.DataFrame(columns=cols['docs']))
        self.assertEqual(len(fg), 0)
        self.assertIn('forget_hi', fg.columns)


class BumpTest(unittest.TestCase):
    def test_rewarm_bump(self):
        """Measured from each arm's own step 0: a plateau branch that starts
        above the base and falls has no bump; a loss still rising at the
        window's end is no transient."""
        logs = pd.DataFrame(dict(
            arm=['base'] + ['A'] * 5 + ['C'] * 3 + ['branch'] * 4,
            step=[0, 0, 3, 6, 9, 45, 0, 3, 6, 0, 3, 6, 9],
            general_loss=[3.0, 3.0, 3.2, 3.1, 3.05, 2.9, 3.0, 3.01, 3.02,
                          3.3, 3.28, 3.25, 3.2]))
        runs = d2l.CptRuns(None, logs, None, None, None, None, '')
        b = d2l.rewarm_bump(runs).set_index('arm')
        self.assertEqual(list(b.index), ['A', 'C', 'branch'])
        self.assertAlmostEqual(b.bump['A'], 0.2)
        self.assertEqual(b.step['A'], 3)
        self.assertAlmostEqual(b.healed['A'], 0.3)
        self.assertTrue(math.isnan(b.bump['C']))  # still rising at step 6
        self.assertEqual(b.step['C'], 6)
        self.assertAlmostEqual(b.bump['branch'], -0.02)  # not 0.28
        out = run_cell('continued-pretraining-bump-table', dict(
            page_ns(), runs=runs._replace(table=pd.DataFrame(dict(
                arm=['A', 'C', 'branch'], peak_lr=[1e-4, 1e-5, 6e-4])))))
        self.assertIn('no transient', out.splitlines()[2])


class BookSplitTest(unittest.TestCase):
    def test_split(self):
        docs = {f'k{i}': dict(doc_id=f'{i:03d}', text=f't{i}')
                for i in range(23)}
        corpus = mock.Mock(docs=dict(reversed(list(docs.items()))))
        with mock.patch.object(d2l, 'book_corpus', lambda: corpus):
            train, held = d2l.book_split()
            again = d2l.book_split()
        self.assertEqual((train, held), again)
        self.assertEqual([d for d, _ in held], ['004', '009', '014', '019'])
        self.assertFalse({d for d, _ in train} & {d for d, _ in held})
        self.assertEqual(len(train) + len(held), 23)


class DataTest(unittest.TestCase):
    def test_t2_normal_run_opens_nothing(self):
        boom = mock.Mock(side_effect=AssertionError('opened a shard'))
        table = pd.DataFrame(dict(stream=['domain: peS2o'], documents=[1],
                                  tokens=[2], source=['s'], licence=['l']))
        manifest = mock.Mock(table=table)
        with mock.patch.dict(os.environ, {'D2L_RERUN': ''}), \
                mock.patch.object(d2l, '_cpt_qwen_shards', boom), \
                mock.patch.object(d2l, 'TokenShards', boom), \
                mock.patch.object(d2l, 'download_keyed', boom), \
                mock.patch.dict(d2l.DATA_HUB, {'lm-cpt-qwen-manifest-v1':
                                               ('x', 'y')}), \
                mock.patch.object(d2l, 'load_artifact',
                                  lambda name: manifest):
            data = d2l.cpt_data('t2')
        self.assertIsNone(data.shards)
        self.assertIs(data.table, table)

    def test_t2_pending_and_missing(self):
        hub = {k: v for k, v in d2l.DATA_HUB.items()
               if k != 'lm-cpt-qwen-manifest-v1'}
        with mock.patch.dict(d2l.DATA_HUB, hub, clear=True):
            with mock.patch.dict(os.environ, {'D2L_RERUN': ''}):
                with self.assertRaises(KeyError):
                    d2l.cpt_data('t2')
            with mock.patch.dict(os.environ, {'D2L_RERUN': 'cpt-t1'}), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                data = d2l.cpt_data('t2')
        self.assertEqual(out.getvalue(), '[cpt-qwen-manifest pending]\n')
        self.assertEqual(len(data.table), 0)
        self.assertIn('licence', data.table.columns)
        with self.assertRaises(ValueError):
            d2l.cpt_data('t3')


class ScheduleTest(unittest.TestCase):
    def test_lr(self):
        cfg = dict(steps=305, warmup=10, final_frac=0.1, schedule='cosine')
        lr = [d2l._cpt_lr(s, cfg, 1e-4) for s in range(305)]
        self.assertAlmostEqual(lr[0], 1e-5)
        self.assertAlmostEqual(lr[9], 1e-4)
        self.assertAlmostEqual(lr[10], 1e-4)
        self.assertTrue(all(a >= b for a, b in zip(lr[10:], lr[11:])))
        self.assertAlmostEqual(lr[-1], 1e-5, delta=1e-8)
        lin = dict(cfg, warmup=0, schedule='linear', steps=100)
        lr = [d2l._cpt_lr(s, lin, 6e-4) for s in range(100)]
        self.assertAlmostEqual(lr[0], 6e-4)
        self.assertAlmostEqual(lr[-1], 6e-6)

    def test_configs(self):
        c = d2l._cpt_configs(**dict(d2l._cpt_qwen_call(), base=d2l.
                                    _cpt_qwen_call()['base']))
        self.assertEqual({a: x['steps'] for a, x in c.items()},
                         dict.fromkeys(c, 305))
        self.assertEqual(c['A']['global_batch'], 128)
        self.assertEqual({a: x['equal_step'] for a, x in c.items()},
                         {'A': 229, 'B': None, 'C': None, 'A′': 229})
        self.assertEqual({a: x['suite_step'] for a, x in c.items()},
                         {'A': 15, 'B': None, 'C': None, 'A′': None})
        self.assertEqual(c['A']['z_loss'], 0.0)
        self.assertEqual(c['A']['tier'], 't2')
        t1 = d2l._cpt_t1_call()
        c = d2l._cpt_configs(t1['arms'], t1['base'], t1['tokens'],
                             t1['seq_len'], t1['batch_tokens'], 10, 0.1)
        self.assertEqual(c['A']['steps'], 183)
        self.assertEqual(c['A']['global_batch'], 64)
        self.assertEqual({a: x['equal_step'] for a, x in c.items()},
                         {'A': 137, 'A′': 137, 'A″': 137, 'B': None,
                          'cold': None, 'branch': None})
        self.assertEqual((c['cold']['warmup'], c['branch']['start'],
                          c['branch']['schedule']), (0, 'plateau', 'linear'))
        self.assertEqual(c['A']['z_loss'], 1e-4)

    def test_eval_steps(self):
        s = d2l._cpt_eval_steps(305, 't2')
        self.assertEqual(s[:10], [3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
        self.assertIn(300, s)
        self.assertEqual(s[-1], 305)
        self.assertIn(35, d2l._cpt_eval_steps(183, 't1'))


def toy_model(seed=0):
    torch.manual_seed(seed)
    return d2l.PackedGPT.from_arch(TOY)


class DocNllTest(Fixtures):
    def test_against_direct_loss(self):
        model, s, T = toy_model().eval(), self.shards['val_domain'], 32
        docs = d2l._doc_nll(model, s, T, tokens=2000, batch=3)
        n_win = (2000 - 1) // T
        toks = torch.from_numpy(s.read(0, n_win * T + 1).astype(np.int64))
        W = torch.stack([toks[j * T:(j + 1) * T + 1] for j in range(n_win)])
        X, Y = W[:, :-1], W[:, 1:].clone()
        Y[X == 0] = -100
        ctx = d2l.PackCtx(*d2l.doc_ids_from_eos(X.int(), 0))
        with torch.no_grad():
            logits = model(X, ctx)
        ce = F.cross_entropy(logits.flatten(0, 1).float(), Y.flatten(),
                             ignore_index=-100, reduction='sum')
        self.assertEqual(int(docs['n_tokens'].sum()), int((Y != -100).sum()))
        self.assertAlmostEqual(docs['sum_nll'].sum() / float(ce), 1.0,
                               places=5)
        # a document's targets: its tokens after its first, and its EOS
        eos = np.flatnonzero(toks.numpy() == 0)
        self.assertEqual(int(docs.set_index('doc_id').n_tokens[0]), eos[0])


class ArmsEndToEndTest(Fixtures):
    """Two toy arms through cpt_arms with D2L_RERUN=cpt-t1: the training
    loop, the evaluation grid, the equal-token and suite files, the merge
    and the stored result."""
    def patches(self, root):
        items = lambda: pd.DataFrame(dict(task=['t1', 't1', 't2'],
                                          item_id=['a', 'b', 'c'],
                                          score=[1.0, 0.0, 1.0]))
        base = toy_model()
        return [mock.patch.object(torch.cuda, 'is_available', lambda: False),
                mock.patch.object(d2l, '_job_root', lambda: root),
                mock.patch.object(d2l, '_rerun_dir',
                                  lambda: os.path.join(root, 'rerun')),
                mock.patch.object(d2l, '_cpt_base', lambda cfg, dtype=None,
                                  device='cpu': (copy_model(base), None,
                                                 1e-3, '')),
                mock.patch.object(d2l, 'cpt_data', lambda tier: d2l.CptData(
                    None, self.shards, '')),
                mock.patch.object(d2l, '_cpt_base_id',
                                  lambda cfg: ['toy', 'base-0']),
                mock.patch.object(d2l, '_cpt_tokenizer', lambda tier: None),
                mock.patch.object(d2l, '_cpt_suite', lambda *a, **k: None),
                mock.patch.object(d2l, '_suite_items',
                                  lambda *a, **k: items()),
                mock.patch.object(d2l, '_CPT_EVAL_TOKENS', 600),
                mock.patch.object(d2l, '_CPT_FULL_TOKENS', 1200),
                mock.patch.dict(d2l._CPT_MICRO, {'t1': 4}),
                mock.patch.dict(os.environ, {'D2L_RERUN': 'cpt-t1'})]

    def test_two_arms(self):
        arms = {'A': dict(peak_lr=1e-3, replay=0.0, data_seed=0),
                'B': dict(peak_lr=1e-3, replay=0.25, data_seed=0)}
        with tempfile.TemporaryDirectory() as root, contextlib.ExitStack() \
                as stack, contextlib.redirect_stdout(io.StringIO()):
            for p in self.patches(root):
                stack.enter_context(p)
            runs = d2l.cpt_arms(arms, base='r2', tokens=16 * 256,
                                seq_len=32, batch_tokens=256)
            self.assertTrue(os.path.exists(os.path.join(root, 'cpt', 't1-A',
                                                        'DONE')))
            self.assertFalse(os.path.exists(os.path.join(root, 'cpt', 't1-A',
                                                         'state')))
            stored = d2l._read_result(os.path.join(
                root, 'rerun', d2l._next_key('lm-cpt-t1-v1') + '.json.gz'),
                'json-table')
        self.assertEqual(list(runs.table['arm']), ['base', 'A', 'B'])
        self.assertEqual(list(runs.table.columns),
                         d2l._CPT_COLUMNS['table'])
        t = runs.table.set_index('arm')
        self.assertEqual(t.steps['A'], 16)
        self.assertEqual(t.domain_tokens['A'], 16 * 8 * 32)
        self.assertLess(t.domain_tokens['B'], t.domain_tokens['A'])
        self.assertEqual(list(runs.equal['arm']), ['A'])
        self.assertEqual(int(runs.equal['step'].iloc[0]), 12)
        logs = runs.logs
        self.assertEqual(sorted(logs.loc[logs.arm == 'A', 'step']),
                         list(range(0, 17)))
        graded = logs[(logs.arm == 'A') & logs.domain_loss.notna()]
        self.assertEqual(list(graded.step), [0, 3, 5, 6, 9, 10, 12, 15, 16])
        # step 0 is the arm's own model before its first update: the base
        start = logs[(logs.arm == 'A') & (logs.step == 0)].iloc[0]
        self.assertAlmostEqual(start.general_loss, logs[
            logs.arm == 'base'].general_loss.iloc[0], places=5)
        self.assertEqual(sorted(runs.items.loc[runs.items.arm == 'A',
                                               'step'].unique()), [15, 16])
        self.assertEqual(set(runs.docs['set']), {'domain', 'general', 'book'})
        self.assertAlmostEqual(t.suite['base'], 0.75)
        self.assertEqual(list(stored.table['arm']), ['base', 'A', 'B'])
        self.assertEqual(stored.table.attrs['config']['tokens'], 4096)

    def test_job_identity(self):
        """A finished or started folder of another arm is refused."""
        arms = {'A': dict(peak_lr=1e-3, replay=0.0, data_seed=0)}
        with tempfile.TemporaryDirectory() as root, contextlib.ExitStack() \
                as stack, contextlib.redirect_stdout(io.StringIO()):
            for p in self.patches(root):
                stack.enter_context(p)
            cfg = d2l._cpt_configs(arms, 'r2', 4 * 256, 32, 256, 2, 0.1)['A']
            d2l._cpt_arm_run(cfg)
            with open(os.path.join(root, 'cpt', 't1-A', 'DONE')) as f:
                self.assertEqual(json.load(f)['base'], ['toy', 'base-0'])
            d2l._cpt_arm_run(cfg)  # the same arm: read back
            with self.assertRaisesRegex(ValueError, 'config.peak_lr'):
                d2l._cpt_arm_run(dict(cfg, peak_lr=5e-4))
            with mock.patch.object(d2l, '_cpt_base_id',
                                   lambda cfg: ['toy', 'base-1']), \
                    self.assertRaisesRegex(ValueError, 'another arm'):
                d2l._cpt_arm_run(cfg)
            os.remove(os.path.join(root, 'cpt', 't1-A', 'DONE'))
            with self.assertRaisesRegex(ValueError, 'config.data_seed'):
                d2l._cpt_arm_run(dict(cfg, data_seed=1))  # arm.json

    def test_resume_equal(self):
        cfg = dict(d2l._cpt_configs(
            {'A': dict(peak_lr=1e-3, replay=0.25, data_seed=0)}, 'r2',
            60 * 256, 32, 256, 5, 0.1)['A'], micro_batch=4, evaluate=False,
            suite_step=None)
        logs = []
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            for name, stops in (('straight', (None,)),
                                ('resumed', (55, None))):
                job = os.path.join(root, name)
                for stop in stops:
                    mix = d2l._cpt_mix(self.shards, cfg)
                    log = d2l._cpt_train(toy_model(), mix, self.shards, cfg,
                                         job, peak=1e-3, stop=stop)
                logs.append(log.set_index('step')['loss'])
        self.assertEqual(len(logs[0]), 60)
        np.testing.assert_array_equal(logs[0].loc[51:].to_numpy(),
                                      logs[1].loc[51:].to_numpy())


def copy_model(m):
    import copy
    return copy.deepcopy(m)


class PendingPageTest(unittest.TestCase):
    """The page's T2 cells run on a pending result (empty tables)."""
    def test_cells(self):
        names = set(d2l._CPT_QWEN_NAMES.values()) | {'cpt-qwen-evals',
                                                     'cpt-t1'}
        hub = {k: v for k, v in d2l.DATA_HUB.items()
               if k not in ('lm-cpt-runs-v1', 'lm-cpt-t1-v1')}
        ns = page_ns()
        with mock.patch.object(d2l, 'PRETRAIN_PENDING', frozenset(names)), \
                mock.patch.dict(os.environ, {'D2L_RERUN': ''}), \
                mock.patch.dict(d2l.DATA_HUB, hub, clear=True):
            out = run_cell('continued-pretraining-opening', ns)
            self.assertEqual(out.splitlines()[0], '[cpt-qwen-evals pending]')
            for cell in ('arms', 'bump-table', 'frontier', 'equal-tokens',
                         'suite-deltas'):
                out = run_cell(f'continued-pretraining-{cell}', ns)
                if cell == 'frontier':
                    self.assertIn('C: needs arm B, C (T2 only)', out)
        self.assertIn('pending', ns['runs'].label)

    def test_fallback_label_kept(self):
        """The T1 fallback keeps its own label ('t1 preset')."""
        t1 = d2l.CptRuns(pd.DataFrame(), pd.DataFrame(), None, None, None,
                         None, 't1 preset')
        with mock.patch.object(d2l, 'load_artifact', lambda name: t1):
            res = d2l._pending_result('cpt-qwen-evals', 'cpt-t1')
        self.assertEqual(res.label, 'T2 run pending; showing the T1 '
                                    'configuration (t1 preset)')


class SmolLM3Test(unittest.TestCase):
    def test_merge(self):
        names = list(d2l._smollm3_call()['names'])
        pts = {}
        for i, n in enumerate(names):
            items = pd.DataFrame([dict(task=t, item_id=f'{t}{j}',
                                       score=float((i + j) % 2),
                                       checkpoint=d2l._SMOLLM3[n][0])
                                  for t in d2l.SMOLLM3_TASKS
                                  for j in range(4)])
            docs = pd.DataFrame([dict(checkpoint=d2l._SMOLLM3[n][0],
                                      domain=dom, doc=k, bpb=1.0 + k,
                                      bytes=100 * (k + 1))
                                 for dom in d2l._SMOLLM3_DOMAINS
                                 for k in range(2)])
            pts[n] = dict(items=items, docs=docs,
                          ctx=65536 if n in ('lc64k', 'mid') else 4096)
        traj = d2l._smollm3_merge(pts, names)
        self.assertEqual(list(traj.table['tokens']),
                         ['8.1T', '9.9T', '11.1T', '11.2T', '11.2T', '11.4T'])
        self.assertEqual(list(traj.table['ctx'])[-2:], ['64K', '64K'])
        self.assertEqual(list(traj.table.columns),
                         ['checkpoint', 'tokens', 'ctx', *d2l.SMOLLM3_TASKS])
        # byte-weighted: (1 * 100 + 2 * 200) / 300
        self.assertAlmostEqual(traj.bpb['web'].iloc[0], 500 / 300)
        self.assertEqual(len(traj.bpb), 6)
        short = dict(hellaswag='hswag', arc_easy='arc-e',
                     arc_challenge='arc-c', winogrande='wino')
        line = traj.table.rename(columns=short).to_string(
            index=False, float_format=lambda v:
                                    f'{100 * v:.1f}').splitlines()[0]
        self.assertLessEqual(len(line), 80)

    def test_stage_deltas(self):
        """Paired changes between adjacent checkpoints: a planted +10
        points on one transition and +20% bits per byte on one domain are
        recovered, the rest is 0 within the interval; the page's cell fits
        12 lines of at most 80 columns."""
        names = list(d2l._smollm3_call()['names'])
        rng = np.random.default_rng(0)
        base = rng.random((len(d2l.SMOLLM3_TASKS), 300)) < 0.6
        flip = rng.random(base.shape) < 0.25  # stage 2 -> 3 gains 10 points
        nbytes = rng.integers(500, 5000, 60)
        pts = {}
        for i, n in enumerate(names):
            score = base | flip if i >= 2 else base
            items = pd.DataFrame([dict(task=t, item_id=j,
                                       score=float(score[k, j]),
                                       checkpoint=d2l._SMOLLM3[n][0])
                                  for k, t in enumerate(d2l.SMOLLM3_TASKS)
                                  for j in range(300)])
            bpb = 0.7 + 0.1 * np.sin(np.arange(60))
            docs = pd.DataFrame([dict(checkpoint=d2l._SMOLLM3[n][0],
                                      domain=dom, doc=k, bytes=nbytes[k],
                                      bpb=bpb[k] * (1.2 if dom == 'book'
                                                    and i >= 3 else 1.0))
                                 for dom in d2l._SMOLLM3_DOMAINS
                                 for k in range(60)])
            pts[n] = dict(items=items, docs=docs, ctx=4096)
        traj = d2l._smollm3_merge(pts, names)
        deltas = d2l.stage_deltas(traj).set_index('transition')
        self.assertEqual(list(deltas.index), ['S1 → S2', 'S2 → S3',
                                              'S3 → LC32K', 'LC32K → LC64K',
                                              'LC64K → mid'])
        gain = 100 * (base | flip).mean() - 100 * base.mean()
        self.assertAlmostEqual(deltas.suite['S2 → S3'], gain, places=6)
        self.assertGreater(deltas.suite_lo['S2 → S3'], 0)
        self.assertEqual(deltas.suite['S1 → S2'], 0)
        self.assertAlmostEqual(deltas.book['S3 → LC32K'], 20, places=6)
        self.assertGreater(deltas.book_lo['S3 → LC32K'], 19.9)
        self.assertAlmostEqual(deltas.web['S3 → LC32K'], 0)
        out = run_cell('continued-pretraining-stage-deltas',
                       dict(page_ns(), traj=traj))
        lines = out.splitlines()
        self.assertEqual(len(lines), 12)
        self.assertLessEqual(max(map(len, lines)), 80)
        self.assertIn('+20.0 [+20.0, +20.0]', lines[9])


class JobsTest(unittest.TestCase):
    """The regeneration jobs call what the page calls."""
    def literal(self, name):
        code = '\n'.join(page_cells().values())
        m = re.search(rf'^{name} = (\{{.*?\}}|\(.*?\))$', code, re.S | re.M)
        return eval(re.sub(r'#.*', '', m.group(1)), {'dict': dict})

    def test_arms(self):
        self.assertEqual(self.literal('ARMS'), d2l._cpt_qwen_call()['arms'])
        self.assertEqual(self.literal('ARMS_T1'), d2l._cpt_t1_call()['arms'])
        cells = page_cells()
        self.assertIn("d2l.cpt_arms(ARMS, base='Qwen/Qwen3-0.6B-Base', "
                      "tokens=80_000_000,\n                    seq_len=2048, "
                      "batch_tokens=262_144, warmup=10,\n                    "
                      "final_frac=0.1)", cells['continued-pretraining-arms'])
        self.assertIn("d2l.cpt_arms(ARMS_T1, base='r2', tokens=12_000_000, "
                      "seq_len=1024,\n                  batch_tokens=65_536)",
                      cells['continued-pretraining-t1-arms'])
        self.assertEqual(self.literal('SMOLLM3_STAGES'),
                         d2l._smollm3_call()['names'])
        for k in ('A', 'B', 'C', 'A2', 'evals'):
            self.assertEqual(d2l.PRETRAIN_JOBS[f'cpt-qwen-{k}'][0],
                             'cpt_arms')
        self.assertEqual(d2l.PRETRAIN_JOBS['cpt-t1'][1](),
                         d2l._cpt_t1_call())
        for k in range(6):
            self.assertEqual(d2l.PRETRAIN_JOBS[f'smollm3-{k}'][0],
                             'smollm3_trajectory')


if __name__ == '__main__':
    unittest.main()
