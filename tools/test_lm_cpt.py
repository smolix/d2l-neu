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
paired resamples), `frontier_excess` against the C -> A chord on a
synthetic concave frontier (on the chord 0, under it negative, a replay
arm forgetting less than C against the base -> C segment, one forgetting
more than A, or one that does not forget, no verdict; notes when an arm
is missing or a span is not resolved from 0) and `replay_fractions`; an arm's two-stream mixture
(traces and replay in (1 - r, r), no book); the data in the job identity
(T2 manifest layout, T1 tar sha1) and in a result's configuration;
cold-plateau's fresh moments against the branch's restored ones; GSM8K in
two formats (the ChatML prompt and its tokens, the answer after
</think> with a box preferred, a cut or never-closed reply has none, the
forced-answer readout for exactly those replies, chat markers in the
completion format, budgets and end tokens; the Transformers weight map
inverted; the per-model cache of cpt-qwen-evals, refused under another
setting or batch size); the page's T2 cells on a synthetic result (shapes,
widths); `rewarm_bump` (from each arm's own step 0, transient or not);
`early_gaps` (paired gaps and the seed yardstick over steps 1-40);
`book_split`
(deterministic, disjoint, every fifth section held out); `cpt_data('t2')`
in a normal run opens no shard; the arm schedule, configurations and
evaluation grid; `_doc_nll` against a direct loss; a toy two-arm run of
`cpt_arms` end to end (job folders, equal-token and suite files, the
merge, the stored json-table read back, the step-0 row), a job folder of
another arm refused, and a resumed arm equal to an uninterrupted one; the
page's T2 cells on a pending result and the fallback's own label kept; the
SmolLM3 merge (token counts, byte-weighted bits per byte) and
`stage_deltas` with the page's cell; the regeneration jobs against the
page's literals; the reasoning-trace corpus of
tools/build_pretraining_data.py on a 30-document fixture (ids from the
ChatML text, stable under the order of the files; decontamination and
special strings; training selection in id order; held-out questions
disjoint, one per question; the overlap rule; the check's invariants
failing on a changed byte, file list or target; the T1 epoch line), the
endings (open, capped, complete; counted per split; a selection keeping
complete traces only), the licence re-read on both build paths (a changed
licence stops the build), `chatml`, `trace_conversations` (refuses
conversations the committed manifest does not list) and `trace_windows`
(one window per conversation, shorter ones whole; on the 500 built
held-out conversations when they are on disk).
"""
import contextlib
import io
import json
import math
import os
import re
import sys
import tempfile
import zlib
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from d2l import torch as d2l

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pretraining_data as builder  # noqa: E402

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
                      for i, k in enumerate(('traces', 'replay',
                                             'val_domain', 'val_general',
                                             'val_book'))}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def streams(self, T=16, B=8, seeds=(0, 1)):
        return [d2l.PackedStream(self.shards[k], T, B, seed=s)
                for k, s in zip(('traces', 'replay'), seeds)]


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

    def test_cpt_mix_two_streams(self):
        """An arm's mixture: reasoning traces and replay in the shares
        (1 - r, r), within three binomial SE over 10,000 steps; the book is
        not one of its streams."""
        cfg = dict(seq_len=16, global_batch=128, data_seed=0, replay=0.25)
        mix = d2l._cpt_mix(self.shards, cfg)
        self.assertEqual([st.layout_hash for st in mix.streams],
                         [d2l._layout_hash(self.shards[k])
                          for k in ('traces', 'replay')])
        counts = np.stack([mix.counts(s) for s in range(10_000)])
        share = counts[:, 1].sum() / counts.sum()
        self.assertLess(abs(share - 0.25),
                        3 * math.sqrt(0.25 * 0.75 / counts.sum()))
        none = d2l._cpt_mix(self.shards, dict(cfg, replay=0.0))
        self.assertEqual(int(none.cursors(100)[1]), 0)
        self.assertEqual(d2l._mix_tokens(none, 100, (0,)), 100 * 128 * 16)
        with self.assertRaises(KeyError):  # no 'book' stream is read
            d2l._cpt_mix({k: v for k, v in self.shards.items()
                          if k != 'traces'}, cfg)

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


def shifted(base, gain, forget, noise, seed):
    """An arm from base's documents: per-token loss lower by gain on the
    domain set, higher by forget on the general set, plus noise."""
    rng = np.random.default_rng(seed)
    n = base['n_tokens'].to_numpy()
    dom = (base['set'] == 'domain').to_numpy()
    per = base['sum_nll'].to_numpy() / n + np.where(dom, -gain, forget)
    return base.assign(sum_nll=(per + noise * rng.standard_normal(len(n)))
                       * n)


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
        _, c = synthetic_docs(gain=0.05, forget=0.01, noise=0.01)
        _, b = synthetic_docs(gain=0.08, forget=0.02, noise=0.01)
        docs = pd.concat([a.assign(arm='A'), b.assign(arm='B'),
                          c.assign(arm='C')])
        fg = d2l.forgetting_gain(base, docs)
        self.assertEqual(list(fg['arm']), ['A', 'B', 'C'])
        r = fg.set_index('arm')
        for arm, g, f in (('A', .10, .04), ('B', .08, .02), ('C', .05, .01)):
            self.assertAlmostEqual(r.gain[arm], g, delta=0.005)
            self.assertAlmostEqual(r.forget[arm], f, delta=0.005)
            self.assertLess(r.gain_lo[arm], r.gain[arm])
        ex = d2l.frontier_excess(fg, 'B', chord=('C', 'A'))
        self.assertAlmostEqual(ex.value, 0.08 - (0.05 + 0.05 / 3),
                               delta=0.005)
        self.assertGreater(ex.lo, 0)
        self.assertEqual((ex.note, ex.ref), ('', 'C → A chord'))
        miss = d2l.frontier_excess(fg, 'B', chord=('C', 'Z'))
        self.assertTrue(math.isnan(miss.value))
        self.assertEqual(miss.note, 'needs arm Z: not in this result')
        kept, forgot = d2l.replay_fractions(fg, 'B', of='A')
        self.assertAlmostEqual(kept.value, 0.8, delta=0.06)
        self.assertAlmostEqual(forgot.value, 0.5, delta=0.15)
        self.assertLess(kept.lo, kept.value)
        self.assertGreater(kept.hi, kept.value)
        self.assertIn('needs arm', d2l.replay_fractions(fg, 'B', 'Z')[0].note)

    def frontier(self, b_gain, b_forget, noise=0.0):
        """Arms C and A on the concave frontier G = 0.3 sqrt(F) (F = 0.01,
        0.04) and B at (b_forget, b_gain), each with its own noise."""
        base, _ = synthetic_docs()
        docs = pd.concat([shifted(base, g, f, noise, seed).assign(arm=arm)
                          for arm, g, f, seed in (
                              ('C', 0.03, 0.01, 1), ('A', 0.06, 0.04, 2),
                              ('B', b_gain, b_forget, 3))])
        return d2l.forgetting_gain(base, docs)

    def test_frontier_cases(self):
        """The pre-registered cases of claim 6 on a synthetic concave
        frontier: on the chord about 0, under it negative, a B that forgets
        less than C against the segment from the base, a B that forgets
        more than A no verdict."""
        on = d2l.frontier_excess(self.frontier(0.045, 0.025))
        self.assertEqual(on.ref, 'C → A chord')
        self.assertLess(abs(on.value), 1e-9)
        inside = d2l.frontier_excess(self.frontier(0.035, 0.025))
        self.assertAlmostEqual(inside.value, -0.010, places=9)
        # the frontier lies above the chord: a noisy B on it is above it
        top = d2l.frontier_excess(self.frontier(0.3 * 0.025 ** 0.5, 0.025,
                                                noise=0.003))
        self.assertGreater(top.lo, 0)
        self.assertLess(top.lo, top.value)
        low = d2l.frontier_excess(self.frontier(0.02, 0.0049))
        self.assertEqual(low.ref, 'base → C segment')
        self.assertAlmostEqual(low.value, 0.02 - 0.03 * 0.49, places=9)
        for f in (0.0, -0.004):  # B does not forget: no lower bound there
            none = d2l.frontier_excess(self.frontier(0.01, f))
            self.assertTrue(math.isnan(none.value))
            self.assertEqual(none.note, 'B does not forget (outside the '
                                        'measured no-replay range): no '
                                        'verdict')
        far = d2l.frontier_excess(self.frontier(0.07, 0.06))
        self.assertTrue(math.isnan(far.value))
        self.assertEqual(far.note, "B's forgetting is outside the measured "
                                   "no-replay range (above A's): no verdict")
        self.assertEqual(far.ref, '')

    def test_frontier_unresolved(self):
        """A reference whose span of forgetting is not resolved from 0, or
        an A that forgets no more than C, gives a note instead."""
        base, _ = synthetic_docs()
        arms = lambda *spec: d2l.forgetting_gain(base, pd.concat([
            shifted(base, g, f, nz, sd).assign(arm=arm)
            for arm, g, f, nz, sd in spec]))
        fg = arms(('C', 0.03, 0.010, 0.0, 1), ('A', 0.06, 0.012, 0.05, 3),
                  ('B', 0.05, 0.011, 0.0, 3))
        r = fg.set_index('arm')
        self.assertGreater(r.forget['A'], r.forget['C'])
        ex = d2l.frontier_excess(fg)
        self.assertTrue(math.isnan(ex.value))
        self.assertIn('not resolved', ex.note)
        fg = arms(('C', 0.03, 0.002, 0.05, 1), ('A', 0.06, 0.04, 0.0, 2),
                  ('B', 0.02, 0.001, 0.0, 3))  # forgets, less than C
        ex = d2l.frontier_excess(fg)
        self.assertIn("C's forgetting is not resolved", ex.note)
        fg = arms(('C', 0.03, 0.04, 0.0, 1), ('A', 0.06, 0.01, 0.0, 2),
                  ('B', 0.05, 0.02, 0.0, 3))
        self.assertEqual(d2l.frontier_excess(fg).note,
                         'A forgets no more than C: no chord')

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
        runs = d2l.CptRuns(None, logs, None, None, None, None, None, '')
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


class EarlyGapsTest(unittest.TestCase):
    def test_gaps(self):
        """Largest a - b over steps 1-40 at shared evaluation steps (step 0
        and later steps ignored); the seed yardstick is the largest
        difference between any two seeds; defined when no arm has a
        transient (all still rising); a missing arm drops its row."""
        steps = [0, 3, 6, 40, 45]
        curve = lambda *v: list(v)
        L = {'A': curve(3.0, 3.10, 3.20, 3.30, 9.0),
             'A′': curve(3.0, 3.12, 3.21, 3.31, 0.0),
             'A″': curve(3.0, 3.09, 3.25, 3.30, 5.0),
             'cold': curve(3.0, 3.30, 3.35, 3.40, 3.0),
             'cold-plateau': curve(3.1, 3.30, 3.25, 3.40, 3.0),
             'branch': curve(3.1, 3.12, 3.20, 3.35, 3.0)}
        logs = pd.DataFrame([dict(arm=a, step=s, general_loss=v)
                             for a, vs in L.items()
                             for s, v in zip(steps, vs)])
        runs = d2l.CptRuns(None, logs, None, None, None, None, None, '')
        self.assertTrue(d2l.rewarm_bump(runs)['bump'].isna().all())
        g = d2l.early_gaps(runs, [('cold-plateau', 'branch'), ('cold', 'A'),
                                  ('cold', 'Z')], ['A', 'A′', 'A″'])
        self.assertEqual(list(g['comparison']), [
            'cold-plateau − branch', 'cold − A', "between A's seeds"])
        self.assertEqual(list(g['step']), [3, 3, 6])
        for got, want in zip(g['gap'], (0.18, 0.20, 0.05)):
            self.assertAlmostEqual(got, want)
        empty = d2l.early_gaps(runs._replace(logs=logs[logs['arm'] == 'A']),
                               [('cold', 'A')], ['A', 'A′'])
        self.assertEqual(len(empty), 0)
        pending = runs._replace(logs=pd.DataFrame(
            columns=d2l._CPT_COLUMNS['logs']))
        self.assertEqual(len(d2l.early_gaps(pending, [('cold', 'A')],
                                            ['A', 'A′'])), 0)


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
                mock.patch.dict(d2l.DATA_HUB, {d2l.PRETRAIN_ARTIFACTS[
                    'cpt-qwen-manifest'][0]: ('x', 'y')}), \
                mock.patch.object(d2l, 'load_artifact',
                                  lambda name: manifest):
            data = d2l.cpt_data('t2')
        self.assertIsNone(data.shards)
        self.assertIs(data.table, table)

    def test_t2_pending_and_missing(self):
        key = d2l.PRETRAIN_ARTIFACTS['cpt-qwen-manifest'][0]
        hub = {k: v for k, v in d2l.DATA_HUB.items() if k != key}
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
                          'cold': None, 'cold-plateau': None,
                          'branch': None})
        self.assertEqual((c['cold']['warmup'], c['branch']['start'],
                          c['branch']['schedule']), (0, 'plateau', 'linear'))
        self.assertEqual(c['A']['z_loss'], 1e-4)
        # cold-plateau is the branch with fresh moments, nothing else
        moments = {a: x['moments'] for a, x in c.items()}
        self.assertEqual(moments, dict(dict.fromkeys(c, 'fresh'),
                                       branch='restored'))
        diff = {k for k in c['branch'] if c['branch'][k] !=
                c['cold-plateau'][k]}
        self.assertEqual(diff, {'arm', 'moments'})
        with self.assertRaisesRegex(ValueError, 'moments'):
            d2l._cpt_configs({'X': dict(peak_lr=1e-4, moments='restored')},
                             'Qwen/Qwen3-0.6B-Base', 2 ** 20, 2048,
                             262_144, 10, 0.1)

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
                mock.patch.object(d2l, '_cpt_data_id',
                                  lambda tier: {'traces': 'data-0'}),
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
                root, 'rerun', d2l._next_key(d2l.PRETRAIN_ARTIFACTS['cpt-t1'][0])
                + '.json.gz'),
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
        self.assertEqual(stored.table.attrs['config']['data'],
                         {'traces': 'data-0'})
        self.assertEqual(len(runs.gsm8k), 0)  # T2 only
        self.assertEqual(list(runs.gsm8k.columns),
                         d2l._CPT_COLUMNS['gsm8k'])
        self.assertTrue(t[['gsm8k_completion', 'gsm8k_chat']].isna().all(
            ).all())

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
            with mock.patch.object(d2l, '_cpt_data_id',  # another corpus
                                   lambda tier: {'traces': 'data-1'}), \
                    self.assertRaisesRegex(ValueError, 'data.traces'):
                d2l._cpt_arm_run(cfg)
            os.remove(os.path.join(root, 'cpt', 't1-A', 'DONE'))
            with self.assertRaisesRegex(ValueError, 'config.data_seed'):
                d2l._cpt_arm_run(dict(cfg, data_seed=1))  # arm.json
            with mock.patch.object(d2l, '_cpt_data_id',
                                   lambda tier: {'traces': 'data-1'}), \
                    self.assertRaisesRegex(ValueError, 'data.traces'):
                d2l._cpt_arm_run(cfg)  # arm.json of another corpus

    def test_t2_parts_then_evals(self):
        """F23's sequence on toy arms: the four part jobs (cpt-qwen-A, -B,
        -C, -A2) one at a time, each pending until the merge; then
        cpt-qwen-evals, which reads the parts and GSM8K files scored side
        by side (arms=()), merges, and stores the configuration with the
        data; the page's T2 cells run on the merged result."""
        arms = d2l._cpt_qwen_call()['arms']
        call = dict(base='Qwen/Qwen3-0.6B-Base', tokens=20 * 256,
                    seq_len=32, batch_tokens=256)

        def score(model, tok, items, **kw):
            rng = np.random.default_rng(len(str(model['w'])))
            ids = [str(i) for i in range(20)]
            return pd.concat([pd.DataFrame(dict(
                format=fmt, item_id=ids, score=rng.integers(0, 2, 20) * 1.0,
                forced=1.0, cut=fmt == 'chat', thinks=fmt == 'chat',
                closed=False, chat_tokens=False, new_tokens=9))
                for fmt in ('completion', 'chat')], ignore_index=True)

        def load(model, path):
            model['w'] = path
        extra = [mock.patch.dict(d2l._CPT_MICRO, {'t2': 4}),
                 mock.patch.dict(d2l._CPT_MIN_FREE_GB, {'t2': 0}),
                 mock.patch.object(d2l, '_cpt_hf_qwen',
                                   lambda name: (dict(w='base'), 'tok')),
                 mock.patch.object(d2l, '_load_into_hf', load),
                 mock.patch.object(d2l, 'score_gsm8k', score),
                 mock.patch.object(d2l, 'gsm8k_items', lambda n, seed: [])]
        with tempfile.TemporaryDirectory() as root, contextlib.ExitStack() \
                as stack, contextlib.redirect_stdout(io.StringIO()):
            for p in self.patches(root) + extra:
                stack.enter_context(p)
            for part in ('cpt-qwen-A', 'cpt-qwen-B', 'cpt-qwen-C',
                         'cpt-qwen-A2'):
                with mock.patch.dict(os.environ, {'D2L_RERUN': part}):
                    res = d2l.cpt_arms(arms, **call)
                self.assertIn('pending', res.label)
                key = d2l.PRETRAIN_ARTIFACTS['cpt-runs'][0]
                self.assertTrue(os.path.exists(os.path.join(
                    root, 'rerun', f'{d2l._next_key(key)}.{part}.json.gz')))
            t2 = d2l._cpt_configs(arms, warmup=10, final_frac=0.1, **call)
            d2l._cpt_gsm8k_all(t2, arms=())  # the base on its own card
            for arm in ('A', 'C'):  # two arms scored before the merge
                d2l._cpt_gsm8k_all(t2, arms=[arm])
            with mock.patch.dict(os.environ,
                                 {'D2L_RERUN': 'cpt-qwen-evals'}):
                runs = d2l.cpt_arms(arms, **call)
            stored = d2l._read_result(os.path.join(
                root, 'rerun', d2l._next_key(key) + '.json.gz'), 'json-table')
            ns = dict(page_ns(), runs=runs)
            outs = {c: run_cell(f'continued-pretraining-{c}', ns) for c in (
                'gsm8k', 'bump-table', 'frontier', 'frontier-chord',
                'equal-tokens', 'suite-deltas')}
        self.assertEqual(list(runs.table['arm']), ['base', 'A', 'B', 'C',
                                                   'A′'])
        self.assertEqual(stored.table.attrs['config']['data'],
                         {'traces': 'data-0'})
        self.assertEqual(sorted(runs.gsm8k['arm'].unique()),
                         sorted(['base', 'A', 'B', 'C', 'A′']))
        self.assertEqual(len(runs.gsm8k), 5 * 40)
        t = runs.table.set_index('arm')
        self.assertTrue(t.gsm8k_chat.notna().all())
        self.assertEqual(t.gsm8k_forced['A'], 1.0)
        # A and A′ at B's domain-token count (equal-token accounting)
        self.assertEqual(sorted(runs.equal['arm']), sorted(['A', 'A′']))
        self.assertEqual(set(runs.equal['step']), {15})
        self.assertEqual(sorted(runs.items.loc[runs.items.arm == 'A',
                                               'step'].unique()), [15, 20])
        self.assertEqual(len(outs['gsm8k'].splitlines()), 12)
        self.assertEqual(len(outs['frontier'].splitlines()), 7)
        # the toy arms barely move: B forgets more than A, and A′ no more
        # than C, so both chords print their notes (the second on a line
        # of its own)
        self.assertEqual(outs['frontier-chord'].splitlines(), [
            "B's forgetting is outside the measured no-replay range (above "
            "A's): no verdict", 'seed spread (A vs A′): gain 0.000; C → A′ '
            'chord:', '  A′ forgets no more than C: no chord'])
        self.assertEqual(len(outs['equal-tokens'].splitlines()), 3)
        self.assertEqual(len(outs['suite-deltas'].splitlines()), 4)
        for c, out in outs.items():
            self.assertLessEqual(max(map(len, out.splitlines())), 80, c)

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
        stored = {d2l.PRETRAIN_ARTIFACTS[k][0] for k in ('cpt-runs',
                                                         'cpt-t1')}
        hub = {k: v for k, v in d2l.DATA_HUB.items() if k not in stored}
        ns = page_ns()
        with mock.patch.object(d2l, 'PRETRAIN_PENDING', frozenset(names)), \
                mock.patch.dict(os.environ, {'D2L_RERUN': ''}), \
                mock.patch.dict(d2l.DATA_HUB, hub, clear=True), \
                mock.patch.object(d2l, '_cpt_data_id', lambda tier: {}):
            out = run_cell('continued-pretraining-opening', ns)
            self.assertEqual(out.splitlines()[0], '[cpt-qwen-evals pending]')
            for cell in ('arms', 'gsm8k', 'bump-table', 'frontier',
                         'frontier-chord', 'equal-tokens', 'suite-deltas'):
                out = run_cell(f'continued-pretraining-{cell}', ns)
                if cell == 'gsm8k':
                    self.assertEqual(out, 'GSM8K: scored for the Qwen arms '
                                          'only\n')
                if cell == 'frontier-chord':
                    self.assertEqual(out.splitlines()[0],
                                     'needs arm A, B, C: not in this '
                                     'result')
        self.assertIn('pending', ns['runs'].label)

    def test_fallback_label_kept(self):
        """The T1 fallback keeps its own label ('t1 preset')."""
        t1 = d2l.CptRuns(pd.DataFrame(), pd.DataFrame(), None, None, None,
                         None, None, 't1 preset')
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


class Gen:
    """A d2l.Generation."""
    def __new__(cls, text, reason='eos', n=5):
        return d2l.Generation(text, list(range(n)), None, reason)


class FakeChatTokenizer:
    ids = {'<|endoftext|>': 151643, '<|im_end|>': 151645}

    def convert_tokens_to_ids(self, t):
        return self.ids[t]


class GSM8KTest(unittest.TestCase):
    def test_chat_prompt(self):
        self.assertEqual(d2l.gsm8k_chat_prompt('How many?'),
                         '<|im_start|>user\nHow many?<|im_end|>\n'
                         '<|im_start|>assistant\n')

    def test_chat_prompt_tokens(self):
        """Under Qwen3's tokenizer the prompt's markers are its special
        tokens, and generate adds nothing before the prompt."""
        try:
            tok = d2l._cpt_tokenizer('t2')
        except Exception as e:  # the tokenizer is not on this machine
            self.skipTest(f'Qwen3 tokenizer not available: {e}')
        prompt = d2l.gsm8k_chat_prompt('What is 2 + 2?')
        ids = d2l._lm_prompt_ids(tok, [prompt])[0]
        self.assertEqual(ids, tok(prompt, add_special_tokens=False)[
            'input_ids'])
        self.assertEqual((ids[0], ids.count(151644), ids.count(151645)),
                         (151644, 2, 1))
        self.assertEqual(tok.decode(ids), prompt)

    def test_chat_answer(self):
        ans = d2l.gsm8k_chat_answer
        self.assertEqual(ans('<think> 3 + 15 </think> The answer is 18.',
                             'eos'), '18')
        self.assertIsNone(ans('<think> 3 + 15 = 18', 'length'))  # cut
        self.assertIsNone(ans('<think> so 18 </think> 18 and', 'length'))
        self.assertIsNone(ans('<think> so it is 18', 'eos'))  # never closed
        self.assertEqual(ans('The answer is 18.', 'eos'), '18')  # no think
        self.assertEqual(ans('<think> 17 </think> 18', 'eos'), '18')
        self.assertIsNone(ans('<think> a </think> b <think> 18', 'eos'))
        # a box wins over a later number; a box without a number does not
        self.assertEqual(ans('<think> x </think> So $\\boxed{18}$ dollars, '
                             'i.e. 3 bags.', 'eos'), '18')
        self.assertEqual(ans('</think> \\boxed{1,800.00} in 2 days', 'eos'),
                         '1800')
        self.assertEqual(ans('</think> \\boxed{x} so 18', 'eos'), '18')

    def test_force_prompt(self):
        cue = 'The final answer is $\\boxed{'
        self.assertEqual(d2l.gsm8k_force_prompt('P', '<think> 3 + 15'),
                         'P<think> 3 + 15\n</think>\n\n' + cue)
        self.assertEqual(d2l.gsm8k_force_prompt('P', 'Q: q Q: q'),
                         'PQ: q Q: q\n\n' + cue)
        self.assertEqual(d2l.gsm8k_force_prompt('P', '<think> a </think> b'),
                         'P<think> a </think> b\n\n' + cue)

    def test_score(self):
        """score_gsm8k on canned replies: the scoring rules, the flags,
        each format's budget, stop and end tokens."""
        items = [dict(id=str(i), question=f'q{i}', prompt=f'P{i}',
                      reference='18') for i in range(5)]
        replies = {
            'chat': [Gen('<think> 18 </think> The answer is 18.'),
                     Gen('<think> 18 18 18', 'length', 4096),
                     Gen('<think> it is 18'),
                     Gen('The answer is 18.'),
                     Gen('<think> 17 </think> 17')],
            'completion': [Gen('So 18. The answer is 18.\n\n', 'stop'),
                           Gen('The answer is 18.<|im_end|>\nThe answer '
                               'is 19.', 'stop'),
                           Gen('The answer is 17.', 'stop'),
                           Gen('so 3', 'length', 256),
                           Gen('<think> The answer is 18.', 'stop')]}
        replies['forced'] = [Gen('18', 'stop'), Gen('17', 'stop')]
        calls = []

        def generate(model, tok, prompts, max_new, **kw):
            fmt = 'forced' if kw['stop'] == ['}'] else 'chat' if prompts[
                0].startswith('<|im_start|>') else 'completion'
            calls.append((fmt, prompts, max_new, kw))
            return replies[fmt]
        with mock.patch.object(d2l, 'generate', generate):
            res = d2l.score_gsm8k(None, FakeChatTokenizer(), items)
        self.assertEqual(list(res.columns), d2l._CPT_COLUMNS['gsm8k'][1:])
        comp, chat, forced = calls
        # forcing: only the cut reply (1) and the never-closed one (2)
        self.assertEqual(forced[1], [d2l.gsm8k_force_prompt(
            chat[1][i], replies['chat'][i].text) for i in (1, 2)])
        self.assertEqual((forced[2], forced[3]['batch_size']), (32, 8))
        with mock.patch.object(d2l, 'generate', generate):
            calls.clear()
            off = d2l.score_gsm8k(None, FakeChatTokenizer(), items,
                                  formats=('chat',), force=0)
        self.assertEqual([c[0] for c in calls], ['chat'])
        self.assertEqual(list(off['forced']), list(off['score']))
        self.assertEqual((comp[0], comp[1][0], comp[2], comp[3]['stop'],
                          comp[3]['eos_token_ids']),
                         ('completion', 'P0', 256, ['Q:'], [151643]))
        self.assertEqual((chat[1][0], chat[2], chat[3]['stop'],
                          sorted(chat[3]['eos_token_ids'])),
                         (d2l.gsm8k_chat_prompt('q0'), 4096, None,
                          [151643, 151645]))
        r = res.set_index(['format', 'item_id'])
        self.assertEqual(list(r.loc['chat', 'score']), [1, 0, 0, 1, 0])
        self.assertEqual(list(r.loc['chat', 'forced']), [1, 1, 0, 1, 0])
        self.assertEqual(list(r.loc['completion', 'forced']),
                         list(r.loc['completion', 'score']))
        self.assertEqual(list(r.loc['chat', 'cut']),
                         [False, True, False, False, False])
        open_think = r.loc['chat', 'thinks'] & ~r.loc['chat', 'closed']
        self.assertEqual(list(open_think), [False, True, True, False, False])
        # completion: the text before a chat marker is the reply
        self.assertEqual(list(r.loc['completion', 'score']), [1, 1, 0, 0, 0])
        self.assertEqual(list(r.loc['completion', 'chat_tokens']),
                         [False, True, False, False, True])
        self.assertEqual(int(r.loc[('chat', '1'), 'new_tokens']), 4096)

    def test_items(self):
        try:
            items = d2l.gsm8k_items(5)
        except Exception as e:  # the dataset is not on this machine
            self.skipTest(f'GSM8K not available: {e}')
        self.assertEqual(len(items), 5)
        self.assertEqual(items, d2l.gsm8k_items(5))
        it = items[0]
        self.assertTrue(it['prompt'].startswith('Q: There are 15 trees'))
        self.assertTrue(it['prompt'].endswith(it['question'] + '\nA:'))
        self.assertRegex(it['reference'], r'^-?\d+$')

    def test_hf_state_round_trip(self):
        """_qwen3_hf_state inverts map_qwen3_state_dict, and the weights it
        loads into a Transformers Qwen3 give that model's logits."""
        import transformers
        from safetensors.torch import save_file
        cfg = transformers.Qwen3Config(
            vocab_size=96, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
            tie_word_embeddings=True)
        torch.manual_seed(0)
        hf = transformers.Qwen3ForCausalLM(cfg).eval()
        with torch.no_grad():  # norms away from 1, so a permutation shows
            for name, p in hf.named_parameters():
                if 'norm' in name:
                    p.add_(torch.randn_like(p) * 0.3)
        ref = {k: v.clone() for k, v in hf.state_dict().items()}
        ours = d2l.map_qwen3_state_dict(dict(ref), cfg)
        back = d2l._qwen3_hf_state(ours, cfg)
        self.assertEqual(set(back), set(ref) - {'lm_head.weight'})
        for k, v in back.items():
            self.assertTrue(torch.equal(v, ref[k]), k)
        with self.assertRaisesRegex(KeyError, 'not consumed'):
            d2l._qwen3_hf_state(dict(ours, extra=torch.zeros(1)), cfg)
        X = torch.randint(0, 96, (2, 12))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'arm.safetensors')
            save_file({k: v.contiguous() for k, v in ours.items()}, path)
            other = transformers.Qwen3ForCausalLM(cfg).eval()
            d2l._load_into_hf(other, path)
        with torch.no_grad():
            self.assertTrue(torch.allclose(other(X).logits, hf(X).logits,
                                           atol=1e-6))

    def test_all_models_cached(self):
        """cpt-qwen-evals' GSM8K: the base first, then each arm's weights;
        each model's replies kept in its folder and read back; a file
        written under other settings is refused."""
        t2 = d2l._cpt_qwen_call()
        cfgs = d2l._cpt_configs(t2['arms'], t2['base'], t2['tokens'],
                                t2['seq_len'], t2['batch_tokens'], 10, 0.1)
        log = []

        def score(model, tok, items):
            log.append(('score', model['w']))
            return pd.DataFrame(dict(format=['chat'], item_id=['0'],
                                     score=[1.0], forced=[1.0], cut=[False],
                                     thinks=[True], closed=[True],
                                     chat_tokens=[False], new_tokens=[9]))

        def load(model, path):
            log.append(('load', path))
            model['w'] = path
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(d2l, '_job_root', lambda: root), \
                mock.patch.object(d2l, '_cpt_hf_qwen',
                                  lambda name: (dict(w='base'), 'tok')), \
                mock.patch.object(d2l, '_load_into_hf', load), \
                mock.patch.object(d2l, 'score_gsm8k', score), \
                mock.patch.object(d2l, 'gsm8k_items', lambda n, seed: []), \
                mock.patch.object(d2l, '_file_id', lambda p: p), \
                mock.patch.object(d2l, '_cpt_base_id',
                                  lambda cfg: ['qwen', 'rev']):
            res = d2l._cpt_gsm8k_all(cfgs)
            self.assertEqual(list(res['arm']), ['base', 'A', 'B', 'C', 'A′'])
            self.assertEqual(list(res.columns), d2l._CPT_COLUMNS['gsm8k'])
            self.assertEqual(log[0], ('score', 'base'))
            self.assertEqual([k for k, _ in log], ['score'] + ['load',
                                                               'score'] * 4)
            self.assertTrue(log[1][1].endswith(os.path.join(
                't2-A', 'ckpt', 's305.safetensors')))
            log.clear()
            again = d2l._cpt_gsm8k_all(cfgs)  # read back: nothing scored
            self.assertEqual(log, [])
            pd.testing.assert_frame_equal(again, res, check_dtype=False)
            with mock.patch.dict(d2l._CPT_GSM8K, n=100), \
                    self.assertRaisesRegex(ValueError, 'settings.n'):
                d2l._cpt_gsm8k_all(cfgs)
            # another batch size is another setting (not batch-invariant)
            with mock.patch.dict(d2l._GSM8K_BATCH, chat=32), \
                    self.assertRaisesRegex(ValueError, 'settings.batch'):
                d2l._cpt_gsm8k_all(cfgs)

    def test_models_side_by_side(self):
        """F23: the base alone (arms=()), then one arm per process; the
        merge's call reads every file back and scores nothing."""
        t2 = d2l._cpt_qwen_call()
        cfgs = d2l._cpt_configs(t2['arms'], t2['base'], t2['tokens'],
                                t2['seq_len'], t2['batch_tokens'], 10, 0.1)
        log, n = [], [0]

        def score(model, tok, items):
            log.append(('score', model['w']))
            n[0] += 1  # the order in which the models were scored
            return pd.DataFrame(dict(format=['chat'], item_id=['0'],
                                     score=[float(n[0])],
                                     forced=[1.0], cut=[False],
                                     thinks=[True], closed=[True],
                                     chat_tokens=[False], new_tokens=[9]))

        def load(model, path):
            log.append(('load', path))
            model['w'] = path
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(d2l, '_job_root', lambda: root), \
                mock.patch.object(d2l, '_cpt_hf_qwen',
                                  lambda name: (dict(w='base'), 'tok')), \
                mock.patch.object(d2l, '_load_into_hf', load), \
                mock.patch.object(d2l, 'score_gsm8k', score), \
                mock.patch.object(d2l, 'gsm8k_items', lambda n, seed: []), \
                mock.patch.object(d2l, '_file_id', lambda p: p), \
                mock.patch.object(d2l, '_cpt_base_id',
                                  lambda cfg: ['qwen', 'rev']):
            base = d2l._cpt_gsm8k_all(cfgs, arms=())
            self.assertEqual(list(base['arm']), ['base'])
            self.assertEqual(log, [('score', 'base')])
            for arm in ('C', 'A′', 'A', 'B'):  # any order, one per process
                log.clear()
                one = d2l._cpt_gsm8k_all(cfgs, arms=[arm])
                self.assertEqual(list(one['arm']), ['base', arm])
                self.assertEqual([k for k, _ in log], ['load', 'score'])
                self.assertIn(f't2-{d2l._arm_slug(arm)}', log[0][1])
            log.clear()
            merged = d2l._cpt_gsm8k_all(cfgs)  # what cpt-qwen-evals calls
            self.assertEqual(log, [])
            self.assertEqual(list(merged['arm']), ['base', 'A', 'B', 'C',
                                                   'A′'])
            self.assertEqual(list(merged['score']), [1.0, 4.0, 5.0, 2.0,
                                                     3.0])
            with self.assertRaisesRegex(ValueError, 'no configuration'):
                d2l._cpt_gsm8k_all(cfgs, arms=['D'])


class IdentityTest(Fixtures):
    def test_data_id_t2(self):
        layout = dict(traces='t', replay='r', val_domain='vd',
                      val_general='vg', val_book='b')
        with mock.patch.object(d2l, '_cpt_manifest_committed', lambda: True), \
                mock.patch.object(d2l, 'load_artifact', lambda name: mock.Mock(
                    layout=layout)):
            self.assertEqual(d2l._cpt_data_id('t2'), dict(
                traces='t', replay='r', val_domain='vd', val_general='vg'))
        with mock.patch.object(d2l, '_cpt_manifest_committed', lambda: False), \
                self.assertRaises(KeyError):
            d2l._cpt_data_id('t2')
        with self.assertRaises(ValueError):
            d2l._cpt_data_id('t3')

    def test_data_id_t1(self):
        with mock.patch.object(d2l, 'reference_shards',
                               lambda which: self.shards['val_general']):
            got = d2l._cpt_data_id('t1')
        hub = lambda name: d2l.DATA_HUB[d2l.PRETRAIN_ARTIFACTS[name][0]][1]
        self.assertEqual(got, dict(
            traces=hub('cpt-traces-12m'), replay=hub('cpt-fwedu-replay-12m'),
            val_domain=hub('cpt-traces-val'),
            val_general=d2l._layout_hash(self.shards['val_general'])))

    def test_identity_has_data(self):
        cfg = d2l._cpt_configs(**dict(d2l._cpt_qwen_call()))['A']
        with mock.patch.object(d2l, '_cpt_base_id', lambda cfg: ['q', 'r']):
            ids = []
            for data in ({'traces': 'x'}, {'traces': 'y'}):
                with mock.patch.object(d2l, '_cpt_data_id',
                                       lambda tier, d=data: d):
                    ids.append(d2l._cpt_identity(cfg))
        self.assertEqual(ids[0]['config'], ids[1]['config'])
        self.assertNotEqual(ids[0]['data'], ids[1]['data'])

    def test_plateau_moments(self):
        """cold-plateau starts from the plateau state's weights with fresh
        moments; the branch restores them."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 's5100.pt')
            model = toy_model()
            torch.save(dict(cfg=dict(arch=TOY._asdict(), peak_lr=6e-4),
                            model=model.state_dict(),
                            optim=dict(state={}, param_groups=[])), path)
            t1 = d2l._cpt_t1_call()
            cfgs = d2l._cpt_configs(t1['arms'], t1['base'], t1['tokens'],
                                    t1['seq_len'], t1['batch_tokens'], 10,
                                    0.1)
            with mock.patch.object(d2l, '_cpt_r2_files',
                                   lambda: ('r2', 'ckpt', path)):
                got = {a: d2l._cpt_base(cfgs[a]) for a in
                       ('branch', 'cold-plateau')}
        self.assertEqual(got['branch'][1], dict(state={}, param_groups=[]))
        self.assertIsNone(got['cold-plateau'][1])
        for m, _, peak, _ in got.values():
            self.assertEqual(peak, 6e-4)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(m.state_dict()[k], v), k)


def synthetic_runs(n=120, seed=0):
    """CptRuns of the T2 shape: per-document losses for four arms on a
    concave frontier and GSM8K replies of five models."""
    base, _ = synthetic_docs()
    spec = (('A', 0.06, 0.04, 2), ('B', 0.05, 0.02, 3), ('C', 0.03, 0.01, 1),
            ('A′', 0.061, 0.041, 4))
    docs = pd.concat([shifted(base, g, f, 0.003, sd).assign(arm=a)
                      for a, g, f, sd in spec])
    rng = np.random.default_rng(seed)
    rows = [dict(arm=m, format=fmt, item_id=str(i),
                 score=float(rng.random() < p), forced=float(
                     rng.random() < p + 0.1), cut=bool(rng.random() < 0.2),
                 thinks=True, closed=bool(rng.random() < 0.6),
                 chat_tokens=bool(rng.random() < 0.05), new_tokens=100)
            for m, p in (('base', 0.4), ('A', 0.3), ('B', 0.35), ('C', 0.4),
                         ('A′', 0.3))
            for fmt in ('completion', 'chat') for i in range(n)]
    table = pd.DataFrame(dict(arm=['base', 'A', 'B', 'C', 'A′'],
                              peak_lr=[None, 1e-4, 1e-4, 1e-5, 1e-4],
                              domain_loss=[1.3, 1.1, 1.15, 1.25, 1.1],
                              general_loss=[2.5, 2.6, 2.55, 2.52, 2.6],
                              book_loss=[2.0, 2.2, 2.1, 2.05, 2.2],
                              suite=[0.6, 0.55, 0.58, 0.6, 0.55],
                              gsm8k_completion=[0.4] * 5,
                              gsm8k_chat=[0.35] * 5)).reindex(
        columns=d2l._CPT_COLUMNS['table'])
    return d2l.CptRuns(table, pd.DataFrame(columns=d2l._CPT_COLUMNS['logs']),
                       base, docs, None, None, pd.DataFrame(rows), '')


class PageCellsTest(unittest.TestCase):
    """The page's T2 cells on a synthetic result: shapes and widths."""
    def test_arms_and_gsm8k(self):
        runs = synthetic_runs()
        ns = page_ns()
        with mock.patch.object(d2l, 'cpt_arms', lambda *a, **k: runs), \
                mock.patch.object(d2l, 'plot_cpt_curves'):
            out = run_cell('continued-pretraining-arms', ns).splitlines()
        self.assertEqual(len(out), 6)
        self.assertEqual(out[0].split(), ['arm', 'trace', 'loss', 'general',
                                          'loss', 'book', 'loss', 'suite',
                                          'mean', 'GSM8K', 'chat'])
        self.assertLessEqual(max(map(len, out)), 80)
        out = run_cell('continued-pretraining-gsm8k', ns).splitlines()
        self.assertEqual(len(out), 12)
        self.assertLessEqual(max(map(len, out)), 80)
        self.assertEqual(out[1].split()[0], 'base')
        self.assertEqual(out[6], 'change against the base, points '
                                 '[95% CI, paired items]')
        self.assertRegex(out[8], r'^ *A +[-+]\d+\.\d \[[-+]\d+\.\d, '
                                 r'[-+]\d+\.\d\] +[-+]')

    def test_frontier_cells(self):
        ns = dict(page_ns(), runs=synthetic_runs())
        with mock.patch.object(d2l, 'plot_tradeoff'):
            out = run_cell('continued-pretraining-frontier', ns).splitlines()
        self.assertEqual(len(out), 7)
        out = run_cell('continued-pretraining-frontier-chord',
                       ns).splitlines()
        self.assertEqual(len(out), 2)
        self.assertRegex(out[0], r"^B − no-replay reference \(C → A chord\):"
                                 r" \+0\.0\d\d \[")
        self.assertTrue(out[1].startswith('seed spread (A vs A′): gain '))
        self.assertLessEqual(max(map(len, out)), 80)

    def test_t1_cell(self):
        arms = ['base', 'A', 'A′', 'A″', 'B', 'cold', 'cold-plateau',
                'branch']
        table = pd.DataFrame(dict(arm=arms, domain_loss=1.0,
                                  general_loss=3.0, book_loss=2.5)).reindex(
            columns=d2l._CPT_COLUMNS['table'])
        logs = pd.DataFrame(dict(arm=np.repeat(arms[1:], 3),
                                 step=np.tile([0, 3, 40], 7),
                                 general_loss=np.tile([3.0, 3.2, 3.0], 7)))
        logs.loc[(logs['arm'] == 'cold-plateau') & (logs['step'] == 3),
                 'general_loss'] = 3.5
        runs = d2l.CptRuns(table, logs, None, None, None, None, None, '')
        with mock.patch.object(d2l, 'cpt_arms', lambda *a, **k: runs):
            out = run_cell('continued-pretraining-t1-arms',
                           page_ns()).splitlines()
        self.assertEqual(len(out), 14)
        self.assertEqual(out[0].split(), ['arm', 'bump', 'trace', 'general',
                                          'book'])
        self.assertIn('cold-plateau', out[7])
        self.assertEqual(out[9], 'largest general-loss gap over steps 1-40 '
                                 '(nats/token)')
        self.assertEqual([r.split()[-2:] for r in out[11:]],
                         [['0.300', '3'], ['0.000', '3'], ['0.000', '3']])
        self.assertLessEqual(max(map(len, out)), 80)


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


def conversation(question, answer, closed=True):
    reply = f'<think> {answer} </think> done' if closed else \
        f'<think> {answer}'
    return [dict(role='user', content=question),
            dict(role='assistant', content=reply)]


def words_ending(near_cap):
    """The fixture's ending(record, ids): the assistant turn's words."""
    return lambda rec, ids: builder.trace_ending(
        rec, len(rec['messages'][1]['content'].split()), near_cap)


def words(rng, n):
    return ' '.join(f'w{x}' for x in rng.integers(0, 10 ** 6, n))


def word_ids(records):
    """A stand-in tokenizer: one id per word of the ChatML text, never 0
    (the fixture's EOS)."""
    return [np.array([zlib.crc32(w.encode()) % 50_000 + 1 for w in
                      d2l.chatml(r['messages']).split()], dtype=np.int64)
            for r in records]


class TracesCorpusTest(unittest.TestCase):
    """The trace corpus on 30 conversations: 12 questions with one to
    three annotations (24 conversations, padded to 29 with single ones),
    one question with a GSM8K test 13-gram, one with a special-token
    string, one exact duplicate, and one held-out candidate that copies a
    training document's reasoning."""
    GSM = ('a baker sells twelve loaves every morning and four more in the '
           'evening how many loaves')

    def fixture(self):
        rng = np.random.default_rng(0)
        convs, k = [], 0
        for q in range(12):
            question = words(rng, 30)
            if q == 2:
                question += ' ' + self.GSM
            for a in range(1 + q % 3):
                text = question + (' <|im_end|>' if q == 5 else '')
                convs.append((q, conversation(
                    text, words(rng, 60 + 20 * (q % 4 == 3)),
                    closed=(q + a) % 3 != 1)))
        while len(convs) < 29:
            k += 1
            convs.append((100 + k, conversation(words(rng, 30),
                                                words(rng, 60),
                                                closed=k % 2 == 0)))
        convs.append(convs[0])  # an exact duplicate
        grams = {'gsm8k': builder.word_ngrams(self.GSM, 13),
                 'math500': set()}
        records = []
        for i, (q, msgs) in enumerate(convs):
            r = builder.trace_record(d2l, msgs, grams, 13, ['<|im_end|>'])
            r.update(file=i % 3, row=i, domain='math' if q % 2 else 'code',
                     source='s', difficulty=1, q=q)
            records.append(r)
        return records

    def select(self, docs, **kw):
        args = dict(train_tokens=600, n_heldout=4, gram=8, max_overlap=0.2,
                    batch=5)
        args.update(kw)
        return builder.select_traces(d2l, docs, word_ids, **args)

    def test_ids_and_order(self):
        records = self.fixture()
        self.assertEqual(len(records), 30)
        r = records[0]
        self.assertEqual(r['doc_id'], __import__('hashlib').sha1(
            d2l.chatml(r['messages']).encode()).hexdigest())
        docs, dups = builder.dedupe_sort(records)
        self.assertEqual((len(docs), dups), (29, 1))
        shuffled = [dict(x, file=2 - x['file']) for x in reversed(records)]
        again, _ = builder.dedupe_sort(shuffled)
        self.assertEqual([d['doc_id'] for d in docs],
                         [d['doc_id'] for d in again])
        a, b = self.select(docs), self.select(again)
        for part in ('train', 'heldout'):
            self.assertEqual([x['doc_id'] for x, _ in a[part]],
                             [x['doc_id'] for x, _ in b[part]])
        self.assertEqual(builder.question_id('What  is\n2+2?'),
                         builder.question_id('what is 2+2?'))

    def test_split_rules(self):
        docs, _ = builder.dedupe_sort(self.fixture())
        flagged = {d['doc_id'] for d in docs if d['contam'] or d['special']}
        self.assertEqual(sum(d['q'] == 2 for d in docs if d['contam']), 3)
        self.assertEqual(sum(1 for d in docs if d['special']), 3)
        sel = self.select(docs)
        train = [r for r, _ in sel['train']]
        held = [r for r, _ in sel['heldout']]
        usable = [d for d in docs if d['doc_id'] not in flagged]
        self.assertEqual([r['doc_id'] for r in train],
                         [d['doc_id'] for d in usable[:len(train)]])
        n = sum(len(i) + 1 for _, i in sel['train'])
        self.assertGreaterEqual(n, 600)
        self.assertLess(n - len(sel['train'][-1][1]) - 1, 600)
        self.assertFalse({r['doc_id'] for r in train + held} & flagged)
        self.assertFalse({r['question_id'] for r in train} &
                         {r['question_id'] for r in held})
        self.assertEqual(len({r['question_id'] for r in held}), 4)
        self.assertTrue(all(h <= 0.2 * w for h, w in sel['overlap']))
        val = builder.prefix_docs(sel['heldout'], 150)
        self.assertGreaterEqual(sum(len(i) + 1 for _, i in val), 150)
        self.assertLess(sum(len(i) + 1 for _, i in val[:-1]), 150)

    def test_overlap_rule(self):
        docs, _ = builder.dedupe_sort(self.fixture())
        sel = self.select(docs)
        first = sel['heldout'][0][0]
        donor = sel['train'][0][0]
        copy = dict(first, messages=[first['messages'][0],
                                     donor['messages'][1]])
        docs2 = [copy if d['doc_id'] == first['doc_id'] else d for d in docs]
        sel2 = self.select(docs2)
        self.assertEqual(sel2['dropped'], [first['doc_id']])
        self.assertNotIn(first['question_id'],
                         {r['question_id'] for r, _ in sel2['heldout']})
        self.assertEqual(len(sel2['heldout']), 4)
        with self.assertRaises(RuntimeError):
            self.select(docs, n_heldout=40)

    def test_check_invariants(self):
        docs, dups = builder.dedupe_sort(self.fixture())
        sel = self.select(docs)
        targets = dict(builder.TRACES, train=600, heldout=4, val=150,
                       seq_len=4, batch=2, steps=1, gram=8,
                       overlap_target=0.5)
        split = builder.split_record(d2l, docs, sel, dups, targets,
                                     words_ending(70))
        self.assertEqual(split['counts']['documents'], 29)
        for part in ('train', 'heldout'):
            e = split['counts'][part]['endings']
            self.assertEqual(sum(v['documents'] for v in e.values()),
                             split['counts'][part]['documents'])
            self.assertEqual(sum(v['tokens'] for v in e.values()),
                             split['counts'][part]['tokens'])
        self.assertEqual(split['counts']['contaminated'],
                         {'gsm8k': 3, 'math500': 0})
        with tempfile.TemporaryDirectory() as root:
            shards = {
                'traces': builder.write_traces(
                    d2l, sel['train'], os.path.join(root, 'traces'), 60_000,
                    0),
                'val_domain': builder.write_traces(
                    d2l, builder.prefix_docs(sel['heldout'], 150),
                    os.path.join(root, 'val_domain'), 60_000, 0)}
            check = lambda sp=split, t=targets: [
                text for good, text in builder.check_trace_shards(
                    d2l, sp, shards, t, 't2') if not good]
            self.assertEqual(check(), [])
            s = shards['traces']
            self.assertTrue(np.array_equal(s.doc(0)[:-1],
                                           sel['train'][0][1]))
            self.assertEqual(int(s.doc(0)[-1]), 0)
            self.assertEqual(len(check(dict(split, files=split['files'][:-1]))),
                             1)
            self.assertEqual(len(check(t=dict(targets, heldout=5))), 1)
            self.assertEqual(len(check(t=dict(targets, train=10 ** 6))), 1)
            self.assertEqual(len(check(t=dict(targets,
                                              endings=('complete',)))), 1)
            t1 = dict(targets, t1_train=600, t1_val=150, t1_seq_len=4,
                      t1_batch=2, t1_steps=1)
            t1_check = lambda t: [text for good, text in
                                  builder.check_trace_shards(
                                      d2l, split, shards, t, 't1')
                                  if not good]
            self.assertEqual(t1_check(t1), [])
            self.assertEqual(len(t1_check(dict(t1, t1_steps=10 ** 6))), 1)
            path = os.path.join(root, 'traces',
                                s.shards[0]['file'] + '.bin')
            with open(path, 'r+b') as f:
                f.seek(8)
                b = f.read(1)
                f.seek(8)
                f.write(bytes([b[0] ^ 1]))
            self.assertTrue(any('SHA-256' in t for t in check()))


    def test_endings(self):
        docs, _ = builder.dedupe_sort(self.fixture())
        ending = words_ending(70)
        kinds = {ending(d, None) for d in docs}
        self.assertEqual(kinds, set(builder.ENDINGS))
        sel = self.select(docs, train_tokens=300, n_heldout=2,
                          keep=lambda r, i: ending(r, i) == 'complete')
        kept = [r for r, _ in sel['train'] + sel['heldout']]
        self.assertTrue(kept and all(ending(r, None) == 'complete'
                                     for r in kept))
        usable = [d for d in docs if not d['contam'] and not d['special']
                  and ending(d, None) == 'complete']
        self.assertEqual([r['doc_id'] for r, _ in sel['train']],
                         [d['doc_id'] for d in usable[:len(sel['train'])]])
        count = builder.assistant_tokens(7)
        self.assertEqual(count(np.array([7, 1, 1, 7, 1, 1, 1])), 3)
        self.assertEqual(count(np.array([1, 1])), 2)
        rec = dict(messages=conversation('q', 'a'))
        self.assertEqual(builder.trace_ending(rec, 10, 11), 'complete')
        self.assertEqual(builder.trace_ending(rec, 11, 11), 'capped')
        self.assertEqual(builder.trace_ending(
            dict(messages=conversation('q', 'a', closed=False)), 1, 11),
            'open')

    def test_licence_reread(self):
        _, repo, rev, licence = d2l.PRETRAIN_SOURCES['openthoughts3']
        card = lambda lic, sha=rev: mock.patch(
            'huggingface_hub.HfApi', lambda: mock.Mock(dataset_info=lambda
                *a, **k: mock.Mock(sha=sha, card_data={'license': lic})))
        with card(licence):
            self.assertEqual(builder.trace_licence(d2l), licence)
        for bad in (card('cc-by-nc-4.0'), card(licence, sha='0' * 40)):
            with bad, self.assertRaises(SystemExit):
                builder.trace_licence(d2l)
        targets = dict(near_cap=builder.TRACES['near_cap'],
                       endings=list(builder.TRACES['endings']))
        with tempfile.TemporaryDirectory() as root:
            def write(**kw):
                with open(os.path.join(root, 'split.json'), 'w') as f:
                    json.dump(dict(dict(licence=licence, targets=targets),
                                   **kw), f)
            write()
            with card(licence):  # the re-read path also reads the card
                split, sel = builder.traces_split(d2l, None, root)
            self.assertIsNone(sel)
            with card('cc-by-nc-4.0'), self.assertRaises(SystemExit):
                builder.traces_split(d2l, None, root)
            write(targets=dict(targets, endings=['complete']))
            with card(licence), self.assertRaises(SystemExit):
                builder.traces_split(d2l, None, root)


class FakeTokenizer:
    """ChatML markers as ids 1 and 2, every other character its code."""
    unk_token_id = 0

    def __init__(self, marks=True):
        self.marks = marks

    def convert_tokens_to_ids(self, toks):
        return [1, 2] if self.marks else [0, 0]

    def __call__(self, texts, add_special_tokens=False, verbose=False):
        out = []
        for t in texts:
            ids = []
            for part in re.split(r'(<\|im_start\|>|<\|im_end\|>)', t):
                ids += ([1] if part == '<|im_start|>' else [2]
                        if part == '<|im_end|>' else [ord(c) for c in part])
            out.append(ids)
        return dict(input_ids=out)


class TraceWindowsTest(unittest.TestCase):
    def test_chatml(self):
        self.assertEqual(d2l.chatml(conversation('q?', 'a')),
                         '<|im_start|>user\nq?<|im_end|>\n<|im_start|>'
                         'assistant\n<think> a </think> done<|im_end|>\n')

    def test_windows(self):
        conv = pd.DataFrame(dict(doc_id=['a', 'b', 'c'], messages=[
            conversation('x' * 50, 'y' * 3000), conversation('q', 'z'),
            conversation('r' * 10, 's' * 100)]))
        with mock.patch.object(d2l, 'trace_conversations', lambda: conv):
            w = d2l.trace_windows(FakeTokenizer(), T=64)
            short = len(FakeTokenizer()([d2l.chatml(conv['messages'][1])])[
                'input_ids'][0])
            self.assertEqual(len(d2l.trace_windows(FakeTokenizer())[1]),
                             short)
            self.assertEqual([len(x) for x in w], [64, 45, 64])
            self.assertEqual(int(w[0][0]), 1)
            self.assertEqual(len(d2l.trace_windows(FakeTokenizer(), n=2)), 2)
            with self.assertRaises(ValueError):
                d2l.trace_windows(FakeTokenizer(marks=False))
        self.assertEqual(builder.TRACES['heldout'], 500)
        import inspect
        self.assertEqual(inspect.signature(d2l.trace_windows).parameters[
            'T'].default, 2048)

    def test_conversations_identity(self):
        rows = [dict(doc_id=i, question_id=i, domain='math', source='s',
                     messages=conversation('q', 'a')) for i in ('a', 'b')]
        with tempfile.TemporaryDirectory() as root:
            import gzip
            with gzip.open(os.path.join(root, 'heldout.jsonl.gz'),
                           'wt') as f:
                f.writelines(json.dumps(r) + '\n' for r in rows)
            ok = __import__('hashlib').sha1(b'a b').hexdigest()
            man = lambda h: mock.patch.object(
                d2l, 'load_artifact', lambda name: mock.Mock(
                    ids_sha1=dict(heldout=h)))
            with mock.patch.dict(d2l._CPT_DIRS, t2=root), mock.patch.object(
                    d2l, '_cpt_manifest_committed', lambda: True):
                with man(ok):
                    self.assertEqual(list(d2l.trace_conversations()[
                        'doc_id']), ['a', 'b'])
                with man('0' * 40), self.assertRaises(ValueError):
                    d2l.trace_conversations()

    @unittest.skipUnless(os.path.exists(os.path.join(
        ROOT, 'data', 'lm', 'cpt-qwen-traces-v1', 'heldout.jsonl.gz')),
        'the trace corpus is not built here')
    def test_built_heldout(self):
        cwd = os.getcwd()
        os.chdir(os.path.join(ROOT, 'chapter_lm-pretraining'))
        try:
            conv = d2l.trace_conversations()
            self.assertEqual(len(conv), 500)
            self.assertEqual(conv['question_id'].nunique(), 500)
            self.assertEqual(list(conv['doc_id']), sorted(conv['doc_id']))
            w = d2l.trace_windows(d2l._cpt_tokenizer('t2'), n=4)
            self.assertEqual([len(x) for x in w], [2048] * 4)
            self.assertTrue(all(int(x[0]) == 151644 for x in w))
        finally:
            os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
