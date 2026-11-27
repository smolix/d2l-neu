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
widths); `rewarm_bump` (from each arm's own step 0, transient or not, a flat
top no transient);
`early_gaps` (paired gaps and the seed yardstick over steps 1-40);
`book_split`
(deterministic, disjoint, every fifth section held out); `cpt_data('t2')`
in a normal run opens no shard; the arm schedule, configurations and
evaluation grid; `_doc_nll` against a direct loss; a toy two-arm run of
`cpt_arms` end to end (job folders, equal-token and suite files, the
merge, the stored json-table read back, the step-0 row), a job folder of
another arm refused, and a resumed arm equal to an uninterrupted one; the
page's T2 cells on a pending result and the fallback's own label kept
(and its configuration checked when given); the
SmolLM3 trajectory (F25): the merge (token counts, byte-weighted bits per
byte, GSM8K accuracy and cut share, no reply text, a conditional rung's
answers), the rungs only once they are regeneration names,
`stage_deltas` (planted cloze, GSM8K and bits-per-byte changes; the
MATH-500 line) and the page's four cells (shapes, widths, pending), the
calibration table (best temperature, the share of the rise recovered), the
cloze scores copied from version 1, the live call; bits per byte with no
prefix token (`_scored_doc` on a hand count, `_doc_stats` against
`bits_per_byte` without a boundary token and a full forward,
`_logit_stats` on fixed logits); MATH-500's numeric filter and answers,
the Minerva prompt, GSM-Symbolic items, the chat template of the
mid-training tokenizer, `score_gsm8k` under SmolLM3's end token and with
kind='math'; the regeneration jobs against the page's literals; the reasoning-trace corpus of
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
        cfg = dict(seq_len=16, global_batch=128, data_seed=0, replay=0.25,
                   steps=4)
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

    def test_cpt_mix_seen_once(self):
        """An arm whose steps would wrap a stream is refused: the last step
        that fits passes, one more step is refused; replay at 0 never
        wraps the replay stream."""
        cfg = dict(seq_len=16, global_batch=128, data_seed=0, replay=0.0)
        n = d2l.PackedStream(self.shards['traces'], 16, 128).n_seq
        d2l._cpt_mix(self.shards, dict(cfg, steps=n // 128))
        with self.assertRaisesRegex(ValueError, 'traces: .* once'):
            d2l._cpt_mix(self.shards, dict(cfg, steps=n // 128 + 1))
        m = d2l.PackedStream(self.shards['replay'], 16, 128).n_seq
        mix = d2l._cpt_mix(self.shards, dict(cfg, replay=1.0,
                                             steps=m // 128))
        self.assertEqual(int(mix.cursors(m // 128)[0]), 0)
        with self.assertRaisesRegex(ValueError, 'replay: .* once'):
            d2l._cpt_mix(self.shards, dict(cfg, replay=1.0,
                                           steps=m // 128 + 1))

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

    def test_flat_top(self):
        """A peak the loss leaves by no more than tol inside the window is
        no transient (F24 review 1: a flat top, forgetting, not a bump);
        a fall of more than tol is one, even if the end lies above it."""
        steps = [0, 3, 6, 9, 12, 45]
        L = {'flat': [3.0, 3.1, 3.2, 3.24, 3.239, 3.24],
             'drop': [3.0, 3.1, 3.3, 3.25, 3.28, 3.35]}
        logs = pd.DataFrame([dict(arm=a, step=t, general_loss=v)
                             for a, vs in L.items()
                             for t, v in zip(steps, vs)])
        runs = d2l.CptRuns(None, logs, None, None, None, None, None, '')
        b = d2l.rewarm_bump(runs).set_index('arm')
        self.assertTrue(math.isnan(b.bump['flat']))
        self.assertEqual(b.step['flat'], 9)
        self.assertAlmostEqual(b.bump['drop'], 0.3)
        self.assertAlmostEqual(b.healed['drop'], -0.05)  # ends above
        self.assertAlmostEqual(d2l.rewarm_bump(runs, tol=1e-4).set_index(
            'arm').bump['flat'], 0.24)


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
        # T1 measures the cold arms' first updates (F24 review 2)
        self.assertEqual(d2l._cpt_eval_steps(183, 't1')[:4], [1, 2, 3, 5])
        self.assertEqual(d2l._cpt_eval_steps(1, 't1'), [1])


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
        self.assertEqual(list(graded.step),
                         [0, 1, 2, 3, 5, 6, 9, 10, 12, 15, 16])
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
            with mock.patch.object(d2l, 'cpt_qwen_runs', lambda: runs):
                outs = {c: run_cell(f'continued-pretraining-{c}', ns)
                        for c in ('opening', 'gsm8k', 'bump-table',
                                  'frontier', 'frontier-chord',
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
        self.assertEqual(len(outs['opening'].splitlines()), 3)
        self.assertRegex(outs['opening'].splitlines()[0],
                         r'^after \d+\.\dM trace tokens \(arm A\): '
                         r'held-out trace loss \d\.\d\d → \d\.\d\d '
                         r'nats/token$')
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

    def test_fallback_config_checked(self):
        """A fallback given with its configuration must have been run with
        it (F24 review 4)."""
        config = dict(arms={'A': dict(peak_lr=3e-4)}, base='r2')
        t1 = d2l.CptRuns(pd.DataFrame(), pd.DataFrame(), None, None, None,
                         None, None, 't1 preset')
        t1.table.attrs['config'] = config
        with mock.patch.object(d2l, 'load_artifact', lambda name: t1):
            res = d2l._pending_result('cpt-qwen-evals', ('cpt-t1', config))
            self.assertIn('T1 configuration', res.label)
            with self.assertRaisesRegex(ValueError, 'arms.A.peak_lr'):
                d2l._pending_result('cpt-qwen-evals', ('cpt-t1', dict(
                    config, arms={'A': dict(peak_lr=1e-4)})))


def smollm3_points(n_items=300, n_docs=60, seed=0, gain=None, rise=None,
                   gsm=None):
    """Synthetic smollm3-k parts: per-item cloze scores, per-document
    calibration sums and GSM8K answers. gain(i) flips cloze items to
    right from checkpoint i on; rise(i, dom) multiplies a domain's NLL;
    gsm(i, fmt) is a checkpoint's GSM8K accuracy (items 0.. in order)."""
    names = list(d2l._smollm3_call()['names'])
    rng = np.random.default_rng(seed)
    base = rng.random((len(d2l.SMOLLM3_TASKS), n_items)) < 0.6
    flip = rng.random(base.shape) < 0.25
    nbytes = rng.integers(500, 5000, n_docs)
    ntok = (nbytes / 4).astype(int)
    per_tok = 1.5 + 0.2 * np.sin(np.arange(n_docs))
    temps = np.asarray(d2l._SMOLLM3_TEMPS)
    pts = {}
    for i, n in enumerate(names):
        label = d2l._SMOLLM3[n][0]
        score = base | flip if gain and gain(i) else base
        items = pd.DataFrame([dict(task=t, item_id=j,
                                   score=float(score[k, j]),
                                   checkpoint=label)
                              for k, t in enumerate(d2l.SMOLLM3_TASKS)
                              for j in range(n_items)])
        rows = []
        for dom in d2l._SMOLLM3_DOMAINS:
            f = rise(i, dom) if rise else 1.0
            for k in range(n_docs):
                nll = per_tok[k] * f * ntok[k]
                row = dict(domain=dom, doc=k, bytes=int(nbytes[k]))
                # the T = 1 loss is the grid's best for a factor of 1
                row.update({f'nll_t{t:g}': nll * (1 + 0.1 * (t - 1) ** 2)
                            for t in temps})
                row.update(entropy=nll, top1=0.5 * ntok[k],
                           tokens=int(ntok[k]),
                           bpb=nll / (math.log(2) * nbytes[k]),
                           checkpoint=label)
                rows.append(row)
        answers = []
        for fmt in ('completion', 'chat'):
            p = gsm(i, fmt) if gsm else 0.5
            answers += [dict(format=fmt, item_id=str(j),
                             score=float(j < p * 500), forced=float(
                                 j < p * 500), cut=fmt == 'chat' and j >= 480,
                             thinks=fmt == 'chat', closed=fmt == 'chat',
                             chat_tokens=False, new_tokens=10, task='gsm8k',
                             checkpoint=label, text='x', pred='1',
                             forced_pred='1', forced_text=None)
                        for j in range(500)]
        pts[f'smollm3-{i}'] = dict(items=items, docs=pd.DataFrame(rows),
                                   answers=pd.DataFrame(answers),
                                   ctx=65536 if n in ('lc64k', 'mid')
                                   else 4096)
    return pts


def with_math_rung(pts):
    """pts with the MATH-500 rung's three parts (stage 3, LC 64K and
    mid-training; their GSM8K answers relabelled)."""
    pts = dict(pts)
    for k in (2, 4, 5):
        pts[f'smollm3-math-{k}'] = dict(answers=pts[f'smollm3-{k}'][
            'answers'].assign(task='math500'))
    return pts


class ByteTokenizer:
    """UTF-8 bytes as token ids (no boundary token), with the two calls
    _scored_doc makes and the encode/decode bits_per_byte makes."""
    def __call__(self, text, add_special_tokens=False):
        return dict(input_ids=list(text.encode('utf-8')))

    def encode(self, text):
        return list(text.encode('utf-8'))

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return bytes(ids).decode('utf-8', errors='replace')


def tiny_llama(vocab=256, seed=0):
    import transformers
    cfg = transformers.LlamaConfig(
        vocab_size=vocab, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True)
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(cfg).eval()


class SmolLM3Test(unittest.TestCase):
    def test_merge(self):
        pts = smollm3_points(n_items=4, n_docs=2)
        names = list(d2l._smollm3_call()['names'])
        traj = d2l._smollm3_merge(pts, names)
        self.assertEqual(list(traj.table['tokens']),
                         ['8.1T', '9.9T', '11.1T', '11.2T', '11.2T', '11.4T'])
        self.assertEqual(list(traj.table['ctx'])[-2:], ['64K', '64K'])
        self.assertEqual(list(traj.table.columns), [
            'checkpoint', 'tokens', 'ctx', 'cloze', '8-shot', 'chat', 'cut',
            'web', 'science', 'book', 'traces'])
        self.assertEqual(list(traj.bpb.columns),
                         ['checkpoint', *d2l._SMOLLM3_DOMAINS])
        # byte-weighted bits per byte over the documents
        d = pts['smollm3-0']['docs']
        web = d[d['domain'] == 'web']
        self.assertAlmostEqual(traj.bpb['web'].iloc[0], (
            web['bpb'] * web['bytes']).sum() / web['bytes'].sum())
        self.assertAlmostEqual(traj.table['chat'].iloc[0], 0.5)
        self.assertAlmostEqual(traj.table['cut'].iloc[0], 0.04)
        self.assertNotIn('text', traj.answers.columns)  # no reply text
        self.assertEqual(list(traj.answers.columns),
                         d2l._smollm3_answer_columns())

    def test_merge_with_a_rung(self):
        """A conditional rung's answers join the merge (claim 10h)."""
        pts = smollm3_points(n_items=4, n_docs=2)
        rung = pts['smollm3-4']['answers'].assign(task='math500')
        pts['smollm3-math-4'] = dict(answers=rung)
        traj = d2l._smollm3_merge(pts, list(d2l._smollm3_call()['names']))
        self.assertEqual(set(traj.answers['task']), {'gsm8k', 'math500'})
        self.assertAlmostEqual(traj.table['8-shot'].iloc[4], 0.5)

    def test_rungs_only_when_named(self):
        """The conditional rungs are parts of the merge only once their
        names are regeneration names (the gate ruled them in, F26)."""
        seen = {}

        def merged(name, parts, merge, config, **kw):
            seen['parts'] = sorted(parts)
            seen['config'], seen['part'] = config, parts['smollm3-0'][1]
            return d2l._pending_result('smollm3-trajectory')
        names = list(d2l._smollm3_call()['names'])
        with mock.patch.object(d2l, '_pretrain_merged', merged), \
                contextlib.redirect_stdout(io.StringIO()):
            d2l.smollm3_trajectory(names)
            self.assertEqual(seen['parts'],
                             sorted(f'smollm3-{k}' for k in range(6)))
            # the merged artifact records every setting it was made with
            # (F25 review 5), each part its scoring rules (review 3)
            cfg = seen['config']
            self.assertEqual(sorted(cfg), ['cloze', 'ctx', 'gsm8k', 'names',
                                           'rungs', 'scorer', 'temps'])
            self.assertEqual((cfg['names'], cfg['rungs'], cfg['scorer']),
                             (names, {}, d2l._BPB_SCORER))
            self.assertEqual(cfg['gsm8k'], d2l._smollm3_settings('gsm8k'))
            self.assertEqual(seen['part']['scorer'], d2l._BPB_SCORER)
            self.assertEqual(seen['part']['cloze'], d2l._SMOLLM3_CLOZE)
            with mock.patch.object(d2l, 'PRETRAIN_RERUN_NAMES',
                                   d2l.PRETRAIN_RERUN_NAMES + (
                                       'smollm3-math-5', 'smollm3-symbolic')):
                traj = d2l.smollm3_trajectory(names)
        self.assertEqual(seen['parts'], sorted(
            [f'smollm3-{k}' for k in range(6)] + ['smollm3-math-5',
                                                 'smollm3-symbolic']))
        self.assertEqual(sorted(seen['config']['rungs']),
                         ['smollm3-math-5', 'smollm3-symbolic'])
        self.assertEqual(seen['config']['rungs']['smollm3-math-5'],
                         d2l._smollm3_settings('math500'))
        self.assertEqual(list(traj.answers.columns),
                         d2l._smollm3_answer_columns())
        self.assertEqual(d2l._SMOLLM3_RUNGS['smollm3-math-2'],
                         ('math500', ('stage3',)))

    def test_stage_deltas(self):
        """Paired changes between adjacent checkpoints: a planted cloze
        gain, a +20% bits-per-byte rise on one domain and a +10-point
        GSM8K chat gain are recovered, the rest is 0 within the interval;
        the page's cell fits 13 lines of at most 80 columns."""
        pts = smollm3_points(gain=lambda i: i >= 2,
                             rise=lambda i, dom: 1.2 if dom == 'book'
                             and i >= 3 else 1.0,
                             gsm=lambda i, fmt: 0.6 if fmt == 'chat'
                             and i == 5 else 0.5)
        traj = d2l._smollm3_merge(pts, list(d2l._smollm3_call()['names']))
        deltas = d2l.stage_deltas(traj).set_index('transition')
        self.assertEqual(list(deltas.index), ['S1 → S2', 'S2 → S3',
                                              'S3 → LC32K', 'LC32K → LC64K',
                                              'LC64K → mid'])
        self.assertGreater(deltas.cloze_lo['S2 → S3'], 0)
        self.assertEqual(deltas.cloze['S1 → S2'], 0)
        self.assertAlmostEqual(deltas.book['S3 → LC32K'], 20, places=6)
        self.assertGreater(deltas.book_lo['S3 → LC32K'], 19.9)
        self.assertAlmostEqual(deltas.web['S3 → LC32K'], 0)
        self.assertAlmostEqual(deltas.gsm8k_chat['LC64K → mid'], 10,
                               places=6)
        self.assertGreater(deltas.gsm8k_chat_lo['LC64K → mid'], 0)
        self.assertEqual(deltas.gsm8k_completion['LC64K → mid'], 0)
        self.assertNotIn('math500_chat', deltas.columns)
        out = run_cell('continued-pretraining-stage-deltas',
                       dict(page_ns(), traj=traj))
        lines = out.splitlines()
        self.assertEqual(len(lines), 13)
        self.assertLessEqual(max(map(len, lines)), 80)
        self.assertTrue(lines[0].startswith('Δ points      cloze [95% CI]'))
        self.assertIn('+10.0 [+', lines[5])
        self.assertEqual(lines[6].split(), ['Δ', 'BPB,', '%', 'web',
                                            'science', 'book', 'traces'])
        self.assertEqual(lines[9].split()[-2], '+20.0*')
        # the MATH-500 rung (all three checkpoints) adds its own header and
        # S3 → LC64K, LC64K → mid
        traj = d2l._smollm3_merge(with_math_rung(pts),
                                  list(d2l._smollm3_call()['names']))
        lines = run_cell('continued-pretraining-stage-deltas',
                         dict(page_ns(), traj=traj)).splitlines()
        self.assertEqual(len(lines), 16)
        self.assertLessEqual(max(map(len, lines)), 80)
        self.assertEqual(lines[6].split(), ['Δ', 'points', 'MATH-500',
                                            '8-shot', 'MATH-500', 'chat'])
        self.assertTrue(lines[7].startswith('S3 → LC64K' + ' ' * 25))
        self.assertTrue(lines[8].startswith('LC64K → mid' + ' ' * 24))
        # the MATH columns sit under GSM8K's 8-shot and chat columns
        self.assertEqual(lines[6].index('MATH-500 8-shot'),
                         lines[0].index('GSM8K 8-shot'))
        self.assertIn('+10.0 [+', lines[8])  # mid's planted chat gain
        self.assertTrue(lines[9].startswith('Δ BPB, %'))

    def test_answer_deltas_and_table(self):
        """The MATH-500 rung on stage 3, LC 64K and mid-training: per
        checkpoint n, both accuracies and the cut share; paired changes
        between consecutive rung checkpoints; explicit pairs."""
        pts = with_math_rung(smollm3_points(
            n_items=4, n_docs=2, gsm=lambda i, fmt: 0.6 if fmt == 'chat'
            and i == 5 else 0.5))
        traj = d2l._smollm3_merge(pts, list(d2l._smollm3_call()['names']))
        tab = d2l.answer_table(traj, 'math500')
        self.assertEqual(list(tab['checkpoint']),
                         ['stage 3 end', 'LC 64K', 'mid-training'])
        self.assertEqual(list(tab.columns),
                         ['checkpoint', 'n', '8-shot', 'chat', 'cut'])
        self.assertEqual(list(tab['n']), [500] * 3)
        self.assertAlmostEqual(tab['chat'].iloc[2], 0.6)
        self.assertAlmostEqual(tab['cut'].iloc[0], 0.04)
        d = d2l.answer_deltas(traj, 'math500').set_index('transition')
        self.assertEqual(list(d.index), ['S3 → LC64K', 'LC64K → mid'])
        self.assertAlmostEqual(d.chat['LC64K → mid'], 10, places=6)
        self.assertEqual(d.completion['S3 → LC64K'], 0)
        e = d2l.answer_deltas(traj, 'math500', [('stage 3 end',
                                                 'mid-training'),
                                                ('LC 32K', 'mid-training')])
        self.assertEqual(list(e['transition']), ['S3 → mid', 'LC32K → mid'])
        self.assertTrue(np.isnan(e['chat'].iloc[1]))  # no LC 32K rung
        self.assertEqual(len(d2l.answer_table(traj, 'gsm-symbolic')), 0)
        self.assertEqual(len(d2l.answer_deltas(traj, 'gsm-symbolic')), 0)

    def test_symbolic_check(self):
        """Both chat gains of mid-training over LC 64K and their ratio."""
        pts = smollm3_points(n_items=4, n_docs=2, gsm=lambda i, fmt: 0.7
                             if fmt == 'chat' and i == 5 else 0.5)
        sym = {k: pts[f'smollm3-{k}']['answers'] for k in (4, 5)}
        pts['smollm3-symbolic'] = dict(answers=pd.concat([
            sym[4].assign(task='gsm-symbolic'),
            sym[5].assign(task='gsm-symbolic', score=lambda d: np.where(
                d['format'] == 'chat', (d['item_id'].astype(int) < 300)
                .astype(float), d['score']))]))
        traj = d2l._smollm3_merge(pts, list(d2l._smollm3_call()['names']))
        chk = d2l.symbolic_check(traj).set_index('task')
        self.assertAlmostEqual(chk.loc['gsm8k', 'gain'], 20, places=6)
        self.assertAlmostEqual(chk.loc['gsm-symbolic', 'gain'], 10, places=6)
        self.assertAlmostEqual(chk.loc['gsm-symbolic', 'ratio'], 0.5)
        self.assertGreater(chk.loc['gsm8k', 'lo'], 0)
        none = d2l.symbolic_check(d2l._smollm3_merge(
            smollm3_points(n_items=4, n_docs=2),
            list(d2l._smollm3_call()['names']))).set_index('task')
        self.assertTrue(np.isnan(none.loc['gsm-symbolic', 'gain']))

    def test_cloze_task_deltas(self):
        """Per transition and task, a paired change with its interval."""
        pts = smollm3_points(n_items=50, n_docs=2, gain=lambda i: i >= 2)
        traj = d2l._smollm3_merge(pts, list(d2l._smollm3_call()['names']))
        d = d2l.cloze_task_deltas(traj, n_boot=200)
        self.assertEqual(len(d), 5 * len(d2l.SMOLLM3_TASKS))
        s23 = d[d['transition'] == 'S2 → S3']
        self.assertTrue((s23['delta'] > 0).all())
        self.assertTrue((d[d['transition'] == 'S1 → S2']['delta'] == 0).all())
        self.assertTrue((s23['lo'] <= s23['delta']).all())

    def test_page_table_and_plot(self):
        traj = d2l._smollm3_merge(smollm3_points(n_items=20, n_docs=4),
                                  list(d2l._smollm3_call()['names']))
        ns = page_ns()
        with mock.patch.object(d2l, 'smollm3_trajectory',
                               lambda names: traj):
            lines = run_cell('continued-pretraining-smollm3',
                             ns).splitlines()
        self.assertEqual(len(lines), 7)
        self.assertLessEqual(max(map(len, lines)), 80)
        self.assertEqual(lines[0].split(), [
            'checkpoint', 'tokens', 'ctx', 'cloze', '8-shot', 'chat', 'cut',
            'web', 'science', 'book', 'traces'])
        self.assertRegex(lines[6], r'^mid-training +11\.4T +64K +\d+\.\d '
                                   r'+50 +50 +4 +\d\.\d{3}')
        d2l.plt.close('all')
        # with the MATH-500 rung: a second block of 4 lines, 11 in all, and
        # the plot's two extra series
        traj = d2l._smollm3_merge(with_math_rung(smollm3_points(
            n_items=20, n_docs=4)), list(d2l._smollm3_call()['names']))
        with mock.patch.object(d2l, 'smollm3_trajectory',
                               lambda names: traj):
            lines = run_cell('continued-pretraining-smollm3',
                             ns).splitlines()
        self.assertEqual(len(lines), 11)
        self.assertLessEqual(max(map(len, lines)), 80)
        self.assertEqual(lines[7].split(), ['MATH-500', '(n', '=', '500)',
                                            '8-shot', 'chat', 'cut'])
        self.assertEqual([r.split()[0] for r in lines[8:]],
                         ['stage', 'LC', 'mid-training'])
        self.assertEqual(lines[10].split()[-3:], ['50', '50', '4'])
        ax = d2l.plt.gcf().axes[0]
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()][
            -2:], ['MATH-500 8-shot', 'MATH-500 chat'])
        d2l.plt.close('all')

    def test_calibration(self):
        """Best temperature and the share of the rise it recovers, on
        planted sums: LC 64K calibrated (best T 1), mid-training's loss
        3.0 at T = 1 and 2.7 at T = 1.5 after LC 64K's 2.0."""
        temps = d2l._SMOLLM3_TEMPS
        curves = {'LC 64K': (2.2, 2.0, 2.1, 2.3, 2.6),
                  'mid-training': (3.6, 3.0, 2.8, 2.7, 2.9)}
        docs = pd.DataFrame([dict(checkpoint=c, domain='web', doc=k,
                                  bytes=400, tokens=100, top1=40 + k,
                                  entropy=100 * (2.0 if c == 'LC 64K'
                                                 else 1.5),
                                  **{f'nll_t{t:g}': 100 * v
                                     for t, v in zip(temps, curve)})
                             for c, curve in curves.items()
                             for k in range(2)])
        table = pd.DataFrame(dict(checkpoint=list(curves)))
        traj = d2l.SmolTrajectory(table, None, None, docs, None, '')
        calib = d2l.smollm3_calibration(traj).set_index('checkpoint')
        self.assertEqual(list(calib.columns), [
            'domain', 'loss', 'entropy', 'top-1', 'best T',
            'loss at best T', 'recovered'])
        self.assertEqual(calib.loc['LC 64K', 'best T'], 1.0)
        self.assertTrue(np.isnan(calib.loc['LC 64K', 'recovered']))
        self.assertEqual(calib.loc['mid-training', 'best T'], 1.5)
        self.assertAlmostEqual(calib.loc['mid-training', 'loss at best T'],
                               2.7)
        self.assertAlmostEqual(calib.loc['mid-training', 'recovered'], 0.3)
        self.assertAlmostEqual(calib.loc['mid-training', 'top-1'], 0.405)
        self.assertAlmostEqual(calib.loc['mid-training', 'entropy'], 1.5)

    def test_calibration_cell(self):
        traj = d2l._smollm3_merge(smollm3_points(n_items=4, n_docs=4,
                                                 rise=lambda i, dom: 1.5
                                                 if i == 5 else 1.0),
                                  list(d2l._smollm3_call()['names']))
        lines = run_cell('continued-pretraining-smollm3-calibration',
                         dict(page_ns(), traj=traj)).splitlines()
        self.assertEqual(len(lines), 7)
        self.assertLessEqual(max(map(len, lines)), 80)
        self.assertEqual(lines[0].split()[:4], ['checkpoint', 'domain',
                                                'loss', 'entropy'])
        self.assertEqual([r.split()[0] for r in lines[1:]],
                         ['LC'] * 3 + ['mid-training'] * 3)
        self.assertTrue(lines[1].rstrip().endswith('–'))
        d2l.plt.close('all')

    def test_live_cell(self):
        """Cell 17: four lines, the committed value next to the live one."""
        traj = d2l._smollm3_merge(smollm3_points(n_items=4, n_docs=2),
                                  list(d2l._smollm3_call()['names']))
        live = pd.DataFrame([dict(checkpoint=c, format=f, item_id=str(j),
                                  score=float(j < 20))
                             for c in ('LC 64K', 'mid-training')
                             for f in ('completion', 'chat')
                             for j in range(50)])
        with mock.patch.object(d2l, 'smollm3_gsm8k',
                               lambda names, n: live):
            lines = run_cell('continued-pretraining-smollm3-live',
                             dict(page_ns(), traj=traj)).splitlines()
        self.assertEqual(len(lines), 5)
        self.assertLessEqual(max(map(len, lines)), 80)
        self.assertEqual(lines[0].split()[:6], ['checkpoint', 'format',
                                                'live', '(n', '=', '50)'])
        self.assertTrue(lines[0].endswith('committed: same items  all 500'))
        # committed: 100% on the first 50 items (j < 250), 50% on all 500
        self.assertEqual(lines[1].split()[-2:], ['100%', '50%'])
        self.assertIn('8-shot', lines[1])
        self.assertRegex(lines[2],
                         r'^LC 64K +chat +40% \[\d\d, \d\d\] +100% +50%$')
        self.assertEqual(lines[0].index('committed'),
                         lines[2].index('100%'))

    def test_pending_cells(self):
        """The four SmolLM3 cells run on a pending result."""
        ns = page_ns()
        hub = {k: v for k, v in d2l.DATA_HUB.items()
               if k != d2l.PRETRAIN_ARTIFACTS['smollm3-trajectory'][0]}
        names = {f'smollm3-{k}' for k in range(6)}
        empty = pd.DataFrame(columns=['checkpoint', 'format', 'item_id',
                                      'score'])
        with mock.patch.object(d2l, 'PRETRAIN_PENDING', frozenset(names)), \
                mock.patch.dict(os.environ, {'D2L_RERUN': ''}), \
                mock.patch.dict(d2l.DATA_HUB, hub, clear=True), \
                mock.patch.object(d2l, 'smollm3_gsm8k',
                                  lambda names, n: empty), \
                warnings_as_errors():
            out = run_cell('continued-pretraining-smollm3', ns)
            self.assertEqual(out.splitlines()[0],
                             '[smollm3-trajectory pending]')
            for cell in ('stage-deltas', 'smollm3-calibration',
                         'smollm3-live'):
                run_cell(f'continued-pretraining-{cell}', ns)
        d2l.plt.close('all')

    def test_texts_identity(self):
        """The texts' identity: order, count, content and the NUL between
        documents count; NFC forms are equal (F25 review 3)."""
        tid = d2l._bpb_texts_id
        a = tid(dict(web=['ab', 'c']))
        self.assertEqual(a['web']['n'], 2)
        for other in (['ab', 'd'], ['a', 'bc'], ['c', 'ab'], ['ab']):
            self.assertNotEqual(tid(dict(web=other)), a, other)
        self.assertEqual(tid(dict(t=['e\u0301'])), tid(dict(t=['\u00e9'])))

    def test_merge_checks_texts(self):
        """Parts that read other texts are refused at the merge."""
        names = list(d2l._smollm3_call()['names'])
        ids = d2l._bpb_texts_id(dict(web=['x']))
        pts = smollm3_points(n_items=4, n_docs=2)
        for p in pts.values():
            p['texts'] = ids
        d2l._smollm3_merge(pts, names, ids)  # the same texts: merged
        with self.assertRaisesRegex(ValueError, 'read other texts'):
            d2l._smollm3_merge(pts, names, d2l._bpb_texts_id(dict(web=['y'])))
        pts['smollm3-3'] = dict(pts['smollm3-3'],
                                texts=d2l._bpb_texts_id(dict(web=['y'])))
        with self.assertRaisesRegex(ValueError, 'smollm3-3 read other'):
            d2l._smollm3_merge(pts, names)

    def test_point_cache_identity(self):
        """The bits-per-byte cache of a checkpoint carries its pin, the
        scoring version and the texts' identity, and the part returns the
        identity for the merge's check."""
        seen = {}

        def cached(name, what, ident, compute):
            seen[what] = ident
            return pd.DataFrame(dict(domain=['web'], doc=[0]))
        texts = dict(web=['some text'], traces=['<|im_start|>user'])
        cfg = mock.Mock(max_position_embeddings=65536)
        with mock.patch.object(d2l, '_smollm3_tokenizer', lambda n: n), \
                mock.patch.object(d2l, '_smollm3_cloze',
                                  lambda n: pd.DataFrame()), \
                mock.patch.object(d2l, '_smollm3_cached', cached), \
                mock.patch.object(d2l, '_smollm3_answers',
                                  lambda *a: pd.DataFrame()), \
                mock.patch('transformers.AutoConfig.from_pretrained',
                           lambda *a, **k: cfg), \
                mock.patch.object(d2l, 'lm_free', lambda: None):
            part = d2l._smollm3_point('mid', texts)
        ident = seen['docs']
        self.assertEqual(ident['scorer'], d2l._BPB_SCORER)
        self.assertEqual(ident['texts'], d2l._bpb_texts_id(texts))
        self.assertEqual(ident['pin'], d2l.PRETRAIN_SOURCES['smollm3-mid'][2])
        self.assertEqual(part['texts'], d2l._bpb_texts_id(texts))
        self.assertEqual(part['ctx'], 65536)

    def test_cloze_copied_from_v1(self):
        """v1's per-item cloze scores are copied only for the pins v1
        scored and its suite (MMLU's 2,000-item subset; F25 review 6)."""
        def v1(mmlu=2000, label='LC 64K'):
            return mock.Mock(items=pd.DataFrame([
                dict(task=t, item_id=j, score=1.0, checkpoint=c)
                for c in (label, 'mid-training') for t in d2l.SMOLLM3_TASKS
                for j in range(mmlu if t == 'mmlu' else 2)]))
        n = 2 * (len(d2l.SMOLLM3_TASKS) - 1) + 2000
        with mock.patch.object(d2l, '_read_result', lambda p, k: v1()), \
                mock.patch.object(d2l, 'download_keyed', lambda k: k):
            got = d2l._smollm3_cloze('mid')
            self.assertEqual(len(got), n)
            self.assertEqual(set(got['checkpoint']), {'mid-training'})
            self.assertIsNone(d2l._smollm3_cloze('stage1'))
            # a re-pinned checkpoint is scored afresh
            src = dict(d2l.PRETRAIN_SOURCES)
            src['smollm3-mid'] = src['smollm3-mid'][:2] + ('f' * 40,) + \
                src['smollm3-mid'][3:]
            with mock.patch.dict(d2l.PRETRAIN_SOURCES, src):
                self.assertIsNone(d2l._smollm3_cloze('mid'))
        with mock.patch.object(d2l, '_read_result',
                               lambda p, k: v1(mmlu=1000)), \
                mock.patch.object(d2l, 'download_keyed', lambda k: k):
            self.assertIsNone(d2l._smollm3_cloze('mid'))  # another subset
        with mock.patch.dict(d2l.DATA_HUB, {}, clear=True):
            self.assertIsNone(d2l._smollm3_cloze('mid'))
        # today's pins are the ones v1 scored
        for k, rev in d2l._SMOLLM3_V1_PINS.items():
            self.assertEqual(d2l.PRETRAIN_SOURCES[f'smollm3-{k}'][2], rev)

    def test_live_call(self):
        """smollm3_gsm8k: the first n of the 500 committed items, both
        formats, SmolLM3's end token, no forced answers, the live
        batch."""
        items = [dict(id=str(i), question='q', prompt='P', reference='1')
                 for i in range(500)]
        calls = []

        def score(model, tok, its, **kw):
            calls.append((model, its, kw))
            return pd.DataFrame(dict(format='chat', item_id=[
                it['id'] for it in its], score=1.0, text='t', pred='1',
                forced_pred='1', forced_text=None))
        with mock.patch.object(d2l, 'gsm8k_items', lambda n, seed: items), \
                mock.patch.object(d2l, 'score_gsm8k', score), \
                mock.patch.object(d2l, '_smollm3_load', lambda name: name), \
                mock.patch.object(d2l, '_smollm3_tokenizer', lambda n: n), \
                mock.patch.object(d2l, 'lm_free', lambda: None):
            res = d2l.smollm3_gsm8k(['lc64k', 'mid'], n=50)
        self.assertEqual([c[0] for c in calls], ['lc64k', 'mid'])
        self.assertEqual(calls[0][1], items[:50])
        kw = calls[0][2]
        self.assertEqual((kw['force'], kw['end_token'], kw['budget']),
                         (0, '<|end_of_text|>', dict(completion=256,
                                                     chat=4096)))
        self.assertEqual(kw['batch_size'], d2l._SMOLLM3_LIVE['batch'])
        self.assertEqual(list(res.columns)[:2], ['checkpoint', 'format'])
        self.assertNotIn('text', res.columns)
        self.assertEqual(set(res['checkpoint']), {'LC 64K', 'mid-training'})


class NoPrefixBpbTest(unittest.TestCase):
    """Bits per byte with no prefix token (F25): the first token, which
    has no context, goes unscored with the bytes it spells."""
    def test_scored_doc_hand_count(self):
        tok = ByteTokenizer()
        ids, first, nb = d2l._scored_doc(tok, 'héllo wörld')
        self.assertEqual((len(ids), first, nb), (13, 1, 12))
        # a first character of two bytes: both go unscored
        ids, first, nb = d2l._scored_doc(tok, 'éa')
        self.assertEqual((ids, first, nb), ([195, 169, 97], 2, 1))
        # a limit inside a character drops the cut character
        ids, first, nb = d2l._scored_doc(tok, 'abé', limit=3)
        self.assertEqual((ids, first, nb), ([97, 98], 1, 1))
        ids, first, nb = d2l._scored_doc(tok, 'abcdef', limit=4)
        self.assertEqual((ids, first, nb), ([97, 98, 99, 100], 1, 3))

    def test_stats_against_bits_per_byte(self):
        """At T = 1 the summed NLL over bytes equals bits_per_byte with no
        boundary token, windows and all (ctx 8, stride 4); every target
        is scored once; a short document equals one full forward."""
        model, tok = tiny_llama(), ByteTokenizer()
        texts = ['the quick brown fox jumps over the lazy dog', 'éa ok',
                 'short']
        docs = [d2l._scored_doc(tok, t) for t in texts]
        stats = d2l._doc_stats(model, [(i, f) for i, f, _ in docs],
                               ctx_len=8, stride=4, batch_size=3)
        k1 = d2l._SMOLLM3_TEMPS.index(1.0)
        self.assertEqual(list(stats[:, -1]), [len(i) - f for i, f, _ in docs])
        with contextlib.redirect_stderr(io.StringIO()):
            for t, (ids, f, nb), s in zip(texts, docs, stats):
                want = d2l.bits_per_byte(model, tok, [t], ctx_len=8,
                                         stride=4)
                self.assertAlmostEqual(s[k1] / (math.log(2) * nb), want,
                                       places=5)
        ids, f, _ = docs[2]
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0].float()
        direct = F.cross_entropy(logits[f - 1:-1], torch.tensor(ids[f:]),
                                 reduction='sum').item()
        one = d2l._doc_stats(model, [(ids, f)], ctx_len=8)
        self.assertAlmostEqual(one[0, k1], direct, places=4)

    def test_logit_stats_grid(self):
        """The NLL at each temperature, the entropy and top-1 on fixed
        logits, against a direct computation."""
        z = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]])
        y = torch.tensor([0, 0])
        temps = d2l._SMOLLM3_TEMPS
        got = d2l._logit_stats(z, y, temps).numpy()
        for j, t in enumerate(temps):
            p = np.exp(z.numpy() / t)
            p /= p.sum(1, keepdims=True)
            np.testing.assert_allclose(got[:, j], -np.log(p[[0, 1], [0, 0]]),
                                       rtol=1e-6)
        p = np.exp(z.numpy())
        p /= p.sum(1, keepdims=True)
        np.testing.assert_allclose(got[:, len(temps)],
                                   -(p * np.log(p)).sum(1), rtol=1e-6)
        self.assertEqual(list(got[:, -1]), [0.0, 1.0])
        # a uniform distribution: the same loss at every temperature
        u = d2l._logit_stats(torch.zeros(1, 4), torch.tensor([2]), temps)
        np.testing.assert_allclose(u[0, :len(temps)].numpy(),
                                   [math.log(4)] * len(temps), rtol=1e-6)


class SmolLM3AnswersTest(unittest.TestCase):
    """The reasoning instruments: MATH-500's numeric subset, the two
    formats' prompts, and score_gsm8k under SmolLM3's tokens."""
    def test_math_value(self):
        cases = {'9': '9', '-50': '-50', '\\frac{14}{3}': '14/3',
                 '\\dfrac{6}{4}': '3/2', '-\\frac34': '-3/4', '0.15': '3/20',
                 '1.25': '5/4', '2,000': '2000', '10{,}000': '10000',
                 '$-5$': '-5', '3/56': '3/56', '-.5': '-1/2'}
        for s, want in cases.items():
            self.assertEqual(d2l.math_value(s), want, s)
        for s in ('90^\\circ', '1,-2', '-\\frac{\\pi}{6}', '3\\sqrt{13}',
                  'x=5', '\\text{Evelyn}', '52_8', '5/0', '6 - 5i', '1,00',
                  None, '(-2,1)'):
            self.assertIsNone(d2l.math_value(s), s)

    def test_math_prediction_value(self):
        """A prediction loses \\$, \\%, a degree sign and a trailing unit;
        a reference stays strict (F25 review 11)."""
        cases = {'\\$18.90': '189/10', '90^\\circ': '90', '90^{\\circ}': '90',
                 '25\\%': '25', '5\\text{ cm}': '5', '12 \\mbox{inches}': '12',
                 '\\frac{1}{2}\\text{ m}': '1/2', '7': '7'}
        for s, want in cases.items():
            self.assertEqual(d2l.math_prediction_value(s), want, s)
        for s in ('\\text{Evelyn}', 'x=5', '3\\sqrt{13}', None, '\\text{5}'):
            self.assertIsNone(d2l.math_prediction_value(s), s)
        self.assertIsNone(d2l.math_value('90^\\circ'))  # references strict
        self.assertEqual(d2l.math_chat_answer(
            '<think> a </think> \\boxed{30^\\circ}', 'eos'), '30')
        self.assertEqual(d2l.math_completion_answer(
            'Final Answer: The final answer is $\\$5$. I hope it is correct.'),
            '5')

    def test_math_answers(self):
        tail = ('\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope'
                ' it is correct.')
        self.assertEqual(d2l.math_completion_answer('so x = 3.' + tail),
                         '-2/3')
        self.assertEqual(d2l.math_completion_answer('so \\boxed{24}.'), '24')
        self.assertEqual(d2l.math_completion_answer(
            'The final answer is $24$'), '24')
        self.assertIsNone(d2l.math_completion_answer('no answer'))
        ans = d2l.math_chat_answer
        self.assertEqual(ans('<think> 2/3 </think> $\\boxed{\\frac{2}{3}}$',
                             'eos'), '2/3')
        self.assertIsNone(ans('<think> \\boxed{1}', 'eos'))  # never closed
        self.assertIsNone(ans('<think> a </think> \\boxed{1}', 'length'))
        self.assertEqual(ans('\\boxed{0.5}', 'eos'), '1/2')

    def test_math500_items(self):
        """The numeric filter and the Minerva prompt on a fixture file."""
        rows = [dict(problem='p1', answer='\\frac{1}{2}', unique_id='a'),
                dict(problem='p2', answer='x^2', unique_id='b'),
                dict(problem='p3', answer='42', unique_id='c')]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'test.jsonl')
            with open(path, 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in rows)
            with mock.patch('huggingface_hub.hf_hub_download',
                            lambda *a, **k: path):
                items = d2l.math500_items()
        self.assertEqual([(i['id'], i['reference']) for i in items],
                         [('a', '1/2'), ('c', '42')])
        p = items[1]['prompt']
        self.assertTrue(p.startswith('Problem:\nFind the domain'))
        self.assertTrue(p.endswith('Problem:\np3\n\nSolution:'))
        self.assertEqual(p.count('Problem:'), 5)
        self.assertEqual(p.count('I hope it is correct.'), 4)
        # each demonstration's own answer is read by the completion rule
        for q, a in d2l._MINERVA_DEMOS[1:]:
            self.assertIsNotNone(d2l.math_completion_answer(a), q)

    @unittest.skipUnless(os.path.exists(os.path.join(
        os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')),
        'hub', 'datasets--HuggingFaceH4--MATH-500')),
        'MATH-500 is not on this machine')
    def test_math500_count(self):
        items = d2l.math500_items()
        self.assertEqual(len(items), 376)  # F25: counted at build

    def test_templates(self):
        """One prompt per format for every checkpoint: the chat prompt is
        the mid-training tokenizer's own chat template, and the
        completion prompt is gsm8k_cot's 8-shot prompt."""
        try:
            tok = d2l._smollm3_tokenizer('mid')
        except Exception as e:  # the tokenizer is not on this machine
            self.skipTest(f'SmolLM3 tokenizer not available: {e}')
        msgs = [dict(role='user', content='What is 2 + 2?')]
        self.assertEqual(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True),
            d2l.gsm8k_chat_prompt('What is 2 + 2?'))
        base = d2l._smollm3_tokenizer('lc64k')
        prompt = d2l.gsm8k_chat_prompt('What is 2 + 2?')
        for t in (tok, base):  # the same ids: ChatML markers are special
            ids = d2l._lm_prompt_ids(t, [prompt])[0]
            self.assertEqual((ids[0], ids.count(128011), ids.count(128012)),
                             (128011, 2, 1))
        self.assertEqual(d2l._lm_prompt_ids(tok, [prompt]),
                         d2l._lm_prompt_ids(base, [prompt]))
        self.assertEqual(tok.convert_tokens_to_ids('<|end_of_text|>'),
                         128001)
        self.assertIsNone(tok.bos_token)  # nothing goes before a prompt
        self.assertFalse(tok.clean_up_tokenization_spaces)  # no warning
        with self.assertNoLogs('transformers', level='WARNING'):
            tok.batch_decode([[9906, 1174]], skip_special_tokens=True)

    def test_gsm_symbolic_items(self):
        rows = [dict(id=t, instance=i, question=f'q{t}{i}',
                     answer=f'so #### {t + i}') for t in range(3)
                for i in range(6)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'test.jsonl')
            with open(path, 'w') as f:
                f.writelines(json.dumps(r) + '\n' for r in rows)
            with mock.patch.object(d2l, 'download_keyed', lambda k: path):
                items = d2l.gsm_symbolic_items(5)
        self.assertEqual(len(items), 15)
        self.assertEqual((items[6]['id'], items[6]['reference']),
                         ('1-1', '2'))
        self.assertEqual(items[0]['prompt'], d2l.fewshot_prompt(
            d2l.GSM8K_COT_DEMOS, 'q00', 'qa'))

    def test_score_smollm3_tokens_and_math(self):
        """score_gsm8k with SmolLM3's end token, and kind='math': stop at
        'Problem:', fractions compared by value, forced answers read from
        a box closed by '}$'."""
        class SmolTok:
            ids = {'<|end_of_text|>': 128001, '<|im_end|>': 128012}

            def convert_tokens_to_ids(self, t):
                return self.ids.get(t)
        items = [dict(id=str(i), question=f'q{i}', prompt=f'P{i}',
                      reference='2/3') for i in range(3)]
        replies = {
            'completion': [Gen('...\nFinal Answer: The final answer is '
                               '$\\frac{4}{6}$. I hope it is correct.',
                               'stop'), Gen('x = 3', 'stop'),
                           Gen('\\boxed{0.5}', 'stop')],
            'chat': [Gen('<think> a </think> \\boxed{\\dfrac{2}{3}}'),
                     Gen('<think> long', 'length', 8192),
                     Gen('<think> b </think> 0.66')],
            'forced': [Gen('\\frac{2}{3}', 'stop')]}
        calls = []

        def generate(model, tok, prompts, max_new, **kw):
            fmt = 'forced' if kw['stop'] == ['}$'] else 'chat' if prompts[
                0].startswith('<|im_start|>') else 'completion'
            calls.append((fmt, max_new, kw))
            return replies[fmt]
        cfg = d2l._SMOLLM3_ANSWERS['math500']
        with mock.patch.object(d2l, 'generate', generate):
            res = d2l.score_gsm8k(None, SmolTok(), items, budget=cfg[
                'budget'], batch_size=cfg['batch'], end_token=
                '<|end_of_text|>', kind='math')
        comp, chat, forced = calls
        self.assertEqual((comp[1], comp[2]['stop'], comp[2]['eos_token_ids']),
                         (1024, ['Problem:'], [128001]))
        self.assertEqual((chat[1], sorted(chat[2]['eos_token_ids'])),
                         (8192, [128001, 128012]))
        self.assertEqual(forced[2]['batch_size'], 8)
        r = res.set_index(['format', 'item_id'])
        self.assertEqual(list(r.loc['completion', 'score']), [1, 0, 0])
        self.assertEqual(list(r.loc['chat', 'score']), [1, 0, 0])
        self.assertEqual(list(r.loc['chat', 'forced']), [1, 1, 0])
        with self.assertRaises(ValueError):
            d2l.score_gsm8k(None, SmolTok(), items, kind='aime')


@contextlib.contextmanager
def warnings_as_errors():
    import warnings
    with warnings.catch_warnings():
        for kind in (UserWarning, RuntimeWarning, FutureWarning):
            warnings.simplefilter('error', kind)
        yield


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
        self.assertEqual(list(res.columns), d2l._CPT_COLUMNS['gsm8k'][1:]
                         + d2l._GSM8K_TEXT)
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
        # the replies, answers and forced continuations are kept (F23
        # review finding 2): enough to read and re-score every item
        self.assertEqual(list(r.loc['chat', 'text']),
                         [g.text for g in replies['chat']])
        kept = lambda col: [None if pd.isna(v) else v for v in col]
        self.assertEqual(kept(r.loc['chat', 'pred']),
                         ['18', None, None, '18', '17'])
        self.assertEqual(kept(r.loc['chat', 'forced_pred']),
                         ['18', '18', '17', '18', '17'])
        self.assertEqual(kept(r.loc['chat', 'forced_text']),
                         [None, '18', '17', None, None])
        self.assertEqual(kept(r.loc['completion', 'pred']),
                         ['18', '18', '17', '3', None])
        self.assertEqual(kept(r.loc['completion', 'pred']),
                         kept(r.loc['completion', 'forced_pred']))
        self.assertTrue(r.loc['completion', 'forced_text'].isna().all())

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
                                     chat_tokens=[False], new_tokens=[9],
                                     text=['<think> x </think> 18'],
                                     pred=['18'], forced_pred=['18'],
                                     forced_text=[None]))

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
            # the model's own file keeps the replies; the merge does not
            own = d2l._read_result(os.path.join(root, 'cpt', 't2-base',
                                                'gsm8k.json.gz'), 'json-table')
            self.assertEqual(list(own['text']), ['<think> x </think> 18'])
            self.assertIsNone(own['forced_text'].iloc[0])
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
            # a file scored under other answer rules is refused (F23
            # review finding 2)
            with mock.patch.object(d2l, '_GSM8K_SCORER',
                                   d2l._GSM8K_SCORER + 1), \
                    self.assertRaisesRegex(ValueError, 'settings.scorer'):
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
        self.assertRegex(out[1], r'; B − C → A′ chord: [-+]0\.\d{3}$')
        self.assertLessEqual(max(map(len, out)), 80)

    def test_frontier_chord_below_c(self):
        """F23 review finding 1: a B that forgets less than C is measured
        against the base → C segment, and the second line names no A′
        chord (the reference does not involve A′)."""
        base, _ = synthetic_docs()
        fg = d2l.forgetting_gain(base, pd.concat([
            shifted(base, g, f, 0.0, sd).assign(arm=arm)
            for arm, g, f, sd in (('C', 0.03, 0.01, 1), ('A', 0.06, 0.04, 2),
                                  ('A′', 0.061, 0.041, 4),
                                  ('B', 0.02, 0.0049, 3))]))
        ns = dict(page_ns(), fg=fg,
                  seed=fg.set_index('arm').reindex(['A', 'A′']))
        out = run_cell('continued-pretraining-frontier-chord',
                       ns).splitlines()
        self.assertEqual(len(out), 2)
        self.assertRegex(out[0], r'^B − no-replay reference \(base → C '
                                 r'segment\): \+0\.\d{3} \[')
        self.assertEqual(out[1], 'seed spread (A vs A′): gain 0.001; '
                                 'C → A′ chord: none, B forgets less than C')
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
