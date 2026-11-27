#!/usr/bin/env python3
"""CPU tests for sec_lm_reference_model (chapter_lm-pretraining/
reference-model.md and reference-model-lib.md) and the chapter's
load-or-regenerate plumbing (data-path-lib.md): `wsd_lr` (fixture, 1e-9);
`param_groups` (name sets, r2's decayed count 113,246,208 on the meta
device); `flops_per_token` against the formula for every preset (exact
ints), r2 = 1,509,949,440, the Transformers-config path on an offline
Qwen3-0.6B-Base config (6 N_mm = 3,575,906,304, F(T=2,048) =
4,985,192,448), `causal_half`, `T=None`; `mfu` arithmetic; `r2_config`
overrides (an unknown field raises); `_RUN_NAMES`, `PRETRAIN_JOBS` and
`_RERUN_ARTIFACT` against `PRETRAIN_RERUN_NAMES`, which equals the
chapter's 40 names; `pretrain` on a 2-layer toy (30 steps, accumulation 1
vs 4 at the same global batch, fp32 1e-5), resume after 20 of 40 steps
(losses 1e-6) and a branch from a saved state; the stats-step columns
`entropy_<l>` and `log_z` (finite; `log_z` the explicit mean logsumexp
1e-5; only at stats steps, absent without stats); `clip_from` (clipping
after that many updates; losses before it equal an unclipped run's;
clip_from = bad_step - 1 clips the bad batch, `_spike` refuses a
configuration that clips; `_cpt_step` refuses clip_from);
`_cfg_json` of a default config equal to the committed `lm-r2-config-v1`;
`attn_logit_stats` against the explicit softmax input; the `qk_clip_`
contract; `softcap`; `make_bad_batch` ('ngram': shape, period, one n-gram
of the batch's own ids, untouched rows bit-identical; an unknown kind
raises); `_BadBatchStream` argument checks (kind, period 1-13, k, one
rank for 'ngram'; `source` puts the same bad batch at any step);
`lr_sensitivity`; `_pretrain_result` and
`_pretrain_merged` with a temporary registry (regenerate with the config
stored, load, configuration mismatch, absent in a normal run, pending,
two regeneration names with other calls in the same process, D2L_RERUN=1
refused); `t_interval` against `scipy.stats.t` (dof 1-10, 1e-4);
`packed_to_hf` against `LlamaForCausalLM` (fp32 1e-5);
`load_reference_model` before the checkpoints are registered; the T1
grid: each variant's arm config (schedule, clip, warmup 305, softcap
backend, QK-norm; unknown variants refused), the zero-layer model trains,
`_arm_summary` (first step above 1,000, smallest and peak-phase entropy),
`first_collapsed_rate`, `departure` and `collapse_crossings` on synthetic
traces (a diverged arm's loss crossing is its first non-finite step);
`update_rms_by_phase` (phase medians; the absolute update = the per-lr
median x the phase's median lr) and `moved` (white noise fires at the same
rate, under 12%, dense and every 10 steps; at most 2 of 20 stable steps of
a noisy convex a + b/sqrt(t) curve, which fires inside its ramp; a shift
found within the window, also every 10 steps; spikes and moves past the
horizon ignored), the
softcap entropy from capped scores with the raw maximum, the seed sets the
data order, a resumed job reads its finished arms, the merge
keeps the zero-layer loss, the regeneration stores the recipe (with the
statistics version; a changed one is refused); a 2-process
gloo DDP `pretrain` equal to one process at the same global batch (loss and
`log_z`, 1e-5); the size transfer: `t_quantile` against `scipy.stats.t`,
`logit_fit` (a noiseless line recovered exactly; over 1,000 noisy draws the
slope's interval and the prediction interval for a two-seed mean cover the
truth at 95% within 2 points; non-finite logits and two-point rates left
out), the run configurations (shapes at T = 1,024, N = 12.6M to 113.3M, the
grid's recipe, the seed sets the data order), the pilot's gate, the driver
(the pilot runs only its 8 runs; a no-go stops the full job before any
other run; the prediction is registered before the held-out shape's first
run and equals the page's fit of the stored runs; a changed prediction or
a held-out run kept without one is refused), a real 2-layer run kept and
read back, the regeneration names (pilot, full after a go, pending) and
the plot; one bad batch under Adam: the arm table's checks,
`spike_excess` on a synthetic pair (peak, steps until it stays below twice
the floor, sum; the bad step excluded; a stopped replay), the paired driver
on a 2-layer toy (two skip replays of one state bit-identical, so the
floor is 0; every arm equal to its skip replay before the bad step; clip
twins clip from their state; a committed late state of another beta2
refused; the run states removed), the regeneration (v2, the recipe, a
changed period or recipe refused, pending), the page's 10-line table and
the plot; OLMo 2's repeated n-gram rule (32 repeats of periods 1, 7 and 13
flagged with period, repeats and start, 31 not; max_n); Adam's epsilon
headroom: the readout of a hand-built AdamW state (bias-corrected sqrt(v)
per group, medians and shares exact, the pooled matrix median), the pair
check (any field but eps refused), the driver on mocked runs (eps
registered before the run, the grid cell read not re-run, kept and
resumed; a changed readout, an unregistered kept run, a grid without the
cell or of another recipe refused), the regeneration, the page's table;
the z-loss pilot's no-go (claim 26): no z-loss cell or call on the page,
no pilot in the library, the name unregistered.
"""
import collections
import contextlib
import dataclasses
import io
import math
import os
import re
import sys
import tempfile
import time
import types
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import torch

from d2l import torch as d2l


def uncommitted():
    """d2l.DATA_HUB without chapter 21's results: the state before a
    campaign's result is registered (the tests of pending and regeneration
    paths must not depend on which results run A has committed)."""
    keys = {key for key, _ in d2l.PRETRAIN_ARTIFACTS.values()}
    weights = re.compile(r'^lm-r2(t1)?-s\d+(-state)?-v\d+$|^lm-r2-16k-')
    return mock.patch.object(d2l, 'DATA_HUB', {
        k: v for k, v in d2l.DATA_HUB.items()
        if k not in keys and not weights.match(k)})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_lm_data_path as dp  # noqa: E402

torch.set_num_threads(min(8, torch.get_num_threads()))

RERUN_NAMES = (
    'pdp-throughput', 'pdp-mask-runs', 't1-runs', 'inst-t1-none',
    'inst-t1-qkclip', 'inst-t1-qknorm', 'inst-t1-softcap', 'inst-t1-clip',
    'inst-t1-warmup', 'size-xfer-pilot', 'size-xfer', 'spike-replay',
    'eps-headroom',
    'reference-run', 'ladder', 'inst-t2', 'reference-evals', 't1-evals',
    'ladder-evals',
    'gemm-sweep', 'lowp-runs', 'cpt-t1', 'cpt-qwen-A', 'cpt-qwen-B',
    'cpt-qwen-C', 'cpt-qwen-A2', 'cpt-qwen-evals', 'smollm3-0', 'smollm3-1',
    'smollm3-2', 'smollm3-3', 'smollm3-4', 'smollm3-5', 'lc-t1',
    'lc-zeroshot', 'lc-t2-pi', 'lc-t2-yarn', 'lc-t2-prog', 'lc-t2-evals',
    'decode-rates')

TOY = d2l.Arch(L=2, d=32, H=4, H_kv=2, d_h=8, f=64, vocab=512, T=32,
               rope_base=1e4, qk_norm=False, tie=True)


def toy_config(**kw):
    args = dict(preset='toy', arch=TOY, steps=30, global_batch=8,
                micro_batch=8, peak_lr=3e-3, warmup=3, decay=6, dtype='fp32',
                compile=False, probe_every=0, stats_every=0, val_every=0,
                z_loss=1e-4)
    args.update(kw)
    return d2l.PretrainConfig(**args)


PRE_F04_FIELDS = (  # PretrainConfig's fields when the runs were committed
    'preset', 'arch', 'steps', 'global_batch', 'micro_batch', 'peak_lr',
    'warmup', 'decay', 'final_lr', 'betas', 'eps', 'wd', 'clip', 'z_loss',
    'dtype', 'compile', 'ddp', 'probe_every', 'stats_every', 'val_every',
    'val_tokens', 'ckpt_steps', 'state_steps', 'state_every', 'branches',
    'data', 'data_seed', 'seed', 'out_dir')


def toy_model(seed=0):
    torch.manual_seed(seed)
    return d2l.PackedGPT.from_arch(TOY, backend='sdpa')


class ScheduleTest(unittest.TestCase):
    def test_wsd_fixture(self):
        lr = [d2l.wsd_lr(s, 10, 2, 4, 1.0, 0.1) for s in range(10)]
        ref = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1 + 0.9 * 4 / 4,
               0.1 + 0.9 * 3 / 4, 0.1 + 0.9 * 2 / 4, 0.1 + 0.9 * 1 / 4]
        for a, b in zip(lr, ref):
            self.assertLess(abs(a - b), 1e-9)

    def test_wsd_spec_fixture(self):
        steps = (0, 1, 99, 100, 1000, 8499, 8500, 9999)
        want = (1e-5, 2e-5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3 / 1500)
        for s, w in zip(steps, want):
            self.assertLess(abs(d2l.wsd_lr(s, 10000, 100, 1500, 1e-3) - w),
                            1e-9, s)

    def test_param_groups(self):
        m = toy_model()
        decay, rest = d2l.param_groups(m, 0.1)
        self.assertEqual((decay['weight_decay'], rest['weight_decay']),
                         (0.1, 0.0))
        self.assertIn('blks.0.attention.W_q.weight', decay['names'])
        self.assertIn('token_emb.weight', rest['names'])
        self.assertIn('blks.0.norm1.weight', rest['names'])
        self.assertEqual(len(decay['names']) + len(rest['names']),
                         len(list(m.parameters())))
        with torch.device('meta'):
            r2 = d2l.PackedGPT.from_arch(d2l.r2_config().arch)
        decay, _ = d2l.param_groups(r2, 0.1)
        self.assertEqual(sum(p.numel() for p in decay['params']),
                         113_246_208)


class FlopsTest(unittest.TestCase):
    def test_presets(self):
        for preset in d2l.R2_ARCHS:
            a = d2l.r2_config(preset).arch
            N = a.L * (2 * a.d * a.H * a.d_h + 2 * a.d * a.H_kv * a.d_h
                       + 3 * a.d * a.f) + a.vocab * a.d
            F = 6 * N + 12 * a.L * a.H * a.d_h * a.T
            self.assertEqual(d2l.flops_per_token(a, T=a.T), F, preset)
            self.assertEqual(d2l.flops_per_token(a), 6 * N)
            self.assertEqual(d2l.flops_per_token(a, T=a.T, causal_half=True),
                             6 * N + 6 * a.L * a.H * a.d_h * a.T)
            self.assertIsInstance(d2l.flops_per_token(a, T=a.T), int)
        self.assertEqual(d2l.flops_per_token(d2l.r2_config().arch, T=4096),
                         1_509_949_440)

    def test_hf_config(self):
        import transformers
        cfg = transformers.Qwen3Config(
            vocab_size=151936, hidden_size=1024, intermediate_size=3072,
            num_hidden_layers=28, num_attention_heads=16,
            num_key_value_heads=8, head_dim=128, tie_word_embeddings=True)
        self.assertEqual(d2l.flops_per_token(cfg), 3_575_906_304)
        self.assertEqual(d2l.flops_per_token(cfg, T=2048), 4_985_192_448)
        self.assertEqual(d2l.flops_per_token(cfg.to_dict(), T=2048),
                         4_985_192_448)
        # A raw config.json without the key is tied (Transformers' default)
        raw = {k: v for k, v in cfg.to_dict().items()
               if k != 'tie_word_embeddings'}
        self.assertEqual(d2l.flops_per_token(raw), 3_575_906_304)

    def test_mfu(self):
        a = d2l.r2_config().arch
        F = d2l.flops_per_token(a, T=a.T)
        self.assertAlmostEqual(d2l.mfu(1e5, a, 165e12, 4),
                               1e5 * F / (4 * 165e12))
        self.assertAlmostEqual(
            d2l.mfu(1e5, a, 165e12, causal_half=True),
            1e5 * d2l.flops_per_token(a, T=a.T, causal_half=True) / 165e12)
        self.assertAlmostEqual(d2l.mfu(1e5, a, 1e12, T=1024),
                               1e5 * d2l.flops_per_token(a, T=1024) / 1e12)


class ConfigTest(unittest.TestCase):
    def test_overrides(self):
        c = d2l.r2_config('t1', seed=1, clip=None)
        self.assertEqual((c.seed, c.clip, c.preset), (1, None, 't1'))
        self.assertEqual(c.arch.T, 1024)
        self.assertEqual(c.micro_batch, 16)
        self.assertEqual(c.accum(1), 4)
        with self.assertRaises(TypeError):
            d2l.r2_config('r2', no_such_field=1)
        with self.assertRaises(ValueError):
            d2l.r2_config('r2', micro_batch=7).accum(1)
        r2 = d2l.r2_config()
        # micro-batch 4: at 8, r2 ran out of memory in run A's D1 probe
        self.assertEqual((r2.steps, r2.global_batch, r2.micro_batch,
                          r2.accum(4)), (6000, 128, 4, 8))
        self.assertEqual(d2l.r2_config('ladder-88m').micro_batch, 8)
        self.assertEqual(r2.branches, ((2550, 450), (1275, 225)))
        self.assertIn(5100, r2.state_steps)
        self.assertEqual(r2.stats_every, 10)
        self.assertEqual(r2.ckpt_steps, d2l.REFERENCE_STEPS)
        t1 = d2l.r2_config('t1')
        self.assertEqual(t1.stats_every, 50)
        # the t1 checkpoints sit at the reference run's fractions
        self.assertEqual(t1.ckpt_steps, d2l.T1_STEPS)
        for a, b in zip(d2l.REFERENCE_STEPS, d2l.T1_STEPS):
            self.assertLessEqual(abs(a / r2.steps - b / t1.steps),
                                 0.5 / t1.steps + 1e-12)  # rounded
        self.assertIn(t1.steps - t1.decay, t1.state_steps)
        self.assertEqual(d2l._reference_key('t1', 1297), 'lm-r2t1-s1297-v1')
        with self.assertRaises(ValueError):
            d2l._reference_key('t1', 1300)

    def test_job_checkpoint_only_for_evals(self):
        for rerun, want in (('spike-replay', None), ('reference-evals', 'r2'),
                            ('t1-evals', 't1')):
            with mock.patch.dict(os.environ, {'D2L_RERUN': rerun}):
                for preset in ('r2', 't1'):
                    path = d2l._job_checkpoint(preset, None)
                    self.assertEqual(path is not None, preset == want,
                                     (rerun, preset))

    def test_names(self):
        self.assertEqual(d2l.PRETRAIN_RERUN_NAMES, RERUN_NAMES)
        names = set(RERUN_NAMES)
        self.assertLessEqual(set(d2l._RUN_NAMES.values()), names)
        self.assertLessEqual(set(d2l.PRETRAIN_JOBS), names)
        self.assertEqual(set(d2l._RERUN_ARTIFACT), names)
        self.assertLessEqual(set(d2l._RERUN_ARTIFACT.values()),
                             set(d2l.PRETRAIN_ARTIFACTS))
        needs_run = {'ladder-evals', 'inst-t2', 'inst-t1-none',
                     'inst-t1-qkclip', 'inst-t1-qknorm', 'inst-t1-softcap',
                     'inst-t1-clip', 'inst-t1-warmup'}
        with mock.patch.object(d2l, '_rerun_dir', lambda: '/nonexistent'), \
                uncommitted():
            for name, (driver, kwargs) in d2l.PRETRAIN_JOBS.items():
                self.assertTrue(callable(getattr(d2l, driver)), name)
                if name in needs_run:  # reads an earlier job's result
                    with self.assertRaisesRegex(FileNotFoundError,
                                                'has not run yet'):
                        kwargs()
                else:
                    kwargs()  # builds without error
        for key, kind in d2l.PRETRAIN_ARTIFACTS.values():
            self.assertRegex(key, r'^lm-[a-z0-9.-]+-v\d+$')
            self.assertIn(kind, d2l._ARTIFACT_EXT)

    def test_cfg_json_round_trip(self):
        c = d2l.r2_config('ladder-54m', seed=2)
        self.assertEqual(d2l._cfg_from_json(d2l._cfg_json(c)), c)
        c = d2l.r2_config('t1', clip_from=250)
        self.assertEqual(d2l._cfg_json(c)['clip_from'], 250)
        self.assertEqual(d2l._cfg_from_json(d2l._cfg_json(c)), c)

    def test_cfg_json_identity_of_committed_runs(self):
        # fields added later (clip_from) are left out at their default, so
        # the configurations stored with committed runs still match
        self.assertEqual(set(d2l._cfg_json(d2l.r2_config('r2'))), set(
            PRE_F04_FIELDS))
        self.assertNotIn('clip_from', d2l._cfg_json(d2l.r2_config('t1')))
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'data', 'lm')
        key = 'lm-r2-config-v1'
        paths = [os.path.join(root, key, key + '.json'),
                 os.path.join(root, 'rerun', key + '.json')]
        path = next((p for p in paths if os.path.exists(p)), None)
        if path is None:
            self.skipTest(f'{key} is not on disk')
        import hashlib
        import json
        with open(path, 'rb') as f:
            raw = f.read()
        self.assertEqual(hashlib.sha1(raw).hexdigest(), d2l.DATA_HUB[key][1])
        stored = json.loads(raw)['jobs']['r2-s0']['pretrain_config']
        mine = d2l._cfg_json(d2l.r2_config('r2', ddp=True))
        mine.pop('out_dir')
        self.assertEqual(mine, stored)
        self.assertEqual(d2l._cfg_from_json(dict(stored, out_dir=None)),
                         d2l.r2_config('r2', ddp=True))

    def test_load_reference_model_unregistered(self):
        with uncommitted(), self.assertRaises(KeyError) as cm:
            d2l.load_reference_model()
        self.assertIn('reference checkpoints not committed yet',
                      str(cm.exception))


class PretrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.shards = dp.fixture_shards(os.path.join(cls.tmp.name, 'set'),
                                       n_docs=600)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stream(self, **kw):
        return d2l.PackedStream(self.shards, TOY.T, 8, seed=1, **kw)

    def run_toy(self, cfg, resume=None, seed=0):
        with contextlib.redirect_stdout(io.StringIO()):
            return d2l.pretrain(toy_model(seed), self.stream(), cfg,
                                resume=resume)

    def test_accumulation(self):
        a = self.run_toy(toy_config(micro_batch=8))
        b = self.run_toy(toy_config(micro_batch=2))
        self.assertEqual(len(a), 30)
        self.assertEqual(a['step'].tolist(), list(range(1, 31)))
        self.assertLess(np.abs(a['loss'] - b['loss']).max(), 1e-5)
        self.assertLess(a['loss'].iloc[-1], a['loss'].iloc[0])
        self.assertTrue({'lr', 'grad_norm', 'clipped', 'tok_s', 'mfu',
                         'run', 'seed', 'spike', 'time', 'world', 'gpu',
                         'job_start'} <= set(a.columns))

    def test_resume_and_branch(self):
        with tempfile.TemporaryDirectory() as out:
            full = self.run_toy(toy_config(steps=40, decay=8,
                                           out_dir=os.path.join(out, 'a')))
            part = toy_config(steps=40, decay=8, state_steps=(20,),
                              out_dir=os.path.join(out, 'b'))
            first = d2l.pretrain(toy_model(), self.stream(), part,
                                 hooks=[self.stop_at(20)])
            self.assertEqual(first['step'].max(), 20)
            self.assertEqual(d2l._state_steps(part.out_dir), [20])
            resumed = self.run_toy(part, resume='auto', seed=5)
            self.assertEqual(resumed['step'].tolist(), list(range(1, 41)))
            self.assertLess(np.abs(resumed['loss'].values
                                   - full['loss'].values).max(), 1e-6)
            # a branch from step 20 with 6 decay steps
            br = toy_config(steps=30, decay=6, state_steps=(20,), ckpt_steps=(30,),
                            branches=((20, 6),),
                            out_dir=os.path.join(out, 'c'))
            log = self.run_toy(br)
            self.assertEqual(sorted(set(log['run'])), ['b26', 'toy'])
            b = log[log['run'] == 'b26']
            self.assertEqual(b['step'].tolist(), list(range(21, 27)))
            self.assertAlmostEqual(b['lr'].iloc[-1], 3e-3 / 6)
            self.assertTrue(os.path.exists(os.path.join(
                br.out_dir, 'ckpt', 'b26-s26.safetensors')))
            self.assertTrue(os.path.exists(os.path.join(
                br.out_dir, 'ckpt', 's30.safetensors')))
            m = d2l._load_bf16(os.path.join(br.out_dir, 'ckpt',
                                            's30.safetensors'),
                               dtype=torch.float32)
            self.assertEqual(m.config()['num_blks'], 2)

    def test_resume_after_crash_in_branch(self):
        class Crash(Exception):
            pass

        def crash(step, model, opt, batch, row):
            if row['run'] == 'b26' and row['step'] == 23:
                raise Crash
        with tempfile.TemporaryDirectory() as out:
            ref = self.run_toy(toy_config(
                steps=30, decay=6, state_steps=(20,), ckpt_steps=(30,),
                branches=((20, 6),), out_dir=os.path.join(out, 'ref')))
            cfg = toy_config(steps=30, decay=6, state_steps=(20,),
                             ckpt_steps=(30,), branches=((20, 6),),
                             out_dir=os.path.join(out, 'job'))
            with self.assertRaises(Crash), \
                    contextlib.redirect_stdout(io.StringIO()):
                d2l.pretrain(toy_model(), self.stream(), cfg, hooks=[crash])
            log = self.run_toy(cfg, resume='auto', seed=5)
        b = log[log['run'] == 'b26']
        self.assertEqual(b['step'].tolist(), list(range(21, 27)))
        self.assertEqual(log[log['run'] == 'toy']['step'].tolist(),
                         list(range(1, 31)))
        r = ref[ref['run'] == 'b26']
        self.assertLess(np.abs(b['loss'].values - r['loss'].values).max(),
                        1e-6)

    @staticmethod
    def stop_at(n):
        class Stop:
            def __call__(self, step, model, opt, batch, row):
                if row['step'] == n:
                    row['loss'] = float('nan')  # stopped() ends the run
        return Stop()

    def test_probes(self):
        cfg = toy_config(steps=4, probe_every=2, stats_every=2)
        log = self.run_toy(cfg)
        self.assertIn('rms_W_q', log.columns)
        self.assertIn('max_logit_0', log.columns)
        self.assertTrue(log['rms_W_q'].iloc[[1, 3]].notna().all())

    def test_entropy_and_log_z(self):
        log = self.run_toy(toy_config(steps=3, stats_every=1))
        cols = ['entropy_0', 'entropy_1', 'log_z']
        self.assertTrue(set(cols) <= set(log.columns))
        vals = log[cols].to_numpy()
        self.assertTrue(np.isfinite(vals).all())
        ent = log[['entropy_0', 'entropy_1']].to_numpy()
        self.assertTrue(((ent >= 0) & (ent <= math.log(TOY.T))).all())
        # log_z of step 1: the mean logsumexp of the initial model's logits
        # over the valid targets of batch 0
        m, b = toy_model(), self.stream().batch(0)
        with torch.no_grad():
            H, W = m(b['input_ids'], d2l.PackCtx(b['doc_ids'], b['pos_ids']),
                     return_hidden=True)
            lse = torch.logsumexp(torch.nn.functional.linear(H, W), -1)
        valid = b['labels'] != -100
        self.assertLess(abs(log['log_z'].iloc[0] - lse[valid].mean().item()),
                        1e-5)
        # the last step's entropy: attn_logit_stats' mean over heads of
        # the trained model on that step's batch
        m = toy_model()
        with contextlib.redirect_stdout(io.StringIO()):
            d2l.pretrain(m, self.stream(), toy_config(steps=3, stats_every=3))
        stats = d2l.attn_logit_stats(m, self.stream().batch(2))
        for l, s in stats.items():
            self.assertAlmostEqual(log[f'entropy_{l}'].iloc[2],
                                   float(s.entropy.mean()), places=5)
        # stats every 2 steps: the columns only at steps 2, 4
        log = self.run_toy(toy_config(steps=4, stats_every=2))
        for c in cols + ['max_logit_0']:
            self.assertEqual(log[c].notna().tolist(),
                             [False, True, False, True], c)
        # no stats: none of the new columns (the old logs' columns)
        log = self.run_toy(toy_config(steps=3))
        self.assertFalse({'log_z', 'entropy_0', 'max_logit_0'}
                         & set(log.columns))

    def test_clip_from(self):
        tiny = 1e-3  # every step's norm is above it
        free = self.run_toy(toy_config(steps=10, clip=None))
        late = self.run_toy(toy_config(steps=10, clip=tiny, clip_from=5))
        always = self.run_toy(toy_config(steps=10, clip=tiny))
        # steps 0-4 (log steps 1-5) unclipped, from step 5 on clipped
        self.assertEqual(late['clipped'].tolist(), [False] * 5 + [True] * 5)
        self.assertTrue(always['clipped'].all())
        self.assertFalse(free['clipped'].any())
        self.assertTrue(np.array_equal(late['loss'][:6], free['loss'][:6]))
        self.assertTrue(np.array_equal(late['grad_norm'][:6],
                                       free['grad_norm'][:6]))
        self.assertGreater(abs(late['loss'].iloc[6] - free['loss'].iloc[6]),
                           1e-6)
        # pretrain_step without a step: clipping from cfg.clip, as before
        m = toy_model()
        opt = torch.optim.AdamW(d2l.param_groups(m, 0.1), lr=1e-3)
        b = self.stream().batch(0)
        n = (b['labels'] != -100).sum()
        out = d2l.pretrain_step(m, opt, [b], toy_config(clip=tiny,
                                                         clip_from=5), n)
        self.assertTrue(bool(out['clipped']))
        # the continued-pretraining step clips from the start: it refuses
        # clip_from instead of ignoring it
        with self.assertRaisesRegex(ValueError, 'clip_from'):
            d2l._cpt_step(m, opt, [b], toy_config(clip_from=5), n)
        out = d2l._cpt_step(m, opt, [b], toy_config(clip=tiny), n)
        self.assertTrue(bool(out['clipped']))

    def test_attn_logit_stats(self):
        m = toy_model().eval()
        b = self.stream().batch(0)
        stats = d2l.attn_logit_stats(m, b, max_seqs=2, chunk=7)
        X, ctx = b['input_ids'][:2], d2l.PackCtx(b['doc_ids'][:2],
                                                 b['pos_ids'][:2])
        seen = d2l._attention_inputs(m, X, ctx)
        for l, (attn, Y, tables) in enumerate(seen):
            q, k, _ = attn.qkv(Y, tables=tables)
            k = k.repeat_interleave(2, 1)
            s = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
            d = ctx.doc_ids
            T = X.shape[1]
            ok = (torch.ones(T, T, dtype=torch.bool).tril()
                  & (d[:, :, None] == d[:, None, :]))[:, None]
            s = s.masked_fill(~ok, -math.inf)
            self.assertLess((stats[l].max_logit - s.amax((0, 2, 3))).abs()
                            .max(), 1e-5)
            p = torch.softmax(s, -1)
            ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1).mean((0, 2))
            self.assertLess((stats[l].entropy - ent).abs().max(), 1e-5)

    def test_attn_logit_stats_softcap(self):
        """A softcap model: the entropy is that of the capped softmax the
        model computes, the largest logit is the raw one (before the cap)."""
        torch.manual_seed(0)
        m = d2l.PackedGPT.from_arch(TOY, softcap=2.0, backend='math').eval()
        with torch.no_grad():
            for blk in m.blks:  # raw logits well above the cap
                blk.attention.W_q.weight *= 12
                blk.attention.W_k.weight *= 12
        b = self.stream().batch(0)
        stats = d2l.attn_logit_stats(m, b)
        X, ctx = b['input_ids'][:1], d2l.PackCtx(b['doc_ids'][:1],
                                                 b['pos_ids'][:1])
        T = X.shape[1]
        ok = (torch.ones(T, T, dtype=torch.bool).tril()
              & (ctx.doc_ids[:, :, None] == ctx.doc_ids[:, None, :]))[:, None]

        def entropy(s):
            p = torch.softmax(s.masked_fill(~ok, -math.inf), -1)
            return -(p * torch.log(p.clamp_min(1e-30))).sum(-1).mean((0, 2))
        for l, (attn, Y, tables) in enumerate(
                d2l._attention_inputs(m, X, ctx)):
            with torch.no_grad():
                q, k, _ = attn.qkv(Y, tables=tables)
            k = k.repeat_interleave(2, 1)
            s = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
            raw = s.masked_fill(~ok, -math.inf).amax((0, 2, 3))
            self.assertGreater(float(raw.max()), 4.0)  # the cap matters
            self.assertLess((stats[l].max_logit - raw).abs().max(), 1e-5)
            capped = entropy(d2l.softcap(s, 2.0))
            self.assertLess((stats[l].entropy - capped).abs().max(), 1e-5)
            self.assertGreater((entropy(s) - capped).abs().max(), 1e-2)

    def test_qk_clip(self):
        # one block: clipping a block changes the inputs of the next ones
        torch.manual_seed(0)
        m = d2l.PackedGPT.from_arch(TOY._replace(L=1), backend='sdpa').eval()
        b = self.stream().batch(0)
        with torch.no_grad():
            for blk in m.blks:  # large logits in some heads
                blk.attention.W_q.weight[:8] *= 6
                blk.attention.W_k.weight[:8] *= 3
        stats = d2l.attn_logit_stats(m, b)
        tau = float(min(s.max_logit.max() for s in stats.values())) * 0.6
        before = {n: p.clone() for n, p in m.named_parameters()}
        d2l.qk_clip_(m, stats, tau)
        after = d2l.attn_logit_stats(m, b)
        for l in stats:
            S, S2 = stats[l].max_logit, after[l].max_logit
            hit = S > tau
            self.assertTrue(hit.any())
            # a clipped head's maximum is tau when its key head is not
            # shared with a larger one; never above tau
            self.assertTrue((S2[hit] <= tau * (1 + 1e-5)).all())
            own = hit & (S == S.view(2, -1).amax(1).repeat_interleave(2))
            self.assertLess(((S2[own] - tau).abs() / tau).max(), 1e-5)
            q = m.blks[l].attention.W_q.weight.view(4, 8, -1)
            q0 = before[f'blks.{l}.attention.W_q.weight'].view(4, 8, -1)
            self.assertTrue(torch.equal(q[~hit], q0[~hit]))
        again = {n: p.clone() for n, p in m.named_parameters()}
        d2l.qk_clip_(m, after, tau * (1 + 1e-4))
        for n, p in m.named_parameters():
            self.assertTrue(torch.equal(p, again[n]), n)
        # a head whose largest logit is negative is left alone (run A: the
        # factor (tau/S)^alpha was NaN there and the t1 qk_clip arm at lr
        # 0.032 turned non-finite at step 18 with every logit below tau)
        neg = {l: s._replace(max_logit=-s.max_logit.abs())
               for l, s in after.items()}
        f = d2l.qk_clip_(m, neg, tau)
        self.assertTrue(all(torch.equal(v, torch.ones_like(v))
                            for v in f.values()))
        for n, p in m.named_parameters():
            self.assertTrue(torch.equal(p, again[n]), n)
        torch.manual_seed(0)
        qn = d2l.PackedGPT.from_arch(TOY._replace(qk_norm=True))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(d2l.qk_clip_(qn, stats, tau), {})
        self.assertTrue(w)

    def test_small_functions(self):
        x = torch.linspace(-100, 100, 101)
        y = d2l.softcap(x, 30.0)
        self.assertTrue((y.abs() < 30).all())
        self.assertLess((d2l.softcap(torch.tensor(0.1), 30.0) - 0.1).abs(),
                        1e-5)
        b = self.stream().batch(0)
        for kind in ('random', 'repeat'):
            bad = d2l.make_bad_batch(b, kind, vocab=512, seed=1)
            self.assertEqual(bad['input_ids'].shape, b['input_ids'].shape)
            self.assertTrue(torch.equal(bad['labels'][:, :-1],
                                        bad['input_ids'][:, 1:]))
            self.assertTrue((bad['input_ids'] >= 32).all())
            self.assertTrue((bad['doc_ids'] == 0).all())
        self.assertEqual(len(set(d2l.make_bad_batch(b, 'repeat')[
            'input_ids'].flatten().tolist())), 1)
        # random: unchanged by the n-gram kind (fixed ids for seed 1)
        g = torch.Generator().manual_seed(1)
        ref = torch.randint(32, 512, (8, TOY.T + 1), generator=g)
        self.assertTrue(torch.equal(d2l.make_bad_batch(
            b, 'random', vocab=512, seed=1)['input_ids'], ref[:, :-1]))

    def test_make_bad_batch_ngram(self):
        b = d2l.PackedStream(self.shards, TOY.T, 16, seed=1).batch(0)
        bad = d2l.make_bad_batch(b, 'ngram', seed=3, k=8, period=7)
        self.assertEqual(set(bad), set(b))
        for key in b:
            self.assertEqual(bad[key].shape, b[key].shape, key)
            self.assertEqual(bad[key].dtype, b[key].dtype, key)
        hit = [r for r in range(16) if not torch.equal(bad['input_ids'][r],
                                                       b['input_ids'][r])]
        self.assertEqual(len(hit), 8)
        rest = [r for r in range(16) if r not in hit]
        for key in b:  # untouched rows: every field bit-identical
            self.assertTrue(torch.equal(bad[key][rest], b[key][rest]), key)
        X, Y = bad['input_ids'][hit], bad['labels'][hit]
        self.assertTrue(torch.equal(X[:, 7:], X[:, :-7]))  # period 7
        self.assertTrue(torch.equal(Y[:, :-1], X[:, 1:]))  # shifted targets
        self.assertTrue(torch.equal(Y[:, -1], X[:, -7]))
        self.assertTrue((X == X[0]).all())  # one n-gram in every row
        gram = X[0, :7]
        self.assertTrue((gram >= 32).all())
        self.assertTrue(set(gram.tolist()) <= set(b['input_ids'].flatten()
                                                  .tolist()))
        self.assertTrue((bad['doc_ids'][hit] == 0).all())
        self.assertTrue(torch.equal(bad['pos_ids'][hit], torch.arange(
            TOY.T, dtype=b['pos_ids'].dtype).expand(8, -1)))
        # deterministic per seed; k=None replaces every row; period 1
        again = d2l.make_bad_batch(b, 'ngram', seed=3, k=8, period=7)
        self.assertTrue(all(torch.equal(again[k], bad[k]) for k in b))
        every = d2l.make_bad_batch(b, 'ngram', seed=3, period=13)
        self.assertTrue((every['input_ids'] == every['input_ids'][0]).all())
        X = every['input_ids']
        self.assertTrue(torch.equal(X[:, 13:], X[:, :-13]))
        one = d2l.make_bad_batch(b, 'ngram', seed=0, k=1, period=1)
        r = [i for i in range(16) if not torch.equal(one['input_ids'][i],
                                                     b['input_ids'][i])]
        self.assertEqual(len(r), 1)
        self.assertEqual(len(set(one['input_ids'][r[0]].tolist())), 1)
        self.assertTrue(torch.equal(b['input_ids'], d2l.PackedStream(
            self.shards, TOY.T, 16, seed=1).batch(0)['input_ids']))
        self.assertAlmostEqual(d2l.lr_sensitivity(
            [3.0, 2.5, float('nan'), 4.5], 4.0), (0.5 + 0 + 1.5 + 1.5) / 4)

    def test_t_interval(self):
        from scipy import stats
        rng = np.random.default_rng(0)
        for n in range(2, 12):
            x = rng.normal(size=n)
            m, h = d2l.t_interval(x)
            ref = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
            self.assertAlmostEqual(m, x.mean())
            self.assertLess(abs(h - ref), 1e-4)
        m, h = d2l.t_interval(rng.normal(size=20), alpha=0.1)
        self.assertGreater(h, 0)

    def test_packed_to_hf(self):
        torch.manual_seed(0)
        m = d2l.PackedGPT.from_arch(TOY._replace(H=4, H_kv=2)).eval()
        hf = d2l.packed_to_hf(m)
        X = torch.randint(0, 512, (2, 20))
        with torch.no_grad():
            ref = m(X)
            out = hf(X).logits
        self.assertLess((ref - out).abs().max(), 1e-5)


def _ddp_worker(rank, world, shard_dir, out, port):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=str(rank), MASTER_ADDR='127.0.0.1',
                      MASTER_PORT=str(port))
    torch.set_num_threads(2)
    shards = d2l.TokenShards(shard_dir)
    stream = d2l.PackedStream(shards, TOY.T, 8, seed=1, rank=rank,
                              world=world)
    cfg = toy_config(steps=10, ddp=True, micro_batch=2, stats_every=5)
    log = d2l.pretrain(toy_model(), stream, cfg)
    if rank == 0:
        log.to_json(out)
    import torch.distributed as dist
    dist.destroy_process_group()


class DDPTest(unittest.TestCase):
    def test_two_process_gloo(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as d:
            shard_dir = os.path.join(d, 'set')
            shards = dp.fixture_shards(shard_dir, n_docs=600)
            stream = d2l.PackedStream(shards, TOY.T, 8, seed=1)
            ref = d2l.pretrain(toy_model(), stream,
                               toy_config(steps=10, micro_batch=4,
                                          stats_every=5))
            out = os.path.join(d, 'log.json')
            env = dict(os.environ)
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # gloo on the CPU
            try:
                mp.start_processes(_ddp_worker, args=(2, shard_dir, out,
                                                      29500 + os.getpid()
                                                      % 1000),
                                   nprocs=2, start_method='spawn')
            finally:
                os.environ.clear()
                os.environ.update(env)
            log = pd.read_json(out)
        self.assertLess(np.abs(log['loss'].values - ref['loss'].values)
                        .max(), 1e-5)
        # log_z: the mean over both ranks' targets (one stacked all-reduce
        # with the loss), at the stats steps only
        self.assertEqual(log['log_z'].notna().tolist(),
                         ref['log_z'].notna().tolist())
        self.assertEqual(int(log['log_z'].notna().sum()), 2)
        self.assertLess(np.nanmax(np.abs(log['log_z'].values
                                         - ref['log_z'].values)), 1e-5)


def toy_log(steps=600, every=10, layers=2, seed=0):
    """A synthetic run log with the columns of pretrain's."""
    rng = np.random.default_rng(seed)
    step = np.arange(1, steps + 1)
    df = pd.DataFrame(dict(step=step, loss=3 + 1 / np.sqrt(step),
                           grad_norm=1 + 0.1 * rng.random(steps), run='toy',
                           seed=0, job='toy-s0', tokens=step * 1000,
                           mfu=0.3, time=step * 2.0))
    for l in range(layers):
        df[f'max_logit_{l}'] = np.where(step % every == 0, 10 + l + step / 50,
                                        np.nan)
    df['rms_W_q'] = np.where(step % every == 0, 0.2, np.nan)
    df['val_loss'] = np.where(step % 100 == 0, 3.1, np.nan)
    return df


class MeasurementTest(unittest.TestCase):
    """The section's drivers on CPU: tau_rule, the arm summaries, the
    stand-in, the bad-batch stream, the T1 arms and the spike protocol on a
    2-layer toy, the result tuples, the launcher's job names and arguments,
    the ladder's points, the dashboard."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.shards = dp.fixture_shards(os.path.join(cls.tmp.name, 'set'),
                                       n_docs=600)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tau_rule(self):
        log = toy_log()
        # largest logit over steps 300-500: layer 1 at step 500 = 21
        self.assertEqual(d2l.tau_rule(log, 300, 500), 30)
        self.assertEqual(d2l.tau_rule(log, 10, 40), 20)  # 11.8 -> 20

    def test_arm_summary_and_trace(self):
        log = toy_log(100)
        row = d2l._arm_summary(log, 'none', 1e-3, 0)
        self.assertAlmostEqual(row['final_loss'], log['loss'].iloc[-20:].mean())
        self.assertIsNone(row['first_nonfinite'])
        self.assertAlmostEqual(row['max_logit'], 13.0)
        self.assertEqual(row['val_loss'], 3.1)
        log.loc[60:, 'loss'] = np.nan
        row = d2l._arm_summary(log, 'none', 1e-3, 0)
        self.assertEqual((row['final_loss'], row['first_nonfinite']),
                         (math.inf, 61))
        tr = d2l._arm_trace(log, 'none', 1e-3, 0)
        self.assertIn('max_logit_1', tr.columns)
        self.assertEqual(tr['max_logit'].dropna().iloc[0], 11.2)

    def test_stand_in(self):
        rows = [dict(variant=v, lr=lr, seed=s, final_loss=1.0)
                for v in ('none', 'qk_clip', 'qk_norm') for lr in (1e-3, 1.6e-2)
                for s in (0, 1)]
        tab = pd.DataFrame(rows)
        tr = tab.assign(step=1)
        res = d2l.InstabilityRuns(tab, tr, 50, 'T2 run pending; showing the '
                                               'T1 configuration')
        out = d2l._t1_stand_in(res)
        self.assertEqual(out.table[['variant', 'lr']].values.tolist(),
                         [['baseline', 1e-3], ['none', 1.6e-2],
                          ['qk_clip', 1.6e-2]])
        self.assertTrue(out.label.startswith('T2 run pending'))

    def test_as_tuple(self):
        pending = pd.DataFrame()
        pending.attrs['label'] = 'spike-replay pending'
        r = d2l._as_tuple(d2l.SpikeReplay, pending, ('table', 'traces'))
        self.assertEqual(r.label, 'spike-replay pending')
        self.assertTrue(r.table.empty and r.floor is None)
        r = d2l._as_tuple(d2l.SpikeReplay, pending, dict(
            table=['run', 'max_dloss'], traces=[]), dict(floor=math.nan))
        self.assertEqual(list(r.table.columns), ['run', 'max_dloss'])
        self.assertTrue(math.isnan(r.floor))
        R = collections.namedtuple('Tables', 'table traces floor label')
        r = d2l._as_tuple(d2l.SpikeReplay, R(pd.DataFrame({'a': [1]}),
                                             pd.DataFrame(), 0.5, 'x'),
                          ('table', 'traces'))
        self.assertEqual((r.floor, r.label, len(r.table)), (0.5, 'x', 1))

    def test_bad_batch_stream(self):
        s = d2l._BadBatchStream(self.shards, TOY.T, 8, 5, 'repeat', 512)
        plain = d2l.PackedStream(self.shards, TOY.T, 8)
        self.assertTrue(torch.equal(s.batch(3)['input_ids'],
                                    plain.batch(3)['input_ids']))
        bad = s.batch(4)['input_ids']  # log step 5 = stream step 4
        self.assertEqual(len(set(bad.flatten().tolist())), 1)
        skipped = d2l._BadBatchStream(self.shards, TOY.T, 8, 5, 'repeat', 512,
                                      skip=[4])
        self.assertTrue(torch.equal(skipped.batch(4)['input_ids'],
                                    plain.batch(5)['input_ids']))
        none = d2l._BadBatchStream(self.shards, TOY.T, 8, None, 'repeat', 512)
        self.assertTrue(torch.equal(none.batch(4)['input_ids'],
                                    plain.batch(4)['input_ids']))
        # 'ngram' passes k and period through; the other rows are the plain
        # stream's
        ng = d2l._BadBatchStream(self.shards, TOY.T, 8, 5, 'ngram', 512, k=3,
                                 period=4)
        ref = d2l.make_bad_batch(plain.batch(4), 'ngram', 512, seed=5, k=3,
                                 period=4)
        got = ng.batch(4)
        self.assertTrue(all(torch.equal(got[k], ref[k]) for k in ref))
        hit = (got['input_ids'] != plain.batch(4)['input_ids']).any(1)
        self.assertEqual(int(hit.sum()), 3)
        # arguments that would give an arm of another configuration
        for kind, kw in (('n-gram', {}), ('ngrams', {}),
                         ('ngram', dict(period=None)),
                         ('ngram', dict(period=0)),
                         ('ngram', dict(period=14)),
                         ('ngram', dict(period=7, k=0)),
                         ('ngram', dict(period=7, k=9)),
                         ('ngram', dict(period=7, rank=0, world=2))):
            with self.assertRaises(ValueError, msg=(kind, kw)):
                d2l._BadBatchStream(self.shards, TOY.T, 8, 5, kind, 512, **kw)
        with self.assertRaisesRegex(ValueError, 'unknown bad-batch kind'):
            d2l.make_bad_batch(plain.batch(0), 'n-gram')
        d2l._BadBatchStream(self.shards, TOY.T, 8, 5, 'ngram', 512, k=8,
                            period=13)  # the bounds themselves are fine

    def test_clip_from_and_bad_step(self):
        """clip_from counts updates, bad_step log steps: clip_from = bad_step
        - 1 clips the bad step, clip_from = bad_step does not (and the
        spike driver refuses it)."""
        tiny = 1e-3  # every step's norm is above it

        def clipped(clip_from):
            stream = d2l._BadBatchStream(self.shards, TOY.T, 8, 5, 'ngram',
                                         512, k=4, period=3, seed=1)
            cfg = toy_config(steps=7, clip=tiny, clip_from=clip_from)
            with contextlib.redirect_stdout(io.StringIO()):
                log = d2l.pretrain(toy_model(), stream, cfg)
            return log.set_index('step')['clipped']
        on = clipped(4)
        self.assertEqual(on.tolist(), [False] * 4 + [True] * 3)
        self.assertTrue(on[5])  # the bad batch's update is clipped
        self.assertFalse(clipped(5)[5])
        # the spike driver sets clipping per arm (from the state, so the bad
        # step is always clipped) and refuses a configuration that clips
        arms = ((0.95, 6, 8, 'random', 8, True),)
        for kw in (dict(clip=1.0), dict(clip=None, clip_from=4)):
            with self.assertRaisesRegex(ValueError, 'clip=None'):
                d2l._spike(self.toy_cfg(**kw), arms, 3, 8, steps=12,
                           window=6, device='cpu')

    def toy_cfg(self, **kw):
        return toy_config(preset='t1', compile=False, warmup=2, **kw)

    def test_inst_t1(self):
        with mock.patch.object(d2l, '_train_shards', lambda c: self.shards), \
                mock.patch.object(d2l, '_job_root', lambda: self.tmp.name), \
                mock.patch.object(d2l, '_val_fn', lambda c: (lambda m: 1.0)):
            runs = {v: d2l._inst_t1(self.toy_cfg(clip=None), (1e-3, 1e-1),
                                    v, (0, 1), 12, 0.5, device='cpu')
                    for v in ('none', 'qk_clip', 'qk_norm')}
        for r in runs.values():
            self.assertEqual(len(r.table), 4)
            self.assertEqual(set(r.traces['step']), set(range(1, 13)))
            self.assertTrue(r.traces['max_logit'].notna().any())
            self.assertTrue(r.traces['min_entropy'].notna().any())
            self.assertTrue(r.table['min_entropy'].notna().all())
        # the clip pulls every head back to tau after each step
        hi = {v: r.table[r.table['lr'] == 1e-1]['max_logit'].values
              for v, r in runs.items()}
        self.assertTrue((hi['qk_clip'] < 0.2 * hi['none']).all())
        self.assertEqual(runs['none'].zero_layer, 1.0)  # the zero-layer run
        self.assertTrue(math.isnan(runs['qk_norm'].zero_layer))
        merged = d2l._merge_arms(runs)
        self.assertEqual(len(merged.table), 12)
        self.assertEqual(merged.zero_layer, 1.0)

    def test_inst_arm_configs(self):
        cfg = d2l.r2_config('t1', clip=None)
        arms = {v: d2l._inst_arm(cfg, v, 8e-3, 1) for v in d2l._INST_PARTS}
        for v, (c, extra) in arms.items():  # the full t1 schedule
            self.assertEqual((c.steps, c.warmup if v != 'warmup' else 76,
                              c.decay, c.peak_lr, c.seed), (1526, 76, 229,
                                                            8e-3, 1), v)
            self.assertEqual((c.stats_every, c.val_every, c.probe_every,
                              c.out_dir, c.state_steps), (10, 0, 50, None,
                                                          ()), v)
            self.assertEqual(c.arch.qk_norm, v == 'qk_norm', v)
            self.assertEqual(c.clip, 1.0 if v == 'clip' else None, v)
            self.assertEqual(extra, dict(softcap=50.0, backend='math')
                             if v == 'softcap' else {}, v)
            self.assertEqual(c.micro_batch, 16, v)
            self.assertEqual(c.data_seed, 1, v)  # the seed sets the order
        self.assertEqual(arms['warmup'][0].warmup, 305)
        # a pair's later run differs from its grid cell in one field only
        c, _ = d2l._inst_arm(cfg, 'clip', 1e-3, 0)
        diff = {k for k, v in d2l._cfg_json(c).items()
                if d2l._cfg_json(dataclasses.replace(c, eps=1e-6))[k] != v}
        self.assertEqual(diff, {'eps'})
        # the 10x arms and the spike runs: constant rate after the warmup
        c, _ = d2l._inst_arm(cfg, 'none', 6e-3, 0, steps=500)
        self.assertEqual((c.steps, c.decay, c.probe_every, c.warmup),
                         (500, 0, 0, 76))
        c, _ = d2l._inst_arm(cfg, 'none', 6e-3, 1, steps=500)
        self.assertEqual(c.data_seed, cfg.data_seed)  # as before F10
        c, _ = d2l._inst_arm(cfg, 'none', 1e-3, 0, L=0)
        self.assertEqual(c.arch.L, 0)
        with self.assertRaisesRegex(ValueError, 'unknown variant'):
            d2l._inst_arm(cfg, 'qknorm', 1e-3, 0)
        with self.assertRaisesRegex(ValueError, 'unknown variants'):
            d2l.instability_arms(cfg, (1e-3,), ('none', 'clip_1'))
        with self.assertRaisesRegex(ValueError, 'given number of steps'):
            d2l.instability_arms(d2l.r2_config('r2'), (6e-3,), ('none',))
        with self.assertRaisesRegex(ValueError, 'none, qk_clip and qk_norm'):
            d2l._inst_t2(d2l.r2_config('r2'), (6e-3,), ('clip',), 500, 30)
        job = d2l.PRETRAIN_JOBS['inst-t1-warmup'][1]
        with mock.patch.object(d2l, '_job_log', lambda n: toy_log()):
            kw = job()
        self.assertEqual(kw['variants'], ('none', 'qk_norm', 'qk_clip',
                                          'softcap', 'clip', 'warmup'))
        self.assertNotIn('steps', kw)  # the full schedule
        self.assertEqual((kw['seeds'], kw['cfg'].clip), ((0, 1), None))

    def test_zero_layer_model(self):
        arch = TOY._replace(L=0)
        model = d2l.PackedGPT.from_arch(arch, backend='sdpa')
        self.assertEqual(len(model.blks), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         TOY.vocab * TOY.d + TOY.d)  # embedding, final norm
        stream = d2l.PackedStream(self.shards, TOY.T, 8)
        cfg = toy_config(arch=arch, steps=6, stats_every=2, probe_every=2)
        before = model.token_emb.weight.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            log = d2l.pretrain(model, stream, cfg, val_fn=lambda m: 2.0)
        self.assertTrue(np.isfinite(log['loss']).all())
        self.assertTrue(np.isfinite(log['grad_norm']).all())
        self.assertFalse(torch.equal(model.token_emb.weight, before))
        self.assertFalse(any(c.startswith(('max_logit_', 'entropy_'))
                             for c in log))
        row = d2l._arm_summary(log, 'none', 1e-3, 0)
        self.assertTrue(math.isnan(row['min_entropy']))
        self.assertEqual(row['val_loss'], 2.0)

    def test_arm_summary_entropy(self):
        log = toy_log(100)  # largest logit 10 + l + step / 50
        for l in range(2):
            log[f'entropy_{l}'] = np.where(log['step'] % 10 == 0,
                                           3 - log['step'] / 40 - l, np.nan)
        log.loc[log['step'] == 70, 'max_logit_1'] = 2000.0
        row = d2l._arm_summary(log, 'none', 1e-3, 0, peak_end=55)
        self.assertEqual(row['first_logit_1000'], 70)
        self.assertAlmostEqual(row['min_entropy'], 3 - 100 / 40 - 1)
        self.assertAlmostEqual(row['peak_entropy'], 3 - 50 / 40 - 1)
        row = d2l._arm_summary(log, 'none', 1e-3, 0)
        self.assertAlmostEqual(row['peak_entropy'], row['min_entropy'])
        self.assertIsNone(d2l._arm_summary(toy_log(100), 'none', 1e-3, 0)[
            'first_logit_1000'])
        tr = d2l._arm_trace(log.assign(log_z=1.0), 'none', 1e-3, 0)
        for c in ('entropy_0', 'entropy_1', 'min_entropy', 'log_z',
                  'rms_W_q'):
            self.assertIn(c, tr.columns)
        self.assertAlmostEqual(tr.loc[tr['step'] == 40, 'min_entropy'].item(),
                               3 - 1 - 1)

    def test_first_collapsed_rate(self):
        ent = {'none': (2.0, 0.05, 0.0), 'qk_norm': (2.0, 1.5, 1.2)}
        rows = [dict(variant=v, lr=lr, seed=s, min_entropy=e)
                for v in ent for s in (0, 1)
                for lr, e in zip((1e-3, 8e-3, 3.2e-2), ent[v])]
        arms = pd.DataFrame(rows)
        self.assertEqual(d2l.first_collapsed_rate(arms, 'none'), 8e-3)
        self.assertIsNone(d2l.first_collapsed_rate(arms, 'qk_norm'))
        arms.loc[(arms['lr'] == 8e-3) & (arms['seed'] == 0),
                 'min_entropy'] = 0.5
        self.assertEqual(d2l.first_collapsed_rate(arms, 'none', seed=0),
                         3.2e-2)
        self.assertEqual(d2l.first_collapsed_rate(arms, 'none'), 8e-3)
        self.assertIsNone(d2l.first_collapsed_rate(arms, 'none', floor=0.0))

    def test_departure(self):
        rng = np.random.default_rng(0)
        step = np.arange(1, 1501)
        x = pd.Series(0.3 + 0.05 * rng.standard_normal(1500), index=step)
        x[step < 76] = np.linspace(2.0, 0.3, 75)  # a warmup ramp: ignored
        self.assertIsNone(d2l.departure(x, 76))
        y = x.where(step < 900, x + 1.0)  # a sustained shift at step 900
        self.assertTrue(900 <= d2l.departure(y, 76) <= 912)
        z = x.copy()
        z[[500, 501, 700]] = 50.0  # single spikes: the median ignores them
        self.assertIsNone(d2l.departure(z, 76))
        y[y.index > 1000] = np.nan  # a stopped run: its missing tail
        self.assertTrue(900 <= d2l.departure(y, 76) <= 912)
        # never inside the reference steps 76-275
        early = x.where(step < 200, x + 1.0)
        self.assertGreaterEqual(d2l.departure(early, 76) or 276, 276)

    def test_update_rms_by_phase(self):
        cfg = toy_config(steps=600, warmup=50, decay=100, peak_lr=1e-3)
        step = np.arange(1, 601)
        rng = np.random.default_rng(0)
        lr = [d2l.wsd_lr(s - 1, cfg.steps, cfg.warmup, cfg.decay,
                         cfg.peak_lr) for s in step]
        probe = step % 10 == 0
        log = pd.DataFrame(dict(step=step, lr=lr, loss=3.0))
        for k, base in (('W_q', 0.2), ('W_v', 0.15)):
            val = base + 0.01 * rng.random(600) + 0.02 * (step > 500)
            log[f'rms_{k}'] = np.where(probe, val, np.nan)
        got = d2l.update_rms_by_phase(log, cfg, 100)
        self.assertEqual(sorted(got.index), ['W_q', 'W_v'])
        run = log.set_index('step')
        for k in ('W_q', 'W_v'):
            for phase, rows in (('stable', range(100, 501, 10)),
                                ('decay', range(510, 601, 10))):
                rows = list(rows)
                med = run.loc[rows, f'rms_{k}'].median()
                self.assertAlmostEqual(got.at[k, phase], med)
                # the absolute update = per-lr median x the median lr
                self.assertAlmostEqual(
                    got.at[k, f'abs_{phase}'],
                    med * run.loc[rows, 'lr'].median())
        self.assertAlmostEqual(got.at['W_q', 'abs_stable'],
                               got.at['W_q', 'stable'] * 1e-3)
        # per lr the decay rises here; the update itself falls
        self.assertTrue((got['decay'] > got['stable']).all())
        self.assertTrue((got['abs_decay'] < got['abs_stable']).all())
        with self.assertRaises(AssertionError):  # no probe in the decay
            d2l.update_rms_by_phase(log[log['step'] <= 500], cfg, 100)

    def test_moved(self):
        rng = np.random.default_rng(0)
        step = np.arange(1, 6001)
        every10 = step % 10 == 0
        # white noise: the same false-alarm rate dense and every 10 steps
        # (the grid of 10-step blocks makes it one detector)
        rate = {}
        for name, keep in (('dense', step > 0), ('every 10', every10)):
            fires = 0
            for _ in range(20):
                x = pd.Series(rng.standard_normal(6000), index=step)[keep]
                fires += sum(d2l.moved(x, t) is not None
                             for t in range(300, 5700, 270))
            rate[name] = fires / 400
        self.assertLess(max(rate.values()), 0.12)
        self.assertLess(abs(rate['dense'] - rate['every 10']), 0.05)
        # a convex, flattening curve with no change (the loss's shape):
        # quiet at stable-phase steps, dense or sparse, reproducibly ...
        curve = pd.Series(2.71 + 28 / np.sqrt(step)
                          + 0.025 * rng.standard_normal(6000), index=step)
        null = np.linspace(1200, 4900, 20).round().astype(int)
        for x in (curve, curve[every10]):
            lags = [d2l.moved(x, int(t)) for t in null]
            self.assertEqual(lags, [d2l.moved(x, int(t)) for t in null])
            self.assertLessEqual(sum(m is not None for m in lags), 2)
        # ... but a line fitted inside its steep start misfits and fires:
        # why the page reads no event whose reference lies in warmup
        self.assertIsNotNone(d2l.moved(curve, 300))
        # a sustained shift 30 steps after the event: found within the
        # rolling window, on dense and on every-10-steps series
        jump = curve + np.where(step >= 2030, 0.2, 0.0)
        self.assertTrue(30 <= d2l.moved(jump, 2000) <= 60)
        self.assertTrue(30 <= d2l.moved(jump[every10], 2000) <= 60)
        self.assertIsNone(d2l.moved(curve, 2000))
        # a change beyond the horizon is not counted
        late = curve + np.where(step >= 2300, 0.2, 0.0)
        self.assertIsNone(d2l.moved(late, 2000, horizon=200))
        # single spikes: the block medians ignore them
        spiky = curve.copy()
        spiky[[2010, 2100]] = 50.0
        self.assertIsNone(d2l.moved(spiky, 2000))

    def test_collapse_crossings(self):
        rng = np.random.default_rng(0)
        step = np.arange(1, 1001)
        stats = step % 10 == 0
        twin = pd.DataFrame(dict(
            step=step, loss=3 + 0.01 * rng.standard_normal(1000),
            grad_norm=0.5 * np.exp(0.05 * rng.standard_normal(1000)),
            max_logit=np.where(stats, 10.0, np.nan),
            min_entropy=np.where(stats, 2.0, np.nan), variant='qk_norm'))
        run = twin.assign(variant='none')
        run['loss'] = twin['loss'] + np.where(step >= 600, 0.3, 0.0)
        run['grad_norm'] = twin['grad_norm'] * np.where(step >= 500, 3, 1)
        run['max_logit'] = np.where(stats, np.where(step >= 300, 2e3, 50),
                                    np.nan)
        run['min_entropy'] = np.where(stats, np.where(step >= 350, .05, 1),
                                      np.nan)
        other = run.assign(seed=1, loss=run['loss'] + 1)  # another seed
        traces = pd.concat([run, twin], ignore_index=True).assign(
            lr=8e-3, seed=0)
        traces = pd.concat([traces, other.assign(lr=8e-3)],
                           ignore_index=True)
        table = pd.DataFrame([dict(variant='none', lr=8e-3, seed=0,
                                   first_nonfinite=math.nan)])
        inst = d2l.InstabilityRuns(table, traces, 30, '')
        got = d2l.collapse_crossings(inst, 8e-3, 76)
        self.assertEqual(got['largest logit > 1,000'], 300)
        self.assertEqual(got['entropy < 0.1'], 350)
        self.assertTrue(500 <= got['grad norm moves'] <= 512)
        self.assertTrue(600 <= got['loss 0.1 above QK-norm'] <= 620)
        # a run that diverges at step 450 and stops: no rolling crossing,
        # the first non-finite step is the loss crossing
        stop = traces[(traces['variant'] == 'qk_norm')
                      | (traces['step'] < 450)].copy()
        stop.loc[(stop['variant'] == 'none') & (stop['step'] == 449),
                 'loss'] = np.nan
        inst = d2l.InstabilityRuns(table.assign(first_nonfinite=449.0),
                                   stop, 30, '')
        got = d2l.collapse_crossings(inst, 8e-3, 76)
        self.assertEqual(got['loss 0.1 above QK-norm'], 449)
        self.assertIsNone(got['grad norm moves'])  # stopped before 500

    def test_inst_kept(self):
        path = os.path.join(self.tmp.name, 'kept', 'arm.json.gz')
        calls = []

        def compute():
            calls.append(1)
            return (dict(variant='none', lr=1e-3, seed=0,
                         final_loss=math.inf, first_nonfinite=None),
                    toy_log(20))
        row, tr = d2l._inst_kept(path, dict(a=1), compute)
        again, tr2 = d2l._inst_kept(path, dict(a=1), compute)
        self.assertEqual(len(calls), 1)  # resumed: read, not trained
        self.assertTrue(pd.isna(row['final_loss']))  # inf is stored as null
        self.assertEqual(str(row), str(again))
        pd.testing.assert_frame_equal(tr, tr2)
        d2l._inst_kept(path, dict(a=2), compute)  # another configuration
        self.assertEqual(len(calls), 2)

    def test_launch_jobs_arguments(self):
        calls = []

        def call(cmd, cwd, stdout, stderr):
            calls.append(cmd)
            job = os.path.dirname(cmd[cmd.index('--config') + 1])
            with open(os.path.join(job, 'log.jsonl'), 'w') as f:
                f.write('{"step": 1, "loss": 1.0}\n')
            open(os.path.join(job, 'DONE'), 'w').close()
            return 0
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(d2l, '_job_root', lambda: root), \
                mock.patch.object(d2l, '_train_shards', lambda c: None), \
                mock.patch('subprocess.call', call):
            cfgs = [d2l.r2_config('t1'), d2l.r2_config('t1', seed=1)]
            df = d2l._launch_jobs(cfgs, 1, jobs=['a', 'b'],
                                  extra=[['--qk-clip', 30], []])
        self.assertEqual(df['job'].tolist(), ['a', 'b'])
        self.assertEqual(calls[0][-2:], ['--qk-clip', '30'])
        self.assertEqual(calls[1][-2:], ['--resume', 'auto'])

    def test_ladder_points(self):
        logs = pd.DataFrame(dict(job=['ladder-25m-s0', 'r2-s0']))
        with mock.patch.object(d2l, '_job_root', lambda: '/r'):
            pts = d2l._ladder_points(logs)
        fin = [p for p in pts if p[3]]
        self.assertEqual(len(fin), 6)
        self.assertEqual(len(pts) - len(fin), 4)
        self.assertIn(('r2-s0', 'b3000', 3000, True,
                       '/r/r2-s0/ckpt/b3000-s3000.safetensors'), pts)
        self.assertIn(('ladder-25m-s0', 'ladder-25m', 816, False,
                       '/r/ladder-25m-s0/state/s816.pt'), pts)

    def test_gradient_event(self):
        log = toy_log(1000)
        log.loc[log['step'] == 600, 'grad_norm'] = 20.0  # a 18x event
        ev = d2l.gradient_event(log)
        self.assertEqual((ev.step, ev.at, ev.mild), (600, 600, False))
        self.assertEqual(ev.source, 'toy run')
        self.assertGreater(ev.ratio, 15)
        self.assertAlmostEqual(ev.hour, (1200 - 2) / 3600)
        self.assertEqual(ev.layer, 2)  # layer 1 (0-based) is the larger
        self.assertAlmostEqual(ev.max_logit, 11 + 600 / 50)
        loss = log.set_index('step')['loss']
        self.assertAlmostEqual(ev.dloss, loss.loc[601:800].mean()
                               - loss.loc[401:600].mean())
        self.assertIsNone(ev.nonfinite)
        # mild: the 10x arm's first step above the baseline + delta
        base, arm = toy_log(500), toy_log(500)
        arm.loc[arm['step'] >= 300, 'loss'] += 0.5
        arm.loc[arm['step'] >= 350, 'loss'] = np.nan
        traces = pd.concat([d2l._arm_trace(base, 'baseline', 6e-4, 0),
                            d2l._arm_trace(arm, 'none', 6e-3, 0)])
        arms = d2l.InstabilityRuns(pd.DataFrame(), traces, 30, '')
        with mock.patch.object(d2l, 'instability_arms',
                               lambda *a, **k: arms):
            ev = d2l.gradient_event(toy_log(1000))
        self.assertTrue(ev.mild)
        self.assertEqual((ev.at, ev.nonfinite), (300, 350))
        self.assertEqual(ev.source, 'unclipped arm at lr 0.006')
        empty = d2l.InstabilityRuns(pd.DataFrame(), pd.DataFrame(
            columns=d2l._INST_TABLES['traces']), None, 'inst-t2 pending')
        with mock.patch.object(d2l, 'instability_arms',
                               lambda *a, **k: empty), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            ev = d2l.gradient_event(toy_log(1000))
        self.assertTrue(ev.mild and ev.at is None)
        self.assertEqual(out.getvalue(), '[inst-t2 pending]\n')

    def test_reference_evals_stand_in(self):
        tab = pd.DataFrame(dict(step=d2l.T1_STEPS, val_loss=1.0, bpb=1.0,
                                piqa=0.5))
        Tables = collections.namedtuple('Tables', 'table items domains '
                                                  'chance label')
        stored = Tables(tab, tab[['step']].assign(task='piqa', item_id=0,
                                                  score=1),
                        pd.DataFrame(), {'piqa': 0.5}, 'T2 run pending; '
                        'showing the T1 configuration')
        with mock.patch.object(d2l, '_pretrain_result',
                               lambda *a, **k: stored), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            ev = d2l.reference_evals(d2l.REFERENCE_STEPS, ('piqa',))
        self.assertEqual(ev.table['step'].tolist(), list(d2l.REFERENCE_STEPS))
        self.assertEqual(ev.items['step'].iloc[-1], 6000)
        self.assertEqual(ev.label, d2l.T1_STAND_IN)
        self.assertIn(d2l.T1_STAND_IN, out.getvalue())
        pending = pd.DataFrame()
        pending.attrs['label'] = 'reference-evals pending'
        with mock.patch.object(d2l, '_pretrain_result',
                               lambda *a, **k: pending), \
                contextlib.redirect_stdout(io.StringIO()):
            ev = d2l.reference_evals(d2l.REFERENCE_STEPS, ('piqa',))
        self.assertEqual(list(ev.table.columns), ['step', 'val_loss', 'bpb',
                                                  'piqa'])
        self.assertTrue(math.isnan(ev.chance['piqa']))

    def test_load_reference_models_stand_in(self):
        calls = []

        def load(step, dtype, device, preset='r2'):
            calls.append((step, preset))
            return preset
        with mock.patch.object(d2l, 'load_reference_model', load), \
                mock.patch.object(d2l, 'PRETRAIN_PENDING',
                                  frozenset({'reference-run'})), \
                uncommitted(), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            models = d2l.load_reference_models((5100, 6000))
        self.assertEqual(calls, [(1297, 't1'), (1526, 't1')])
        self.assertEqual(set(models), {5100, 6000})
        self.assertIn('T2 run pending', out.getvalue())

    def test_plot_traces_and_ladder(self):
        import matplotlib
        matplotlib.use('Agg')
        traces = pd.concat([d2l._arm_trace(toy_log(100), v, 1e-3, 0)
                            for v in ('none', 'qk_clip')])
        d2l.plot_traces(traces, ['max_logit', 'loss'], by=['lr', 'variant'],
                        logy=['max_logit'], steps=(10, 90))
        fig = d2l.plt.gcf()
        self.assertEqual(len(fig.axes), 2)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        self.assertEqual(labels, ['0.001 none', '0.001 qk_clip'])
        d2l.plt.close('all')
        d2l.plot_traces(traces, ['loss'], by='variant', dashed=['qk_clip'])
        styles = [ln.get_linestyle() for ln in d2l.plt.gcf().axes[0].lines]
        self.assertEqual(styles, ['-', '--'])
        d2l.plt.close('all')
        d2l.plot_traces(traces, ['max_logit', 'loss'], by='variant',
                        refs={'loss': 0.1})  # a threshold on one panel
        top, bottom = d2l.plt.gcf().axes
        self.assertEqual(len(top.lines), 2)  # no line where no ref
        ref = bottom.lines[-1]
        self.assertEqual((list(ref.get_ydata()), ref.get_linestyle()),
                         ([0.1, 0.1], ':'))
        d2l.plt.close('all')
        arms = pd.DataFrame([dict(variant=v, lr=lr, seed=s, loss=3 + lr + s,
                                  max_logit=10 * lr)
                             for v in ('none', 'qk_norm')
                             for lr in (1e-3, 1e-2) for s in (0, 1)])
        d2l.plot_lr_grid(arms, {'loss': ('loss', 3.5),
                                'max_logit': ('logit', None)},
                         ('none', 'qk_norm'), logy=['max_logit'])
        fig = d2l.plt.gcf()
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(len(fig.axes[0].lines), 5)  # 2 means, 2 seeds, ref
        self.assertEqual(len(fig.axes[1].lines), 4)
        self.assertEqual(fig.axes[1].get_yscale(), 'log')
        labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
        self.assertEqual(labels, ['none', 'qk_norm'])  # beside the last
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(),
                                   [3.5 + 1e-3, 3.5 + 1e-2])  # seed mean
        d2l.plt.close('all')
        log = toy_log(200).assign(job='ladder-25m-s0')
        log.loc[log['step'] > 150, 'run'] = 'b150'
        empty = pd.DataFrame()
        d2l.plot_ladder_runs(empty, log)
        self.assertEqual(len(d2l.plt.gca().get_lines()), 2)
        d2l.plt.close('all')

    def test_plot_run_log(self):
        import matplotlib
        matplotlib.use('Agg')
        d2l.plot_run_log(toy_log(), ['loss', 'grad_norm', 'rms', 'max_logit',
                                     'mfu'], marks=(30, 500))
        fig = d2l.plt.gcf()
        self.assertEqual(len(fig.axes), 5)
        d2l.plt.close('all')


class ResultTest(unittest.TestCase):
    """_pretrain_result and _pretrain_merged with a temporary registry."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        chapter = os.path.join(self.tmp.name, 'chapter')
        os.makedirs(chapter)
        self.cwd = os.getcwd()
        os.chdir(chapter)
        self.patches = [
            uncommitted(),
            mock.patch.object(d2l, 'PRETRAIN_PENDING', frozenset()),
            mock.patch.dict(os.environ, {'D2L_RERUN': ''})]
        for p in self.patches:
            p.start()
        self.calls = 0

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def compute(self):
        self.calls += 1
        df = pd.DataFrame(dict(T=[1024, 4096], tok_s=[1.5e5, 1.2e5]))
        df.attrs['label'] = 't1 preset'
        return df

    def result(self, config=None, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return d2l._pretrain_result('pdp-throughput', self.compute,
                                        config or dict(T=[1024, 4096]), **kw)

    def register(self):
        path = os.path.join('..', 'data', 'lm', 'rerun',
                            'lm-pdp-throughput-v1.parquet')
        d2l.DATA_HUB['lm-pdp-throughput-v1'] = ('url', 'sha1')
        return mock.patch.object(d2l, 'download_keyed', lambda key: path)

    def test_regenerate_then_load(self):
        os.environ['D2L_RERUN'] = 'pdp-throughput'
        df = self.result()
        self.assertEqual(self.calls, 1)
        self.assertEqual(df.attrs['label'], 't1 preset')
        path = '../data/lm/rerun/lm-pdp-throughput-v1.parquet'
        self.assertTrue(os.path.exists(path))
        os.environ['D2L_RERUN'] = ''
        with self.register():
            df = self.result()
            self.assertEqual(self.calls, 1)
            self.assertEqual(df['T'].tolist(), [1024, 4096])
            self.assertEqual(df.attrs['label'], 't1 preset')
            self.assertEqual(df.attrs['config'], dict(T=[1024, 4096]))
            with self.assertRaises(ValueError) as cm:
                self.result(dict(T=[1024, 8192]))
            self.assertIn('page configuration differs from the committed run '
                          'lm-pdp-throughput-v1: T[1] 8192 != 4096',
                          str(cm.exception))
            # a regeneration of a registered key writes the next version
            os.environ['D2L_RERUN'] = 'pdp-throughput'
            self.result()
            self.assertTrue(os.path.exists(
                '../data/lm/rerun/lm-pdp-throughput-v2.parquet'))

    def test_absent_and_pending(self):
        with self.assertRaises(KeyError):
            self.result()
        with mock.patch.object(d2l, 'PRETRAIN_PENDING',
                               frozenset({'pdp-throughput'})):
            df = self.result()
        self.assertEqual(df.attrs['label'], 'pdp-throughput pending')
        os.environ['D2L_RERUN'] = 't1-runs,ladder'
        df = self.result()  # another name in the same process: no error
        self.assertEqual(self.calls, 0)
        self.assertEqual(df.attrs['label'], 'pdp-throughput pending')
        os.environ['D2L_RERUN'] = '1'
        with self.assertRaises(ValueError) as cm:
            self.result()
        self.assertIn('t1-runs', str(cm.exception))

    def test_compute_incomplete(self):
        os.environ['D2L_RERUN'] = 'pdp-throughput'
        with contextlib.redirect_stdout(io.StringIO()):
            df = d2l._pretrain_result('pdp-throughput', lambda: None, {})
        self.assertEqual(df.attrs['label'], 'pdp-throughput pending')
        self.assertFalse(os.path.exists('../data/lm/rerun'))

    def test_json_table_and_jsonl(self):
        for kind in ('json-table', 'jsonl'):
            path = os.path.join(self.tmp.name, 'x.' + kind)
            df = self.compute()
            if kind == 'json-table':
                import collections
                R = collections.namedtuple('R', 'table extra label')
                d2l._write_result(path, kind, R(df, 3, 'lbl'),
                                  dict(config={'a': 1}, label='lbl'))
                r = d2l._read_result(path, kind)
                self.assertEqual((r.extra, r.label), (3, 'lbl'))
                self.assertEqual(r.table['T'].tolist(), [1024, 4096])
                self.assertEqual(d2l._stored_config(r), {'a': 1})
            else:
                d2l._write_result(path, kind, df, dict(config={'a': 1},
                                                       label='lbl'))
                r = d2l._read_result(path, kind)
                self.assertEqual(r['tok_s'].tolist(), [1.5e5, 1.2e5])
                self.assertEqual(r.attrs['config'], {'a': 1})

    def test_merged(self):
        parts = {n: (self.compute, dict(arm=n)) for n in (
            'cpt-qwen-A', 'cpt-qwen-B', 'cpt-qwen-C', 'cpt-qwen-A2')}

        def merge(results):
            return pd.concat(results.values(), ignore_index=True)

        def call():
            with contextlib.redirect_stdout(io.StringIO()):
                return d2l._pretrain_merged('cpt-qwen-evals', parts, merge,
                                            dict(arms=4))
        os.environ['D2L_RERUN'] = 'cpt-qwen-A,cpt-qwen-B'
        self.assertEqual(call().attrs['label'], 'cpt-qwen-evals pending')
        self.assertEqual(self.calls, 2)
        os.environ['D2L_RERUN'] = 'cpt-qwen-C,cpt-qwen-A2,cpt-qwen-evals'
        df = call()
        self.assertEqual(self.calls, 4)
        self.assertEqual(len(df), 8)
        name = d2l.PRETRAIN_ARTIFACTS['cpt-runs'][0]
        self.assertTrue(os.path.exists(f'../data/lm/rerun/{name}.json.gz'))
        os.environ['D2L_RERUN'] = ''
        with self.assertRaises(KeyError):
            call()

    def test_merged_last_part_merges(self):
        """Parts run side by side: the one that finishes last merges, even
        when the others' files appear while it computes (run A: smollm3-5
        started before smollm3-4 finished and left the merge pending)."""
        names = [f'smollm3-{i}' for i in range(3)]
        key = d2l.PRETRAIN_ARTIFACTS['smollm3-trajectory'][0]
        d2l.DATA_HUB.pop(key, None)  # a copy

        def call(compute):
            parts = {n: (compute, dict(name=n)) for n in names}
            with contextlib.redirect_stdout(io.StringIO()):
                return d2l._pretrain_merged(
                    None, parts, lambda r: pd.concat(r.values(),
                                                     ignore_index=True),
                    dict(names=names))
        os.environ['D2L_RERUN'] = 'smollm3-0'
        self.assertTrue(call(self.compute).attrs['label'].endswith('pending'))

        def late():  # smollm3-1 finishes while smollm3-2 runs
            os.environ['D2L_RERUN'] = 'smollm3-1'
            call(self.compute)
            os.environ['D2L_RERUN'] = 'smollm3-2'
            return self.compute()
        os.environ['D2L_RERUN'] = 'smollm3-2'
        self.assertEqual(len(call(late)), 6)
        merged = f'../data/lm/rerun/{key}.json.gz'
        self.assertTrue(os.path.exists(merged))
        os.remove(merged)
        # --merge: the named part's file exists, so nothing is computed
        calls = self.calls
        with mock.patch.dict(os.environ, {'D2L_PRETRAIN_MERGE': '1',
                                          'D2L_RERUN': 'smollm3-0'}):
            self.assertEqual(len(call(self.compute)), 6)
        self.assertEqual(self.calls, calls)
        self.assertTrue(os.path.exists(merged))


    def test_inst_t1_regeneration(self):
        """Each variant is its own regeneration; the merged result keeps the
        zero-layer loss of the 'none' part and stores the recipe."""
        cfg = d2l.r2_config('t1', clip=None)

        def fake(cfg, lrs, v, seeds, steps, tau):
            self.calls += 1
            rows = [dict(variant=v, lr=lr, seed=s, val_loss=3.0)
                    for lr in lrs for s in seeds]
            return d2l.InstabilityRuns(
                pd.DataFrame(rows), pd.DataFrame(rows).assign(step=1), tau,
                '', 7.5 if v == 'none' else math.nan)

        def call():
            with contextlib.redirect_stdout(io.StringIO()):
                return d2l.instability_arms(cfg, (1e-3, 8e-3),
                                            ('none', 'clip'), seeds=(0, 1),
                                            tau=30)
        with mock.patch.object(d2l, '_inst_t1', fake):
            os.environ['D2L_RERUN'] = 'inst-t1-clip'
            self.assertTrue(call().label.endswith('pending'))
            os.environ['D2L_RERUN'] = 'inst-t1-none'
            res = call()
        self.assertEqual(self.calls, 2)
        self.assertEqual((len(res.table), res.zero_layer, res.tau),
                         (8, 7.5, 30))
        path = '../data/lm/rerun/lm-r2-inst-t1-v2.json.gz'
        stored = d2l._read_result(path, 'json-table')
        config = d2l._stored_config(stored)
        self.assertEqual((config['steps'], config['variants']),
                         (None, ['none', 'clip']))
        self.assertEqual((config['recipe']['warmup'],
                          config['recipe']['stats']), (0.2, 2))
        self.assertEqual(stored.zero_layer, 7.5)
        # the committed grid loads; a changed recipe is refused
        d2l.DATA_HUB['lm-r2-inst-t1-v2'] = ('url', 'sha1')
        os.environ['D2L_RERUN'] = ''
        with mock.patch.object(d2l, 'download_keyed', lambda key: path):
            self.assertEqual(call().zero_layer, 7.5)
            with mock.patch.object(d2l, 'INST_CLIP', 0.5), \
                    self.assertRaisesRegex(ValueError, 'recipe.clip'):
                call()
            with mock.patch.object(d2l, 'INST_STATS', 3), \
                    self.assertRaisesRegex(ValueError, 'recipe.stats'):
                call()

    def test_size_transfer_regeneration(self):
        """The pilot writes the result (its 8 runs, the gate); a normal run
        loads it under the page's configuration; after a go the full job
        (its name added) computes every run; another regeneration run
        shows the pending result."""
        calls = []

        def fake(shapes, lrs, seeds, steps, full):
            calls.append(full)
            rows = [dict(shape='t1', N=1.0, lr=4e-3, seed=0, logit=2.0,
                         loss=4.0, first_nonfinite=None)]
            fit = pd.DataFrame(columns=d2l._XFER_TABLES['fit'])
            return d2l.SizeTransfer(pd.DataFrame(rows), pd.DataFrame(
                dict(step=[300], loss=[4.0], max_logit=[2.0])), fit,
                'no-go', None, '')
        kw = d2l._size_xfer_job()

        def call(**over):
            with contextlib.redirect_stdout(io.StringIO()):
                return d2l.size_transfer(**dict(kw, **over))
        path = '../data/lm/rerun/lm-r2-size-xfer-v1.json.gz'
        before_go = tuple(n for n in d2l.PRETRAIN_RERUN_NAMES
                          if n != 'size-xfer')  # the names before the go
        with mock.patch.object(d2l, '_size_transfer', fake):
            os.environ['D2L_RERUN'] = 'size-xfer-pilot'
            with self.assertRaisesRegex(ValueError, 'regenerate size-xfer'):
                call()  # after the go the pilot alone is refused
            self.assertEqual(calls, [])
            with mock.patch.object(d2l, 'PRETRAIN_RERUN_NAMES', before_go):
                res = call()
            self.assertEqual((calls, res.gate, len(res.table)),
                             ([False], 'no-go', 1))
            self.assertTrue(os.path.exists(path))
            config = d2l._stored_config(d2l._read_result(path, 'json-table'))
            self.assertEqual(config['recipe']['pilot'],
                             dict(shapes=['t1', 'ladder-88m'],
                                  lrs=[4e-3, 8e-3]))
            self.assertEqual(config['runs']['r2']['arch']['L'], 18)
            self.assertNotIn('peak_lr', config['runs']['t1'])
            os.environ['D2L_RERUN'] = 't1-runs'  # another job's run
            res = call()
            self.assertTrue(res.label.endswith('pending'))
            self.assertIsNone(res.gate)
            self.assertEqual(list(res.fit.columns), d2l._XFER_TABLES['fit'])
            # after a go: the full job (its name was added after the
            # pilot's go, spec 5.3)
            os.environ['D2L_RERUN'] = 'size-xfer'
            call()
            self.assertEqual(calls, [False, True])
        os.environ['D2L_RERUN'] = ''
        d2l.DATA_HUB['lm-r2-size-xfer-v1'] = ('url', 'sha1')
        with mock.patch.object(d2l, 'download_keyed', lambda key: path):
            res = call()
            self.assertEqual((res.gate, res.label), ('no-go', ''))
            with self.assertRaisesRegex(ValueError, 'steps 400 != 300'):
                call(steps=400)
            with mock.patch.object(d2l, 'XFER_PILOT', dict(
                    shapes=('t1', 'ladder-54m'), lrs=(4e-3, 8e-3))), \
                    self.assertRaisesRegex(ValueError, 'recipe.pilot'):
                call()


class SizeTransferTest(unittest.TestCase):
    """The size transfer of the largest logit (cell 25c) on CPU."""
    SHAPES = ('t1', 'ladder-54m', 'ladder-88m', 'r2')
    LRS = (2e-3, 4e-3, 8e-3, 1.6e-2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, 'ch21')

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def points(Ns, slope=0.7, a=-5.0, sigma=0.0, seeds=(0, 1), rng=None,
               lrs=(1e-3,)):
        rows = []
        for lr in lrs:
            for N in Ns:
                for s in seeds:
                    e = rng.normal(0, sigma) if sigma else 0.0
                    rows.append(dict(lr=lr, seed=s, N=N, logit=math.exp(
                        a + slope * math.log(N) + e)))
        return pd.DataFrame(rows)

    def test_t_quantile(self):
        from scipy import stats
        for dof in range(1, 16):
            for level in (0.95, 0.9):
                self.assertAlmostEqual(d2l.t_quantile(dof, level), float(
                    stats.t.ppf((1 + level) / 2, dof)), delta=1e-4)

    def test_logit_fit_exact(self):
        Ns = (12.6e6, 35.4e6, 62.9e6)
        fit = d2l.logit_fit(self.points(Ns, lrs=(2e-3, 4e-3)), at=113.2e6)
        self.assertEqual(list(fit.columns), d2l._XFER_TABLES['fit'])
        self.assertEqual(fit['lr'].tolist(), [2e-3, 4e-3])
        truth = math.exp(-5 + 0.7 * math.log(113.2e6))
        for r in fit.itertuples():
            self.assertAlmostEqual(r.slope, 0.7, places=9)
            self.assertAlmostEqual(r.intercept, -5.0, places=7)
            self.assertAlmostEqual(r.predicted / truth, 1, places=7)
            self.assertLess(r.slope_hw, 1e-6)
            self.assertLess((r.hi - r.lo) / truth, 1e-6)

    def test_logit_fit_coverage(self):
        """Six points (three sizes, two seeds) per draw; the measured value
        at the held-out size is the mean of log logit over two new seeds."""
        rng = np.random.default_rng(0)
        Ns, at, slope, a, sigma = (12.6e6, 35.4e6, 62.9e6), 113.2e6, 0.7, -5, .3
        covered = slopes = 0
        est = []
        for _ in range(1000):
            fit = d2l.logit_fit(self.points(Ns, slope, a, sigma, rng=rng),
                                at).iloc[0]
            new = a + slope * math.log(at) + rng.normal(0, sigma, 2).mean()
            covered += math.log(fit.lo) <= new <= math.log(fit.hi)
            slopes += abs(fit.slope - slope) <= fit.slope_hw
            est.append(fit.slope)
        self.assertLess(abs(covered / 1000 - 0.95), 0.02)
        self.assertLess(abs(slopes / 1000 - 0.95), 0.02)
        self.assertLess(abs(np.mean(est) - slope), 0.02)  # unbiased
        # with one seed the interval is wider (1/m with m = 1)
        one = d2l.logit_fit(self.points(Ns, sigma=sigma, seeds=(0,),
                                        rng=rng), at).iloc[0]
        two = d2l.logit_fit(self.points(Ns, sigma=sigma, rng=rng), at).iloc[0]
        self.assertGreater(one.hi / one.lo, 1)
        self.assertGreater(two.hi / two.lo, 1)

    def test_logit_fit_non_finite(self):
        """A rate with a non-finite logit gets no line (the point is not
        dropped: that would censor the largest logits); so does a rate
        with fewer than three points."""
        pts = self.points((1e7, 3e7, 6e7), lrs=(2e-3, 4e-3, 8e-3))
        pts.loc[0, 'logit'] = math.nan
        pts.loc[(pts['lr'] == 4e-3) & (pts['N'] == 6e7), 'logit'] = math.inf
        fit = d2l.logit_fit(pts, at=1e8)
        self.assertEqual(fit['lr'].tolist(), [8e-3])
        self.assertAlmostEqual(fit['slope'].iloc[0], 0.7, places=9)
        two = self.points((1e7, 3e7), seeds=(0,))
        fit = d2l.logit_fit(two, at=1e8)
        self.assertEqual((len(fit), list(fit.columns)),
                         (0, d2l._XFER_TABLES['fit']))

    def test_xfer_config(self):
        N = {s: d2l._non_embedding(d2l._xfer_config(s, 4e-3, 1, 300).arch)
             for s in self.SHAPES}
        self.assertEqual([round(n / 1e5) / 10 for n in N.values()],
                         [12.6, 35.4, 62.9, 113.3])
        for s in self.SHAPES:
            c = d2l._xfer_config(s, 4e-3, 1, 300)
            ref = d2l.r2_config(s).arch
            self.assertEqual(c.arch, ref._replace(T=1024))
            self.assertEqual((c.preset, c.steps, c.warmup, c.decay, c.clip,
                              c.global_batch, c.micro_batch, c.peak_lr,
                              c.seed, c.data_seed, c.stats_every, c.z_loss,
                              c.wd, c.data, c.val_every, c.probe_every,
                              c.ckpt_steps, c.state_steps, c.out_dir),
                             (s, 300, 76, 0, None, 64, 16, 4e-3, 1, 1, 10,
                              1e-4, 0.1, '100m', 0, 0, (), (), None))
        # a constant rate after warmup
        lr = [d2l.wsd_lr(k, c.steps, c.warmup, c.decay, c.peak_lr,
                         c.final_lr) for k in range(76, 300)]
        self.assertEqual(set(lr), {4e-3})
        # N by formula (the embedding tied, counted once, left out)
        a = d2l.r2_config('r2').arch
        self.assertEqual(N['r2'], a.L * (2 * a.d * a.H * a.d_h + 2 * a.d
                                         * a.H_kv * a.d_h + 3 * a.d * a.f
                                         + 2 * a.d) + a.d)
        with self.assertRaises(KeyError):
            d2l._xfer_config('r3', 4e-3, 0, 300)

    def table(self, larger=(1, 1, 1, 1)):
        """A pilot table: at (4e-3, s0), (4e-3, s1), (8e-3, s0), (8e-3, s1)
        the ladder-88m logit is above t1's where larger says 1."""
        rows, i = [], 0
        for lr in (4e-3, 8e-3):
            for s in (0, 1):
                rows.append(dict(shape='t1', lr=lr, seed=s, logit=100.0,
                                 first_nonfinite=None))
                rows.append(dict(shape='ladder-88m', lr=lr, seed=s,
                                 logit=150.0 if larger[i] else 90.0,
                                 first_nonfinite=None))
                i += 1
        return pd.DataFrame(rows)

    @staticmethod
    def diverge(t, shape, lr, seed, step=123):
        i = t.index[(t['shape'] == shape) & (t['lr'] == lr)
                    & (t['seed'] == seed)]
        t.loc[i, 'logit'], t.loc[i, 'first_nonfinite'] = math.nan, step
        return t

    def test_gate(self):
        self.assertEqual(d2l._xfer_gate(self.table(), (0, 1)), 'go')
        for i in range(4):  # one seed disagrees at one rate
            larger = [1] * 4
            larger[i] = 0
            self.assertEqual(d2l._xfer_gate(self.table(larger), (0, 1)),
                             'no-go')
        self.assertEqual(d2l._xfer_gate(self.table((0, 0, 0, 0)), (0, 1)),
                         'no-go')  # agreeing on the wrong order
        t = self.table()
        t.loc[1, 'logit'] = math.nan  # a non-finite run
        self.assertEqual(d2l._xfer_gate(t, (0, 1)), 'no-go')
        self.assertEqual(d2l._xfer_gate(self.table().iloc[1:], (0, 1)),
                         'no-go')  # a missing run
        # divergence: only the larger shape diverged = above (the trend in
        # its strongest form); the smaller one or both = against
        t = self.diverge(self.table((1, 1, 0, 1)), 'ladder-88m', 8e-3, 0)
        self.assertTrue(d2l._xfer_above(t, 8e-3, 0))
        self.assertEqual(d2l._xfer_gate(t, (0, 1)), 'go')
        t = self.diverge(self.table(), 't1', 8e-3, 0)
        self.assertFalse(d2l._xfer_above(t, 8e-3, 0))
        self.assertEqual(d2l._xfer_gate(t, (0, 1)), 'no-go')
        t = self.diverge(t, 'ladder-88m', 8e-3, 0)
        self.assertEqual(d2l._xfer_gate(t, (0, 1)), 'no-go')
        # the stored JSON's NaN for "did not diverge" reads the same
        t = self.table().astype({'first_nonfinite': float})
        self.assertEqual(d2l._xfer_gate(t, (0, 1)), 'go')

    def test_diverged_note(self):
        t = self.table()
        self.assertEqual(d2l.diverged_note(t), '')
        t = self.diverge(self.diverge(t, 't1', 4e-3, 1, 250), 'ladder-88m',
                         8e-3, 0, 212)
        self.assertEqual(d2l.diverged_note(t), '2 of 8 runs diverged '
                         '(first: ladder-88m seed 0, step 212)')

    def fake_runs(self, order, pilot_up=True, diverge=(), finished=None):
        """A _xfer_run that trains nothing: log logit linear in log N (two
        seeds apart by 0.1), ladder-88m below t1 at 8e-3 unless pilot_up,
        the (shape, lr, seed) in diverge diverged at step 123; finished =
        now unless given; records the call order and whether the
        prediction existed."""
        pred = os.path.join(self.root, 'size-xfer', 'prediction.json')

        def run(shape, lr, seed, steps, shards, device):
            N = d2l._non_embedding(d2l._xfer_config(shape, lr, seed,
                                                    steps).arch)
            y = 1 + 0.5 * math.log(N / 1e6) + 50 * lr + 0.1 * seed
            if not pilot_up and shape == 'ladder-88m' and lr == 8e-3:
                y = 0.0
            order.append((shape, lr, seed, os.path.exists(pred)))
            bad = (shape, lr, seed) in diverge
            row = dict(shape=shape, N=N, lr=lr, seed=seed,
                       logit=math.nan if bad else math.exp(y),
                       loss=math.inf if bad else 4.0,
                       first_nonfinite=123 if bad else None,
                       finished=finished or time.strftime(
                           '%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            tr = pd.DataFrame(dict(step=[10, steps], loss=[5.0, 4.0],
                                   max_logit=[1.0, math.exp(y)]))
            return row, tr.assign(shape=shape, lr=lr, seed=seed)
        return run

    def driver(self, full, run):
        with mock.patch.object(d2l, '_xfer_run', run), \
                mock.patch.object(d2l, '_train_shards', lambda c: None), \
                mock.patch.object(d2l, '_job_root', lambda: self.root), \
                contextlib.redirect_stdout(io.StringIO()):
            return d2l._size_transfer(self.SHAPES, self.LRS, (0, 1), 300,
                                      full, device='cpu')

    def test_driver_pilot_and_full(self):
        order = []
        res = self.driver(False, self.fake_runs(order))
        pilot = [(s, lr, seed) for s in ('t1', 'ladder-88m')
                 for lr in (4e-3, 8e-3) for seed in (0, 1)]
        self.assertEqual([o[:3] for o in order], pilot)
        self.assertEqual((res.gate, len(res.table), len(res.fit),
                          res.registered), ('go', 8, 0, None))
        self.assertEqual(list(res.fit.columns), d2l._XFER_TABLES['fit'])
        order.clear()
        res = self.driver(True, self.fake_runs(order))
        self.assertEqual(len(res.table), 32)
        self.assertEqual(res.table[['shape', 'lr', 'seed']].values.tolist(),
                         [[s, lr, seed] for s in self.SHAPES
                          for lr in self.LRS for seed in (0, 1)])
        r2 = [o for o in order if o[0] == 'r2']
        self.assertEqual(len(r2), 8)
        self.assertTrue(all(o[3] for o in r2))  # registered before r2 ran
        self.assertFalse(any(o[3] for o in order if o[0] != 'r2'))
        self.assertEqual(order[-8:], r2)  # r2 last
        self.assertRegex(res.registered, r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$')
        # the page's check: its own fit of the stored runs = the prediction
        held = res.table['shape'] == 'r2'
        N = res.table.loc[held, 'N'].iloc[0]
        page = d2l.logit_fit(res.table[~held], at=N)
        np.testing.assert_allclose(page.to_numpy(float),
                                   res.fit[page.columns].to_numpy(float))
        self.assertTrue(np.allclose(res.fit['slope'], 0.5))
        # a restart reads the same prediction; a changed one is refused
        again = self.driver(True, self.fake_runs([]))
        self.assertEqual(again.registered, res.registered)
        path = os.path.join(self.root, 'size-xfer', 'prediction.json')
        import json
        with open(path) as f:
            doc = json.load(f)
        doc['fit']['predicted'][0] *= 1.01
        with open(path, 'w') as f:
            json.dump(doc, f)
        with self.assertRaisesRegex(ValueError, 'differs from the fit'):
            self.driver(True, self.fake_runs([]))

    def test_driver_refusals(self):
        order = []
        res = self.driver(False, self.fake_runs(order, pilot_up=False))
        self.assertEqual(res.gate, 'no-go')
        order.clear()
        with self.assertRaisesRegex(RuntimeError, 'no-go'):
            self.driver(True, self.fake_runs(order, pilot_up=False))
        self.assertEqual({o[0] for o in order}, {'t1', 'ladder-88m'})
        self.assertEqual(len(order), 8)  # no run beyond the pilot's
        # a held-out run kept before any prediction was written
        with mock.patch.object(d2l, '_job_root', lambda: self.root):
            held = d2l._xfer_path('r2', 2e-3, 0)
        os.makedirs(os.path.dirname(held), exist_ok=True)
        open(held, 'w').close()
        with self.assertRaisesRegex(RuntimeError, 'without a prediction'):
            self.driver(True, self.fake_runs([]))
        with self.assertRaisesRegex(ValueError, 'pilot'):
            d2l.size_transfer(('t1', 'ladder-54m', 'r2'), self.LRS)
        with self.assertRaisesRegex(ValueError, 'pilot'):
            d2l.size_transfer(self.SHAPES, (2e-3, 8e-3))
        # a held-out run that finished before the registration time
        os.remove(held)
        with self.assertRaisesRegex(RuntimeError, 'finished before'):
            self.driver(True, self.fake_runs(
                [], finished='2000-01-01T00:00:00Z'))

    def test_pilot_only_go_refused(self):
        """A stored pilot result whose gate is go must never pass for the
        full transfer (the page would print nothing): refused outside the
        pilot's own regeneration."""
        res = self.driver(False, self.fake_runs([]))
        self.assertEqual(res.gate, 'go')
        with mock.patch.object(d2l, '_pretrain_result', lambda *a: res), \
                mock.patch.object(d2l, '_regenerating', lambda: ()), \
                self.assertRaisesRegex(ValueError, 'pilot only'):
            d2l.size_transfer(self.SHAPES, self.LRS)
        before_go = tuple(n for n in d2l.PRETRAIN_RERUN_NAMES
                          if n != 'size-xfer')
        with mock.patch.object(d2l, '_pretrain_result', lambda *a: res), \
                mock.patch.object(d2l, '_regenerating',
                                  lambda: ('size-xfer-pilot',)), \
                mock.patch.object(d2l, 'PRETRAIN_RERUN_NAMES', before_go):
            self.assertEqual(d2l.size_transfer(self.SHAPES, self.LRS).gate,
                             'go')
        with mock.patch.object(d2l, '_pretrain_result', lambda *a: res), \
                mock.patch.object(d2l, '_regenerating',
                                  lambda: ('size-xfer-pilot',)), \
                self.assertRaisesRegex(ValueError, 'regenerate size-xfer'):
            d2l.size_transfer(self.SHAPES, self.LRS)  # the pilot alone
        with mock.patch.object(d2l, '_pretrain_result', lambda *a: res), \
                mock.patch.object(d2l, '_regenerating',
                                  lambda: ('size-xfer-pilot', 'size-xfer')):
            self.assertEqual(d2l.size_transfer(self.SHAPES, self.LRS).gate,
                             'go')  # both names: the full job
        full = self.driver(True, self.fake_runs([]))
        with mock.patch.object(d2l, '_pretrain_result', lambda *a: full), \
                mock.patch.object(d2l, '_regenerating', lambda: ()):
            self.assertEqual(len(d2l.size_transfer(self.SHAPES,
                                                   self.LRS).table), 32)

    def page(self, res):
        """Cell 25c's printed output on res (d2l.size_transfer mocked)."""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'chapter_lm-pretraining',
            'reference-model.md')
        with open(path) as f:
            src = f.read().split(
                '#reference-model-the-largest-logit-and-model-size-2}\n')[1]
        src = src.split('```')[0].replace('%%tab pytorch\n', '')
        out, plots = io.StringIO(), []
        with mock.patch.object(d2l, 'size_transfer', lambda *a, **k: res), \
                mock.patch.object(d2l, 'plot_size_transfer',
                                  lambda *a, **k: plots.append(a)), \
                contextlib.redirect_stdout(out):
            exec(src, dict(d2l=d2l, np=np, math=math, pd=pd,
                           logit_fit=d2l.logit_fit))
        return out.getvalue().splitlines(), plots

    def test_page_cell(self):
        """Cell 25c: the no-go line names the range the pilot tested; a
        diverged run gives 'no fit' at its rate (smaller shapes) or
        'diverged (step k)' in the measured column (the held-out shape);
        warnings-free."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            lines, plots = self.page(self.driver(
                False, self.fake_runs([], pilot_up=False)))
            self.assertEqual(lines, ['pilot: no size trend from 12.6M to '
                                     '62.9M parameters at 300 steps '
                                     '(gate failed)'])
            self.assertEqual(plots, [])
            bad = {('ladder-54m', 1.6e-2, 1), ('r2', 8e-3, 0)}
            res = self.driver(True, self.fake_runs([], diverge=bad))
            self.assertEqual(res.fit['lr'].tolist(), [2e-3, 4e-3, 8e-3])
            lines, plots = self.page(res)
        self.assertEqual(len(lines), 5)
        self.assertIn('logit at 113.3M, predicted [95%]', lines[0])
        self.assertTrue(lines[1].endswith('inside'))
        self.assertTrue(lines[3].endswith('diverged (step 123)'), lines[3])
        self.assertEqual(lines[4], ' 0.016 no fit: 1 of 6 runs diverged '
                         '(first: ladder-54m seed 1, step 123)')
        self.assertEqual(len(plots), 1)
        self.assertLessEqual(max(map(len, lines)), 79)

    def test_xfer_run_trains_and_keeps(self):
        shards = dp.fixture_shards(os.path.join(self.tmp.name, 'set'),
                                   n_docs=300)

        def config(shape, lr, seed, steps):
            return toy_config(preset=shape, steps=steps, warmup=2, decay=0,
                              clip=None, stats_every=5, peak_lr=lr, seed=seed,
                              data_seed=seed)
        with mock.patch.object(d2l, '_xfer_config', config), \
                mock.patch.object(d2l, '_job_root', lambda: self.root), \
                contextlib.redirect_stdout(io.StringIO()):
            row, tr = d2l._xfer_run('toy', 3e-3, 1, 10, shards, 'cpu')
            with mock.patch.object(d2l, '_inst_run', None):  # not trained
                again, tr2 = d2l._xfer_run('toy', 3e-3, 1, 10, shards, 'cpu')
        self.assertEqual(row['N'], d2l._non_embedding(TOY))
        self.assertEqual(row['N'], sum(p.numel() for p in toy_model()
                                       .parameters()) - TOY.vocab * TOY.d)
        self.assertTrue(math.isfinite(row['logit']) and row['logit'] > 0)
        self.assertEqual(tr['step'].tolist(), [5, 10])
        self.assertEqual(row['logit'], tr['max_logit'].iloc[-1])
        self.assertIsNone(row['first_nonfinite'])
        self.assertEqual(str(row), str(again))
        pd.testing.assert_frame_equal(tr, tr2)

    def test_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        res = self.driver(True, self.fake_runs([]))
        held = res.table['shape'] == 'r2'
        fit = d2l.logit_fit(res.table[~held], at=res.table.loc[held, 'N']
                            .iloc[0])
        d2l.plot_size_transfer(res, fit, held='r2')
        ax = d2l.plt.gcf().axes[0]
        self.assertEqual((ax.get_xscale(), ax.get_yscale()), ('log', 'log'))
        self.assertEqual(len(ax.lines), 12)  # 4 fits, 4 fitted, 4 held out
        self.assertEqual([ln.get_markerfacecolor() for ln in ax.lines[8:]],
                         ['none'] * 4)
        self.assertEqual(len(ax.lines[4].get_xdata()), 6)  # 3 shapes x 2
        self.assertEqual(ax.get_ylabel(), 'largest logit at step 300')
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels, ['lr 0.002', 'lr 0.004', 'lr 0.008',
                                  'lr 0.016'])
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()],
                         ['12.6M', '35.4M', '62.9M', '113.3M'])
        d2l.plt.close('all')
        d2l.plot_size_transfer(res, fit.iloc[:0], held='r2')  # nothing drawn
        self.assertEqual(d2l.plt.get_fignums(), [])


class _Shards:
    """A token array that reads like TokenShards (n_tokens, read)."""
    def __init__(self, x):
        self.x, self.n_tokens = np.asarray(x, np.int32), len(x)

    def read(self, start, n):
        return self.x[start:start + n]


class SpikeTest(unittest.TestCase):
    """One bad batch under Adam (cell 27) and OLMo 2's repeated n-gram rule
    (cell 27b) on CPU: the arm table's checks, the excess statistic, the
    same bad batch at every state, the paired driver on a 2-layer toy
    (skip replays of one state bit-identical, so the floor is 0; every arm
    equals its skip replay before the bad step; states removed), the
    regeneration, the page's table and the plot, and the n-gram flags."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.shards = dp.fixture_shards(os.path.join(cls.tmp.name, 'set'),
                                       n_docs=600)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_arms(self):
        job = d2l._spike_job()
        arms = d2l._spike_arms(job['arms'], 64, warmup=job['cfg'].warmup)
        self.assertEqual(job['cfg'].warmup, 76)
        self.assertEqual(len(arms), 8)
        self.assertEqual(arms[6], (0.95, 76, 80, 'ngram', 8, False))
        self.assertEqual(arms[0].clip, True)
        for bad in ((0.95, 250, 300, 'random', 8, True),  # random = all rows
                    (0.95, 250, 300, 'ngram', 0, True),
                    (0.95, 250, 300, 'ngram', 65, True),
                    (0.95, 250, 300, 'n-gram', 8, True),
                    (0.95, 0, 30, 'ngram', 8, True),  # no state 0
                    (0.95, 250, 250, 'ngram', 8, True),
                    (0.95, 250, 301, 'ngram', 8, True),  # window past 400
                    (1.0, 250, 300, 'ngram', 8, True),
                    (0.95, 250, 300, 'ngram', 8, 1),
                    (0.95, 250, 300, 'ngram', 8)):
            with self.assertRaises(ValueError, msg=bad):
                d2l._spike_arms([bad], 64)
        d2l._spike_arms([(0.95, 1, 51, 'ngram', 8, True)], 64)  # the bounds
        d2l._spike_arms([(0.95, 76, 77, 'ngram', 8, True)], 64, warmup=76)
        with self.assertRaisesRegex(ValueError, '76 warmup steps'):
            d2l._spike_arms([(0.95, 75, 80, 'ngram', 8, True)], 64, warmup=76)
        with self.assertRaisesRegex(ValueError, '76 warmup steps'):
            d2l.spike_replay(job['cfg'], [(0.95, 50, 80, 'ngram', 8, False)])
        with self.assertRaisesRegex(ValueError, 'beta2 0.95, not 0.999'):
            d2l._spike_arms([(0.999, 1297, 1301, 'ngram', 8, False)], 64)
        with self.assertRaisesRegex(ValueError, 'twice'):
            d2l._spike_arms([(0.95, 250, 300, 'ngram', 8, True),
                             (0.95, 250, 300, 'ngram', 8, False)], 64)
        with self.assertRaisesRegex(ValueError, 'no arms'):
            d2l._spike_arms([], 64)

    def test_excess(self):
        steps = range(1, 21)
        ref = pd.Series(1.0, index=steps)
        e = pd.Series(0.0, index=steps)
        e.loc[10] = 5.0  # the bad step itself is not read
        e.loc[11:15] = [0.5, -0.2, 0.05, 0.01, 0.004]
        ex = d2l.spike_excess(ref + e, ref, 10, 0.003, window=8)
        self.assertEqual(ex['recover'], 4)  # last |e| >= 0.006 at step 14
        self.assertAlmostEqual(ex['peak'], 0.5)
        self.assertAlmostEqual(ex['total'], 0.5 - 0.2 + 0.05 + 0.01 + 0.004)
        self.assertEqual(d2l.spike_excess(ref, ref, 10, 0.003, 8)['recover'],
                         0)
        late = e.copy()
        late.loc[18] = 0.01  # still above at the window's last step
        self.assertIsNone(d2l.spike_excess(ref + late, ref, 10, 0.003,
                                           8)['recover'])
        late.loc[18] = 0.0
        late.loc[17] = 0.01
        self.assertEqual(d2l.spike_excess(ref + late, ref, 10, 0.003,
                                          8)['recover'], 7)
        nan = (ref + e).copy()
        nan.loc[13] = math.nan
        self.assertEqual(d2l.spike_excess(nan, ref, 10, 0.003, 8),
                         dict(peak=math.inf, recover=None, total=math.inf))
        short = (ref + e).loc[:15]  # a replay that stopped
        self.assertEqual(d2l.spike_excess(short, ref, 10, 0.003, 8)['peak'],
                         math.inf)

    def test_same_bad_batch_at_every_state(self):
        """source fixes the bad batch: the n-gram rows put at log step 4
        equal those put at step 8 (both built from step 8's data); the
        default (source None) is the old stream; make_bad_batch's pool."""
        plain = d2l.PackedStream(self.shards, TOY.T, 8)

        def bad(step, source):
            s = d2l._BadBatchStream(self.shards, TOY.T, 8, step, 'ngram', 512,
                                    k=3, period=4, source=source)
            b = s.batch(step - 1)
            rows = (b['input_ids'] != plain.batch(step - 1)['input_ids']).any(1)
            return b, rows
        (b4, r4), (b8, r8) = bad(4, 8), bad(8, 8)
        self.assertEqual(int(r4.sum()), 3)
        self.assertTrue(torch.equal(r4, r8))
        for key in ('input_ids', 'labels', 'pos_ids', 'doc_ids'):
            self.assertTrue(torch.equal(b4[key][r4], b8[key][r8]), key)
        self.assertTrue(torch.equal(b4['input_ids'][~r4],
                                    plain.batch(3)['input_ids'][~r4]))
        default, _ = bad(8, None)
        self.assertTrue(torch.equal(default['input_ids'], b8['input_ids']))
        ref = d2l.make_bad_batch(plain.batch(3), 'ngram', 512, seed=8, k=3,
                                 period=4, pool=plain.batch(7)['input_ids'])
        self.assertTrue(torch.equal(ref['input_ids'], b4['input_ids']))
        # random: the same ids at any step
        rnd = [d2l._BadBatchStream(self.shards, TOY.T, 8, s, 'random', 512,
                                   source=8).batch(s - 1)['input_ids']
               for s in (4, 8)]
        self.assertTrue(torch.equal(*rnd))
        with self.assertRaisesRegex(ValueError, 'source'):
            d2l._BadBatchStream(self.shards, TOY.T, 8, 4, 'random', 512,
                                source=0)

    def toy_cfg(self, **kw):
        return toy_config(preset='t1', compile=False, warmup=2, clip=None,
                          z_loss=0.0, **kw)

    def test_driver(self):
        root = tempfile.mkdtemp(dir=self.tmp.name)
        cfg = self.toy_cfg()
        late = os.path.join(root, 'late')  # a committed state at step 10
        with contextlib.redirect_stdout(io.StringIO()):
            d2l.pretrain(toy_model(), d2l.PackedStream(self.shards, TOY.T, 8),
                         d2l._inst_config(cfg, cfg.peak_lr, 0, 10,
                                          stats_every=0, state_steps=(10,),
                                          out_dir=late))
        late = {10: (d2l._state_path(late, 10), 0.95)}
        arms = ((0.95, 6, 8, 'random', 8, True),
                (0.95, 6, 8, 'ngram', 2, True),
                (0.999, 6, 8, 'ngram', 2, True),
                (0.95, 2, 4, 'ngram', 2, False),
                (0.95, 10, 12, 'ngram', 2, False))
        old = os.path.join(root, 'spike', 'log.jsonl')  # an earlier job's
        os.makedirs(os.path.dirname(old))
        with open(old, 'w') as f:
            f.write('{}\n')
        with mock.patch.object(d2l, '_train_shards', lambda c: self.shards), \
                mock.patch.object(d2l, '_job_root', lambda: root), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, '2 warmup steps'):
                d2l._spike(cfg, [(0.95, 1, 2, 'ngram', 2, False)], 3, 8,
                           steps=12, window=6, late=late, device='cpu')
            r = d2l._spike(cfg, arms, 3, 8, steps=12, window=6, late=late,
                           device='cpu')
            with self.assertRaisesRegex(ValueError, 'betas'):  # not its
                d2l._spike(dataclasses.replace(cfg, betas=(0.8, 0.95)),
                           arms[-1:], 3, 8, steps=12, window=6, late=late,
                           device='cpu')
        # the driver removes its own states only, not the earlier log
        self.assertFalse(os.path.exists(os.path.join(root, 'spike', 'v2')))
        self.assertEqual(os.listdir(os.path.join(root, 'spike')),
                         ['log.jsonl'])
        t = r.table
        self.assertEqual(list(t.columns), d2l._SPIKE_TABLES['table'])
        self.assertEqual([tuple(x) for x in t[['beta2', 'state', 'bad_step',
                                               'batch', 'k']].values],
                         [a[:5] for a in arms])
        self.assertEqual(r.floor, 0.0)  # CPU: two replays bit-identical
        self.assertTrue((t['drift'] == 0).all())
        orig = d2l.pretrain

        def drifting(model, stream, c, **kw):  # 1e-3 off at the first step
            log = orig(model, stream, c, **kw)
            if stream.bad_step is not None:
                log.loc[log.index[0], 'loss'] += 1e-3
            return log
        with mock.patch.object(d2l, '_train_shards', lambda c: self.shards), \
                mock.patch.object(d2l, '_job_root', lambda: root), \
                mock.patch.object(d2l, 'pretrain', drifting), \
                contextlib.redirect_stdout(io.StringIO()):
            d = d2l._spike(cfg, (arms[0], arms[3]), 3, 8, steps=12, window=6,
                           late=late, device='cpu')
        # a replay drift before the bad step is part of the floor
        np.testing.assert_allclose(d.table['drift'], [1e-3, 1e-3], rtol=1e-6)
        self.assertAlmostEqual(d.floor, 1e-3, places=9)
        tr = r.traces
        self.assertEqual(list(tr.columns), d2l._SPIKE_TABLES['traces'])
        runs = tr.groupby(['beta2', 'state', 'arm', 'batch', 'k'])['step']
        self.assertEqual(len(runs), 7 + 4 + 2 + 2)  # skip, clip, floor
        self.assertTrue((runs.count() == 12).all())

        def loss(b2, state, arm, batch='none', k=0):
            g = tr[(tr['beta2'] == b2) & (tr['state'] == state)
                   & (tr['arm'] == arm) & (tr['batch'] == batch)
                   & (tr['k'] == k)]
            return g.set_index('step')['loss']
        for a in arms:  # paired: equal to the skip replay before the bad step
            b2, state, step, batch, k, clip = a
            for arm, ref in (('bad', 'skip'),) + ((('bad+clip', 'skip+clip'),)
                                                  if clip else ()):
                x, y = loss(b2, state, arm, batch, k), loss(b2, state, ref)
                self.assertEqual(x.index[0], state + 1)
                pd.testing.assert_series_equal(x.loc[:step - 1],
                                               y.loc[:step - 1])
                self.assertGreater(abs(x[step] - y[step]), 1e-3, a)
        row = t.iloc[0]
        self.assertGreater(row.peak, 0)
        self.assertIsNone(row.recover)  # 2 x floor = 0 on CPU: never below
        bad = tr[(tr['arm'] == 'bad') & (tr['state'] == 6)
                 & (tr['batch'] == 'random')].set_index('step')
        self.assertAlmostEqual(row.norm, bad.loc[8, 'grad_norm'])
        self.assertAlmostEqual(row['median'], tr[
            (tr['arm'] == 'skip') & (tr['state'] == 6)
            & (tr['beta2'] == 0.95)]['grad_norm'].median())
        self.assertAlmostEqual(row.norm_ratio, row.norm / row['median'])
        self.assertAlmostEqual(row.clip_ratio, row.peak_clip / row.peak)
        self.assertAlmostEqual(row.clip_predicted, min(1, 1 / row.norm_clip))
        self.assertGreaterEqual(row.clipped_before, 0)
        self.assertTrue(t.iloc[3:][['peak_clip', 'clip_ratio']].isna()
                        .all().all())
        # the clip twins clip at 1.0 from their first step on (the bad step
        # included); the other replays never clip
        twin = tr['arm'].str.endswith('+clip')
        self.assertTrue((tr.loc[twin, 'clipped']
                         == (tr.loc[twin, 'grad_norm'] > 1.0)).all())
        self.assertFalse(tr.loc[~twin, 'clipped'].any())
        self.assertTrue(tr.loc[twin, 'clipped'].any())
        # the two beta2 states differ (their runs), the late arm starts at 10
        self.assertNotEqual(loss(0.95, 6, 'skip')[7],
                            loss(0.999, 6, 'skip')[7])
        self.assertEqual(loss(0.95, 10, 'skip').index[0], 11)

    def test_regeneration_and_page(self):
        kw = d2l._spike_job()
        page = self.page_cells()
        self.assertEqual(eval(page['arms'], dict(d2l=d2l)), kw['arms'])
        self.assertIn('period=7', page['call'])
        self.assertIn('source=300', page['call'])
        res = self.fake_result()
        with tempfile.TemporaryDirectory() as tmp, uncommitted(), \
                mock.patch.object(d2l, 'PRETRAIN_PENDING', frozenset()), \
                mock.patch.dict(os.environ, {'D2L_RERUN': 'spike-replay'}):
            cwd = os.getcwd()
            os.makedirs(os.path.join(tmp, 'chapter'))
            os.chdir(os.path.join(tmp, 'chapter'))
            try:
                calls = []
                with mock.patch.object(d2l, '_spike', lambda *a: calls.append(
                        a) or res), contextlib.redirect_stdout(io.StringIO()):
                    got = d2l.spike_replay(**kw)
                    path = '../data/lm/rerun/lm-r2-spike-v2.json.gz'
                    config = d2l._stored_config(d2l._read_result(
                        path, 'json-table'))
                    self.assertEqual(config['recipe']['late'], [
                        [1297, 'lm-r2t1-s1297-state-v1', 0.95]])
                    self.assertEqual((config['period'], config['source'],
                                      len(config['arms'])), (7, 300, 8))
                    os.environ['D2L_RERUN'] = 't1-runs'
                    pending = d2l.spike_replay(**kw)
                    os.environ['D2L_RERUN'] = ''
                    d2l.DATA_HUB['lm-r2-spike-v2'] = ('url', 'sha1')
                    with mock.patch.object(d2l, 'download_keyed',
                                           lambda key: path):
                        loaded = d2l.spike_replay(**kw)
                        with self.assertRaisesRegex(ValueError, 'period'):
                            d2l.spike_replay(**dict(kw, period=5))
                        with mock.patch.object(d2l, 'SPIKE_STEPS', 200), \
                                self.assertRaisesRegex(ValueError, 'recipe'):
                            d2l.spike_replay(**kw)
                    with self.assertRaises(ValueError):  # before any lookup
                        d2l.spike_replay(kw['cfg'], kw['arms'][:1] * 2)
            finally:
                os.chdir(cwd)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(got.table), 8)
        self.assertTrue(pending.label.endswith('pending'))
        self.assertEqual(list(pending.table.columns),
                         d2l._SPIKE_TABLES['table'])
        self.assertTrue(math.isnan(pending.floor))
        self.assertEqual(loaded.label, '')
        pd.testing.assert_frame_equal(loaded.table, got.table,
                                      check_dtype=False)
        self.assertAlmostEqual(loaded.floor, res.floor)
        # the page's table: header, 8 rows, floor line
        lines = self.table_output(loaded)
        self.assertEqual(len(lines), 10)
        self.assertLessEqual(max(map(len, lines)), 80)
        self.assertIn('peak excess', lines[0])
        self.assertEqual(lines[1].split()[:3], ['0.95', '250', 'random'])
        self.assertTrue(lines[7].rstrip().endswith('/ —'))
        self.assertIn('>100 / 3', lines[2])  # not back within the window
        self.assertEqual(lines[-1], 'replay floor (same state, same batches, '
                         'no bad batch): 3.0e-03')
        self.assertEqual(self.table_output(pending), [])

    def page_cells(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'chapter_lm-pretraining',
            'reference-model.md')
        with open(path) as f:
            src = f.read()
        call = src.split('#reference-model-spike}\n')[1].split('```')[0]
        table = src.split('#reference-model-spike-table}\n')[1].split(
            '```')[0]
        arms = call.split('ARMS = ')[1].split('\nspike = ')[0]
        return dict(arms=arms, call=call, table=table.replace(
            '%%tab pytorch\n', ''))

    def table_output(self, spike):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(self.page_cells()['table'], dict(d2l=d2l, pd=pd,
                                                  spike=spike))
        return out.getvalue().splitlines()

    def fake_result(self):
        rows = []
        for i, a in enumerate(d2l._spike_job()['arms']):
            b2, state, step, batch, k, clip = a
            row = dict(beta2=b2, state=state, bad_step=step, batch=batch, k=k,
                       norm=12.0 + i, median=0.4, norm_ratio=(12.0 + i) / 0.4,
                       peak=0.5 + i, recover=None if i == 1 else 12 + i,
                       total=4.0 + i, drift=2e-4 * i)
            if clip:
                row.update(norm_clip=12.0 + i, peak_clip=0.05, recover_clip=3,
                           total_clip=0.3, clipped_before=0,
                           clip_ratio=0.05 / (0.5 + i),
                           clip_predicted=1 / (12.0 + i))
            rows.append(row)
        traces = []
        for r in rows:
            for arm in ('bad', 'skip') + (('bad+clip', 'skip+clip')
                                          if 'peak_clip' in r else ()):
                bad = arm.startswith('bad')
                st = np.arange(r['state'] + 1, r['state'] + 151)
                traces.append(pd.DataFrame(dict(
                    beta2=r['beta2'], state=r['state'],
                    bad_step=r['bad_step'] if bad else 0,
                    batch=r['batch'] if bad else 'none',
                    k=r['k'] if bad else 0, arm=arm, step=st,
                    loss=3.0 + bad * np.exp(-(st - r['bad_step']).clip(0)
                                            / 10.0) * (st >= r['bad_step']),
                    grad_norm=0.4, clipped=False)))
        traces = pd.concat(traces, ignore_index=True).drop_duplicates(
            ['beta2', 'state', 'arm', 'batch', 'k', 'step'])
        return d2l.SpikeReplay(pd.DataFrame(rows).reindex(
            columns=d2l._SPIKE_TABLES['table']), traces, 3e-3, '')

    def test_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        d2l.plot_spike(self.fake_result())
        ax = d2l.plt.gcf().axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(labels, [
            'β2 0.95, random', 'β2 0.95, n-gram ×1', 'β2 0.95, n-gram ×8',
            'β2 0.95, n-gram ×64', 'β2 0.999, random',
            'β2 0.95, random, clip 1.0', 'replay floor'])
        self.assertEqual(ax.get_yscale(), 'log')
        self.assertEqual(ax.lines[0].get_xdata()[0], 1)
        self.assertEqual(len(ax.lines[0].get_xdata()), 100)
        self.assertEqual(ax.lines[-1].get_linestyle(), ':')
        d2l.plt.close('all')
        empty = d2l.SpikeReplay(pd.DataFrame(columns=d2l._SPIKE_TABLES[
            'table']), pd.DataFrame(), math.nan, 'spike-replay pending')
        n = len(d2l.plt.get_fignums())
        d2l.plot_spike(empty)
        self.assertEqual(len(d2l.plt.get_fignums()), n)

    def test_repeated_ngram_flags(self):
        """Windows of T = 512 with 32 and 31 repeats of n-grams of periods 1,
        7 and 13 (distinct ids, so the smallest period is the n-gram's
        length) in random text: the 32s are flagged with their period,
        repeats and start, the 31s are not; max_n = 12 misses period 13."""
        rng = np.random.default_rng(0)
        T, windows = 512, 7
        x = rng.integers(32, 30000, windows * T + 1)
        for w, (n, r) in enumerate([(1, 32), (1, 31), (7, 32), (7, 31),
                                    (13, 32), (13, 31)]):
            gram = np.arange(30001 + 20 * w, 30001 + 20 * w + n)
            x[w * T + 50:w * T + 50 + n * r] = np.tile(gram, r)
        tok = types.SimpleNamespace(decode=lambda ids: ' '.join(map(str,
                                                                    ids)))
        flags = d2l.repeated_ngram_flags(_Shards(x), T=T, tokenizer=tok)
        self.assertEqual(flags.windows, windows)
        t = flags.table
        self.assertEqual(t['window'].tolist(), [0, 2, 4])
        self.assertEqual(t['period'].tolist(), [1, 7, 13])
        self.assertEqual(t['repeats'].tolist(), [32, 32, 32])
        self.assertEqual(t['start'].tolist(), [50, 50, 50])
        self.assertEqual(t['text'][0], '"' + ' '.join(['30001'] * 32)[:50]
                         + '"')
        short = d2l.repeated_ngram_flags(_Shards(x), T=T, max_n=12,
                                         tokenizer=tok)
        self.assertEqual(short.table['window'].tolist(), [0, 2])
        clean = d2l.repeated_ngram_flags(_Shards(x[3 * T:4 * T + 1]), T=T,
                                         tokenizer=None)
        self.assertEqual((len(clean.table), clean.windows), (0, 1))


def eps_state(beta2=0.5, step=1):
    """A hand-built full state of _PretrainRun's form: sqrt of the
    bias-corrected second moment is a power of two per coordinate."""
    groups = [  # (names, sqrt(v-hat) per parameter)
        (['blks.0.attention.W_q.weight', 'blks.0.ffn.W_1.weight'],
         [[2 ** -14, 2 ** -14, 2 ** -20],
          [2 ** -16, 2 ** -18, 2 ** -25, 2 ** -12]]),
        (['token_emb.weight', 'blks.0.norm1.weight', 'norm.weight'],
         [[2 ** -24, 2 ** -17], [1.0, 2 ** -10], [2 ** -30]])]
    state, param_groups, i = {}, [], 0
    for names, values in groups:
        idx = []
        for s in values:
            v = torch.tensor(s, dtype=torch.float64) ** 2 * (
                1 - beta2 ** step)
            state[i] = dict(step=torch.tensor(float(step)),
                            exp_avg=torch.zeros(len(s)),
                            exp_avg_sq=v.float())
            idx.append(i)
            i += 1
        param_groups.append(dict(names=names, params=idx,
                                 betas=(0.9, beta2)))
    return dict(optim=dict(state=state, param_groups=param_groups),
                step=step)


class EpsHeadroomTest(unittest.TestCase):
    """Adam's epsilon headroom (cell 15b) on CPU: the readout, the pair
    check, the driver with mocked runs, the regeneration and the page."""
    LR = 1e-3

    def test_readout(self):
        table, pooled = d2l._eps_readout(eps_state(), 't1')
        self.assertEqual(table['group'].tolist(),
                         ['embedding', 'attention', 'MLP', 'gains'])
        self.assertEqual(table['n'].tolist(), [2, 3, 4, 3])
        self.assertEqual(table['median'].tolist(),  # lower middle value
                         [2 ** -24, 2 ** -14, 2 ** -18, 2 ** -10])
        self.assertEqual(table['below_hi'].tolist(), [1.0, 1 / 3, 0.5, 1 / 3])
        self.assertEqual(table['below_lo'].tolist(), [0.5, 0.0, 0.25, 1 / 3])
        self.assertEqual(set(table['model']), {'t1'})
        self.assertEqual(pooled, 2 ** -16)  # 4th of the 7 matrix values
        self.assertEqual(float(f'{pooled:.0e}'), 2e-5)  # the new eps
        # the bias correction at a real step and beta2
        table, pooled = d2l._eps_readout(eps_state(0.95, 1297), 'r2')
        self.assertAlmostEqual(pooled / 2 ** -16, 1, places=6)
        self.assertEqual(list(table.columns), d2l._EPS_TABLES['readout'])

    def test_pair_check(self):
        cfg = d2l.r2_config('t1', clip=None)
        cell, _ = d2l._inst_arm(cfg, 'clip', self.LR, 0)
        new = dataclasses.replace(cell, eps=5e-5)
        d2l._eps_pair_check(cell, new)  # eps only: accepted
        for field, value in dict(
                seed=1, data_seed=1, peak_lr=2e-3, clip=None, warmup=305,
                steps=1000, decay=100, z_loss=0.0, wd=0.0,
                betas=(0.9, 0.999), micro_batch=8, stats_every=50,
                probe_every=10, clip_from=10, data='4b',
                arch=cell.arch._replace(qk_norm=True)).items():
            with self.assertRaisesRegex(ValueError, f"'{field}'"):
                d2l._eps_pair_check(cell, dataclasses.replace(
                    new, **{field: value}))
        with self.assertRaisesRegex(ValueError, 'nothing'):
            d2l._eps_pair_check(cell, cell)

    def test_update_rms(self):
        c, _ = d2l._inst_arm(d2l.r2_config('t1', clip=None), 'clip',
                             self.LR, 0)
        tr = self.trace(0.2)
        tr.loc[tr['step'] < 76, 'rms_W_q'] = 9.0  # warmup: not read
        tr.loc[tr['step'] > 1297, 'rms_W_1'] = 9.0  # the decay: not read
        self.assertAlmostEqual(d2l._eps_update_rms(tr, c), 0.2)
        tr.loc[tr['step'] == 600, 'rms_W_2'] = 0.9  # one step: the median
        self.assertAlmostEqual(d2l._eps_update_rms(tr, c), 0.2)
        tr['rms_W_o'] = np.where(tr['rms_W_o'].notna(), 0.9, np.nan)
        self.assertAlmostEqual(d2l._eps_update_rms(tr, c),
                               (6 * 0.2 + 0.9) / 7)

    @staticmethod
    def trace(rms, steps=1526, every=50, loss=4.0):
        step = np.arange(1, steps + 1)
        t = pd.DataFrame(dict(step=step, loss=loss, grad_norm=0.3,
                              max_logit_0=np.where(step % 10 == 0, 20.0,
                                                   np.nan),
                              entropy_0=np.where(step % 10 == 0, 2.0,
                                                 np.nan)))
        for k in ('token_emb', 'norm', 'W_q', 'W_k', 'W_v', 'W_o', 'W_g',
                  'W_1', 'W_2'):
            t[f'rms_{k}'] = np.where(step % every == 0, rms, np.nan)
        return t

    def grid(self, lrs=(1e-3, 2e-3), variants=('none', 'clip'),
             steps=None):
        rows, traces = [], []
        for v in variants:
            for lr in lrs:
                for s in (0, 1):
                    rows.append(dict(variant=v, lr=lr, seed=s,
                                     final_loss=4.0, val_loss=3.9 + s / 100,
                                     first_nonfinite=None))
                    traces.append(self.trace(0.2 + s / 100).assign(
                        variant=v, lr=lr, seed=s))
        table = pd.DataFrame(rows)
        table.attrs['config'] = d2l._jsonable(dict(
            cfg=d2l._cfg_json(d2l.r2_config('t1', clip=None)),
            lrs=list(lrs), variants=list(variants), seeds=[0, 1],
            steps=steps, tau=30, recipe=d2l._inst_recipe()))
        return d2l.InstabilityRuns(table, pd.concat(traces,
                                                    ignore_index=True), 30,
                                   '', 7.0)

    def test_driver(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            states = {}
            for m, (b2, step) in dict(t1=(0.5, 1), r2=(0.95, 1297)).items():
                states[m] = os.path.join(tmp, f'{m}.pt')
                torch.save(eps_state(b2, step), states[m])
            reg = os.path.join(tmp, 'eps-headroom', 'eps.json')

            def run(c, extra, variant, tau, shards, val, device):
                self.assertTrue(os.path.exists(reg))  # registered first
                calls.append((c, extra, variant, tau))
                log = self.trace(0.1, loss=4.2)
                log['val_loss'] = np.where(log['step'] == 1526, 3.95, np.nan)
                return log
            grid = self.grid()
            patches = [
                mock.patch.object(d2l, 'EPS_STATES', states),
                mock.patch.object(d2l, '_job_root', lambda: tmp),
                mock.patch.object(d2l, '_inst_run', run),
                mock.patch.object(d2l, '_train_shards', lambda c: None),
                mock.patch.object(d2l, '_val_fn', lambda c: None),
                mock.patch.object(d2l, 'load_artifact', lambda n: grid)]
            for p in patches:
                p.start()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = d2l._eps_headroom(self.LR, device='cpu')
                    again = d2l._eps_headroom(self.LR, device='cpu')
                self.assertEqual(len(calls), 1)  # the kept run is read
                pd.testing.assert_frame_equal(res.pair, again.pair)
                self.assertEqual(res.registered, again.registered)
                # a changed readout against the registered eps: refused
                changed = eps_state(0.95, 1297)
                changed['optim']['state'][0]['exp_avg_sq'] *= 4
                torch.save(changed, states['r2'])
                with self.assertRaisesRegex(ValueError, 'readout'):
                    d2l._eps_headroom(self.LR, device='cpu')
                torch.save(eps_state(0.95, 1297), states['r2'])
                os.remove(reg)  # a kept run without its registration
                with self.assertRaisesRegex(RuntimeError, 'kept without'):
                    d2l._eps_headroom(self.LR, device='cpu')
                grid = self.grid(lrs=(2e-3,))
                with self.assertRaisesRegex(ValueError, 'no full-schedule'):
                    d2l._eps_headroom(self.LR, device='cpu')
                grid = self.grid(steps=400)
                with self.assertRaisesRegex(ValueError, 'no full-schedule'):
                    d2l._eps_headroom(self.LR, device='cpu')
                grid = self.grid()
                with mock.patch.object(d2l, 'INST_CLIP', 0.5), \
                        self.assertRaisesRegex(ValueError, 'recipe.clip'):
                    d2l._eps_headroom(self.LR, device='cpu')
            finally:
                for p in reversed(patches):
                    p.stop()
        c, extra, variant, tau = calls[0]
        self.assertEqual((res.eps, c.eps, extra, variant, tau),
                         (2e-5, 2e-5, {}, 'clip', None))
        self.assertEqual(res.matrix_median, 2 ** -16)
        cell, _ = d2l._inst_arm(d2l.r2_config('t1', clip=None), 'clip',
                                self.LR, 0)
        self.assertEqual(dataclasses.replace(c, eps=1e-8), cell)
        self.assertEqual((c.clip, c.steps, c.decay, c.stats_every),
                         (1.0, 1526, 229, 10))
        self.assertEqual(list(res.readout['model']), ['t1'] * 4 + ['r2'] * 4)
        p = res.pair
        self.assertEqual(list(p.columns), d2l._EPS_TABLES['pair'])
        self.assertEqual(p[['eps', 'seed', 'source']].values.tolist(), [
            [1e-8, 0, 'instability-t1'], [1e-8, 1, 'instability-t1'],
            [2e-5, 0, 'eps-headroom']])
        np.testing.assert_allclose(p['val_loss'], [3.9, 3.91, 3.95])
        np.testing.assert_allclose(p['update_rms'], [0.2, 0.21, 0.1])
        self.assertEqual(list(res.traces.columns), d2l._EPS_TABLES['traces'])
        self.assertEqual(sorted(set(res.traces['eps'])), [1e-8, 2e-5])
        self.assertEqual(len(res.traces), 2 * 1526)
        self.assertEqual(res.traces['update_rms'].notna().sum(), 2 * 30)

    def fake_result(self):
        rows = []
        for m, scale in (('t1', 1.0), ('r2', 0.15)):
            for g, med, hi, lo in (('embedding', 1.9e-5, 0.187, 0.0074),
                                   ('attention', 6.7e-5, 0.0072, 0.0),
                                   ('MLP', 4.5e-5, 0.0, 0.0),
                                   ('gains', 9.5e-5, 0.0, 0.0)):
                rows.append(dict(model=m, group=g, n=1000, median=med * scale,
                                 below_hi=hi, below_lo=lo))
        pair = pd.DataFrame(dict(
            eps=[1e-8, 1e-8, 5e-5], seed=[0, 1, 0],
            val_loss=[3.912, 3.906, 3.95], final_loss=4.0,
            first_nonfinite=None, update_rms=[0.183, 0.185, 0.121],
            source=['instability-t1'] * 2 + ['eps-headroom']))
        traces = pd.DataFrame(dict(step=[1, 2], loss=4.0, grad_norm=0.3,
                                   update_rms=np.nan, eps=1e-8, seed=0))
        return d2l.EpsHeadroom(pd.DataFrame(rows), pair, traces, 5e-5,
                               4.7e-5, '2026-11-27T10:00:00Z', '')

    def page_cell(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'chapter_lm-pretraining',
            'reference-model.md')
        with open(path) as f:
            src = f.read()
        return src.split('#reference-model-epsilon-headroom}\n')[1].split(
            '```')[0].replace('%%tab pytorch\n', '')

    def page_output(self, res):
        out = io.StringIO()
        with mock.patch.object(d2l, 'adam_eps_headroom', lambda lr: res), \
                contextlib.redirect_stdout(out):
            exec(self.page_cell(), dict(d2l=d2l, pd=pd))
        return out.getvalue().splitlines()

    def test_regeneration_and_page(self):
        page = self.page_cell()
        self.assertIn('PAIR_LR = 1e-3', page)
        self.assertNotIn("'source'", page)  # rows by eps, not load names
        self.assertEqual(d2l.PRETRAIN_JOBS['eps-headroom'],
                         ('adam_eps_headroom',
                          d2l.PRETRAIN_JOBS['eps-headroom'][1]))
        self.assertEqual(d2l.PRETRAIN_JOBS['eps-headroom'][1](),
                         dict(lr=1e-3))
        res = self.fake_result()
        calls = []
        with tempfile.TemporaryDirectory() as tmp, uncommitted(), \
                mock.patch.object(d2l, 'PRETRAIN_PENDING', frozenset()), \
                mock.patch.dict(os.environ, {'D2L_RERUN': 'eps-headroom'}):
            cwd = os.getcwd()
            os.makedirs(os.path.join(tmp, 'chapter'))
            os.chdir(os.path.join(tmp, 'chapter'))
            try:
                with mock.patch.object(d2l, '_eps_headroom', lambda lr:
                                       calls.append(lr) or res), \
                        contextlib.redirect_stdout(io.StringIO()):
                    got = d2l.adam_eps_headroom(self.LR)
                    path = '../data/lm/rerun/lm-r2-eps-v1.json.gz'
                    config = d2l._stored_config(d2l._read_result(
                        path, 'json-table'))
                    os.environ['D2L_RERUN'] = 't1-runs'
                    pending = d2l.adam_eps_headroom(self.LR)
                    os.environ['D2L_RERUN'] = ''
                    d2l.DATA_HUB['lm-r2-eps-v1'] = ('url', 'sha1')
                    with mock.patch.object(d2l, 'download_keyed',
                                           lambda key: path):
                        loaded = d2l.adam_eps_headroom(self.LR)
                        with self.assertRaisesRegex(ValueError, 'lr'):
                            d2l.adam_eps_headroom(2e-3)
                        with mock.patch.object(d2l, 'EPS_STATES', dict(
                                t1='lm-r2t1-s1526-state-v1', r2='x')), \
                                self.assertRaisesRegex(ValueError,
                                                       'recipe.states'):
                            d2l.adam_eps_headroom(self.LR)
                        with mock.patch.dict(d2l.PRETRAIN_ARTIFACTS, {
                                'instability-t1': ('lm-r2-inst-t1-v3',
                                                   'json-table')}), \
                                self.assertRaisesRegex(ValueError, 'cell'):
                            d2l.adam_eps_headroom(self.LR)
            finally:
                os.chdir(cwd)
        self.assertEqual(calls, [self.LR])
        self.assertEqual((config['lr'], config['cell']['key'],
                          config['recipe']['base']),
                         (1e-3, 'lm-r2-inst-t1-v2', 1e-8))
        self.assertNotIn('eps', config['run'])  # set by the job
        self.assertEqual(config['run']['clip'], 1.0)
        self.assertEqual(got.eps, 5e-5)
        self.assertTrue(pending.label.endswith('pending'))
        self.assertEqual(list(pending.pair.columns),
                         d2l._EPS_TABLES['pair'])
        self.assertTrue(math.isnan(pending.eps))
        self.assertEqual((loaded.label, loaded.eps, loaded.matrix_median),
                         ('', 5e-5, 4.7e-5))
        pd.testing.assert_frame_equal(loaded.pair, got.pair,
                                      check_dtype=False)
        pd.testing.assert_frame_equal(loaded.readout, got.readout,
                                      check_dtype=False)
        lines = self.page_output(loaded)
        self.assertEqual(len(lines), 11)  # header, 8 groups, the pair
        self.assertLessEqual(max(map(len, lines)), 80)
        self.assertEqual(lines[0].split(), ['model', 'group', 'median',
                                            'sqrt(v)', 'below', '1e-5',
                                            'below', '1e-7'])
        self.assertEqual(lines[1].split(), ['t1', 'embedding', '1.9e-05',
                                            '18.70%', '0.74%'])
        self.assertEqual(lines[9], 't1 pair at lr 1e-3: eps 1e-8 → val '
                         '3.912 (seed 1: 3.906), update RMS ÷ lr 0.183')
        self.assertEqual(lines[10], ' ' * 20 + 'eps 5e-5 → val 3.950, '
                         'update RMS ÷ lr 0.121')
        self.assertEqual(self.page_output(pending), [])
        bad = loaded._replace(eps=6e-5)  # not the median's rounding
        with self.assertRaises(AssertionError):
            self.page_output(bad)


class ZlossNoGoTest(unittest.TestCase):
    """The z-loss pilot did not meet claim 26's rule (S/fix/gates/zloss.md,
    2026-11-27: |1e-4 minus none| = 1.36 nats, rule > 5), so the page has
    no z-loss subsection or cell (spec 21.2 cell 27c, removed), the library
    no pilot and the name no registration (the exercise takes its place)."""

    def test_no_go(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'chapter_lm-pretraining',
            'reference-model.md')
        with open(path) as f:
            src = f.read()
        for gone in ('#reference-model-a-z-loss-pilot', 'zloss_pilot(',
                     '### A z-Loss Pilot'):
            self.assertNotIn(gone, src)
        for table in (d2l.PRETRAIN_RERUN_NAMES, d2l.PRETRAIN_JOBS,
                      d2l.PRETRAIN_ARTIFACTS, d2l._RERUN_ARTIFACT):
            self.assertNotIn('zloss-pilot', table)
        self.assertFalse(hasattr(d2l, 'zloss_pilot'))


class LibraryStructureTest(unittest.TestCase):
    """The chapter's #@save blocks against two freshness hazards (lib
    review findings 1 and 2): a block that defines no name and is no
    registry entry lands in the `_preamble` shard, on which every notebook
    of the book depends; and a name defined in two Language Models files
    with different code keeps only the later one, silently."""

    @staticmethod
    def _blocks(pattern):
        import glob
        import build_lib
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = []
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            with contextlib.redirect_stdout(io.StringIO()):
                blocks = build_lib.extract_save_blocks(path, 'pytorch')
            out += [(b[0], os.path.relpath(path, root)) for b in blocks]
        return build_lib, out

    def test_no_block_in_preamble_shard(self):
        import re
        build_lib, blocks = self._blocks('chapter_lm-pretraining/*.md')
        self.assertGreater(len(blocks), 100)
        bad = []
        for code, path in blocks:
            names = [n for n in build_lib._block_shard_names(code)
                     if n != 'd2l']
            entries, rest = build_lib.registry_entries(code)
            if names or (entries and not rest):
                continue
            if not re.match(r'(?:d2l\.)?(\w+)\[', code.lstrip()):
                bad.append(f'{path}: {code.splitlines()[0]}')
        self.assertEqual(bad, [])

    def test_no_name_redefined_across_files(self):
        import re
        build_lib, blocks = self._blocks('chapter_lm-*/*.md')
        seen, bad = {}, []
        for code, path in blocks:
            names = re.findall(r'^(?:def|class)\s+(\w+)', code, re.MULTILINE)
            if len(set(names)) != 1:
                continue
            # A method added to a class is keyed by its class
            cls = re.match(r'@(?:d2l\.)?add_to_class\(([\w.]+)\)', code)
            key = f'{cls.group(1)}.{names[0]}' if cls else names[0]
            first = seen.setdefault(key, (path, code))
            if first[0] != path and first[1] != code:
                bad.append(f'{key}: {first[0]} and {path}')
        self.assertEqual(bad, [])



class DomainsStageTest(unittest.TestCase):
    """The URL-host classes of build_pretraining_data.py --stage domains."""

    def test_url_domain(self):
        import build_pretraining_data as b
        cases = {'en.wikipedia.org': 'wikipedia', 'wikipedia.org': 'wikipedia',
                 'cs.stanford.edu': 'edu', 'www.ox.ac.uk': 'edu',
                 'www.unsw.edu.au': 'edu', 'www.nasa.gov': 'gov',
                 'www.gov.uk': 'gov', 'www.khanacademy.org': 'org',
                 'notwikipedia.org': 'org', 'example.com': 'other',
                 'education.com': 'other', '': 'other'}
        for host, dom in cases.items():
            self.assertEqual(b.url_domain(host), dom, host)


if __name__ == '__main__':
    unittest.main()
