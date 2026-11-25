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
chapter's 35 names; `pretrain` on a 2-layer toy (30 steps, accumulation 1
vs 4 at the same global batch, fp32 1e-5), resume after 20 of 40 steps
(losses 1e-6) and a branch from a saved state; `attn_logit_stats` against
the explicit softmax input; the `qk_clip_` contract; `softcap`;
`make_bad_batch`; `lr_sensitivity`; `_pretrain_result` and
`_pretrain_merged` with a temporary registry (regenerate with the config
stored, load, configuration mismatch, absent in a normal run, pending,
two regeneration names with other calls in the same process, D2L_RERUN=1
refused); `t_interval` against `scipy.stats.t` (dof 1-10, 1e-4);
`packed_to_hf` against `LlamaForCausalLM` (fp32 1e-5);
`load_reference_model` before the checkpoints are registered; a 2-process
gloo DDP `pretrain` equal to one process at the same global batch (1e-5).
"""
import collections
import contextlib
import io
import math
import os
import sys
import tempfile
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
    return mock.patch.object(d2l, 'DATA_HUB', {
        k: v for k, v in d2l.DATA_HUB.items() if k not in keys})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_lm_data_path as dp  # noqa: E402

torch.set_num_threads(min(8, torch.get_num_threads()))

RERUN_NAMES = (
    'pdp-throughput', 'pdp-mask-runs', 't1-runs', 'inst-t1-none',
    'inst-t1-qkclip', 'inst-t1-qknorm', 'inst-t1-softcap', 'spike-replay',
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
        self.assertEqual((r2.steps, r2.global_batch, r2.micro_batch,
                          r2.accum(4)), (6000, 128, 8, 4))
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
                     'inst-t1-qkclip', 'inst-t1-qknorm', 'inst-t1-softcap'}
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

    def test_load_reference_model_unregistered(self):
        with self.assertRaises(KeyError) as cm:
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
    cfg = toy_config(steps=10, ddp=True, micro_batch=2)
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
                               toy_config(steps=10, micro_batch=4))
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

    def toy_cfg(self, **kw):
        return toy_config(preset='t1', compile=False, warmup=2, **kw)

    def test_inst_t1(self):
        with mock.patch.object(d2l, '_train_shards', lambda c: self.shards):
            val = None
            with mock.patch.object(d2l, '_val_fn', lambda c: (
                    lambda m: 1.0)):
                runs = {v: d2l._inst_t1(self.toy_cfg(clip=None), (1e-3, 1e-1),
                                        v, (0, 1), 12, 0.5, device='cpu')
                        for v in ('none', 'qk_clip', 'qk_norm')}
        for r in runs.values():
            self.assertEqual(len(r.table), 4)
            self.assertEqual(set(r.traces['step']), set(range(1, 13)))
            self.assertTrue(r.traces['max_logit'].notna().any())
        # the clip pulls every head back to tau after each step
        hi = {v: r.table[r.table['lr'] == 1e-1]['max_logit'].values
              for v, r in runs.items()}
        self.assertTrue((hi['qk_clip'] < 0.2 * hi['none']).all())
        merged = d2l._merge_arms(runs)
        self.assertEqual(len(merged.table), 12)
        del val

    def test_spike(self):
        replays = (('from 6, same batches', 6, (), None),
                   ('from 6, skip 8', 6, (8,), None),
                   ('from 2, same batches', 2, (), None),
                   ('clip 1.0', 0, (), 1.0))
        with mock.patch.object(d2l, '_train_shards', lambda c: self.shards), \
                mock.patch.object(d2l, '_job_root', lambda: self.tmp.name):
            r = d2l._spike(self.toy_cfg(clip=None, z_loss=0.0), 8, 'random',
                           replays, steps=14, window=(1, 14), device='cpu')
        self.assertEqual(r.table['run'].tolist(), ['spiked run'] + [
            x[0] for x in replays])
        t = r.table.set_index('run')
        # same state, same batches: equal up to the floor (CPU: exact)
        self.assertLessEqual(t.loc['from 6, same batches', 'max_dloss'],
                             2 * r.floor + 1e-6)
        self.assertGreater(t.loc['from 6, skip 8', 'max_dloss'], 1e-4)
        self.assertNotAlmostEqual(t.loc['spiked run', 'loss_at_bad'],
                                  t.loc['from 6, skip 8', 'loss_at_bad'], 4)
        self.assertAlmostEqual(t.loc['spiked run', 'loss_at_bad'],
                               t.loc['from 2, same batches', 'loss_at_bad'], 4)
        self.assertEqual(set(r.traces['run']), set(t.index))

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
        self.assertTrue(os.path.exists('../data/lm/rerun/lm-cpt-runs-v1'
                                       '.json.gz'))
        os.environ['D2L_RERUN'] = ''
        with self.assertRaises(KeyError):
            call()

    def test_merged_last_part_merges(self):
        """Parts run side by side: the one that finishes last merges, even
        when the others' files appear while it computes (run A: smollm3-5
        started before smollm3-4 finished and left the merge pending)."""
        names = [f'smollm3-{i}' for i in range(3)]
        d2l.DATA_HUB.pop('lm-smollm3-trajectory-v1', None)  # a copy

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
        merged = '../data/lm/rerun/lm-smollm3-trajectory-v1.json.gz'
        self.assertTrue(os.path.exists(merged))
        os.remove(merged)
        # --merge: the named part's file exists, so nothing is computed
        calls = self.calls
        with mock.patch.dict(os.environ, {'D2L_PRETRAIN_MERGE': '1',
                                          'D2L_RERUN': 'smollm3-0'}):
            self.assertEqual(len(call(self.compute)), 6)
        self.assertEqual(self.calls, calls)
        self.assertTrue(os.path.exists(merged))


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
