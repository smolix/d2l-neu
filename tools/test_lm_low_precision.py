#!/usr/bin/env python3
"""CPU tests for sec_lm_low_precision (chapter_lm-pretraining/
low-precision.md and low-precision-lib.md): `fp8_quant` at every
granularity (scale shapes, element error at most half an E4M3/E5M2 step,
saturation at the format maximum, the tile round trip, bad shapes refused)
and with delayed scaling (history of 16, first call equal to current
scaling, a jump of six against a history of ones saturates); `Fp8Linear`
(simulated) against fp32 `F.linear` and autograd (3-D input, bias, block
granularity, the operand statistics table); `fake_quant_fp4` (grid values
exact, round-to-nearest idempotent, stochastic rounding unbiased, the
power-of-two scale's amax / s in [4, 8), dim equal to a transpose, dtype
kept) and `block_nmse`; `_hadamard` (orthogonal, sign flips, non powers of
two refused) and `_rotate_rows` (dY^T X unchanged); `_fp4_weight` 16 x 16
tiles read the same along both dimensions; `Fp4Linear` gradients against
fp32; `_swap_linears` on a toy PackedGPT (outputs unchanged in fp32 within
the fp8 error); `mlp_input` against a hook-free recomputation;
`_stats_hook`; `_profile_table` on a stub profiler; `_lowp_merge`;
`lowp_arms` and `fp8_gemm_sweep` pending; the jobs of `PRETRAIN_JOBS`
against the page's literals.
"""
import ast
import collections
import contextlib
import io
import math
import os
import re
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from d2l import torch as d2l

torch.set_num_threads(min(8, torch.get_num_threads()))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, 'chapter_lm-pretraining', 'low-precision.md')

TOY = d2l.Arch(L=2, d=32, H=4, H_kv=2, d_h=8, f=64, vocab=512, T=32,
               rope_base=1e4, qk_norm=False, tie=True)

rel = lambda a, b: float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


class Fp8QuantTest(unittest.TestCase):
    def test_granularities(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(256, 384, generator=g) * torch.rand(256, 1,
                                                            generator=g)
        shapes = {'tensor': (), 'row': (256, 1), 'block1x128': (256, 384),
                  'block128x128': (256, 384)}
        for gran, shape in shapes.items():
            for fmt, m in (('e4m3', 3), ('e5m2', 2)):
                q, s = d2l.fp8_quant(x, fmt, gran)
                self.assertEqual(tuple(s.shape), shape, gran)
                fmax = torch.finfo(q.dtype).max
                self.assertLessEqual(float(q.float().abs().max()), fmax)
                xr = q.float() * s
                # normal values: at most half a step, 2^-(m+1) relative
                z = (x / s).abs()
                normal = z >= torch.finfo(q.dtype).tiny
                err = ((xr - x).abs() / x.abs())[normal]
                self.assertLessEqual(float(err.max()), 2.0 ** -(m + 1) + 1e-6)
                # the tile's amax lands on the format maximum
                self.assertAlmostEqual(float(q.float().abs().max()), fmax)
        # 1x128 tiles: one scale per row and 128 columns
        _, s = d2l.fp8_quant(x, granularity='block1x128')
        self.assertEqual(len(torch.unique(s[0, :128])), 1)
        self.assertGreater(len(torch.unique(s[0])), 1)
        _, s = d2l.fp8_quant(x, granularity='block128x128')
        self.assertEqual(len(torch.unique(s[:128, :128])), 1)

    def test_tiles_round_trip(self):
        x = torch.arange(256 * 256, dtype=torch.float32).reshape(2, 128, 256)
        for gran in ('tensor', 'row', 'block1x128', 'block128x128'):
            tiles, untile = d2l._fp8_tiles(x, gran)
            self.assertTrue(torch.equal(untile(tiles), x), gran)
        tiles, _ = d2l._fp8_tiles(x, 'block128x128')
        self.assertEqual(tuple(tiles.shape), (4, 128 * 128))
        # a tile holds a 128 x 128 block of the matrix
        X = x.reshape(256, 256)
        self.assertTrue(torch.equal(tiles[1], X[:128, 128:].reshape(-1)))
        with self.assertRaises(ValueError):
            d2l._fp8_tiles(torch.ones(100, 100), 'block1x128')
        with self.assertRaises(ValueError):
            d2l._fp8_tiles(x, 'columns')

    def test_delayed(self):
        state = {}
        x = torch.randn(64, 128)
        q1, s1 = d2l.fp8_quant(x, scale_mode='delayed', state=state)
        q0, s0 = d2l.fp8_quant(x)
        self.assertTrue(torch.equal(s1, s0))  # first call: the current amax
        self.assertEqual(tuple(state['amax'].shape), (16, 1, 1))
        # the scale comes from earlier calls only
        _, s2 = d2l.fp8_quant(6 * x, scale_mode='delayed', state=state)
        self.assertTrue(torch.equal(s2, s0))
        q3, s3 = d2l.fp8_quant(x, scale_mode='delayed', state=state)
        self.assertAlmostEqual(float(s3), 6 * float(s0), places=5)
        # the page's jump: a history of amax 1, then values up to 6
        state = {}
        for _ in range(16):
            d2l.fp8_quant(2 * torch.rand(4096) - 1, scale_mode='delayed',
                          state=state)
        y = 12 * torch.rand(4096) - 6
        q, s = d2l.fp8_quant(y, scale_mode='delayed', state=state)
        above = y.abs() > float(s) * 448  # beyond the old amax
        self.assertTrue(bool((q.float().abs()[above] == 448).all()))
        self.assertGreater(float(above.float().mean()), 0.8)


class Fp8LinearTest(unittest.TestCase):
    def test_against_fp32(self):
        torch.manual_seed(0)
        lin = d2l.Fp8Linear(256, 128, bias=True)
        X = torch.randn(4, 64, 256, requires_grad=True)
        Y = lin(X)
        Y_ref = F.linear(X, lin.weight, lin.bias)
        self.assertEqual(Y.shape, Y_ref.shape)
        analytic = math.sqrt(2 * 2 ** -6 / 24)
        self.assertLess(rel(Y, Y_ref), 2 * analytic)
        dY = torch.randn_like(Y)
        g = torch.autograd.grad(Y, (X, lin.weight, lin.bias), dY)
        g_ref = torch.autograd.grad(Y_ref, (X, lin.weight, lin.bias), dY)
        self.assertLess(rel(g[0], g_ref[0]), 0.1)
        self.assertLess(rel(g[1], g_ref[1]), 0.1)
        torch.testing.assert_close(g[2], g_ref[2])

    def test_block_and_stats(self):
        torch.manual_seed(1)
        lin = d2l.Fp8Linear(256, 256, granularity='block')
        lin.track = True
        X = torch.randn(128, 256, requires_grad=True)
        Y = lin(X)
        self.assertLess(rel(Y, F.linear(X, lin.weight)), 0.05)
        Y.sum().backward()
        table = lin.stats()
        self.assertEqual(list(table.index), ['x', 'w', 'dy'])
        self.assertEqual(tuple(table.columns), d2l._FP8_STAT_COLUMNS)
        # current scaling never saturates
        self.assertTrue((table['saturation'] == 0).all())
        self.assertTrue((table['saturation_current'] == 0).all())
        self.assertTrue((table['error'] < 0.05).all())
        self.assertEqual(lin.records, [])
        self.assertTrue(lin.stats().empty)

    def test_delayed_layer(self):
        torch.manual_seed(2)
        lin = d2l.Fp8Linear(128, 64, scale_mode='delayed')
        X = torch.randn(32, 128)
        for _ in range(3):
            lin(X.requires_grad_()).sum().backward()
        self.assertEqual(tuple(lin.state['x']['amax'].shape), (16, 1, 1))
        self.assertIn('amax', lin.state['dy'])


class Fp4Test(unittest.TestCase):
    def test_grid_and_rounding(self):
        grid = torch.tensor(d2l.E2M1 + tuple(-v for v in d2l.E2M1)) / 4
        for scale in ('fp32', 'e4m3', 'pow2'):
            self.assertTrue(torch.equal(
                d2l.fake_quant_fp4(grid, 16, scale), grid), scale)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4096, 64, generator=g)
        q = d2l.fake_quant_fp4(x)
        self.assertTrue(torch.equal(d2l.fake_quant_fp4(q), q))
        # every value is a grid value times its block's scale
        xb, qb = x.reshape(-1, 16), q.reshape(-1, 16)
        s = qb.abs().amax(-1, keepdim=True) / 6
        z = (qb.abs() / s)
        on = (z[..., None] - torch.tensor(d2l.E2M1)).abs().amin(-1)
        self.assertLess(float(on.max()), 1e-4)
        self.assertTrue(torch.equal(torch.sign(qb[qb != 0]),
                                    torch.sign(xb[qb != 0])))
        # stochastic rounding: unbiased
        v = torch.tensor([0.3, 1.2, 2.5, 4.7])
        blocks = torch.cat([v, torch.tensor([6.0]), torch.zeros(11)]).repeat(
            50_000, 1)
        mean = d2l.fake_quant_fp4(blocks, 16, 'fp32', 'sr',
                                  generator=g)[:, :4].mean(0)
        self.assertLess(float(((mean - v) / v).abs().max()), 0.01)

    def test_scales_dim_dtype(self):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(512, 64, generator=g)
        # pow2 (MXFP4): amax / s in [4, 8)
        xb = x.reshape(-1, 32)
        amax = xb.abs().amax(-1)
        s = torch.exp2(torch.floor(torch.log2(amax)) - 2)
        self.assertTrue(bool(((amax / s >= 4) & (amax / s < 8)).all()))
        q = d2l.fake_quant_fp4(x, 32, 'pow2')
        self.assertLessEqual(float(q.abs().max()), float(6 * s.max()))
        # dim: quantizing along dim 0 equals quantizing the transpose
        torch.testing.assert_close(d2l.fake_quant_fp4(x, dim=0),
                                   d2l.fake_quant_fp4(x.t()).t())
        self.assertEqual(d2l.fake_quant_fp4(x.bfloat16()).dtype,
                         torch.bfloat16)
        # the E4M3 scale costs little against an fp32 one
        n32 = d2l.block_nmse(x, lambda b, k: d2l.fake_quant_fp4(b, k, 'fp32'),
                             16)
        n8 = d2l.block_nmse(x, lambda b, k: d2l.fake_quant_fp4(b, k), 16)
        self.assertLess(n8 / n32, 1.1)
        manual = float(((d2l.fake_quant_fp4(x) - x) ** 2).sum() /
                       (x ** 2).sum())
        self.assertAlmostEqual(n8, manual, places=6)

    def test_hadamard(self):
        for n in (1, 2, 16, 64):
            H = d2l._hadamard(n, torch.Generator().manual_seed(n))
            torch.testing.assert_close(H @ H.T, torch.eye(n))
        H0, H1 = d2l._hadamard(16), d2l._hadamard(16, torch.Generator(
            ).manual_seed(0))
        self.assertTrue(torch.equal(H0.abs(), H1.abs()))
        self.assertFalse(torch.equal(H0, H1))
        with self.assertRaises(ValueError):
            d2l._hadamard(12)
        X, dY = torch.randn(64, 8), torch.randn(64, 5)
        torch.testing.assert_close(
            d2l._rotate_rows(dY, H1).T @ d2l._rotate_rows(X, H1), dY.T @ X,
            atol=1e-5, rtol=1e-5)
        with self.assertRaises(ValueError):
            d2l._rotate_rows(torch.randn(10, 3), H1)

    def test_fp4_weight(self):
        W = torch.randn(64, 48)
        a = d2l._fp4_weight(W, 16, True)
        b = d2l._fp4_weight(W, 16, True, dim=0)
        self.assertTrue(torch.equal(a, b))  # one quantized W for both GEMMs
        tiles = W.reshape(4, 16, 3, 16).transpose(1, 2).reshape(-1, 256)
        q = d2l.fake_quant_fp4(tiles, 256)  # tile (1, 2) is row 5
        torch.testing.assert_close(a[16:32, 32:48], q[5].reshape(16, 16))
        self.assertFalse(torch.equal(d2l._fp4_weight(W, 16, False),
                                     d2l._fp4_weight(W, 16, False, dim=0)))
        with self.assertRaises(ValueError):
            d2l._fp4_weight(torch.randn(20, 32), 16, True)

    def test_fp4_linear(self):
        torch.manual_seed(3)
        for hadamard in (16, 0):
            lin = d2l.Fp4Linear(128, 64, hadamard=hadamard)
            self.assertEqual(lin.H is None, not hadamard)
            self.assertNotIn('H', lin.state_dict())
            X = torch.randn(2, 64, 128, requires_grad=True)
            Y = lin(X)
            Y_ref = F.linear(X, lin.weight)
            self.assertLess(rel(Y, Y_ref), 0.2)
            dY = torch.randn_like(Y)
            g = torch.autograd.grad(Y, (X, lin.weight), dY)
            g_ref = torch.autograd.grad(Y_ref, (X, lin.weight), dY)
            self.assertLess(rel(g[0], g_ref[0]), 0.3)
            self.assertLess(rel(g[1], g_ref[1]), 0.3)
            self.assertTrue(all(bool(torch.isfinite(t).all()) for t in g))


class ModelTest(unittest.TestCase):
    def model(self):
        torch.manual_seed(0)
        return d2l.PackedGPT.from_arch(TOY)

    def batch(self):
        X = torch.randint(0, TOY.vocab, (2, TOY.T))
        ctx = d2l.PackCtx(*d2l.doc_ids_from_eos(X.int(), 0))
        return X, ctx

    def test_swap_linears(self):
        model, (X, ctx) = self.model(), self.batch()
        ref = model(X, ctx)
        d2l._swap_linears(model, lambda i, o: d2l.Fp8Linear(i, o))
        self.assertIsInstance(model.blks[0].ffn.W_2, d2l.Fp8Linear)
        self.assertIsInstance(model.blks[1].attention.W_k, d2l.Fp8Linear)
        Y = model(X, ctx)
        self.assertLess(rel(Y, ref), 0.1)
        Y.float().mean().backward()
        self.assertIsNotNone(model.blks[0].attention.W_q.weight.grad)

    def test_stats_hook(self):
        model = self.model()
        d2l._swap_linears(model, lambda i, o: d2l.Fp8Linear(
            i, o, scale_mode='delayed'))
        rows = []
        hook = d2l._stats_hook(model, rows, every=2)
        X, ctx = self.batch()
        for step in range(4):
            model(X, ctx).float().mean().backward()
            hook(step, model, None, None, dict(step=step + 1))
        # steps 1 and 3 (0-based) are tracked: their rows carry step 2, 4
        self.assertEqual(sorted({r['step'] for r in rows}), [2, 4])
        self.assertEqual({r['role'] for r in rows}, {'x', 'w', 'dy'})

    def test_mlp_input(self):
        model = self.model()
        rng = np.random.default_rng(0)
        toks = rng.integers(1, TOY.vocab, 4 * TOY.T)
        toks[[5, 40, 100]] = 0
        shards = mock.Mock(eos_id=0)
        shards.read = lambda start, n: toks[start:start + n]
        with mock.patch.object(d2l, 'fwedu_shards', lambda *a, **k: shards):
            X = d2l.mlp_input(model, 1, n_seqs=4, T=TOY.T)
        self.assertEqual(tuple(X.shape), (4 * TOY.T, TOY.d))
        seen = []
        h = model.blks[1].ffn.register_forward_pre_hook(
            lambda m, a: seen.append(a[0]))
        t = torch.from_numpy(toks.reshape(4, TOY.T).astype(np.int64))
        with torch.no_grad():
            model(t, d2l.PackCtx(*d2l.doc_ids_from_eos(t.int(), 0)))
        h.remove()
        torch.testing.assert_close(X, seen[0].float().flatten(0, -2))


class _Event:
    def __init__(self, key, us, shapes=(), device='cpu'):
        self.key, self.self_device_time_total = key, us
        self.input_shapes = list(shapes)
        self.device_type = (torch.autograd.DeviceType.CUDA if device == 'cuda'
                            else torch.autograd.DeviceType.CPU)


class ArmsTest(unittest.TestCase):
    def test_profile_table(self):
        cfg = d2l.r2_config('fp8')
        prof = mock.Mock()
        prof.key_averages.return_value = [
            _Event('aten::mm', 40_000, [[16384, 768], [768, 2048]]),
            _Event('aten::_scaled_mm', 20_000, [[16384, 2048], [2048, 768]]),
            _Event('aten::mm', 10_000, [[16384, 768], [768, cfg.arch.vocab]]),
            _Event('triton_poi_fused', 20_000),  # CPU side: not counted
            _Event('sgemm', 70_000, device='cuda'),
            _Event('triton_poi_fused', 20_000, device='cuda'),
            _Event('## Call CompiledFxGraph', 90_000, device='cuda')]
        t = d2l._profile_table(prof, cfg, wall=0.0100, steps=10)
        t = t.set_index('kernels')
        self.assertAlmostEqual(t.loc['block GEMMs', 'ms'], 6.0)
        self.assertAlmostEqual(t.loc['head GEMMs', 'ms'], 1.0)
        self.assertAlmostEqual(t.loc['other kernels', 'ms'], 2.0)
        self.assertAlmostEqual(t.loc['host and launches', 'ms'], 1.0)
        self.assertAlmostEqual(t['share'].sum(), 1.0)

    def runs(self):
        def log(rate, val):
            steps = np.arange(1, 501)
            return pd.DataFrame(dict(
                step=steps, tokens=steps * 1000, loss=5.0, tok_s=rate + steps % 3,
                val_loss=np.where(steps % 250 == 0, val, np.nan)))
        empty = lambda k: pd.DataFrame(columns=d2l._LOWP_COLUMNS[k])
        stats = pd.DataFrame([dict(step=10, role='x', saturation=0.0,
                                   underflow=0.1, error=0.02,
                                   saturation_current=0.0,
                                   underflow_current=0.1)])
        prof = pd.DataFrame([('block GEMMs', 6.0, 0.6)],
                            columns=d2l._LOWP_COLUMNS['profile'])
        mk = lambda r, v, n, s=None, p=None: dict(
            log=log(r, v), native=n, gib=3.0,
            stats=s if s is not None else empty('stats'),
            profile=p if p is not None else empty('profile'))
        arms = {'bf16': ('bf16', (0, 1)), 'tensorwise':
                ('torchao-tensorwise', (0,)), 'delayed': ('fp8-delayed', (0,))}
        runs = {('bf16', 0): mk(100, 3.0, '—', p=prof),
                ('bf16', 1): mk(102, 3.2, '—'),
                ('tensorwise', 0): mk(150, 3.1, 'yes'),
                ('delayed', 0): mk(20, 3.3, 'simulated', s=stats)}
        return d2l._lowp_merge(runs, arms)

    def test_merge(self):
        r = self.runs()
        t = r.table.set_index('arm')
        self.assertEqual(list(r.table.columns), d2l._LOWP_COLUMNS['table'])
        self.assertAlmostEqual(t.loc['bf16', 'val loss'], 3.1)
        self.assertEqual(t.loc['bf16', 'n'], 2)
        self.assertAlmostEqual(t.loc['bf16', '÷bf16'], 1.0)
        self.assertAlmostEqual(t.loc['tensorwise', '÷bf16'],
                               t.loc['tensorwise', 'tok/s'] /
                               t.loc['bf16', 'tok/s'])
        self.assertEqual(t.loc['delayed', 'native'], 'simulated')
        self.assertEqual(len(r.stats), 1)
        self.assertEqual(len(r.profile), 1)
        self.assertEqual(len(r.losses), 4 * 500)
        final = r.losses.dropna(subset=['val_loss']).groupby(
            ['arm', 'seed'])['val_loss'].last()
        self.assertAlmostEqual(final['bf16'][1], 3.2)

    def test_pending(self):
        with mock.patch.object(d2l, 'PRETRAIN_PENDING',
                               frozenset({'lowp-runs', 'gemm-sweep'})), \
                mock.patch.dict(os.environ, {'D2L_RERUN': ''}), \
                contextlib.redirect_stdout(io.StringIO()):
            runs = d2l.lowp_arms(d2l.r2_config('fp8'),
                                 {'bf16': ('bf16', (0,))})
            sweep = d2l.fp8_gemm_sweep(16384, (512,), [(16384, 768, 768)])
        self.assertIn('pending', runs.label)
        self.assertEqual(list(runs.table.columns),
                         d2l._LOWP_COLUMNS['table'])
        self.assertEqual(list(sweep.shapes.columns),
                         ['m', 'k', 'n', *d2l._SWEEP_COLUMNS])


class JobsTest(unittest.TestCase):
    """The regeneration jobs call what the page calls."""
    def page_code(self):
        src = open(PAGE).read()
        return '\n'.join(re.findall(r'```\{\.python \.input #[^}]*\}\n'
                                    r'%%tab pytorch\n(.*?)```', src, re.S))

    def test_arms_literal(self):
        code = self.page_code()
        m = re.search(r'^ARMS = (\{.*?\})$', code, re.S | re.M)
        arms = ast.literal_eval(re.sub(r'#.*', '', m.group(1)))
        name, kw = d2l.PRETRAIN_JOBS['lowp-runs']
        self.assertEqual(name, 'lowp_arms')
        kw = kw()
        self.assertEqual(kw['arms'], arms)
        self.assertEqual(d2l._cfg_json(kw['cfg']),
                         d2l._cfg_json(d2l.r2_config('fp8')))

    def test_sweep_shapes(self):
        arch = d2l.r2_config('fp8').arch
        M = 16 * 1024
        shapes = [(M, arch.d, n) for n in (arch.d, arch.H_kv * arch.d_h,
                                           arch.f)] + [(M, arch.f, arch.d)]
        name, kw = d2l.PRETRAIN_JOBS['gemm-sweep']
        kw = kw()
        self.assertEqual(name, 'fp8_gemm_sweep')
        self.assertEqual(kw['shapes'], shapes)
        self.assertEqual(kw['m'], M)
        code = self.page_code()
        self.assertIn('ns=(512, 768, 1024, 1536, 2048, 3072, 4096)', code)
        self.assertEqual(kw['ns'], (512, 768, 1024, 1536, 2048, 3072, 4096))

    def test_fp8_preset(self):
        cfg = d2l.r2_config('fp8')
        with torch.device('meta'):
            P = sum(p.numel() for p in d2l.PackedGPT.from_arch(
                cfg.arch).parameters())
        self.assertEqual(P, 75_511_552)
        self.assertEqual(cfg.steps * cfg.global_batch * cfg.arch.T,
                         20_004_864)


if __name__ == '__main__':
    unittest.main()
