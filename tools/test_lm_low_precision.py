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
`_stats_hook`; a no-grad forward leaves the amax histories and records
alone; `_profile_table` on a stub profiler; `_lowp_merge` (speeds only
for natively compiled arms, a profile per arm); `lowp_arms` and
`fp8_gemm_sweep` pending, and a stored labelled result prints its label;
`has_fp8_cores` and `device_name` on the CPU; the jobs of `PRETRAIN_JOBS`
against the page's literals.

The native FP4 path (Blackwell; claim 13 of the section, the only on-box
evidence for it: the build box has no FP4 tensor cores), on the CPU with the
compute capability mocked and `_fp4_scaled_mm` stubbed: dispatch by
capability ((8, 9), (9, 0), (11, 0) simulated; (10, 0), (10, 3), (12, 0),
(12, 1) native after one probe, cached), the probe (a failing or wrong GEMM
falls back to the simulator with a one-line reason in `fp4_backend`), a
torch without the FP4 dtypes; the E2M1 packing (known bytes, round trip)
and the blocked scale layout (inverse, and against the cuBLAS offset
formula looped as in torch's own `from_blocked`); operand equality: the
packed codes, un-swizzled E4M3/E8M0 scales and tensor scale handed to the
GEMM, decoded by this file's own E2M1 reference, equal the simulator's
quantized operands of the forward, input-gradient and weight-gradient GEMMs
exactly (NVFP4 with 16 x 16 weight tiles repeated into 1 x 16 blocks, 1-D
weight blocks, MXFP4; fp32 inputs exactly, bf16 inputs up to the
simulator's rounding to bf16), and the layer's outputs and gradients match;
the backend never enters `lowp_arms`'s configuration or the state dict;
`_lowp_convert`'s FP4 recipes; the `lowp-blackwell` job, its run folder and
its campaign entry in `tools/lm_campaigns.yaml`.
"""
import ast
import collections
import contextlib
import io
import math
import os
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F

from d2l import torch as d2l


def uncommitted():
    """d2l.DATA_HUB without chapter 21's results: the state before a
    campaign's result is registered (the tests of pending and regeneration
    paths must not depend on which results run A has committed)."""
    keys = {key for key, _ in d2l.PRETRAIN_ARTIFACTS.values()}
    return mock.patch.object(d2l, 'DATA_HUB', {
        k: v for k, v in d2l.DATA_HUB.items() if k not in keys})

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

    def test_delayed_needs_state(self):
        with self.assertRaises(ValueError):
            d2l.fp8_quant(torch.randn(8, 8), scale_mode='delayed')

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

    def test_no_grad_forward_is_read_only(self):
        torch.manual_seed(4)
        lin = d2l.Fp8Linear(128, 64, scale_mode='delayed')
        X = torch.randn(32, 128)
        lin(X.requires_grad_()).sum().backward()
        lin.track = True
        before = {r: {k: v.clone() for k, v in st.items()}
                  for r, st in lin.state.items()}
        with torch.no_grad():
            Y = lin(100 * X)  # a validation pass with a far larger amax
        self.assertEqual(lin.records, [])
        for role, st in lin.state.items():
            for k, v in st.items():
                self.assertTrue(torch.equal(v, before[role][k]), role)
        # it read the history: the training step's scale, so 100 X
        # saturates and the output is far from the exact one
        with torch.no_grad():
            self.assertGreater(rel(Y, F.linear(100 * X, lin.weight)), 0.5)
        lin(X).sum().backward()  # a training step records again
        self.assertEqual({r for r, _ in lin.records}, {'x', 'w', 'dy'})


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

    def test_ties_to_even(self):
        # midpoints between grid values (scale 1: the block holds a 6)
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        even = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
        x = torch.cat([ties, torch.tensor([6.0]), torch.zeros(8)])
        for sign in (1, -1):
            q = d2l.fake_quant_fp4(sign * x, 16, 'fp32')[:7]
            self.assertTrue(torch.equal(q, sign * even), q)

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


# References for the native FP4 tests, written from the formats' definitions
# (not from the library): E2M1 by its bit fields (sign, two exponent bits
# with bias 1, one mantissa bit), element 2j in the low half of byte j (as
# torch's pack_uint4 and torchao), and the cuBLAS offset of scale (i, j) in
# the blocked layout, looped as torch.testing's from_blocked does.
def ref_e2m1(n):
    n = n.long()
    e, m = (n >> 1) & 3, (n & 1).float()
    v = torch.where(e == 0, 0.5 * m, torch.exp2(e.float() - 1) * (1 + m / 2))
    return torch.where((n >> 3) & 1 == 1, -v, v)


def ref_unpack(packed):
    b = packed.view(torch.uint8).long()
    return ref_e2m1(torch.stack([b & 15, b >> 4], -1).flatten(-2))


def ref_unblock(flat, rows, cols):
    flat, out = flat.view(torch.uint8), torch.zeros(rows, cols,
                                                    dtype=torch.uint8)
    tiles_per_row = -(-cols // 4)
    for i in range(rows):
        for j in range(cols):
            outer, inner = i % 128, j % 4
            off = (512 * ((i // 128) * tiles_per_row + j // 4)
                   + (outer % 32) * 16 + (outer // 32) * 4 + inner)
            out[i, j] = flat[off]
    return out


def ref_dequant(packed, scales, t, block, kind):
    """The operand (rows, K) that codes, blocked scales and tensor scale t
    stand for, in the simulator's order: code x (block scale x t)."""
    v = ref_unpack(packed)
    bits = ref_unblock(scales, v.shape[0], v.shape[1] // block)
    if kind == 'e4m3':
        s = bits.view(torch.float8_e4m3fn).float() * t
    else:
        s = torch.exp2(bits.float() - 127)
    return v * s.repeat_interleave(block, 1)


def ref_gemm(a, b, scale_a, scale_b, block, kind):
    """What the hardware computes for _fp4_scaled_mm's arguments (tensor
    scales are applied by the caller)."""
    A = ref_dequant(a, scale_a, 1.0, block, kind)
    B = ref_dequant(b.T, scale_b, 1.0, block, kind)
    return A @ B.T


@contextlib.contextmanager
def mocked_gpu(cc, name='NVIDIA B200', probe=None):
    """A CUDA device of compute capability cc (no GPU touched): the
    probe's answer is probe (None: it passes); the probe cache is empty."""
    with mock.patch.object(torch.cuda, 'get_device_capability',
                           lambda d=None: cc), \
            mock.patch.object(torch.cuda, 'get_device_name',
                              lambda d=None: name), \
            mock.patch.dict(d2l._FP4_PROBES, clear=True), \
            mock.patch.object(d2l, '_fp4_probe',
                              mock.Mock(return_value=probe)) as p:
        yield p


class Fp4NativeTest(unittest.TestCase):
    """The Blackwell path, on the CPU (claim 13's on-box evidence)."""

    def test_api_and_pin(self):
        self.assertIsNone(d2l._fp4_api())
        with mock.patch.dict(torch.__dict__):
            del torch.__dict__['float4_e2m1fn_x2']
            self.assertIn('float4_e2m1fn_x2', d2l._fp4_api())
        # the scaled-mm contract in _fp4_scaled_mm's docstring was checked
        # against torch 2.11.0 and torchao 0.17: re-check it on a bump
        self.assertTrue(torch.__version__.startswith('2.11.'),
                        torch.__version__)
        doc = d2l._fp4_scaled_mm.__doc__
        for word in ('torch._scaled_mm', 'torch.nn.functional.scaled_mm',
                     'BlockWise1x32', 'SWIZZLE_32_4_4', 'float4_e2m1fn_x2',
                     'column-major', 'float8_e4m3fn', 'float8_e8m0fnu',
                     '2.11.0'):
            self.assertIn(word, doc)
        # the call each format takes (the GPU is not touched: both mocked)
        a = torch.zeros(128, 64, dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2)
        for kind, dtype in (('e4m3', torch.float8_e4m3fn),
                            ('pow2', torch.float8_e8m0fnu)):
            s = torch.zeros(512, dtype=torch.uint8).view(dtype)
            with mock.patch.object(torch, '_scaled_mm') as old, \
                    mock.patch.object(F, 'scaled_mm') as new:
                d2l._fp4_scaled_mm(a, a.T, s, s)
            if kind == 'e4m3':
                new.assert_not_called()
                self.assertEqual(old.call_args.kwargs,
                                 {'out_dtype': torch.float32})
            else:
                old.assert_not_called()
                args, kw = new.call_args
                self.assertEqual((args[3], args[5]), (
                    F.ScalingType.BlockWise1x32,) * 2)
                self.assertEqual(kw, dict(
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.float32))

    def test_dispatch(self):
        cases = {(8, 0): False, (8, 9): False, (9, 0): False, (11, 0): False,
                 (10, 0): True, (10, 3): True, (12, 0): True, (12, 1): True}
        for cc, native in cases.items():
            with mocked_gpu(cc) as probe:
                self.assertEqual(d2l.has_fp4_cores('cuda:0'), native, cc)
                self.assertEqual(d2l.has_fp4_cores('cuda:0'), native, cc)
                # one probe per device and format, only where it can pass
                self.assertEqual(probe.call_count, int(native), cc)
                d2l.has_fp4_cores('cuda:0', 'pow2')
                self.assertEqual(probe.call_count, 2 * int(native), cc)
        self.assertFalse(d2l.has_fp4_cores('cpu'))
        with mocked_gpu((10, 0)), mock.patch.object(
                d2l, '_fp4_api', lambda: 'torch 2.4 lacks float4_e2m1fn_x2'):
            self.assertFalse(d2l.has_fp4_cores('cuda'))
            self.assertIn('torch 2.4 lacks', d2l.fp4_backend('cuda'))

    def test_backend_lines(self):
        lines = {}
        with mocked_gpu((8, 9), 'NVIDIA GeForce RTX 4090'):
            lines['ada'] = d2l.fp4_backend('cuda:0')
        with mocked_gpu((10, 0)):
            lines['b200'] = d2l.fp4_backend('cuda:0')
            lines['mx'] = d2l.fp4_backend('cuda:0', 'pow2')
        with mocked_gpu((12, 0), 'NVIDIA GeForce RTX 5090',
                        probe='the probe GEMM failed: RuntimeError: x'):
            lines['failed'] = d2l.fp4_backend('cuda:0')
        lines['cpu'] = d2l.fp4_backend('cpu')
        self.assertEqual(lines, {
            'ada': 'FP4 GEMMs on RTX 4090 (sm_89): simulated (native needs '
                   'Blackwell: sm_100/103/120/121)',
            'b200': 'FP4 GEMMs on B200 (sm_100): native (NVFP4: E4M3 scales,'
                    ' blocks of 16)',
            'mx': 'FP4 GEMMs on B200 (sm_100): native (MXFP4: E8M0 scales, '
                  'blocks of 32)',
            'failed': 'FP4 GEMMs on RTX 5090 (sm_120): simulated (the probe '
                      'GEMM failed: RuntimeError: x)',
            'cpu': 'FP4 GEMMs on CPU: simulated (native needs a Blackwell '
                   'GPU)'})
        for line in lines.values():  # one short line (code-guide 7.2)
            self.assertNotIn('\n', line)
            self.assertLessEqual(len(line), 100)

    def test_probe(self):
        for kind, block in (('e4m3', 16), ('pow2', 32)):
            good = lambda a, b, sa, sb: ref_gemm(a, b, sa, sb, block, kind)
            with mock.patch.object(d2l, '_fp4_scaled_mm', good):
                self.assertIsNone(d2l._fp4_probe('cpu', kind), kind)
            swapped = lambda a, b, sa, sb: ref_gemm(a, b, sb, sa, block, kind)
            with mock.patch.object(d2l, '_fp4_scaled_mm', swapped):
                self.assertIn('off the simulator by', d2l._fp4_probe('cpu',
                                                                    kind))

        def fails(*args):
            raise RuntimeError('CUBLAS_STATUS_NOT_SUPPORTED when calling '
                               'cublasLtMatmul\nmore detail')
        with mock.patch.object(d2l, '_fp4_scaled_mm', fails):
            why = d2l._fp4_probe('cpu', 'e4m3')
        self.assertTrue(why.startswith('the probe GEMM failed: RuntimeError:'
                                       ' CUBLAS_STATUS_NOT_SUPPORTED'), why)
        self.assertNotIn('\n', why)
        # on a Blackwell card whose probe fails (run here on the CPU), the
        # backend line gives the reason and the layer stays on the simulator
        real_probe, stub = d2l._fp4_probe, mock.Mock(side_effect=fails)
        with mocked_gpu((10, 0)) as probe, \
                mock.patch.object(d2l, '_fp4_scaled_mm', stub):
            probe.side_effect = lambda dev, kind: real_probe('cpu', kind)
            self.assertFalse(d2l.has_fp4_cores('cuda:0'))
            self.assertEqual(stub.call_count, 1)  # the probe's one call
            self.assertEqual(d2l.fp4_backend('cuda:0'),
                             f'FP4 GEMMs on B200 (sm_100): simulated ({why})')
            on_card = lambda dev, kind='e4m3': d2l._fp4_why_not(
                'cuda:0', kind) is None  # the layer's tensors stay on the CPU
            with mock.patch.object(d2l, 'has_fp4_cores', on_card):
                d2l.Fp4Linear(32, 16)(torch.randn(16, 32)).sum().backward()
            self.assertEqual(stub.call_count, 1)
            self.assertEqual(probe.call_count, 1)  # cached

    def test_layer_dispatch(self):
        X = torch.randn(32, 64, requires_grad=True)
        for has, native in ((True, None), (False, None), (False, True)):
            lin = d2l.Fp4Linear(64, 32, native=native)
            stub = mock.Mock(side_effect=lambda a, b, sa, sb: ref_gemm(
                a, b, sa, sb, 16, 'e4m3'))
            with mock.patch.object(d2l, 'has_fp4_cores',
                                   lambda dev, kind='e4m3': has), \
                    mock.patch.object(d2l, '_fp4_scaled_mm', stub):
                lin(X).sum().backward()
            self.assertEqual(stub.call_count, 3 if has or native else 0)
        with self.assertRaisesRegex(ValueError, 'block 16'):
            d2l.Fp4Linear(64, 32, block=32, native=True)(X)
        # the format MXFP4 needs: E8M0 scales on blocks of 32
        lin = d2l.Fp4Linear(64, 32, block=32, native=None)
        with mock.patch.object(d2l, 'has_fp4_cores', mock.Mock()) as has:
            lin(X)
        has.assert_not_called()  # NVFP4 scales on blocks of 32: no hardware
        self.assertEqual(set(d2l.Fp4Linear(64, 32, native=True).state_dict()),
                         {'weight'})

    def test_packing(self):
        x = torch.cat([torch.tensor([0.5, 1.0, -6.0, 3.0]), torch.zeros(12)])
        q = d2l.fake_quant_fp4(x[None], 16)
        packed, scales, t = d2l._fp4_operand(q, x[None], 16, 'e4m3')
        self.assertEqual(packed.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(tuple(packed.shape), (1, 8))
        self.assertEqual(packed.view(torch.uint8)[0].tolist(),
                         [0x21, 0x5F, 0, 0, 0, 0, 0, 0])
        self.assertEqual(scales.dtype, torch.float8_e4m3fn)
        self.assertAlmostEqual(float(scales[0]) * float(t), 1.0, places=6)
        g = torch.Generator().manual_seed(4)
        for kind, block in (('e4m3', 16), ('pow2', 32)):
            x = torch.randn(40, 96, generator=g) * torch.rand(40, 1,
                                                              generator=g)
            x[3] = 0  # an all-zero row of blocks
            for rounding in ('rtn', 'sr'):
                q = d2l.fake_quant_fp4(x, block, kind, rounding,
                                       generator=g)
                op = d2l._fp4_operand(q, x, block, kind)
                self.assertTrue(torch.equal(
                    ref_dequant(*op, block, kind), q), (kind, rounding))
        codes = torch.randint(0, 16, (4, 32), generator=g, dtype=torch.uint8)
        codes[:, 0] = 7  # each block holds a 6: MXFP4's scale is 2^0
        vals = ref_e2m1(codes)
        op = d2l._fp4_operand(vals, vals, 32, 'pow2')
        self.assertTrue(torch.equal(ref_unpack(op[0]), vals))
        self.assertEqual(op[0].view(torch.uint8)[:, 0].tolist(),
                         (codes[:, 0] | codes[:, 1] << 4).tolist())

    def test_blocked_layout(self):
        g = torch.Generator().manual_seed(5)
        for rows, cols in ((128, 4), (200, 6), (37, 2), (256, 12)):
            s = torch.randint(0, 256, (rows, cols), generator=g,
                              dtype=torch.uint8)
            flat = d2l._fp4_to_blocked(s)
            self.assertEqual(flat.numel(), (-(-rows // 128) * 128)
                             * (-(-cols // 4) * 4))
            self.assertTrue(torch.equal(
                d2l._fp4_from_blocked(flat, rows, cols), s))
            self.assertTrue(torch.equal(ref_unblock(flat, rows, cols), s))

    def run_layer(self, lin, X, dY, native):
        """lin(X) and its backward with the GEMM operands recorded: the
        simulator's (Aq, Bq) per GEMM, or the hardware's (codes, scales, t)
        of A and B^T plus the stub's arguments."""
        lin.native, ops, calls, sims = native, [], [], []
        real_op, real_mm = d2l._fp4_operand, d2l._f32_mm
        kind, block = lin.scale, lin.block

        def spy_op(*args, **kw):
            ops.append(real_op(*args, **kw))
            return ops[-1]

        def stub(a, b, sa, sb):
            calls.append((a, b, sa, sb))
            return ref_gemm(a, b, sa, sb, block, kind)

        def spy_mm(A, B, out):
            sims.append((A, B))
            return real_mm(A, B, out)
        X = X.detach().clone().requires_grad_()
        torch.manual_seed(7)  # the same stochastic-rounding draws
        with mock.patch.object(d2l, '_fp4_operand', spy_op), \
                mock.patch.object(d2l, '_fp4_scaled_mm', stub), \
                mock.patch.object(d2l, '_f32_mm', spy_mm):
            Y = lin(X)
            dX, dW = torch.autograd.grad(Y, (X, lin.weight), dY)
        return (Y, dX, dW), ops, calls, sims

    def check_operands(self, lin, X, dY):
        kind, block = lin.scale, lin.block
        sim_out, _, _, sims = self.run_layer(lin, X, dY, False)
        nat_out, ops, calls, none = self.run_layer(lin, X, dY, True)
        self.assertEqual((len(sims), len(ops), len(calls), len(none)),
                         (3, 6, 3, 0))
        for k, (Aq, Bq) in enumerate(sims):  # forward, dgrad, wgrad
            op_a, op_bt = ops[2 * k], ops[2 * k + 1]
            a, b, sa, sb = calls[k]
            # the stub got exactly these operands, B column-major
            self.assertTrue(torch.equal(a.view(torch.uint8),
                                        op_a[0].view(torch.uint8)))
            self.assertEqual(b.stride(0), 1)
            self.assertTrue(torch.equal(b.T.view(torch.uint8),
                                        op_bt[0].view(torch.uint8)))
            self.assertIs(sa, op_a[1])
            self.assertIs(sb, op_bt[1])
            A = ref_dequant(*op_a, block, kind)
            B = ref_dequant(*op_bt, block, kind).T
            # exact for fp32 inputs; the simulator rounds bf16 inputs' operands
            self.assertTrue(torch.equal(A.to(Aq.dtype), Aq), (kind, k))
            self.assertTrue(torch.equal(B.to(Bq.dtype), Bq), (kind, k))
        for got, want in zip(nat_out, sim_out):  # bf16 outputs: one rounding
            tol = 1e-2 if got.dtype == torch.bfloat16 else 1e-5
            self.assertEqual(got.dtype, want.dtype)
            self.assertLess(rel(got.detach().float(), want.float()), tol)
        return ops

    def test_operand_equality_nvfp4(self):
        g = torch.Generator().manual_seed(6)
        torch.manual_seed(6)
        lin = d2l.Fp4Linear(64, 32)
        with torch.no_grad():  # rows of different size: tiles differ
            lin.weight.mul_(torch.linspace(0.1, 4.0, 32)[:, None])
        X, dY = torch.randn(48, 64, generator=g), torch.randn(48, 32,
                                                             generator=g)
        X[:, 5] *= 30  # an outlier channel
        ops = self.check_operands(lin, X, dY)
        # the page's 16 x 16 weight tiles: one scale repeated over 16 rows
        for op, rows in ((ops[1], 32), (ops[3], 64)):  # W (fwd), W^T (dgrad)
            s = d2l._fp4_from_blocked(op[1].view(torch.uint8), rows,
                                      op[0].shape[1] * 2 // 16)
            tiles = s.reshape(rows // 16, 16, -1)
            self.assertTrue(bool((tiles == tiles[:, :1]).all()))
            self.assertGreater(len(torch.unique(s)), 1)
        # 1-D weight blocks and no rotation; bf16 activations and gradients
        lin = d2l.Fp4Linear(64, 32, weight_block2d=False, hadamard=0)
        self.check_operands(lin, X, dY)
        self.check_operands(d2l.Fp4Linear(64, 32), X.bfloat16(),
                            dY.bfloat16())

    def test_operand_equality_mxfp4(self):
        g = torch.Generator().manual_seed(8)
        torch.manual_seed(8)
        lin = d2l.Fp4Linear(64, 64, block=32, scale='pow2')
        X, dY = torch.randn(64, 64, generator=g), torch.randn(64, 64,
                                                             generator=g)
        ops = self.check_operands(lin, X, dY)
        for packed, scales, t in ops:
            self.assertEqual(scales.dtype, torch.float8_e8m0fnu)
            self.assertEqual(float(t), 1.0)

    def test_backend_not_in_config(self):
        cfg = d2l.r2_config('fp8')
        arms = {'bf16': ('bf16', (0,)), 'nvfp4-mlp': ('nvfp4-mlp', (0,))}
        configs = []

        class Seen(Exception):
            pass

        def fake(name, compute, config, **kw):
            configs.append((name, config))
            raise Seen
        for has in (True, False):
            with mock.patch.object(d2l, 'has_fp4_cores',
                                   lambda dev, kind='e4m3': has), \
                    mock.patch.object(d2l, '_pretrain_result', fake), \
                    self.assertRaises(Seen):
                d2l.lowp_arms(cfg, arms)
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0][0], 'lowp-runs')
        self.assertNotIn('native', str(configs[0][1]))

    def test_convert(self):
        def convert(recipe, has):
            torch.manual_seed(0)
            model = d2l.PackedGPT.from_arch(TOY)
            with mock.patch.object(d2l, 'has_fp4_cores',
                                   lambda dev, kind='e4m3': has):
                got = d2l._lowp_convert(model, recipe)
            return got, model.blks[0].ffn.W_1
        for recipe, has, want, native in (
                ('nvfp4-mlp', True, 'eager', True),
                ('nvfp4-mlp', False, 'simulated', False),
                ('nvfp4-mlp-sim', True, 'simulated', False),
                ('nvfp4-mlp-native', True, 'eager', True)):
            got, lin = convert(recipe, has)
            self.assertEqual(got, want, recipe)
            self.assertIsInstance(lin, d2l.Fp4Linear)
            self.assertIs(lin.native, native)
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            convert('nvfp4-mlp-native', False)
        with self.assertRaises(ValueError):
            convert('nvfp4-mlp-x', True)

    def test_blackwell_job(self):
        name, kw = d2l.PRETRAIN_JOBS['lowp-blackwell']
        self.assertEqual(name, 'lowp_arms')
        kw = kw()
        self.assertEqual(kw['job'], 'lowp-blackwell')
        self.assertEqual(kw['cfg'], d2l.r2_config('fp8', steps=400, decay=60,
                                                  val_every=100))
        self.assertEqual({r for r, _ in kw['arms'].values()},
                         {'bf16', 'nvfp4-mlp-sim', 'nvfp4-mlp-native'})
        self.assertEqual(kw['arms']['bf16'][1], (0, 1, 2, 3))
        self.assertEqual(d2l._RERUN_ARTIFACT['lowp-blackwell'],
                         'lowp-blackwell')
        key = d2l.PRETRAIN_ARTIFACTS['lowp-blackwell'][0]
        self.assertEqual(key, 'lm-lowp-blackwell-v1')
        with open(os.path.join(ROOT, 'tools', 'lm_campaigns.yaml')) as f:
            camp = yaml.safe_load(f)
        entry = next(e for e in camp['campaigns']
                     if e['id'] == '21.3-lowp-blackwell')
        self.assertEqual(entry['command'],
                         'python tools/lm_regen.py lowp-blackwell')
        self.assertTrue(any(o.endswith(f'-> {key}')
                            for o in entry['outputs']))
        # its runs keep their own folder (the lowp-runs files never clash)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(d2l, '_job_root', lambda: tmp), \
                mock.patch.object(d2l, '_lowp_run', side_effect=RuntimeError
                                  ('stop')) as run:
            for job in ('lowp-runs', 'lowp-blackwell'):
                with self.assertRaisesRegex(RuntimeError, 'stop'):
                    d2l._lowp_run_cached(kw['cfg'], 'bf16', 'bf16', 0, False,
                                         False, job=job)
            self.assertEqual(sorted(os.listdir(tmp)),
                             ['lowp', 'lowp-blackwell'])
            self.assertEqual(run.call_count, 2)


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
    def __init__(self, key, us, shapes=(), device='cpu', total=None):
        self.key, self.self_device_time_total = key, us
        self.device_time_total = us if total is None else total
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
            _Event('d2l::fp8_copy', 0, total=5_000),  # kernels in children
            _Event('sgemm', 70_000, device='cuda'),
            _Event('triton_poi_fused', 20_000, device='cuda'),
            _Event('copy_kernel', 5_000, device='cuda'),
            _Event('## Call CompiledFxGraph', 90_000, device='cuda')]
        t = d2l._profile_table(prof, cfg, wall=0.0100, steps=10)
        self.assertEqual(tuple(t.columns), d2l._PROFILE_COLUMNS)
        t = t.set_index('kernels')
        self.assertAlmostEqual(t.loc['block GEMMs', 'ms'], 6.0)
        self.assertAlmostEqual(t.loc['head GEMMs', 'ms'], 1.0)
        self.assertAlmostEqual(t.loc['fp8 copies', 'ms'], 0.5)
        self.assertAlmostEqual(t.loc['other kernels', 'ms'], 2.0)
        self.assertAlmostEqual(t.loc['host and launches', 'ms'], 0.5)
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
                            columns=d2l._PROFILE_COLUMNS)
        mk = lambda r, v, n, s=None, p=None: dict(
            log=log(r, v), native=n, gib=3.0,
            stats=s if s is not None else empty('stats'),
            profile=p if p is not None else pd.DataFrame(
                columns=d2l._PROFILE_COLUMNS))
        arms = {'bf16': ('bf16', (0, 1)), 'tensorwise':
                ('torchao-tensorwise', (0,)), 'delayed': ('fp8-delayed', (0,))}
        runs = {('bf16', 0): mk(100, 3.0, '—', p=prof),
                ('bf16', 1): mk(102, 3.2, '—'),
                ('tensorwise', 0): mk(150, 3.1, 'yes', p=prof),
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
        # no speeds for an arm whose fp8 GEMMs run outside the compiled step
        self.assertTrue(t.loc['delayed', ['tok/s', 'IQR', '÷bf16']]
                        .isna().all())
        self.assertFalse(np.isnan(t.loc['delayed', 'val loss']))
        self.assertEqual(len(r.stats), 1)
        self.assertEqual(list(r.profile.columns),
                         d2l._LOWP_COLUMNS['profile'])
        self.assertEqual(list(r.profile['arm']), ['bf16', 'tensorwise'])
        self.assertEqual(len(r.losses), 4 * 500)
        final = r.losses.dropna(subset=['val_loss']).groupby(
            ['arm', 'seed'])['val_loss'].last()
        self.assertAlmostEqual(final['bf16'][1], 3.2)

    def test_pending(self):
        with mock.patch.object(d2l, 'PRETRAIN_PENDING',
                               frozenset({'lowp-runs', 'gemm-sweep'})), \
                mock.patch.dict(os.environ, {'D2L_RERUN': ''}), \
                uncommitted(), contextlib.redirect_stdout(io.StringIO()):
            runs = d2l.lowp_arms(d2l.r2_config('fp8'),
                                 {'bf16': ('bf16', (0,))})
            sweep = d2l.fp8_gemm_sweep(16384, (512,), [(16384, 768, 768)])
        self.assertIn('pending', runs.label)
        self.assertEqual(list(runs.table.columns),
                         d2l._LOWP_COLUMNS['table'])
        self.assertEqual(list(sweep.shapes.columns),
                         ['m', 'k', 'n', *d2l._SWEEP_COLUMNS])

    def test_stored_label_printed(self):
        runs = self.runs()._replace(label='smoke')
        sweep = d2l.GemmSweep(pd.DataFrame(columns=['n']),
                              pd.DataFrame(columns=['m']), label='H100')
        for fn, result, args in (
                (d2l.lowp_arms, runs, (d2l.r2_config('fp8'), {})),
                (d2l.fp8_gemm_sweep, sweep, (16384, (512,), []))):
            out = io.StringIO()
            with mock.patch.object(d2l, '_pretrain_result',
                                   lambda *a, **k: result), \
                    contextlib.redirect_stdout(out):
                got = fn(*args)
            self.assertEqual(got.label, result.label)
            self.assertEqual(out.getvalue(), f'[{result.label}]\n')

    def test_devices(self):
        self.assertFalse(d2l.has_fp8_cores('cpu'))
        self.assertEqual(d2l.device_name('cpu'), 'CPU')
        with mock.patch.object(torch.cuda, 'get_device_name',
                               lambda d=None: 'NVIDIA GeForce RTX 4090'):
            self.assertEqual(d2l.device_name('cuda:0'), 'RTX 4090')
        arms = {'bf16': ('bf16', (0,)), 'tw': ('torchao-tensorwise', (0,)),
                'rw': ('torchao-rowwise', (0,))}
        with mock.patch.object(d2l, '_rowwise_ok', lambda dev: False):
            self.assertEqual(d2l._torchao_recipes(arms), ('tensorwise',))
        with mock.patch.object(d2l, '_rowwise_ok', lambda dev: True):
            self.assertEqual(d2l._torchao_recipes(arms),
                             ('rowwise', 'tensorwise'))


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
