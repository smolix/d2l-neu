#!/usr/bin/env python3
"""GPU tests for chapter 22 (chapter_lm-hardware/), run only with
D2L_TEST_GPU=1 and a CUDA GPU, through the GPU-lock helper (they time
nothing that is quoted): `hw_copy_bench` byte accounting (GB/s x ms equals
the bytes moved: passes x nbytes for a read, twice nbytes for a copy),
`read_passes` (the sum of every pass, also with the smaller blocks of a
small working set), `_hw_steady` (a clock and a power sampled), `hw_matmul_bench` FLOP
accounting (FlopCounterMode on the CUDA product equals 2 m n k, and
TFLOP/s x ms equals it), the FP8 product against the bf16 product of the
dequantized operands, the Triton L1 kernel's sum against torch, and the
pointer-chase array (one cycle through every line); for 22.2 (kernels):
the Triton RMSNorm against `F.rms_norm` (FP32, 1e-5), the Triton attention
against the FP32 reference (GQA, both page shapes at 1,024 tokens, a
sequence that is not a multiple of the tiles, non-causal), the chunked
attention against SDPA's flash kernel (BF16, 2e-2), `sdpa_causal` refusing
FP32 (no math fallback), `_time_arms` accounting (TFLOP/s x ms = FLOPs),
the register metadata `triton_spills` reads, and the Inductor kernel list;
for 22.3 (workload-hardware): the benchmark attention's decode step against
Transformers' SDPA on the same static cache (BF16 logits), the static cache
refill (every replay attends over L + 1 positions), the CUDA-graph kernel
count, and the decode, prefill and training benches on a two-layer
Qwen3-0.6B (finite times, peak memory reported), and the MoE router hooks
on a two-layer random Qwen3-30B-A3B.
"""
import contextlib
import io
import os
import unittest

import pandas as pd
import torch
import torch.nn.functional as F

from d2l import torch as d2l

GPU = os.environ.get('D2L_TEST_GPU') == '1' and torch.cuda.is_available()


@unittest.skipUnless(GPU, 'D2L_TEST_GPU=1 and a CUDA GPU')
class TestAcceleratorDesignGPU(unittest.TestCase):
    def test_copy_bytes(self):
        n = 2**26
        for kind, moved in (('read', (2**28 // n) * n), ('copy', 2 * n)):
            r = d2l.hw_copy_bench(n, kind, repeats=5)
            self.assertAlmostEqual(r['gbs'] * r['ms'] * 1e6 / moved, 1, 9)
            self.assertGreater(r['gbs'], 10)

    def test_matmul_flops(self):
        from torch.utils.flop_counter import FlopCounterMode
        m, n, k = 256, 512, 1024
        for fmt in ('fp32', 'tf32', 'bf16', 'fp16', 'fp8'):
            fn = d2l.hw_matmul_op(m, n, k, fmt)
            with FlopCounterMode(display=False) as c:
                out = fn()
            self.assertEqual(c.get_total_flops(), 2 * m * n * k, fmt)
            self.assertEqual(tuple(out.shape), (m, n))
            r = d2l.hw_matmul_bench(m, n, k, fmt, repeats=5)
            self.assertAlmostEqual(r['tflops'] * r['ms'] * 1e9
                                   / (2 * m * n * k), 1, 9)

    def test_fp8_product(self):
        A = torch.randn(128, 256, device='cuda')
        B = torch.randn(64, 256, device='cuda')
        (A8, sA), (B8, sB) = d2l.fp8_per_tensor(A), d2l.fp8_per_tensor(B)
        out = torch._scaled_mm(A8, B8.T, scale_a=sA, scale_b=sB,
                               out_dtype=torch.bfloat16).float()
        ref = (A8.float() * sA) @ (B8.float() * sB).T
        self.assertLess(float((out - ref).abs().max() / ref.abs().max()),
                        1e-2)

    def test_l1_kernel_sum(self):
        work, block, iters, programs = 2048, 1024, 6, 3
        x = torch.rand(programs * work, device='cuda')
        out = torch.empty(programs * block, device='cuda')
        d2l._l1_read_kernel[(programs,)](x, out, iters, WORK=work,
                                         BLOCK=block, num_warps=4)
        ref = torch.zeros(programs, block, device='cuda')
        xv = x.view(programs, work)
        for i in range(iters):
            idx = (torch.arange(block, device='cuda') + i * block) % work
            ref += xv[:, idx]
        torch.testing.assert_close(out.view(programs, block), ref)

    def test_read_passes(self):
        x = torch.rand(3 * 2048 + 77, device='cuda')
        out = d2l.read_passes(x, passes=3, block=2048)
        torch.testing.assert_close(out.sum(), 3 * x.sum(), rtol=1e-4,
                                   atol=1e-3)
        n_sm = torch.cuda.get_device_properties(0).multi_processor_count
        x = torch.rand(2**16, device='cuda')  # 256 KiB: smaller blocks
        out = d2l.read_passes(x, passes=3)
        self.assertEqual(out.numel(), min(2**16, 4 * n_sm * 128))
        torch.testing.assert_close(out.sum(), 3 * x.sum(), rtol=1e-4,
                                   atol=1e-2)

    def test_steady_clock(self):
        r = d2l._hw_steady(lambda: d2l.hw_copy_bench(2**24, 'read',
                                                     repeats=3),
                           warm_s=0.2)
        self.assertGreater(r['gbs'], 10)
        self.assertGreater(r['sm_mhz'], 100)  # sampled while it ran
        self.assertGreater(r['power_w'], 1)

    def test_chase_cycle(self):
        nbytes = 2**16
        nxt = d2l._chase_next(nbytes).cpu()
        lines, stride = nbytes // 128, 32
        j, seen = 0, set()
        for _ in range(lines):
            self.assertEqual(j % stride, 0)
            seen.add(j)
            j = int(nxt[j])
        self.assertEqual(len(seen), lines)  # every line, then back
        self.assertEqual(j, 0)
        r = d2l._chase_ns(nbytes, steps=1024, repeats=2)
        self.assertGreater(r['ns_per_load'], 1)


@unittest.skipUnless(GPU, 'D2L_TEST_GPU=1 and a CUDA GPU')
class TestKernelsGPU(unittest.TestCase):
    def shapes(self):
        R = d2l.r2_config('r2').arch
        qwen = d2l.HWShape(num_hidden_layers=1, hidden_size=4096,
                           num_attention_heads=32, num_key_value_heads=8,
                           head_dim=128, intermediate_size=1, vocab_size=1,
                           tie_word_embeddings=False)
        return {'reference': (R, 2), 'Qwen3-8B': (qwen, 1)}

    def test_rmsnorm(self):
        x = torch.randn(300, 4096, device='cuda')
        w = torch.randn(4096, device='cuda')
        ref = F.rms_norm(x, (4096,), w, eps=1e-6)
        torch.testing.assert_close(d2l.triton_rmsnorm(x, w), ref, atol=1e-5,
                                   rtol=0)
        torch.testing.assert_close(d2l.rms_norm_formula(x, w), ref,
                                   atol=1e-5, rtol=0)
        y = d2l.triton_rmsnorm(x[:, :1000].bfloat16(), w[:1000])  # d not 2^k
        torch.testing.assert_close(y.float(), F.rms_norm(
            x[:, :1000], (1000,), w[:1000], eps=1e-6), atol=3e-2, rtol=2e-2)

    def test_flash_against_fp32(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for name, shape in self.shapes().items():
                d2l.attention_error(d2l.flash_attention_triton, shape, 1024,
                                    name=name)
                d2l.attention_error(d2l.chunked_causal_attention, shape, 1024,
                                    name=name)
        q, k, v = d2l.attention_inputs(self.shapes()['reference'][0], 1, 1000)
        G = q.shape[1] // k.shape[1]
        K, V = k.repeat_interleave(G, 1).float(), v.repeat_interleave(
            G, 1).float()
        for causal in (True, False):
            for bq, bk in ((64, 128), (128, 32)):
                out = d2l.flash_attention_triton(q, k, v, causal=causal,
                                                 block_q=bq, block_k=bk)
                ref = F.scaled_dot_product_attention(q.float(), K, V,
                                                     is_causal=causal)
                self.assertLess(float((out.float() - ref).abs().max()
                                      / ref.abs().max()), 2e-2)

    def test_chunked_against_sdpa(self):
        for cfg, batch in self.shapes().values():
            q, k, v = d2l.attention_inputs(cfg, batch, 1024)
            ref = d2l.sdpa_causal(q, k, v).float()
            out = d2l.chunked_causal_attention(q, k, v).float()
            self.assertLess(float((out - ref).abs().max() / ref.abs().max()),
                            2e-2)

    def test_tile_matmul(self):
        g = torch.Generator(device='cuda').manual_seed(0)
        for m, n, k in ((300, 1000, 200), (4096, 1088, 4096)):
            a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16,
                            generator=g)
            b = torch.randn(k, n, device='cuda', dtype=torch.bfloat16,
                            generator=g)
            ref = a.float() @ b.float()
            out = d2l.tile_matmul(a, b).float()
            self.assertLess(float((out - ref).abs().max()
                                  / ref.abs().max()), 1e-2)
        r = d2l._wave_bench('Triton 128 x 128', 1024, 1024, 1024, 3)
        self.assertAlmostEqual(r['tflops'] * r['ms'] * 1e9 / 2**31, 1, 9)

    def test_sdpa_refuses_math(self):
        q, k, v = d2l.attention_inputs(self.shapes()['reference'][0], 1, 128,
                                       dtype=torch.float32)
        with self.assertRaises(RuntimeError), d2l._quiet():
            d2l.sdpa_causal(q, k, v)

    def test_time_arms(self):
        q, k, v = d2l.attention_inputs(self.shapes()['reference'][0], 1, 1024)
        flops = d2l._attn_flops(q)
        r = d2l._time_arms({'Triton': d2l.flash_attention_triton}, (q, k, v),
                           flops=flops, repeats=3, warm_s=0.1)
        self.assertAlmostEqual(r.tflops[0] * r.ms[0] * 1e9 / flops, 1, 9)
        self.assertEqual(d2l._arm_launches(
            {'Triton': d2l.flash_attention_triton}, (q, k, v)),
            {'Triton': (1, '_flash_fwd_kernel')})

    def test_spills_and_inductor(self):
        flash = pd.DataFrame([dict(shape='reference', arm='64 x 64, 4 warps',
                                   H=12, H_kv=4, d_h=64, block_q=64,
                                   block_k=64, num_warps=4, tflops=100.0)])
        flash.attrs['gpu'] = d2l._gpu_short(torch.cuda.get_device_name(0))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d2l.triton_spills(d2l.flash_attention_triton, flash)
        line = buf.getvalue().splitlines()[1].split()
        self.assertEqual(line[:2], ['64x64', 'w4'])
        self.assertGreater(int(line[3].replace(',', '')), 32)  # registers
        flash.attrs['gpu'] = 'RTX 9999'  # committed elsewhere: no TF/s
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d2l.triton_spills(d2l.flash_attention_triton, flash)
        lines = buf.getvalue().splitlines()
        self.assertNotIn('TF/s', lines[0])
        self.assertTrue(lines[-1].startswith('(TF/s omitted: the sweep ran '
                                             'on RTX 9999'))
        q, k, v = d2l.attention_inputs(self.shapes()['reference'][0], 1, 1024)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d2l.inductor_kernels(torch.compile(d2l.chunked_causal_attention,
                                               dynamic=False), q, k, v)
        text = buf.getvalue()
        self.assertIn('x Inductor', text)
        self.assertIn('x library', text)
        for line in text.splitlines():
            self.assertLessEqual(len(line), 79)


@unittest.skipUnless(GPU, 'D2L_TEST_GPU=1 and a CUDA GPU')
class TestWorkloadHardwareGPU(unittest.TestCase):
    MODEL = 'Qwen/Qwen3-0.6B'

    def tearDown(self):
        d2l.lm_free()

    def test_full_cache_attention(self):
        """One decode step over a full static cache: the benchmark attention
        (no mask, SDPA's GQA path) gives Transformers' SDPA logits."""
        import transformers
        model = d2l.hw_random_model(self.MODEL, n_layers=2)
        cfg = d2l._hw_model_config(self.MODEL, 2)
        with torch.device('cuda'):
            ref = transformers.AutoModelForCausalLM.from_config(
                cfg, dtype=torch.bfloat16, attn_implementation='sdpa')
        ref.load_state_dict(model.state_dict())
        ids = torch.randint(cfg.vocab_size, (2, 65), device='cuda')
        out = []
        with torch.inference_mode():
            for m in (model, ref.eval()):
                cache = transformers.StaticCache(config=cfg, max_cache_len=65)
                m(ids[:, :64], past_key_values=cache,
                  cache_position=torch.arange(64, device='cuda'))
                out.append(m(ids[:, 64:], past_key_values=cache,
                             cache_position=torch.tensor([64], device='cuda'),
                             logits_to_keep=1).logits.float())
        err = (out[0] - out[1]).abs().max().item()
        self.assertLess(err, 0.05 * out[1].abs().max().item())

    def test_refill_and_kernels(self):
        model = d2l.hw_random_model(self.MODEL, n_layers=2)
        cache, refill = d2l.hw_static_cache(model, 2, 128)
        refill()
        self.assertEqual(int(cache.layers[0].cumulative_length), 128)
        self.assertEqual(int(cache.layers[1].cumulative_length), 128)
        n = d2l._decode_kernels(model, 1, 128)
        self.assertIsNotNone(n)
        self.assertGreater(n, 20)
        self.assertLess(n, 1000)

    def test_benches(self):
        model = d2l.hw_random_model(self.MODEL, n_layers=2)
        for r in (d2l._hw_measure(d2l.hw_decode_bench, model, 2, 256,
                                  steps=8),
                  d2l._hw_measure(d2l.hw_eager_decode, model, 1, 256,
                                  steps=4),
                  d2l._hw_measure(d2l.hw_prefill_bench, model, 512,
                                  repeats=2),
                  d2l._hw_measure(d2l.hw_train_bench, model, 1, 256,
                                  repeats=2, warmup=1)):
            self.assertGreater(r['ms'], 0)
            self.assertGreaterEqual(r['mad'], 0)
            self.assertGreater(r['peak_gib'], 0)

    def test_moe_routes(self):
        """The router hooks of a two-layer random Qwen3-30B-A3B: K distinct
        sorted experts per sequence and layer, the router's own top-K of
        the last position's router input."""
        model = d2l.hw_random_model('Qwen/Qwen3-30B-A3B', n_layers=2)
        seen = {}
        model.model.layers[0].mlp.gate.register_forward_pre_hook(
            lambda module, args: seen.update(x=args[0].detach().clone()))
        ids = torch.randint(1000, (5, 32)).numpy()
        routes = d2l._moe_routes(model, ids, micro_batch=2)
        self.assertEqual(routes.shape, (5, 2, 8))
        for r in routes.reshape(-1, 8):
            self.assertEqual(len(set(r.tolist())), 8)
            self.assertEqual(r.tolist(), sorted(r.tolist()))
        gate = model.model.layers[0].mlp.gate
        x = seen['x'].reshape(1, 32, -1)[:, -1]  # the last micro-batch: 1
        with torch.no_grad():
            top = F.linear(x, gate.weight).float().topk(8, dim=-1).indices
        self.assertEqual(sorted(top[0].tolist()), routes[-1, 0].tolist())


if __name__ == '__main__':
    unittest.main()
