#!/usr/bin/env python3
"""GPU tests for chapter 22 (chapter_lm-hardware/), run only with
D2L_TEST_GPU=1 and a CUDA GPU, through the GPU-lock helper (they time
nothing that is quoted): `hw_copy_bench` byte accounting (GB/s x ms equals
the bytes moved: passes x nbytes for a read, twice nbytes for a copy),
`read_passes` (the sum of every pass), `hw_matmul_bench` FLOP
accounting (FlopCounterMode on the CUDA product equals 2 m n k, and
TFLOP/s x ms equals it), the FP8 product against the bf16 product of the
dequantized operands, the Triton L1 kernel's sum against torch, and the
pointer-chase array (one cycle through every line).
"""
import os
import unittest

import torch

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


if __name__ == '__main__':
    unittest.main()
