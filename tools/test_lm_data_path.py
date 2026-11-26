#!/usr/bin/env python3
"""CPU tests for sec_lm_data_path (chapter_lm-pretraining/data-path.md and
data-path-lib.md), the chapter-21 exports chapter 22 consumes first:
`doc_ids_from_eos` (fixture and the packing fixtures [5,3,9,2], [20],
[1,1,1,30] at T=8); packed attention (a document packed with another
equals the document alone, with and without position resets under RoPE;
unmasked differs; learned positions need the reset); `ctx=None` equals
`Qwen3Attention`/`Qwen3GPT` with the same weights (exact); 'sdpa' vs
'math' (1e-5); `token_weighted_loss` accumulation over 1/2/4/8
micro-batches (1e-6) against the naive mean of means (differs, and by the
covariance identity); `chunked_lm_loss` value and gradient (1e-6), z-loss
against autograd; `PackedStream` on a 1,000-document fixture shard set
(repeatable, resume at 37, world 1/2/4 multisets, seed, skip, labels -100
after EOS, layout-hash mismatch, windows across shards equal
`TokenShards.read`); `PackedGPT.adopt` (logits exact); `config()` round
trip; `PackedGPT.from_arch` for every preset on the meta device (P exact);
`set_rope(scale=4)` and `max_len`; `forward_cached(all_logits=True)` after
a prefill equals `forward` (fp32 1e-5); `_quiet()`; the DDP identity on
simulated ranks (global vs local denominators, 1e-6); `_rope_tables`
equals `qwen3_rope_tables`; `reseeded_restart_overlap`; the measurement
drivers on the fixture (`_UnmaskedStream`, `_padded_batch`,
`_foreign_mass` against an explicit softmax and uniform attention, the
throughput table and the masked/unmasked runs on a tiny CPU model, and the
`packing_comparison` load path with its configuration check).

Gated (D2L_TEST_GPU=1 and CUDA, run through the GPU-lock helper): 'flex'
vs 'sdpa' in bf16 (2e-2), logits and parameter gradients."""
import contextlib
import io
import os
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from d2l import torch as d2l


def uncommitted():
    """d2l.DATA_HUB without chapter 21's results: the state before a
    campaign's result is registered (the tests of pending and regeneration
    paths must not depend on which results run A has committed)."""
    from unittest import mock
    keys = {key for key, _ in d2l.PRETRAIN_ARTIFACTS.values()}
    return mock.patch.object(d2l, 'DATA_HUB', {
        k: v for k, v in d2l.DATA_HUB.items() if k not in keys})

torch.set_num_threads(min(8, torch.get_num_threads()))


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


def fixture_shards(out, n_docs=1000, seed=0, shard_tokens=20_000):
    """A shard set of n_docs random documents (ids 32-511, lengths 5-120)."""
    rng = np.random.default_rng(seed)
    docs = [(f'd{i}', ' '.join(map(str, rng.integers(32, 512, rng.integers(
        5, 121))))) for i in range(n_docs)]
    d2l.write_shards(docs, IdTokenizer(), out, 'train',
                     shard_tokens=shard_tokens, eos_id=0)
    return d2l.TokenShards(out)


def tiny_model(seed=0, **kw):
    torch.manual_seed(seed)
    args = dict(head_dim=16, ffn_hidden=96, max_len=64)
    args.update(kw)
    return d2l.PackedGPT(64, 64, 4, 2, 2, **args).eval()


def pack(lengths, T):
    """The first window of T tokens of documents of these lengths (each
    ending in EOS 0), as (tokens, doc_ids, pos_ids)."""
    toks, docs, pos = [], [], []
    for k, n in enumerate(lengths):
        toks += [7] * (n - 1) + [0]
        docs += [k] * n
        pos += list(range(n))
    return (torch.tensor(toks[:T]), torch.tensor(docs[:T]),
            torch.tensor(pos[:T]))


class DocIdsTest(unittest.TestCase):
    def test_fixture(self):
        t = torch.tensor([[5, 7, 0, 3, 0, 9, 9, 0, 4]])
        d, p = d2l.doc_ids_from_eos(t)
        self.assertEqual(d.tolist(), [[0, 0, 0, 1, 1, 2, 2, 2, 3]])
        self.assertEqual(p.tolist(), [[0, 1, 2, 0, 1, 0, 1, 2, 0]])
        self.assertEqual(d.dtype, t.dtype)

    def test_spec_fixture(self):
        t = torch.tensor([5, 6, 0, 7, 0, 0, 8], dtype=torch.int32)
        d, p = d2l.doc_ids_from_eos(t)
        self.assertEqual(d.tolist(), [0, 0, 0, 1, 1, 2, 3])
        self.assertEqual(p.tolist(), [0, 1, 2, 0, 1, 0, 0])
        self.assertEqual(d.dtype, torch.int32)

    def test_packing_fixtures(self):
        for lengths in ([5, 3, 9, 2], [20], [1, 1, 1, 30]):
            toks, docs, pos = pack(lengths, 8)
            d, p = d2l.doc_ids_from_eos(toks[None])
            self.assertEqual(d[0].tolist(), docs.tolist(), lengths)
            self.assertEqual(p[0].tolist(), pos.tolist(), lengths)
        self.assertEqual(pack([5, 3, 9, 2], 8)[1].tolist(),
                         [0, 0, 0, 0, 0, 1, 1, 1])
        self.assertEqual(pack([1, 1, 1, 30], 8)[2].tolist(),
                         [0, 0, 0, 0, 1, 2, 3, 4])


class PackedAttentionTest(unittest.TestCase):
    def window(self):
        g = torch.Generator().manual_seed(1)
        a = torch.randint(32, 64, (5,), generator=g)
        b = torch.randint(32, 64, (9,), generator=g)
        a[-1] = 0
        X = torch.cat([a, b])[None]
        return a, b, X, d2l.PackCtx(*d2l.doc_ids_from_eos(X))

    def test_packed_equals_alone(self):
        a, b, X, ctx = self.window()
        for backend in ('sdpa', 'math'):
            m = tiny_model(backend=backend)
            with torch.no_grad():
                packed = m(X, ctx)[0]
                alone = torch.cat([m(a[None])[0], m(b[None])[0]])
                no_reset = m(X, d2l.PackCtx(ctx.doc_ids, torch.arange(
                    14)[None]))[0]
                unmasked = m(X)[0]
            self.assertLess((packed - alone).abs().max(), 1e-5)
            self.assertLess((no_reset - alone).abs().max(), 1e-5)  # RoPE
            self.assertGreater((unmasked[5:] - alone[5:]).abs().max(), 1e-3)
        m = tiny_model(pos='learned')
        with torch.no_grad():
            alone = m(b[None])[0]
            packed = m(X, ctx)[0, 5:]
            no_reset = m(X, d2l.PackCtx(ctx.doc_ids, torch.arange(
                14)[None]))[0, 5:]
        self.assertLess((packed - alone).abs().max(), 1e-5)
        self.assertGreater((no_reset - alone).abs().max(), 1e-3)

    def test_sdpa_vs_math(self):
        _, _, X, ctx = self.window()
        m = tiny_model(backend='sdpa')
        with torch.no_grad():
            y1 = m(X, ctx)
            for blk in m.blks:
                blk.attention.backend = 'math'
            y2 = m(X, ctx)
        self.assertLess((y1 - y2).abs().max(), 1e-5)

    def test_ctx_none_is_qwen3(self):
        torch.manual_seed(0)
        q = d2l.Qwen3GPT(64, 64, 4, 2, 2, head_dim=16, qk_norm=True,
                         rope_theta=1e6, ffn_hidden=96, max_len=64).eval()
        p = d2l.PackedGPT.adopt(q)
        X = torch.randint(0, 64, (2, 12))
        with torch.no_grad():
            self.assertTrue(torch.equal(q(X), p(X)))
            Y = torch.randn(2, 12, 64)
            self.assertTrue(torch.equal(
                q.blks[0].attention(Y, Y, Y), p.blks[0].attention(Y, Y, Y)))
        self.assertEqual(p.token_emb.weight.data_ptr(),  # no copy
                         q.token_emb.weight.data_ptr())

    def test_adopt_shape_mismatch(self):
        q = d2l.Qwen3GPT(64, 64, 4, 2, 2, head_dim=16, ffn_hidden=96)
        p = d2l.PackedGPT(64, 64, 4, 2, 2, head_dim=16, ffn_hidden=64)
        with self.assertRaises(ValueError):
            d2l._check_same_tensors(p.state_dict(), q.state_dict())

    def test_rope_tables(self):
        c1, s1 = d2l.qwen3_rope_tables(10, 16, 1e6, offset=3)
        c2, s2 = d2l._rope_tables(torch.arange(10) + 3, 16, 1e6)
        self.assertTrue(torch.equal(c1, c2) and torch.equal(s1, s2))
        c3, _ = d2l._rope_tables(torch.arange(10)[None].repeat(2, 1), 16, 1e6)
        self.assertEqual(tuple(c3.shape), (2, 1, 10, 8))
        c4, _ = d2l._rope_tables(torch.arange(10), 16, 1e6, mscale=1.5)
        c0, _ = d2l.qwen3_rope_tables(10, 16, 1e6)
        self.assertLess((c4 - 1.5 * c0).abs().max(), 1e-6)


class PackedModelTest(unittest.TestCase):
    def test_config_round_trip(self):
        m = tiny_model(qk_norm=True, softcap=30.0, backend='math')
        m.set_rope([0.5] * 8, 1.2, 'yarn', 2.0)
        c = m.config()
        m2 = d2l._packed_from_config(c)
        self.assertEqual(m2.config(), c)
        self.assertEqual(c['max_len'], 128)
        self.assertEqual(set(m2.state_dict()), set(m.state_dict()))

    def test_from_arch_param_count(self):
        for preset in d2l.R2_ARCHS:
            arch = d2l.r2_config(preset).arch
            with torch.device('meta'):
                m = d2l.PackedGPT.from_arch(arch)
            L, d, H, H_kv, d_h, f, V, tie = d2l._arch_fields(arch)
            P = V * d + L * (2 * d * H * d_h + 2 * d * H_kv * d_h + 3 * d * f
                             + 2 * d + (2 * d_h if arch.qk_norm else 0)) + d
            self.assertEqual(sum(p.numel() for p in m.parameters()), P,
                             preset)
        with torch.device('meta'):
            m = d2l.PackedGPT.from_arch(d2l.r2_config('r2').arch)
        self.assertEqual(sum(p.numel() for p in m.parameters()), 138_440_448)

    def test_set_rope_max_len(self):
        with torch.device('meta'):
            m = d2l.PackedGPT(64, 64, 4, 2, 2, max_len=4096)
        m.set_rope(scale=4)
        self.assertEqual(m.max_len, 16384)
        m.set_rope()
        self.assertEqual(m.max_len, 4096)

    def test_forward_cached_all_logits(self):
        m = tiny_model()
        m.set_rope([1 / 10000 ** (i / 8) * 0.25 for i in range(8)], 1.1)
        X = torch.randint(0, 64, (2, 20))
        ks, vs = m.init_cache(2, 32)
        with torch.no_grad():
            full = m(X)
        m.forward_cached(X[:, :12], ks, vs, 0)
        logits, _, _ = m.forward_cached(X[:, 12:], ks, vs, 12,
                                        all_logits=True)
        self.assertLess((logits - full[:, 12:]).abs().max(), 1e-5)
        last, _, _ = m.forward_cached(X[:, 19:], ks, vs, 19)
        self.assertLess((last - full[:, 19]).abs().max(), 1e-5)

    def test_forward_cached_learned_raises(self):
        m = tiny_model(pos='learned')
        ks, vs = m.init_cache(1, 8)
        with self.assertRaises(NotImplementedError):
            m.forward_cached(torch.zeros(1, 4, dtype=torch.long), ks, vs, 0)

    def test_option_checks(self):
        X = torch.randint(0, 64, (1, 6))
        with self.assertRaises(ValueError):
            tiny_model(backend='fast')(X)
        with self.assertRaises(ValueError):
            tiny_model(pos='alibi')(X)

    def test_return_hidden(self):
        m = tiny_model()
        X = torch.randint(0, 64, (1, 9))
        with torch.no_grad():
            H, W = m(X, return_hidden=True)
            self.assertTrue(torch.allclose(F.linear(H, W), m(X)))


class LossTest(unittest.TestCase):
    def data(self, n=64, V=50, seed=0):
        g = torch.Generator().manual_seed(seed)
        logits = torch.randn(8, n // 8, V, generator=g, dtype=torch.float64)
        y = torch.randint(0, V, (8, n // 8), generator=g)
        y[torch.rand(8, n // 8, generator=g) < 0.3] = -100
        y[0, :5] = -100  # unequal counts per micro-batch
        return logits, y

    def test_accumulation(self):
        logits, y = self.data()
        s, n = d2l.token_weighted_loss(logits, y)
        ref = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                              ignore_index=-100)
        self.assertLess(abs(s / n - ref), 1e-6)
        for k in (1, 2, 4, 8):
            parts = [d2l.token_weighted_loss(logits[i:i + 8 // k],
                                             y[i:i + 8 // k])
                     for i in range(0, 8, 8 // k)]
            tw = sum(p[0] for p in parts) / sum(p[1] for p in parts)
            self.assertLess(abs(tw - ref), 1e-6)
        parts = [d2l.token_weighted_loss(logits[i:i + 2], y[i:i + 2])
                 for i in range(0, 8, 2)]
        means = torch.stack([p[0] / p[1] for p in parts])
        counts = torch.stack([p[1] for p in parts]).double()
        naive = means.mean()
        self.assertGreater(abs(naive - ref), 1e-3)
        # covariance identity: mean of means - pooled mean
        #   = -cov(count, mean) / mean(count)
        cov = ((counts - counts.mean()) * (means - means.mean())).mean()
        self.assertLess(abs((naive - ref) + cov / counts.mean()), 1e-6)

    def test_chunked(self):
        torch.manual_seed(0)
        H = torch.randn(3, 37, 16, dtype=torch.float64, requires_grad=True)
        W = torch.randn(50, 16, dtype=torch.float64, requires_grad=True)
        y = torch.randint(0, 50, (3, 37))
        y[0, :4] = -100
        s, n = d2l.chunked_lm_loss(H, W, y, chunk=10)
        gH, gW = torch.autograd.grad(s, (H, W))
        s2, n2 = d2l.token_weighted_loss(F.linear(H, W), y)
        rH, rW = torch.autograd.grad(s2, (H, W), retain_graph=True)
        self.assertEqual(int(n), int(n2))
        self.assertLess(abs(s - s2), 1e-6)
        self.assertLess((gH - rH).abs().max(), 1e-6)
        self.assertLess((gW - rW).abs().max(), 1e-6)
        z = 1e-2
        s3, _, nll = d2l.chunked_lm_loss_and_nll(H, W, y, chunk=8, z_loss=z)
        s4, _ = d2l.chunked_lm_loss(H, W, y, chunk=8, z_loss=z)
        # return_lse: the same three values, then the summed logsumexp
        s5, n5, nll5, lse_sum = d2l.chunked_lm_loss_and_nll(
            H, W, y, chunk=8, z_loss=z, return_lse=True)
        self.assertTrue(torch.equal(s5, s3) and torch.equal(nll5, nll))
        self.assertEqual(int(n5), int((y != -100).sum()))
        lse_ref = torch.logsumexp(F.linear(H, W), -1)[y != -100].sum()
        self.assertLess(abs(lse_sum - lse_ref), 1e-6)
        self.assertLess(abs(s4 - s3), 1e-9)
        logits = F.linear(H, W)
        valid = y != -100
        lse = torch.logsumexp(logits, -1)
        ref = s2 + z * (lse[valid] ** 2).sum()
        self.assertLess(abs(s3 - ref), 1e-6)
        self.assertLess(abs(nll - s2), 1e-6)
        g3 = torch.autograd.grad(s3, H)[0]
        g_ref = torch.autograd.grad(ref, H)[0]
        self.assertLess((g3 - g_ref).abs().max(), 1e-6)

    def test_ddp_identity(self):
        """Gradients summed over simulated ranks with the global count equal
        the one-process gradient; local denominators do not."""
        torch.manual_seed(0)
        H = torch.randn(8, 12, 16, dtype=torch.float64, requires_grad=True)
        W = torch.randn(50, 16, dtype=torch.float64)
        y = torch.randint(0, 50, (8, 12))
        y[:2, :9] = -100
        s, n = d2l.chunked_lm_loss(H, W, y)
        ref = torch.autograd.grad(s / n, H)[0]
        world = 4
        g_glob, g_loc = torch.zeros_like(H), torch.zeros_like(H)
        for r in range(world):
            sl = slice(r * 2, r * 2 + 2)
            s_r, n_r = d2l.chunked_lm_loss(H[sl], W, y[sl])
            # DDP averages: each rank's (loss * world / N) / world
            g_glob += torch.autograd.grad(s_r * world / n, H)[0] / world
            s_r, n_r = d2l.chunked_lm_loss(H[sl], W, y[sl])
            g_loc += torch.autograd.grad(s_r / n_r, H)[0] / world
        self.assertLess((g_glob - ref).abs().max(), 1e-6)
        self.assertGreater((g_loc - ref).abs().max(), 1e-6)


class StreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.shards = fixture_shards(os.path.join(cls.tmp.name, 'set'))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stream(self, **kw):
        return d2l.PackedStream(self.shards, 32, 16, **kw)

    def test_shape_and_labels(self):
        self.assertGreater(self.shards.n_shards, 1)
        b = self.stream().batch(3)
        X, Y = b['input_ids'], b['labels']
        self.assertEqual(tuple(X.shape), (16, 32))
        self.assertTrue(torch.equal(Y == -100, X == 0))
        self.assertTrue(torch.equal(Y[Y != -100], torch.cat(
            [b['input_ids'][:, 1:], Y[:, -1:]], 1)[Y != -100]))
        d, p = d2l.doc_ids_from_eos(X.int(), 0)
        self.assertTrue(torch.equal(b['doc_ids'], d))
        self.assertTrue(torch.equal(b['pos_ids'], p))

    def test_windows_equal_read(self):
        s = self.stream()
        idx = d2l._window_indices(s, 5)
        W = d2l._windows(s, idx)
        for i, w in zip(idx, W):
            self.assertTrue(np.array_equal(w, self.shards.read(int(i) * 32,
                                                               33)))
        first = self.shards.shards[0]['n_tokens']
        k = first // 32  # the window that crosses into shard 1
        w = d2l._windows(s, [k])[0]
        self.assertTrue(np.array_equal(w, self.shards.read(k * 32, 33)))

    def test_repeatable_and_resume(self):
        s = self.stream(seed=3)
        a = [s.batch(i)['input_ids'] for i in range(50)]
        self.assertTrue(torch.equal(a[7], self.stream(seed=3).batch(7)[
            'input_ids']))
        it = iter(s)
        for _ in range(37):
            next(it)
        state = s.state_dict()
        s2 = self.stream(seed=0)
        s2.load_state_dict(state)
        it2 = iter(s2)
        rest = [next(it2)['input_ids'] for _ in range(13)]
        for x, y in zip(rest, a[37:]):
            self.assertTrue(torch.equal(x, y))
        self.assertFalse(torch.equal(self.stream(seed=4).batch(0)[
            'input_ids'], a[0]))

    def test_world_multisets(self):
        def rows(world):
            out = []
            for step in range(10):
                for r in range(world):
                    s = self.stream(rank=r, world=world)
                    out += [tuple(x) for x in s.batch(step)['input_ids']
                            .tolist()]
            return sorted(out)
        one = rows(1)
        self.assertEqual(one, rows(2))
        self.assertEqual(one, rows(4))

    def test_skip(self):
        s = self.stream(skip=(3,))
        ref = self.stream()
        self.assertEqual(s.physical_index(3), 4)
        self.assertTrue(torch.equal(s.batch(3)['input_ids'],
                                    ref.batch(4)['input_ids']))
        self.assertTrue(torch.equal(s.batch(2)['input_ids'],
                                    ref.batch(2)['input_ids']))
        s.skip(5)
        self.assertEqual(s.physical_index(4), 6)
        self.assertEqual(s.state_dict()['skip'], [3, 5])

    def test_layout_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            other = fixture_shards(os.path.join(d, 'set'), n_docs=300, seed=9)
            s = d2l.PackedStream(other, 32, 16)
            with self.assertRaises(ValueError):
                s.load_state_dict(self.stream().state_dict())
        with self.assertRaises(ValueError):
            d2l.PackedStream(self.shards, 32, 16, world=3).batch(0)

    def test_skip_order_from_state(self):
        s = self.stream()
        s.load_state_dict(dict(self.stream().state_dict(), skip=[5, 3]))
        self.assertEqual(s.physical_index(4), 6)

    def test_restart_overlap(self):
        r = d2l.reseeded_restart_overlap()
        self.assertEqual((r.steps, r.windows), (10, 640))
        self.assertEqual((r.naive_match, r.stateless_match), (0, 10))
        self.assertGreater(r.naive_seen, 0)
        self.assertEqual(r.stateless_seen, 0)
        # the windows seen before are the repeats over the whole run
        perm = np.random.default_rng([0, 0]).permutation(10_000)
        after = np.random.default_rng([1, 0]).permutation(10_000)
        run = np.concatenate([perm[:40 * 64], after[40 * 64:50 * 64]])
        self.assertEqual(r.naive_seen, len(run) - len(np.unique(run)))


class QuietTest(unittest.TestCase):
    def test_quiet(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with d2l._quiet() as buf:
                print('downloading x')
        self.assertEqual(out.getvalue(), '')
        self.assertIn('downloading x', buf.getvalue())
        with self.assertRaises(RuntimeError) as cm:
            with contextlib.redirect_stdout(out), d2l._quiet():
                print('progress 50%')
                raise RuntimeError('boom')
        self.assertIn('progress 50%', ''.join(cm.exception.__notes__))


def tiny_cfg(**kw):
    """A t1-shaped config small enough for CPU: one block, d = 32, T = 32,
    vocabulary 512 (the fixture's ids), fp32, eager."""
    arch = d2l.Arch(1, 32, 2, 1, 16, 64, 512, 32, 1e4, False, True)
    return d2l.r2_config('t1', arch=arch, global_batch=8, micro_batch=4,
                         compile=False, dtype='fp32', **kw)


class MeasurementDriverTest(unittest.TestCase):
    """The drivers behind packing_throughput and packing_comparison, on
    the fixture shards (CPU, tiny model)."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.shards = fixture_shards(os.path.join(cls.tmp.name, 'set'))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_unmasked_stream(self):
        a = d2l.PackedStream(self.shards, 32, 16, seed=2).batch(4)
        b = d2l._UnmaskedStream(self.shards, 32, 16, seed=2).batch(4)
        for k in ('input_ids', 'labels'):
            self.assertTrue(torch.equal(a[k], b[k]))
        self.assertTrue((b['doc_ids'] == 0).all())
        self.assertTrue(torch.equal(b['pos_ids'][3], torch.arange(
            32, dtype=torch.int32)))
        self.assertGreater(int(a['doc_ids'].max()), 0)

    def test_padded_batch(self):
        b = d2l._padded_batch(self.shards, 64, 4, 2)
        for r in range(4):
            d = torch.from_numpy(self.shards.doc(8 + r)).long()
            n = min(len(d) - 1, 64)
            self.assertTrue(torch.equal(b['input_ids'][r, :n], d[:n]))
            self.assertTrue(torch.equal(b['labels'][r, :n], d[1:n + 1]))
            self.assertTrue((b['labels'][r, n:] == -100).all())
            self.assertTrue((b['input_ids'][r, n:] == 0).all())
        self.assertTrue((b['doc_ids'] == 0).all())

    def test_foreign_mass(self):
        B, H, T, d_h = 2, 2, 6, 4
        q, k = torch.zeros(B, H, T, d_h), torch.randn(B, H, T, d_h)
        doc_ids = torch.tensor([[0, 0, 1, 1, 1, 2], [0] * 6])
        m, u = d2l._foreign_mass(q, k, doc_ids, chunk=4)
        # uniform attention puts exactly the uniform share on other documents
        self.assertAlmostEqual(m[0].item(), u[0].item(), places=6)
        share = np.mean([2 / 3, 2 / 4, 2 / 5, 5 / 6])  # queries 2-5
        self.assertAlmostEqual(u[0].item(), share, places=6)
        self.assertTrue(torch.isnan(m[1]) and torch.isnan(u[1]))
        q = torch.randn(B, H, T, d_h)
        m2, _ = d2l._foreign_mass(q, k, doc_ids, chunk=6)
        s = (q @ k.transpose(-2, -1) / 2).masked_fill(
            ~torch.ones(T, T, dtype=torch.bool).tril(), float('-inf'))
        p = torch.softmax(s, -1)[0]  # window 0, (H, T, T)
        foreign = (doc_ids[0][:, None] != doc_ids[0][None, :]).tril()
        ref = (p * foreign).sum(-1).mean(0)[2:].mean()
        self.assertAlmostEqual(m2[0].item(), ref.item(), places=5)

    def test_throughput_table_cpu(self):
        cfg = tiny_cfg()
        df = d2l._throughput_table(cfg, ('padded', 'packed', 'sdpa-mask'),
                                   (32, 64), steps=3, warmup=1,
                                   shards=self.shards,
                                   device=torch.device('cpu'))
        self.assertEqual(len(df), 6)
        self.assertTrue((df.status == 'ok').all())
        self.assertTrue((df.tok_s_q1 <= df.tok_s).all() and (
            df.tok_s <= df.tok_s_q3).all())
        self.assertTrue((df[df.pipeline == 'padded'].padding > 0).all())
        self.assertTrue((df[df.pipeline != 'padded'].padding == 0).all())

    def test_mask_runs_cpu(self):
        cfg = d2l._mask_run_config(tiny_cfg(), tokens=8 * 32 * 10)
        self.assertEqual((cfg.steps, cfg.warmup, cfg.decay), (10, 0, 1))
        X = torch.from_numpy(np.stack([self.shards.read(i * 32, 32)
                                       for i in range(6)]).astype(np.int64))

        def val(model):  # the training targets of one fixed batch
            b = d2l.PackedStream(self.shards, 32, 8, seed=9).batch(0)
            with torch.no_grad():
                H, W = model(b['input_ids'], d2l.PackCtx(b['doc_ids'],
                                                         b['pos_ids']),
                             return_hidden=True)
                s, n = d2l.chunked_lm_loss(H, W, b['labels'])
            return float(s / n)
        runs = d2l._mask_runs(cfg, (0, 1), self.shards, val, X,
                              torch.device('cpu'), every=5)
        self.assertEqual(runs.table.seed.tolist(), [0, 1])
        self.assertTrue(np.isfinite(runs.table[['masked', 'unmasked']]
                                    .values).all())
        self.assertNotEqual(runs.table.masked[0], runs.table.unmasked[0])
        self.assertEqual(sorted(set(runs.losses.step)), [5, 10])
        self.assertEqual(len(runs.losses), 2 * 2 * 2)
        leak = runs.leak
        self.assertEqual(leak.layer.tolist(), [0])
        self.assertTrue((leak.lo <= leak.leak).all() and (
            leak.leak <= leak.hi).all())
        self.assertTrue(0 < leak.uniform[0] < 1 and (leak.leak > 0).all())

    def test_stale_rerun_file(self):
        """A next-version file in the rerun folder is read only by a run
        whose D2L_RERUN names a campaign that writes it."""
        from unittest import mock
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            d2l._write_result(os.path.join(tmp, 'lm-pdp-throughput-v1.parquet'),
                              'parquet', pd.DataFrame(dict(T=[1024])),
                              dict(config=None, label='smoke'))
            with mock.patch.object(d2l, '_rerun_dir', lambda: tmp), \
                    uncommitted():
                with mock.patch.dict(os.environ,
                                     {'D2L_RERUN': 'decode-rates'}):
                    with self.assertRaises(KeyError):
                        d2l.load_artifact('packing-throughput')
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        tp = d2l.packing_throughput(d2l.r2_config('t1'))
                    self.assertTrue(tp.empty)  # pending, not the stale file
                    self.assertEqual(out.getvalue(),
                                     '[pdp-throughput pending]\n')
                with mock.patch.dict(os.environ,
                                     {'D2L_RERUN': 'pdp-throughput'}):
                    got = d2l.load_artifact('packing-throughput')
                    self.assertEqual(d2l._label_of(got), 'smoke')

    def test_comparison_round_trip(self):
        """packing_comparison loads a committed json-table result as
        PackingRuns and refuses one stored under another configuration."""
        from unittest import mock
        import pandas as pd
        cfg = d2l.r2_config('t1')
        config = dict(cfg=d2l._cfg_json(d2l._mask_run_config(cfg, 1000)),
                      tokens=1000, seeds=[0], leak_windows=4)
        runs = d2l.PackingRuns(pd.DataFrame(dict(seed=[0], masked=[3.5],
                                                 unmasked=[3.6])),
                               pd.DataFrame(dict(arm=['masked'], seed=[0],
                                                 step=[10], loss=[4.0])),
                               pd.DataFrame(dict(layer=[0], leak=[0.1],
                                                 lo=[0.05], hi=[0.2],
                                                 uniform=[0.3],
                                                 windows=[4])), '')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'lm-pdp-mask-runs-v1.json.gz')
            d2l._write_result(path, 'json-table', runs,
                              dict(config=d2l._jsonable(config), label=''))
            with mock.patch.object(d2l, 'DATA_HUB', dict(
                    d2l.DATA_HUB, **{'lm-pdp-mask-runs-v1': ('u', 's')})), \
                    mock.patch.object(d2l, 'download_keyed',
                                      lambda key: path), \
                    mock.patch.dict(os.environ, {'D2L_RERUN': ''}):
                got = d2l.packing_comparison(cfg, tokens=1000, seeds=(0,),
                                             leak_windows=4)
                self.assertIsInstance(got, d2l.PackingRuns)
                self.assertEqual(got.table.unmasked.tolist(), [3.6])
                self.assertEqual(got.leak.hi.tolist(), [0.2])
                with self.assertRaises(ValueError):
                    d2l.packing_comparison(cfg, tokens=2000, seeds=(0,),
                                           leak_windows=4)


@unittest.skipUnless(os.environ.get('D2L_TEST_GPU') == '1'
                     and torch.cuda.is_available(), 'GPU test')
class FlexTest(unittest.TestCase):
    def test_flex_vs_sdpa(self):
        torch.manual_seed(0)
        m = d2l.PackedGPT(256, 128, 4, 2, 2, head_dim=32, max_len=256,
                          backend='sdpa').cuda().eval()
        X = torch.randint(1, 256, (2, 256), device='cuda')
        X[0, [40, 100, 200]] = 0
        X[1, [10, 128]] = 0
        ctx = d2l.PackCtx(*d2l.doc_ids_from_eos(X))
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            y1 = m(X, ctx).float()
            for blk in m.blks:
                blk.attention.backend = 'flex'
            y2 = m(X, ctx).float()
        self.assertLess((y1 - y2).abs().max().item(), 2e-2)

    def test_flex_vs_sdpa_gradients(self):
        """The masked training arm runs through flex forward and backward:
        its parameter gradients match sdpa's in bf16 (2e-2 of the largest
        gradient entry)."""
        torch.manual_seed(0)
        m = d2l.PackedGPT(256, 128, 4, 2, 2, head_dim=32, max_len=256,
                          backend='sdpa').cuda()
        X = torch.randint(1, 256, (2, 256), device='cuda')
        X[0, [40, 100, 200]] = 0
        X[1, [10, 128]] = 0
        Y = torch.roll(X, -1, 1)
        ctx = d2l.PackCtx(*d2l.doc_ids_from_eos(X))
        grads = []
        for backend in ('sdpa', 'flex'):
            for blk in m.blks:
                blk.attention.backend = backend
            m.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                H, W = m(X, ctx, return_hidden=True)
                s, n = d2l.chunked_lm_loss(H, W, Y)
            (s / n).backward()
            grads.append({k: p.grad.float().clone()
                          for k, p in m.named_parameters()})
        for k in grads[0]:
            scale = grads[0][k].abs().max().item()
            gap = (grads[0][k] - grads[1][k]).abs().max().item()
            self.assertLess(gap, 2e-2 * scale + 1e-6, k)


if __name__ == '__main__':
    unittest.main()
