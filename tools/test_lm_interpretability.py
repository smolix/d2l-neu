#!/usr/bin/env python3
"""CPU tests for the library code of sec_lm_interpretability
(chapter_lm-pretrained/interpretability.md): `intervene`, `logit_lens`,
`TunedLens`, `TopKSAE`, `train_sae`, `load_qwen_scope_sae`, `sae_metrics`,
and the closed forms of the section's derivations (attribution patching,
the L1 and top-k codes of an orthonormal dictionary).

The model is a random tiny `d2l.Qwen3GPT` (vocabulary 97, width 32, 4 query
and 2 key-value heads of width 16, 3 blocks, fp32). References are a forward
written out from the block's modules. Nothing is downloaded: the Qwen-Scope
loader reads a fake repository through a patched `hf_hub_download`.

Run: `.venv-lm/bin/python -m unittest tools/test_lm_interpretability.py`,
or `make test-lm`.
"""

import copy
import io
import json
import os
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

V, D, HEADS, KV, HD, L = 97, 32, 4, 2, 16, 3
EOS = 96
KINDS = ('resid_post', 'resid_mid', 'attn_out', 'mlp_out', 'head_z')
SAE_REPO = 'Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50'


def tiny(seed=0, tied=True):
    """A random Qwen3GPT with non-trivial norm gains, frozen, eval mode."""
    torch.manual_seed(seed)
    model = d2l.Qwen3GPT(V, D, HEADS, KV, L, head_dim=HD,
                         tie_embeddings=tied, max_len=256)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(1 + 0.3 * torch.randn_like(p) if 'norm' in name
                    else 0.3 * torch.randn_like(p))
    return model.eval().requires_grad_(False)


def tokens(B=2, T=9, seed=1):
    return torch.randint(0, EOS, (B, T),
                         generator=torch.Generator().manual_seed(seed))


def manual_forward(model, X, edit=None, leaf=None):
    """Logits of model on X written out from the modules of each block.
    edit = (site, fn) overwrites one site; leaf = site replaces that
    activation by a fresh leaf tensor, returned as the second value."""
    T = X.shape[1]
    tables = d2l.qwen3_rope_tables(T, model.head_width, model.rope_theta)
    H, out = model.token_emb(X), {}

    def at(kind, l, x):
        if edit is not None and edit[0] == (kind, l):
            x = edit[1](x)
        if leaf == (kind, l):
            x = out['leaf'] = x.detach().requires_grad_()
        return x

    for l, blk in enumerate(model.blks):
        att = blk.attention
        q, k, v = att.qkv(blk.norm1(H), tables=tables)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=True)
        z = at('head_z', l, z.transpose(1, 2).flatten(2))
        a = at('attn_out', l, att.W_o(z))
        H = at('resid_mid', l, H + a)
        m = at('mlp_out', l, blk.ffn(blk.norm2(H)))
        H = at('resid_post', l, H + m)
    head = model.lm_head.weight if hasattr(model, 'lm_head') else (
        model.token_emb.weight)
    return F.linear(model.norm(H), head), out.get('leaf')


def no_hooks(model):
    return all(not m._forward_hooks and not m._forward_pre_hooks
               for m in model.modules())


class CharTok:
    """One token per printable ASCII character (id = code - 32); EOS 96."""
    eos_token_id = EOS

    def _enc(self, text):
        return [ord(c) - 32 for c in text]

    def __call__(self, text, add_special_tokens=True):
        if isinstance(text, list):
            return {'input_ids': [self._enc(t) for t in text]}
        return {'input_ids': self._enc(text)}

    def decode(self, ids):
        return ''.join(chr(i + 32) for i in ids if i < 95)


# ------------------------------------------------------------ intervene
class TestIntervene(unittest.TestCase):
    def setUp(self):
        self.model, self.X = tiny(), tokens()
        with torch.no_grad():
            self.ref = self.model(self.X)

    def test_manual_forward_is_the_model(self):
        with torch.no_grad():
            torch.testing.assert_close(manual_forward(self.model, self.X)[0],
                                       self.ref, rtol=0, atol=1e-6)

    def test_record_only_is_bitwise(self):  # spec test 1
        sites = [(k, l) for k in KINDS for l in range(L)]
        with torch.no_grad(), d2l.intervene(self.model, record=sites) as ctx:
            out = self.model(self.X)
        self.assertTrue(torch.equal(out, self.ref))
        for s in sites:
            self.assertEqual(len(ctx.cache[s]), 1)
            self.assertFalse(ctx.cache[s][0].requires_grad)
        self.assertEqual(ctx.cache[('head_z', 0)][0].shape, (2, 9, HEADS * HD))
        self.assertTrue(no_hooks(self.model))

    def test_identity_fn(self):  # spec test 1
        for kind in KINDS:
            for l in range(L):
                with torch.no_grad(), d2l.intervene(
                        self.model, {(kind, l): lambda x: x}):
                    out = self.model(self.X)
                self.assertTrue(torch.equal(out, self.ref), (kind, l))

    def test_resid_writes_bitwise_in_bf16(self):
        """Review 18.6 F1: a resid site returned fn(r) - stream and the
        block added the stream back, which is not r in bf16. An identity
        fn, and positions a splice leaves alone, must come back bitwise."""
        model = tiny()
        with torch.no_grad():  # large stream norms, as in real models
            for blk in model.blks:
                blk.attention.W_o.weight.mul_(30)
                blk.ffn.W_2.weight.mul_(30)
        model, X = model.to(torch.bfloat16), tokens(B=4, T=64)
        with torch.no_grad():
            ref = model(X)
            for kind in ('resid_mid', 'resid_post'):
                for l in range(L):
                    with d2l.intervene(model, {(kind, l): lambda x: x}):
                        self.assertTrue(torch.equal(model(X), ref), (kind, l))
            keep = torch.zeros(4, 64, 1, dtype=torch.bool)
            keep[:, 5:9] = True  # splice a few positions only
            later = ('resid_post', 2)
            with d2l.intervene(model, record=[later]) as clean:
                model(X)
            with d2l.intervene(model, {('resid_post', 1): lambda x:
                                       torch.where(keep, 2 * x, x)},
                               record=[later]) as spliced:
                model(X)
        a, b = clean.cache[later][0], spliced.cache[later][0]
        self.assertTrue(torch.equal(a[:, :5], b[:, :5]))  # causal: before
        self.assertFalse(torch.equal(a[:, 5:9], b[:, 5:9]))

    def test_equals_collect_hiddens(self):  # spec test 2
        points = ('attn_out', 'resid_mid', 'mlp_out', 'resid_post')
        sites = [(k, l) for k in points for l in range(L)]
        with torch.no_grad(), d2l.intervene(self.model, record=sites) as ctx:
            self.model(self.X)
        ref = d2l.collect_hiddens(self.model, self.X)
        for k in points:
            ours = torch.stack([ctx.cache[(k, l)][0] for l in range(L)])
            self.assertTrue(torch.equal(ours, ref[k]), k)

    def test_overwrites_equal_manual_forward(self):  # spec test 3
        g = torch.Generator().manual_seed(2)
        delta = torch.randn(D, generator=g)
        for kind in ('resid_mid', 'resid_post', 'attn_out', 'mlp_out'):
            fn = lambda x: x + delta
            with torch.no_grad():
                want, _ = manual_forward(self.model, self.X,
                                         edit=((kind, 1), fn))
                with d2l.intervene(self.model, {(kind, 1): fn}):
                    got = self.model(self.X)
            torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
            self.assertGreater((got - self.ref).abs().max(), 1e-2)
        # resid_mid changes both the FFN input and the stream: an edit of
        # attn_out by the same delta is the same edit
        with torch.no_grad():
            with d2l.intervene(self.model, {('attn_out', 1): fn}):
                a = self.model(self.X)
            with d2l.intervene(self.model, {('resid_mid', 1): fn}):
                b = self.model(self.X)
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5)

    def test_head_z_zeroing_equals_zeroed_W_o(self):  # spec test 4
        for h in range(HEADS):
            def zero(x, h=h):
                x = x.clone()
                x[..., h * HD:(h + 1) * HD] = 0
                return x
            ablated = tiny()
            with torch.no_grad():
                ablated.blks[2].attention.W_o.weight[:, h * HD:(h + 1) * HD] = 0
                want = ablated(self.X)
                with d2l.intervene(self.model, {('head_z', 2): zero}):
                    got = self.model(self.X)
            torch.testing.assert_close(got, want, rtol=0, atol=1e-6)

    def test_hooks_removed_and_errors(self):  # spec test 5
        with self.assertRaisesRegex(ZeroDivisionError, 'boom'):
            with d2l.intervene(self.model, {('resid_post', 2): lambda x: x}):
                raise ZeroDivisionError('boom')  # unfired, but not masked
        self.assertTrue(no_hooks(self.model))
        with self.assertRaisesRegex(RuntimeError, 'never fired'):
            with d2l.intervene(self.model, record=[('mlp_out', 1)]):
                pass
        self.assertTrue(no_hooks(self.model))
        with self.assertRaisesRegex(ValueError, 'None'):
            d2l.intervene(self.model, {('attn_out', 0): None})
        for bad in [('resid', 0), ('attn_out', L), ('attn_out', -1),
                    ['attn_out', 0], 'attn_out', ('attn_out', 1.0)]:
            with self.assertRaises(ValueError):
                d2l.intervene(self.model, record=[bad])
        with self.assertRaisesRegex(ValueError, 'shape'):
            with torch.no_grad(), d2l.intervene(
                    self.model, {('mlp_out', 0): lambda x: x[:, :1]}):
                self.model(self.X)
        with self.assertRaisesRegex(ValueError, 'dtype'):
            with torch.no_grad(), d2l.intervene(
                    self.model, {('mlp_out', 0): lambda x: x.double()}):
                self.model(self.X)
        self.assertTrue(no_hooks(self.model))

    def test_fn_error_propagates(self):
        def bad(x):
            raise KeyError('inside fn')
        with self.assertRaisesRegex(KeyError, 'inside fn'):
            with torch.no_grad(), d2l.intervene(self.model,
                                                {('head_z', 1): bad}):
                self.model(self.X)
        self.assertTrue(no_hooks(self.model))

    def test_nesting_applies_outer_first(self):  # spec test 5
        for kind in KINDS:
            site = (kind, 1)
            with torch.no_grad():
                with d2l.intervene(self.model, {site: lambda x: 2 * x}), \
                        d2l.intervene(self.model, {site: lambda x: x + 1}):
                    nested = self.model(self.X)
                with d2l.intervene(self.model, {site: lambda x: 2 * x + 1}):
                    once = self.model(self.X)
                with d2l.intervene(self.model, {site: lambda x: 2 * (x + 1)}):
                    swapped = self.model(self.X)
            torch.testing.assert_close(nested, once, rtol=0, atol=1e-5)
            self.assertGreater((nested - swapped).abs().max(), 1e-3)

    def test_nested_contexts_read_their_own_hooks(self):
        """Review 18.6 F2, the documented semantics: an outer context's
        resid_post record precedes an inner context's mlp_out edit in the
        same block; the inner record is the stream the model carries."""
        site, edit = ('resid_post', 1), ('mlp_out', 1)
        with torch.no_grad():
            with d2l.intervene(self.model, record=[site]) as outer:
                with d2l.intervene(self.model, {edit: torch.zeros_like},
                                   record=[site]) as inner:
                    self.model(self.X)
            with d2l.intervene(self.model, record=[site]) as clean:
                self.model(self.X)
            with d2l.intervene(self.model, {edit: torch.zeros_like},
                               record=[site]) as edited:
                self.model(self.X)
        self.assertTrue(torch.equal(outer.cache[site][0],
                                    clean.cache[site][0]))
        self.assertTrue(torch.equal(inner.cache[site][0],
                                    edited.cache[site][0]))

    def test_model_checks(self):
        with self.assertRaises(AssertionError):
            d2l.intervene(nn.Linear(2, 2))
        post = d2l.GPT(V, num_hiddens=D, num_heads=4, num_blks=2,
                       pre_norm=False)
        with self.assertRaisesRegex(AssertionError, 'pre-norm'):
            d2l.intervene(post)
        dropped = d2l.GPT(V, num_hiddens=D, num_heads=4, num_blks=2,
                          dropout=0.1)
        with self.assertRaisesRegex(AssertionError, 'dropout'):
            d2l.intervene(dropped.train())
        d2l.intervene(dropped.eval())  # inactive in eval mode

    def test_plain_gpt(self):
        """A d2l.GPT (fused W_qkv, the book's RoPE) has the same sites."""
        torch.manual_seed(0)
        gpt = d2l.GPT(V, num_hiddens=D, num_heads=4, num_blks=2).eval()
        sites = [(k, l) for k in KINDS for l in range(2)]
        with torch.no_grad():
            ref = gpt(self.X)
            with d2l.intervene(gpt, record=sites) as ctx:
                out = gpt(self.X)
        self.assertTrue(torch.equal(out, ref))
        hid = d2l.collect_hiddens(gpt, self.X)
        self.assertTrue(torch.equal(
            torch.stack([ctx.cache[('resid_post', l)][0] for l in range(2)]),
            hid['resid_post']))

    def test_record_grad(self):  # spec test 6
        c, d = 3, 7
        all_sites = [(k, l) for k in KINDS for l in range(L)]
        with d2l.intervene(self.model, record=all_sites,
                           record_grad=True) as ctx:
            out = self.model(self.X)
            (out[:, -1, c] - out[:, -1, d]).sum().backward()
        self.assertTrue(torch.equal(out.detach(), self.ref))
        for site in all_sites:
            logits, a = manual_forward(self.model, self.X, leaf=site)
            want, = torch.autograd.grad(
                (logits[:, -1, c] - logits[:, -1, d]).sum(), a)
            got = ctx.grad_cache[site][0]
            torch.testing.assert_close(got, want, rtol=0, atol=1e-6,
                                       msg=str(site))
            self.assertTrue(torch.equal(ctx.cache[site][0].detach(),
                                        a.detach()), site)
        # one site at a time gives the same gradient
        with d2l.intervene(self.model, record=[('resid_post', 0)],
                           record_grad=True) as one:
            out = self.model(self.X)
            (out[:, -1, c] - out[:, -1, d]).sum().backward()
        torch.testing.assert_close(one.grad_cache[('resid_post', 0)][0],
                                   ctx.grad_cache[('resid_post', 0)][0],
                                   rtol=0, atol=1e-6)
        self.assertFalse(any(p.grad is not None
                             for p in self.model.parameters()))
        with self.assertRaisesRegex(RuntimeError, 'gradients'):
            with torch.no_grad(), d2l.intervene(
                    self.model, record=[('mlp_out', 0)], record_grad=True):
                self.model(self.X)

    def test_record_grad_with_fn(self):
        """A recorded and overwritten site: the cache holds fn's value and
        the gradient is taken there."""
        site, fn = ('resid_mid', 1), lambda x: 0.5 * x
        with d2l.intervene(self.model, {site: fn}, record=[site],
                           record_grad=True) as ctx:
            self.model(self.X)[:, -1, 4].sum().backward()
        logits, a = manual_forward(self.model, self.X, edit=(site, fn),
                                   leaf=site)
        want, = torch.autograd.grad(logits[:, -1, 4].sum(), a)
        torch.testing.assert_close(ctx.grad_cache[site][0], want, rtol=0,
                                   atol=1e-6)
        torch.testing.assert_close(ctx.cache[site][0].detach(), a.detach(),
                                   rtol=0, atol=1e-6)

    def test_stop_after(self):  # spec test 8
        calls = []
        h = self.model.blks[2].norm1.register_forward_pre_hook(
            lambda m, a: calls.append(1))
        try:
            with torch.no_grad(), d2l.intervene(
                    self.model, {('mlp_out', 2): lambda x: x},
                    record=[('attn_out', 1), ('resid_post', 2)],
                    stop_after=('resid_post', 1)) as ctx:
                self.model(self.X)
                raise AssertionError('the forward did not stop')
        finally:
            h.remove()
        self.assertEqual(calls, [])
        self.assertEqual(len(ctx.cache[('resid_post', 1)]), 1)
        self.assertEqual(len(ctx.cache[('attn_out', 1)]), 1)
        self.assertEqual(ctx.cache[('resid_post', 2)], [])
        ref = d2l.collect_hiddens(self.model, self.X)['resid_post'][1]
        self.assertTrue(torch.equal(ctx.cache[('resid_post', 1)][0], ref))
        self.assertTrue(no_hooks(self.model))

    def test_nested_stop(self):
        """An inner context's stop is not swallowed by the outer one, and
        each context swallows only its own."""
        with torch.no_grad():
            with d2l.intervene(self.model, record=[('attn_out', 0)],
                               stop_after=('mlp_out', 0)) as outer:
                with d2l.intervene(self.model, record=[('attn_out', 2)],
                                   stop_after=('attn_out', 2)):
                    self.model(self.X)
                raise AssertionError('not reached')
        self.assertEqual(len(outer.cache[('mlp_out', 0)]), 1)
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError,
                                                     'never fired'):
            with d2l.intervene(self.model, record=[('attn_out', 2)]):
                with d2l.intervene(self.model, stop_after=('attn_out', 0)):
                    self.model(self.X)
        self.assertTrue(no_hooks(self.model))

    def test_cached_generation(self):  # spec test 7
        tok, model = CharTok(), self.model
        prompts = ['ab cd ef', 'gh ij kl']  # equal lengths: one bucket
        T0, n_new = len(prompts[0]), 6
        calls = []
        orig = model.forward_cached
        with mock.patch.object(model, 'forward_cached',
                               side_effect=lambda *a: calls.append(1)
                               or orig(*a)):
            with d2l.intervene(model,
                               record=[('resid_post', 1)]) as ctx:
                gens = d2l.generate(model, tok, prompts, n_new,
                                    eos_token_ids=[])
        cache = ctx.cache[('resid_post', 1)]
        self.assertEqual(len(cache), len(calls))
        self.assertEqual(len(calls), n_new)  # prefill + n_new - 1 steps
        self.assertEqual(cache[0].shape, (2, T0, D))
        self.assertTrue(all(c.shape == (2, 1, D) for c in cache[1:]))
        # the steering rule of the notebook, under the cached path
        g = torch.Generator().manual_seed(3)
        for scale, changes in [(0.5, None), (60.0, True)]:
            v = scale * torch.randn(D, generator=g)

            def steer(x):  # every position but the first
                if x.shape[1] > 1:
                    return torch.cat([x[:, :1], x[:, 1:] + v], 1)
                return x + v
            with d2l.intervene(model, {('resid_post', 1): steer}):
                cached = [r.token_ids for r in d2l.generate(
                    model, tok, prompts, n_new, eos_token_ids=[])]
            X = torch.tensor([tok._enc(p) for p in prompts])

            def add_from_1(x):  # the uncached reference: whole sequences
                y = x.clone()
                y[:, 1:] += v
                return y
            with torch.no_grad():
                for _ in range(n_new):
                    with d2l.intervene(model, {('resid_post', 1): add_from_1}):
                        nxt = model(X)[:, -1].argmax(-1)
                    X = torch.cat([X, nxt[:, None]], 1)
            self.assertEqual(cached, X[:, T0:].tolist())
            if changes:
                self.assertNotEqual(cached, [r.token_ids for r in gens])


# --------------------------------------------------------------- lenses
class TestLenses(unittest.TestCase):
    def test_logit_lens_last_block(self):  # spec test 9
        X = tokens()
        for tied in (True, False):
            model = tiny(tied=tied)
            self.assertEqual(hasattr(model, 'lm_head'), not tied)
            with torch.no_grad(), d2l.intervene(
                    model, record=[('resid_post', L - 1)]) as ctx:
                ref = model(X)
            out = d2l.logit_lens(model, ctx.cache[('resid_post', L - 1)][0])
            self.assertEqual(out.dtype, torch.float32)
            torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)

    def test_norms_without_affine_or_bias(self):
        """Review 18.6 F6: a LayerNorm without bias or affine weights, and
        an RMSNorm without weights, as the model's final norm."""
        torch.manual_seed(0)
        gpt = d2l.GPT(V, num_hiddens=D, num_heads=4, num_blks=1,
                      norm='layer').eval()
        h = torch.randn(2, 3, D)
        for norm in (nn.LayerNorm(D, bias=False),
                     nn.LayerNorm(D, elementwise_affine=False),
                     nn.RMSNorm(D, elementwise_affine=False)):
            gpt.norm = norm
            with torch.no_grad():
                torch.testing.assert_close(
                    d2l.logit_lens(gpt, h),
                    F.linear(norm(h), gpt.token_emb.weight), rtol=0,
                    atol=1e-5)

    def test_bf16_model_lens_in_fp32(self):
        model = tiny().to(torch.bfloat16)
        h = torch.randn(2, 3, D, dtype=torch.bfloat16)
        out = d2l.logit_lens(model, h)
        self.assertEqual(out.dtype, torch.float32)
        ref = d2l.logit_lens(tiny(), h.float())
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_layernorm_gpt(self):
        torch.manual_seed(0)
        gpt = d2l.GPT(V, num_hiddens=D, num_heads=4, num_blks=2,
                      norm='layer').eval()
        X = tokens()
        with torch.no_grad(), d2l.intervene(
                gpt, record=[('resid_post', 1)]) as ctx:
            ref = gpt(X)
        torch.testing.assert_close(
            d2l.logit_lens(gpt, ctx.cache[('resid_post', 1)][0]), ref,
            rtol=0, atol=1e-5)

    def test_direct_logit_attribution(self):  # D1.6a, spec test 9
        model, X = tiny(), tokens()
        sites = [(k, l) for k in ('attn_out', 'mlp_out', 'resid_post')
                 for l in range(L)]
        with torch.no_grad(), d2l.intervene(model, record=sites) as ctx:
            logits = model(X)
            h0 = model.token_emb(X)
        c = {s: ctx.cache[s][0] for s in sites}
        hL = c[('resid_post', L - 1)]
        gamma, E = model.norm.weight, model.token_emb.weight
        rho = (hL.pow(2).mean(-1, keepdim=True) + model.norm.eps).sqrt()
        parts = [h0] + [c[(k, l)] for l in range(L)
                        for k in ('attn_out', 'mlp_out')]
        contrib = [F.linear(p * gamma / rho, E) for p in parts]
        torch.testing.assert_close(sum(contrib), logits, rtol=0, atol=1e-4)
        # the lens at block l has the argmax of the unrescaled partial sum
        for l in range(L):
            h = c[('resid_post', l)]
            self.assertTrue(torch.equal(
                d2l.logit_lens(model, h).argmax(-1),
                F.linear(h * gamma, E).argmax(-1)))

    def test_smallest_case(self):  # D1.6a's numbers
        h0, d1 = torch.tensor([1., 0.]), torch.tensor([0., 3.])
        rho = lambda h: h.pow(2).mean().sqrt()
        h1 = h0 + d1
        final = h1 / rho(h1)
        torch.testing.assert_close(final, torch.tensor([0.447, 1.342]),
                                   atol=1e-3, rtol=0)
        torch.testing.assert_close(h0 / rho(h0), torch.tensor([1.414, 0.]),
                                   atol=1e-3, rtol=0)
        self.assertAlmostEqual(float(rho(h1) / rho(h0)), 3.162, places=3)


class TestTunedLens(unittest.TestCase):
    def setUp(self):
        self.model = tiny()
        sites = [('resid_post', l) for l in range(L)]
        with torch.no_grad(), d2l.intervene(self.model, record=sites) as ctx:
            self.model(tokens())
        self.H = [ctx.cache[s][0] for s in sites]

    def test_fresh_is_logit_lens(self):  # spec test 10
        lens = d2l.TunedLens(D, L)
        self.assertEqual(len(lens.translators), L - 1)
        for l, h in enumerate(self.H):
            self.assertTrue(torch.equal(lens.logits(self.model, h, l),
                                        d2l.logit_lens(self.model, h)))
        self.assertIs(lens(self.H[-1], L - 1), self.H[-1])
        with self.assertRaises(IndexError):
            lens(self.H[0], L)

    def batches(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        return [torch.randint(0, EOS, (2, 12), generator=g) for _ in range(n)]

    def test_fit_reduces_kl_and_fp16_round_trip(self):  # spec test 10
        lens = d2l.TunedLens(D, L)
        losses = lens.fit(self.model, self.batches(1) * 40, steps=40,
                          lr=1e-2)
        self.assertEqual(len(losses), 40)
        self.assertLess(losses[-1], 0.3 * losses[0])
        state = {k: v.half() for k, v in lens.state_dict().items()}
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        back = d2l.TunedLens(D, L)
        back.load_state_dict(torch.load(buf, weights_only=True))
        for l in range(L):
            a, b = lens(self.H[l], l), back(self.H[l], l)
            tol = 1e-3 * (1 + self.H[l].abs().max())
            torch.testing.assert_close(a, b, rtol=0, atol=float(tol))
        with self.assertRaisesRegex(ValueError, 'ran out'):
            lens.fit(self.model, self.batches(2), steps=3)
        with self.assertRaisesRegex(AssertionError, 'freeze'):
            lens.fit(tiny().requires_grad_(True), self.batches(1), steps=1)

    def test_layer_count_must_match(self):
        """Review 18.6 F3: a lens with fewer layers than the model would
        train toward an intermediate block's logit lens."""
        for n in (L - 1, L + 1):
            lens = d2l.TunedLens(D, n)
            with self.assertRaisesRegex(ValueError, 'blocks'):
                lens.fit(self.model, self.batches(2), steps=1)
            with self.assertRaisesRegex(ValueError, 'blocks'):
                lens.logits(self.model, self.H[0], 0)

    def test_per_layer_backward_is_summed_gradient(self):  # spec test 10
        lens = d2l.TunedLens(D, L)
        with torch.no_grad():  # away from zero, so every term matters
            for p in lens.parameters():
                p.copy_(0.05 * torch.randn_like(p))
        X = self.batches(1, seed=5)
        before = copy.deepcopy(lens)  # fit's Adam step moves lens
        captured = {}

        class Capture(torch.optim.Adam):
            def step(self, closure=None):
                for i, p in enumerate(self.param_groups[0]['params']):
                    captured[i] = p.grad.clone()
                return super().step(closure)

        with mock.patch('torch.optim.Adam', Capture):
            lens.fit(self.model, X, steps=1, frac=1.0, chunk=5)
        # reference: the summed loss over all positions >= 1, one backward
        lens, params = before, list(before.parameters())
        sites = [('resid_post', l) for l in range(L)]
        with torch.no_grad(), d2l.intervene(self.model, record=sites) as ctx:
            self.model(X[0])
        H = [ctx.cache[s][0][:, 1:].reshape(-1, D) for s in sites]
        target = F.log_softmax(d2l.logit_lens(self.model, H[-1]), -1)
        total = sum(F.kl_div(F.log_softmax(lens.logits(self.model, H[l], l),
                                           -1), target, reduction='sum',
                             log_target=True) / len(H[0])
                    for l in range(L - 1))
        want = torch.autograd.grad(total, params)
        for i, w in enumerate(want):
            torch.testing.assert_close(captured[i], w, rtol=0, atol=1e-6)


# ------------------------------------------------------------------ SAEs
def app_topk_relu(x, k):
    """The release demo's encoder nonlinearity (app.py at the pinned
    revision): ReLU, then the top k, scattered into zeros."""
    relu_x = torch.relu(x)
    values, indices = torch.topk(relu_x, k, dim=-1)
    out = torch.zeros_like(relu_x)
    out.scatter_(-1, indices, values)
    return out


def readme_topk(x, k):
    """The README snippet: the top k pre-activations, without a ReLU."""
    values, indices = x.topk(k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, indices, values)
    return out


class TestTopKSAE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.sae = d2l.TopKSAE(16, 64, 5)
        with torch.no_grad():
            for p in (self.sae.b_enc, self.sae.b_dec):
                p.copy_(0.3 * torch.randn_like(p))
        self.x = torch.randn(200, 16)

    def test_layout(self):
        shapes = {n: tuple(p.shape) for n, p in self.sae.named_parameters()}
        self.assertEqual(shapes, {'W_dec': (16, 64), 'W_enc': (64, 16),
                                  'b_enc': (64,), 'b_dec': (16,)})
        torch.testing.assert_close(self.sae.W_dec.norm(dim=0),
                                   torch.ones(64))

    def test_encode(self):  # spec test 11
        z = self.sae.encode(self.x)
        self.assertTrue(((z != 0).sum(-1) <= 5).all())
        self.assertTrue((z >= 0).all())
        pre = self.x @ self.sae.W_enc.T + self.sae.b_enc
        self.assertTrue(torch.equal(z, app_topk_relu(pre, 5)))
        rows = (pre > 0).sum(-1) >= 5
        self.assertGreater(int(rows.sum()), 50)
        self.assertTrue(torch.equal(z[rows], readme_topk(pre, 5)[rows]))
        x_hat, z2 = self.sae(self.x)
        self.assertTrue(torch.equal(z2, z))
        torch.testing.assert_close(
            x_hat, z @ self.sae.W_dec.T + self.sae.b_dec)
        with torch.no_grad():  # most pre-activations negative: they differ
            self.sae.b_enc.sub_(4)
        pre = self.x @ self.sae.W_enc.T + self.sae.b_enc
        self.assertTrue(((pre > 0).sum(-1) < 5).any())
        self.assertTrue(torch.equal(self.sae.encode(self.x),
                                    app_topk_relu(pre, 5)))
        self.assertFalse(torch.equal(self.sae.encode(self.x),
                                     readme_topk(pre, 5)))

    def test_center(self):  # spec test 11
        self.sae.center = True
        z = self.sae.encode(self.x)
        self.sae.center = False
        self.assertTrue(torch.equal(z, self.sae.encode(self.x
                                                       - self.sae.b_dec)))
        self.assertFalse(torch.equal(z, self.sae.encode(self.x)))

    def test_ties_by_index(self):  # spec test 11
        pre = torch.tensor([[1., 3., 2., 2., 2., 0.5],
                            [2., 2., 2., 2., 2., 2.],
                            [-1., -1., 0., -1., -2., -3.]])
        z = d2l.TopKSAE.topk_relu(pre, 3)
        self.assertEqual(z.tolist(), [[0., 3., 2., 2., 0., 0.],
                                      [2., 2., 2., 0., 0., 0.],
                                      [0., 0., 0., 0., 0., 0.]])
        # repeated features of the encoder give exactly tied pre-activations
        sae = d2l.TopKSAE(4, 6, 2)
        with torch.no_grad():
            sae.W_enc.copy_(torch.ones(6, 4))
        z = sae.encode(torch.ones(3, 4))
        self.assertEqual((z != 0).nonzero()[:, 1].tolist(), [0, 1] * 3)

    def test_normalize_and_parallel_grad(self):  # spec test 12
        sae = d2l.TopKSAE(16, 64, 5)
        with torch.no_grad():
            sae.W_dec.mul_(torch.rand(64) * 3 + 0.1)
        before = {n: p.detach().clone() for n, p in sae.named_parameters()}
        sae.normalize_decoder_()
        torch.testing.assert_close(sae.W_dec.norm(dim=0), torch.ones(64),
                                   rtol=0, atol=1e-6)
        for n in ('W_enc', 'b_enc', 'b_dec'):
            self.assertTrue(torch.equal(getattr(sae, n), before[n]))
        out = sae(self.x)
        sae.normalize_decoder_()  # already unit: nothing changes
        for a, b in zip(sae(self.x), out):
            self.assertTrue(torch.equal(a, b))
        mse, _ = sae.loss(self.x)
        mse.backward()
        sae.remove_parallel_grad_()
        dots = (sae.W_dec * sae.W_dec.grad).sum(0)
        torch.testing.assert_close(dots, torch.zeros(64), rtol=0, atol=1e-6)
        self.assertGreater(sae.W_dec.grad.abs().max(), 1e-4)

    def test_loss(self):
        mse, aux = self.sae.loss(self.x)
        x_hat, _ = self.sae(self.x)
        want = ((x_hat - self.x) ** 2).sum(-1).mean() / (
            (self.x - self.x.mean(0)) ** 2).sum(-1).mean()
        torch.testing.assert_close(mse, want)
        self.assertEqual(float(aux), 0.0)
        dead = torch.zeros(64, dtype=torch.bool)
        dead[:10] = True
        mse2, aux = self.sae.loss(self.x, dead, k_aux=3)
        self.assertTrue(torch.equal(mse, mse2))
        pre = self.x @ self.sae.W_enc.T + self.sae.b_enc
        e = self.x - x_hat
        z = app_topk_relu(pre[:, :10], 3)
        e_hat = z @ self.sae.W_dec[:, :10].T
        want = ((e_hat - e) ** 2).sum(-1).mean() / (
            (e - e.mean(0)) ** 2).sum(-1).mean()
        torch.testing.assert_close(aux, want)
        aux.backward()  # the residual target is detached
        self.assertIsNone(self.sae.b_dec.grad)
        with torch.no_grad():  # k_aux is capped at the number of dead
            self.assertEqual(float(self.sae.loss(self.x, dead, k_aux=99)[1]),
                             float(self.sae.loss(self.x, dead, k_aux=10)[1]))

    def test_saved_artifact_loads_with_weights_only(self):
        sae = d2l.TopKSAE(16, 64, 5, center=True)
        art = {'state_dict': {k: v.half() for k, v in
                              sae.state_dict().items()},
               'config': {'d_model': 16, 'd_sae': 64, 'k': 5, 'center': True}}
        buf = io.BytesIO()
        torch.save(art, buf)
        buf.seek(0)
        got = torch.load(buf, weights_only=True)
        back = d2l.TopKSAE(**got['config'])
        back.load_state_dict(got['state_dict'])
        self.assertTrue(back.center)
        torch.testing.assert_close(back.W_dec, sae.W_dec, rtol=1e-3,
                                   atol=1e-3)

    def test_train_sae_planted_dictionary(self):  # spec test 13
        g = torch.Generator().manual_seed(0)
        atoms = torch.randn(32, 64, generator=g)
        atoms /= atoms.norm(dim=0)
        N = 20000
        idx = torch.rand(N, 64, generator=g).topk(4, -1).indices
        coef = torch.zeros(N, 64).scatter_(-1, idx, 1 + torch.rand(
            N, 4, generator=g))
        acts = (coef @ atoms.T + 0.5).half()  # an offset for b_dec
        torch.manual_seed(0)
        sae = d2l.TopKSAE(32, 128, 4, center=True)
        losses = d2l.train_sae(sae, acts, steps=1500, batch_size=512,
                               lr=3e-3, dead_after=50)
        self.assertEqual(len(losses), 1500)
        torch.testing.assert_close(sae.W_dec.norm(dim=0), torch.ones(128),
                                   rtol=0, atol=1e-5)
        x = acts[:4096].float()
        with torch.no_grad():
            x_hat, _ = sae(x)
        fvu = ((x_hat - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
        self.assertLess(float(fvu), 0.1)
        cos = (atoms.T @ sae.W_dec.detach()).max(-1).values
        self.assertGreaterEqual(float((cos > 0.9).float().mean()), 0.9)

    def test_seed_determines_trained_sae(self):
        """Review 18.6 F4: TopKSAE(seed=...) and train_sae(seed=...) fix
        the result whatever the global RNG state."""
        acts = torch.randn(4000, 16, generator=torch.Generator()
                           .manual_seed(0)).half()
        runs = []
        for g in (0, 1):
            torch.manual_seed(g)  # whatever a notebook consumed before
            sae = d2l.TopKSAE(16, 32, 4, seed=0)
            runs.append((d2l.train_sae(sae, acts, steps=20, batch_size=256,
                                       seed=0), sae.state_dict()))
        self.assertEqual(runs[0][0], runs[1][0])
        for name, p in runs[0][1].items():
            self.assertTrue(torch.equal(p, runs[1][1][name]), name)
        torch.manual_seed(5)
        a = d2l.TopKSAE(16, 32, 4).W_dec
        torch.manual_seed(5)
        self.assertTrue(torch.equal(a, d2l.TopKSAE(16, 32, 4).W_dec))

    def test_train_sae_initialization(self):
        acts = torch.randn(1000, 8) + 3
        sae = d2l.TopKSAE(8, 16, 2)
        with torch.no_grad():
            sae.W_dec.mul_(5)
        with mock.patch('torch.optim.Adam') as adam:
            d2l.train_sae(sae, acts, steps=0)
        adam.assert_called_once()
        torch.testing.assert_close(sae.b_dec, acts.mean(0))
        torch.testing.assert_close(sae.W_dec.norm(dim=0), torch.ones(16))
        self.assertTrue(torch.equal(sae.W_enc, sae.W_dec.T))


class TestClosedForms(unittest.TestCase):
    def test_attribution_patching_toy(self):  # D1.6b, spec test 16
        def atp(g, a_c, a_x):
            a = a_c.clone().requires_grad_()
            grad, = torch.autograd.grad(g(a), a)
            return -((a_x - a_c) * grad).sum(), g(a_c) - g(a_x)
        w = torch.tensor([0.5, -2.0, 1.5])
        affine = lambda a: (w * a).sum() + 3
        a_c, a_x = torch.tensor([1., 2., 3.]), torch.tensor([0., -1., 4.])
        est, exact = atp(affine, a_c, a_x)
        torch.testing.assert_close(est, exact)
        est, exact = atp(lambda a: (a ** 2).sum(), torch.tensor([1.]),
                         torch.tensor([0.]))
        self.assertEqual((float(exact), float(est)), (1.0, 2.0))
        self.assertEqual(float(est - exact), 0.5 * 2 * 1.0 ** 2)

    def test_attribution_patching_on_the_model(self):
        """AtP from intervene's gradients against the exact noising effect,
        for a small corruption (first order) and a large one."""
        model, X = tiny(), tokens(B=1)
        site, c, d = ('resid_mid', 1), 3, 7
        with d2l.intervene(model, record=[site], record_grad=True) as clean:
            out = model(X)
            (out[:, -1, c] - out[:, -1, d]).sum().backward()
        a_c, grad = clean.cache[site][0].detach(), clean.grad_cache[site][0]
        m_clean = float((out[:, -1, c] - out[:, -1, d]).sum().detach())
        g = torch.Generator().manual_seed(4)
        direction = torch.randn(a_c.shape, generator=g)
        errors = []
        for eps in (1e-3, 1.0):
            a_x = a_c + eps * direction
            with torch.no_grad(), d2l.intervene(model,
                                                {site: lambda x: a_x}):
                out = model(X)
            exact = m_clean - float((out[:, -1, c] - out[:, -1, d]).sum())
            est = float(-((a_x - a_c) * grad).sum())
            errors.append(abs(est - exact) / max(abs(exact), 1e-12))
        self.assertLess(errors[0], 1e-2)
        self.assertGreater(errors[1], errors[0])

    def test_l1_and_topk_codes(self):  # D1.6c, spec test 17
        g = torch.Generator().manual_seed(0)
        W = torch.linalg.qr(torch.randn(64, 64, generator=g,
                                        dtype=torch.float64))[0].float()
        z_star = torch.zeros(64)
        support = torch.tensor([3, 17, 40, 59])
        z_star[support] = 1 + torch.rand(4, generator=g)
        x, lam = W @ z_star, 0.3
        z = torch.zeros(64)  # the nonnegative L1 code by proximal gradient
        for _ in range(100):
            z = (z - 0.5 * (W.T @ (W @ z - x)) - 0.5 * lam).clamp_min(0)
        want = torch.where(z_star > 0, z_star - lam, torch.zeros(64))
        torch.testing.assert_close(z, want, rtol=0, atol=1e-5)
        sae = d2l.TopKSAE(64, 64, 4)
        with torch.no_grad():
            sae.W_dec.copy_(W)
            sae.W_enc.copy_(W.T)
            sae.b_enc.zero_()
            sae.b_dec.zero_()
        with torch.no_grad():
            x_hat, z_top = sae(x)
        torch.testing.assert_close(z_top, z_star, rtol=0, atol=1e-6)
        gap = ((x - W @ z) ** 2).sum() - ((x - x_hat) ** 2).sum()
        self.assertAlmostEqual(float(gap), 4 * lam ** 2, places=5)

    def test_smallest_case(self):  # x = 3w, lambda = 1
        y, lam = 3.0, 1.0
        z_l1 = max(y - lam, 0)
        self.assertEqual((z_l1, (y - z_l1) ** 2, 0.5 * (y - z_l1) ** 2
                          + lam * z_l1), (2.0, 1.0, 2.5))
        c = 10.0  # w -> c w, z -> z / c: same reconstruction, smaller penalty
        self.assertAlmostEqual(lam * y / c, 0.3)

    def test_unit_norm_constraint(self):  # spec test 17
        """A one-feature autoencoder with an L1 penalty: without the norm
        bound the decoder grows to shrink the penalty."""
        g = torch.Generator().manual_seed(0)
        u = torch.randn(8, generator=g)
        u /= u.norm()
        x = (1 + torch.rand(256, 1, generator=g)) * u
        norms = []
        for constrain in (False, True):
            torch.manual_seed(0)
            sae = d2l.TopKSAE(8, 1, 1)
            opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
            for _ in range(300):
                x_hat, z = sae(x)
                loss = 0.5 * ((x_hat - x) ** 2).sum(-1).mean() + 0.5 * z.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if constrain:
                    sae.normalize_decoder_()
            norms.append(float(sae.W_dec.detach().norm()))
        self.assertGreater(norms[0], 1.5)
        self.assertAlmostEqual(norms[1], 1.0, places=5)


# -------------------------------------------------------------- loading
class TestLoadQwenScope(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__('shutil').rmtree(self.dir))
        self.cfg = {'model_type': 'topk_sae', 'd_model': 8, 'd_sae': 24,
                    'k': 3, 'layers': [0, 1, 2], 'hook_point': 'resid_post'}
        self.calls = []

    def write(self, state, cfg=None):
        with open(os.path.join(self.dir, 'config.json'), 'w') as f:
            json.dump(cfg or self.cfg, f)
        torch.save(state, os.path.join(self.dir, 'layer1.sae.pt'))

    def fake_download(self, repo_id, filename, revision=None, **kw):
        self.calls.append((repo_id, filename, revision))
        return os.path.join(self.dir, filename)

    def load(self, layer=1, **kw):
        with mock.patch('huggingface_hub.hf_hub_download',
                        self.fake_download):
            return d2l.load_qwen_scope_sae(SAE_REPO, layer, **kw)

    def good_state(self):
        g = torch.Generator().manual_seed(0)
        return {'W_enc': torch.randn(24, 8, generator=g),
                'W_dec': torch.randn(8, 24, generator=g),
                'b_enc': torch.randn(24, generator=g),
                'b_dec': torch.randn(8, generator=g)}

    def test_load(self):  # spec test 14
        state = self.good_state()
        self.write(state)
        sae = self.load(center=True)
        self.assertIsInstance(sae, d2l.TopKSAE)
        self.assertEqual((sae.k, sae.d_model, sae.d_sae, sae.center),
                         (3, 8, 24, True))
        for name, t in state.items():
            self.assertTrue(torch.equal(getattr(sae, name), t))
            self.assertFalse(getattr(sae, name).is_meta)
        pinned = d2l.HF_REVISIONS[SAE_REPO]
        self.assertEqual(self.calls, [(SAE_REPO, 'config.json', pinned),
                                      (SAE_REPO, 'layer1.sae.pt', pinned)])
        x = torch.randn(5, 8)
        torch.testing.assert_close(
            sae.encode(x), app_topk_relu((x - state['b_dec'])
                                         @ state['W_enc'].T
                                         + state['b_enc'], 3))
        self.assertEqual(self.load(revision='abc', dtype=torch.float16)
                         .W_enc.dtype, torch.float16)
        self.assertEqual(self.calls[-1][2], 'abc')

    def test_rejects(self):  # spec test 14
        state = self.good_state()
        self.write(state)
        with self.assertRaisesRegex(ValueError, 'no layer 5'):
            self.load(layer=5)
        self.write({**state, 'W_dec': torch.zeros(24, 8)})
        with self.assertRaisesRegex(ValueError, 'W_dec'):
            self.load()
        self.write({k: v for k, v in state.items() if k != 'b_enc'})
        with self.assertRaises(KeyError):
            self.load()
        self.write({**state, 'extra': torch.zeros(1)})
        with self.assertRaises(KeyError):
            self.load()
        self.write(state, cfg={**self.cfg, 'model_type': 'jumprelu_sae'})
        with self.assertRaisesRegex(ValueError, 'top-k'):
            self.load()

    def test_refuses_pickled_objects(self):  # spec test 14
        self.write(self.good_state())
        with open(os.path.join(self.dir, 'layer1.sae.pt'), 'wb') as f:
            pickle.dump({'W_enc': io.StringIO('not a tensor')}, f, protocol=2)
        with self.assertRaises(pickle.UnpicklingError):
            self.load()


# --------------------------------------------------------------- metrics
class TestSAEMetrics(unittest.TestCase):
    def setUp(self):
        self.model = tiny()
        g = torch.Generator().manual_seed(7)
        self.batches = [torch.randint(0, EOS, (3, 10), generator=g)
                        for _ in range(2)]
        self.layer = 1

    def stream(self):
        site = ('resid_post', self.layer)
        xs = []
        with torch.no_grad():
            for X in self.batches:
                with d2l.intervene(self.model, record=[site]) as ctx:
                    self.model(X)
                xs.append(ctx.cache[site][0])
        return xs

    def test_exact_sae(self):  # spec test 15
        shift = float(max(x.abs().max() for x in self.stream())) + 1
        sae = d2l.TopKSAE(D, D, D)
        with torch.no_grad():  # pre = x + shift > 0: every feature fires
            sae.W_enc.copy_(torch.eye(D))
            sae.W_dec.copy_(torch.eye(D))
            sae.b_enc.fill_(shift)
            sae.b_dec.fill_(-shift)
        m = d2l.sae_metrics(sae, self.model, self.batches, self.layer)
        self.assertLess(m['fvu'], 1e-8)
        self.assertAlmostEqual(m['loss_recovered'], 1.0, places=3)
        self.assertAlmostEqual(m['ce_splice'], m['ce_orig'], places=4)
        self.assertGreater(m['ce_mean_ablate'], m['ce_orig'] + 0.05)
        self.assertEqual(m['l0'], D)
        self.assertEqual(m['unfired_frac'], 0.0)
        np.testing.assert_array_equal(m['density'], np.ones(D))
        self.assertEqual(m['n_tokens'], 2 * 3 * 9)  # position 0 excluded
        self.assertEqual(m['per_doc']['sse'].shape, (6,))
        np.testing.assert_array_equal(m['per_doc']['n'], np.full(6, 9))
        self.assertAlmostEqual(
            float(m['per_doc']['ce_orig'].mean()), m['ce_orig'], places=5)

    def test_mean_sae(self):  # spec test 15
        xs = self.stream()
        mean = torch.cat([x[:, 1:].reshape(-1, D) for x in xs]).double()
        sae = d2l.TopKSAE(D, 64, 4)
        with torch.no_grad():
            sae.W_dec.zero_()
            sae.b_dec.copy_(mean.mean(0).float())
        m = d2l.sae_metrics(sae, self.model, self.batches, self.layer)
        self.assertAlmostEqual(m['loss_recovered'], 0.0, places=4)
        self.assertAlmostEqual(m['fvu'], 1.0, places=4)
        self.assertEqual(m['l0'], 4)
        self.assertAlmostEqual(m['unfired_frac'] * 64
                               + (m['density'] > 0).sum(), 64)

    def test_outliers_and_width(self):
        """A position whose stream norm exceeds 10 times the median is left
        out of every statistic and is not spliced."""
        model = tiny()
        with torch.no_grad():
            model.token_emb.weight[5] *= 1000
        batches = [b.clone() for b in self.batches]
        batches[0][0, 4] = batches[1][2, 7] = 5
        sae = d2l.TopKSAE(D, D, D)
        with torch.no_grad():
            sae.W_dec.zero_()  # x_hat = 0: a splice would change the run
        m = d2l.sae_metrics(sae, model, iter(batches), self.layer)
        self.assertEqual(m['n_tokens'], 2 * 3 * 9 - 2)
        self.assertEqual(m['per_doc']['n'].tolist(), [8, 9, 9, 9, 9, 8])
        with self.assertRaisesRegex(ValueError, 'width'):
            d2l.sae_metrics(d2l.TopKSAE(D + 1, 8, 2), model, batches, 1)

    def test_per_doc_ce_counts(self):
        """Review 18.6 F5: n_ce counts the kept positions that have a next
        token, so n_ce-weighted per-row CE is the pooled CE (an outlier at
        the last position counts in n but not in n_ce); a row with no such
        position has CE NaN, not 0."""
        model = tiny()
        with torch.no_grad():
            model.token_emb.weight[5] *= 1000
        batches = [b.clone() for b in self.batches]
        batches[0][0, 4] = batches[0][1, 9] = 5  # 9: the last position
        batches[1][2, 1:] = 5  # every position >= 1 an outlier
        batches[1][2, 1] = 6  # ... but the first
        sae = d2l.TopKSAE(D, D, D)
        m = d2l.sae_metrics(sae, model, batches, self.layer)
        pd = m['per_doc']
        self.assertEqual(pd['n'].tolist(), [8, 8, 9, 9, 9, 1])
        self.assertEqual(pd['n_ce'].tolist(), [7, 8, 8, 8, 8, 1])
        for key in ('ce_orig', 'ce_splice'):
            w = pd['n_ce']
            weighted = float((pd[key] * w).sum() / w.sum())
            self.assertLess(abs(weighted / m[key] - 1), 1e-6, key)
            n_weighted = float((pd[key] * pd['n']).sum() / pd['n'].sum())
            self.assertGreater(abs(n_weighted / m[key] - 1), 1e-4, key)
        batches[1][2, 1] = 5  # now no position of row 6 is kept
        pd = d2l.sae_metrics(sae, model, batches, self.layer)['per_doc']
        self.assertEqual(pd['n_ce'][5], 0)
        self.assertTrue(np.isnan(pd['ce_orig'][5]))
        self.assertTrue(np.isnan(pd['ce_splice'][5]))


if __name__ == '__main__':
    unittest.main()
