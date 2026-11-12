#!/usr/bin/env python3
"""CPU tests for the library code of sec_lm_d2l_checkpoint
(chapter_lm-pretrained/d2l-checkpoint.md): the Qwen3 port
(`qwen3_rope_tables`, `Qwen3Attention`, `SwiGLU`, `Qwen3GPT`,
`qwen3_gpt_kwargs`, `qwen3_rope_permutation`, `map_qwen3_state_dict`,
`_qwen3_files`, `load_qwen3`), the layer-by-layer comparison
(`collect_hiddens`, `layer_errors`, `first_divergence`) and the part's
sampler (`Generation`, `truncated_probs`, `sample_logits`, `generate`).

Reference models are tiny random `transformers.Qwen3ForCausalLM`s (eager
attention) saved to a temporary directory; the configurations of the real
checkpoints are the committed fixtures of tools/lm/fixtures/hf_models/.
Nothing is downloaded.

Run: `.venv-lm/bin/python -m unittest tools/test_lm_pretrained_port.py`, or
`make test-lm`.
"""

import collections
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

TOOLS = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(TOOLS, 'lm', 'fixtures', 'hf_models')
QWEN3 = ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3-4B',
         'Qwen/Qwen3-8B']
POINTS = ('attn_out', 'resid_mid', 'mlp_out', 'resid_post')
TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=192,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            num_hidden_layers=2, rope_theta=1e6, rms_norm_eps=1e-6,
            tie_word_embeddings=True)
PAD, EOS = 0, 1


def fixture_config(model_id):
    with open(os.path.join(FIXTURES, model_id, 'config.json')) as fh:
        return json.load(fh)


def tiny_hf(seed=0, dtype=torch.float32, **overrides):
    """A random Qwen3ForCausalLM with non-trivial norm gains, eager
    attention, in eval mode."""
    import transformers
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(**{**TINY, **overrides},
                                      attn_implementation='eager')
    model = transformers.Qwen3ForCausalLM(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'norm' in name:  # ones would hide a missing permutation
                p.copy_(1 + 0.3 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
    model.generation_config.eos_token_id = EOS
    model.generation_config.pad_token_id = PAD
    return model.to(dtype).eval()


def tiny_tokenizer(vocab_size=128):
    """Whitespace word-level tokenizer over the ids of the tiny models:
    `<pad>` 0, `<eos>` 1, `<s>` 2, `</s>` 3, and `w<i>` for every other i;
    its chat template marks enable_thinking=False with `w5`."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {'<pad>': PAD, '<eos>': EOS, '<s>': 2, '</s>': 3,
             **{f'w{i}': i for i in range(4, vocab_size)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token='<pad>'))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.decoder = decoders.WordPiece(prefix='##')  # joins words by spaces
    template = ("{% for m in messages %}<s> {{ m['content'] }} </s> "
                "{% endfor %}{% if enable_thinking is defined and "
                "enable_thinking is false %}w5 {% endif %}"
                "{% if add_generation_prompt %}<s>{% endif %}")
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token='<pad>',
                                   eos_token='<eos>', chat_template=template)


def words(ids):
    return ' '.join(f'w{i}' for i in ids)


@contextlib.contextmanager
def saved(model):
    """Save a Transformers model and point d2l._qwen3_files at it."""
    path = tempfile.mkdtemp()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            model.save_pretrained(path)
        files = (os.path.join(path, 'config.json'),
                 os.path.join(path, 'generation_config.json'),
                 [os.path.join(path, 'model.safetensors')])
        with mock.patch.object(d2l, '_qwen3_files',
                               return_value=files) as patched:
            yield patched
    finally:
        shutil.rmtree(path)


def port_of(hf_model, dtype=torch.float32, rope_layout='permute'):
    """The Qwen3GPT with the weights of a Transformers model, on the CPU."""
    cfg = hf_model.config.to_dict()
    with torch.device('meta'):
        model = d2l.Qwen3GPT(**d2l.qwen3_gpt_kwargs(cfg))
    state = d2l.map_qwen3_state_dict(
        {k: v.detach().clone() for k, v in hf_model.state_dict().items()},
        cfg, rope_layout)
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()},
                          strict=True, assign=True)
    return model.eval()


def rotate_half_rope(x, cos, sin):
    """Transformers' RoPE: pairs (i, i + d/2); cos, sin are (T, d/2)."""
    x1, x2 = x.chunk(2, -1)
    cos, sin = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


def sample_next_probs(logits, temperature=1.0, k=None, p=None, min_p=None):
    """sec_decoding's sample_next truncation in NumPy, returned as the
    distribution it draws from (stable sort, as truncated_probs's ties)."""
    logits = np.asarray(logits, dtype=np.float64)
    probs = np.exp((logits - logits.max()) / temperature)
    order = np.argsort(-probs, kind='stable')
    probs = probs[order] / probs.sum()
    keep = len(probs)
    if k is not None:
        keep = min(keep, k)
    if p is not None:
        keep = min(keep, int(np.searchsorted(probs.cumsum(), p)) + 1)
    if min_p is not None:
        keep = min(keep, int((probs >= min_p * probs[0]).sum()))
    out = np.zeros_like(probs)
    out[order[:keep]] = probs[:keep] / probs[:keep].sum()
    return out


TOY = torch.tensor([0.45, 0.25, 0.15, 0.10, 0.05]).log()


class Counter(nn.Module):
    """Greedy continuation of token t is t + 1 (mod vocab); a stand-in for
    any module mapping ids to logits."""
    def __init__(self, vocab_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, X):
        return self.scale * F.one_hot((X + 1) % self.vocab_size,
                                      self.vocab_size).float()


class Fixed(nn.Module):
    """The same next-token logits for every row and position, so a row's
    samples depend only on its random draws."""
    def __init__(self, vocab_size=128, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.logits = nn.Parameter(torch.randn(vocab_size, generator=g))

    def forward(self, X):
        logits = self.logits.clone()
        logits[EOS] = -torch.inf  # never stop early
        return logits.expand(*X.shape, -1)


class TestRopeTables(unittest.TestCase):
    def test_equals_transformers(self):
        import transformers
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding)
        config = transformers.Qwen3Config(
            hidden_size=2048, num_attention_heads=16, head_dim=128,
            rope_theta=1e6, max_position_embeddings=4096)
        rotary = Qwen3RotaryEmbedding(config)
        x = torch.zeros(1, 4096, 128)
        cos_hf, sin_hf = rotary(x, torch.arange(4096)[None])
        cos, sin = d2l.qwen3_rope_tables(4096, 128, 1e6)
        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(cos.shape, (4096, 64))
        self.assertTrue(torch.equal(cos, cos_hf[0, :, :64]))
        self.assertTrue(torch.equal(sin, sin_hf[0, :, :64]))
        # The book's formula rounds 24 of the 64 frequencies differently
        cos_b, sin_b = d2l.rope_tables(4096, 128, base=1e6)
        differ = int((cos_b != cos).sum() + (sin_b != sin).sum())
        self.assertGreater(differ, 0)
        angle = torch.atan2(sin, cos) - torch.atan2(sin_b, cos_b)
        angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
        self.assertGreater(angle.abs()[-1].max().item(), 1e-5)

    def test_offset(self):
        cos, sin = d2l.qwen3_rope_tables(10, 16, 1e4)
        cos5, sin5 = d2l.qwen3_rope_tables(5, 16, 1e4, offset=5)
        self.assertTrue(torch.equal(cos[5:], cos5))
        self.assertTrue(torch.equal(sin[5:], sin5))

    def test_meta_context(self):  # the frequencies stay on the CPU
        with torch.device('meta'):
            cos, _ = d2l.qwen3_rope_tables(4, 8, 1e4, device='cpu')
        self.assertTrue(torch.equal(cos, d2l.qwen3_rope_tables(4, 8, 1e4)[0]))


class TestQwen3Attention(unittest.TestCase):
    def test_shapes(self):
        with torch.device('meta'):
            attn = d2l.Qwen3Attention(1024, 16, 8, head_dim=128, qk_norm=True,
                                      rope_theta=1e6)
        shapes = {n: tuple(p.shape) for n, p in attn.named_parameters()}
        self.assertEqual(shapes, {
            'W_q.weight': (2048, 1024), 'W_k.weight': (1024, 1024),
            'W_v.weight': (1024, 1024), 'W_o.weight': (1024, 2048),
            'q_norm.weight': (128,), 'k_norm.weight': (128,)})
        self.assertEqual((attn.head_dim, attn.num_heads, attn.num_kv_heads,
                          attn.rope_theta, attn.kv_hiddens, attn.rope),
                         (128, 16, 8, 1e6, 1024, True))
        self.assertEqual(attn.q_norm.eps, 1e-6)
        self.assertIsInstance(attn, d2l.CausalAttention)
        self.assertFalse(hasattr(attn, 'W_qkv'))
        no_norm = d2l.Qwen3Attention(64, 4, 2)
        self.assertFalse(hasattr(no_norm, 'q_norm'))
        self.assertEqual(no_norm.head_dim, 16)
        with self.assertRaises(ValueError):
            d2l.Qwen3Attention(64, 4, 2, causal=False)

    def test_features_off_equals_causal_attention(self):
        torch.manual_seed(0)
        book = d2l.CausalAttention(64, 4, rope=True, num_kv_heads=2)
        ours = d2l.Qwen3Attention(64, 4, 2, rope_theta=1e4)
        with torch.no_grad():
            W_q, W_k, W_v = book.W_qkv.weight.split([64, 32, 32])
            ours.W_q.weight.copy_(W_q)
            ours.W_k.weight.copy_(W_k)
            ours.W_v.weight.copy_(W_v)
            ours.W_o.weight.copy_(book.W_o.weight)
        X = torch.randn(3, 20, 64)
        tables = d2l.rope_tables(20, 16)
        diff = (ours(X, tables=tables) - book(X, tables=tables)).abs().max()
        self.assertLessEqual(diff.item(), 1e-6)

    def test_forward_step_matches_forward(self):
        torch.manual_seed(1)
        attn = d2l.Qwen3Attention(64, 4, 2, head_dim=24, qk_norm=True,
                                  rope_theta=1e6).eval()
        X = torch.randn(2, 15, 64)
        full = attn(X)
        ks, vs = torch.zeros(2, 2, 15, 24), torch.zeros(2, 2, 15, 24)
        outs = [attn.forward_step(X[:, :7], (ks, vs, 0))]
        for t in range(7, 12):
            outs.append(attn.forward_step(X[:, t:t + 1], (ks, vs, t)))
        outs.append(attn.forward_step(X[:, 12:15], (ks, vs, 12)))
        diff = (torch.cat(outs, 1) - full).abs().max()
        self.assertLessEqual(diff.item(), 1e-5)
        # tables of the step's positions, built by the caller
        ks2, vs2 = torch.zeros_like(ks), torch.zeros_like(vs)
        attn.forward_step(X[:, :12], (ks2, vs2, 0))
        step = attn.forward_step(X[:, 12:15], (ks2, vs2, 12),
                                 d2l.qwen3_rope_tables(3, 24, 1e6, 12))
        self.assertLessEqual((step - outs[-1]).abs().max().item(), 1e-6)

    def test_gqa_grouping(self):
        # Uniform attention (W_q = 0, no RoPE) and KV head j's values all
        # equal to j + 1: query head h must read KV head h // (H / H_kv)
        attn = d2l.Qwen3Attention(8, 4, 2, head_dim=4, rope=False)
        with torch.no_grad():
            attn.W_q.weight.zero_()
            attn.W_v.weight.copy_(torch.repeat_interleave(
                torch.tensor([1.0, 2.0]), 4)[:, None].expand(8, 8) / 8)
        heads = []
        attn.W_o.register_forward_pre_hook(lambda m, a: heads.append(a[0]))
        attn(torch.ones(1, 5, 8))
        per_head = heads[0].reshape(1, 5, 4, 4)[0, :, :, 0]  # (T, heads)
        expected = torch.tensor([1.0, 1.0, 2.0, 2.0]).expand(5, 4)
        self.assertTrue(torch.allclose(per_head, expected, atol=1e-6))

    def test_default_tables(self):
        torch.manual_seed(2)
        attn = d2l.Qwen3Attention(64, 4, 2, qk_norm=True, rope_theta=1e6)
        X = torch.randn(2, 9, 64)
        self.assertTrue(torch.equal(
            attn(X), attn(X, tables=d2l.qwen3_rope_tables(9, 16, 1e6))))
        q, k, _ = attn.qkv(X[:, 4:], 4)
        q2, k2, _ = attn.qkv(X[:, 4:],
                             tables=d2l.qwen3_rope_tables(5, 16, 1e6, 4))
        self.assertTrue(torch.equal(q, q2) and torch.equal(k, k2))

    def test_kv_cache_bytes(self):
        cfg = fixture_config('Qwen/Qwen3-0.6B')
        with torch.device('meta'):
            model = d2l.Qwen3GPT(**d2l.qwen3_gpt_kwargs(cfg))
        self.assertEqual(d2l.kv_cache_bytes(model, 1, bytes_per_elem=2),
                         114688)


class TestSwiGLU(unittest.TestCase):
    def test_swiglu(self):
        torch.manual_seed(0)
        ffn = d2l.SwiGLU(16, 40)
        self.assertIsInstance(ffn, d2l.FeedForward)
        self.assertEqual(ffn.act, 'swiglu')
        self.assertEqual({n: tuple(p.shape) for n, p in ffn.named_parameters()},
                         {'W_g.weight': (40, 16), 'W_1.weight': (40, 16),
                          'W_2.weight': (16, 40)})
        self.assertEqual(d2l.num_params(ffn), 3 * 16 * 40)
        X = torch.randn(3, 16)
        ref = ffn.W_2.weight @ (F.silu(ffn.W_g.weight @ X.T)
                                * (ffn.W_1.weight @ X.T))
        self.assertLessEqual((ffn(X) - ref.T).abs().max().item(), 1e-6)
        self.assertEqual(d2l.num_params(d2l.SwiGLU(16, 40, bias=True)),
                         3 * 16 * 40 + 2 * 40 + 16)


def small_gpt_and_port():
    """A small d2l.GPT and a Qwen3GPT with its features off and its weights."""
    torch.manual_seed(0)
    gpt = d2l.GPT(100, num_hiddens=64, num_heads=4, num_blks=2).eval()
    port = d2l.Qwen3GPT(100, 64, 4, 4, 2, qk_norm=False, rope_theta=1e4,
                        norm_eps=1e-5, ffn_hidden=round(8 * 64 / 3)).eval()
    state = {}
    for name, W in gpt.state_dict().items():
        if name.endswith('W_qkv.weight'):
            prefix = name[:-len('W_qkv.weight')]
            for part, rows in zip(('W_q', 'W_k', 'W_v'), W.split(64)):
                state[f'{prefix}{part}.weight'] = rows
        else:
            state[name] = W
    port.load_state_dict(state, strict=True)
    return gpt, port


class TestQwen3GPT(unittest.TestCase):
    def test_generalizes_gpt(self):
        gpt, port = small_gpt_and_port()
        self.assertIsInstance(port, d2l.GPT)
        X = torch.randint(0, 100, (4, 32))
        diff = (port(X, tables=d2l.rope_tables(32, 16)) - gpt(X)).abs().max()
        self.assertLessEqual(diff.item(), 1e-5)
        self.assertEqual((port.pos, port.max_len, port.head_width),
                         ('rope', 40960, 16))

    def test_cached_forward(self):
        torch.manual_seed(3)
        for tie in (True, False):
            model = d2l.Qwen3GPT(50, 32, 4, 2, 2, head_dim=16,
                                 tie_embeddings=tie).eval()
            X = torch.randint(0, 50, (2, 20))
            with torch.no_grad():
                full = model(X)
            ks, vs = model.init_cache(2, 20)
            self.assertEqual(ks.shape, (2, 2, 2, 20, 16))
            logits, ks, vs = model.forward_cached(X[:, :9], ks, vs, 0)
            rows = [logits]
            for t in range(9, 17):
                logits, ks, vs = model.forward_cached(X[:, t:t + 1], ks, vs, t)
                rows.append(logits)
            logits, ks, vs = model.forward_cached(X[:, 17:20], ks, vs, 17)
            rows.append(logits)
            expected = full[:, [8, *range(9, 17), 19]]
            diff = (torch.stack(rows, 1) - expected).abs().max()
            self.assertLessEqual(diff.item(), 1e-5, f'tie={tie}')

    def test_generate_cached(self):
        torch.manual_seed(4)
        model = d2l.Qwen3GPT(50, 32, 4, 2, 2, head_dim=16).eval()
        prefix = [3, 1, 4, 1, 5]
        self.assertEqual(model.generate_cached(prefix, 12, top_k=1),
                         model.generate(prefix, 12, top_k=1))

    def test_untied_head(self):
        torch.manual_seed(5)
        tied = d2l.Qwen3GPT(50, 32, 4, 2, 2).eval()
        untied = d2l.Qwen3GPT(50, 32, 4, 2, 2, tie_embeddings=False).eval()
        state = tied.state_dict()
        state['lm_head.weight'] = state['token_emb.weight'].clone()
        untied.load_state_dict(state, strict=True)
        X = torch.randint(0, 50, (2, 7))
        with torch.no_grad():
            self.assertTrue(torch.allclose(untied(X), tied(X), atol=1e-6))
        self.assertEqual(untied.lm_head.weight.shape, (50, 32))
        self.assertFalse(hasattr(tied, 'lm_head'))

    def test_state_dict_names(self):
        model = d2l.Qwen3GPT(50, 32, 4, 2, 2, head_dim=16)
        layer = ['norm1.weight', 'norm2.weight', 'attention.W_q.weight',
                 'attention.W_k.weight', 'attention.W_v.weight',
                 'attention.W_o.weight', 'attention.q_norm.weight',
                 'attention.k_norm.weight', 'ffn.W_g.weight', 'ffn.W_1.weight',
                 'ffn.W_2.weight']
        expected = (['token_emb.weight']
                    + [f'blks.{i}.{n}' for i in range(2) for n in layer]
                    + ['norm.weight'])
        self.assertEqual(sorted(model.state_dict()), sorted(expected))
        untied = d2l.Qwen3GPT(50, 32, 4, 2, 2, tie_embeddings=False)
        self.assertEqual(sorted(untied.state_dict()),
                         sorted(expected + ['lm_head.weight']))
        attn = model.blks[0].attention
        self.assertEqual((attn.head_dim, attn.num_heads, attn.num_kv_heads,
                          attn.rope_theta), (16, 4, 2, 1e6))
        norms = [model.norm] + [n for b in model.blks
                                for n in (b.norm1, b.norm2, b.attention.q_norm,
                                          b.attention.k_norm)]
        self.assertTrue(all(n.eps == 1e-6 for n in norms))
        self.assertTrue(all(b.pre_norm for b in model.blks))

    def test_meta_and_parameter_counts(self):
        expected = {'Qwen/Qwen3-0.6B': 596049920,
                    'Qwen/Qwen3-8B': 8190735360}
        for model_id in QWEN3:
            cfg = fixture_config(model_id)
            with torch.device('meta'):
                model = d2l.Qwen3GPT(**d2l.qwen3_gpt_kwargs(cfg))
            self.assertTrue(all(p.is_meta for p in model.parameters()))
            self.assertEqual(d2l.num_params(model), d2l.config_params(cfg),
                             model_id)
            if model_id in expected:
                self.assertEqual(d2l.num_params(model), expected[model_id])
            self.assertEqual(hasattr(model, 'lm_head'),
                             not cfg['tie_word_embeddings'])
        cfg = fixture_config('Qwen/Qwen3-0.6B')
        with torch.device('meta'):
            model = d2l.Qwen3GPT(**d2l.qwen3_gpt_kwargs(cfg))
        self.assertEqual(d2l.num_params(model.blks[0]), 15730944)

    def test_cached_path_calls_modules(self):
        # sec_lm_interpretability hooks norm1, norm2, ffn and W_o; they must
        # fire on the cached path as on the full forward
        torch.manual_seed(7)
        model = d2l.Qwen3GPT(50, 32, 4, 2, 2, head_dim=16).eval()
        counts = collections.Counter()
        for i, blk in enumerate(model.blks):
            for name in ('norm1', 'norm2', 'ffn'):
                getattr(blk, name).register_forward_hook(
                    lambda m, a, y, key=(name, i): counts.update([key]))
            blk.attention.W_o.register_forward_hook(
                lambda m, a, y, key=('W_o', i): counts.update([key]))
        ks, vs = model.init_cache(1, 8)
        X = torch.randint(0, 50, (1, 8))
        model.forward_cached(X[:, :5], ks, vs, 0)
        for t in range(5, 8):
            model.forward_cached(X[:, t:t + 1], ks, vs, t)
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(len(counts), 8)

    def test_logits_to_keep(self):
        torch.manual_seed(6)
        model = d2l.Qwen3GPT(50, 32, 4, 2, 2).eval()
        X = torch.randint(0, 50, (2, 11))
        with torch.no_grad():
            full, last3 = model(X), model(X, logits_to_keep=3)
        self.assertEqual(last3.shape, (2, 3, 50))
        self.assertTrue(torch.allclose(last3, full[:, -3:], atol=1e-6))


class TestKwargs(unittest.TestCase):
    def test_fixtures(self):
        self.assertEqual(
            d2l.qwen3_gpt_kwargs(fixture_config('Qwen/Qwen3-0.6B')),
            dict(vocab_size=151936, num_hiddens=1024, num_heads=16,
                 num_kv_heads=8, num_blks=28, head_dim=128, qk_norm=True,
                 rope_theta=1e6, ffn_hidden=3072, norm_eps=1e-6,
                 tie_embeddings=True, max_len=40960, bias=False))
        rows = {model_id: d2l.qwen3_gpt_kwargs(fixture_config(model_id))
                for model_id in QWEN3}
        self.assertEqual(
            [(r['num_hiddens'], r['num_blks'], r['ffn_hidden'],
              r['tie_embeddings']) for r in rows.values()],
            [(1024, 28, 3072, True), (2048, 28, 6144, True),
             (2560, 36, 9728, True), (4096, 36, 12288, False)])
        self.assertTrue(all(r['head_dim'] == 128 and r['rope_theta'] == 1e6
                            for r in rows.values()))

    def test_v4_and_v5_configs(self):
        import transformers
        for model_id in QWEN3:
            raw = fixture_config(model_id)
            config = transformers.Qwen3Config.from_dict(raw)
            self.assertNotIn('rope_theta', config.to_dict())  # v5 layout
            self.assertEqual(d2l.qwen3_gpt_kwargs(config),
                             d2l.qwen3_gpt_kwargs(raw))
            self.assertEqual(d2l.qwen3_gpt_kwargs(config.to_dict()),
                             d2l.qwen3_gpt_kwargs(raw))

    def test_raises(self):
        raw = fixture_config('Qwen/Qwen3-0.6B')
        for change in [{'use_sliding_window': True},
                       {'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0}},
                       {'rope_parameters': {'rope_theta': 1e6,
                                            'rope_type': 'yarn'}},
                       {'attention_bias': True},
                       {'model_type': 'llama'}]:
            with self.assertRaises(ValueError, msg=str(change)):
                d2l.qwen3_gpt_kwargs({**raw, **change})
        with self.assertRaisesRegex(NotImplementedError, 'qwen3_moe'):
            d2l.qwen3_gpt_kwargs({**raw, 'model_type': 'qwen3_moe'})
        with self.assertRaisesRegex(NotImplementedError,
                                    'sec_lm_hybrids_at_scale'):
            d2l.qwen3_gpt_kwargs(fixture_config('Qwen/Qwen3.5-2B'))
        with self.assertRaisesRegex(NotImplementedError,
                                    'sec_lm_hybrids_at_scale'):
            d2l.qwen3_gpt_kwargs({**raw, 'model_type': 'qwen3_5_text'})


class TestPermutation(unittest.TestCase):
    def test_values(self):
        self.assertEqual(d2l.qwen3_rope_permutation(4).tolist(), [0, 2, 1, 3])
        self.assertEqual(d2l.qwen3_rope_permutation(8).tolist(),
                         [0, 4, 1, 5, 2, 6, 3, 7])
        idx = d2l.qwen3_rope_permutation(128)
        self.assertEqual(sorted(idx.tolist()), list(range(128)))

    def test_score_identity(self):
        # R_pos P = P R'_pos: the book's rotation of permuted vectors equals
        # the permuted split-half rotation; RMSNorm needs its gain permuted
        torch.manual_seed(0)
        d_h, T = 128, 64
        idx = d2l.qwen3_rope_permutation(d_h)
        q, k = torch.randn(T, d_h), torch.randn(T, d_h)
        gamma = 1 + 0.5 * torch.randn(d_h)
        norm = lambda x, g: F.rms_norm(x, (d_h,), g, 1e-6)
        cos, sin = d2l.qwen3_rope_tables(T, d_h, 1e6)
        scores_hf = (rotate_half_rope(norm(q, gamma), cos, sin)
                     @ rotate_half_rope(norm(k, gamma), cos, sin).T)
        tables = (cos, sin)
        rot = lambda x, g: d2l.rope(norm(x[:, idx], g), tables=tables)
        scores = rot(q, gamma[idx]) @ rot(k, gamma[idx]).T
        scale = scores_hf.abs().max().item()
        self.assertLessEqual((scores - scores_hf).abs().max().item(),
                             1e-5 * scale)
        wrong = rot(q, gamma) @ rot(k, gamma).T  # gain not permuted
        self.assertGreater((wrong - scores_hf).abs().max().item(),
                           1e-2 * scale)


def synthetic_hf_state(num_layers=2, d=16, H=4, H_kv=2, d_h=8, f=24, V=30,
                       head=None, meta=False):
    """A Transformers-style Qwen3 state dict and its configuration."""
    g = torch.Generator().manual_seed(0)
    make = ((lambda *s: torch.empty(*s, device='meta')) if meta
            else (lambda *s: torch.randn(*s, generator=g)))
    state = {'model.embed_tokens.weight': make(V, d),
             'model.norm.weight': make(d)}
    for i in range(num_layers):
        p = f'model.layers.{i}.'
        state.update({
            p + 'input_layernorm.weight': make(d),
            p + 'self_attn.q_proj.weight': make(H * d_h, d),
            p + 'self_attn.k_proj.weight': make(H_kv * d_h, d),
            p + 'self_attn.v_proj.weight': make(H_kv * d_h, d),
            p + 'self_attn.o_proj.weight': make(d, H * d_h),
            p + 'self_attn.q_norm.weight': make(d_h),
            p + 'self_attn.k_norm.weight': make(d_h),
            p + 'post_attention_layernorm.weight': make(d),
            p + 'mlp.gate_proj.weight': make(f, d),
            p + 'mlp.up_proj.weight': make(f, d),
            p + 'mlp.down_proj.weight': make(d, f)})
    if head == 'tied':
        state['lm_head.weight'] = state['model.embed_tokens.weight'].clone()
    elif head == 'untied':
        state['lm_head.weight'] = make(V, d)
    cfg = dict(model_type='qwen3', vocab_size=V, hidden_size=d,
               num_attention_heads=H, num_key_value_heads=H_kv, head_dim=d_h,
               num_hidden_layers=num_layers, intermediate_size=f,
               rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=64,
               tie_word_embeddings=head != 'untied', attention_bias=False,
               use_sliding_window=False, rope_scaling=None)
    return state, cfg


class TestMap(unittest.TestCase):
    def check_round_trip(self, head):
        hf, cfg = synthetic_hf_state(head=head)
        state = d2l.map_qwen3_state_dict(hf, cfg)
        model = d2l.Qwen3GPT(**d2l.qwen3_gpt_kwargs(cfg))
        self.assertEqual(
            {k: v.shape for k, v in state.items()},
            {k: v.shape for k, v in model.state_dict().items()})
        model.load_state_dict(state, strict=True)
        idx = d2l.qwen3_rope_permutation(8)
        q = hf['model.layers.1.self_attn.q_proj.weight']
        self.assertTrue(torch.equal(
            state['blks.1.attention.W_q.weight'],
            q.reshape(4, 8, 16)[:, idx].reshape(32, 16)))
        k_norm = hf['model.layers.0.self_attn.k_norm.weight']
        self.assertTrue(torch.equal(state['blks.0.attention.k_norm.weight'],
                                    k_norm[idx]))
        for hf_name, ours in [('mlp.gate_proj', 'ffn.W_g'),
                              ('mlp.up_proj', 'ffn.W_1'),
                              ('self_attn.v_proj', 'attention.W_v')]:
            self.assertIs(state[f'blks.0.{ours}.weight'],
                          hf[f'model.layers.0.{hf_name}.weight'])
        return hf, state

    def test_round_trip(self):
        for head in (None, 'tied', 'untied'):
            hf, state = self.check_round_trip(head)
            self.assertEqual(len(state), len(hf) - (head == 'tied'))
        hf, cfg = synthetic_hf_state()
        none = d2l.map_qwen3_state_dict(hf, cfg, rope_layout='none')
        self.assertIs(none['blks.0.attention.W_q.weight'],
                      hf['model.layers.0.self_attn.q_proj.weight'])

    def test_strict_accounting(self):
        hf, cfg = synthetic_hf_state()
        extra = {**hf, 'model.layers.0.self_attn.rotary.weight': torch.ones(1)}
        with self.assertRaisesRegex(KeyError,
                                    r'not consumed: \[.*rotary.weight'):
            d2l.map_qwen3_state_dict(extra, cfg)
        missing = dict(hf)
        del missing['model.layers.1.self_attn.k_norm.weight']
        with self.assertRaisesRegex(
                KeyError, r'blks\.1\.attention\.k_norm\.weight \(from '
                r'model\.layers\.1\.self_attn\.k_norm\.weight\)'):
            d2l.map_qwen3_state_dict(missing, cfg)
        hf_untied, cfg_untied = synthetic_hf_state(head='untied')
        del hf_untied['lm_head.weight']
        with self.assertRaisesRegex(KeyError, 'lm_head'):
            d2l.map_qwen3_state_dict(hf_untied, cfg_untied)

    def test_unequal_tied_head(self):
        hf, cfg = synthetic_hf_state(head='tied')
        hf['lm_head.weight'] = hf['lm_head.weight'] + 1
        with self.assertRaisesRegex(ValueError, 'tied'):
            d2l.map_qwen3_state_dict(hf, cfg)

    def test_meta(self):
        hf, cfg = synthetic_hf_state(meta=True, head='tied')
        state = d2l.map_qwen3_state_dict(hf, cfg)
        self.assertTrue(all(v.is_meta for v in state.values()))
        self.assertEqual(state['blks.0.attention.W_q.weight'].shape, (32, 16))
        self.assertEqual(state['blks.0.attention.q_norm.weight'].shape, (8,))

    def test_dry_runs_of_released_layouts(self):
        # Transformers' own Qwen3 on meta stands in for the header: 0.6B
        # stores a duplicate head (311 tensors -> 310 keys), 8B an untied one
        import transformers
        for model_id, consumed, produced in [('Qwen/Qwen3-0.6B', 311, 310),
                                             ('Qwen/Qwen3-8B', 399, 399)]:
            cfg = fixture_config(model_id)
            with torch.device('meta'):
                hf = transformers.Qwen3ForCausalLM(
                    transformers.Qwen3Config.from_dict(cfg))
                port = d2l.Qwen3GPT(**d2l.qwen3_gpt_kwargs(cfg))
            hf_state = dict(hf.state_dict())
            self.assertEqual(len(hf_state), consumed)
            state = d2l.map_qwen3_state_dict(hf_state, cfg)
            self.assertEqual(len(state), produced)
            port.load_state_dict(state, strict=True, assign=True)
            self.assertTrue(all(p.is_meta for p in port.parameters()))
            self.assertEqual('lm_head.weight' in state,
                             not cfg['tie_word_embeddings'])


class TestLoadQwen3(unittest.TestCase):
    def test_tiny_checkpoints(self):
        X = torch.randint(4, 128, (2, 16), generator=torch.Generator()
                          .manual_seed(0))
        for tie in (True, False):
            hf = tiny_hf(tie_word_embeddings=tie, seed=int(tie))
            with torch.no_grad():
                expected = hf(X).logits
            with saved(hf) as files, contextlib.redirect_stdout(io.StringIO()):
                port = d2l.load_qwen3('Qwen/Qwen3-0.6B', dtype=torch.float32,
                                      device='cpu')
                port64 = d2l.load_qwen3('Qwen/Qwen3-0.6B', 'abc',
                                        dtype=torch.float64, device='cpu')
            files.assert_any_call('Qwen/Qwen3-0.6B',
                                  d2l.HF_REVISIONS['Qwen/Qwen3-0.6B'])
            files.assert_any_call('Qwen/Qwen3-0.6B', 'abc')
            self.assertIsInstance(port, d2l.Qwen3GPT)
            self.assertFalse(port.training)
            self.assertEqual(port.revision, d2l.HF_REVISIONS['Qwen/Qwen3-0.6B'])
            self.assertEqual(port.hf_config['tie_word_embeddings'], tie)
            self.assertEqual(port.generation_config.eos_token_id, EOS)
            self.assertEqual(hasattr(port, 'lm_head'), not tie)
            self.assertEqual(d2l.num_params(port), hf.num_parameters())
            self.assertTrue(all(p.dtype == torch.float64
                                for p in port64.parameters()))
            with torch.no_grad():
                diff = (port(X) - expected).abs().max().item()
                diff64 = (port64(X).float() - expected).abs().max().item()
            self.assertLessEqual(diff, 1e-5, f'tie={tie}')
            self.assertLessEqual(diff64, 1e-5, f'tie={tie}')

    def test_bf16_and_no_generation_config(self):
        hf = tiny_hf()
        with saved(hf) as files:
            config, _, weights = files.return_value
            files.return_value = (config, None, weights)
            port = d2l.load_qwen3('Qwen/Qwen3-0.6B', device='cpu')
        self.assertIsNone(port.generation_config)
        self.assertTrue(all(p.dtype == torch.bfloat16
                            for p in port.parameters()))
        self.assertTrue(all(p.device.type == 'cpu' for p in port.parameters()))

    def test_files(self):
        import huggingface_hub
        from huggingface_hub.errors import EntryNotFoundError
        for repo in ({'config.json': '{}', 'model.safetensors': ''},
                     {'config.json': '{}', 'generation_config.json': '{}',
                      'model.safetensors.index.json': json.dumps(
                          {'weight_map': {'a': 'm-2.safetensors',
                                          'b': 'm-1.safetensors',
                                          'c': 'm-2.safetensors'}}),
                      'm-1.safetensors': '', 'm-2.safetensors': ''}):
            path = tempfile.mkdtemp()
            calls = []

            def download(model_id, filename, revision=None):
                calls.append((model_id, filename, revision))
                if filename not in repo:
                    raise EntryNotFoundError(filename)
                with open(os.path.join(path, filename), 'w') as fh:
                    fh.write(repo[filename])
                return os.path.join(path, filename)

            try:
                with mock.patch.object(huggingface_hub, 'hf_hub_download',
                                       download):
                    config, generation, weights = d2l._qwen3_files('o/m', 'r')
            finally:
                shutil.rmtree(path)
            self.assertTrue(config.endswith('config.json'))
            self.assertEqual(generation is None,
                             'generation_config.json' not in repo)
            self.assertEqual([os.path.basename(w) for w in weights],
                             ['model.safetensors'] if len(repo) == 2
                             else ['m-1.safetensors', 'm-2.safetensors'])
            self.assertTrue(all(c[0] == 'o/m' and c[2] == 'r' for c in calls))
        with mock.patch.object(huggingface_hub, 'hf_hub_download',
                               side_effect=EntryNotFoundError('x')):
            with self.assertRaises(EntryNotFoundError):
                d2l._qwen3_files('o/m', 'r')


class TestHiddens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hf = tiny_hf(seed=3)
        cls.hf64 = tiny_hf(seed=3, dtype=torch.float64)
        cls.port = port_of(cls.hf)
        cls.X = torch.randint(4, 128, (3, 12),
                              generator=torch.Generator().manual_seed(1))

    def test_port_matches_transformers(self):
        ref = d2l.collect_hiddens(self.hf, self.X)
        ours = d2l.collect_hiddens(self.port, self.X)
        self.assertEqual(list(ours), list(POINTS))
        for name in POINTS:
            self.assertEqual(ours[name].shape, (2, 3, 12, 64))
            self.assertEqual(ours[name].dtype, torch.float32)
            self.assertLessEqual((ours[name] - ref[name]).abs().max().item(),
                                 1e-5, name)
        # resid_mid = block input + attn_out; resid_post = resid_mid + mlp_out
        self.assertTrue(torch.allclose(
            ours['resid_post'], ours['resid_mid'] + ours['mlp_out']))
        with torch.no_grad():  # the last capture feeds the final norm
            logits = F.linear(self.port.norm(ours['resid_post'][-1]),
                              self.port.token_emb.weight)
            self.assertTrue(torch.allclose(logits, self.port(self.X),
                                           atol=1e-5))

    def test_dtypes_and_hooks_removed(self):
        ref64 = d2l.collect_hiddens(self.hf64, self.X)
        self.assertEqual(ref64['attn_out'].dtype, torch.float64)
        bf16 = port_of(self.hf, torch.bfloat16)
        self.assertEqual(d2l.collect_hiddens(bf16, self.X)['mlp_out'].dtype,
                         torch.float32)
        hooked = lambda m: any(len(x._forward_hooks) + len(
            x._forward_pre_hooks) for x in m.modules())
        self.assertFalse(hooked(self.port) or hooked(self.hf))
        with self.assertRaises(IndexError):  # ids beyond the vocabulary
            d2l.collect_hiddens(self.port, self.X + 1000)
        self.assertFalse(hooked(self.port))

    def test_layer_errors(self):
        ref = {n: torch.ones(2, 3, 4, 5) for n in POINTS}
        ours = {n: torch.ones(2, 3, 4, 5) for n in POINTS}
        ours['mlp_out'][1, 2, 1:] *= 1.5   # layer 1, prompt 2, positions 1-3
        ours['resid_post'][0, 0, 0] = 3.0  # position 0 only
        errors = d2l.layer_errors(ref, ours)
        self.assertEqual(errors.shape, (2, 4, 3))
        expected = np.zeros((2, 4, 3))
        expected[1, 2, 2] = 0.5
        np.testing.assert_allclose(errors, expected, atol=1e-12)
        with_first = d2l.layer_errors(ref, ours, skip_first=0)
        np.testing.assert_allclose(with_first[0, 3, 0], 2 / 2, atol=1e-12)
        np.testing.assert_allclose(with_first[1, 2, 2],
                                   (0.5 * 15 ** 0.5) / 20 ** 0.5, atol=1e-12)

    def test_first_divergence_synthetic(self):
        floor = np.full((3, 4, 5), 1e-6)
        self.assertIsNone(d2l.first_divergence(floor * 9.9, floor))
        errors = floor.copy()
        errors[2, 1] = 1e-4                   # every prompt at (2, resid_mid)
        errors[1, 3, :2] = 1.0                # two of five prompts: median not
        self.assertEqual(d2l.first_divergence(errors, floor), (2, 'resid_mid'))
        errors[1, 3, :3] = 1.0                # three of five: the median moves
        self.assertEqual(d2l.first_divergence(errors, floor),
                         (1, 'resid_post'))
        self.assertIsNone(d2l.first_divergence(errors, floor, factor=1e7))

    def test_first_divergence_of_porting_bugs(self):
        ref = d2l.collect_hiddens(self.hf64, self.X)
        floor = d2l.layer_errors(ref, d2l.collect_hiddens(self.port, self.X))
        locate = lambda m: d2l.first_divergence(
            d2l.layer_errors(ref, d2l.collect_hiddens(m, self.X)), floor)
        self.assertIsNone(locate(port_of(self.hf)))
        self.assertEqual(locate(port_of(self.hf, rope_layout='none')),
                         (0, 'attn_out'))
        swapped = port_of(self.hf)  # B3: gate and up projections swapped
        with torch.no_grad():
            ffn = swapped.blks[0].ffn
            W_g = ffn.W_g.weight.clone()
            ffn.W_g.weight.copy_(ffn.W_1.weight)
            ffn.W_1.weight.copy_(W_g)
        self.assertEqual(locate(swapped), (0, 'mlp_out'))
        # Blocks 1 and 2 of a 4-layer model in the wrong order
        hf4 = tiny_hf(seed=4, num_hidden_layers=4)
        ref4 = d2l.collect_hiddens(hf4.double(), self.X)
        hf4.float()
        clean = port_of(hf4)
        floor4 = d2l.layer_errors(ref4, d2l.collect_hiddens(clean, self.X))
        reordered = port_of(hf4)
        reordered.blks = nn.ModuleList([reordered.blks[i]
                                        for i in (0, 2, 1, 3)])
        errors = d2l.layer_errors(ref4, d2l.collect_hiddens(reordered, self.X))
        self.assertTrue(np.array_equal(errors[0], floor4[0]))  # layer 0 exact
        self.assertEqual(d2l.first_divergence(errors, floor4), (1, 'attn_out'))


class TestSampler(unittest.TestCase):
    def test_generation_record(self):
        g = d2l.Generation('hi', [5, 6], None, 'eos')
        self.assertEqual(d2l.Generation._fields,
                         ('text', 'token_ids', 'logprobs', 'finish_reason'))
        self.assertEqual(tuple(g), ('hi', [5, 6], None, 'eos'))
        self.assertEqual(g.finish_reason, 'eos')

    def test_toy_supports(self):
        for kw, support in [({'k': 2}, {0, 1}), ({'p': 0.8}, {0, 1, 2}),
                            ({'min_p': 0.5}, {0, 1}), ({}, set(range(5)))]:
            probs = d2l.truncated_probs(TOY[None], **kw)[0]
            self.assertEqual(set(torch.nonzero(probs)[:, 0].tolist()), support)
            np.testing.assert_allclose(probs.numpy(),
                                       sample_next_probs(TOY.numpy(), **kw),
                                       atol=1e-6)

    def test_random_rows(self):
        g = torch.Generator().manual_seed(0)
        logits = 3 * torch.randn(20, 1000, generator=g)
        logits[:, :10] = logits[:, :1]  # ties at the top, broken by index
        for temperature in (0.5, 1.0, 1.7):
            for k in (None, 1, 50):
                for p in (None, 0.3, 0.9):
                    for min_p in (None, 0.05, 0.5):
                        probs = d2l.truncated_probs(logits, temperature, k, p,
                                                    min_p)
                        expected = np.stack([sample_next_probs(
                            row.numpy(), temperature, k, p, min_p)
                                             for row in logits])
                        np.testing.assert_allclose(
                            probs.numpy(), expected, atol=1e-6,
                            err_msg=f'{temperature} {k} {p} {min_p}')

    def test_draws_follow_the_distribution(self):
        n = 200000
        for kw in ({}, {'p': 0.8}, {'temperature': 2.0, 'k': 3}):
            draws = d2l.sample_logits(
                TOY.expand(n, 5), 'sample', generator=torch.Generator()
                .manual_seed(0), **kw)
            freq = torch.bincount(draws, minlength=5).double() / n
            probs = d2l.truncated_probs(TOY[None], **kw)[0].double()
            self.assertLessEqual(0.5 * (freq - probs).abs().sum().item(),
                                 3 / n ** 0.5, str(kw))
            self.assertTrue(set(draws.unique().tolist())
                            <= set(torch.nonzero(probs)[:, 0].tolist()))

    def test_greedy_and_generators(self):
        logits = torch.tensor([[0.0, 2.0, 2.0, 1.0], [5.0, 5.0, 0.0, 0.0]])
        self.assertEqual(d2l.sample_logits(logits).tolist(), [1, 0])
        with self.assertRaises(ValueError):
            d2l.sample_logits(logits, 'beam')
        # one generator per row: a row's draw ignores the other rows
        flat = torch.zeros(3, 100)
        gens = lambda *seeds: [torch.Generator().manual_seed(s) for s in seeds]
        both = d2l.sample_logits(flat, 'sample', generator=gens(7, 8, 9))
        alone = d2l.sample_logits(flat[:1], 'sample', generator=gens(8))
        self.assertEqual(both[1].item(), alone[0].item())
        same = d2l.sample_logits(flat, 'sample',
                                 generator=torch.Generator().manual_seed(1))
        again = d2l.sample_logits(flat, 'sample',
                                  generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(same, again))


class TestGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fp32: in fp64, Transformers' eager attention turns a left-padded
        # row into NaN (its fp64 mask minimum becomes -inf in the fp32
        # softmax), so fp64 batches with padding would compare NaN with NaN
        cls.tok = tiny_tokenizer()
        cls.hf = tiny_hf(seed=5)
        cls.port = port_of(cls.hf)
        g = torch.Generator().manual_seed(2)
        cls.prompts = [words(torch.randint(4, 128, (n,), generator=g).tolist())
                       for n in (5, 9, 5, 3, 9, 7)]

    def test_hf_equals_transformers_generate(self):
        logits, _ = d2l._lm_hf_steps(self.hf, [[5, 6], [7, 8, 9, 10]], PAD)
        self.assertTrue(torch.isfinite(logits).all())
        unpadded = self.hf(torch.tensor([[5, 6]])).logits[0, -1]
        self.assertLessEqual((logits[0] - unpadded).abs().max().item(), 1e-5)
        out = d2l.generate(self.hf, self.tok, self.prompts, 12, batch_size=4)
        for start in (0, 4):
            batch = self.tok(self.prompts[start:start + 4], padding=True,
                             padding_side='left', return_tensors='pt')
            ref = self.hf.generate(**batch, do_sample=False, max_new_tokens=12)
            for row, g in zip(ref[:, batch['input_ids'].shape[1]:].tolist(),
                              out[start:start + 4]):
                cut = row.index(EOS) if EOS in row else len(row)
                self.assertEqual(g.token_ids, row[:cut])
                self.assertEqual(g.finish_reason,
                                 'eos' if EOS in row else 'length')

    def test_port_equals_transformers(self):
        ours = d2l.generate(self.port, self.tok, self.prompts, 12)
        ref = d2l.generate(self.hf, self.tok, self.prompts, 12)
        self.assertEqual(ours, ref)
        self.assertTrue(any(len(g.token_ids) > 3 for g in ours))

    def test_batched_equals_one_by_one(self):
        recompute = d2l.GPT(128, num_hiddens=32, num_heads=4, num_blks=1)
        for model in (self.hf, self.port, recompute.double().eval()):
            batched = d2l.generate(model, self.tok, self.prompts, 8,
                                   batch_size=3)
            single = [d2l.generate(model, self.tok, [x], 8)[0]
                      for x in self.prompts]
            self.assertEqual(batched, single, type(model).__name__)

    def test_paths(self):
        calls = []
        forward_cached = self.port.forward_cached
        with mock.patch.object(self.port, 'forward_cached',
                               lambda *a: calls.append(a[0].shape)
                               or forward_cached(*a)):
            d2l.generate(self.port, self.tok, self.prompts[:2], 4,
                         eos_token_ids=[])
        self.assertEqual(calls, [(1, 5), (1, 1), (1, 1), (1, 1),
                                 (1, 9), (1, 1), (1, 1), (1, 1)])
        counter = Counter()
        with mock.patch.object(counter, 'forward',
                               wraps=counter.forward) as forward:
            d2l.generate(counter, self.tok, ['w4 w5', 'w6 w7', 'w8'], 3)
        self.assertEqual([c.args[0].shape for c in forward.call_args_list],
                         [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 4)])

    def test_finish_reasons(self):
        counter = Counter()
        out = d2l.generate(counter, self.tok, ['w4', 'w120', 'w10'], 6,
                           eos_token_ids=[123], stop=['w8 w9'])
        self.assertEqual(out[0], d2l.Generation('w5 w6 w7 ', [5, 6, 7, 8, 9],
                                                None, 'stop'))
        self.assertEqual(out[1], d2l.Generation('w121 w122', [121, 122], None,
                                                'eos'))
        self.assertEqual(out[2], d2l.Generation(
            'w11 w12 w13 w14 w15 w16', [11, 12, 13, 14, 15, 16], None,
            'length'))
        # defaults: the tokenizer's end token joins the model's
        counter.generation_config = mock.Mock(eos_token_id=[3],
                                              max_new_tokens=None)
        out = d2l.generate(counter, self.tok, ['<s>', 'w126'])
        self.assertEqual([g.finish_reason for g in out], ['eos', 'eos'])
        self.assertEqual([g.token_ids for g in out], [[], [127, 0]])
        self.assertEqual(len(d2l.generate(Counter(), self.tok, ['w4'],
                                          eos_token_ids=[])[0].token_ids), 256)

    def test_sampling_seeds(self):
        fixed = Fixed()
        prompts = ['w4', 'w5 w6', 'w7', 'w8 w9 w10']
        kw = dict(strategy='sample', temperature=1.3, p=0.95, k=40)
        once = d2l.generate(fixed, self.tok, prompts, 10, seed=3, **kw)
        self.assertEqual(once, d2l.generate(fixed, self.tok, prompts, 10,
                                            seed=3, **kw))
        self.assertNotEqual(once, d2l.generate(fixed, self.tok, prompts, 10,
                                               seed=4, **kw))
        seeds = [11, 12, 13, 14]
        rows = d2l.generate(fixed, self.tok, prompts, 10, seed=seeds, **kw)
        for i in range(4):  # alone, permuted, or in other chunks
            alone = d2l.generate(fixed, self.tok, [prompts[i]], 10,
                                 seed=[seeds[i]], **kw)
            self.assertEqual(rows[i], alone[0])
        order = [2, 0, 3, 1]
        permuted = d2l.generate(fixed, self.tok, [prompts[i] for i in order],
                                10, seed=[seeds[i] for i in order],
                                batch_size=1, **kw)
        self.assertEqual(permuted, [rows[i] for i in order])
        with self.assertRaises(AssertionError):
            d2l.generate(fixed, self.tok, prompts, 2, seed=[1, 2], **kw)

    def test_per_row_seeds_ignore_finished_rows(self):
        # Rows that stop early keep drawing, so later rows are unaffected
        prompts = ['w4', 'w5']
        kw = dict(strategy='sample', max_new_tokens=6, seed=[1, 2])
        with_stop = d2l.generate(Fixed(), self.tok, prompts, stop=['w'], **kw)
        without = d2l.generate(Fixed(), self.tok, prompts, **kw)
        self.assertEqual([g.finish_reason for g in with_stop], ['stop'] * 2)
        self.assertEqual(without[1], d2l.generate(
            Fixed(), self.tok, ['w5'], 6, strategy='sample', seed=[2])[0])

    def test_logprobs(self):
        out = d2l.generate(self.hf, self.tok, self.prompts[:3], 6,
                           strategy='sample', temperature=0.7, k=5, seed=0,
                           return_logprobs=True, eos_token_ids=[])
        for prompt, g in zip(self.prompts[:3], out):
            ids = self.tok(prompt)['input_ids'] + g.token_ids
            with torch.no_grad():
                logits = self.hf(torch.tensor([ids])).logits[0].float()
            lp = F.log_softmax(logits, -1)[len(ids) - 7:-1].gather(
                -1, torch.tensor(g.token_ids)[:, None])[:, 0]
            np.testing.assert_allclose(g.logprobs, lp.tolist(), atol=1e-5)
        self.assertIsNone(d2l.generate(self.hf, self.tok, self.prompts[:1],
                                       2)[0].logprobs)

    def test_chat(self):
        calls = []
        tok = self.tok
        render = tok.apply_chat_template

        def spy(messages, **kw):
            calls.append((messages, kw))
            return render(messages, **kw)

        with mock.patch.object(tok, 'apply_chat_template', spy):
            off = d2l.generate(Counter(), tok, ['w7 w8'], 2, chat=True,
                               enable_thinking=False)
            default = d2l.generate(
                Counter(), tok, [[{'role': 'user', 'content': 'w7 w8'}]], 2,
                chat=True)
        self.assertEqual(calls[0], ([{'role': 'user', 'content': 'w7 w8'}],
                                    {'tokenize': False,
                                     'add_generation_prompt': True,
                                     'enable_thinking': False}))
        self.assertNotIn('enable_thinking', calls[1][1])
        # '<s> w7 w8 </s> w5 <s>' ends in <s> (2): the counter continues 3, 4
        self.assertEqual(off[0].token_ids, [3, 4])
        self.assertEqual(default[0].token_ids, [3, 4])


class TestQwen3Tokenizer(unittest.TestCase):
    """With the real Qwen3 tokenizer from the local Hugging Face cache (one
    `AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', revision=...)` caches
    it); skipped when it is not cached."""
    @classmethod
    def setUpClass(cls):
        import transformers
        try:
            cls.tok = transformers.AutoTokenizer.from_pretrained(
                'Qwen/Qwen3-0.6B', revision=d2l.HF_REVISIONS['Qwen/Qwen3-0.6B'],
                local_files_only=True)
        except OSError:
            raise unittest.SkipTest('Qwen3 tokenizer not in the HF cache')

    def test_prompts_and_stop_tokens(self):
        seen = []
        model = Counter(len(self.tok))
        model.forward = mock.Mock(side_effect=lambda X: (
            seen.append(X[0].tolist()), Counter.forward(model, X))[1])
        model.generation_config = mock.Mock(eos_token_id=[151645, 151643],
                                            max_new_tokens=None)
        # no BOS is added; <|im_end|> - 1 continues to <|im_end|>, a stop token
        d2l.generate(model, self.tok, ['Hello'], 3)
        self.assertEqual(seen[0], self.tok('Hello')['input_ids'])
        self.assertNotIn(self.tok.bos_token_id or -1, seen[0])
        seen.clear()
        d2l.generate(model, self.tok, ['Hi'], 5, chat=True,
                     enable_thinking=False)
        rendered = self.tok.apply_chat_template(
            [{'role': 'user', 'content': 'Hi'}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(rendered.endswith(
            '<|im_start|>assistant\n<think>\n\n</think>\n\n'), rendered)
        self.assertEqual(seen[0], self.tok(rendered, add_special_tokens=False)
                         ['input_ids'])
        stop = d2l.generate(model, self.tok, ['<|im_start|>'], 4)[0]
        self.assertEqual(stop.finish_reason, 'eos')  # 151644 + 1 = <|im_end|>
        self.assertEqual(stop.token_ids, [])


if __name__ == '__main__':
    unittest.main()
