#!/usr/bin/env python3
"""CPU tests for the library code of sec_lm_hf_models
(chapter_lm-pretrained/hf-models.md): `d2l.HF_REVISIONS`,
`d2l.safetensors_header`, `d2l.load_hf`, `d2l.config_params` and
`d2l.config_memory`.

The configurations and safetensors headers of the studied checkpoints are
committed fixtures (tools/lm/fixtures/hf_models/, Apache-2.0 metadata, no
weights). Models are built from them on the meta device (no memory) or as
tiny random models on the CPU; nothing is downloaded. The one network test,
which reads the Qwen3.5-2B header from the Hub and compares it with the
fixture, runs only with D2L_TEST_NETWORK=1.

Run: `.venv-lm/bin/python -m unittest tools/test_lm_hf_models.py`, or
`make test-lm`.
"""

import contextlib
import glob
import http.server
import io
import json
import math
import os
import re
import shutil
import socket
import tempfile
import threading
import traceback
import unittest
import urllib.error
import urllib.request
import warnings
from unittest import mock

import torch
from d2l import torch as d2l

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
FIXTURES = os.path.join(TOOLS, 'lm', 'fixtures', 'hf_models')
QWEN35, GEMMA = 'Qwen/Qwen3.5-2B', 'google/gemma-4-E2B-it'
DENSE = ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3-1.7B-Base',
         'Qwen/Qwen3-4B', 'Qwen/Qwen3-8B']
SHA = re.compile(r'^[0-9a-f]{40}$')


def fixture_config(model_id):
    with open(os.path.join(FIXTURES, model_id, 'config.json')) as fh:
        return json.load(fh)


def fixture_header(model_id):
    """Name -> (shape, dtype), the format of `d2l.safetensors_header`."""
    with open(os.path.join(FIXTURES, model_id, 'header.json')) as fh:
        tensors = json.load(fh)['tensors']
    return {k: (tuple(v['shape']), v['dtype']) for k, v in tensors.items()}


def numel(header, keep=lambda name: True):
    return sum(math.prod(shape) for name, (shape, _) in header.items()
               if keep(name))


def auto_config(model_id):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(os.path.join(FIXTURES, model_id))


def meta_params(config, cls=None):
    """`num_parameters()` of the model Transformers builds, on meta."""
    from transformers import AutoModelForCausalLM
    with torch.device('meta'):
        model = (cls(config) if cls else
                 AutoModelForCausalLM.from_config(config))
    return model.num_parameters()


# Tiny random models: the same configuration shapes as the studied ones
# (gated attention and Gated DeltaNet; sliding and global layers, KV
# sharing, double-wide MLPs and per-layer embeddings), at a few KB.
TINY_QWEN35 = dict(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
    linear_value_head_dim=8, linear_conv_kernel_dim=4,
    layer_types=['linear_attention'] * 3 + ['full_attention'],
    max_position_embeddings=64)
TINY_QWEN35_VISION = dict(depth=1, hidden_size=16, intermediate_size=32,
                          num_heads=2, out_hidden_size=32,
                          num_position_embeddings=16)
TINY_GEMMA = dict(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=5,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    global_head_dim=32, sliding_window=8, num_kv_shared_layers=2,
    layer_types=['sliding_attention', 'full_attention', 'sliding_attention',
                 'sliding_attention', 'full_attention'],
    hidden_size_per_layer_input=8, vocab_size_per_layer_input=128,
    use_double_wide_mlp=True, max_position_embeddings=64)
TINY_GEMMA_VISION = dict(hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, head_dim=8,
                         position_embedding_size=16)
TINY_GEMMA_AUDIO = dict(hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, output_proj_dims=16)


def tiny_gemma_config(**text):
    from transformers import Gemma4Config
    return Gemma4Config(text_config={**TINY_GEMMA, **text},
                        vision_config=TINY_GEMMA_VISION,
                        audio_config=TINY_GEMMA_AUDIO)


def tiny_qwen35_config(**text):
    from transformers.models.qwen3_5 import Qwen3_5Config
    return Qwen3_5Config(text_config={**TINY_QWEN35, **text},
                         vision_config=TINY_QWEN35_VISION)


def save_tiny_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel({'<pad>': 0, '<unk>': 1, 'cache': 2},
                                     unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='<unk>',
                            pad_token='<pad>').save_pretrained(path)


def cache_bytes(cache):
    """Bytes of the floating-point tensors a Transformers cache holds,
    counting each view once (a sliding layer's keys are a view)."""
    seen, total = set(), 0

    def visit(x):
        nonlocal total
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            key = (x.data_ptr(), tuple(x.shape))
            if key not in seen:
                seen.add(key)
                total += x.nbytes
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)

    for layer in cache.layers:
        visit(list(vars(layer).values()))
    return total


def quiet():
    """Silence Transformers' loader logs and progress bars in a test."""
    import transformers
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    stack.callback(transformers.logging.enable_progress_bar)
    stack.callback(transformers.logging.set_verbosity, verbosity)
    return stack


class Revisions(unittest.TestCase):
    def test_values_are_commit_shas(self):
        for model_id, sha in d2l.HF_REVISIONS.items():
            self.assertRegex(sha, SHA, model_id)

    def test_studied_models_pinned(self):
        for model_id in DENSE + [QWEN35, GEMMA, 'Qwen/Qwen3-1.7B',
                                 'Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-0.8B-Base',
                                 'Qwen/Qwen3.5-4B']:
            self.assertIn(model_id, d2l.HF_REVISIONS)

    def test_every_hub_id_in_the_chapter_is_pinned(self):
        """Every Hub repository named in a code cell of the chapter has a
        pin. The API section names hosted models (OpenRouter ids), not Hub
        repositories, and is left out."""
        cell = re.compile(r'```\{\.python \.input[^}]*\}\n(.*?)```', re.S)
        hub_id = re.compile(r"""['"]((?:Qwen|google)/[\w.-]+)['"]""")
        found = set()
        for path in glob.glob(os.path.join(ROOT, 'chapter_lm-pretrained',
                                           '*.md')):
            if os.path.basename(path).startswith('api'):
                continue
            with open(path) as fh:
                for code in cell.findall(fh.read()):
                    found |= set(hub_id.findall(code))
        self.assertTrue(found)
        self.assertEqual(sorted(found - set(d2l.HF_REVISIONS)), [])


class ConfigParams(unittest.TestCase):
    def test_qwen35_2b_reconciles_with_header(self):
        cfg, header = fixture_config(QWEN35), fixture_header(QWEN35)
        self.assertEqual(d2l.config_params(cfg), 1_881_825_088)
        self.assertEqual(d2l.config_params(cfg, include_vision=True),
                         2_213_241_664)
        groups = d2l.config_params(cfg, True, True, by_group=True)
        self.assertEqual(sum(groups.values()), 2_274_069_824)
        self.assertEqual(sum(groups.values()), numel(header))
        self.assertEqual(len(header), 632)
        buckets = {
            'model.language_model.': sum(v for k, v in groups.items()
                                         if k not in ('vision', 'mtp')),
            'model.visual.': groups['vision'], 'mtp.': groups['mtp']}
        for prefix, count in buckets.items():
            self.assertEqual(
                numel(header, lambda name: name.startswith(prefix)), count,
                prefix)
        self.assertEqual(groups['vision'], 331_416_576)
        self.assertEqual(groups['mtp'], 60_828_160)

    def test_qwen35_layer_counts(self):
        """The per-layer counts of the derivation (eq_lm_hf_params)."""
        groups = d2l.config_params(fixture_config(QWEN35), by_group=True)
        self.assertEqual(groups['attention'] + 6 * (3 * 2048 * 6144 + 2 * 2048),
                         6 * 52_433_408)
        self.assertEqual(groups['linear_attention']
                         + 18 * (3 * 2048 * 6144 + 2 * 2048), 18 * 58_814_624)
        self.assertEqual(groups['embed'], 508_559_360)
        self.assertEqual(groups['lm_head'], 0)  # tied

    def test_dense_qwen3(self):
        expected = {'Qwen/Qwen3-0.6B': 596_049_920,
                    'Qwen/Qwen3-0.6B-Base': 596_049_920,
                    'Qwen/Qwen3-8B': 8_190_735_360}
        for model_id, count in expected.items():
            self.assertEqual(d2l.config_params(fixture_config(model_id)),
                             count, model_id)
        groups = d2l.config_params(fixture_config('Qwen/Qwen3-8B'),
                                   by_group=True)
        self.assertEqual(groups['lm_head'], 151_936 * 4096)  # untied

    def test_gemma_text_model(self):
        cfg, header = fixture_config(GEMMA), fixture_header(GEMMA)
        text = 'model.language_model.'
        shared_kv = re.compile(
            r'model\.language_model\.layers\.(\d+)\.self_attn\.'
            r'(k_proj|v_proj|k_norm)\.')
        first_shared = 35 - 20

        def unused(name):  # shared layers' K/V, never instantiated
            m = shared_kv.match(name)
            return bool(m) and int(m.group(1)) >= first_shared

        scalars = [k for k in header if k.endswith('.layer_scalar')]
        self.assertEqual(len(scalars), 35)  # buffers, not parameters
        self.assertEqual(numel(header, unused), 18_880_512)
        self.assertEqual(numel(header, lambda k: k.startswith(text)),
                         4_647_449_891)
        loaded = numel(header, lambda k: k.startswith(text) and not unused(k)
                       and not k.endswith('.layer_scalar'))
        groups = d2l.config_params(cfg, by_group=True)
        self.assertEqual(sum(groups.values()), d2l.config_params(cfg))
        self.assertEqual(loaded, 4_628_569_344)
        self.assertEqual(d2l.config_params(cfg), loaded)
        self.assertEqual(groups['per_layer_embed'], 262_144 * 35 * 256)
        self.assertEqual(d2l.config_params(cfg) - groups['per_layer_embed'],
                         2_279_759_104)

    def test_groups_sum_to_total(self):
        for model_id in DENSE + [QWEN35, GEMMA]:
            cfg = fixture_config(model_id)
            for flags in [(False, False), (False, True), (True, True)]:
                if model_id == GEMMA and flags[0]:
                    continue
                groups = d2l.config_params(cfg, *flags, by_group=True)
                self.assertEqual(sorted(groups), sorted([
                    'embed', 'per_layer_embed', 'attention',
                    'linear_attention', 'mlp', 'per_layer_proj', 'norm',
                    'lm_head', 'vision', 'mtp']))
                self.assertEqual(sum(groups.values()),
                                 d2l.config_params(cfg, *flags), model_id)

    def test_equals_transformers_on_meta(self):
        """The count equals the parameters Transformers instantiates for
        every fixture configuration (built on the meta device)."""
        from transformers.models.qwen3_5 import (
            Qwen3_5ForConditionalGeneration)
        for model_id in DENSE + [QWEN35, GEMMA]:
            cfg = auto_config(model_id)
            self.assertEqual(meta_params(cfg.get_text_config()),
                             d2l.config_params(cfg), model_id)
        cfg = auto_config(QWEN35)  # the multimodal model: text and vision
        self.assertEqual(meta_params(cfg, Qwen3_5ForConditionalGeneration),
                         d2l.config_params(cfg, include_vision=True))

    def test_config_forms_agree(self):
        """Raw config.json, the loaded configuration and its text part."""
        for model_id in [QWEN35, GEMMA, 'Qwen/Qwen3-8B']:
            cfg = auto_config(model_id)
            counts = {d2l.config_params(fixture_config(model_id)),
                      d2l.config_params(cfg),
                      d2l.config_params(cfg.get_text_config())}
            self.assertEqual(len(counts), 1, model_id)

    def test_tiny_models(self):
        """Global KV heads with K = V, and the tiny shapes used below."""
        from transformers.models.gemma4 import Gemma4ForConditionalGeneration
        from transformers.models.qwen3_5 import (
            Qwen3_5ForConditionalGeneration)
        variants = [tiny_gemma_config(),
                    tiny_gemma_config(attention_k_eq_v=True,
                                      num_global_key_value_heads=2),
                    tiny_gemma_config(num_kv_shared_layers=0,
                                      use_double_wide_mlp=False)]
        for cfg in variants:
            text = cfg.get_text_config()
            self.assertEqual(meta_params(text), d2l.config_params(cfg))
            self.assertEqual(d2l.config_params(text.to_dict()),
                             d2l.config_params(cfg))
        cfg = tiny_qwen35_config()
        self.assertEqual(meta_params(cfg.get_text_config()),
                         d2l.config_params(cfg))
        self.assertEqual(meta_params(cfg, Qwen3_5ForConditionalGeneration),
                         d2l.config_params(cfg, include_vision=True))
        with torch.device('meta'):  # the whole Gemma file is larger
            model = Gemma4ForConditionalGeneration(tiny_gemma_config())
        self.assertGreater(model.num_parameters(),
                           d2l.config_params(tiny_gemma_config()))

    def test_moe_raises(self):
        message = 'use d2l.config_stats (14.1)'
        moe = dict(fixture_config('Qwen/Qwen3-0.6B'), num_experts=128)
        with self.assertRaisesRegex(NotImplementedError, re.escape(message)):
            d2l.config_params(moe)
        routed = dict(fixture_config('Qwen/Qwen3-0.6B'), n_routed_experts=64)
        with self.assertRaisesRegex(NotImplementedError, re.escape(message)):
            d2l.config_params(routed)
        gemma = fixture_config(GEMMA)
        self.assertIsNone(gemma['text_config']['num_experts'])
        d2l.config_params(gemma)  # num_experts: null is not MoE
        gemma['text_config']['enable_moe_block'] = True
        with self.assertRaisesRegex(NotImplementedError, re.escape(message)):
            d2l.config_params(gemma)
        with self.assertRaisesRegex(NotImplementedError, re.escape(message)):
            d2l.config_memory(moe, 16, 1)

    def test_unknown_layouts_raise(self):
        """Regression: other families differ in ways the formula does not
        model (Qwen2.5's Q/K/V biases, OLMo 2's full-width QK norms, Gemma
        2 without QK norms, Mistral's sliding window without layer_types),
        so they raise instead of returning a silently wrong count."""
        dense = fixture_config('Qwen/Qwen3-0.6B')
        for model_type in ('qwen2', 'olmo2', 'gemma2', 'gemma3_text',
                           'mistral', 'llama'):
            cfg = dict(dense, model_type=model_type)
            with self.assertRaisesRegex(NotImplementedError, model_type):
                d2l.config_params(cfg)
            with self.assertRaisesRegex(NotImplementedError, model_type):
                d2l.config_memory(cfg, 16, 1)
        for bias in ('attention_bias', 'mlp_bias'):
            with self.assertRaisesRegex(NotImplementedError, 'biases'):
                d2l.config_params(dict(dense, **{bias: True}))
        gemma = fixture_config(GEMMA)
        gemma['text_config']['model_type'] = 'gemma2'
        with self.assertRaisesRegex(NotImplementedError, 'gemma2'):
            d2l.config_params(gemma)
        for model_id in [QWEN35, GEMMA] + DENSE:  # the studied families pass
            d2l.config_params(fixture_config(model_id))
            d2l.config_params(auto_config(model_id))

    def test_unsupported_vision_raises(self):
        with self.assertRaisesRegex(NotImplementedError, 'gemma4_vision'):
            d2l.config_params(fixture_config(GEMMA), include_vision=True)
        cfg = fixture_config(QWEN35)
        cfg['vision_config']['deepstack_visual_indexes'] = [5, 11]
        with self.assertRaises(NotImplementedError):
            d2l.config_params(cfg, include_vision=True)
        dense = fixture_config('Qwen/Qwen3-0.6B')  # no tower: nothing to add
        self.assertEqual(d2l.config_params(dense, include_vision=True),
                         d2l.config_params(dense))


class ConfigMemory(unittest.TestCase):
    KEYS = ['params', 'weights', 'kv_cache', 'recurrent_state',
            'activations', 'logits', 'total']

    def test_qwen35_2b(self):
        m = d2l.config_memory(fixture_config(QWEN35), 1024, 1)
        self.assertEqual(list(m), self.KEYS)
        self.assertEqual(m['kv_cache'], 12_582_912)
        self.assertEqual(m['recurrent_state'], 19_759_104)
        self.assertEqual(m['params'], 1_881_825_088)
        self.assertEqual(m['weights'], 2 * 1_881_825_088)
        self.assertEqual(m['logits'], 248_320 * 2)
        self.assertEqual(m['total'], sum(m[k] for k in self.KEYS[1:-1]))
        three = d2l.config_memory(fixture_config(QWEN35), 1024, 1,
                                  conv_taps=3)
        self.assertEqual(three['recurrent_state'], 19_537_920)

    def test_batch_scales_all_but_weights(self):
        for model_id in [QWEN35, GEMMA, 'Qwen/Qwen3-0.6B']:
            cfg = fixture_config(model_id)
            one, four = (d2l.config_memory(cfg, 1000, B) for B in (1, 4))
            self.assertEqual(one['weights'], four['weights'])
            self.assertEqual(one['params'], four['params'])
            for key in ['kv_cache', 'recurrent_state', 'activations',
                        'logits']:
                self.assertEqual(four[key], 4 * one[key], (model_id, key))
            self.assertEqual(four['total'] - four['weights'],
                             4 * (one['total'] - one['weights']))

    def test_qwen3_kv_formula(self):
        m = d2l.config_memory(fixture_config('Qwen/Qwen3-0.6B'), 1000, 1)
        self.assertEqual(m['kv_cache'], 114_688_000)  # 2 L H_kv d_h b n
        self.assertEqual(m['recurrent_state'], 0)

    def test_gemma_sharing_and_windows(self):
        cfg = fixture_config(GEMMA)
        for n, kv in [(100, 1_843_200), (10_000, 67_719_168)]:
            m = d2l.config_memory(cfg, n, 1)
            self.assertEqual(m['kv_cache'], kv)
            self.assertEqual(m['kv_cache'], 6_144 * n + 12_288 * min(n, 511))
            self.assertEqual(m['recurrent_state'], 0)
        m = d2l.config_memory(cfg, 10_000, 1, window_cached=512)
        self.assertEqual(m['kv_cache'], 67_731_456)
        self.assertEqual(d2l.config_memory(auto_config(GEMMA), 10_000, 1),
                         d2l.config_memory(cfg, 10_000, 1))

    def test_hybrid_crossover(self):
        """Cache plus state of the hybrid against an all-attention stack of
        the same shape: equal at 536 tokens."""
        cfg = fixture_config(QWEN35)
        full = json.loads(json.dumps(cfg))
        full['text_config']['layer_types'] = ['full_attention'] * 24
        per_token = d2l.config_memory(full, 1, 1)['kv_cache']
        self.assertEqual(per_token, 49_152)
        m = d2l.config_memory(cfg, 536, 1)
        self.assertEqual(m['kv_cache'] + m['recurrent_state'],
                         d2l.config_memory(full, 536, 1)['kv_cache'])

    def test_logits_to_keep(self):
        cfg = fixture_config(QWEN35)
        for keep in (None, 0):
            m = d2l.config_memory(cfg, 16_384, 1, logits_to_keep=keep)
            self.assertEqual(m['logits'], 8_136_949_760)
        m = d2l.config_memory(cfg, 4_096, 1, logits_to_keep=0)
        self.assertEqual(m['logits'], 2_034_237_440)
        m = d2l.config_memory(cfg, 4, 1, logits_to_keep=10)
        self.assertEqual(m['logits'], 4 * 248_320 * 2)

    def test_zero_conv_taps(self):
        """Regression: conv_taps=0 counts no convolution inputs instead of
        falling back to linear_conv_kernel_dim."""
        m = d2l.config_memory(fixture_config(QWEN35), 1024, 1, conv_taps=0)
        self.assertEqual(m['recurrent_state'], 18 * 16 * 128 * 128 * 4)

    def test_dtype_arguments(self):
        cfg = fixture_config(QWEN35)
        m = d2l.config_memory(cfg, 1024, 1, dtype_bytes=4, state_bytes=4,
                              conv_bytes=2)
        self.assertEqual(m['kv_cache'], 2 * 12_582_912)
        self.assertEqual(m['recurrent_state'], 19_759_104)
        self.assertEqual(m['weights'], 4 * m['params'])
        vision = d2l.config_memory(cfg, 1, 1, include_vision=True,
                                   include_mtp=True)
        self.assertEqual(vision['params'], 2_274_069_824)

    def test_warns_beyond_max_positions(self):
        cfg = fixture_config(GEMMA)
        with self.assertWarns(UserWarning):
            m = d2l.config_memory(cfg, 131_073, 1)
        self.assertGreater(m['total'], 0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            d2l.config_memory(cfg, 131_072, 1)

    def test_live_cache_matches(self):
        """The bytes a Transformers cache holds after a bf16 prefill on the
        CPU equal kv_cache + recurrent_state (fp32 state, bf16 conv inputs,
        4 conv taps, w - 1 window positions, no cache in shared layers)."""
        from transformers import Gemma4ForCausalLM
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM
        torch.manual_seed(0)
        k_eq_v = tiny_gemma_config(attention_k_eq_v=True,
                                   num_global_key_value_heads=2)
        cases = [(Qwen3_5ForCausalLM, tiny_qwen35_config().get_text_config()),
                 (Gemma4ForCausalLM, tiny_gemma_config().get_text_config()),
                 # K = V layers still cache both (V is normalized apart)
                 (Gemma4ForCausalLM, k_eq_v.get_text_config())]
        for cls, cfg in cases:
            model = cls(cfg).to(torch.bfloat16).eval()
            for n, B in [(5, 1), (20, 2)]:
                ids = torch.randint(0, 128, (B, n))
                with quiet(), torch.no_grad():
                    cache = model(ids, use_cache=True,
                                  logits_to_keep=1).past_key_values
                m = d2l.config_memory(model.config, n, B)
                self.assertEqual(cache_bytes(cache),
                                 m['kv_cache'] + m['recurrent_state'],
                                 (cls.__name__, n, B))


class _Hub(http.server.BaseHTTPRequestHandler):
    """A Hub stand-in with the redirect shape of huggingface.co (checked
    with curl, 2026-11-12). /<org>/<repo>/resolve/<rev>/<file> answers a
    shard with a 302 to a CDN on another host (`cdn` + /cdn/...) and any
    other file with a relative 307 to /api/resolve-cache/models/... on the
    Hub itself. The org 'gated' answers 401 on the Hub unless the request
    carries `Bearer hf_ok`; the CDN needs no token and honours byte ranges
    (or ignores them if `ranges` is False)."""
    root, ranges, log, cdn = None, True, None, None

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.log.append((self.path, dict(self.headers)))
        parts = self.path.split('?')[0].strip('/').split('/')
        if parts[0] == 'cdn':  # cdn/<org>/<repo>/<rev>/<file>
            self.serve(parts[1], parts[2], parts[4])
            return
        if parts[0] == 'api':  # api/resolve-cache/models/<org>/<repo>/...
            org, repo, rev, filename = parts[3:]
        else:  # <org>/<repo>/resolve/<rev>/<file>
            org, repo, _, rev, filename = parts
        if org == 'gated' and (self.headers.get('Authorization')
                               != 'Bearer hf_ok'):
            self.send_error(401)
        elif not os.path.exists(os.path.join(self.root, org, repo, filename)):
            self.send_error(404)
        elif parts[0] == 'api':
            self.serve(org, repo, filename)
        else:
            shard = filename.endswith('.safetensors')
            self.send_response(302 if shard else 307)
            self.send_header('Location', (
                f'{self.cdn}/cdn/{org}/{repo}/{rev}/{filename}' if shard else
                f'/api/resolve-cache/models/{org}/{repo}/{rev}/{filename}'
                '?etag=x'))
            self.end_headers()

    def serve(self, org, repo, filename):
        path = os.path.join(self.root, org, repo, filename)
        if not os.path.exists(path):
            self.send_error(404)
            return
        with open(path, 'rb') as fh:
            data = fh.read()
        byte_range = self.headers.get('Range')
        if byte_range and self.ranges:
            start, stop = map(int, byte_range.split('=')[1].split('-'))
            data = data[start:stop + 1]
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class SafetensorsHeader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from safetensors.torch import save_file
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name
        single = os.path.join(root, 'org', 'single')
        sharded = os.path.join(root, 'org', 'sharded')
        os.makedirs(single)
        os.makedirs(sharded)
        save_file({'w': torch.zeros(3, 4), 'b': torch.zeros(4,
                                                            dtype=torch.bfloat16),
                   'step': torch.zeros((), dtype=torch.int64)},
                  os.path.join(single, 'model.safetensors'),
                  metadata={'format': 'pt'})
        shards = {'model-00001-of-00002.safetensors':
                  {'a.weight': torch.zeros(2, 5, dtype=torch.bfloat16),
                   'a.bias': torch.zeros(5)},
                  'model-00002-of-00002.safetensors':
                  {'z.weight': torch.zeros(7, 1, 2, dtype=torch.float16)}}
        weight_map = {}
        for name, tensors in shards.items():
            save_file(tensors, os.path.join(sharded, name),
                      metadata={'format': 'pt'})
            weight_map.update({k: name for k in tensors})
        with open(os.path.join(sharded, 'model.safetensors.index.json'),
                  'w') as fh:
            json.dump({'metadata': {'total_size': 0},
                       'weight_map': weight_map}, fh)
        cls.files = {'org/single': [os.path.join(single, 'model.safetensors')],
                     'org/sharded': [os.path.join(sharded, n) for n in shards]}
        shutil.copytree(os.path.join(root, 'org'), os.path.join(root, 'gated'))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @contextlib.contextmanager
    def hub(self, ranges=True, token=None):
        """Serve the Hub and its CDN on two ports (two hosts to urllib)."""
        handler = type('Hub', (_Hub,), dict(root=self.tmp.name, ranges=ranges,
                                            log=[]))
        servers = []
        for _ in range(2):  # the CDN first: the Hub redirects to it
            servers.append(http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                                                           handler))
            threading.Thread(target=servers[-1].serve_forever,
                             kwargs=dict(poll_interval=0.05),
                             daemon=True).start()
        handler.cdn, base = (f'http://127.0.0.1:{s.server_address[1]}'
                             for s in servers)

        def url(repo_id, filename, revision=None, **kwargs):
            return f'{base}/{repo_id}/resolve/{revision}/{filename}'

        try:
            with mock.patch('huggingface_hub.hf_hub_url', side_effect=url), \
                    mock.patch('huggingface_hub.get_token',
                               return_value=token):
                yield handler.log
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()

    def expected(self, repo):
        from safetensors import safe_open
        out = {}
        for path in self.files[repo]:
            with safe_open(path, 'pt') as f:
                for k in f.keys():
                    s = f.get_slice(k)
                    out[k] = (tuple(s.get_shape()), s.get_dtype())
        return out

    def test_equals_safe_open(self):
        for ranges in (True, False):
            for repo in self.files:
                with self.hub(ranges=ranges) as log:
                    header = d2l.safetensors_header(repo, revision='r1')
                self.assertEqual(header, self.expected(repo), (repo, ranges))
                self.assertNotIn('__metadata__', header)
                cdn = [h for path, h in log if path.startswith('/cdn/')]
                tensor_reads = [h for path, h in log
                                if path.startswith('/cdn/')
                                and path.endswith('.safetensors')]
                self.assertEqual(len(tensor_reads), 2 * len(self.files[repo]))
                self.assertTrue(all('Range' in h for h in tensor_reads))
                self.assertTrue(all('r1' in path for path, _ in log))
                self.assertTrue(cdn)
        self.assertEqual(header['z.weight'], ((7, 1, 2), 'F16'))
        self.assertEqual(self.expected('org/single')['step'], ((), 'I64'))

    def test_token_stays_on_the_hub(self):
        """The token travels as an unredirected header on every request to
        the Hub, including its same-host 307 to the resolve cache, and
        never reaches the CDN."""
        requests, open_ = [], urllib.request.OpenerDirector.open

        def spy(self, request, *args, **kwargs):  # sees every redirect hop
            requests.append(request)
            return open_(self, request, *args, **kwargs)

        with self.hub(token='hf_secret') as log, \
                mock.patch.object(urllib.request.OpenerDirector, 'open', spy):
            d2l.safetensors_header('org/sharded', revision='r1')
        hub = [h for path, h in log if not path.startswith('/cdn/')]
        cdn = [h for path, h in log if path.startswith('/cdn/')]
        self.assertTrue(any(path.startswith('/api/') for path, _ in log))
        self.assertEqual(len(cdn), 4)
        self.assertTrue(all(h.get('Authorization') == 'Bearer hf_secret'
                            for h in hub))
        self.assertFalse(any('Authorization' in h for h in cdn))
        self.assertEqual(len(requests), len(log))
        hub_host = requests[0].host
        for request in requests:
            self.assertNotIn('Authorization', request.headers)
            self.assertEqual(
                request.unredirected_hdrs.get('Authorization'),
                'Bearer hf_secret' if request.host == hub_host else None)
        with self.hub(token=None) as log:
            d2l.safetensors_header('org/sharded', revision='r1')
        self.assertFalse(any('Authorization' in h for _, h in log))

    def test_gated_repository(self):
        """Regression: the index of a gated sharded repository sits behind
        the Hub's same-host 307, which must keep the token."""
        for repo in self.files:
            gated = repo.replace('org/', 'gated/')
            with self.hub(token='hf_ok'):
                header = d2l.safetensors_header(gated, revision='r1')
            self.assertEqual(header, self.expected(repo), repo)
            for token in (None, 'hf_wrong'):
                with self.hub(token=token):
                    with self.assertRaises(urllib.error.HTTPError) as e:
                        d2l.safetensors_header(gated, revision='r1')
                self.assertEqual(e.exception.code, 401)

    def test_revision_defaults(self):
        pin = 'a' * 40
        with mock.patch.dict(d2l.HF_REVISIONS, {'org/single': pin}), \
                self.hub() as log:
            d2l.safetensors_header('org/single')
        self.assertTrue(all(pin in path for path, _ in log))
        head, out = 'f' * 40, io.StringIO()
        with mock.patch('huggingface_hub.HfApi') as api, self.hub() as log, \
                contextlib.redirect_stdout(out):
            api.return_value.model_info.return_value.sha = head
            d2l.safetensors_header('org/sharded')  # unpinned: its head commit
        api.return_value.model_info.assert_called_once_with('org/sharded')
        self.assertIn(f'unpinned: org/sharded@{head}', out.getvalue())
        self.assertTrue(all(head in path for path, _ in log))

    def test_stalled_server_times_out(self):
        """Regression: a server that accepts the connection and never
        answers must not block the notebook until its execution limit."""
        import inspect
        self.assertEqual(inspect.signature(d2l.safetensors_header)
                         .parameters['timeout'].default, 30)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            outcome = []

            def url(repo_id, filename, revision=None, **kwargs):
                return f'http://127.0.0.1:{port}/{repo_id}/{filename}'

            def call():
                try:
                    d2l.safetensors_header('org/x', revision='r', timeout=0.5)
                except Exception as e:
                    outcome.append(e)

            with mock.patch('huggingface_hub.hf_hub_url', side_effect=url), \
                    mock.patch('huggingface_hub.get_token',
                               return_value=None):
                thread = threading.Thread(target=call, daemon=True)
                thread.start()
                thread.join(10)
        self.assertFalse(thread.is_alive(), 'still blocked after 10 s')
        self.assertIsInstance(outcome[0], OSError)  # timeout or URLError

    def test_fixture_header(self):
        header = fixture_header(QWEN35)
        dtypes = {}
        for _, dtype in header.values():
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
        self.assertEqual(dtypes, {'BF16': 596, 'F32': 36})
        self.assertEqual(numel(header), 2_274_069_824)

    @unittest.skipUnless(os.environ.get('D2L_TEST_NETWORK') == '1',
                         'network test: set D2L_TEST_NETWORK=1')
    def test_hub_qwen35_2b(self):
        header = d2l.safetensors_header(QWEN35)
        self.assertEqual(len(header), 632)
        self.assertEqual(numel(header), 2_274_069_824)
        self.assertEqual(header, fixture_header(QWEN35))


class LoadHF(unittest.TestCase):
    def load(self, model_id, config, revision=None, missing=(), n_gpus=0,
             device_map='auto', hf_device_map=None, unexpected=(),
             config_eos=None, tokenizer_eos=None):
        import transformers
        model = mock.MagicMock()
        model.eval.return_value = model
        model.hf_device_map = hf_device_map or {'': 'cpu'}
        model.generation_config = transformers.GenerationConfig(
            eos_token_id=config_eos)
        tokenizer = mock.NonCallableMock(eos_token_id=tokenizer_eos)
        info = {'missing_keys': set(missing),
                'unexpected_keys': list(unexpected), 'mismatched_keys': [],
                'error_msgs': []}
        out = io.StringIO()
        with mock.patch.object(transformers.AutoConfig, 'from_pretrained',
                               return_value=config) as auto_config_, \
                mock.patch.object(transformers.AutoModelForCausalLM,
                                  'from_pretrained',
                                  return_value=(model, info)) as auto_model, \
                mock.patch.object(transformers.AutoTokenizer,
                                  'from_pretrained',
                                  return_value=tokenizer) as auto_tok, \
                mock.patch('torch.cuda.device_count', return_value=n_gpus), \
                contextlib.redirect_stdout(out):
            result = d2l.load_hf(model_id, revision=revision,
                                 device_map=device_map)
        self.assertEqual(result, (model, tokenizer))
        model.eval.assert_called_once()
        return (model, auto_config_.call_args, auto_model.call_args,
                auto_tok.call_args, out.getvalue())

    def test_one_revision_for_model_and_tokenizer(self):
        pin = d2l.HF_REVISIONS[QWEN35]
        model, c, m, t, out = self.load(QWEN35, auto_config(QWEN35))
        for call in (c, m, t):
            self.assertEqual(call.args, (QWEN35,))
            self.assertEqual(call.kwargs['revision'], pin)
            self.assertIs(call.kwargs['trust_remote_code'], False)
        self.assertIs(m.kwargs['use_safetensors'], True)
        self.assertIs(m.kwargs['output_loading_info'], True)
        self.assertEqual(m.kwargs['dtype'], torch.bfloat16)
        self.assertEqual(model.revision, pin)
        self.assertIn('0 missing', out)
        _, c, m, t, _ = self.load(QWEN35, auto_config(QWEN35),
                                  revision='b' * 40)
        self.assertEqual({c.kwargs['revision'], m.kwargs['revision'],
                          t.kwargs['revision']}, {'b' * 40})

    def test_unpinned_resolves_head(self):
        with mock.patch('huggingface_hub.HfApi') as api:
            api.return_value.model_info.return_value.sha = 'f' * 40
            _, c, m, t, out = self.load('org/new-model',
                                        auto_config('Qwen/Qwen3-0.6B'))
        api.return_value.model_info.assert_called_once_with('org/new-model')
        self.assertIn(f'unpinned: org/new-model@{"f" * 40}', out)
        self.assertEqual(m.kwargs['revision'], 'f' * 40)
        self.assertEqual(t.kwargs['revision'], 'f' * 40)

    def test_text_class_and_key_mapping(self):
        from transformers import Gemma4TextConfig, Qwen3Config
        from transformers.models.qwen3_5 import Qwen3_5TextConfig
        _, _, m, _, _ = self.load(GEMMA, auto_config(GEMMA))
        self.assertIsInstance(m.kwargs['config'], Gemma4TextConfig)
        self.assertEqual(m.kwargs['key_mapping'],
                         {r'^model\.language_model\.': 'model.'})
        _, _, m, _, _ = self.load(QWEN35, auto_config(QWEN35))
        self.assertIsInstance(m.kwargs['config'], Qwen3_5TextConfig)
        self.assertNotIn('key_mapping', m.kwargs)
        dense = auto_config('Qwen/Qwen3-0.6B')
        _, _, m, _, _ = self.load('Qwen/Qwen3-0.6B', dense)
        self.assertIsInstance(m.kwargs['config'], Qwen3Config)
        self.assertIs(m.kwargs['config'], dense)
        self.assertNotIn('key_mapping', m.kwargs)

    def test_missing_keys_raise(self):
        try:  # not assertRaises: it drops the traceback
            self.load(QWEN35, auto_config(QWEN35),
                      missing=['model.norm.weight'])
            self.fail('no RuntimeError')
        except RuntimeError as e:
            self.assertIn('model.norm.weight', str(e))
            tb = e.__traceback__
        # Regression: the traceback Jupyter keeps must not hold the model
        frames = [frame for frame, _ in traceback.walk_tb(tb)
                  if frame.f_code.co_name == 'load_hf']
        self.assertEqual(len(frames), 1)
        self.assertNotIn('model', frames[0].f_locals)

    def test_stop_tokens(self):
        """Regression: without generation_config.json (Qwen3.5-0.8B, 2B and
        4B), `generate` stops only at the configuration's <|endoftext|>
        (248044), not at the tokenizer's <|im_end|> (248046) ending a chat
        turn; load_hf adds the tokenizer's end token."""
        cfg = auto_config(QWEN35)
        cases = [(248044, 248046, [248044, 248046]),
                 ([1, 106], 1, [1, 106]), ([1, 106], 106, [1, 106]),
                 (151643, None, [151643]), (None, 7, [7]), (None, None, [])]
        for config_eos, tokenizer_eos, expected in cases:
            model, _, _, _, out = self.load(QWEN35, cfg, config_eos=config_eos,
                                            tokenizer_eos=tokenizer_eos)
            stop = model.generation_config.eos_token_id
            self.assertEqual([stop] if isinstance(stop, int) else stop or [],
                             expected, (config_eos, tokenizer_eos))
            self.assertIn(f'stop tokens {expected}', out)

    def test_load_report(self):
        unexpected = ['model.audio_tower.a', 'model.audio_tower.b',
                      'model.vision_tower.c']
        _, _, _, _, out = self.load(GEMMA, auto_config(GEMMA),
                                    unexpected=unexpected)
        self.assertEqual(len(out.strip().splitlines()), 1)
        self.assertIn("'model.audio_tower': 2", out)
        self.assertIn("'model.vision_tower': 1", out)
        self.assertIn(d2l.HF_REVISIONS[GEMMA][:12], out)

    def test_device_map(self):
        cfg = auto_config('Qwen/Qwen3-0.6B')
        for n_gpus, expected in [(0, 'cpu'), (1, {'': 0}), (2, 'auto')]:
            _, _, m, _, _ = self.load('Qwen/Qwen3-0.6B', cfg, n_gpus=n_gpus)
            self.assertEqual(m.kwargs['device_map'], expected, n_gpus)
        _, _, m, _, _ = self.load('Qwen/Qwen3-0.6B', cfg, n_gpus=4,
                                  device_map='cuda:1')
        self.assertEqual(m.kwargs['device_map'], 'cuda:1')

    def test_warns_when_spread_over_devices(self):
        cfg = auto_config('Qwen/Qwen3-0.6B')
        with self.assertWarnsRegex(UserWarning, 'spread over'):
            self.load('Qwen/Qwen3-0.6B', cfg, n_gpus=2,
                      hf_device_map={'model.layers.0': 0, 'lm_head': 'cpu'})
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.load('Qwen/Qwen3-0.6B', cfg, n_gpus=2,
                      hf_device_map={'model.layers.0': 0, 'lm_head': 0})

    def test_tiny_gemma_end_to_end(self):
        """A tiny random multimodal Gemma 4 checkpoint, with the unused K/V
        of its shared layers stored as in the released file, loads through
        load_hf as the text class: no missing keys, the text weights equal
        the saved ones, and the parameter count equals config_params."""
        from safetensors.torch import load_file, save_file
        from transformers import Gemma4ForCausalLM
        from transformers.models.gemma4 import Gemma4ForConditionalGeneration
        cfg = tiny_gemma_config()
        torch.manual_seed(0)
        full = Gemma4ForConditionalGeneration(cfg)
        with tempfile.TemporaryDirectory() as path:
            with quiet():
                full.save_pretrained(path)
            save_tiny_tokenizer(path)
            weights = load_file(os.path.join(path, 'model.safetensors'))
            first_shared = TINY_GEMMA['num_hidden_layers'] - TINY_GEMMA[
                'num_kv_shared_layers']
            for i in range(first_shared, TINY_GEMMA['num_hidden_layers']):
                prefix = f'model.language_model.layers.{i}.self_attn.'
                source = f'model.language_model.layers.0.self_attn.'
                for name in ('k_proj.weight', 'v_proj.weight',
                             'k_norm.weight'):
                    weights[prefix + name] = weights[source + name].clone()
            save_file(weights, os.path.join(path, 'model.safetensors'),
                      metadata={'format': 'pt'})
            with quiet() as stack:
                out = io.StringIO()
                stack.enter_context(contextlib.redirect_stdout(out))
                model, tokenizer = d2l.load_hf(path, revision='main',
                                               dtype=torch.float32,
                                               device_map='cpu')
        self.assertIsInstance(model, Gemma4ForCausalLM)
        self.assertFalse(model.training)
        self.assertEqual(model.revision, 'main')
        self.assertIn('0 missing keys', out.getvalue())
        self.assertIn('model.audio_tower', out.getvalue())
        self.assertEqual(model.num_parameters(), d2l.config_params(cfg))
        saved = full.state_dict()
        for name, tensor in model.state_dict().items():
            source = ('model.language_model.' + name[len('model.'):]
                      if name.startswith('model.') else
                      'model.language_model.embed_tokens.weight')
            self.assertTrue(torch.equal(tensor, saved[source]), name)
        self.assertEqual(tokenizer('cache')['input_ids'], [2])

    def test_tiny_qwen35_stops_at_tokenizer_eos(self):
        """End to end: a repository without generation_config.json whose
        tokenizer ends turns with a token the configuration does not name;
        greedy `generate` stops there instead of running to max_new_tokens."""
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
        cfg = tiny_qwen35_config()
        torch.manual_seed(0)
        full = Qwen3_5ForConditionalGeneration(cfg).eval()
        prompt = torch.tensor([[2]])
        with torch.no_grad():  # the token greedy decoding emits first
            first = int(full(input_ids=prompt).logits[0, -1].argmax())
        self.assertIsNone(cfg.text_config.eos_token_id)
        vocab = {f't{i}': i for i in range(TINY_QWEN35['vocab_size'])}
        tok = Tokenizer(models.WordLevel(vocab, unk_token='t1'))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        with tempfile.TemporaryDirectory() as path:
            with quiet():
                full.save_pretrained(path)
            os.remove(os.path.join(path, 'generation_config.json'))
            PreTrainedTokenizerFast(
                tokenizer_object=tok, unk_token='t1',
                eos_token=f't{first}').save_pretrained(path)
            with quiet():
                model, tokenizer = d2l.load_hf(path, revision='main',
                                               dtype=torch.float32,
                                               device_map='cpu')
        self.assertEqual(tokenizer.eos_token_id, first)
        self.assertEqual(model.generation_config.eos_token_id, [first])
        with quiet():
            ids = model.generate(prompt, do_sample=False, max_new_tokens=4,
                                 pad_token_id=0)
        self.assertEqual(ids.tolist(), [[2, first]])

    def test_tiny_qwen35_end_to_end(self):
        """The library renames Qwen3.5's keys itself: no key_mapping."""
        from transformers.models.qwen3_5 import (
            Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration)
        cfg = tiny_qwen35_config()
        torch.manual_seed(0)
        full = Qwen3_5ForConditionalGeneration(cfg)
        with tempfile.TemporaryDirectory() as path:
            with quiet():
                full.save_pretrained(path)
            save_tiny_tokenizer(path)
            with quiet():
                model, _ = d2l.load_hf(path, revision='main',
                                       dtype=torch.float32, device_map='cpu')
        self.assertIsInstance(model, Qwen3_5ForCausalLM)
        self.assertEqual(model.num_parameters(), d2l.config_params(cfg))
        saved = full.state_dict()
        for name, tensor in model.model.state_dict().items():
            self.assertTrue(torch.equal(
                tensor, saved['model.language_model.' + name]), name)


if __name__ == '__main__':
    unittest.main()
