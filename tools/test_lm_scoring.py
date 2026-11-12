#!/usr/bin/env python3
"""CPU tests for sec_lm_scoring (chapter_lm-pretrained/scoring.md).

Library code (spec 18.4 part 5):
- `format_mc`: exact strings for a 3- and a 10-option item with one shot,
  in both formats; the 26-letter limit and the other errors.
- `score_choices` and `ScoreComponents`: a toy bigram module with fixed
  logits and a character tokenizer stub (right-padded path); a tiny random
  Transformers Qwen3 and a tiny hybrid Qwen3.5 (left-padded path,
  `logits_to_keep`, position ids) with the real Qwen3-0.6B-Base tokenizer;
  tiny `d2l.Qwen3GPT` and `d2l.GPT`. Against a hand loop over the full
  log-softmax; batch sizes; ragged choices; the whitespace move; the
  harness's character and byte lengths; empty continuations; empty and
  over-long prompts; the letter format against `d2l.label_logprobs`.
- `bits_per_byte`: uniform logits give T log2 V / B exactly; the boundary
  token (resolution order, first-token rule without one); windows score
  every token once over a grid of (T, ctx_len, stride) and follow the
  spec's worked example; the chunked head of a Transformers model equals
  its full logits across the 1,024-position chunk boundary and padding.
- `pass_at_k`, `pass_at_k_variance`: closed forms, the plug-in bound,
  broadcasting, errors, and a Monte Carlo of the moments.
- `extract_answer`: the twelve strings of cell `scoring-extract-check`
  plus `strict=True`.
- `run_python_tests`: the four fixtures of cell `scoring-run-tests-check`,
  stdin, the address-space limit.
- `plot_intervals` (book-side, `sec_utils`, requested by this section):
  a smoke test of both kinds.
- `import d2l.torch` and the scorers work with `transformers` blocked.

The Transformers tests load the Qwen3-0.6B-Base tokenizer at its pinned
revision from the HF cache (no weights) and skip, with the command that
caches it, when it is missing.

Run: `.venv-lm/bin/python -m unittest tools/test_lm_scoring.py`, or
`make test-lm`.
"""

import math
import os
import subprocess
import sys
import time
import unittest
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

BASE = 'Qwen/Qwen3-0.6B-Base'
CACHE_CMD = ('.venv-lm/bin/python -c "from transformers import '
             f"AutoTokenizer; AutoTokenizer.from_pretrained('{BASE}', "
             "revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd')\"")
QUESTION = 'Which gas do plants take in from the air to make their food?'
OPTIONS = ['oxygen', 'carbon dioxide', 'nitrogen', 'water vapor']


# ---------------------------------------------------------------- helpers
class CharTokenizer:
    """One token per character (ids 0-255, Latin-1); optional special
    tokens: '<|endoftext|>' = 256 when `endoftext`, an EOS = 257 when
    `eos`. Without `vocab` it has no convert_tokens_to_ids (like
    tiktoken's Encoding)."""
    V = 258

    def __init__(self, endoftext=False, eos=False, vocab=True):
        self.special = {'<|endoftext|>': 256} if endoftext else {}
        if eos:
            self.eos_token_id = 257
        if vocab:
            self.convert_tokens_to_ids = lambda t: self.special.get(t)

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return ''.join(chr(i) for i in ids if i < 256)


class Bigram(nn.Module):
    """Fixed logits W[previous token]; returns a plain tensor, or an object
    with `.logits` when `wrap`."""
    def __init__(self, V=CharTokenizer.V, seed=0, wrap=False, max_len=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(V, V, generator=g))
        self.wrap = wrap
        if max_len is not None:
            self.max_len = max_len
        self.inputs = []

    def forward(self, X):
        self.inputs.append(X.tolist())
        z = self.W[X]
        if self.wrap:
            return type('Out', (), {'logits': z})()
        return z


class Uniform(Bigram):
    def forward(self, X):
        self.inputs.append(X.tolist())
        return torch.zeros(*X.shape, self.W.shape[0]) + 0 * self.W[0, 0]


def hand_ll(model, ids_ctx, ids_cont):
    """Sum of log p(cont_t | ctx, cont_<t) from one unpadded forward pass
    over the full sequence, full-vocabulary log-softmax in fp32."""
    X = torch.tensor([ids_ctx + ids_cont])
    with torch.no_grad():
        out = model(X) if not hasattr(model, 'config') else model(
            X, use_cache=False)
    z = getattr(out, 'logits', out)[0].float().log_softmax(-1)
    n = len(ids_ctx)
    return sum(z[n - 1 + t, u].item() for t, u in enumerate(ids_cont))


def qwen_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise unittest.SkipTest(f'transformers not installed: {e}')
    try:
        return AutoTokenizer.from_pretrained(
            BASE, revision=d2l.HF_REVISIONS[BASE], local_files_only=True)
    except OSError:
        raise unittest.SkipTest(
            f'{BASE} tokenizer not in the HF cache; cache it with '
            f'{CACHE_CMD}')


def _randomize(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(1 + 0.3 * torch.randn_like(p) if 'norm' in name
                    else 0.08 * torch.randn_like(p))
    return model.eval()


def tiny_qwen3(seed=0):
    """A random Qwen3ForCausalLM over the real 151,936-row vocabulary:
    2 layers, hidden size 64, fp32."""
    import transformers
    config = transformers.Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_hidden_layers=2, tie_word_embeddings=True,
        attn_implementation='sdpa')
    return _randomize(transformers.Qwen3ForCausalLM(config), seed)


def tiny_qwen35(seed=0):
    """A random hybrid Qwen3_5ForCausalLM (three Gated DeltaNet layers, one
    attention layer), hidden size 64, over Qwen3's vocabulary size."""
    import transformers
    config = transformers.Qwen3_5TextConfig(
        vocab_size=151936, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=['linear_attention'] * 3 + ['full_attention'],
        tie_word_embeddings=True, attn_implementation='sdpa')
    return _randomize(transformers.Qwen3_5ForCausalLM(config), seed)


# ---------------------------------------------------------------- format_mc
class TestFormatMC(unittest.TestCase):
    def test_three_options_one_shot(self):
        shot = ('What is ice?', ['solid water', 'a gas'], 0)
        prompt, choices = d2l.format_mc('Pick one.', ['red', 'green', 'blue'],
                                        shots=[shot])
        self.assertEqual(prompt, 'Question: What is ice?\nAnswer: solid '
                         'water\n\nQuestion: Pick one.\nAnswer:')
        self.assertEqual(choices, [' red', ' green', ' blue'])
        prompt, choices = d2l.format_mc('Pick one.', ['red', 'green', 'blue'],
                                        'letter', shots=[shot])
        self.assertEqual(prompt, 'Question: What is ice?\nA. solid water\n'
                         'B. a gas\nAnswer: A\n\nQuestion: Pick one.\n'
                         'A. red\nB. green\nC. blue\nAnswer:')
        self.assertEqual(choices, [' A', ' B', ' C'])

    def test_ten_options(self):
        opts = [f'o{i}' for i in range(10)]
        shot = ('S?', ['x', 'y', 'z'], 2)
        prompt, choices = d2l.format_mc('Q?', opts, 'letter', shots=[shot])
        self.assertEqual(prompt, 'Question: S?\nA. x\nB. y\nC. z\nAnswer: C'
                         '\n\nQuestion: Q?\n' + ''.join(
                             f'{L}. o{i}\n' for i, L in
                             enumerate('ABCDEFGHIJ')) + 'Answer:')
        self.assertEqual(choices, [' ' + L for L in 'ABCDEFGHIJ'])
        prompt, choices = d2l.format_mc('Q?', opts)
        self.assertEqual(prompt, 'Question: Q?\nAnswer:')
        self.assertEqual(choices, [' ' + o for o in opts])

    def test_no_shots_default(self):
        self.assertEqual(d2l.format_mc(QUESTION, OPTIONS),
                         (f'Question: {QUESTION}\nAnswer:',
                          [' oxygen', ' carbon dioxide', ' nitrogen',
                           ' water vapor']))

    def test_errors(self):
        opts = [f'o{i}' for i in range(27)]
        with self.assertRaisesRegex(ValueError, '26'):
            d2l.format_mc('Q?', opts, 'letter')
        d2l.format_mc('Q?', opts, 'cloze')  # no letters, no limit
        with self.assertRaisesRegex(ValueError, '26'):  # in a shot too
            d2l.format_mc('Q?', ['a'], 'letter', shots=[('S', opts, 0)])
        with self.assertRaises(ValueError):
            d2l.format_mc('Q?', ['a'], 'mcq')
        with self.assertRaises(ValueError):
            d2l.format_mc('Q?', [], 'letter')
        with self.assertRaises(ValueError):
            d2l.format_mc('Q?', ['a'], shots=[('S', ['x', 'y'], 2)])


# ---------------------------------------------------------- score_choices
class TestScoreChoicesToy(unittest.TestCase):
    """The right-padded path: a bigram module and a character tokenizer."""
    def setUp(self):
        self.tok = CharTokenizer(eos=True)
        self.model = Bigram()
        self.prompts = ['Answer:', 'Q: 2+2?\nAnswer:', 'Long prompt here:']
        self.choices = [[' carbon dioxide', ' x'], [' 4', ' five', ' 22'],
                        [' a']]

    def test_sum_against_hand_loop(self):
        s = d2l.score_choices(self.model, self.tok, self.prompts,
                              self.choices, 'sum')
        self.assertEqual(s.shape, (3, 3))
        self.assertEqual(s.dtype, torch.float32)
        for i, (p, cs) in enumerate(zip(self.prompts, self.choices)):
            for j, c in enumerate(cs):
                ref = hand_ll(self.model, self.tok.encode(p),
                              self.tok.encode(c))
                self.assertAlmostEqual(s[i, j].item(), ref, delta=1e-5)

    def test_ragged_padding(self):
        s = d2l.score_choices(self.model, self.tok, self.prompts,
                              self.choices)
        self.assertEqual(s[0, 2].item(), -math.inf)
        self.assertTrue(torch.isinf(s[2, 1:]).all())
        self.assertTrue(torch.isfinite(s[1]).all())
        self.assertTrue((s.argmax(-1) < torch.tensor([2, 3, 1])).all())
        comps = d2l.score_choices(self.model, self.tok, self.prompts,
                                  self.choices, return_components=True)
        self.assertEqual(comps.n_tokens[0, 2].item(), 0)

    def test_batch_sizes_identical(self):
        for wrap in (False, True):  # with and without .logits
            model = Bigram(wrap=wrap)
            one = d2l.score_choices(model, self.tok, self.prompts,
                                    self.choices, batch_size=1)
            for bs in (3, 64):
                self.assertTrue(torch.equal(one, d2l.score_choices(
                    model, self.tok, self.prompts, self.choices,
                    batch_size=bs)))

    def test_shared_choices(self):
        shared = [' A', ' B', ' C']
        a = d2l.score_choices(self.model, self.tok, self.prompts, shared)
        b = d2l.score_choices(self.model, self.tok, self.prompts,
                              [shared] * 3)
        self.assertTrue(torch.equal(a, b))

    def test_whitespace_move(self):
        a = d2l.score_choices(self.model, self.tok, ['Answer: '], [['x']],
                              'sum', return_components=True)
        b = d2l.score_choices(self.model, self.tok, ['Answer:'], [[' x']],
                              'sum', return_components=True)
        for u, v in zip(a, b):
            self.assertTrue(torch.equal(u, v))

    def test_lengths(self):
        c = d2l.score_choices(self.model, self.tok, ['Answer:', 'Menu:'],
                              [[' carbon dioxide'], [' café']],
                              return_components=True)
        self.assertEqual(c.n_chars[0, 0].item(), 14)
        self.assertEqual(c.n_bytes[0, 0].item(), 14)
        self.assertEqual(c.n_chars[1, 0].item(), 4)
        self.assertEqual(c.n_bytes[1, 0].item(), 5)
        self.assertEqual(c.n_tokens[0, 0].item(), 15)  # one per character
        # only one leading whitespace character is the delimiter
        c = d2l.score_choices(self.model, self.tok, ['A:'], [['  x']],
                              return_components=True)
        self.assertEqual(c.n_chars[0, 0].item(), 2)

    def test_components_give_every_norm(self):
        comps = d2l.score_choices(self.model, self.tok, self.prompts,
                                  self.choices, return_components=True)
        self.assertIsInstance(comps, d2l.ScoreComponents)
        self.assertEqual(comps._fields,
                         ('ll', 'n_tokens', 'n_chars', 'n_bytes'))
        for norm, den in (('sum', 1), ('token', comps.n_tokens),
                          ('char', comps.n_chars),
                          ('byte', comps.n_bytes)):
            s = d2l.score_choices(self.model, self.tok, self.prompts,
                                  self.choices, norm)
            torch.testing.assert_close(s, comps.ll / den)
        with self.assertRaises(ValueError):
            d2l.score_choices(self.model, self.tok, self.prompts,
                              self.choices, 'mean')

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, 'empty continuation'):
            d2l.score_choices(self.model, self.tok, ['Answer:'], [['x', '']])
        with self.assertRaises(ValueError):
            d2l.score_choices(self.model, self.tok, ['a', 'b'], [['x']] * 3)

    def test_empty_prompt_gets_boundary(self):
        for tok, b in ((CharTokenizer(endoftext=True, eos=True), 256),
                       (CharTokenizer(eos=True), 257)):
            s = d2l.score_choices(self.model, tok, [''], [['ab']], 'sum')
            self.assertAlmostEqual(s[0, 0].item(), hand_ll(
                self.model, [b], tok.encode('ab')), delta=1e-5)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            d2l.score_choices(self.model, CharTokenizer(), [''], [['ab']])

    def test_long_prompt_cut_from_left(self):
        model = Bigram(max_len=8)
        prompts, choices = ['0123456789:', 'ab:'], [[' xy', ' z'], [' q']]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            s = d2l.score_choices(model, self.tok, prompts, choices, 'sum')
        msgs = [str(x.message) for x in w]
        self.assertEqual(len(msgs), 1)  # once per item, naming it
        self.assertIn('item 0', msgs[0])
        self.assertTrue(all(len(x) <= 8 for X in model.inputs for x in X))
        # the bigram sees only the previous token: the cut changes nothing
        ref = hand_ll(Bigram(), self.tok.encode('0123456789:'),
                      self.tok.encode(' xy'))
        self.assertAlmostEqual(s[0, 0].item(), ref, delta=1e-5)
        with self.assertRaisesRegex(ValueError, 'does not fit'):
            d2l.score_choices(Bigram(max_len=4), self.tok, ['ab'],
                              [[' long choice']])


class TestScoreChoicesD2l(unittest.TestCase):
    """Tiny d2l.Qwen3GPT (logits_to_keep) and d2l.GPT, right padding."""
    def test_d2l_models(self):
        tok = CharTokenizer(eos=True)
        prompts = ['Question: which?\nAnswer:', 'Q?\nAnswer:']
        choices = [[' carbon dioxide', ' oxygen'], [' A', ' Bee', ' C']]
        torch.manual_seed(0)
        qwen = d2l.Qwen3GPT(CharTokenizer.V, 32, 4, 2, 2, head_dim=8).eval()
        gpt = d2l.GPT(CharTokenizer.V, num_hiddens=32, num_heads=4,
                      num_blks=2).eval()
        for model in (qwen, gpt):
            one = d2l.score_choices(model, tok, prompts, choices, 'sum',
                                    batch_size=1)
            torch.testing.assert_close(one, d2l.score_choices(
                model, tok, prompts, choices, 'sum', batch_size=5),
                atol=1e-5, rtol=0)
            for i, (p, cs) in enumerate(zip(prompts, choices)):
                for j, c in enumerate(cs):
                    self.assertAlmostEqual(one[i, j].item(), hand_ll(
                        model, tok.encode(p), tok.encode(c)), delta=1e-4)
        keeps = []
        orig = qwen.forward
        qwen.forward = lambda X, **kw: (keeps.append((X.shape, kw)),
                                        orig(X, **kw))[1]
        d2l.score_choices(qwen, tok, prompts, choices, batch_size=5)
        # five rows, longest input 24 + 15 - 1 = 38 tokens; the earliest
        # continuation is predicted from position 10 - 1 = 9
        (shape, kw), = keeps
        self.assertEqual(shape, (5, 38))
        self.assertEqual(kw, {'logits_to_keep': 38 - 9})


class TestScoreChoicesHF(unittest.TestCase):
    """The left-padded Transformers path."""
    @classmethod
    def setUpClass(cls):
        cls.tok = qwen_tokenizer()
        cls.model = tiny_qwen3()
        cls.items = [d2l.format_mc(QUESTION, OPTIONS),
                     d2l.format_mc('Which is a metal?', ['iron', 'wood',
                                                         'glass'])]
        cls.prompts = [p for p, _ in cls.items]
        cls.choices = [c for _, c in cls.items]

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)['input_ids']

    def test_against_hand_loop(self):
        s = d2l.score_choices(self.model, self.tok, self.prompts,
                              self.choices, 'sum')
        for i, (p, cs) in enumerate(self.items):
            for j, c in enumerate(cs):
                full, ctx = self.encode(p + c), self.encode(p)
                self.assertEqual(full[:len(ctx)], ctx)  # no boundary merge
                self.assertAlmostEqual(s[i, j].item(), hand_ll(
                    self.model, ctx, full[len(ctx):]), delta=1e-5)
        comps = d2l.score_choices(self.model, self.tok, self.prompts,
                                  self.choices, return_components=True)
        self.assertEqual(comps.n_tokens[0].tolist(), [1, 2, 1, 2])
        self.assertEqual(comps.n_chars[0].tolist(), [6, 14, 8, 11])

    def test_batch_sizes(self):
        for model in (self.model, tiny_qwen35()):  # hybrid: left padding
            one = d2l.score_choices(model, self.tok, self.prompts,
                                    self.choices, batch_size=1)
            for bs in (3, 8):
                torch.testing.assert_close(d2l.score_choices(
                    model, self.tok, self.prompts, self.choices,
                    batch_size=bs), one, atol=1e-5, rtol=0)

    def test_hybrid_against_hand_loop(self):
        model = tiny_qwen35()
        s = d2l.score_choices(model, self.tok, self.prompts, self.choices,
                              'sum', batch_size=8)
        for i, (p, cs) in enumerate(self.items):
            for j, c in enumerate(cs):
                full, ctx = self.encode(p + c), self.encode(p)
                self.assertAlmostEqual(s[i, j].item(), hand_ll(
                    model, ctx, full[len(ctx):]), delta=1e-5)

    def test_transformers_call(self):
        calls = []
        h = self.model.register_forward_pre_hook(
            lambda m, args, kw: calls.append(kw), with_kwargs=True)
        try:
            d2l.score_choices(self.model, self.tok, self.prompts,
                              self.choices, batch_size=8)
        finally:
            h.remove()
        (kw,) = calls  # seven rows, one batch
        self.assertEqual(kw['logits_to_keep'], 2)  # longest continuation
        self.assertFalse(kw['use_cache'])
        mask, pos = kw['attention_mask'], kw['position_ids']
        self.assertTrue((mask[:, -1] == 1).all())  # left padding
        self.assertTrue((mask[:, :1] == 0).any())
        lengths = mask.sum(-1)
        for row, n in zip(pos, lengths):
            self.assertEqual(row[-n:].tolist(), list(range(n)))
        self.assertEqual(kw['input_ids'][mask == 0].unique().tolist(),
                         [self.tok.pad_token_id])
        # length-sorted: the longest inputs first
        self.assertEqual(lengths.tolist(), sorted(lengths.tolist(),
                                                  reverse=True))

    def test_whitespace_move(self):
        a = d2l.score_choices(self.model, self.tok, ['Answer: '],
                              [['carbon']], 'sum')
        b = d2l.score_choices(self.model, self.tok, ['Answer:'],
                              [[' carbon']], 'sum')
        self.assertEqual(a.item(), b.item())

    def test_letter_format_is_label_logprobs(self):
        prompts, choices = zip(*[d2l.format_mc(QUESTION, OPTIONS, 'letter'),
                                 d2l.format_mc('Which is a metal?',
                                               ['iron', 'wood', 'glass',
                                                'tin'], 'letter')])
        s = d2l.score_choices(self.model, self.tok, list(prompts),
                              list(choices), 'sum')
        ref = d2l.label_logprobs(self.model, self.tok, list(prompts),
                                 list('ABCD'))
        torch.testing.assert_close(s.log_softmax(-1), ref, atol=1e-5,
                                   rtol=0)
        # one token per letter: every norm ranks alike
        for norm in ('token', 'char', 'byte'):
            self.assertTrue(torch.equal(d2l.score_choices(
                self.model, self.tok, list(prompts), list(choices),
                norm).argmax(-1), s.argmax(-1)))


# ---------------------------------------------------------- bits_per_byte
class TestBitsPerByte(unittest.TestCase):
    def test_uniform_logits(self):
        V = CharTokenizer.V
        tok = CharTokenizer(endoftext=True, eos=True)
        texts = ['café au lait', 'naïve', 'plain']
        T = sum(len(t) for t in texts)
        B = sum(len(t.encode()) for t in texts)
        self.assertNotEqual(T, B)
        for ctx, stride in ((4, 2), (64, None), (3, 3)):
            got = d2l.bits_per_byte(Uniform(), tok, texts, ctx, stride)
            self.assertAlmostEqual(got, T * math.log2(V) / B, places=5)

    def test_without_boundary(self):
        V = CharTokenizer.V
        tok = CharTokenizer()  # neither <|endoftext|> nor an EOS
        texts = ['éclair', 'tea']
        got = d2l.bits_per_byte(Uniform(), tok, texts, 4, 2)
        T = sum(len(t) - 1 for t in texts)  # first tokens unscored
        B = (len('éclair'.encode()) - 2) + (3 - 1)  # and their bytes gone
        self.assertAlmostEqual(got, T * math.log2(V) / B, places=5)

    def test_boundary_resolution(self):
        both = CharTokenizer(endoftext=True, eos=True)
        self.assertEqual(d2l._lm_boundary_id(both), 256)
        self.assertEqual(d2l._lm_boundary_id(CharTokenizer(eos=True)), 257)
        self.assertEqual(d2l._lm_boundary_id(CharTokenizer(vocab=False,
                                                           eos=True)), 257)
        self.assertIsNone(d2l._lm_boundary_id(CharTokenizer()))
        model = Bigram()
        d2l.bits_per_byte(model, both, ['abc'], 8)
        self.assertEqual(model.inputs, [[[256, 97, 98]]])

        class Unk(CharTokenizer):  # maps unknown tokens to its UNK
            unk_token_id = 0

            def __init__(self):
                super().__init__(eos=True)
                self.convert_tokens_to_ids = lambda t: 0
        self.assertEqual(d2l._lm_boundary_id(Unk()), 257)

    def test_tiktoken_boundary(self):
        try:
            import tiktoken
        except ImportError:
            self.skipTest('tiktoken not installed')
        enc = tiktoken.Encoding(
            'toy', pat_str=r'.', mergeable_ranks={bytes([i]): i
                                                  for i in range(256)},
            special_tokens={'<|endoftext|>': 256})
        self.assertEqual(d2l._lm_boundary_id(enc), 256)

    def test_window_coverage(self):
        """A bigram sees only the previous token, which every window holds:
        any (ctx_len, stride) gives the single-pass value iff every token
        is scored exactly once."""
        tok = CharTokenizer(endoftext=True)
        model = Bigram()
        for T in (1, 2, 5, 10, 17):
            text = ''.join(chr(97 + i % 26) for i in range(T))
            ids = [256] + tok.encode(text)
            ref = -sum(F.log_softmax(model.W[a], -1)[b].item()
                       for a, b in zip(ids, ids[1:])) / (math.log(2) * T)
            for ctx in (1, 2, 3, 4, 8, 32):
                for stride in sorted({1, max(ctx // 2, 1), ctx}):
                    for bs in (1, 3):
                        model.inputs = []
                        got = d2l.bits_per_byte(model, tok, text, ctx,
                                                stride, batch_size=bs)
                        self.assertAlmostEqual(got, ref, places=5,
                                               msg=(T, ctx, stride))
                        self.assertTrue(all(len(x) <= ctx for X in
                                            model.inputs for x in X))
        with self.assertRaises(ValueError):
            d2l.bits_per_byte(model, tok, 'abc', 4, 5)
        with self.assertRaises(ValueError):
            d2l.bits_per_byte(model, tok, ['', ''], 4)

    def test_worked_example(self):
        """Ten tokens after the boundary, ctx_len 4, stride 2: windows
        start at positions 0, 2, 4, 6 and score 4, 2, 2, 2 tokens."""
        tok = CharTokenizer(endoftext=True)
        model = Bigram()
        d2l.bits_per_byte(model, tok, 'abcdefghij', 4, 2)
        seq = [256] + tok.encode('abcdefghij')
        starts = [seq.index(X[0][0]) for X in model.inputs]
        self.assertEqual(starts, [0, 2, 4, 6])
        self.assertEqual([len(X[0]) for X in model.inputs], [4] * 4)

    def test_chunked_head(self):
        """Decoder states and the head in 1,024-position chunks, batched
        and right-padded, equal full logits row by row."""
        tok = qwen_tokenizer()
        model = tiny_qwen3()
        words = ('the quick brown fox jumps over the lazy dog while seven '
                 'wizards quietly judge boxing, ') * 70
        texts = [words, 'Café crème, s’il vous plaît.', 'x = [i ** 2 for i in '
                 'range(10)]\nprint(sum(x))\n']
        b = tok.convert_tokens_to_ids('<|endoftext|>')
        self.assertEqual(b, 151643)
        nll, B = 0.0, 0
        for t in texts:
            ids = [b] + tok(t, add_special_tokens=False)['input_ids']
            with torch.no_grad():
                z = model(torch.tensor([ids]), use_cache=False).logits[0]
            nll += F.cross_entropy(z[:-1].float(), torch.tensor(ids[1:]),
                                   reduction='sum').item()
            B += len(t.encode())
        self.assertTrue(1100 < len(tok(words)['input_ids']) < 2047)
        ref = nll / (math.log(2) * B)
        for bs in (1, 4):
            got = d2l.bits_per_byte(model, tok, texts, ctx_len=2048,
                                    stride=1024, batch_size=bs)
            self.assertAlmostEqual(got, ref, delta=1e-5 * ref)
        # the windowed value differs (less context), the hybrid model runs
        self.assertNotAlmostEqual(d2l.bits_per_byte(
            model, tok, texts, ctx_len=256, stride=128), ref, places=4)
        hybrid = tiny_qwen35()
        one = d2l.bits_per_byte(hybrid, tok, texts[1:], 16, 8, batch_size=1)
        self.assertAlmostEqual(d2l.bits_per_byte(
            hybrid, tok, texts[1:], 16, 8, batch_size=4), one, delta=1e-5)

    def test_softcapped_and_wrapped_models_run_one_window(self):
        tok = qwen_tokenizer()
        model = tiny_qwen3()
        texts = ['A short text.', 'Another one, a little longer than that.']
        ref = d2l.bits_per_byte(model, tok, texts, 8, 4)
        calls = []
        h = model.register_forward_pre_hook(
            lambda m, args, kw: calls.append(kw), with_kwargs=True)
        try:  # a non-linear head: whole-model calls, one window each
            model.config.final_logit_softcapping = 30.0
            got = d2l.bits_per_byte(model, tok, texts, 8, 4, batch_size=4)
        finally:
            h.remove()
            del model.config.final_logit_softcapping
        self.assertAlmostEqual(got, ref, delta=1e-5)  # Qwen3 ignores it
        self.assertTrue(calls)
        self.assertTrue(all(kw['input_ids'].shape[0] == 1 for kw in calls))


# --------------------------------------------------------------- pass@k
class TestPassAtK(unittest.TestCase):
    def test_values(self):
        self.assertEqual(d2l.pass_at_k(4, 1, 2), 0.5)
        self.assertAlmostEqual(d2l.pass_at_k(200, 37, 10), 0.877375,
                               delta=1e-6)
        self.assertAlmostEqual(d2l.pass_at_k(5, 2, 1), 0.4, delta=1e-12)
        self.assertIsInstance(d2l.pass_at_k(4, 1, 2), float)
        for n in range(1, 13):
            for k in range(1, n + 1):
                self.assertEqual(d2l.pass_at_k(n, 0, k), 0.0)
                for c in range(n - k + 1, n + 1):
                    self.assertEqual(d2l.pass_at_k(n, c, k), 1.0)

    def test_closed_form_and_plugin_bound(self):
        for n in range(1, 17):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    u = d2l.pass_at_k(n, c, k)
                    exact = 1 - math.comb(n - c, k) / math.comb(n, k)
                    self.assertAlmostEqual(u, exact, delta=1e-12)
                    plug = 1 - (1 - c / n) ** k
                    self.assertLessEqual(plug, u + 1e-12)
                    if k == 1 or c in (0, n):
                        self.assertAlmostEqual(plug, u, delta=1e-12)

    def test_broadcasting(self):
        n = np.array([[8], [16]])
        c = np.array([0, 1, 4, 8])
        out = d2l.pass_at_k(n, c, 4)
        self.assertEqual(out.shape, (2, 4))
        for i in range(2):
            for j in range(4):
                self.assertEqual(out[i, j], d2l.pass_at_k(
                    int(n[i, 0]), int(c[j]), 4))
        out = d2l.pass_at_k(8, np.arange(9), np.array([1, 2, 4, 8])[:, None])
        self.assertEqual(out.shape, (4, 9))

    def test_errors(self):
        for args in ((4, -1, 2), (4, 5, 2), (4, 1, 0), (4, 1, 5)):
            with self.assertRaises(ValueError):
                d2l.pass_at_k(*args)
        with self.assertRaises(ValueError):
            d2l.pass_at_k(np.array([4, 4]), np.array([1, 5]), 2)
        with self.assertRaises(TypeError):
            d2l.pass_at_k(4.0, 1, 2)


class TestPassAtKVariance(unittest.TestCase):
    def test_special_cases(self):
        for n in (1, 3, 8, 20):
            for p in (0.05, 0.3, 0.5, 0.9):
                self.assertAlmostEqual(d2l.pass_at_k_variance(n, p, 1),
                                       p * (1 - p) / n, delta=1e-12)
                q = (1 - p) ** n
                self.assertAlmostEqual(d2l.pass_at_k_variance(n, p, n),
                                       q * (1 - q), delta=1e-12)
        self.assertEqual(d2l.pass_at_k_variance(10, 0.0, 3), 0.0)
        self.assertEqual(d2l.pass_at_k_variance(10, 1.0, 3), 0.0)
        for args in ((10, -0.1, 1), (10, 1.1, 1), (10, 0.5, 0),
                     (10, 0.5, 11)):
            with self.assertRaises(ValueError):
                d2l.pass_at_k_variance(*args)

    def test_numeric_case(self):
        var = d2l.pass_at_k_variance(20, 0.3, 5)
        self.assertAlmostEqual(var, 0.018063, delta=5e-7)
        self.assertAlmostEqual(math.sqrt(var), 0.1344, delta=5e-5)
        self.assertGreater(d2l.pass_at_k_variance(2000, 0.3, 5), 0)

    def test_monte_carlo(self):
        n, p, k, draws = 20, 0.3, 5, 200_000
        rng = np.random.default_rng(0)
        c = rng.binomial(n, p, size=draws)
        table = d2l.pass_at_k(n, np.arange(n + 1), k)
        U = table[c]
        mean, var = 1 - (1 - p) ** k, d2l.pass_at_k_variance(n, p, k)
        pmf = np.array([math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                        for i in range(n + 1)])
        m4 = np.sum(pmf * (table - mean) ** 4)
        self.assertLess(abs(U.mean() - mean), 3 * math.sqrt(var / draws))
        self.assertLess(abs(U.var() - var),
                        3 * math.sqrt((m4 - var ** 2) / draws))


# --------------------------------------------------------- extract_answer
EXTRACT_CASES = [  # the twelve strings of cell scoring-extract-check
    ('so she has 18 left.\n#### 18', 'gsm8k', '18'),
    ('#### 1,234', 'gsm8k', '1234'),
    ('The total is $1,000.', 'gsm8k', '1000'),
    ('The temperature drops to -7 degrees.\n#### -7', 'gsm8k', '-7'),
    ('He pays 3.50 each, 2 times: 7.00.', 'gsm8k', '7'),
    ('The answer is 42.', 'gsm8k', '42'),
    ('the answer is 12, not 15 as first thought', 'gsm8k', '12'),
    ('First 5 apples, then 3 more, so 8 apples in all', 'gsm8k', '8'),
    ('I cannot tell from the question.', 'gsm8k', None),
    (r'So \boxed{B}, or perhaps \boxed{C}.', 'boxed', 'C'),
    (r'\boxed{ \frac{1}{2} }', 'boxed', r'\frac{1}{2}'),
    (r'The letter is B.', 'boxed', None)]


class TestExtractAnswer(unittest.TestCase):
    def test_cases(self):
        for text, kind, want in EXTRACT_CASES:
            self.assertEqual(d2l.extract_answer(text, kind), want, text)

    def test_cases_match_the_notebook(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'chapter_lm-pretrained', 'scoring.md')
        with open(path) as f:
            src = f.read()
        cell = src.split('#scoring-extract-check}')[1].split('```')[0]
        cases = eval(cell.split('cases = ')[1].split('\nfor ')[0])
        self.assertEqual(cases, EXTRACT_CASES)

    def test_strict(self):
        for text, kind, want in EXTRACT_CASES:
            if kind != 'gsm8k':
                continue
            fallback = '####' not in text and 'answer is' not in text.lower()
            self.assertEqual(d2l.extract_answer(text, strict=True),
                             None if fallback else want, text)

    def test_more(self):
        ex = d2l.extract_answer
        self.assertEqual(ex('#### 0.50'), '0.5')
        self.assertEqual(ex('#### -0'), '0')
        self.assertEqual(ex('#### $-'), None)  # no number after the marker
        self.assertEqual(ex('7 then\n#### none', strict=True), None)
        self.assertEqual(ex('#### 3\nThe answer is 4'), '3')  # '####' first
        self.assertEqual(ex('The Answer Is: $5'), '5')
        self.assertEqual(ex(r'\boxed{\boxed{3}}', 'boxed'), '3')
        self.assertEqual(ex(r'\boxed{A} and \boxed{B', 'boxed'), 'A')
        self.assertIsNone(ex(r'\boxed{ }', 'boxed'))
        with self.assertRaises(ValueError):
            ex('42', 'math')


# ------------------------------------------------------- run_python_tests
class TestRunPythonTests(unittest.TestCase):
    ADD = 'def add(a, b):\n    return a + b'
    TESTS = 'assert add(2, 3) == 5\nassert add(-1, 1) == 0'

    def test_fixtures(self):
        timeout = 2.0
        for program, want in ((self.ADD, True),
                              ('def add(a, b):\n    return a - b', False),
                              ('def add(a, b):\n    while True:\n'
                               '        pass', False),
                              ('def add(a, b)\n    return a + b', False)):
            tic = time.time()
            self.assertEqual(d2l.run_python_tests(program, self.TESTS,
                                                  timeout=timeout), want)
            self.assertLess(time.time() - tic, timeout + 1)

    def test_isolation(self):
        # stdin is closed, the directory is new and empty, -I is set
        self.assertFalse(d2l.run_python_tests('input()', ''))
        self.assertTrue(d2l.run_python_tests(
            'import os, sys', 'assert os.listdir() == ["main.py"]\n'
            'assert sys.flags.isolated'))
        self.assertTrue(d2l.run_python_tests(
            '', 'raise SystemExit(__name__ != "__main__")'))

    @unittest.skipUnless(os.name == 'posix', 'resource limits are POSIX')
    def test_memory_limit(self):
        self.assertFalse(d2l.run_python_tests(
            'x = bytearray(2 << 30)', '', timeout=10))

    def test_child_processes_killed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'pid')
            tic = time.time()
            self.assertFalse(d2l.run_python_tests(
                'import subprocess, sys\n'
                'p = subprocess.Popen([sys.executable, "-c", "import time; '
                'time.sleep(30)"])\n'
                f'open({path!r}, "w").write(str(p.pid))\n'
                'p.wait()', '', timeout=1.0))
            self.assertLess(time.time() - tic, 2.0)
            with open(path) as f:
                pid = int(f.read())
        if os.name != 'posix':
            return
        for _ in range(50):  # the grandchild is gone (or a zombie)
            try:
                with open(f'/proc/{pid}/stat') as f:
                    if f.read().split(')')[-1].split()[0] == 'Z':
                        break
            except FileNotFoundError:
                break
            time.sleep(0.05)
        else:
            os.kill(pid, 9)
            self.fail('the program\'s child survived the timeout')


# ----------------------------------------------------------- plot_intervals
class TestPlotIntervals(unittest.TestCase):
    def test_smoke(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        series = {'a': ([0.4, 0.5, 0.6], [0.3, 0.4, 0.5], [0.5, 0.6, 0.7]),
                  'b': ([0.5, 0.5, np.nan], [0.4, 0.4, np.nan],
                        [0.6, 0.6, np.nan])}
        fig, ax = plt.subplots()
        out = d2l.plot_intervals(['x', 'y', 'z'], series, 'convention',
                                 'accuracy', reference=0.25, axes=ax)
        self.assertIs(out, ax)
        self.assertEqual(len(ax.containers), 2)  # two errorbar series
        fig, ax = plt.subplots()
        d2l.plot_intervals([1, 2, 4, 8], {'a': ([.2, .4, .6, .7],
                                                [.1, .3, .5, .6],
                                                [.3, .5, .7, .8])},
                           'k', 'pass@k', kind='band', xscale='log2',
                           axes=ax)
        self.assertEqual(len(ax.collections), 1)
        plt.close('all')
        with self.assertRaises(ValueError):
            d2l.plot_intervals([1], {'a': ([1], [0], [2])}, 'x', 'y',
                               kind='bars', axes=ax)


# ---------------------------------------------------- import without HF
class TestWithoutTransformers(unittest.TestCase):
    def test_import_and_score(self):
        code = (
            'import sys\n'
            'for m in ("transformers", "openai", "accelerate", '
            '"huggingface_hub", "tokenizers", "safetensors"):\n'
            '    sys.modules[m] = None\n'
            'sys.path.insert(0, sys.argv[1])\n'
            'import torch\n'
            'from d2l import torch as d2l\n'
            'from test_lm_scoring import Bigram, CharTokenizer\n'
            'tok = CharTokenizer(eos=True)\n'
            's = d2l.score_choices(Bigram(), tok, ["Q:"], [[" a", " bc"]])\n'
            'assert s.shape == (1, 2)\n'
            'b = d2l.bits_per_byte(Bigram(), tok, ["some text"], 4)\n'
            'assert b > 0\n'
            'assert d2l.pass_at_k(4, 1, 2) == 0.5\n'
            'assert d2l.extract_answer("#### 5") == "5"\n'
            'print("ok")\n')
        tools = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run([sys.executable, '-c', code, tools],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], 'ok')


if __name__ == '__main__':
    unittest.main()
