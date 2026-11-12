#!/usr/bin/env python3
"""CPU tests for sec_lm_prompting (chapter_lm-pretrained/prompting.md).

Library code (spec 18.3 part 5):
- `fewshot_prompt`: the four layouts, `fmt`, `system` and `label_map`,
  against exact fixture strings, and every documented error.
- `calibrate_labels`: a uniform content-free row, the two-class worked
  case, and a simulation of `eq_lm_ctx_calibration` (Gaussian classes at the
  vertices of a regular simplex, a skewed prior, a content-free input that is
  computed, not set) including the residual bias of Exercise 3.
- `label_logprobs` (and its helper `_label_token_ids`): with a tiny random
  `Qwen3ForCausalLM` (2 layers, hidden size 64, fp32) and the real Qwen3
  tokenizer at the pinned revision, loaded with `local_files_only=True`;
  against the full-vocabulary softmax, batched against one at a time, the
  leading-space label ids, the three errors in their order, the
  Transformers call (`logits_to_keep=1`, `use_cache=False`, left padding),
  and the bucketed path of a tiny `d2l.Qwen3GPT` and `d2l.GPT`.  Skipped,
  with the caching command, when the tokenizer is not cached.

`test_chat_figure_tokens` (spec rule T1): the chat-template figure
(`fig_lm_chat_template`, drawn by tools/gen_mdl_lm_pretrained_figures.py)
writes its token strings and ids as literals and does not import
`transformers`.  This test renders the same conversation with the pinned
Qwen3-1.7B tokenizer and asserts that the ids and token strings equal the
figure's rows, that the four label words are single tokens after a newline,
and that the plain layout `Headline: ...\\nTopic:` is followed by ` Sports`
(13180), a different token from the chat layout's `Sports` (40979).  Its
tokenizer comes from the cache, or is downloaded once (about 11 MB, no
weights); it is skipped when neither works.

Run: `.venv-lm/bin/python -m unittest tools/test_lm_prompting.py`, or
`make test-lm`.
"""

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch
from d2l import torch as d2l

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import gen_mdl_lm_pretrained_figures as figs  # noqa: E402

MODEL = 'Qwen/Qwen3-1.7B'


def _tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise unittest.SkipTest(f'transformers not installed: {e}')
    for local_only in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                MODEL, revision=figs.CHAT_REVISION,
                local_files_only=local_only)
        except Exception as e:  # noqa: BLE001 (offline, cache miss)
            err = e
    raise unittest.SkipTest(f'{MODEL} tokenizer unavailable: {err}')


class ChatFigureTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _tokenizer()

    def test_chat_figure_tokens(self):
        tok = self.tok
        messages = [{'role': 'system', 'content': figs.CHAT_SYSTEM},
                    {'role': 'user', 'content': figs.CHAT_USER}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, add_special_tokens=False)['input_ids']
        rows = [t for row in figs.CHAT_ROWS for t in row]
        self.assertEqual(ids, [i for _, i in rows])
        self.assertEqual([tok.decode([i]) for i in ids], [s for s, _ in rows])
        # the generation prefix is the last row: 3 + 4 tokens
        self.assertEqual(len(figs.CHAT_ROWS[2]), 7)
        for word, tid in figs.CHAT_LABELS:
            self.assertEqual(tok.encode(word, add_special_tokens=False),
                             [tid], word)
        plain = f'{figs.CHAT_USER}\nTopic:'
        p_ids = tok.encode(plain, add_special_tokens=False)
        self.assertEqual(p_ids[-len(figs.PLAIN_TAIL):],
                         [i for _, i in figs.PLAIN_TAIL])
        word, tid = figs.PLAIN_LABEL
        self.assertEqual(tok.encode(plain + word, add_special_tokens=False),
                         p_ids + [tid])
        self.assertNotIn(tid, [i for _, i in figs.CHAT_LABELS])
        self.assertEqual(figs.SPECIAL_IDS,
                         set(tok.convert_tokens_to_ids(
                             ['<|im_start|>', '<|im_end|>', '<think>',
                              '</think>'])))


# ---------------------------------------------------------------- 18.3 lib

FMT = 'Headline: {x}\nTopic: {y}'
DEMOS = [('Rain delays the cup final', 'Sports'),
         ('Chip maker shares climb', 'Business'),
         ('New probe reaches the moon', 'Tech')]


class TestFewshotPrompt(unittest.TestCase):
    def test_plain_fmt(self):
        self.assertEqual(d2l.fewshot_prompt([], 'a', fmt=FMT),
                         'Headline: a\nTopic:')
        self.assertEqual(
            d2l.fewshot_prompt(DEMOS[:2], 'Local team wins', fmt=FMT),
            'Headline: Rain delays the cup final\nTopic: Sports\n\n'
            'Headline: Chip maker shares climb\nTopic: Business\n\n'
            'Headline: Local team wins\nTopic:')

    def test_default_plain_and_qa(self):
        self.assertEqual(
            d2l.fewshot_prompt(DEMOS[:1], 'q'),
            'Input: Rain delays the cup final\nOutput: Sports\n\n'
            'Input: q\nOutput:')
        self.assertEqual(
            d2l.fewshot_prompt(DEMOS[:2], 'q', 'qa'),
            'Q: Rain delays the cup final\nA: Sports\n\n'
            'Q: Chip maker shares climb\nA: Business\n\nQ: q\nA:')
        self.assertEqual(d2l.fewshot_prompt([], 'q', 'qa'), 'Q: q\nA:')

    def test_system_line(self):
        self.assertEqual(
            d2l.fewshot_prompt(DEMOS[:1], 'q', fmt=FMT, system='Name it.'),
            'Name it.\n\nHeadline: Rain delays the cup final\n'
            'Topic: Sports\n\nHeadline: q\nTopic:')
        self.assertEqual(d2l.fewshot_prompt([], 'q', 'qa', system='S'),
                         'S\n\nQ: q\nA:')

    def test_trailing_whitespace_and_braces(self):
        # only the query's trailing whitespace goes; braces in x stay
        self.assertEqual(
            d2l.fewshot_prompt([('a {y} b', 'c ')], '{x}',
                               fmt='{x} -> {y}  '),
            'a {y} b -> c   \n\n{x} ->')

    def test_chat_multi(self):
        msgs = d2l.fewshot_prompt(DEMOS, 'q', 'chat_multi', system='S')
        self.assertEqual(len(msgs), 8)
        self.assertEqual([m['role'] for m in msgs],
                         ['system'] + ['user', 'assistant'] * 3 + ['user'])
        self.assertEqual(msgs[0]['content'], 'S')
        self.assertEqual([m['content'] for m in msgs[1:7]],
                         [c for d in DEMOS for c in d])
        self.assertEqual(msgs[-1], {'role': 'user', 'content': 'q'})
        self.assertEqual(d2l.fewshot_prompt([], 'q', 'chat_multi'),
                         [{'role': 'user', 'content': 'q'}])
        # (question, answer) pairs pass as they are, newlines included
        pairs = [('2 + 2?\nThink.', 'It is 4.\n#### 4')]
        self.assertEqual(
            [m['content'] for m in d2l.fewshot_prompt(pairs, 'x',
                                                      'chat_multi')],
            ['2 + 2?\nThink.', 'It is 4.\n#### 4', 'x'])

    def test_chat_single(self):
        msgs = d2l.fewshot_prompt(DEMOS, 'q', 'chat_single', fmt=FMT)
        self.assertEqual(msgs, [{'role': 'user', 'content':
                                 d2l.fewshot_prompt(DEMOS, 'q', fmt=FMT)}])
        for x, y in DEMOS:
            self.assertIn(FMT.format(x=x, y=y), msgs[0]['content'])
        msgs = d2l.fewshot_prompt(DEMOS[:1], 'q', 'chat_single', system='S')
        self.assertEqual(msgs, [
            {'role': 'system', 'content': 'S'},
            {'role': 'user', 'content': 'Input: Rain delays the cup final'
             '\nOutput: Sports\n\nInput: q\nOutput:'}])

    def test_label_map(self):
        self.assertEqual(
            d2l.fewshot_prompt([('a', 0)], 'b', fmt=FMT,
                               label_map={0: 'foo'}),
            'Headline: a\nTopic: foo\n\nHeadline: b\nTopic:')
        msgs = d2l.fewshot_prompt([('a', 1)], 'b', 'chat_multi',
                                  label_map={1: 'bar'})
        self.assertEqual(msgs[1], {'role': 'assistant', 'content': 'bar'})
        with self.assertRaises(KeyError) as cm:
            d2l.fewshot_prompt([('a', 0), ('b', 3)], 'c',
                               label_map={0: 'foo'})
        self.assertIn('3', str(cm.exception))
        for template in ('plain', 'qa', 'chat_single', 'chat_multi'):
            with self.assertRaises(TypeError):
                d2l.fewshot_prompt([('a', 0)], 'b', template)

    def test_fmt_errors(self):
        for template in ('qa', 'chat_multi'):
            with self.assertRaises(ValueError):
                d2l.fewshot_prompt(DEMOS, 'q', template, fmt=FMT)
        for fmt in ('Headline: {x}', 'Topic: {y}', '{x} {y} {z}', '{0}'):
            with self.assertRaises(ValueError):
                d2l.fewshot_prompt(DEMOS, 'q', fmt=fmt)
        # regression (review 18.3, finding 3): the prompt must end after
        # the query's label field, so {y} comes once and last
        for fmt in ('Topic: {y}\nHeadline: {x}', '{x} -> {y}.',
                    '{y}: {x} -> {y}', '{x} {y}\nEnd'):
            with self.assertRaisesRegex(ValueError, 'once and last'):
                d2l.fewshot_prompt(DEMOS, 'q', fmt=fmt)
        for fmt in ('{x} -> {y}\n', '{x} {x} -> {y}', '{x}:{y!s}'):
            self.assertTrue(d2l.fewshot_prompt(DEMOS, 'q', fmt=fmt)
                            .endswith('\n\n' + fmt.format(x='q', y='')
                                      .rstrip()))
        with self.assertRaises(ValueError):
            d2l.fewshot_prompt(DEMOS, 'q', 'Headline: {x}\nTopic: {y}')


class TestCalibrateLabels(unittest.TestCase):
    def test_uniform_content_free(self):
        gen = torch.Generator().manual_seed(0)
        lp = torch.randn(50, 4, generator=gen).log_softmax(-1)
        cf = torch.full((3, 4), -math.log(4))
        torch.testing.assert_close(d2l.calibrate_labels(lp, cf), lp,
                                   atol=1e-6, rtol=0)
        # any constant row is uniform after renormalization
        torch.testing.assert_close(
            d2l.calibrate_labels(lp, torch.full((4,), 7.0)), lp,
            atol=1e-6, rtol=0)

    def test_two_classes(self):
        lp = torch.tensor([[0.6, 0.4]]).log()
        cal = d2l.calibrate_labels(lp, torch.tensor([[0.8, 0.2]]).log())
        torch.testing.assert_close(cal.exp(), torch.tensor([[3 / 11, 8 / 11]]),
                                   atol=1e-4, rtol=0)
        self.assertNotEqual(lp.argmax(-1).item(), cal.argmax(-1).item())

    def test_log_space_mean(self):
        # mirror-image rows: their log-space mean is uniform
        cf = torch.tensor([[0.9, 0.1], [0.1, 0.9]]).log()
        lp = torch.tensor([[0.3, 0.7], [0.55, 0.45]]).log()
        torch.testing.assert_close(d2l.calibrate_labels(lp, cf), lp,
                                   atol=1e-6, rtol=0)
        # the geometric mean (0.9 * 0.5, 0.1 * 0.5) ** 0.5 is 3 : 1, the
        # arithmetic mean (0.7, 0.3) is 7 : 3
        cf = torch.tensor([[0.9, 0.1], [0.5, 0.5]], dtype=torch.float64)
        cal = d2l.calibrate_labels(torch.zeros(1, 2, dtype=torch.float64),
                                   cf.log())
        torch.testing.assert_close(cal.exp(), torch.tensor(
            [[0.25, 0.75]], dtype=torch.float64), atol=1e-12, rtol=0)
        # NumPy arrays and a single content-free row are accepted
        out = d2l.calibrate_labels(lp.numpy(), cf[0].numpy())
        torch.testing.assert_close(
            out, (lp - cf[0]).log_softmax(-1), atol=1e-6, rtol=0)

    def test_shapes(self):
        """Regression (review 18.3, finding 1): content-free arrays over
        another label count, or transposed, were reshaped to (-1, L) and
        used without a word."""
        lp = torch.randn(5, 4).log_softmax(-1)
        for shape in ((4, 3), (4, 2), (3,), (8,), (2, 2, 4), ()):
            with self.assertRaisesRegex(ValueError, 'content-free'):
                d2l.calibrate_labels(lp, torch.zeros(shape))
        for shape in ((4,), (1, 4), (7, 4)):
            self.assertEqual(d2l.calibrate_labels(
                lp, torch.zeros(shape)).shape, (5, 4))

    @staticmethod
    def _simplex(scale=1.2):
        """Four equal-norm means at the vertices of a regular simplex in
        R^3, the Bayes log-posterior under a prior, and 20,000 points from
        a uniform class prior with identity covariance."""
        mu = scale * torch.tensor([[1., 1, 1], [1, -1, -1], [-1, 1, -1],
                                   [-1, -1, 1]], dtype=torch.float64)

        def log_q(x):  # log q(x|y) up to a constant, (N, 4)
            return -0.5 * ((x[:, None] - mu) ** 2).sum(-1)

        def posterior(x, prior):
            return (log_q(x) + prior.log()).log_softmax(-1)
        gen = torch.Generator().manual_seed(0)
        y = torch.randint(0, 4, (20000,), generator=gen)
        x = mu[y] + torch.randn(20000, 3, generator=gen, dtype=torch.float64)
        return mu, log_q, posterior, x, y

    def test_simulation(self):
        mu, log_q, posterior, x, y = self._simplex()
        skew = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float64)
        uniform = torch.full((4,), 0.25, dtype=torch.float64)
        p_m = posterior(x, skew)
        x0 = torch.zeros(1, 3, dtype=torch.float64)  # equidistant
        cal = d2l.calibrate_labels(p_m, posterior(x0, skew))
        torch.testing.assert_close(cal, posterior(x, uniform), atol=1e-5,
                                   rtol=0)
        acc = lambda lp: (lp.argmax(-1) == y).double().mean().item()
        self.assertGreater(acc(cal), acc(p_m) + 0.01)

    def test_residual_bias(self):
        mu, log_q, posterior, x, y = self._simplex()
        skew = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float64)
        uniform = torch.full((4,), 0.25, dtype=torch.float64)
        x0 = 0.5 * mu[:1]  # toward the first mean: q(x0|y) = c(y)
        cal = d2l.calibrate_labels(posterior(x, skew), posterior(x0, skew))
        resid = cal - posterior(x, uniform) + log_q(x0)  # (N, 4)
        torch.testing.assert_close(resid - resid[:, :1],
                                   torch.zeros_like(resid), atol=1e-5,
                                   rtol=0)
        # and the bias is real: without the -log q(x0|y) term it is not
        diff = cal - posterior(x, uniform)
        self.assertGreater((diff - diff[:, :1]).abs().max().item(), 0.1)


LABELS = ['World', 'Sports', 'Business', 'Tech']
PLAIN_IDS = [4337, 13180, 8025, 17374]  # ' World', ' Sports', ...
HEADLINES = ['Local team wins cup final', 'Chip maker shares climb',
             'New probe reaches the moon after a long and quiet trip',
             'Leaders meet for talks']
CACHE_CMD = ('.venv-lm/bin/python -c "from transformers import '
             "AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B',"
             " revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')\"")


def tiny_qwen3(seed=0, attn='sdpa'):
    """A random Qwen3ForCausalLM over the real 151,936-row vocabulary:
    2 layers, hidden size 64, fp32, eval mode."""
    import transformers
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_hidden_layers=2, tie_word_embeddings=True,
        attn_implementation=attn)
    model = transformers.Qwen3ForCausalLM(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(1 + 0.3 * torch.randn_like(p) if 'norm' in name
                    else 0.08 * torch.randn_like(p))
    return model.eval()


class TestLabelLogprobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise unittest.SkipTest(f'transformers not installed: {e}')
        try:
            cls.tok = AutoTokenizer.from_pretrained(
                MODEL, revision=d2l.HF_REVISIONS[MODEL],
                local_files_only=True)
        except OSError:
            raise unittest.SkipTest(
                f'{MODEL} tokenizer not in the HF cache; cache it with '
                f'{CACHE_CMD}')
        cls.model = tiny_qwen3()
        cls.prompts = [f'Headline: {h}\nTopic:' for h in HEADLINES]

    def reference(self, model, prompt, label_ids):
        """Full-vocabulary log-softmax at the last position, gathered at
        the label ids and renormalized; one prompt, no padding."""
        X = torch.tensor([self.tok(prompt, add_special_tokens=False)
                          ['input_ids']])
        with torch.no_grad():
            z = model(X, use_cache=False).logits[0, -1] if hasattr(
                model, 'config') else model(X)[0, -1]
        lp = z.float().log_softmax(-1)[label_ids]
        return lp - lp.logsumexp(-1)

    def test_against_full_softmax(self):
        out = d2l.label_logprobs(self.model, self.tok, self.prompts, LABELS,
                                 batch_size=1)
        self.assertEqual(out.shape, (4, 4))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device.type, 'cpu')
        for row, p in zip(out, self.prompts):
            torch.testing.assert_close(row, self.reference(
                self.model, p, PLAIN_IDS), atol=1e-5, rtol=0)
        torch.testing.assert_close(out.exp().sum(-1), torch.ones(4))
        # the label distributions are not all the same
        self.assertGreater((out - out[:1]).abs().max().item(), 1e-2)

    def test_batched_left_padding(self):
        lengths = [len(self.tok(p)['input_ids']) for p in self.prompts]
        self.assertGreater(len(set(lengths)), 2)  # padding is exercised
        for attn in ('sdpa', 'eager'):
            model = self.model if attn == 'sdpa' else tiny_qwen3(attn=attn)
            one = d2l.label_logprobs(model, self.tok, self.prompts, LABELS,
                                     batch_size=1)
            for bs in (3, 4, 8):
                torch.testing.assert_close(
                    d2l.label_logprobs(model, self.tok, self.prompts,
                                       LABELS, batch_size=bs),
                    one, atol=1e-4, rtol=0, msg=f'{attn}, batch {bs}')

    def test_transformers_call(self):
        tok_side = self.tok.padding_side
        calls = []
        orig = self.model.forward

        def spy(*args, **kwargs):
            calls.append(kwargs)
            out = orig(*args, **kwargs)
            self.assertEqual(out.logits.shape[1], 1)  # one position kept
            return out
        with mock.patch.object(self.model, 'forward', side_effect=spy):
            d2l.label_logprobs(self.model, self.tok, self.prompts, LABELS,
                               batch_size=4)
        self.assertEqual(len(calls), 1)
        kw = calls[0]
        self.assertEqual(kw['logits_to_keep'], 1)
        self.assertIs(kw['use_cache'], False)
        mask, X = kw['attention_mask'], kw['input_ids']
        ids = self.tok(self.prompts, add_special_tokens=False)['input_ids']
        for r, m, x, pos in zip(ids, mask, X, kw['position_ids']):
            pad = len(X[0]) - len(r)  # padded on the left
            self.assertEqual(m.tolist(), [0] * pad + [1] * len(r))
            self.assertEqual(x[pad:].tolist(), r)
            self.assertEqual(pos[pad:].tolist(), list(range(len(r))))
        self.assertEqual(self.tok.padding_side, tok_side)

    def test_special_tokens(self):
        """A plain prompt is encoded as d2l.generate encodes it (with the
        tokenizer's BOS, if it adds one); a rendered chat prompt is not."""
        tok = self.tok

        class BosTokenizer:  # Qwen3's tokenizer, with a BOS of 151643
            def __call__(self, texts, add_special_tokens=True):
                ids = tok(texts, add_special_tokens=False)['input_ids']
                return {'input_ids': [[151643] * add_special_tokens + r
                                      for r in ids]}

            def __getattr__(self, name):
                return getattr(tok, name)
        seen = []
        orig = self.model.forward
        with mock.patch.object(self.model, 'forward', side_effect=lambda
                               *a, **kw: (seen.append(kw['input_ids']),
                                          orig(*a, **kw))[1]):
            out = d2l.label_logprobs(self.model, BosTokenizer(),
                                     self.prompts[:1], LABELS)
            d2l.label_logprobs(self.model, BosTokenizer(), HEADLINES[:1],
                               LABELS, chat=True)
        plain = tok(self.prompts[0], add_special_tokens=False)['input_ids']
        self.assertEqual(seen[0][0].tolist(), [151643] + plain)
        self.assertNotEqual(seen[1][0, 0].item(), 151643)
        X = torch.tensor([[151643] + plain])
        with torch.no_grad():
            lp = self.model(X).logits[0, -1].log_softmax(-1)[PLAIN_IDS]
        torch.testing.assert_close(out[0], lp - lp.logsumexp(-1),
                                   atol=1e-5, rtol=0)

    def test_inference_mode(self):
        calls = []
        orig = self.model.forward

        def spy(*args, **kwargs):
            calls.append((torch.is_inference_mode_enabled(),
                          torch.is_grad_enabled()))
            return orig(*args, **kwargs)
        with mock.patch.object(self.model, 'forward', side_effect=spy):
            out = d2l.label_logprobs(self.model, self.tok, self.prompts,
                                     LABELS)
        self.assertEqual(calls, [(True, False)])
        self.assertFalse(out.requires_grad)
        self.assertFalse(out.is_inference())  # the caller may write to it
        out[0] += 1

    def test_label_ids(self):
        ids, labels = d2l._label_token_ids(self.tok, self.prompts, LABELS)
        self.assertEqual(labels, [PLAIN_IDS] * 4)
        self.assertEqual(ids[0], self.tok(self.prompts[0])['input_ids'])
        self.assertEqual(labels[0][1], 13180)
        chat = self.tok.apply_chat_template(
            [{'role': 'user', 'content': self.prompts[0]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        _, labels = d2l._label_token_ids(self.tok, [chat], LABELS)
        self.assertEqual(labels[0][1], 40979)
        self.assertEqual(labels[0], self.tok.convert_tokens_to_ids(LABELS))
        # a prompt ending in a newline also takes the word without space
        _, labels = d2l._label_token_ids(self.tok, ['Topic?\n'], ['Sports'])
        self.assertEqual(labels, [[40979]])
        # the blueprint's Tech/Technology pair does not collide
        _, labels = d2l._label_token_ids(self.tok, ['Topic:'],
                                         ['Tech', 'Technology'])
        self.assertEqual(labels, [[17374, 11791]])

    def test_chat(self):
        msgs = [d2l.fewshot_prompt(DEMOS[:2], h, 'chat_multi', system='S')
                for h in HEADLINES]
        for thinking in (False, None, True):
            opts = {} if thinking is None else {'enable_thinking': thinking}
            rendered = [self.tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, **opts)
                        for m in msgs]
            out = d2l.label_logprobs(self.model, self.tok, msgs, LABELS,
                                     chat=True, enable_thinking=thinking,
                                     batch_size=4)
            torch.testing.assert_close(out, d2l.label_logprobs(
                self.model, self.tok, rendered, LABELS, batch_size=1),
                                       atol=1e-4, rtol=0)
            chat_ids = self.tok.convert_tokens_to_ids(LABELS)
            torch.testing.assert_close(out[0], self.reference(
                self.model, rendered[0], chat_ids), atol=1e-4, rtol=0)
        self.assertTrue(rendered[0].endswith('assistant\n'))  # thinking on
        # the default is enable_thinking=False; a string is one user turn
        out = d2l.label_logprobs(self.model, self.tok, HEADLINES[:1],
                                 LABELS, chat=True)
        rendered = self.tok.apply_chat_template(
            [{'role': 'user', 'content': HEADLINES[0]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(rendered.endswith('</think>\n\n'))
        torch.testing.assert_close(out, d2l.label_logprobs(
            self.model, self.tok, [rendered], LABELS), atol=1e-6, rtol=0)

    def test_errors(self):
        f = lambda prompts, words: d2l.label_logprobs(
            self.model, self.tok, prompts, words)
        with self.assertRaisesRegex(ValueError, r"'World' and 'World'"):
            f(self.prompts, ['World', 'World', 'Business', 'Tech'])
        with self.assertRaisesRegex(
                ValueError, r"'Sci/Tech' and 'Sci-Fi' .*' Sci' \(40372\)"):
            f(self.prompts, ['Sci/Tech', 'Sci-Fi', 'Business', 'World'])
        with self.assertRaisesRegex(ValueError,
                                    r"'Sci/Tech' after ' ' is 3 tokens"):
            f(self.prompts, ['Sci/Tech', 'Sports', 'Business', 'World'])
        # (a) before (b) before (c)
        for words in (LABELS, ['Sci/Tech', 'Sci-Fi'], ['Sci/Tech']):
            with self.assertRaisesRegex(ValueError,
                                        'strip trailing whitespace'):
                f(self.prompts[:1] + ['Headline: x\nTopic: '], words)
        with self.assertRaisesRegex(ValueError, 'share the first token'):
            f(self.prompts, ['Sci/Tech', 'Sports', 'Sci-Fi'])
        with self.assertRaisesRegex(ValueError, 'empty prompt'):
            f([''], LABELS)
        with self.assertRaisesRegex(ValueError, 'adds no token'):
            f(['Topic:\n'], ['', 'Sports'])

    def test_chinese_labels(self):
        """Regression (review 18.3, finding 2): a space was forced before
        every word, so ' 体育' became [' ', '体育'] after a Chinese prompt
        and raised; next to a wide character no space is added."""
        tok, words = self.tok, ['体育', '科技', '财经', '国际']
        prompts = ['新闻：火箭发射成功\n主题：', '新闻：股市上涨\n主题：',
                   'Headline: rocket launch\nTopic:']
        self.assertEqual([d2l._label_sep(p, w) for p, w in [
            (prompts[0], 'Sports'), (prompts[2], '体育'),
            (prompts[2], 'Sports'), ('Topic:\n', 'Sports'),
            ('Topic:', '')]], ['', '', ' ', '', ' '])
        ids, labels = d2l._label_token_ids(tok, prompts, words)
        for p, h, row in zip(prompts, ids, labels):
            self.assertEqual(h, tok(p, add_special_tokens=False)['input_ids'])
            self.assertEqual(row, [tok(p + w, add_special_tokens=False)[
                'input_ids'][len(h)] for w in words])
        self.assertEqual(labels[0], tok.convert_tokens_to_ids(
            [tok.convert_ids_to_tokens(i) for i in labels[0]]))
        self.assertEqual(tok.decode(labels[0][:1]), '体育')
        out = d2l.label_logprobs(self.model, tok, prompts, words,
                                 batch_size=2)
        for row, p, lab in zip(out, prompts, labels):
            torch.testing.assert_close(row, self.reference(
                self.model, p, lab), atol=1e-5, rtol=0)
        # an explicit sep overrides the rule; the error names it
        _, sp = d2l._label_token_ids(tok, ['Topic:'], ['Sports'], sep='')
        self.assertEqual(sp, [[tok.convert_tokens_to_ids('Sports')]])
        with self.assertRaisesRegex(ValueError, r"'体育' after ' ' is 2 "):
            d2l.label_logprobs(self.model, tok, prompts[:1], words, sep=' ')

    def test_tokenizer_appends_eos(self):
        """Regression (review 18.3, finding 4): the plain path re-encoded
        with special tokens and so read the label after an appended EOS.
        A tokenizer adding <BOS> ... <EOS> now feeds <BOS> ... only."""
        tok, eos = self.tok, self.tok.eos_token_id

        class BosEosTokenizer:
            def __call__(self, texts, add_special_tokens=True):
                ids = tok(texts, add_special_tokens=False)['input_ids']
                return {'input_ids': [[151643] * add_special_tokens + r +
                                      [eos] * add_special_tokens
                                      for r in ids]}

            def __getattr__(self, name):
                return getattr(tok, name)
        self.assertEqual(d2l._lm_bos_ids(BosEosTokenizer()), [151643])
        self.assertEqual(d2l._lm_bos_ids(tok), [])
        seen = []
        orig = self.model.forward
        with mock.patch.object(self.model, 'forward', side_effect=lambda
                               *a, **kw: (seen.append(kw['input_ids']),
                                          orig(*a, **kw))[1]):
            out = d2l.label_logprobs(self.model, BosEosTokenizer(),
                                     self.prompts[:1], LABELS)
        plain = tok(self.prompts[0], add_special_tokens=False)['input_ids']
        self.assertEqual(seen[0][0].tolist(), [151643] + plain)
        X = torch.tensor([[151643] + plain])
        with torch.no_grad():
            lp = self.model(X).logits[0, -1].log_softmax(-1)[PLAIN_IDS]
        torch.testing.assert_close(out[0], lp - lp.logsumexp(-1),
                                   atol=1e-5, rtol=0)

    def test_wrapped_models(self):
        """Regression (review 18.3, finding 5): a torch.compile or PEFT
        wrapper around a Transformers model fell into the d2l branch and
        raised a TypeError on the output object."""
        class PeftLike(torch.nn.Module):  # PEFT's interface, in brief
            def __init__(self, base):
                super().__init__()
                self.base_model = base

            def get_base_model(self):
                return self.base_model

            def forward(self, *args, **kwargs):
                return self.base_model(*args, **kwargs)
        ref = d2l.label_logprobs(self.model, self.tok, self.prompts, LABELS,
                                 batch_size=4)
        compiled = torch.compile(self.model, backend='eager')
        for wrapped in (compiled, PeftLike(self.model), PeftLike(compiled)):
            self.assertIs(d2l._lm_unwrap(wrapped), self.model)
            torch.testing.assert_close(d2l.label_logprobs(
                wrapped, self.tok, self.prompts, LABELS, batch_size=4),
                                       ref, atol=1e-5, rtol=0)
        torch.manual_seed(0)
        qwen = d2l.Qwen3GPT(151936, 64, 4, 2, 2, head_dim=16).eval()
        calls = []
        orig = qwen.forward
        with mock.patch.object(qwen, 'forward', side_effect=lambda X, **kw: (
                calls.append(kw), orig(X, **kw))[1]):
            d2l.label_logprobs(PeftLike(qwen), self.tok, self.prompts[:1],
                               LABELS)
        self.assertEqual(calls, [{'logits_to_keep': 1}])

    def test_d2l_models(self):
        prompts = self.prompts + [p.replace('Topic', 'Label')
                                  for p in self.prompts]
        lengths = [len(self.tok(p)['input_ids']) for p in prompts]
        self.assertLess(len(set(lengths)), len(prompts))  # shared buckets
        torch.manual_seed(0)
        qwen = d2l.Qwen3GPT(151936, 64, 4, 2, 2, head_dim=16).eval()
        gpt = d2l.GPT(151936, num_hiddens=64, num_heads=4, num_blks=2).eval()
        for model in (qwen, gpt):
            one = d2l.label_logprobs(model, self.tok, prompts, LABELS,
                                     batch_size=1)
            for bs in (2, 8):
                torch.testing.assert_close(
                    d2l.label_logprobs(model, self.tok, prompts, LABELS,
                                       batch_size=bs),
                    one, atol=1e-5, rtol=0)
            for row, p in zip(one, prompts):
                torch.testing.assert_close(row, self.reference(
                    model, p, PLAIN_IDS), atol=1e-5, rtol=0)
        # Qwen3GPT gets logits_to_keep=1
        calls = []
        orig = qwen.forward
        with mock.patch.object(qwen, 'forward', side_effect=lambda X, **kw: (
                calls.append(kw), orig(X, **kw))[1]):
            d2l.label_logprobs(qwen, self.tok, prompts, LABELS)
        self.assertTrue(calls)
        self.assertTrue(all(kw == {'logits_to_keep': 1} for kw in calls))


if __name__ == '__main__':
    unittest.main()
