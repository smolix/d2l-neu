#!/usr/bin/env python3
"""CPU tests for sec_lm_prompting (chapter_lm-pretrained/prompting.md).

`test_chat_figure_tokens` (spec rule T1): the chat-template figure
(`fig_lm_chat_template`, drawn by tools/gen_mdl_lm_pretrained_figures.py)
writes its token strings and ids as literals and does not import
`transformers`.  This test renders the same conversation with the pinned
Qwen3-1.7B tokenizer and asserts that the ids and token strings equal the
figure's rows, that the four label words are single tokens after a newline,
and that the plain layout `Headline: ...\\nTopic:` is followed by ` Sports`
(13180), a different token from the chat layout's `Sports` (40979).

The tokenizer files (about 11 MB, no weights) come from the Hugging Face
cache, or are downloaded once at the pinned revision; the test is skipped when
neither works (offline host).  The library tests of 18.3 join this file in
build step 6.

Run: `.venv-lm/bin/python -m unittest tools/test_lm_prompting.py`, or
`make test-lm`.
"""

import os
import sys
import unittest

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


if __name__ == '__main__':
    unittest.main()
