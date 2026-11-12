#!/usr/bin/env python3
"""Literal data of the chapter-18 figures (tools/gen_mdl_lm_pretrained_figures.py).

The generator writes token splits as literals (it imports no `transformers`).
These tests re-tokenize with the pinned tokenizers:

* `fig_lm_scoring_conventions`: the cloze continuations of the invented item
  under Qwen3-0.6B-Base (one token for oxygen and nitrogen, two for carbon
  dioxide and water vapor) and the single-token letters ` A` to ` D`, as cell
  `scoring-format-print` prints them;
* `fig_lm_patching`: the Qwen3-0.6B tokens of the first recall pair (Eiffel
  Tower / Colosseum), equal length, subject last token at the same position.

The chat-template figure is checked by `test_lm_prompting.py`
(`test_chat_figure_tokens`).  The layer types and Qwen3-0.6B shapes are checked
against the committed configs by the generator itself (`check_data`).
Tokenizer files come from the Hugging Face cache or are downloaded once at the
pinned revision; the tests skip on an offline host.

Run: `.venv-lm/bin/python -m unittest tools/test_lm_figures.py`, or
`make test-lm`.
"""

import os
import sys
import unittest

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import gen_mdl_lm_pretrained_figures as figs  # noqa: E402

# Pins of d2l.HF_REVISIONS (chapter_lm-pretrained/hf-models.md).
REVISIONS = {
    'Qwen/Qwen3-0.6B': 'c1899de289a04d12100db370d81485cdf75e47ca',
    'Qwen/Qwen3-0.6B-Base': 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
}


def _tokenizer(model):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise unittest.SkipTest(f'transformers not installed: {e}')
    for local_only in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                model, revision=REVISIONS[model], local_files_only=local_only)
        except Exception as e:  # noqa: BLE001 (offline, cache miss)
            err = e
    raise unittest.SkipTest(f'{model} tokenizer unavailable: {err}')


class FigureTokens(unittest.TestCase):
    def test_revisions_match_registry(self):
        src = os.path.join(os.path.dirname(TOOLS), 'chapter_lm-pretrained',
                           'hf-models.md')
        with open(src) as fh:
            text = fh.read()
        for model, rev in REVISIONS.items():
            self.assertIn(f"'{model}': '{rev}'", text)

    def test_generator_data(self):
        figs.check_data()

    def test_scoring_figure_tokens(self):
        tok = _tokenizer('Qwen/Qwen3-0.6B-Base')
        prompt = f'Question: {figs.SCORE_QUESTION}\nAnswer:'
        for option, pieces in zip(figs.SCORE_OPTIONS, figs.SCORE_CLOZE):
            ids = tok.encode(' ' + option, add_special_tokens=False)
            self.assertEqual([tok.decode([i]) for i in ids], pieces)
            # no merge across the prompt/continuation boundary
            both = tok.encode(prompt + ' ' + option, add_special_tokens=False)
            self.assertEqual(both[-len(ids):], ids)
        for letter in 'ABCD':
            self.assertEqual(len(tok.encode(' ' + letter,
                                            add_special_tokens=False)), 1)

    def test_patching_figure_tokens(self):
        tok = _tokenizer('Qwen/Qwen3-0.6B')
        tmpl = 'The {} is located in the city of'
        for name, pieces in (('Eiffel Tower', figs.PATCH_CLEAN),
                             ('Colosseum', figs.PATCH_CORRUPT)):
            ids = tok.encode(tmpl.format(name), add_special_tokens=False)
            self.assertEqual([tok.decode([i]) for i in ids], pieces)
        k = figs.PATCH_SUBJECT_LAST
        self.assertEqual(figs.PATCH_CLEAN[k + 1], ' is')
        self.assertEqual(figs.PATCH_CORRUPT[k + 1], ' is')


if __name__ == '__main__':
    unittest.main()
