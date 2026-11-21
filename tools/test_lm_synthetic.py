#!/usr/bin/env python3
"""CPU tests for sec_lm_synthetic_data (chapter_lm-data/synthetic-data.md):
chunk_doc (the chunks concatenate to the document, none is longer than the
bound, every chunk that is not a cut of an over-long sentence ends at a
sentence end or a line break); synth_metrics on golden cases; distinct_n of
a repeated text; collapse_sim against the closed forms of the Gaussian
refit (replace and accumulate, 3 Monte-Carlo standard errors); and
rephrase_docs on a scripted model (input order, stop and length finishes,
dropped rows, the thinking-text assertion, one seed per batch). Gated on the
Hugging Face cache: Qwen3.5-0.8B (CPU, fp32) gives the same greedy rewrites
in batches of one and of eight."""
import glob
import os
import re
import tempfile
import unittest

import numpy as np
import torch

from d2l import torch as d2l

HF = os.environ.get('HF_HOME', '/home/smola/.cache/hf-lm')
QWEN_SMALL = glob.glob(f'{HF}/hub/models--Qwen--Qwen3.5-0.8B/snapshots/'
                       f'{d2l.HF_REVISIONS["Qwen/Qwen3.5-0.8B"]}/'
                       'model.safetensors*')

WORDS = ('the model reads seven pages of text and writes one clean '
         'paragraph that keeps every number from the source like 1999 '
         'or 3.14 and adds nothing new to it').split()


def fixture_docs(n=20, seed=0):
    """Documents of sentences (., ?, !, a closing quote), line breaks,
    and a few sentences far longer than the chunk bound."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n):
        parts = []
        for _ in range(int(rng.integers(3, 30))):
            k = int(rng.integers(3, 120 if rng.random() < 0.1 else 25))
            s = ' '.join(rng.choice(WORDS, k))
            parts.append(s[0].upper() + s[1:] + str(rng.choice(
                ['.', '?', '!', '."', ':\n', '.\n\n'])))
        docs.append(' '.join(parts).strip())
    return docs


class ChunkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = fixture_docs()
        raw = d2l.train_hf_bpe(cls.docs * 20, 400, d2l.R2_SPECIALS)
        cls.dir = tempfile.TemporaryDirectory()
        path = os.path.join(cls.dir.name, 'tokenizer.json')
        with open(path, 'wb') as f:
            f.write(raw.to_str(pretty=False).encode('utf-8'))
        cls.tok = d2l._r2_from_file(path)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_reconstruction_and_bound(self):
        cut = 0
        for doc in self.docs:
            ids = self.tok.encode(doc, add_special_tokens=False)
            chunks = d2l.chunk_doc(ids, self.tok, max_tokens=40)
            self.assertEqual(sum(chunks, []), ids)
            self.assertEqual(self.tok.decode(sum(chunks, [])), doc)
            self.assertLessEqual(max(map(len, chunks)), 40)
            for c, nxt in zip(chunks, chunks[1:]):  # the last ends the doc
                text = self.tok.decode(c)
                ends = bool(re.search(r'([.?!]["\')\]]*\s*|\n\s*)$', text))
                if not text[-1:].isspace():
                    ends = ends and self.tok.decode(nxt)[:1].isspace()
                if not ends:  # only a cut inside an over-long sentence
                    self.assertEqual(len(c), 40)
                    cut += 1
        self.assertGreater(cut, 0)  # the fixture has over-long sentences

    def test_greedy_fill(self):
        ids = self.tok.encode('A b c. D e f. G h.', add_special_tokens=False)
        whole = d2l.chunk_doc(ids, self.tok, max_tokens=len(ids))
        self.assertEqual(whole, [ids])
        one = d2l.chunk_doc(ids, self.tok, max_tokens=len(ids) - 1)
        self.assertGreater(len(one), 1)
        self.assertEqual(d2l.chunk_doc([], self.tok), [])


class MetricsTest(unittest.TestCase):
    SRC = 'In 1999 the town had 3,200 people and 14 schools; 2.5% left.'

    def test_identical(self):
        m = d2l.synth_metrics(self.SRC, self.SRC)
        self.assertEqual((m['copy_rate'], m['num_fidelity'],
                          m['len_ratio']), (1.0, 1.0, 1.0))

    def test_one_changed_number(self):
        out = self.SRC.replace('14', '15')
        m = d2l.synth_metrics(self.SRC, out)
        self.assertAlmostEqual(m['num_fidelity'], 1 - 1 / 4)
        self.assertLess(m['copy_rate'], 1.0)
        m = d2l.synth_metrics(self.SRC, out.replace('3,200', '3200'))
        self.assertAlmostEqual(m['num_fidelity'], 1 - 2 / 4)

    def test_no_numbers_and_disjoint(self):
        m = d2l.synth_metrics('no digits here', 'a different text')
        self.assertEqual(m['num_fidelity'], 1.0)
        m = d2l.synth_metrics('abcdefgh', 'stuvwxyz')
        self.assertEqual(m['copy_rate'], 0.0)
        self.assertAlmostEqual(m['len_ratio'], 1.0)

    def test_distinct_n(self):
        s = 'one rewrite of the same short source text again'
        for k in (1, 2, 5):
            self.assertAlmostEqual(d2l.distinct_n([s] * k, 4), 1 / k)
        self.assertEqual(d2l.distinct_n(['too short'], 4), 0.0)
        self.assertEqual(d2l.distinct_n([], 4), 0.0)


class CollapseTest(unittest.TestCase):
    def test_closed_forms(self):
        n, T, reps = 100, 20, 2000
        q = (n - 1) / n
        r = d2l.collapse_sim(n, T, 'replace', reps, seed=0)
        a = d2l.collapse_sim(n, T, 'accumulate', reps, seed=0)
        for out in (r, a):
            self.assertEqual(out['var_mu'].shape, (T + 1,))
            self.assertEqual(out['mean_sigma2'][0], 1.0)
            self.assertEqual(out['var_mu'][0], 0.0)
        for t in (1, 5, T):
            self.assertLess(abs(r['mean_sigma2'][t] - q**t),
                            3 * r['se_sigma2'][t])
            self.assertLess(abs(r['var_mu'][t] - (1 - q**t)),
                            3 * r['se_var_mu'][t])
            partial = sum(1 / s**2 for s in range(1, t + 1)) / n
            self.assertLess(abs(a['var_mu'][t] - partial),
                            3 * a['se_var_mu'][t])
        self.assertLess(a['var_mu'][T], np.pi**2 / 6 / n
                        + 3 * a['se_var_mu'][T])
        self.assertGreater(a['mean_sigma2'][T], 0.95)
        with self.assertRaises(ValueError):
            d2l.collapse_sim(n, 2, 'mix')


class _Cfg:
    eos_token_id = [2, 3]


class _ScriptedModel(torch.nn.Module):
    """generate() writes, per row, a reply chosen by the prompt's last
    character code: 'L' never stops, 'T' replies with thinking text, 'R'
    draws a random token (seed check), anything else replies 'ok' and
    stops; finished rows are padded with token 2."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = _Cfg()

    def generate(self, input_ids, attention_mask, generation_config):
        n = generation_config.max_new_tokens
        rows = []
        for x in input_ids.tolist():
            key = chr(x[-1])
            if key == 'L':
                new = [ord('a')] * n
            elif key == 'T':
                new = [ord(c) for c in '<think>x'] + [3]
            elif key == 'R':
                new = [ord('a') + int(torch.randint(0, 26, (1,)))] + [3]
            else:
                new = [ord('o'), ord('k'), 3]
            rows.append(new + [2] * (n - len(new)))
        L = max(map(len, rows))
        rows = [r + [2] * (L - len(r)) for r in rows]
        return torch.cat([input_ids, torch.tensor(rows)], 1)


class _CharTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt,
                            enable_thinking):
        assert enable_thinking is False and add_generation_prompt
        return msgs[0]['content'][-1]  # the key character

    def __call__(self, texts, add_special_tokens):
        return {'input_ids': [[ord(c) for c in t] for t in texts]}

    def decode(self, ids):
        return ''.join(chr(i) for i in ids)


class RephraseTest(unittest.TestCase):
    MODEL = (_ScriptedModel(), _CharTokenizer())

    def run_(self, chunks, **kw):
        return d2l.rephrase_docs(chunks, '{text}', model=self.MODEL,
                                 max_new_tokens=6, **kw)

    def test_rows(self):
        rows = self.run_(['xa', 'xL', 'xb', 'xL', 'xc'], batch=2)
        self.assertEqual([r['index'] for r in rows], [0, 2, 4])
        self.assertTrue(all(r['out'] == 'ok' and r['n_out'] == 2 and
                            r['finish'] == 'stop' for r in rows))
        self.assertEqual([r['index'] for r in rows.dropped], [1, 3])
        self.assertTrue(all(r['finish'] == 'length' and r['n_out'] == 6
                            for r in rows.dropped))
        self.assertEqual(rows[1]['src'], 'xb')
        styles = d2l.rephrase_docs(['a', 'b'], ['{text}L', '{text}x'],
                                   model=self.MODEL, max_new_tokens=6)
        self.assertEqual([r['index'] for r in styles], [1])
        self.assertEqual(len(styles.dropped), 1)

    def test_thinking_text_fails(self):
        with self.assertRaises(AssertionError):
            self.run_(['xa', 'xT'])

    def test_seeds(self):
        get = lambda seed, batch: [r['out'] for r in self.run_(
            ['R'] * 8, seed=seed, batch=batch)]
        self.assertEqual(get(0, 4), get(0, 4))
        self.assertNotEqual(get(0, 4), get(1, 4))
        self.assertNotEqual(get(0, 8)[4:], get(0, 8)[:4])

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            self.run_(['xa'], backend='other')

    def test_prompts(self):
        for style, p in d2l.REPHRASE_PROMPTS.items():
            self.assertIn('{text}', p)
            self.assertIn('Preserve every fact, name and number', p)
            self.assertEqual(p.format(text='X').count('X'), 1, style)


@unittest.skipUnless(QWEN_SMALL, 'Qwen3.5-0.8B not in the cache')
class QwenBatchTest(unittest.TestCase):
    def test_greedy_batch_one_equals_batch_eight(self):
        model, tok = d2l.load_hf('Qwen/Qwen3.5-0.8B', dtype=torch.float32,
                                 device_map='cpu')
        chunks = ['In 1999 the town of Elm had 3,200 people and 14 schools.',
                  'Water boils at 100 degrees Celsius at sea level.']
        kw = dict(model=(model, tok), max_new_tokens=32, temperature=0.0,
                  style='wiki')
        one = d2l.rephrase_docs(chunks, batch=1, **kw)
        eight = d2l.rephrase_docs(chunks * 4, batch=8, **kw)
        both = lambda rows: sorted(list(rows) + rows.dropped,
                                   key=lambda r: r['index'])
        a, b = both(one), both(eight)
        for i, r in enumerate(b):
            self.assertEqual(r['out'], a[i % 2]['out'], i)
            self.assertEqual(r['finish'], a[i % 2]['finish'], i)


if __name__ == '__main__':
    unittest.main()
