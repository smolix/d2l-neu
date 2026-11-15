#!/usr/bin/env python3
"""CPU tests for sec_lm_data_mixtures (chapter_lm-data/data-mixtures.md),
the shared part written before the section: the FineWeb-Edu bucket rule
(golden ids, the split ranges partition 0-255), `bpb_from_nll` on a
uniform byte model, `token_byte_lengths` for byte-level, SentencePiece-style
and Transformers-wrapped tokenizers (the byte lengths of a text's ids sum to
its UTF-8 length), `load_pool` on fixture pools (one pool, several pools,
a length mismatch), and the chapter's source-file order."""
import json
import math
import os
import tempfile
import unittest
import unicodedata

import numpy as np

from d2l import torch as d2l

TEXTS = ['The quick brown fox jumps over 13 lazy dogs.',
         'Grüße aus Köln: 3.14159 ist π, und 中文 hat keine Leerzeichen.',
         'def f(x):\n    return x ** 2  # squares\n',
         'Café naïve résumé — façade.'] * 50


class BucketTest(unittest.TestCase):
    def test_golden(self):
        golden = {'<urn:uuid:00000000-0000-0000-0000-000000000000>': 87,
                  '<urn:uuid:4ad6d48e-21c8-4a6f-bd16-3f43ce4f5e6b>': 158,
                  '<urn:uuid:ffffffff-ffff-ffff-ffff-ffffffffffff>': 56}
        for doc_id, b in golden.items():
            self.assertEqual(d2l.fwedu_bucket(doc_id), b)

    def test_partition(self):
        seen = [b for r in d2l.FWEDU_BUCKETS.values() for b in r]
        self.assertEqual(sorted(seen), list(range(256)))
        self.assertEqual(list(d2l.FWEDU_BUCKETS['val']), [0])
        self.assertEqual(list(d2l.FWEDU_BUCKETS['side']), list(range(1, 8)))
        counts = np.bincount([d2l.fwedu_bucket(f'<urn:uuid:{i}>')
                              for i in range(25600)], minlength=256)
        self.assertLess(abs(counts.mean() - 100), 1e-9)
        self.assertLess(counts.std(), 20)  # about sqrt(100)

    def test_files(self):
        names = [f.split('/')[-1][:3] for f in d2l._FWEDU_FILES]
        self.assertEqual(names, ['000', '001', '002', '003', '004', '006',
                                 '007', '008', '009', '010', '011'])


class BpbTest(unittest.TestCase):
    def test_uniform(self):
        n = 1000  # 1,000 byte tokens under a uniform model over 256 bytes
        self.assertAlmostEqual(d2l.bpb_from_nll(n * math.log(256), n), 8.0,
                               delta=1e-9)
        self.assertAlmostEqual(d2l.bpb_from_nll(math.log(2) * 3, 3), 1.0,
                               delta=1e-12)


class ByteLengthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = d2l.train_hf_bpe(TEXTS, 400, d2l.R2_SPECIALS)

    def check(self, lengths, encode):
        for t in TEXTS[:4]:
            t = unicodedata.normalize('NFC', t)
            self.assertEqual(int(lengths[encode(t)].sum()),
                             len(t.encode('utf-8')), t)

    def test_byte_level(self):
        lengths = d2l.token_byte_lengths(self.tok)
        self.assertEqual(len(lengths), 400)
        self.assertTrue((lengths[:32] == 0).all())
        self.assertTrue((lengths[32:] > 0).all())
        self.check(lengths, lambda t: self.tok.encode(t).ids)

    def test_transformers_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'tokenizer.json')
            with open(p, 'w', encoding='utf-8') as f:
                f.write(self.tok.to_str(pretty=False))
            wrapped = d2l._r2_from_file(p)
        lengths = d2l.token_byte_lengths(wrapped)
        self.assertTrue((lengths == d2l.token_byte_lengths(self.tok)).all())
        self.check(lengths, lambda t: wrapped.encode(
            t, add_special_tokens=False))

    def test_sentencepiece_style(self):
        from tokenizers import Tokenizer, models
        vocab = {'▁hello': 0, '<0x41>': 1, 'é': 2, '<unk>': 3, '<s>': 4}
        tok = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
        tok.add_special_tokens(['<s>'])
        self.assertEqual(list(d2l.token_byte_lengths(tok)), [6, 1, 2, 5, 0])


class PoolTest(unittest.TestCase):
    def test_load_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = np.arange(100, dtype=np.uint16)
            ids.tofile(os.path.join(tmp, 'web-1k.bin'))
            meta = dict(n_tokens=100, n_docs=3, dtype='uint16', source='x')
            with open(os.path.join(tmp, 'meta.json'), 'w') as f:
                json.dump(meta, f)
            for arg in (tmp, os.path.join(tmp, 'web-1k.bin')):
                pool = d2l.load_pool(arg)
                self.assertEqual(pool.meta, meta)
                self.assertTrue((pool.ids == ids).all())
                with self.assertRaises(ValueError):
                    pool.ids[0] = 1  # read-only
            np.arange(50, dtype=np.uint16).tofile(os.path.join(tmp,
                                                               'math-1k.bin'))
            with self.assertRaisesRegex(ValueError, '2 pools'):
                d2l.load_pool(tmp)
            with open(os.path.join(tmp, 'meta.json'), 'w') as f:
                json.dump({'web-1k.bin': meta,
                           'math-1k.bin': dict(meta, n_tokens=50)}, f)
            self.assertEqual(len(d2l.load_pool(tmp, 'math-1k').ids), 50)
            with open(os.path.join(tmp, 'meta.json'), 'w') as f:
                json.dump(dict(meta, n_tokens=99), f)
            with self.assertRaisesRegex(ValueError, 'meta says'):
                d2l.load_pool(tmp, 'web-1k')


if __name__ == '__main__':
    unittest.main()
