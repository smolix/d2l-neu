#!/usr/bin/env python3
"""CPU tests for sec_lm_tokenizers (chapter_lm-data/tokenizers.md), the
shared part written before the section (the one-way doors of chapter spec
1.6a/b): `train_hf_bpe` on a fixture corpus (special tokens at ids 0-31,
vocabulary size, digits split singly, NFC, two trainings byte-identical,
special strings in text kept as text); the reference-tokenizer wrapper on a
fixture file (`_r2_from_file`: SHA-256 check, EOS 0, no BOS, pad 1,
`_lm_boundary_id`, `_lm_encode`, `bits_per_byte` on a tiny random
`d2l.GPT`); `write_shards`/`TokenShards` round trip (EOS invariants, .idx,
SHA-256, `locate`/`iter_docs` resume, `locate_token`, `read` across a
boundary, `max_tokens`, uint32 for large vocabularies, meta columns in
order, `meta_fn`, a special id inside a document refused, corruption found
by `verify`); `normalize_url`; the registry guards of `r2_tokenizer` and
`fwedu_shards`.

Gated on the local Hugging Face cache (no download): `R2_SPLIT_REGEX`
equals the Split pattern of Qwen3-0.6B's tokenizer.json, and Qwen3's
tokenizer through `_doc_token_ids`/`write_shards` keeps special strings as
text and writes uint32 shards with EOS 151643."""
import hashlib
import json
import os
import random
import tempfile
import unicodedata
import unittest

import numpy as np
import torch

from d2l import torch as d2l

SPEC_REGEX = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
SPECIAL_DOC = ('A page about language models: <|endoftext|> ends a document, '
               '<|pad|> pads a batch, and <|im_start|>user ... <|im_end|> '
               'marks a chat turn.')


def corpus(n_docs=1000, seed=0):
    """Deterministic fixture documents: words, numbers, accents (some in
    NFD form), CJK and code."""
    rng = random.Random(seed)
    words = ['data', 'model', 'token', 'the', 'of', 'and', 'learning',
             'Grüße', 'café', 'café', 'naïve', '中文', '数据',
             'def', 'return', 'x', '==', '(', ')', ':', '\n', '    ']
    syl = ['ka', 'lo', 'mi', 'ne', 'ru', 'sa', 'ti', 'vo', 'zé', 'ße']
    words += [''.join(rng.choice(syl) for _ in range(rng.randint(1, 4)))
              for _ in range(300)]
    docs = []
    for i in range(n_docs):
        k = rng.randint(5, 60)
        parts = [rng.choice(words) if rng.random() > 0.15 else
                 str(rng.randint(0, 10**6)) for _ in range(k)]
        docs.append(' '.join(parts))
    return docs


class BpeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = corpus()
        cls.tok = d2l.train_hf_bpe(cls.docs, 512, d2l.R2_SPECIALS)

    def test_regex_is_the_spec(self):
        self.assertEqual(d2l.R2_SPLIT_REGEX, SPEC_REGEX)

    def test_specials_and_size(self):
        self.assertEqual(len(d2l.R2_SPECIALS), 32)
        self.assertEqual(self.tok.get_vocab_size(), 512)
        for i, s in enumerate(d2l.R2_SPECIALS):
            self.assertEqual(self.tok.token_to_id(s), i)
        with self.assertRaisesRegex(ValueError, 'too little text'):
            d2l.train_hf_bpe(['a b c'], 4096, d2l.R2_SPECIALS)

    def test_digits_nfc_round_trip(self):
        enc = self.tok.encode('x 1234567 y')
        digits = [t for t in enc.tokens if any(c.isdigit() for c in t)]
        self.assertEqual(len(digits), 7)
        self.assertEqual(self.tok.encode('café').ids,
                         self.tok.encode('café').ids)
        for t in self.docs[:50] + [SPECIAL_DOC]:
            nfc = unicodedata.normalize('NFC', t)
            self.assertEqual(self.tok.decode(self.tok.encode(t).ids), nfc)

    def test_special_strings_are_text(self):
        ids = self.tok.encode(SPECIAL_DOC).ids
        self.assertTrue(all(i >= 32 for i in ids))
        self.tok.encode_special_tokens = False  # the default matches them
        try:
            self.assertIn(0, self.tok.encode(SPECIAL_DOC).ids)
        finally:
            self.tok.encode_special_tokens = True
        self.assertTrue(all(i >= 32 for ids in d2l._doc_token_ids(
            self.tok, [SPECIAL_DOC]) for i in ids))

    def test_deterministic(self):
        again = d2l.train_hf_bpe(self.docs, 512, d2l.R2_SPECIALS)
        self.assertEqual(again.to_str(pretty=False),
                         self.tok.to_str(pretty=False))


class WrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tok = d2l.train_hf_bpe(corpus(), 512, d2l.R2_SPECIALS)
        cls.path = os.path.join(cls.tmp.name, 'tokenizer.json')
        data = tok.to_str(pretty=False).encode('utf-8')
        with open(cls.path, 'wb') as f:
            f.write(data)
        cls.sha = hashlib.sha256(data).hexdigest()
        cls.assert_bytes = data == tok.to_str(pretty=False).encode('utf-8')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_contract(self):
        self.assertTrue(self.assert_bytes)  # to_str is stable
        import transformers
        tok = d2l._r2_from_file(self.path, sha256=self.sha)
        self.assertIsInstance(tok, transformers.PreTrainedTokenizerFast)
        self.assertEqual((tok.eos_token_id, tok.pad_token_id), (0, 1))
        self.assertIsNone(tok.bos_token_id)
        self.assertEqual(d2l._lm_boundary_id(tok), 0)
        ids = tok.encode('a b', add_special_tokens=False)
        self.assertEqual(ids, tok('a b')['input_ids'])
        self.assertTrue(all(i >= 32 for i in ids))
        for ids in (tok.encode(SPECIAL_DOC), tok(SPECIAL_DOC)['input_ids'],
                    d2l._doc_token_ids(tok, [SPECIAL_DOC])[0]):
            self.assertTrue(all(i >= 32 for i in ids))
        self.assertEqual(tok.decode(tok.encode(SPECIAL_DOC)), SPECIAL_DOC)
        enc = d2l._lm_encode(tok, ['a b', SPECIAL_DOC])
        self.assertTrue(all(isinstance(i, int) for ids in enc for i in ids))
        raw = d2l._r2_from_file(self.path, raw=True)
        self.assertTrue(raw.encode_special_tokens)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            d2l._r2_from_file(self.path, sha256='0' * 64)

    def test_bits_per_byte(self):
        tok = d2l._r2_from_file(self.path)
        torch.manual_seed(0)
        model = d2l.GPT(len(tok), num_hiddens=32, num_heads=2, num_blks=1,
                        max_len=64)
        bpb = d2l.bits_per_byte(model, tok, corpus(4, seed=1), ctx_len=32)
        self.assertTrue(np.isfinite(bpb) and bpb > 0)

    def test_registry_guards(self):
        with self.assertRaises(KeyError):
            d2l.r2_tokenizer()
        with self.assertRaises(KeyError):
            d2l.fwedu_shards('100m')
        with self.assertRaises(ValueError):
            d2l.fwedu_shards('1b')


class TinyTiktoken:
    """A tiktoken-like encoding with ids above 65,536 (one per character)."""
    n_vocab = 70_000
    special_tokens_set = {'<|endoftext|>'}

    def encode_single_token(self, s):
        return 69_999

    def encode_ordinary(self, t):
        return [66_000 + ord(c) % 3000 for c in t]

    def encode_ordinary_batch(self, ts):
        return [self.encode_ordinary(t) for t in ts]


class ShardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = corpus()
        cls.tok = d2l.train_hf_bpe(cls.docs, 512, d2l.R2_SPECIALS)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, 'set')

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, **kw):
        meta = lambda i: dict(url=f'https://e.org/{i}', dump='CC-MAIN-2024-10',
                              bucket=9 + i % 5, edu_score=3.25, int_score=3,
                              gpt2_tokens=10 + i, src_file='000_00000.parquet',
                              src_row=i)
        docs = [(f'<urn:uuid:{i}>', t, meta(i))
                for i, t in enumerate(self.docs + [SPECIAL_DOC])]
        return d2l.write_shards(docs, self.tok, self.out, 'train',
                                shard_tokens=8000, batch_docs=64, **kw)

    def test_round_trip(self):
        entry = self.write()
        ts = d2l.TokenShards(self.out)
        self.assertGreater(ts.n_shards, 2)
        self.assertEqual((ts.n_docs, ts.n_tokens),
                         (len(self.docs) + 1, entry['n_tokens']))
        self.assertEqual((ts.dtype, ts.eos_id, ts.vocab_size),
                         (np.dtype('uint16'), 0, 512))
        self.assertTrue(ts.verify())
        texts = self.docs + [SPECIAL_DOC]
        for i in (0, 1, 500, len(texts) - 1):
            want = self.tok.encode(texts[i]).ids + [0]
            self.assertEqual(ts.doc(i).tolist(), want)
            self.assertEqual(ts.doc(i).dtype, np.int32)
        for k, s in enumerate(ts.shards):  # closed at the first boundary
            if k < ts.n_shards - 1:
                last = int(ts.offsets(k)[-2])
                self.assertLess(last, 8000)
                self.assertGreaterEqual(s['n_tokens'], 8000)
        lens = [len(ts.doc(i)) for i in range(ts.shards[0]['n_docs'])]
        self.assertEqual(ts.offsets(0).tolist(), [0] + np.cumsum(lens)
                         .tolist())
        i = ts.shards[0]['n_docs'] + 3
        k, j, off = ts.locate(i)
        self.assertEqual((k, j, off), (1, 3, int(ts.offsets(1)[3])))
        resumed = [d.tolist() for _, d in zip(range(3), (
            d for _, d in ts.iter_docs(start=i)))]
        self.assertEqual(resumed, [ts.doc(i + r).tolist() for r in range(3)])
        b = ts.shards[0]['n_tokens']
        self.assertEqual(ts.locate_token(b + 5), (1, 5))
        flat = np.concatenate([ts.doc(i) for i in range(ts.n_docs)])
        self.assertEqual(ts.read(b - 7, 20).tolist(),
                         flat[b - 7:b + 13].tolist())
        self.assertEqual(int((flat == 0).sum()), ts.n_docs)
        self.assertEqual(d2l.TokenShards(self.out, max_tokens=b).n_shards, 1)
        self.assertEqual(d2l.TokenShards(self.out, max_tokens=b + 1)
                         .n_shards, 2)
        m = ts.meta(0)
        self.assertEqual(list(m.columns), ['doc_id', 'n_tokens', 'url',
                                           'dump', 'bucket', 'edu_score',
                                           'int_score', 'gpt2_tokens',
                                           'src_file', 'src_row'])
        import pyarrow.parquet as pq
        schema = pq.read_schema(os.path.join(self.out, 'train-00000'
                                                       '.meta.parquet'))
        self.assertEqual([str(schema.field(c).type) for c in m.columns],
                         ['string', 'int32', 'string', 'string', 'int16',
                          'float', 'int8', 'int32', 'string', 'int64'])
        with open(os.path.join(self.out, 'manifest.json')) as f:
            man = json.load(f)
        self.assertEqual(man['schema'], 'd2l-corpus/1')
        self.assertEqual(man['format'], dict(
            dtype='uint16', eos_id=0, endianness='little', doc_end='eos',
            idx_dtype='uint32', meta_required=['doc_id', 'n_tokens']))
        self.assertEqual(man['splits']['train'], entry)

    def test_deterministic_bytes(self):
        a = self.write()
        self.tmp2 = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp2.name, 'set')
        b = self.write()
        self.tmp2.cleanup()
        self.assertEqual(a, b)

    def test_meta_fn_and_manifest(self):
        m = d2l.CorpusManifest('pes2o').add_stage('nfc', {}, 3, 3, {}, 'a', 'b')
        entry = d2l.write_shards(
            [('a', 'one two', None), ('b', 'three', None), ('c', 'four', None)],
            self.tok, self.out, 'val', manifest=m,
            meta_fn=lambda i, t, meta: dict(n_chars=len(t), field='bio'))
        ts = d2l.TokenShards(self.out, 'val')
        self.assertEqual(list(ts.meta(0).columns), ['doc_id', 'n_tokens',
                                                    'n_chars', 'field'])
        self.assertEqual(entry['n_docs'], 3)
        man = d2l.CorpusManifest.load(os.path.join(self.out, 'manifest.json'))
        self.assertEqual(man.stages[0]['name'], 'nfc')
        self.assertTrue(ts.verify())

    def test_uint32_and_refusal(self):
        enc = TinyTiktoken()
        d2l.write_shards([('a', 'hello'), ('b', 'world!')], enc, self.out,
                         'train')
        ts = d2l.TokenShards(self.out)
        self.assertEqual((ts.dtype, ts.eos_id), (np.dtype('uint32'), 69_999))
        self.assertEqual(ts.doc(1).tolist(), enc.encode_ordinary('world!')
                         + [69_999])
        self.assertTrue(ts.verify())
        bad = TinyTiktoken()
        bad.encode_ordinary_batch = lambda ts: [[66_001, 69_999, 66_002]
                                                for _ in ts]
        with self.assertRaisesRegex(ValueError, 'special id 69999'):
            d2l.write_shards([('a', 'x')], bad, self.out + '2', 'train')

    def test_verify_finds_corruption(self):
        self.write()
        path = os.path.join(self.out, 'train-00001.bin')
        with open(path, 'r+b') as f:
            f.seek(10)
            b = f.read(1)
            f.seek(10)
            f.write(bytes([b[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, 'train-00001.bin'):
            d2l.TokenShards(self.out).verify()


class UrlTest(unittest.TestCase):
    def test_normalize_url(self):
        cases = {'https://www.Example.com/': 'https://example.com',
                 'HTTP://WWW.example.com/a/b/?q=1#frag':
                     'http://example.com/a/b?q=1',
                 'https://example.com/a?x=Y': 'https://example.com/a?x=Y',
                 'https://wwwx.org/Path/': 'https://wwwx.org/Path',
                 ' https://sub.www.org/ ': 'https://sub.www.org'}
        for url, want in cases.items():
            self.assertEqual(d2l.normalize_url(url), want, url)


def qwen3_tokenizer():
    try:
        import transformers
        return transformers.AutoTokenizer.from_pretrained(
            'Qwen/Qwen3-0.6B-Base',
            revision=d2l.HF_REVISIONS['Qwen/Qwen3-0.6B-Base'],
            local_files_only=True)
    except Exception:
        return None


class Qwen3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = qwen3_tokenizer()
        if cls.tok is None:
            raise unittest.SkipTest('Qwen3-0.6B-Base not in the HF cache')

    def test_split_pattern(self):
        cfg = json.loads(self.tok.backend_tokenizer.to_str())
        pres = cfg['pre_tokenizer']['pretokenizers']
        split = [p for p in pres if p['type'] == 'Split'][0]
        self.assertEqual(split['pattern']['Regex'], d2l.R2_SPLIT_REGEX)

    def test_special_strings_and_shards(self):
        (ids,) = d2l._doc_token_ids(self.tok, [SPECIAL_DOC])
        special = d2l._tok_info(self.tok)[2]
        self.assertFalse(set(ids) & special)
        self.assertIn(151643, self.tok(SPECIAL_DOC)['input_ids'])  # default
        with tempfile.TemporaryDirectory() as tmp:
            d2l.write_shards([('a', SPECIAL_DOC), ('b', 'plain text')],
                             self.tok, tmp, 'train')
            ts = d2l.TokenShards(tmp)
            self.assertEqual((ts.dtype, ts.eos_id), (np.dtype('uint32'),
                                                     151643))
            self.assertEqual(ts.doc(0).tolist(), ids + [151643])
            self.assertTrue(ts.verify())


if __name__ == '__main__':
    unittest.main()
