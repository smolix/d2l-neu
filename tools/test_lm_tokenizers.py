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
boundary, a pickled reader without its memory maps, `max_tokens`, uint32
for large vocabularies, meta columns in order, `meta_fn`, a special id inside a document refused, corruption found
by `verify`); `normalize_url`; the registry guards of `r2_tokenizer` and
`fwedu_shards`; lazy shard downloads checked by hash (network mocked) and
the 4B manifest's folder, kept while its sha1 stays and cleared when it
changes.

Section cells (20.6): the unigram model of D2 against brute force
(forward-backward expected counts and log P(s) vs enumeration to 1e-9,
Viterbi = argmax with ties to fewer pieces, EM never lowers the corpus
log-likelihood, sampled segmentations within 3 sigma of their posteriors,
`UnigramTokenizer` trains to its size); `fertility_table` on a toy encoder
pair; `extend_vocab` on tiny random tied and untied Llama models (ASCII ids
unchanged, the decomposition-logit identity to 1e-5, rows of each rule,
config.vocab_size); `sp_train` (thread-pinned pieces identical across two
trainings, lossless round trip, `viterbi_segment` on `_sp_pieces_scores`
reproduces its segmentations); `_bpe_prefix`, `_vocab_bytes`,
`_fallback_ids`; `_rolling_ngram_hashes` against direct hashing.

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
        from unittest import mock
        hub = {k: v for k, v in d2l.DATA_HUB.items()
               if k not in ('lm-r2-tokenizer-v1', 'lm-fwedu-shards-100m-v1')}
        with mock.patch.dict(d2l.DATA_HUB, hub, clear=True):
            with self.assertRaises(KeyError):
                d2l.r2_tokenizer()
            with self.assertRaises(KeyError):
                d2l.fwedu_shards('100m')
        with mock.patch.object(d2l, 'R2_TOKENIZER_SHA256', None):
            with self.assertRaises(KeyError):
                d2l.r2_tokenizer()
        with self.assertRaises(ValueError):
            d2l.fwedu_shards('1b')

    @unittest.skipUnless(os.environ.get('D2L_TEST_NET') == '1',
                         'network test (D2L_TEST_NET=1)')
    def test_registered_tokenizer(self):
        """The registered file (downloaded into a temporary folder) has the
        pinned SHA-256 and the reference contract."""
        with tempfile.TemporaryDirectory() as d:
            path = d2l.download_keyed('lm-r2-tokenizer-v1', root=d)
            tok = d2l._r2_from_file(path, sha256=d2l.R2_TOKENIZER_SHA256)
            raw = d2l._r2_from_file(path, raw=True)
        self.assertEqual(len(tok), 32_768)
        self.assertEqual([raw.id_to_token(i) for i in range(32)],
                         d2l.R2_SPECIALS)
        self.assertEqual((tok.eos_token_id, tok.bos_token_id), (0, None))
        ids = d2l._doc_token_ids(tok, [SPECIAL_DOC])[0]
        self.assertTrue(min(ids) >= 32 and tok.decode(ids) == SPECIAL_DOC)
        self.assertEqual([raw.id_to_token(i) for i in raw.encode('2026').ids],
                         list('2026'))


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
        import pickle
        blob = pickle.dumps(ts)  # opened memory maps are not copied
        self.assertLess(len(blob), 100_000)
        self.assertEqual(pickle.loads(blob).read(b - 7, 20).tolist(),
                         flat[b - 7:b + 13].tolist())
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


    def test_max_shards(self):
        full = self.write()
        self.tmp2 = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp2.name, 'set')
        part = self.write(max_shards=2)
        self.tmp2.cleanup()
        self.assertEqual(len(part['shards']), 2)
        self.assertEqual(part['shards'], full['shards'][:2])


def builder():
    """tools/build_r2_shards.py as a module."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'build_r2_shards.py')
    spec = importlib.util.spec_from_file_location('build_r2_shards', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BuilderTest(unittest.TestCase):
    """The stream, val split and val_dedup rules of build_r2_shards.py on
    fixture streams."""
    @classmethod
    def setUpClass(cls):
        cls.b = builder()

    def test_shuffle_buffer(self):
        sb = self.b.shuffle_buffer
        a = list(sb(range(1000), seed=0, size=100))
        self.assertEqual(a, list(sb(range(1000), seed=0, size=100)))
        self.assertNotEqual(a, list(sb(range(1000), seed=1, size=100)))
        self.assertEqual(sorted(a), list(range(1000)))
        self.assertLess(max(a[:500]), 600)  # emits from its window
        self.assertEqual(sorted(sb(range(30), size=100)), list(range(30)))
        rng, buf, want = np.random.default_rng(0), list(range(4)), []
        for x in range(4, 10):  # the rule, written out
            j = int(rng.integers(4))
            want.append(buf[j])
            buf[j] = x
        want += [buf[int(j)] for j in rng.permutation(4)]
        self.assertEqual(list(sb(range(10), seed=0, size=4)), want)

    def test_train_docs(self):
        ids = [f'<urn:uuid:{i}>' for i in range(400)]
        rows = [dict(id=i, text='' if k % 50 == 7 else
                     unicodedata.normalize('NFD', f'café {k}'), url=f'u{k}',
                     dump='CC-MAIN-2024-10', score=3.0, int_score=3,
                     token_count=k) for k, i in enumerate(ids)]
        stats = {}
        docs = list(self.b.train_docs(
            ['f0'], stats, rows_of=lambda name: enumerate(rows)))
        buckets = [d2l.fwedu_bucket(i) for i in ids]
        want = [k for k in range(400) if k % 50 != 7 and buckets[k] >= 9]
        self.assertEqual([m['src_row'] for _, _, m in docs], want)
        self.assertEqual(stats, dict(rows=400, empty=8))
        self.assertTrue(all(t == unicodedata.normalize('NFC', t)
                            and m['bucket'] >= 9 for _, t, m in docs))
        self.assertEqual(sorted(m['src_row'] for _, _, m in self.b
                                .train_stream(['f0'], rows_of=lambda n:
                                              enumerate(rows), size=16)),
                         want)

    def test_val_split(self):
        b = self.b
        cand = [(f'v{i}', f'validation text {i} ' * (1 + i % 3),
                 dict(url=f'https://www.e.org/p/{i}/')) for i in range(40)]
        keys = {b.text_key(cand[3][1]), b.text_key(cand[5][1])}
        urls = {d2l.normalize_url('HTTPS://e.org/p/5'),
                d2l.normalize_url('https://e.org/p/8#top')}
        count = lambda texts: [len(t.split()) for t in texts]
        kept, drops, n, n_tok = b.val_split(cand, keys, urls, count,
                                            quota=100, seed=1)
        self.assertEqual(drops, dict(text_hash=2, url=1))  # 5: first rule
        self.assertEqual(n, 37)
        rest = [d for d in cand if d[0] not in ('v3', 'v5', 'v8')]
        perm = [rest[int(i)] for i in np.random.default_rng(1)
                .permutation(37)]
        lens = np.cumsum([c + 1 for c in count([d[1] for d in perm])])
        cut = int(np.argmax(lens >= 100)) + 1
        self.assertEqual(kept, perm[:cut])
        self.assertEqual(n_tok, int(lens[cut - 1]))
        self.assertGreaterEqual(n_tok, 100)
        self.assertLess(int(lens[cut - 2]), 100)
        with self.assertRaisesRegex(ValueError, 'R10'):
            b.val_split(cand, keys, urls, count, quota=10**6)

    def test_docs_hash_of_pairs(self):
        docs = [('b', 'two'), ('a', 'one')]
        pairs = [(i, hashlib.sha256(t.encode()).hexdigest()) for i, t in docs]
        self.assertEqual(self.b.docs_hash_of_pairs(pairs),
                         d2l.docs_hash(docs))


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


def segmentations(s, logp, max_len):
    if not s:
        yield []
        return
    for j in range(1, min(max_len, len(s)) + 1):
        if s[:j] in logp:
            for rest in segmentations(s[j:], logp, max_len):
                yield [s[:j]] + rest


def random_pieces(rng, n=12, max_len=4):
    pieces = set('abc') | {''.join(rng.choice(list('abc'), k)) for k in
                           rng.integers(2, max_len + 1, n)}
    w = rng.random(len(pieces))
    return dict(zip(sorted(pieces), np.log(w / w.sum())))


class UnigramTest(unittest.TestCase):
    def test_forward_backward_and_viterbi(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            logp = random_pieces(rng)
            s = ''.join(rng.choice(list('abc'), rng.integers(1, 13)))
            segs = list(segmentations(s, logp, 4))
            lps = np.array([sum(logp[x] for x in g) for g in segs])
            logz, counts = d2l._unigram_fb(s, logp, 4)
            self.assertAlmostEqual(logz, np.logaddexp.reduce(lps), delta=1e-9)
            brute = {}
            for g, lp in zip(segs, lps):
                for x in g:
                    brute[x] = brute.get(x, 0) + np.exp(
                        lp - np.logaddexp.reduce(lps))
            for x in set(brute) | set(counts):
                self.assertAlmostEqual(brute.get(x, 0), counts[x], delta=1e-9)
            v = d2l.viterbi_segment(s, logp, 4)
            top = [len(g) for g, lp in zip(segs, lps)
                   if lp >= lps.max() - 1e-12]
            self.assertEqual(''.join(v), s)
            self.assertGreaterEqual(sum(logp[x] for x in v), lps.max() - 1e-12)
            self.assertEqual(len(v), min(top))

    def test_viterbi_unknown(self):
        logp = {'a': -1.0, 'b': -1.0, 'ab': -1.5}
        self.assertEqual(d2l.viterbi_segment('abab', logp, 2), ['ab', 'ab'])
        with self.assertRaises(ValueError):
            d2l.viterbi_segment('abz', logp, 2)
        self.assertEqual(d2l.viterbi_segment('abz', logp, 2, unk=-20.0),
                         ['ab', 'z'])

    def test_em_monotone(self):
        rng = np.random.default_rng(0)
        words = {}
        for _ in range(5000):
            w = ''.join(rng.choice(list('abc'), rng.integers(3, 9)))
            words[w] = words.get(w, 0) + 1
        uni = d2l.UnigramTokenizer(0, max_piece_len=3)
        logp = random_pieces(np.random.default_rng(1), n=20, max_len=3)
        lls = []
        for _ in range(6):
            logp, ll = uni._em_step(words, logp)
            lls.append(ll)
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(lls, lls[1:])))
        self.assertAlmostEqual(sum(np.exp(v) for v in logp.values()), 1.0,
                               delta=1e-9)

    def test_sampling_frequencies(self):
        s, alpha = 'abcabc', 0.2
        logp = random_pieces(np.random.default_rng(2), n=40)
        segs = [tuple(g) for g in segmentations(s, logp, 4)]
        w = alpha * np.array([sum(logp[x] for x in g) for g in segs])
        post = dict(zip(segs, np.exp(w - np.logaddexp.reduce(w))))
        rng, n, freq = np.random.default_rng(0), 10_000, {}
        for _ in range(n):
            g = tuple(d2l._unigram_sample(s, logp, 4, alpha, rng))
            freq[g] = freq.get(g, 0) + 1
        self.assertLessEqual(set(freq), set(post))
        for g, p in post.items():
            self.assertLess(abs(freq.get(g, 0) / n - p),
                            3 * np.sqrt(p * (1 - p) / n), g)

    def test_train_and_encode(self):
        text = [' '.join(corpus(200, seed=1))]
        uni = d2l.UnigramTokenizer(300, seed_size=2000).train(text)
        self.assertLessEqual(len(uni.pieces), 300)
        self.assertGreater(len(uni.pieces), 250)
        sentence = 'the model of data and learning'
        pieces = uni.encode(sentence)
        self.assertEqual(''.join(pieces), ''.join(
            '▁' + w for w in sentence.split()))
        sample = uni.encode_sample(sentence, alpha=0.2, seed=3)
        self.assertEqual(''.join(sample), ''.join(pieces))


class FertilityTest(unittest.TestCase):
    def test_toy(self):
        chars = lambda s: [ord(c) for c in s]
        pairs = lambda s: [ord(c) for c in s[::2]]
        par = {'eng_Latn': ['abcd', 'ef'], 'xx': ['abcdefgh', 'ü']}
        df = d2l.fertility_table({'chars': (chars, {ord('ü')}),
                                  'pairs': pairs}, par).set_index(
            ['encoder', 'lang'])
        self.assertEqual(df.loc[('chars', 'xx'), 'tokens'], 9)
        self.assertAlmostEqual(df.loc[('chars', 'xx'), 'premium'], 9 / 6)
        self.assertAlmostEqual(df.loc[('pairs', 'xx'), 'premium'], 5 / 3)
        self.assertAlmostEqual(df.loc[('chars', 'xx'), 'fallback_share'], 1 / 9)
        self.assertAlmostEqual(df.loc[('chars', 'xx'), 'bytes_per_token'],
                               10 / 9)
        self.assertEqual(df.loc[('pairs', 'eng_Latn'), 'premium'], 1.0)


def tiny_lm(tok, tied=True, seed=0):
    import transformers
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=tied)
    return transformers.LlamaForCausalLM(cfg).eval()


class ExtendVocabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import transformers
        from tokenizers import Tokenizer
        spec = d2l.train_hf_bpe(corpus(400), 600, ['<|endoftext|>']).to_str()
        cls.make_tok = staticmethod(
            lambda: transformers.PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer.from_str(spec),
                eos_token='<|endoftext|>'))
        cls.new = ['नमस्ते', 'दुनिया', 'ßüé']

    def test_decomp_identity_and_ascii(self):
        tok = self.make_tok()
        model = tiny_lm(tok)
        ascii_text = ['the model of data', 'def return x == 42']
        before = tok(ascii_text, add_special_tokens=False)['input_ids']
        parts = tok(self.new, add_special_tokens=False)['input_ids']
        n_old = len(tok)
        model, tok = d2l.extend_vocab(model, tok, self.new, init='decomp')
        self.assertEqual(tok(ascii_text, add_special_tokens=False)[
            'input_ids'], before)
        self.assertEqual(model.config.vocab_size, n_old + len(self.new))
        W = model.get_output_embeddings().weight
        self.assertIs(W, model.get_input_embeddings().weight)
        x = torch.tensor([before[0]])
        with torch.no_grad():
            h = model.model(input_ids=x).last_hidden_state[0]
            logits = model(input_ids=x).logits[0]
        for k, ids in enumerate(parts):
            torch.testing.assert_close(
                logits[:, n_old + k], (h @ W[ids].T).mean(1), atol=1e-5,
                rtol=0)
        self.assertEqual(tok(self.new[0], add_special_tokens=False)[
            'input_ids'], [n_old])

    def test_rules_and_untied(self):
        for tied in (True, False):
            for init in ('random', 'mean'):
                tok = self.make_tok()
                model = tiny_lm(tok, tied=tied)
                n_old = len(tok)
                old_out = model.get_output_embeddings().weight[:n_old].clone()
                model, tok = d2l.extend_vocab(model, tok, self.new, init=init,
                                              seed=1)
                mats = [model.get_input_embeddings().weight]
                if not tied:
                    mats.append(model.get_output_embeddings().weight)
                    torch.testing.assert_close(
                        mats[1][:n_old], old_out)
                for W in mats:
                    new = W[n_old:].detach()
                    self.assertEqual(new.shape[0], len(self.new))
                    self.assertFalse(torch.equal(new[0], new[1]))
                    if init == 'mean':
                        torch.testing.assert_close(
                            new.mean(0), W[:n_old].mean(0), atol=0.01,
                            rtol=0)
        with self.assertRaises(ValueError):
            d2l.extend_vocab(model, tok, ['x'], init='zeros')


class SentencePieceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            raise unittest.SkipTest('sentencepiece not installed')
        cls.docs = corpus(600, seed=2)
        cls.sp = d2l.sp_train(cls.docs, 'unigram', 500)

    def test_deterministic_and_lossless(self):
        again = d2l.sp_train(self.docs, 'unigram', 500)
        self.assertEqual(d2l._sp_fingerprint(again),
                         d2l._sp_fingerprint(self.sp))
        for t in self.docs[:50] + ['a  b\tc\nd']:
            self.assertEqual(self.sp.decode(self.sp.encode(t)), t)

    def test_viterbi_reproduces(self):
        logp, unk = d2l._sp_pieces_scores(self.sp)
        for line in corpus(100, seed=3):
            s = '▁' + line.replace(' ', '▁')
            ours = []
            for x in d2l.viterbi_segment(s, logp, 16, unk=unk):
                ours += ([x] if x in logp else
                         [f'<0x{b:02X}>' for b in x.encode('utf-8')])
            self.assertEqual(ours, self.sp.encode(line, out_type=str), line)


class VocabHelpersTest(unittest.TestCase):
    def test_prefix_bytes_fallback(self):
        docs = [unicodedata.normalize('NFC', d) for d in corpus(400)]
        tok = d2l.train_hf_bpe(docs, 700, ['<|endoftext|>'])
        full = d2l._bpe_prefix(tok, 700)
        self.assertEqual([e.ids for e in full.encode_batch(docs[:50])],
                         [e.ids for e in tok.encode_batch(docs[:50])])
        small = d2l._bpe_prefix(tok, 400)
        self.assertEqual(small.get_vocab_size(), 400)
        enc = small.encode_batch(docs[:50])
        self.assertTrue(all(max(e.ids) < 400 for e in enc))
        self.assertEqual([small.decode(e.ids) for e in enc], docs[:50])
        self.assertGreater(sum(map(len, enc)),
                           sum(len(e.ids) for e in tok.encode_batch(docs[:50])))
        vb = d2l._vocab_bytes(tok)
        self.assertIsNone(vb[0])  # <|endoftext|>
        for t, e in zip(docs[:20], tok.encode_batch(docs[:20])):
            self.assertEqual(b''.join(vb[i] for i in e.ids), t.encode('utf-8'))
        fb = d2l._fallback_ids(tok)
        self.assertTrue(fb and all(len(vb[i]) == 1 and vb[i][0] >= 0x80
                                   for i in fb))

    def test_rolling_hashes(self):
        rng = np.random.default_rng(0)
        ids = rng.integers(1, 50, 400)
        ids[[100, 250]] = 0
        ids[300:310] = ids[20:30]
        h, ok = d2l._rolling_ngram_hashes(ids, 10, eos=0)
        self.assertEqual(len(h), len(ids) - 9)
        self.assertEqual(h[300], h[20])
        for i in range(len(h)):
            self.assertEqual(bool(ok[i]), 0 not in ids[i:i + 10].tolist())
        direct = {}
        for i in range(len(h)):
            direct.setdefault(tuple(ids[i:i + 10]), set()).add(int(h[i]))
        self.assertTrue(all(len(v) == 1 for v in direct.values()))
        self.assertEqual(len({int(x) for x in h}), len(direct))


class FetchTest(unittest.TestCase):
    """Lazy shard downloads (TokenShards(fetch=True), _fetch_shard_file) and
    the 4B manifest's folder in fwedu_shards, with the network mocked."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, 'set')
        tok = d2l.train_hf_bpe(corpus(200), 400, d2l.R2_SPECIALS)
        d2l.write_shards([(str(i), t) for i, t in enumerate(corpus(200))],
                         tok, self.out, 'train', shard_tokens=2000)
        with open(os.path.join(self.out, 'manifest.json')) as f:
            man = json.load(f)
        self.files = {}  # url -> bytes, the mocked server
        for s in man['splits']['train']['shards']:
            s['url'], s['sha1'] = {}, {}
            for ext, e in d2l._SHARD_EXT.items():
                with open(os.path.join(self.out, s['file'] + e), 'rb') as f:
                    data = f.read()
                s['url'][ext] = f'https://example.org/{s["file"]}{e}'
                s['sha1'][ext] = hashlib.sha1(data).hexdigest()
                self.files[s['url'][ext]] = data
        with open(os.path.join(self.out, 'manifest.json'), 'w') as f:
            json.dump(man, f)
        self.man = man

    def tearDown(self):
        self.tmp.cleanup()

    def get(self, url, **kw):
        from unittest import mock
        r = mock.MagicMock()
        r.__enter__.return_value = r
        r.iter_content = lambda n: [self.files[url][i:i + n] for i in
                                    range(0, len(self.files[url]), n)]
        self.calls.append(url)
        return r

    def test_lazy_fetch(self):
        from unittest import mock
        ref = d2l.TokenShards(self.out)
        want = [ref.doc(i).tolist() for i in range(ref.n_docs)]
        for name in os.listdir(self.out):
            if name.startswith('train-00001'):
                os.remove(os.path.join(self.out, name))
        self.calls = []
        with mock.patch.object(d2l.requests, 'get', self.get):
            with self.assertRaises(FileNotFoundError):
                d2l.TokenShards(self.out).verify()
            ts = d2l.TokenShards(self.out, fetch=True)
            self.assertEqual(ts.doc(0).tolist(), want[0])
            self.assertEqual(self.calls, [])  # shard 0 is on disk
            self.assertEqual([ts.doc(i).tolist() for i in range(ts.n_docs)],
                             want)
            self.assertTrue(ts.verify())
        self.assertEqual(sorted(u.rsplit('/', 1)[1] for u in self.calls),
                         ['train-00001.bin', 'train-00001.idx',
                          'train-00001.meta.parquet'])
        self.assertFalse([n for n in os.listdir(self.out)
                          if n.endswith('.part')])

    def test_bad_download(self):
        from unittest import mock
        s = self.man['splits']['train']['shards'][1]
        path = os.path.join(self.tmp.name, 'x.bin')
        self.files[s['url']['bin']] = b'not the shard'
        self.calls = []
        with mock.patch.object(d2l.requests, 'get', self.get):
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                d2l._fetch_shard_file(s, 'bin', path)
        self.assertEqual(os.listdir(self.tmp.name), ['set'])

    def test_fwedu_4b_folder(self):
        from unittest import mock
        cwd = os.getcwd()
        work = os.path.join(self.tmp.name, 'a', 'b')  # ../data/lm -> a/data
        os.makedirs(work)
        src = os.path.join(self.out, 'manifest.json')
        hub = dict(d2l.DATA_HUB, **{'lm-fwedu-manifest-4b-v1': ('u', 'one')})
        try:
            os.chdir(work)
            with mock.patch.dict(d2l.DATA_HUB, hub, clear=True), \
                    mock.patch.object(d2l, 'download_keyed',
                                      lambda key: src):
                ts = d2l.fwedu_shards('4b', fetch=False)
                root = ts.root
                self.assertEqual(ts.n_docs, 200)
                fetched = os.path.join(root, 'train-00000.bin')
                with open(fetched, 'wb') as f:
                    f.write(b'fetched')
                d2l.fwedu_shards('4b', fetch=False)  # same sha1: kept
                self.assertTrue(os.path.exists(fetched))
                d2l.DATA_HUB['lm-fwedu-manifest-4b-v1'] = ('u', 'two')
                d2l.fwedu_shards('4b', fetch=False)  # new sha1: afresh
                self.assertFalse(os.path.exists(fetched))
                with open(os.path.join(root, '.extracted-'
                                       'lm-fwedu-manifest-4b-v1')) as f:
                    self.assertEqual(f.read(), 'two')
        finally:
            os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
