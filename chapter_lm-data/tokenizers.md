# Tokenizers at Scale
:label:`sec_lm_tokenizers`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "3.6 Tokenizers at Scale · T0").
Blueprint: docs/llm/blueprints/ch03_data.md section 3.6.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Vocabulary size is a compute choice, and the unigram model is the alternative to merges.
**Opening:** The byte-level BPE of `sec_text-sequence` splits a Hindi sentence into three times the tokens of its English translation, so the same sentence costs three times the compute and fits a third as much into the context. The section builds the alternative to merges, measures what one tokenizer costs each language, chooses a vocabulary size from a cost model, freezes the tokenizer the reference model uses, and closes on the question of extending a vocabulary after training.
**Teaches:**
- The unigram model: piece probabilities estimated by expectation-maximization with pruning, Viterbi segmentation (`d2l.viterbi_segment`), subword regularization by sampling segmentations; SentencePiece as the production implementation; the one difference from merges (a probabilistic lexicon instead of a merge order).
- Vocabulary size against embedding and softmax cost and against sequence length: the elasticity of compression in V and the cost model that picks V for a parameter budget; the vocabulary fit of released models.
- Fertility and the tokenization premium by language (`d2l.fertility_table`) on parallel passages, and bits per character as the comparison that is fair across tokenizers.
- Extension of a vocabulary with new pieces and initialization of the new embedding rows (random, mean, decomposition over existing pieces; `d2l.extend_vocab`), with the softmax mass of the new rows at step zero as the check.
- The reference tokenizer (byte-level BPE, 32,768 pieces, Qwen3-style single-digit pre-tokenization, EOS id 0, no BOS, ids 0 to 31 reserved for specials) and the shard format Chapter 4 reads (`d2l.write_shards`, `d2l.TokenShards`).
**Derivation:** none.
**Notebook:**
1. Train SentencePiece unigram, SentencePiece BPE and byte-level BPE with the reference regex at a fixed vocabulary on a 10-language FineWeb-2 slice (5 MB per language, 90% train); implement Viterbi from scratch and check it against SentencePiece's segmentation on 1,000 sentences.
2. Compute fertility, bytes per token and the premium relative to English on parallel passages for the three tokenizers and for the GPT-2, o200k and Qwen3 tokenizers; plot bits per character per language.
3. Train one small `d2l.GPT` per tokenizer on the same 45 MB of text (a byte budget, so token counts differ and are recorded) with two seeds, and evaluate bits per byte per language on the held-out 10% and on the parallel passages.
4. Extend a small released decoder's vocabulary with Hindi pieces under three initializations and continue training briefly on Hindi; report Hindi bits per byte at step 0 and early steps, English retention, and the new tokens' softmax mass at step 0.
5. Verify that the freshly trained reference tokenizer reproduces the committed `tokenizer.json` hash; write a 100-million-token FineWeb-Edu shard with `d2l.write_shards` and read it back with `d2l.TokenShards`, checking EOS placement and the resume cursor.
6. Claim that survives noise: the tokenizer type (unigram against BPE at equal V) matters less for per-language bits per byte than the premium spread across languages does (differences smaller than the larger of the two seed ranges are not interpreted; passages are bootstrapped for the macro-average); informed initializations reach lower Hindi bits per byte at step 0 than random and the arms converge with training (step-0 values are deterministic and bootstrapped over passages).
7. Reused pieces: `d2l.BPETokenizer` and `GPT2_PATTERN` (`sec_text-sequence`, recapped in one paragraph), `d2l.GPT`, `d2l.train_proxy_lm`, `d2l.ProxyCfg`, `d2l.bpb_from_nll`, `d2l.temperature_weights` (3.4), `d2l.CorpusManifest`, `d2l.docs_hash` (3.1), `d2l.load_hf` (1.1); new `d2l.UnigramTokenizer`, `d2l.viterbi_segment`, `d2l.fertility_table`, `d2l.R2_SPLIT_REGEX`, `d2l.train_hf_bpe`, `d2l.r2_tokenizer`, `d2l.extend_vocab`, `d2l.write_shards`, `d2l.TokenShards`.
**Figures:** `fig_lm_fertility_by_language` (plot: bits per byte against the tokenization premium per language and tokenizer, with lines joining a language across tokenizers); `fig_lm_unigram_viterbi` (the lattice of segmentations of one word with piece log-probabilities and the best path).
**Restate / forward:** `subsec_fertility-digits-glitch` (fertility differs by script and digits chunk unevenly under a merge-based tokenizer); `sec_mdl-coding-compression` (bits per character converts a per-token loss into a tokenizer-free quantity, `eq_mdl-bpc`). Forward: 4.1 (the stream that reads these shards).
**Exercises:** 5 to 7. Load-bearing: derive the vocabulary size that balances embedding and softmax cost against sequence length for a stated parameter budget and compression elasticity (derive); measure digit chunking under the three tokenizers on a numeric passage and the effect on a small arithmetic probe (measure); check that `d2l.TokenShards.locate(i)` returns a cursor from which `iter_docs` resumes at document i, and that id 0 appears only at document ends (check); open: edit the merge table instead of appending pieces so the extended tokenizer stays BPE-consistent, and compare segmentation of mixed Hindi-English text.
**Writer notes:**
- The reference tokenizer is a one-way door (blueprint README Decision 1): byte-level BPE with vocabulary exactly 32,768; normalizer NFC; pre-tokenizer `Split(R2_SPLIT_REGEX, 'isolated')` then `ByteLevel(add_prefix_space=False, use_regex=False)` with Qwen3's regex (contractions, letters with an optional leading non-letter, single digits, punctuation runs, newlines, whitespace); specials `<|endoftext|>` = 0 (EOS and document delimiter; no BOS), `<|pad|>` = 1, `<|im_start|>` = 2, `<|im_end|>` = 3, `<|reserved_4|>` to `<|reserved_31|>` = 4 to 31 (so later chat, tool or thinking tokens need no retraining); merges start at id 32 in trainer order (do not assume id minus 32 is a byte value); trained on 200,000 train-bucket documents disjoint from validation; only `<|endoftext|>` occurs in pretraining, so a chapter that starts using a reserved id initializes its row first. Rejected alternative: reuse GPT-2's tokenizer (a 53% larger embedding for a 150M model and worse digit and whitespace handling).
- The shard format is normative for 4.1's `d2l.PackedStream` and 4.2: `manifest.json` in the `d2l-corpus/1` schema with tokenizer hash and format; `{split}-{k:05d}.bin` raw little-endian ids, `uint16` when every id is at most 65,535 and `uint32` otherwise (Qwen's 151,936 in 4.4 needs `uint32`), recorded in `format.dtype`; EOS appended after every document and nowhere else; documents never span shards; `{split}-{k:05d}.idx` as `uint32` offsets of length documents plus one; `.meta.parquet` for audits; `TokenShards.locate(i)` returns the resume cursor; `verify()` recomputes hashes; `fetch=True` downloads missing shards from the manifest's `DATA_HUB` entries, which pin a commit of a d2l-owned Hugging Face dataset repository (never `main`).
- Fertility and digit chunking are restated from `subsec_fertility-digits-glitch` and bits per character from `sec_mdl-coding-compression`, not re-measured on the GPT-2 tokenizer; the section's own content is the unigram model, the vocabulary cost model, extension and initialization, and the reference tokenizer and shards; FineWeb-2 is named as the multilingual corpus; the experiment with small LMs is T1 while the rest is T0 (the section's tier stays T0 in the table, with the T1 cells on committed results by default).
- Data: FineWeb-2 configs for nine languages plus FineWeb-Edu English, 5 MB each (ODC-By); parallel passages from `facebook/belebele` (CC BY-SA 4.0; the language codes differ from FineWeb-2's, for example `zho_Hans` against `cmn_Hani`; passages are translations of the same text, so token counts compare); Hindi documents from FineWeb-2 for the extension and 200 held-out English documents for retention; reference tokenizers GPT-2 and o200k through `tiktoken`, Qwen3-0.6B's `tokenizer.json`, and a small released decoder with a byte-level tokenizer for the extension arm.
- SentencePiece settings: `byte_fallback=True`, identity normalization after the notebook's own NFC (its default applies NFKC-like rules silently), `split_digits=True`, `character_coverage=0.9995`; the pure-Python `d2l.UnigramTokenizer` is the fallback if wheels fail.
- Pin the `tokenizers` library to a 0.x release (the blueprint pinned at least 0.21): the documentation of the 1.0 release candidate, read on 6 November 2026, says its bindings load and run a `tokenizer.json` but do not yet expose building a tokenizer from components, editing or training, which `d2l.train_hf_bpe` and the reference-tokenizer cell need.
- Pitfalls to state: comparing perplexity across tokenizers; counting bytes before normalization for one tokenizer and after for another; training the tokenizer on text the model never sees (glitch tokens); choosing V by habit; the number of new tokens is not the benefit; mean initialization makes every new row identical and can hijack the softmax at step 0 (check the mass); forgetting the output matrix when embeddings are untied and `config.vocab_size`; adding pieces that contain ASCII changes English segmentation; `uint16` with V over 65,536; EOS inside a document.
- Hosting: the 4-billion-token shard set is about 8 GB; if the bucket cannot host it, ship the manifest and build script and host 1 billion tokens.
- Committed artifacts: `lm-r2-tokenizer` (`tokenizer.json`, 2 to 3 MB), `lm-fwedu-val` (at least 2^23 tokens plus the NFC text), `lm-fwedu-shards-100m`, `lm-fwedu-manifest-4b` (40 shards of 10^8 tokens with URLs and hashes), the two result tables as JSON.
- Do not re-teach: the BPE merge rule and pre-tokenization (`sec_text-sequence`), the code-length view of a unigram model beyond one sentence (`sec_mdl-information_theory`), the GPT-2 fertility table.
**Sources:**
- Papers: [SentencePiece](https://aclanthology.org/D18-2012/) (the unigram and BPE trainer with byte fallback); [Subword Regularization](https://aclanthology.org/P18-1007/) (the unigram model, Viterbi and sampled segmentations); [Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623) (optimal vocabulary size as a function of compute: the cost model); [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) (the tokenization premium by language); [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://arxiv.org/abs/2004.03720) (unigram against BPE in their setting, cited with that limit); [WECHSEL](https://aclanthology.org/2022.naacl-main.293/) (embedding initialization for a new vocabulary by decomposition over the old); [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) (BPE's origin, one sentence); [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) (the tokenizer-free alternative, one pointer to Chapter 14's Further Reading).
- Code and documentation: [tokenizers](https://huggingface.co/docs/tokenizers/index) and its [pipeline guide](https://huggingface.co/docs/tokenizers/pipeline) (normalizer, pre-tokenizer, model, decoder: the reference tokenizer is built with these); [tokenization algorithms summary](https://huggingface.co/docs/transformers/tokenizer_summary); [SentencePiece](https://github.com/google/sentencepiece) (the trainer and its normalization defaults); [minbpe](https://github.com/karpathy/minbpe) (the from-scratch BPE oracle); [FineWeb-2 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (the per-language slices).
- Courses, tutorials and blogs: [Hugging Face LLM Course, chapter 6](https://huggingface.co/learn/llm-course/en/chapter6/1) (training a tokenizer with the library); [CS336](https://cs336.stanford.edu/) (the tokenization lecture and assignment A1's tokenizer).
-->

<!-- Placeholder (stage 2, infra step): the cells below are the section's
shared library, written before the section; the prose around them comes with
the section. Cell ids follow the section specification. -->

```{.python .input #tokenizers-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #tokenizers-extras}
%%tab pytorch
import json
import os
import pandas as pd
```

<!-- Placeholder (stage 2): the reference tokenizer's pipeline (chapter spec 1.6a): NFC, Qwen3's split pattern with single digits, byte-level BPE, 32 special tokens first. -->

```{.python .input #tokenizers-bpe-def}
%%tab pytorch
#@save
R2_SPLIT_REGEX = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+"""
                  r"""|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)"""
                  r"""|\s+""")

#@save
def train_hf_bpe(texts, vocab_size, specials, regex=R2_SPLIT_REGEX,
                 min_frequency=2):
    """A byte-level BPE (Hugging Face tokenizers) trained on texts: NFC,
    the split regex, bytes as 256 base symbols, the special tokens first
    (ids 0, 1, ...), no BOS or EOS added on encode. Special-token strings
    inside a text are encoded as text (encode_special_tokens is set; the
    flag is not saved in tokenizer.json, so loaders set it again)."""
    from tokenizers import (Regex, Tokenizer, decoders, models, normalizers,
                            pre_tokenizers, trainers)
    tok = Tokenizer(models.BPE(dropout=None, byte_fallback=False,
                               unk_token=None))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(regex), behavior='isolated', invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False,
                                 use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=list(specials), show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(f'trained {tok.get_vocab_size()} pieces, not '
                         f'{vocab_size}: too little text for this vocabulary')
    tok.encode_special_tokens = True
    return tok
```

<!-- Placeholder (stage 2): the frozen reference tokenizer, checked by SHA-256 and wrapped for Transformers; special strings in a document stay text. -->

```{.python .input #tokenizers-r2-tok-def}
%%tab pytorch
#@save
R2_TOKENIZER_SHA256 = None  # SHA-256 of tokenizer.json; set when it is built

#@save
def r2_tokenizer(raw=False):
    """The reference tokenizer (artifact lm-r2-tokenizer-v1; its file is
    checked against R2_TOKENIZER_SHA256): a Transformers
    PreTrainedTokenizerFast with EOS <|endoftext|> = 0, pad <|pad|> = 1,
    no BOS, and special-token strings in text encoded as text; raw=True
    gives the tokenizers.Tokenizer itself."""
    if 'lm-r2-tokenizer-v1' not in d2l.DATA_HUB or not d2l.R2_TOKENIZER_SHA256:
        raise KeyError('lm-r2-tokenizer-v1 is not registered yet: it is '
                       'built by tools/build_r2_shards.py')
    return d2l._r2_from_file(d2l.download_keyed('lm-r2-tokenizer-v1'),
                             raw=raw, sha256=d2l.R2_TOKENIZER_SHA256)
```

<!-- Placeholder (stage 2): shards are flat token files with document offsets and a metadata table, described by a d2l-corpus/1 manifest (chapter spec 1.6b). -->

```{.python .input #tokenizers-shards-def}
%%tab pytorch
#@save
def write_shards(docs, tokenizer, out_dir, split, shard_tokens=100_000_000,
                 eos_id=None, meta_fn=None, manifest=None, *, batch_docs=1000):
    """Tokenize documents into shards out_dir/{split}-{k:05d}.bin|.idx|
    .meta.parquet and return the split's manifest entry. docs yields
    (doc_id, text) or (doc_id, text, meta); each document becomes its ids
    and one EOS (eos_id, default the tokenizer's <|endoftext|>); special
    strings in a text stay text, and a special id inside a document is an
    error. A shard closes at the first document boundary at or after
    shard_tokens; ids are uint16 for vocabularies up to 65,536, else uint32.
    meta_fn(doc_id, text, meta) gives a document's extra meta columns
    (default: meta itself). The entry is stored in manifest (a
    CorpusManifest, created if None), which is saved as
    out_dir/manifest.json."""
    import itertools
    os.makedirs(out_dir, exist_ok=True)
    vocab, eos, special = d2l._tok_info(tokenizer)
    eos = eos if eos_id is None else eos_id
    if eos is None:
        raise ValueError('the tokenizer has no <|endoftext|>: pass eos_id')
    dtype = np.dtype('uint16' if vocab <= 2**16 else 'uint32')
    special = np.array(sorted(special | {eos}), dtype=np.int64)
    shards, w, docs = [], None, iter(docs)
    while batch := list(itertools.islice(docs, batch_docs)):
        batch = [(d[0], d[1], d[2] if len(d) > 2 else None) for d in batch]
        for (doc_id, text, meta), ids in zip(
                batch, d2l._doc_token_ids(tokenizer, [b[1] for b in batch])):
            ids = np.asarray(ids, dtype=np.int64)
            bad = np.isin(ids, special)
            if bad.any():
                raise ValueError(f'document {doc_id}: special id '
                                 f'{int(ids[bad][0])} inside the text')
            if w is None:
                w = d2l._ShardWriter(out_dir, f'{split}-{len(shards):05d}',
                                     dtype)
            extra = meta_fn(doc_id, text, meta) if meta_fn else (meta or {})
            w.add(doc_id, np.append(ids, eos).astype(dtype), extra)
            if w.n_tokens >= shard_tokens:
                shards.append(w.close())
                w = None
    if w is not None:
        shards.append(w.close())
    entry = dict(n_tokens=sum(s['n_tokens'] for s in shards),
                 n_docs=sum(s['n_docs'] for s in shards), shards=shards)
    manifest = manifest or d2l.CorpusManifest(os.path.basename(
        os.path.normpath(out_dir)))
    manifest.splits = dict(manifest.splits or {}, **{split: entry})
    fmt = dict(dtype=dtype.name, eos_id=int(eos), endianness='little',
               doc_end='eos', idx_dtype='uint32',
               meta_required=['doc_id', 'n_tokens'])
    if manifest.format not in (None, fmt):
        raise ValueError(f'manifest format {manifest.format} != {fmt}')
    manifest.format = fmt
    manifest.tokenizer = manifest.tokenizer or dict(vocab_size=vocab)
    manifest.save(os.path.join(out_dir, 'manifest.json'))
    return entry

#@save
class TokenShards:
    """Read-only access to a shard set written by write_shards: documents
    by global index, token ranges by global position, per-shard meta, and
    verify(), which recomputes every file's SHA-256 and the EOS and offset
    invariants. max_tokens keeps the smallest prefix of shards holding at
    least that many tokens; fetch=True downloads missing files from the
    manifest's url and checks sha1, then SHA-256."""
    def __init__(self, root, split='train', fetch=False, max_tokens=None):
        self.root, self.split, self.fetch = root, split, fetch
        with open(os.path.join(root, 'manifest.json'), encoding='utf-8') as f:
            self.manifest = json.load(f)
        shards = self.manifest['splits'][split]['shards']
        if max_tokens is not None:
            total = np.cumsum([s['n_tokens'] for s in shards])
            shards = shards[:int(np.searchsorted(total, max_tokens)) + 1]
        self.shards = shards
        fmt = self.manifest['format']
        self.dtype, self.eos_id = np.dtype(fmt['dtype']), fmt['eos_id']
        self.vocab_size = (self.manifest.get('tokenizer') or {}).get(
            'vocab_size')
        self._docs = np.cumsum([0] + [s['n_docs'] for s in shards])
        self._toks = np.cumsum([0] + [s['n_tokens'] for s in shards])
        self.n_docs, self.n_tokens = int(self._docs[-1]), int(self._toks[-1])
        self.n_shards = len(shards)
        self._mm, self._idx = {}, {}

    def _path(self, k, ext):
        s = self.shards[k]
        path = os.path.join(self.root, s['file'] + d2l._SHARD_EXT[ext])
        if not os.path.exists(path):
            if not self.fetch or 'url' not in s:
                raise FileNotFoundError(path)
            d2l._fetch_shard_file(s, ext, path)
        return path

    def shard(self, k):
        if k not in self._mm:
            self._mm[k] = np.memmap(self._path(k, 'bin'), dtype=self.dtype,
                                    mode='r')
        return self._mm[k]

    def offsets(self, k):
        if k not in self._idx:
            self._idx[k] = np.fromfile(self._path(k, 'idx'), dtype='<u4')
        return self._idx[k]

    def locate(self, i):
        """(shard, document within it, offset of its first token there)."""
        if not 0 <= i < self.n_docs:
            raise IndexError(i)
        k = int(np.searchsorted(self._docs, i, side='right')) - 1
        j = i - int(self._docs[k])
        return k, j, int(self.offsets(k)[j])

    def locate_token(self, t):
        """(shard, position within it) of global token position t."""
        if not 0 <= t < self.n_tokens:
            raise IndexError(t)
        k = int(np.searchsorted(self._toks, t, side='right')) - 1
        return k, t - int(self._toks[k])

    def doc(self, i):
        k, j, a = self.locate(i)
        return np.asarray(self.shard(k)[a:self.offsets(k)[j + 1]],
                          dtype=np.int32)

    def iter_docs(self, start=0):
        for i in range(start, self.n_docs):
            yield i, self.doc(i)

    def read(self, start, n):
        """n tokens from global position start, across shard boundaries."""
        if start < 0 or start + n > self.n_tokens:
            raise IndexError((start, n))
        out = []
        while n > 0:
            k, a = self.locate_token(start)
            take = min(n, self.shards[k]['n_tokens'] - a)
            out.append(np.asarray(self.shard(k)[a:a + take], dtype=np.int32))
            start, n = start + take, n - take
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int32)

    def meta(self, k):
        return pd.read_parquet(self._path(k, 'meta'))

    def verify(self):
        """Recompute every file's SHA-256 and check that offsets start at 0,
        increase and end at the shard's token count, that EOS ends every
        document and occurs nowhere else, and that the meta rows match."""
        for k, s in enumerate(self.shards):
            for ext in d2l._SHARD_EXT:
                path = self._path(k, ext)
                if d2l._sha256_file(path) != s['sha256'][ext]:
                    raise ValueError(f'{path}: SHA-256 differs from the '
                                     'manifest')
            ids, idx = self.shard(k), self.offsets(k)
            name = s['file']
            if (len(idx) != s['n_docs'] + 1 or idx[0] != 0
                    or idx[-1] != len(ids) or len(ids) != s['n_tokens']
                    or (np.diff(idx.astype(np.int64)) <= 0).any()):
                raise ValueError(f'{name}.idx: offsets do not match the '
                                 'shard')
            if (np.asarray(ids[idx[1:].astype(np.int64) - 1]) != self.eos_id
                    ).any() or int((ids == self.eos_id).sum()) != s['n_docs']:
                raise ValueError(f'{name}.bin: EOS not exactly at the '
                                 'document ends')
            m = self.meta(k)
            if len(m) != s['n_docs'] or (m['n_tokens'].to_numpy()
                                         != np.diff(idx)).any():
                raise ValueError(f'{name}.meta.parquet: rows do not match '
                                 'the offsets')
        return True

#@save
def fwedu_shards(which='100m', fetch=True, max_tokens=None):
    """The reference shards of FineWeb-Edu under the reference tokenizer:
    which='val' (2^23 validation tokens), '100m' (train-00000, 10^8 tokens)
    or '4b' (the 40 training shards, downloaded one by one when first read).
    Each lives in ../data/lm/fwedu-r2-v1/<which>/."""
    keys = {'val': 'lm-fwedu-val-v1', '100m': 'lm-fwedu-shards-100m-v1',
            '4b': 'lm-fwedu-manifest-4b-v1'}
    if which not in keys:
        raise ValueError(f'which is one of {tuple(keys)}')
    key = keys[which]
    if key not in d2l.DATA_HUB:
        raise KeyError(f'{key} is not registered yet: it is built by '
                       'tools/build_r2_shards.py')
    root = os.path.join('..', 'data', 'lm', 'fwedu-r2-v1', which)
    os.makedirs(root, exist_ok=True)
    path = d2l.download_keyed(key)
    if which == '4b':
        import shutil
        shutil.copyfile(path, os.path.join(root, 'manifest.json'))
    elif not os.path.exists(os.path.join(root, '.extracted-' + key)):
        d2l.extract(path, root)
        open(os.path.join(root, '.extracted-' + key), 'w').close()
    return d2l.TokenShards(root, 'val' if which == 'val' else 'train',
                           fetch=fetch, max_tokens=max_tokens)
```

