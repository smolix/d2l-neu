# Tokenizers at Scale: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_tokenizers` explains in a sentence each but does not show:

- the reference tokenizer's special tokens and its loader (`R2_SPECIALS`,
  `_r2_from_file`);
- special-string-safe batched encoding for the three tokenizer kinds the
  shards accept (`_tok_info`, `_doc_token_ids`);
- the shard writer, the meta table with its fixed column types, and the
  checked download of one shard file (`_SHARD_EXT`, `FWEDU_META_COLUMNS`,
  `_ShardWriter`, `_meta_table`, `_fetch_shard_file`);
- URL normalization for the validation split's deduplication
  (`normalize_url`);
- the bytes behind every token id of the four tokenizer kinds the section
  compares, its byte-fallback ids, and the prefix of a byte-level BPE
  (`_byte_level_decoder`, `_vocab_bytes`, `_fallback_ids`, `_bpe_prefix`);
- SentencePiece training through a temporary file and its pieces as a
  log-probability table (`sp_train`, `_sp_fingerprint`, `_sp_pieces_scores`);
- the log-space addition of the unigram lattice (`_logaddexp`);
- vectorized token n-gram hashes for the train-validation overlap
  (`_rolling_ngram_hashes`).

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble (`tokenizers`, `transformers`, `pyarrow`, `urllib`) inside
its functions: a module-level import would join the `_preamble` shard and
move the freshness fingerprint of every notebook in the book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_tokenizers`

```{.python .input #tokenizers-lib-specials}
%%tab pytorch
#@save
R2_SPECIALS = (['<|endoftext|>', '<|pad|>', '<|im_start|>', '<|im_end|>']
               + [f'<|reserved_{i}|>' for i in range(4, 32)])

#@save
def _r2_from_file(path, raw=False, sha256=None):
    """A reference-format tokenizer.json, its bytes checked against sha256
    (if given): the Transformers wrapper (EOS id 0, pad id 1, no BOS,
    special strings in text kept as text), or with raw=True the
    tokenizers.Tokenizer. A fresh object on every call."""
    with open(path, 'rb') as f:
        data = f.read()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f'{path}: SHA-256 {hashlib.sha256(data).hexdigest()}'
                         f', expected {sha256}')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(data.decode('utf-8'))
    tok.encode_special_tokens = True
    if raw:
        return tok
    import transformers
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token='<|endoftext|>', pad_token='<|pad|>',
        split_special_tokens=True, model_max_length=4096)
```

```{.python .input #tokenizers-lib-encode}
%%tab pytorch
#@save
def _tok_info(tokenizer):
    """(vocabulary size, <|endoftext|> id or None, ids of special tokens)
    of a Transformers tokenizer, a tokenizers.Tokenizer or a tiktoken
    encoding."""
    if hasattr(tokenizer, 'encode_ordinary'):  # tiktoken
        return (tokenizer.n_vocab, d2l._lm_boundary_id(tokenizer),
                {tokenizer.encode_single_token(s)
                 for s in tokenizer.special_tokens_set})
    if hasattr(tokenizer, 'all_special_ids'):  # Transformers
        special = set(tokenizer.all_special_ids) | {
            i for i, t in tokenizer.added_tokens_decoder.items() if t.special}
        return len(tokenizer), d2l._lm_boundary_id(tokenizer), special
    special = {i for i, t in tokenizer.get_added_tokens_decoder().items()
               if t.special}
    return (tokenizer.get_vocab_size(with_added_tokens=True),
            tokenizer.token_to_id('<|endoftext|>'), special)

#@save
def _doc_token_ids(tokenizer, texts):
    """Token ids of each document text, no special tokens added, and a
    special-token string inside a text (a page about <|endoftext|>)
    encoded as ordinary text, for a Transformers tokenizer, a
    tokenizers.Tokenizer or a tiktoken encoding."""
    texts = [str(t) for t in texts]
    if not texts:
        return []
    if hasattr(tokenizer, 'encode_ordinary_batch'):  # tiktoken
        return tokenizer.encode_ordinary_batch(texts)
    if hasattr(tokenizer, 'all_special_ids'):  # Transformers
        return tokenizer(texts, add_special_tokens=False,
                         split_special_tokens=True,  # no warning about
                         verbose=False)['input_ids']  # long documents
    old = tokenizer.encode_special_tokens
    tokenizer.encode_special_tokens = True
    try:
        return [e.ids for e in tokenizer.encode_batch(
            texts, add_special_tokens=False)]
    finally:
        tokenizer.encode_special_tokens = old
```

```{.python .input #tokenizers-lib-shards}
%%tab pytorch
#@save
_SHARD_EXT = {'bin': '.bin', 'idx': '.idx', 'meta': '.meta.parquet'}

#@save
FWEDU_META_COLUMNS = (  # (name, arrow type) of the FineWeb-Edu meta columns
    ('url', 'string'), ('dump', 'string'), ('bucket', 'int16'),
    ('edu_score', 'float32'), ('int_score', 'int8'), ('gpt2_tokens', 'int32'),
    ('src_file', 'string'), ('src_row', 'int64'))

#@save
def _meta_table(rows):
    """The meta rows of one shard as a pyarrow Table: doc_id and n_tokens
    first, then the FineWeb-Edu columns present, in their fixed order and
    types, then any other column in order of first appearance."""
    import pyarrow as pa
    known = dict([('doc_id', 'string'), ('n_tokens', 'int32')]
                 + list(d2l.FWEDU_META_COLUMNS))
    seen = dict.fromkeys(k for r in rows for k in r)
    cols = [c for c in known if c in seen] + [c for c in seen
                                              if c not in known]
    return pa.table({c: pa.array([r.get(c) for r in rows],
                                  type=getattr(pa, known[c])()
                                  if c in known else None) for c in cols})

#@save
class _ShardWriter:
    """One shard being written: ids appended to name.bin as they come,
    offsets and meta rows kept until close() writes name.idx and
    name.meta.parquet and returns the shard's manifest entry."""
    def __init__(self, out_dir, name, dtype):
        self.out_dir, self.name = out_dir, name
        self.dtype = np.dtype(dtype).newbyteorder('<')
        self.f = open(os.path.join(out_dir, name + '.bin'), 'wb')
        self.offsets, self.rows, self.n_tokens = [0], [], 0

    def add(self, doc_id, ids, extra):
        self.f.write(np.asarray(ids, dtype=self.dtype).tobytes())
        self.n_tokens += len(ids)
        if self.n_tokens >= 2**32:
            raise ValueError(f'{self.name}: offsets overflow uint32')
        self.offsets.append(self.n_tokens)
        self.rows.append(dict(doc_id=str(doc_id), n_tokens=len(ids),
                              **(extra or {})))

    def close(self):
        import pyarrow.parquet as pq
        self.f.close()
        base = os.path.join(self.out_dir, self.name)
        np.asarray(self.offsets, dtype='<u4').tofile(base + '.idx')
        pq.write_table(d2l._meta_table(self.rows), base + '.meta.parquet',
                       compression='zstd', compression_level=3,
                       row_group_size=65536)
        files = {k: base + e for k, e in d2l._SHARD_EXT.items()}
        return dict(file=self.name, n_tokens=self.n_tokens,
                    n_docs=len(self.rows),
                    sha256={k: d2l._sha256_file(p) for k, p in files.items()},
                    bytes={k: os.path.getsize(p) for k, p in files.items()})

#@save
def _fetch_shard_file(shard, ext, path):
    """Download one file of a shard from the manifest's url, checking its
    sha1 and SHA-256 before it is moved into place. The partial file is
    named per process, so ranks that fetch the same file at once never
    write into each other's download."""
    tmp = f'{path}.{os.getpid()}.part'
    sha1, sha256 = hashlib.sha1(), hashlib.sha256()
    with requests.get(shard['url'][ext], stream=True, timeout=60) as r, \
            open(tmp, 'wb') as f:
        r.raise_for_status()
        for c in r.iter_content(1 << 20):
            f.write(c)
            sha1.update(c)
            sha256.update(c)
    if (sha1.hexdigest(), sha256.hexdigest()) != (shard['sha1'][ext],
                                                  shard['sha256'][ext]):
        os.remove(tmp)
        raise ValueError(f'{shard["url"][ext]}: hash differs from the '
                         'manifest')
    os.replace(tmp, path)
```

```{.python .input #tokenizers-lib-url}
%%tab pytorch
#@save
def normalize_url(url):
    """A URL in the form the validation split deduplicates on: scheme and
    host lower-cased, a leading www. removed, the fragment dropped, a
    trailing / of the path removed, the query kept."""
    import urllib.parse
    p = urllib.parse.urlsplit(url.strip())
    host = p.netloc.lower()
    if host.startswith('www.'):
        host = host[4:]
    return urllib.parse.urlunsplit((p.scheme.lower(), host,
                                    p.path.rstrip('/'), p.query, ''))
```

```{.python .input #tokenizers-lib-vocab}
%%tab pytorch
#@save
def _byte_level_decoder():
    """GPT-2's byte-level alphabet inverted: the printable character that
    stands for each byte -> the byte (the alphabet of byte-level BPE)."""
    bs = (list(range(33, 127)) + list(range(161, 173))
          + list(range(174, 256)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

#@save
def _vocab_bytes(tokenizer):
    """The UTF-8 bytes behind every token id (None for special and unused
    ids) of a tiktoken encoding, a Transformers tokenizer, a
    tokenizers.Tokenizer or a SentencePieceProcessor."""
    if hasattr(tokenizer, 'decode_single_token_bytes'):  # tiktoken
        out = [None] * tokenizer.n_vocab
        special = set(tokenizer.special_tokens_set)
        for i in range(tokenizer.n_vocab):
            try:
                b = tokenizer.decode_single_token_bytes(i)
            except KeyError:
                continue
            if b.decode('utf-8', 'replace') not in special:
                out[i] = b
        return out
    if hasattr(tokenizer, 'id_to_piece'):  # SentencePiece
        out = []
        for i in range(tokenizer.get_piece_size()):
            piece = tokenizer.id_to_piece(i)
            if tokenizer.is_byte(i):
                out.append(bytes([int(piece[3:5], 16)]))
            elif tokenizer.is_control(i) or tokenizer.is_unknown(i):
                out.append(None)
            else:
                out.append(piece.replace('▁', ' ').encode('utf-8'))
        return out
    raw = getattr(tokenizer, 'backend_tokenizer', tokenizer)
    vocab = raw.get_vocab(with_added_tokens=True)
    special = {t.content for t in raw.get_added_tokens_decoder().values()
               if t.special}
    added = {t.content for t in raw.get_added_tokens_decoder().values()}
    byte_level, dec = 'ByteLevel' in raw.to_str(), _byte_level_decoder()
    out = [None] * (max(vocab.values()) + 1)
    for piece, i in vocab.items():
        if piece in special:
            continue
        if byte_level and piece not in added:
            out[i] = bytes(dec[c] for c in piece)
        else:
            out[i] = piece.replace('▁', ' ').encode('utf-8')
    return out

#@save
def _fallback_ids(tokenizer):
    """Ids that spell a character byte by byte: SentencePiece's byte pieces
    and, in a byte-level vocabulary, the single bytes that are no
    character on their own (0x80 and above)."""
    if hasattr(tokenizer, 'is_byte'):
        return {i for i in range(tokenizer.get_piece_size())
                if tokenizer.is_byte(i)}
    return {i for i, b in enumerate(_vocab_bytes(tokenizer))
            if b is not None and len(b) == 1 and b[0] >= 0x80}

#@save
def _bpe_prefix(tok, V):
    """The byte-level BPE `tok` (a tokenizers.Tokenizer) cut to its first
    V ids: the vocabulary entries below V and the merges that make them.
    Encoding with it is encoding with the merges learned first (the prefix
    property of BPE)."""
    from tokenizers import Tokenizer
    spec = json.loads(tok.to_str())
    vocab = spec['model']['vocab']
    spec['model']['vocab'] = {p: i for p, i in vocab.items() if i < V}
    spec['model']['merges'] = [m for m in spec['model']['merges']
                               if vocab[''.join(m)] < V]
    spec['added_tokens'] = [t for t in spec['added_tokens'] if t['id'] < V]
    out = Tokenizer.from_str(json.dumps(spec))
    out.encode_special_tokens = True
    return out
```

```{.python .input #tokenizers-lib-sp}
%%tab pytorch
#@save
def sp_train(texts, model_type, vocab_size, **kw):
    """A SentencePiece model trained on texts (one input line per line of
    text) through a temporary file, as a SentencePieceProcessor. Defaults:
    no normalization (the text is NFC already), whitespace kept as is,
    byte fallback instead of <unk>, digits split, character_coverage
    0.9995, 16 threads (the pieces depend on the thread count)."""
    import io
    import tempfile
    import sentencepiece as spm
    args = dict(model_type=model_type, vocab_size=vocab_size,
                normalization_rule_name='identity',
                remove_extra_whitespaces=False, byte_fallback=True,
                split_digits=True, character_coverage=0.9995,
                num_threads=16, max_sentence_length=1 << 16,
                minloglevel=2)
    args.update(kw)
    model = io.BytesIO()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'input.txt')
        with open(path, 'w', encoding='utf-8') as f:
            for t in texts:
                f.writelines(line + '\n' for line in t.split('\n')
                             if line.strip())
        spm.SentencePieceTrainer.train(input=path, model_writer=model,
                                       **args)
    return spm.SentencePieceProcessor(model_proto=model.getvalue())

#@save
def _sp_fingerprint(sp):
    """SHA-256 of a SentencePiece model's pieces and scores in id order
    (the serialized model also records the training file's path)."""
    rows = [(sp.id_to_piece(i), round(sp.get_score(i), 6))
            for i in range(sp.get_piece_size())]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode(
        'utf-8')).hexdigest()

#@save
def _sp_pieces_scores(sp):
    """A unigram SentencePiece model as viterbi_segment's input: the
    log-probabilities of its ordinary pieces, and the score SentencePiece
    gives a character that is no piece (the lowest score minus 10)."""
    logp = {sp.id_to_piece(i): sp.get_score(i)
            for i in range(sp.get_piece_size())
            if not (sp.is_byte(i) or sp.is_control(i) or sp.is_unknown(i))}
    return logp, min(logp.values()) - 10.0
```

```{.python .input #tokenizers-lib-ngrams}
%%tab pytorch
#@save
def _logaddexp(a, b):
    """log(exp(a) + exp(b)) for Python floats, -inf allowed."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-math.fabs(a - b)))

#@save
def _rolling_ngram_hashes(ids, n, eos=None):
    """64-bit polynomial hashes of every window of n consecutive token ids
    (Horner's rule, arithmetic mod 2^64, vectorized over the windows); a
    window that contains eos gets no hash (its slot is masked). Returns
    (hashes, valid), both of length len(ids) - n + 1. Equal windows hash
    equally; a match is confirmed on the ids themselves."""
    x = np.asarray(ids).astype(np.uint64) + np.uint64(1)
    m = len(x) - n + 1
    if m <= 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    h = np.zeros(m, dtype=np.uint64)
    base = np.uint64(0x9E3779B97F4A7C15)
    for k in range(n):
        h *= base
        h += x[k:k + m]
    valid = np.ones(m, dtype=bool)
    if eos is not None:
        c = np.concatenate([[0], np.cumsum(np.asarray(ids) == eos)])
        valid = c[n:n + m] == c[:m]
    return h, valid
```
