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
  (`normalize_url`).

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
                         split_special_tokens=True)['input_ids']
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
FWEDU_META_COLUMNS = (  # (name, arrow type): the FineWeb-Edu meta columns, in order
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
    sha1 and then its SHA-256 before it is moved into place."""
    tmp = path + '.part'
    with requests.get(shard['url'][ext], stream=True, timeout=60) as r, \
            open(tmp, 'wb') as f:
        r.raise_for_status()
        for c in r.iter_content(1 << 20):
            f.write(c)
    with open(tmp, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    if sha1 != shard['sha1'][ext] or (d2l._sha256_file(tmp)
                                      != shard['sha256'][ext]):
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
