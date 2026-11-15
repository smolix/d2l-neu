# Data Mixtures: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_data_mixtures` explains in a sentence each but does not
show: the token pools of 20.4-20.6 (`TokenPool`, `load_pool`), the byte
length of each token for bits per byte (`token_byte_lengths`), and the
FineWeb-Edu `sample-10BT` file order of the chapter (`_FWEDU_FILES`).

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble inside its functions: a module-level import would join the
`_preamble` shard and move the freshness fingerprint of every notebook in the
book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_data_mixtures`

```{.python .input #mixtures-lib-files}
%%tab pytorch
#@save
_FWEDU_FILES = tuple(  # sample/10BT files read, in order; 005, 012, 013 are chapter 21's
    f'sample/10BT/{k:03d}_00000.parquet' for k in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11))
```

```{.python .input #mixtures-lib-pools}
%%tab pytorch
#@save
TokenPool = collections.namedtuple('TokenPool', 'ids meta')

#@save
def load_pool(path_or_key, name=None):
    """A token pool: a flat array of token ids (read-only memmap; every
    document ends in EOS) and its meta dict. path_or_key is a DATA_HUB key
    (a tar of <name>.bin files and their meta.json), a directory or a .bin
    file. meta.json describes one pool, or maps each .bin file name to its
    meta when a directory holds several (then name picks one)."""
    import pathlib
    p = path_or_key
    if isinstance(p, str) and p in d2l.DATA_HUB:
        tar = d2l.download_keyed(p)
        p = pathlib.Path(tar).with_suffix('')
        if not (p / '.extracted').exists():
            p.mkdir(exist_ok=True)
            d2l.extract(tar, str(p))
            (p / '.extracted').touch()
    p = pathlib.Path(p)
    if p.is_dir():
        bins = sorted(b for b in p.glob('*.bin')
                      if name is None or b.stem == name)
        if len(bins) != 1:
            raise ValueError(f'{p}: {len(bins)} pools match name={name!r} '
                             f'({[b.stem for b in p.glob("*.bin")]})')
        p = bins[0]
    with open(p.parent / 'meta.json') as f:
        meta = json.load(f)
    meta = meta.get(p.name, meta)
    ids = np.memmap(p, dtype=meta.get('dtype', 'uint16'), mode='r')
    if 'n_tokens' in meta and len(ids) != meta['n_tokens']:
        raise ValueError(f'{p}: {len(ids)} tokens, meta says '
                         f'{meta["n_tokens"]}')
    return d2l.TokenPool(ids, meta)

#@save
def token_byte_lengths(tokenizer):
    """np.ndarray of the UTF-8 bytes each token id stands for (0 for
    special tokens): the denominator of bits per byte. Takes a Transformers
    tokenizer, a tokenizers.Tokenizer or a tiktoken encoding."""
    if hasattr(tokenizer, 'decode_single_token_bytes'):  # tiktoken
        n = tokenizer.n_vocab
        special = set(tokenizer.special_tokens_set)
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            try:
                b = tokenizer.decode_single_token_bytes(i)
            except KeyError:  # an unused id
                continue
            if b.decode('utf-8', 'replace') not in special:
                out[i] = len(b)
        return out
    raw = getattr(tokenizer, 'backend_tokenizer', tokenizer)
    vocab = raw.get_vocab(with_added_tokens=True)
    special = {t.content for t in raw.get_added_tokens_decoder().values()
               if t.special}
    byte_level = 'ByteLevel' in raw.to_str()  # one character per byte
    out = np.zeros(max(vocab.values()) + 1, dtype=np.int64)
    for piece, i in vocab.items():
        if piece in special:
            continue
        if byte_level:
            out[i] = len(piece)
        elif re.fullmatch(r'<0x[0-9A-Fa-f]{2}>', piece):  # byte fallback
            out[i] = 1
        else:  # SentencePiece-style: U+2581 marks a space
            out[i] = len(piece.replace('▁', ' ').encode('utf-8'))
    return out
```
