# Legacy Tokenizer Shims (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`.

It keeps two methods that the retired English-French translator of
`seq2seq.md` attached to `d2l.BPETokenizer` (2026-11-19, when the 2026-11
pass cut the translator): `tok['<pad>']` maps a special token to its id and
`to_tokens` maps ids to token strings. Nothing in the book calls them. They
are merged into the `BPETokenizer` class block
(`d2l/_blocks/<fw>/BPETokenizer.py`), on which every notebook that uses the
tokenizer or the shared test models depends, so dropping them would mark
about 50 notebooks (PyTorch and JAX, 2026-11) stale with no code change
(docs/build-system.md §3.4). The cell is byte-for-byte the cell of its
original home and is appended after every chapter, as it was, so the block
is unchanged. Delete the file at the next full re-execution of the book.
-->

```{.python .input #seq2seq-vocab-shims}
%%tab pytorch, jax
@d2l.add_to_class(d2l.BPETokenizer)  #@save
def __getitem__(self, token):
    return self.specials[token]

@d2l.add_to_class(d2l.BPETokenizer)  #@save
def to_tokens(self, ids):
    return [self.decode([int(i)]) for i in ids]
```
