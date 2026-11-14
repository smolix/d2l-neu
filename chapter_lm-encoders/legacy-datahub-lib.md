# Legacy Dataset Registrations (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`.

It keeps the `d2l.DATA_HUB` entries that the two retired NLP chapters
(Natural Language Processing: Pretraining and Applications, removed 2026-11)
registered: SNLI, the IMDb reviews, three GloVe tables and the fastText
`wiki.en` vectors. Every notebook that downloads data depends on the
`DATA_HUB` shard (`d2l/_blocks/<fw>/DATA_HUB.py`), so dropping these entries
would mark 147 notebooks (four frameworks, 2026-11) stale with no code change
(docs/build-system.md §3.4). The blocks are byte-for-byte the cells of their
original homes (`natural-language-inference-and-dataset.md`,
`sentiment-analysis-and-dataset.md`, `similarity-analogy.md`), including a
trailing space, so the shard is unchanged.

Chapter 19 (`chapter_lm-encoders/`) is the natural home of the GloVe and
fastText entries: a section that registers one of them moves the block there
verbatim and deletes it here, which leaves the shard byte-identical. The
remaining entries can be deleted at the next full re-execution of the book.
-->

```{.python .input}
#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')
```

```{.python .input}
#@save
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')
```

```{.python .input}
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```
