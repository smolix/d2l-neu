# Retrieval-Augmented Generation: Question-Set Plumbing (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_rag` explains in a sentence but does not show: the reader of
the question-set artifact (`_book_qa_read`, with the notice on a second load
of the `test` split in one process) and the item list `BookQA` (`stats`,
`subset`).

Every block is tagged `%%tab pytorch`; a module-level import would join the
`_preamble` shard and move the freshness fingerprint of every notebook in the
book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_rag`

```{.python .input #rag-lib-qa}
%%tab pytorch
#@save
_BOOK_QA_LOADS = collections.Counter()  # loads per split in this process

#@save
def _book_qa_read(split, root=os.path.join('..', 'data', 'lm')):
    """The items of one split of the question set, as dicts; a second load
    of the test split in one process prints a notice."""
    if split not in ('train', 'dev', 'test'):
        raise ValueError(f'unknown split {split!r}')
    _BOOK_QA_LOADS[split] += 1
    if split == 'test' and _BOOK_QA_LOADS[split] == 2:
        print('notice: the test split is loaded a second time in this '
              'process; decisions belong to train and dev')
    path = d2l.download_keyed('lm-book-qa-v1', root)
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist()
                    if n.rsplit('/', 1)[-1] == f'qa_{split}.jsonl')
        lines = z.read(name).decode('utf-8').splitlines()
    return [json.loads(l) for l in lines if l.strip()]

class BookQA(list):
    """Items of the book's question set (dicts), from one split."""
    def __init__(self, items, split):
        super().__init__(items)
        self.split = split

    def stats(self):
        """Items per question type, answerable and book-specific (no
        general_knowledge flag)."""
        df = pd.DataFrame([dict(type=q['type'], answerable=q['answerable'],
                                book_specific='general_knowledge' not in
                                q.get('flags', ())) for q in self])
        if df.empty:
            return df
        out = df.groupby('type', sort=False).agg(
            items=('type', 'size'), answerable=('answerable', 'sum'),
            book_specific=('book_specific', 'sum'))
        out.loc['total'] = out.sum()
        return out

    def subset(self, n, seed=0, stratify='type'):
        """n items drawn without replacement, in proportion to the counts
        of stratify (largest remainders), kept in their original order."""
        rng = np.random.default_rng(seed)
        groups = collections.defaultdict(list)
        for i, q in enumerate(self):
            groups[q[stratify] if stratify else None].append(i)
        keys = sorted(groups, key=str)
        quota = {g: n * len(groups[g]) / len(self) for g in keys}
        take = {g: math.floor(quota[g]) for g in keys}
        for g in sorted(keys, key=lambda g: take[g] - quota[g])[
                :n - sum(take.values())]:
            take[g] += 1
        picked = sorted(i for g in keys for i in rng.choice(
            groups[g], size=min(take[g], len(groups[g])), replace=False))
        return BookQA([self[i] for i in picked], self.split)
```
