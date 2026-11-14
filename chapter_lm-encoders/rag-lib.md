# Retrieval-Augmented Generation: Question-Set Plumbing (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_rag` explains in a sentence but does not show: the reader of
the question-set artifact (`_book_qa_read`, with the notice on a second load
of the `test` split in one process), the item list `BookQA` (`stats`,
`subset`), and the judge's frozen prompt `RAG_JUDGE_V1` with its message
builder and JSON parser (the question set's closed-book flags use them, so
they were fixed before the set was built).

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

```{.python .input #rag-lib-judge}
%%tab pytorch
#@save
# The judge's prompt, version 1: frozen before the book's question set was
# built (its closed-book flags use it); a change is RAG_JUDGE_V2 and a new
# question-set version
RAG_JUDGE_V1 = {
    'system': ('You grade answers to questions about the textbook Dive into '
               'Deep Learning. Reply with one line of JSON and nothing '
               'else.'),
    'correct': ('correct: true if the answer states every key fact (in any '
                'wording) and says nothing that contradicts the reference '
                'answer; otherwise false.'),
    'faithful': ('faithful: true if every factual statement in the answer is '
                 'supported by the passages given (ignore the reference '
                 'answer for this); otherwise false.'),
    'supported': ('supported: true if the cited passages alone support the '
                  'answer; false if they do not, or if the answer cites '
                  'nothing.'),
    'retry': 'Reply with the JSON line only, in exactly the requested form.'}

#@save
def _rag_abstained(answer):
    """Whether an answer is the abstention string (case, surrounding
    whitespace and quotes, and a final period ignored)."""
    norm = lambda s: s.strip().strip('"\'*').rstrip('.').strip().lower()
    return norm(answer) == norm(d2l.ABSTAIN)

def _rag_judge_messages(item, answer, context_texts, cited_ids):
    """The judge's chat messages (RAG_JUDGE_V1) for the verdicts it has to
    give on one answer: 'correct' always, 'faithful' and 'supported' when
    there are passages."""
    p = d2l.RAG_JUDGE_V1
    facts = '\n'.join(f'- {f}' for f in item['key_facts']) or '- (none)'
    parts = [f'Question: {item["question"]}',
             f'Reference answer: {item["reference"]}', f'Key facts:\n{facts}']
    keys = ['correct']
    if context_texts:
        keys += ['faithful', 'supported']
        parts.append('Passages given to the system:\n' + '\n\n'.join(
            f'[{n}] {t}' for n, t in enumerate(context_texts, start=1)))
        parts.append('Cited passages: ' + (', '.join(
            f'[{n}]' for n in cited_ids) if cited_ids else 'none'))
    form = json.dumps({k: True for k in keys})
    parts += [f'Answer to grade: {answer}', 'Decide:\n' + '\n'.join(
        p[k] for k in keys), f'Reply exactly in this form: {form}']
    return [{'role': 'system', 'content': p['system']},
            {'role': 'user', 'content': '\n\n'.join(parts)}]

def _rag_parse_json(text, keys):
    """{key: bool} from the first JSON object in a reply that has every
    one of keys as a boolean (or 'yes'/'no', 'true'/'false'); None if
    there is none."""
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        out = {}
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip().lower() in (
                    'yes', 'no', 'true', 'false'):
                v = v.strip().lower() in ('yes', 'true')
            if not isinstance(v, bool):
                break
            out[k] = v
        else:
            return out
    return None
```
