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
`subset`), the judge's frozen prompt `RAG_JUDGE_V1` with its message
builder and JSON parser (the question set's closed-book flags use them, so
they were fixed before the set was built), and the judge chosen on
2026-11-14 against frontier-model labels (`RAG_JUDGE_LOCAL`: Qwen3.8-27B,
thinking on, FP8 weights served by vLLM over two 24 GB GPUs; its server
`rag_judge_server`, its client `rag_judge_client` and `_rag_judge_replies`,
which reads a verdict after the reasoning, from an endpoint or a
Transformers model), and the plumbing of `RAGPipeline`: the prompts for
query rewriting, HyDE, decomposition and sentence verification, the prompt
variants of the pre-registered fixes, and the reader of the run files
(`_rag_records`). It also holds what :numref:`sec_lm_rag` and
:numref:`sec_lm_rag_fixes` share, so that neither notebook repeats it: the
models, artifacts and configurations of the runs (`RAG_ARTIFACTS`,
`RAG_BASE`, `RAG_CONFIGS`, `RAG_STAGES`, `RAG_CONTROLS`, `RAG_FIXES`), the
checked run order (`_rag_rerun`), each run's settings and file, the
generator loaded once (`_rag_generator`), the runner of a configuration
(`_rag_run_config`), the readers of the answer and judgment records, and the
per-question scores with their document-resampled intervals (`_rag_scores`,
`_rag_boot`, `_rag_paired`, `_rag_ready`).

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
    path = d2l.download_keyed('lm-book-qa-v2', root)
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
# The judge of the RAG section, chosen 2026-11-14 against frontier-model
# labels (lm-rag-judge-labels-v1): Qwen3.8-27B with thinking on, FP8 weights
# served by vLLM over two 24 GB GPUs; the same open weights are served in
# BF16 on OpenRouter (qwen/qwen3.8-27b, deepinfra/bf16)
RAG_JUDGE_LOCAL = {'model': 'Qwen/Qwen3.8-27B-FP8',
                   'revision': 'c3a462cc3d38cf351f3b26cbd763469eeb74ef81',
                   'vllm': '0.33.0', 'thinking': True,
                   'max_new_tokens': 3000}

#@save
def rag_judge_server(n_gpus=2, port=None, vllm=None):
    """A d2l.LocalChatServer that serves the judge (RAG_JUDGE_LOCAL) with
    vLLM, its layers split over n_gpus GPUs (pipeline parallel), from the
    executable vllm (default: the environment variable D2L_VLLM, else
    `vllm` on the PATH). vLLM lives in an environment of its own, since it
    pins its own PyTorch."""
    import shutil  # not in the d2l library preamble
    exe = vllm or os.environ.get('D2L_VLLM') or shutil.which('vllm')
    if exe is None:
        raise RuntimeError('no vLLM: install it in its own environment and '
                           'set D2L_VLLM to its vllm executable')
    j = d2l.RAG_JUDGE_LOCAL
    return d2l.LocalChatServer([
        exe, 'serve', j['model'], '--revision', j['revision'],
        '--pipeline-parallel-size', str(n_gpus), '--max-model-len', '16384',
        '--max-num-seqs', '64', '--gpu-memory-utilization', '0.9',
        '--limit-mm-per-prompt', '{"image": 0, "video": 0}', '--seed', '0',
        '--host', '127.0.0.1', '--port', '{port}'], port=port)

def rag_judge_client(url='http://127.0.0.1/v1', cache=None, mode='replay'):
    """A d2l.ChatAPI for the judge served at url, thinking on; the request
    hash keeps the host only, so a recorded cache replays with no server."""
    return d2l.ChatAPI(url, d2l.RAG_JUDGE_LOCAL['model'], cache=cache,
                       mode=mode, default_params={'extra_body': {
                           'chat_template_kwargs': {'enable_thinking': True}}})

#@save
def _rag_abstained(answer):
    """Whether an answer abstains: it contains the abstention sentence
    (case, whitespace, quotes, emphasis and the final period ignored),
    alone or with an explanation of what the passages lack. Generators
    asked to reply with the sentence alone often explain first."""
    norm = lambda s: re.sub(r'[\s"\'*]+', ' ', s).strip().rstrip('.').lower()
    return norm(d2l.ABSTAIN) in norm(answer)

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

def _rag_judge_replies(judge, tokenizer, messages, thinking, max_new_tokens,
                       batch_size):
    """The judge's replies to a list of chat messages, each the text after
    its reasoning ('</think>'), None when the reply was cut off at
    max_new_tokens, or the exception of a request that failed in transport
    (an endpoint that refused, timed out or answered with an error after
    the client's retries: no reply, so not a malformed one). judge is a
    d2l.ChatAPI (greedy, seed 0; batch_size requests in flight) or a
    Transformers model with its tokenizer."""
    if hasattr(judge, 'achat'):  # an OpenAI-compatible endpoint
        import asyncio  # not in the d2l library preamble
        import concurrent.futures
        batch = d2l.chat_batch(judge, [dict(
            messages=m, max_tokens=max_new_tokens, temperature=0, seed=0)
            for m in messages], concurrency=batch_size)
        if hasattr(judge, '_run'):  # a d2l.ChatAPI: its private event loop,
            out = judge._run(batch)  # the same for every batch, so its HTTP
        else:                        # connections never outlive their loop
            # A thread of its own: a notebook's kernel already runs an
            # event loop, in which asyncio.run refuses to start
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                out = pool.submit(asyncio.run, batch).result()
        return [r if isinstance(r, BaseException)  # no reply at all
                else None if r.finish_reason == 'length'
                else (r.text or '').split('</think>')[-1] for r in out]
    gens = d2l.generate(judge, tokenizer, messages,
                        max_new_tokens=max_new_tokens, chat=True,
                        enable_thinking=thinking, batch_size=batch_size)
    if not thinking:
        return [g.text for g in gens]
    return [None if g.finish_reason == 'length' and '</think>' not in g.text
            else g.text.split('</think>')[-1] for g in gens]

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

The prompts with which `RAGPipeline` asks the generator for a search query
(`rewrite`), a hypothetical passage (`hyde`) or sub-questions
(`decompose`), the yes/no check of one cited sentence (`verify`), and the two
prompt variants among the pre-registered fixes of :numref:`sec_lm_rag`:
"quote, then answer" and the abstention rule placed after the passages.

```{.python .input #rag-lib-pipeline}
%%tab pytorch
#@save
def _rag_rewrite_prompt(question):
    """Chat messages asking for a short search query for a question."""
    return [{'role': 'system', 'content': (
                'You turn questions about the textbook Dive into Deep '
                'Learning into search queries.')},
            {'role': 'user', 'content': (
                'Rewrite the question as a search query of at most 15 '
                'words. Keep its technical terms, names and code '
                f'identifiers. Reply with the query only.\n\nQuestion: '
                f'{question}')}]

def _rag_hyde_prompt(question):
    """Chat messages asking for a hypothetical passage that answers a
    question (HyDE: the dense side embeds the passage)."""
    return [{'role': 'user', 'content': (
        'Write a passage of at most 100 words from a deep learning '
        'textbook that answers the question. Reply with the passage '
        f'only.\n\nQuestion: {question}')}]

def _rag_decompose_prompt(question):
    """Chat messages asking to split a question into at most two
    self-contained questions, one per line."""
    return [{'role': 'user', 'content': (
        'If the question asks about two things, write two self-contained '
        'questions, one per line; otherwise repeat the question. Reply '
        f'with the questions only.\n\nQuestion: {question}')}]

def _rag_verify_prompt(sentence, passages):
    """Chat messages asking whether the passages (chunk dicts) support
    every fact of one sentence; the answer is read as P(yes)."""
    text = '\n\n'.join(f'[{n}] {p["text"]}'
                       for n, p in enumerate(passages, start=1))
    return [{'role': 'user', 'content': (
        f'Passages:\n{text}\n\nStatement: {sentence}\n\nDo the passages '
        'support every fact in the statement? Answer yes or no.')}]

def _rag_fix_prompt(question, hits, variant, abstain=None):
    """The grounded prompt of rag_prompt changed by one pre-registered fix:
    'quote' (copy the supporting sentences with their passage numbers,
    then answer after 'Answer:') or 'rule_after' (the abstention rule
    after the passages, in the user turn, instead of the system turn)."""
    abstain = abstain or d2l.ABSTAIN
    msgs = d2l.rag_prompt(question, hits, abstain)
    if variant == 'quote':
        msgs[1]['content'] = msgs[1]['content'].replace(
            'Answer in at most four sentences.',
            'First write "Quotes:" and copy the sentences of the passages '
            'that answer the question, each followed by its passage number '
            'such as [2]. Then write "Answer:" and answer in at most four '
            'sentences, citing the passages.')
    elif variant == 'rule_after':
        rule = 'If the passages do not contain the answer, reply exactly: '
        msgs[0]['content'] = msgs[0]['content'].replace(
            rule + abstain, '').strip()
        msgs[1]['content'] += f'\n{rule}{abstain}'
    elif variant != 'fixed':
        raise ValueError(f'unknown prompt variant {variant!r}')
    return msgs

def _rag_quoted_answer(text):
    """The answer part of a 'quote, then answer' reply (all of it when the
    reply has no 'Answer:')."""
    head, sep, tail = text.rpartition('Answer:')
    return tail.strip() if sep else text.strip()

def _rag_records(paths, on=('config', 'qid')):
    """The records of several JSON-lines run files (missing paths skipped)
    as one DataFrame; the columns on identify a record and may not
    repeat."""
    frames = [d2l.read_jsonl(p) for p in paths if p and os.path.exists(p)]
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=list(on)))
    dup = df.duplicated(list(on))
    if dup.any():
        raise ValueError(f'{dup.sum()} records repeat {on}, e.g. '
                         f'{df[dup].iloc[0][list(on)].tolist()}')
    return df
```

The configurations and run files that :numref:`sec_lm_rag` and
:numref:`sec_lm_rag_fixes` share: the models, the recorded artifacts, the
baseline and every configuration that changes one stage of it, with the fix
written down for each failure class before any class was counted; the
settings that each run's records are checked against, where a run file is
read from, and the generator, loaded once per process.

```{.python .input #rag-lib-runs}
%%tab pytorch
#@save
RAG_GENERATOR = 'Qwen/Qwen3.5-9B'  # revision pinned in the notebooks
RAG_RERANKER = 'Qwen/Qwen3-Reranker-0.6B'
RAG_ARTIFACTS = {  # name: (registered key, the file a regeneration writes)
    'rag-gen-base': ('lm-rag-gen-base-v2', 'lm-rag-gen-base.jsonl'),
    'rag-gen-stages': ('lm-rag-gen-stages-v3', 'lm-rag-gen-stages.jsonl'),
    'rag-gen-fix': ('lm-rag-gen-fix-v2', 'lm-rag-gen-fix.jsonl'),
    'rag-judge': ('lm-rag-judge-v6', 'lm-rag-judge.jsonl'),
    'rag-judge-cache': ('lm-rag-judge-cache-v6', 'lm-rag-judge-cache.zip'),
    'rag-cards': ('lm-rag-cards-v2', 'lm-rag-cards.jsonl')}

#@save
# The baseline: section chunks, hybrid retrieval, the first five chunks in
# rank order within 3,000 tokens, the fixed prompt, 300 new tokens, batches
# of eight prompts
RAG_BASE = dict(chunking='sections', context='retrieved', k=5, budget=3000,
                order='rank', rewrite=None, rerank_k=None, prompt='fixed',
                verify=False, abstain_below=None, max_new_tokens=300,
                batch=8, shuffle=None)
RAG_CONFIGS = {  # name: what changes against the baseline
    'closed': dict(RAG_BASE, context='none'),  # no passages: closed book
    'base': RAG_BASE,
    'oracle': dict(RAG_BASE, context='gold'),  # the gold chunks only
    'base-repeat': dict(RAG_BASE, shuffle=1),  # same prompts, other batches
    'fixed': dict(RAG_BASE, chunking='fixed', k=None, budget='base'),
    'fixed-over': dict(RAG_BASE, chunking='fixed', k=None, budget='base',
                       over=255),  # up to a window past the base's tokens
    'rewrite-rerank': dict(RAG_BASE, rewrite='generator', rerank_k=50),
    'reverse': dict(RAG_BASE, order='reverse'),
    'budget-1500': dict(RAG_BASE, k=None, budget=1500),
    'budget-6000': dict(RAG_BASE, k=None, budget=6000, batch=4)}
RAG_STAGES = ['fixed', 'rewrite-rerank', 'reverse', 'budget-1500',
              'budget-6000']
RAG_CONTROLS = ['fixed-over']  # checks a stage's result; not a candidate
RAG_FIXES = {  # failure class: its fix, written down before any is counted
    'not_retrieved': dict(RAG_BASE, rewrite='fuse'),  # the query's ranking
    'ranked_low': dict(RAG_BASE, rerank_k=100, k=8),
    'ignored': dict(RAG_BASE, prompt='quote', order='edges',
                    max_new_tokens=500),
    'unsupported': dict(RAG_BASE, verify=True),
    'false_answer': dict(RAG_BASE, abstain_below='train'),  # tuned on train
    'false_abstain': dict(RAG_BASE, prompt='rule_after')}
RAG_CONFIGS.update({f'fix-{c}': v for c, v in RAG_FIXES.items()})

#@save
_RAG_LOADED = {}  # the generator and its tokenizer while they are loaded

#@save
def _rag_rerun():
    """The RAG runs this run regenerates (d2l.lm_rerun of RAG_ARTIFACTS):
    regenerating the judge's cache regenerates the judge; the runs must
    come in their order, each result registered before the next."""
    rerun = d2l.lm_rerun(d2l.RAG_ARTIFACTS)
    if 'rag-judge-cache' in rerun:  # the judge's cache is written with it
        rerun.add('rag-judge')
    order = ('run order, each result registered before the next: '
             'rag-gen-base; rag-judge (until every answer is judged); '
             'rag-cards; rag-gen-stages; rag-judge; rag-gen-fix; rag-judge')
    assert not ('rag-judge' in rerun and rerun & {'rag-gen-stages',
                                                  'rag-gen-fix'}), order
    assert not ('rag-gen-fix' in rerun and rerun & {'rag-gen-base',
                                                    'rag-gen-stages'}), order
    return rerun

def _rag_settings(name):
    """What the answers of configuration name are made with (JSON): its
    entry of RAG_CONFIGS, the models at their revisions, the corpus, the
    BM25 parameters, the embedder's task and the question set. Reading the
    records checks it."""
    gen, emb, rr = d2l.RAG_GENERATOR, d2l.DEFAULT_EMBEDDER, d2l.RAG_RERANKER
    rev = {gen: d2l.HF_REVISIONS[gen],
           **{m: d2l.HF_ENCODER_REVISIONS[m] for m in (emb, rr)}}
    return json.dumps(dict(
        d2l.RAG_CONFIGS[name], generator=f'{gen}@{rev[gen]}',
        corpus=d2l.R3_VERSION, embedder=f'{emb}@{rev[emb]}',
        reranker=f'{rr}@{rev[rr]}', bm25=d2l.BookRetriever.BM25_PARAMS,
        task=d2l.BOOK_TASK, qa='lm-book-qa-v2'), sort_keys=True)

def _rag_run_path(name, rerun):
    """The file of RAG artifact name that this run regenerates (name in
    rerun), else the registered one, else None (not registered yet)."""
    key, file = d2l.RAG_ARTIFACTS[name]
    if name in rerun:
        return os.path.join('..', 'data', 'lm', 'rerun', file)
    return d2l.lm_registered(key)

def _rag_generator():
    """The generator (RAG_GENERATOR) and its tokenizer, loaded on first use,
    on the first GPU (never split over two visible cards)."""
    if not d2l._RAG_LOADED:
        gpu = torch.cuda.is_available()
        d2l._RAG_LOADED['gen'] = d2l.load_hf(
            d2l.RAG_GENERATOR, d2l.HF_REVISIONS[d2l.RAG_GENERATOR],
            dtype=torch.bfloat16 if gpu else torch.float32,
            device_map={'': 0} if gpu else 'auto')
    return d2l._RAG_LOADED['gen']

def _rag_free_generator():
    """Drop the generator and free its memory."""
    d2l._RAG_LOADED.clear()
    d2l.lm_free()

def _rag_gold_context(item, corpus):
    """The chunks of an item's gold spans (not the alternatives), in span
    order."""
    ids = [i for s in item['gold_spans'] for i in corpus.span_to_chunk_ids(
        s['doc_id'], s['char_start'], s['char_end'])]
    return [corpus[i] for i in dict.fromkeys(ids)]

def _rag_run_config(name, items, retrievers, keep_gen, depth=50,
                    budgets=None, threshold=None):
    """Answer records of configuration name (RAG_CONFIGS) on items, in item
    order. retrievers maps each chunking to its BookRetriever; keep_gen:
    the generator stays on the card while the retriever works; depth: the
    ranking kept; budgets (qid: tokens) sets each question's budget for a
    configuration whose budget is 'base'; threshold: the reranker score
    below which the pipeline abstains."""
    c, items = d2l.RAG_CONFIGS[name], list(items)
    budget = d2l.RAG_BASE['budget']
    assert (c['budget'] == 'base') == (budgets is not None), name
    order = list(range(len(items)))
    if c['shuffle'] is not None:  # other batches for the same prompts
        random.Random(c['shuffle']).shuffle(order)
    items = [items[i] for i in order]
    if not (keep_gen or c['rewrite']):  # retrieval without the generator
        d2l._rag_free_generator()
    assert keep_gen or not c['rewrite'], 'the rewriter needs the card too'
    pipe = d2l.RAGPipeline(  # the generator joins when it is needed
        retrievers[c['chunking']],
        *(d2l._rag_generator() if c['rewrite'] else (None, None)),
        k=c['k'], order=c['order'],
        budget_tokens=budget if c['budget'] == 'base' else c['budget'],
        rewrite=c['rewrite'], rerank_k=c['rerank_k'], depth=depth,
        max_new_tokens=c['max_new_tokens'], prompt=c['prompt'],
        verify=c['verify'], abstain_below=threshold)
    if c['context'] == 'none':
        ctx = [dict(query_used=None, ranked_ids=[], context=None)
               for _ in items]
    else:
        ctx = pipe.prepare(items, None if budgets is None else
                           [budgets[q['qid']] for q in items])
    if c['context'] == 'gold':  # unanswerable items keep the retrieved
        corpus = retrievers['sections'].corpus
        for p, q in zip(ctx, items):
            if q['answerable']:
                p['context'] = d2l.assemble_context(
                    d2l._rag_gold_context(q, corpus), budget, k=c['k'])[0]
    pipe.model, pipe.tokenizer = d2l._rag_generator()
    res = pipe.answer(items, batch_size=c['batch'], contexts=ctx)
    recs = [dict(config=name, split=q['split'],
                 settings=d2l._rag_settings(name),
                 **{k: v for k, v in r._asdict().items() if k != 'question'})
            for q, r in zip(items, res)]
    return [r for _, r in sorted(zip(order, recs), key=lambda t: t[0])]

def _rag_run_configs(names, items, *args, **kwargs):
    """_rag_run_config for each name, printing its minutes and peak
    memory."""
    recs = []
    for name in names:
        t = time.time()
        recs += d2l._rag_run_config(name, items, *args, **kwargs)
        print(f'{name}: {(time.time() - t) / 60:.1f} min, {d2l.lm_peak()}')
    return recs
```

The recorded answers and verdicts: an answer's key (what its verdict
depends on), the answer records of all runs with their settings checked,
and the judgment records.

```{.python .input #rag-lib-records}
%%tab pytorch
#@save
def _rag_answer_key(r):
    """What a verdict depends on: the answer, its context and citations
    (a regenerated answer needs a new verdict)."""
    return hashlib.sha1(json.dumps([r['answer'], list(r['context_ids']), [
        int(n) for n in r['cited_ids']]]).encode()).hexdigest()[:16]

def _rag_save_gens(name, recs):
    """Save the answer records of RAG artifact name, a table keyed by
    configuration and question."""
    d2l.lm_save_table(d2l.RAG_ARTIFACTS[name][1], pd.DataFrame(recs),
                      on=['config', 'qid'], key=d2l.RAG_ARTIFACTS[name][0])

def _rag_load_gens(rerun):
    """Every answer made so far, one row per (configuration, question),
    with the answer_sha its verdict must carry; each configuration's
    settings are checked against _rag_settings."""
    df = d2l._rag_records([d2l._rag_run_path(n, rerun) for n in (
        'rag-gen-base', 'rag-gen-stages', 'rag-gen-fix')])
    for name, s in (df.groupby('config')['settings'] if len(df) else []):
        assert (s == d2l._rag_settings(name)).all(), f'{name}: other settings'
    if len(df):  # by the current rule, not the one of the run that wrote it
        df['abstained'] = [d2l._rag_abstained(a) for a in df.answer]
    return df.assign(answer_sha=[d2l._rag_answer_key(r)
                                 for r in df.to_dict('records')])

def _rag_load_judgments(path):
    """Judgment records, verdicts as True, False or None; max_new_tokens
    of rows from before the re-judging is the judge's default."""
    df = d2l._rag_records([path], on=['config', 'qid', 'concurrency'])
    for k in ('correct', 'faithful', 'supported'):
        df[k] = [None if v is None or v != v else bool(v)
                 for v in df.get(k, [])]
    for k in ('backend', 'answer_sha'):
        df[k] = df.get(k, [None] * len(df))
    default = d2l.RAG_JUDGE_LOCAL['max_new_tokens']
    df['max_new_tokens'] = [default if v is None or v != v else int(v)
                            for v in df.get('max_new_tokens',
                                            [None] * len(df))]
    return df
```

Per-question scores of a configuration and their means with 95% intervals.
Questions drafted from one document share its difficulty, so the bootstrap
resamples documents (`_rag_clusters`); differences between configurations
are paired by question.

```{.python .input #rag-lib-scores}
%%tab pytorch
#@save
def _rag_clusters(items, corpus):
    """The document of each item (qid: doc_id), the unit the bootstrap
    resamples: that of its first gold span, else of its anchor chunk."""
    return {q['qid']: (q['gold_spans'] or [{'doc_id': corpus[
        q['anchor_ids'][0]]['doc_id']}])[0]['doc_id'] for q in items}

def _rag_signflip_p(d, cluster, n=10_000):
    """Two-sided sign-flip p of the mean of per-document mean
    differences."""
    m = np.array([d[cluster == c].mean() for c in np.unique(cluster)])
    flips = np.random.default_rng(0).choice([-1.0, 1.0], (n, len(m)))
    return np.mean(np.abs(flips @ m) >= np.abs(m.sum()) - 1e-12)

def _rag_ci(a, b=None, cluster=None):
    """The bootstrap of the mean of per-question scores a (or of a - b,
    paired), NaNs dropped, resampling documents when there are 30 or
    more, else questions: (result or None, by documents, a, b,
    cluster)."""
    import warnings  # not in the d2l library preamble
    a = np.asarray(a, float)
    b = None if b is None else np.asarray(b, float)
    keep = ~np.isnan(a) & (True if b is None else ~np.isnan(b))
    a, cluster = a[keep], np.asarray(cluster)[keep]
    b = None if b is None else b[keep]
    by_doc = len(set(cluster)) >= 30
    if len(a) < 2:
        return None, by_doc, a, b, cluster
    with warnings.catch_warnings():  # few items: _rag_boot's * says so
        warnings.simplefilter('ignore')
        r = (d2l.bootstrap_ci(a, cluster=cluster if by_doc else None)
             if b is None else d2l.paired_bootstrap(
                 a, b, cluster=cluster if by_doc else None))
    return r, by_doc, a, b, cluster

def _rag_boot(a, b=None, cluster=None):
    """Mean of per-question scores a (or of a - b, paired), NaNs dropped,
    with a 95% interval that resamples documents, marked * below 100
    questions; below 30 documents questions are resampled instead, marked
    + (and a paired difference gets a document-level sign-flip p)."""
    r, by_doc, a, b, cluster = d2l._rag_ci(a, b, cluster)
    if r is None:
        return 'n/a'
    f = '.3f' if b is None else '+.3f'
    out = f'{r.mean:{f}} [{r.ci[0]:{f}}, {r.ci[1]:{f}}]'
    if by_doc:
        return out + ('' if len(a) >= 100 else '*')
    if b is None:
        return out + '+'
    p = d2l._rag_signflip_p(a - b, cluster)
    return out + f'+ (sign-flip p {p:.3f})'

def _rag_paired(a, b, col, mask=None):
    """Paired difference a - b of score col on the questions both score
    tables have (mask: a filter of the merged rows)."""
    m = a.merge(b, on='qid', suffixes=('_a', '_b'))
    if mask is not None:
        m = m[mask(m)]
    return d2l._rag_boot(m[f'{col}_a'], m[f'{col}_b'], m.cluster_a)

def _rag_scores(config, split, gens, jud, retrievers, items, cluster,
                concurrency=8):
    """One row per judged answer of a configuration on a split, with
    per-question scores (NaN where a score does not apply, and accuracy NaN
    where the judge gave no verdict: an unjudged answer is left out). gens:
    the answer records (_rag_load_gens); jud: the judgment records, of
    which those judged at concurrency count; retrievers maps a chunking to
    its BookRetriever; items: qid to item; cluster: qid to document."""
    verdicts = ['correct', 'faithful', 'supported']
    k, budget = d2l.RAG_BASE['k'], d2l.RAG_BASE['budget']
    g = gens[(gens.config == config) & (gens.split == split)]
    j = jud[jud.concurrency == concurrency][['config', 'qid', 'answer_sha']
                                            + verdicts]
    df = g.merge(j, on=['config', 'qid', 'answer_sha']).sort_values(
        'qid', ignore_index=True)
    corpus = retrievers[d2l.RAG_CONFIGS[config]['chunking']].corpus
    its = [items[q] for q in df.qid]
    ans = np.array([q['answerable'] for q in its])
    rel = [d2l.gold_chunk_ids(q, corpus) for q in its]
    cm = d2l.citation_metrics(df.to_dict('records'), its, corpus)
    has_ctx = df.context_ids.map(len).to_numpy() > 0
    answered = has_ctx & ~df.abstained.to_numpy(bool)
    flag = lambda col: np.array([np.nan if v is None else float(v)
                                 for v in df[col]])
    return pd.DataFrame({
        'qid': df.qid, 'cluster': [cluster[q] for q in df.qid],
        'answerable': ans, 'type': [q['type'] for q in its],
        'book_specific': ['general_knowledge' not in q['flags']
                          for q in its],
        'accuracy': flag('correct'),  # NaN: unjudged, left out
        'faithful': np.where(answered, flag('faithful'), np.nan),
        'supported': np.where(answered, flag('supported'), np.nan),
        'cite_precision': np.where(ans, cm.precision, np.nan),
        'cite_recall': np.where(ans, cm.recall, np.nan),
        'abstained': df.abstained.astype(float),
        'hit': np.where(ans, [float(bool(r & set(c))) for r, c in zip(
            rel, df.context_ids)], np.nan),
        f'recall@{k}': np.where(ans & has_ctx, [float(bool(r & set(x[:k])))
                                                for r, x in zip(
                                                    rel, df.ranked_ids)],
                                np.nan),
        f'cov@{budget}': np.where(ans & has_ctx, [float(d2l.cov_at_budget(
            x, q, corpus, budget)) for x, q in zip(df.ranked_ids, its)],
            np.nan),
        'answer': df.answer, 'n_prompt_tokens': df.n_prompt_tokens,
        'n_new_tokens': df.n_new_tokens, 'latency_s': df.latency_s,
        'record': df.to_dict('records'),
        'verdicts': df[verdicts].to_dict('records')})

def _rag_ready(configs, split, gens, jud, rerun, concurrency=8):
    """Whether every answer of each configuration on split carries a
    verdict on that very answer (answer_sha); says which do not. Partial
    or stale judgments are never reported: between judging runs a
    configuration is left out, and a normal run stops on one."""
    k = jud[jud.concurrency == concurrency]  # not judged: gens may be new
    j = set(zip(k.config, k.qid, k.answer_sha))
    missing = {}
    for c in configs:
        g = gens[(gens.config == c) & (gens.split == split)]
        n = sum(x not in j for x in zip(g.config, g.qid, g.answer_sha))
        if n or not len(g):
            missing[c] = (f'{n} of {len(g)} answers lack a verdict' if len(g)
                          else 'no answers')
    if missing:
        print(f'not ready on {split}: {missing}')
        assert rerun or all(m == 'no answers' for m in missing.values()), (
            'registered answers without verdicts: register the judge run')
    return not missing
```
