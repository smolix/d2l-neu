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
(`_rag_records`).

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
# The judge of 19.5, chosen 2026-11-14 against frontier-model labels
# (lm-rag-judge-labels-v1): Qwen3.8-27B with thinking on, FP8 weights served
# by vLLM over two 24 GB GPUs; the same open weights are served in BF16 on
# OpenRouter (qwen/qwen3.8-27b, deepinfra/bf16)
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
