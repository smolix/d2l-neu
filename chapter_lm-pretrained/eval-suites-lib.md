# Evaluation Suite Registry (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_eval_suites` explains in one sentence and does not show:
the dataset adapters `d2l.EVAL_ADAPTERS`, the task registry
`d2l.EVAL_TASKS`, the frozen suite manifests `d2l.EVAL_SUITES`, the row
loader `_eval_load_rows`, the task normalizer `_eval_canonical`, the item
renderer `_eval_items`, the run-cache
helpers `_eval_model_id` and `_eval_cache_*`, and the manifest and
save/load methods of `d2l.EvalSuite` and `d2l.EvalResult`. Every block is
tagged `%%tab pytorch` and imports third-party modules inside its
functions. Item ids are strings: a dataset's own id where it has one (ARC,
HumanEval, MMLU-Pro), else the row index in the pinned file.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_eval_suites`

```{.python .input #eval-suites-lib-load}
%%tab pytorch
#@save
_EVAL_ROWS = {}  # rows of the files read from the Hub in this process

#@save
def _eval_load_rows(task, split=None):
    """The rows (dicts) of the file of task.files for split (default
    task.split): the one file whose path has the split name as a word.
    Read from the Hugging Face Hub at task.revision, or from the folder
    task.dataset if that is a local directory; parquet or JSON lines."""
    split = split or task.split
    word = re.compile(rf'(^|[/_.-]){re.escape(split)}([/_.-]|$)')
    files = [f for f in task.files if word.search(f)]
    if len(files) != 1:
        raise ValueError(f'task {task.id}: {len(files)} files of split '
                         f'{split!r} in {task.files}')
    local = os.path.isdir(task.dataset)
    key = (task.dataset, task.revision, files[0])
    if local or key not in _EVAL_ROWS:
        if local:
            path = os.path.join(task.dataset, files[0])
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(task.dataset, files[0], repo_type='dataset',
                                   revision=task.revision)
        if path.endswith('.parquet'):
            rows = pd.read_parquet(path).to_dict('records')
        else:
            with open(path, encoding='utf-8') as f:
                rows = [json.loads(line) for line in f if line.strip()]
        if local:
            return rows
        _EVAL_ROWS[key] = rows
    return _EVAL_ROWS[key]
```

```{.python .input #eval-suites-lib-adapters}
%%tab pytorch
#@save
def _eval_arc(row, i):
    """ARC: the question, its options and the index of the answer key."""
    labels = [str(a) for a in row['choices']['label']]
    return {'id': str(row['id']), 'question': row['question'],
            'options': [str(o) for o in row['choices']['text']],
            'gold': labels.index(str(row['answerKey']))}

#@save
def _eval_hellaswag(row, i):
    """HellaSwag, cleaned as lm-evaluation-harness does: 'label: context'
    as the question, the four endings as options. With fmt='continuation'
    the context is the prompt and each ending continues it, as in the
    harness."""
    def clean(text):
        text = text.strip().replace(' [title]', '. ')
        return re.sub(r'\[.*?\]', '', text).replace('  ', ' ')
    ctx = row['ctx_a'] + ' ' + row['ctx_b'].capitalize()
    return {'id': str(i), 'question': clean(row['activity_label'] + ': ' +
                                            ctx),
            'options': [clean(e) for e in row['endings']],
            'gold': int(row['label'])}

#@save
def _eval_mmlu(row, i):
    """MMLU: the question, four options, the answer index, the subject
    (for same-subject demonstrations and as the cluster) and the subject
    header that precedes the prompt."""
    subject = str(row['subject'])
    return {'id': str(i), 'question': row['question'].strip(),
            'options': [str(o) for o in row['choices']],
            'gold': int(row['answer']), 'subject': subject,
            'header': 'The following are multiple choice questions (with '
                      f"answers) about {subject.replace('_', ' ')}.\n\n"}

#@save
def _eval_mmlu_pro(row, i):
    """MMLU-Pro: up to ten options, the category as the subject."""
    subject = str(row['category'])
    return {'id': str(row['question_id']), 'question': row['question'].strip(),
            'options': [str(o) for o in row['options']],
            'gold': int(row['answer_index']), 'subject': subject,
            'header': 'The following are multiple choice questions (with '
                      f'answers) about {subject}.\n\n'}

#@save
def _eval_piqa(row, i):
    """PIQA: the goal and its two solutions."""
    return {'id': str(i), 'question': row['goal'],
            'options': [row['sol1'], row['sol2']], 'gold': int(row['label'])}

#@save
def _eval_sciq(row, i):
    """SciQ: the support paragraph as a header, three distractors and the
    correct answer last, as lm-evaluation-harness orders them."""
    support = (row['support'] or '').strip()
    return {'id': str(i), 'question': row['question'],
            'options': [row['distractor1'], row['distractor2'],
                        row['distractor3'], row['correct_answer']],
            'gold': 3, 'header': support + '\n' if support else ''}

#@save
def _eval_winogrande(row, i):
    """WinoGrande, partially scored: the sentence up to the blank with
    either option as two prompts, the rest of the sentence as their
    shared continuation."""
    s = row['sentence']
    k = s.index('_')
    return {'id': str(i), 'prompts': [s[:k] + row['option1'],
                                      s[:k] + row['option2']],
            'continuation': ' ' + s[k + 1:].strip(),
            'gold': int(row['answer']) - 1}

#@save
def _eval_gsm8k(row, i):
    """GSM8K: the question, the worked answer without its calculator
    annotations <<...>> (a demonstration's target) and the final number
    after '####' (the reference)."""
    return {'id': str(i), 'question': row['question'],
            'target': re.sub(r'<<[^>]*>>', '', row['answer']),
            'reference': d2l.extract_answer(row['answer'], 'gsm8k',
                                            strict=True)}

#@save
def _eval_humaneval(row, i):
    """HumanEval: the function header and docstring as the prompt; the
    tests call check() on the entry point."""
    return {'id': str(row['task_id']), 'prompt': row['prompt'],
            'test': f"{row['test']}\n\ncheck({row['entry_point']})\n",
            'entry_point': row['entry_point']}

#@save
EVAL_ADAPTERS = {  # dataset row and its index -> an item
    'arc': _eval_arc, 'hellaswag': _eval_hellaswag, 'mmlu': _eval_mmlu,
    'mmlu_pro': _eval_mmlu_pro, 'piqa': _eval_piqa, 'sciq': _eval_sciq,
    'winogrande': _eval_winogrande, 'gsm8k': _eval_gsm8k,
    'humaneval': _eval_humaneval}
```

```{.python .input #eval-suites-lib-items}
%%tab pytorch
#@save
def _eval_canonical(task):
    """task with canonical JSON values (numpy numbers and arrays become
    Python ones; files and stop become tuples, a string being one entry).
    Refuses sampling keys that run() sets itself."""
    def plain(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        raise TypeError(f'task {task.id}: {x!r} is not JSON')
    d = json.loads(json.dumps(task._asdict(), default=plain))
    for key in ('files', 'stop'):
        d[key] = (d[key],) if isinstance(d[key], str) else tuple(d[key] or ())
    clash = sorted(set(d['sampling'] or ()) & {
        'seed', 'stop', 'chat', 'enable_thinking', 'batch_size',
        'max_new_tokens'})
    if clash:
        raise ValueError(f'task {task.id}: sampling sets {clash}, which '
                         'run() derives from the task and its seed')
    return task._replace(**d)

#@save
def _eval_items(task, rows, ids):
    """The items rows[ids] of task, adapted by EVAL_ADAPTERS[task.adapter]
    and rendered: 'mc' items get a prompt and choices from format_mc in
    task.fmt with task.shots demonstrations (after the item's header), or,
    with fmt='continuation', the question as the prompt and each option
    after a space as a choice (demonstrations: question and answer, blank
    lines between); 'gen' items get a prompt from fewshot_prompt in
    task.template. The
    demonstrations come from task.shots_from: {'split', 'ids'} (rows of
    that split) or '<split>:same_subject' (the first rows of that split
    with the item's subject)."""
    if task.kind not in ('mc', 'mc_partial', 'gen', 'code', 'ppl'):
        raise ValueError(f'task {task.id}: unknown kind {task.kind!r}')
    if task.template in ('chat_single', 'chat_multi') and not task.chat:
        raise ValueError(f'task {task.id}: template {task.template!r} '
                         'needs chat=True')
    adapt = d2l.EVAL_ADAPTERS[task.adapter]
    items = [adapt(rows[int(i)], int(i)) for i in ids]
    src, k = task.shots_from, task.shots or 0
    if k and src is None:
        raise ValueError(f'task {task.id}: {k} shots but no shots_from')
    if isinstance(src, dict):
        if len(src['ids']) != k:
            raise ValueError(f"task {task.id}: shots={k} but "
                             f"{len(src['ids'])} demonstration ids")
        pool = d2l._eval_load_rows(task, src['split'])
        fixed = [adapt(pool[j], j) for j in src['ids']]
    elif isinstance(src, str):
        split, rule = src.split(':')
        if rule != 'same_subject':
            raise ValueError(f'task {task.id}: unknown rule {rule!r}')
        pool = d2l._eval_load_rows(task, split)
        by_subject = collections.defaultdict(list)
        for j, row in enumerate(pool):
            shot = adapt(row, j)
            if len(by_subject[shot['subject']]) < k:
                by_subject[shot['subject']].append(shot)
    for item in items:
        shots = [] if not k else fixed if isinstance(src, dict) else (
            by_subject[item['subject']])
        if k and len(shots) < k:
            raise ValueError(f"task {task.id}: {len(shots)} of {k} shots "
                             f"for item {item['id']}")
        if task.kind == 'mc' and task.fmt == 'continuation':
            demos = [s['question'] + ' ' + s['options'][s['gold']]
                     for s in shots]
            prompt = '\n\n'.join(demos + [item['question']])
            item['prompt'] = item.get('header', '') + prompt
            item['choices'] = [' ' + o for o in item['options']]
        elif task.kind == 'mc':
            prompt, item['choices'] = d2l.format_mc(
                item['question'], item['options'], task.fmt,
                shots=[(s['question'], s['options'], s['gold'])
                       for s in shots])
            item['prompt'] = item.get('header', '') + prompt
        elif task.kind == 'gen':
            demos = [(s['question'], s['target']) for s in shots]
            item['prompt'] = (item['question'] if task.template is None
                              else d2l.fewshot_prompt(demos, item['question'],
                                                      task.template))
    return items
```

```{.python .input #eval-suites-lib-registry}
%%tab pytorch
#@save
EVAL_TASKS = {t.id: t for t in (  # pinned revisions and explicit files
    EvalTask('arc_challenge', 'allenai/ai2_arc',
             '210d026faf9955653af8916fad021475a3f00453',
             ('ARC-Challenge/test-00000-of-00001.parquet',), 'mc', 'arc',
             config='ARC-Challenge', norm='char'),
    EvalTask('arc_easy', 'allenai/ai2_arc',
             '210d026faf9955653af8916fad021475a3f00453',
             ('ARC-Easy/test-00000-of-00001.parquet',), 'mc', 'arc',
             config='ARC-Easy', norm='char'),
    EvalTask('hellaswag', 'Rowan/hellaswag',
             '218ec52e09a7e7462a5400043bb9a69a41d06b76',
             ('data/validation-00000-of-00001.parquet',), 'mc', 'hellaswag',
             split='validation', fmt='continuation', norm='char'),
    EvalTask('mmlu', 'cais/mmlu', 'c30699e8356da336a370243923dbaf21066bb9fe',
             ('all/test-00000-of-00001.parquet',
              'all/dev-00000-of-00001.parquet'), 'mc', 'mmlu', config='all',
             fmt='letter', norm='sum', shots=5,
             shots_from='dev:same_subject', cluster='subject'),
    EvalTask('mmlu_pro', 'TIGER-Lab/MMLU-Pro',
             'b189ec765aa7ed75c8acfea42df31fdae71f97be',
             ('data/test-00000-of-00001.parquet',
              'data/validation-00000-of-00001.parquet'), 'mc', 'mmlu_pro',
             fmt='letter', norm='sum', shots=5,
             shots_from='validation:same_subject'),
    EvalTask('gsm8k', 'openai/gsm8k',
             '740312add88f781978c0658806c59bc2815b9866',
             ('main/test-00000-of-00001.parquet',
              'main/train-00000-of-00001.parquet'), 'gen', 'gsm8k',
             config='main', template='qa', shots=4,
             shots_from={'split': 'train', 'ids': [0, 1, 2, 3]},
             max_new_tokens=384, stop=('\n\nQ:',), extract='gsm8k'),
    EvalTask('humaneval', 'openai/openai_humaneval',
             '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',
             ('openai_humaneval/test-00000-of-00001.parquet',), 'code',
             'humaneval', max_new_tokens=512,
             stop=('\nclass', '\ndef', '\n#', '\nif', '\nprint')),
    EvalTask('piqa', 'baber/piqa', '142f6d7367fd9877f0fb3b5734ea6a545f54cdd1',
             ('piqa_validation.parquet',), 'mc', 'piqa', split='validation',
             norm='char'),
    EvalTask('sciq', 'allenai/sciq',
             '2c94ad3e1aafab77146f384e23536f97a4849815',
             ('data/test-00000-of-00001.parquet',), 'mc', 'sciq',
             norm='char'),
    EvalTask('winogrande', 'allenai/winogrande',
             '01e74176c63542e6b0bcb004dcdea22d94fb67b5',
             ('winogrande_xl/validation-00000-of-00001.parquet',),
             'mc_partial', 'winogrande', split='validation',
             config='winogrande_xl', norm='sum'))}

#@save
EVAL_SUITES = {  # frozen manifests: a change here is a new suite version
    'dev_v1': {'name': 'dev_v1', 'tasks': [
        EvalTask('arc_challenge', 'allenai/ai2_arc',
                 '210d026faf9955653af8916fad021475a3f00453',
                 ('ARC-Challenge/test-00000-of-00001.parquet',), 'mc', 'arc',
                 config='ARC-Challenge', n=500, norm='char'),
        EvalTask('gsm8k', 'openai/gsm8k',
                 '740312add88f781978c0658806c59bc2815b9866',
                 ('main/test-00000-of-00001.parquet',
                  'main/train-00000-of-00001.parquet'), 'gen', 'gsm8k',
                 config='main', n=100, template='chat_multi', chat=True,
                 enable_thinking=False, shots=4,
                 shots_from={'split': 'train', 'ids': [0, 1, 2, 3]},
                 max_new_tokens=384, extract='gsm8k')]},
    'report_v1': {'name': 'report_v1', 'tasks': [
        EvalTask('mmlu_pro', 'TIGER-Lab/MMLU-Pro',
                 'b189ec765aa7ed75c8acfea42df31fdae71f97be',
                 ('data/test-00000-of-00001.parquet',
                  'data/validation-00000-of-00001.parquet'), 'mc',
                 'mmlu_pro', n=1000, fmt='letter', norm='sum', shots=5,
                 shots_from='validation:same_subject'),
        EvalTask('mmlu', 'cais/mmlu',
                 'c30699e8356da336a370243923dbaf21066bb9fe',
                 ('all/test-00000-of-00001.parquet',
                  'all/dev-00000-of-00001.parquet'), 'mc', 'mmlu',
                 config='all', n=2000, fmt='letter', norm='sum', shots=5,
                 shots_from='dev:same_subject', cluster='subject'),
        EvalTask('gsm8k', 'openai/gsm8k',
                 '740312add88f781978c0658806c59bc2815b9866',
                 ('main/test-00000-of-00001.parquet',
                  'main/train-00000-of-00001.parquet'), 'gen', 'gsm8k',
                 config='main', template='qa', shots=4,
                 shots_from={'split': 'train', 'ids': [0, 1, 2, 3]},
                 max_new_tokens=384, stop=('\n\nQ:',), extract='gsm8k')]}}
```

```{.python .input #eval-suites-lib-manifest}
%%tab pytorch
@d2l.add_to_class(d2l.EvalSuite)  #@save
@classmethod
def from_manifest(cls, name_or_path, batch_size=8):
    """The suite of a frozen manifest: a name in d2l.EVAL_SUITES or a JSON
    file written by to_manifest. Its tasks' n and subset_seed are kept."""
    if name_or_path in d2l.EVAL_SUITES:
        manifest = d2l.EVAL_SUITES[name_or_path]
    else:
        with open(name_or_path, encoding='utf-8') as f:
            manifest = json.load(f)
    return cls(manifest['tasks'], manifest['name'], batch_size, n=None,
               seed=None)

@d2l.add_to_class(d2l.EvalSuite)  #@save
def to_manifest(self, path=None):
    """The suite as a JSON-ready manifest {'name', 'tasks'}; with path,
    also written there as sorted-key JSON."""
    manifest = json.loads(json.dumps(
        {'name': self.name, 'tasks': [t._asdict() for t in self.tasks]}))
    if path is not None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, sort_keys=True, indent=1)
            f.write('\n')
    return manifest
```

```{.python .input #eval-suites-lib-cache}
%%tab pytorch
#@save
def _eval_model_id(model):
    """(model id, revision) of a Transformers model, from its config;
    other models may set the attributes name_or_path and revision, else
    their class name and 'local' are used."""
    m = d2l._lm_unwrap(model)
    cfg = getattr(m, 'config', None)
    name = (getattr(m, 'name_or_path', None) or
            getattr(cfg, '_name_or_path', None) or type(m).__name__)
    revision = (getattr(m, 'revision', None) or
                getattr(cfg, '_commit_hash', None) or 'local')
    return str(name), str(revision)

#@save
def _eval_cache_path(cache_dir, model_id, revision, suite_hash, run_key):
    """<cache_dir>/<model id, '/' as '--'>@<revision>/<suite hash>/<run
    key>.parquet"""
    return os.path.join(cache_dir, f"{model_id.replace('/', '--')}@"
                        f'{revision}', suite_hash, f'{run_key}.parquet')

#@save
def _eval_sorted(rows):
    """Result rows as a DataFrame sorted by task and item id."""
    return pd.DataFrame(rows).sort_values(['task', 'item_id'],
                                          ignore_index=True)

#@save
def _eval_cache_read(path):
    """The result rows cached at path, or [] if there are none."""
    if not os.path.exists(path):
        return []
    return [{k: (None if isinstance(v, float) and k != 'score' and
                 math.isnan(v) else v) for k, v in r.items()}
            for r in pd.read_parquet(path).to_dict('records')]

#@save
def _eval_cache_write(path, rows):
    """Write result rows to path atomically, sorted, without timestamps."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _eval_sorted(rows).to_parquet(path + '.tmp', index=False)
    os.replace(path + '.tmp', path)
```

```{.python .input #eval-suites-lib-save}
%%tab pytorch
@d2l.add_to_class(d2l.EvalResult)  #@save
def save(self, path):
    """Write per_item to path (parquet, rows sorted) and meta, without the
    wall clock, to path with the suffix .json (sorted keys)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    _eval_sorted(self.per_item).to_parquet(path, index=False)
    meta = {k: v for k, v in self.meta.items() if k != 'wall_clock'}
    with open(os.path.splitext(path)[0] + '.json', 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, sort_keys=True, indent=1)
        f.write('\n')

@d2l.add_to_class(d2l.EvalResult)  #@save
@classmethod
def load(cls, path):
    """An EvalResult written by save."""
    with open(os.path.splitext(path)[0] + '.json', encoding='utf-8') as f:
        meta = json.load(f)
    return cls(pd.read_parquet(path), meta)
```
