# Evaluation Suite Registry (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_eval_suites` explains in one sentence and does not show:
the dataset adapters `d2l.EVAL_ADAPTERS`, the task registry
`d2l.EVAL_TASKS` (with the written demonstrations `d2l.GSM8K_COT_DEMOS`
of lm-evaluation-harness's `gsm8k_cot`), the frozen suite manifests
`d2l.EVAL_SUITES`, the row loader `_eval_load_rows`, the task normalizer
`_eval_canonical`, the item renderer `_eval_items`, the run-cache
helpers `_eval_model_id` and `_eval_cache_*`, the manifest and
save/load methods of `d2l.EvalSuite` and `d2l.EvalResult`, the task kinds
the section does not run (`_eval_score_more`), `EvalResult.macro`, the
batching and fp32 read-out of `judge_pairwise` (`_judge_cut`,
`_judge_logodds`), and the chapter helper `lm_save_table` that writes a
regenerated table and checks it against the registered version it
replaces (`_lm_frozen_key`, `_lm_read_table`). Every block is
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
    header that precedes the prompt. With fmt='mmlu' and five
    'dev:same_subject' shots the prompt is lm-evaluation-harness's."""
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
    that split), {'demos': [[question, target], ...]} (written out in the
    task, 'gen' only; an optional 'source' names them) or
    '<split>:same_subject' (the first rows of that split with the item's
    subject)."""
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
        n_src = len(src['demos'] if 'demos' in src else src['ids'])
        if n_src != k:
            raise ValueError(f'task {task.id}: shots={k} but {n_src} '
                             'demonstration ids')
        if 'demos' in src:  # written out in the task: (question, target)
            if task.kind != 'gen':
                raise ValueError(f'task {task.id}: written demonstrations '
                                 "need kind 'gen'")
            fixed = [{'question': q, 'target': t} for q, t in src['demos']]
        else:
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
GSM8K_COT_DEMOS = [  # Wei et al. (2022), as lm-evaluation-harness's
    # gsm8k_cot task writes them (lm_eval/tasks/gsm8k/gsm8k-cot.yaml at
    # commit b733598568fde3afa0e008892f8e7bb18cd472b6)
    ['There are 15 trees in the grove. Grove workers will plant trees in '
     'the grove today. After they are done, there will be 21 trees. How '
     'many trees did the grove workers plant today?',
     'There are 15 trees originally. Then there were 21 trees after some '
     'more were planted. So there must have been 21 - 15 = 6. The answer '
     'is 6.'],
    ['If there are 3 cars in the parking lot and 2 more cars arrive, how '
     'many cars are in the parking lot?',
     'There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The '
     'answer is 5.'],
    ['Leah had 32 chocolates and her sister had 42. If they ate 35, how '
     'many pieces do they have left in total?',
     'Originally, Leah had 32 chocolates. Her sister had 42. So in total '
     'they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The '
     'answer is 39.'],
    ['Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has '
     '12 lollipops. How many lollipops did Jason give to Denny?',
     'Jason started with 20 lollipops. Then he had 12 after giving some to '
     'Denny. So he gave Denny 20 - 12 = 8. The answer is 8.'],
    ['Shawn has five toys. For Christmas, he got two toys each from his '
     'mom and dad. How many toys does he have now?',
     'Shawn started with 5 toys. If he got 2 toys each from his mom and '
     'dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.'],
    ['There were nine computers in the server room. Five more computers '
     'were installed each day, from monday to thursday. How many computers '
     'are now in the server room?',
     'There were originally 9 computers. For each of 4 days, 5 more '
     'computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is '
     '29. The answer is 29.'],
    ['Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On '
     'wednesday, he lost 2 more. How many golf balls did he have at the '
     'end of wednesday?',
     'Michael started with 58 golf balls. After losing 23 on tuesday, he '
     'had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf '
     'balls. The answer is 33.'],
    ['Olivia has $23. She bought five bagels for $3 each. How much money '
     'does she have left?',
     'Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = '
     '15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The '
     'answer is 8.']]

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
             fmt='mmlu', norm='sum', shots=5,
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
    # lm-evaluation-harness's gsm8k_cot: 8 written chain-of-thought
    # demonstrations as 'Q: ...\nA: ... The answer is N.', greedy, stop 'Q:',
    # 256 new tokens (the harness's max_gen_toks)
    EvalTask('gsm8k_cot', 'openai/gsm8k',
             '740312add88f781978c0658806c59bc2815b9866',
             ('main/test-00000-of-00001.parquet',
              'main/train-00000-of-00001.parquet'), 'gen', 'gsm8k',
             config='main', template='qa', shots=8,
             shots_from={'source': 'Wei et al. 2022',
                         'demos': GSM8K_COT_DEMOS},
             max_new_tokens=256, stop=('Q:',), extract='gsm8k'),
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
                 config='all', n=2000, fmt='mmlu', norm='sum', shots=5,
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

```{.python .input #eval-suites-lib-kinds}
%%tab pytorch
#@save
def _eval_score_more(suite, task, items, model, tokenizer, seed):
    """(scores, preds, finish reasons) of the kinds EvalSuite._score does
    not show: 'mc_partial' (one continuation after each of an item's
    prompts; the highest-scoring prompt is the prediction), 'code' (the
    prompt and its completion run against the item's tests) and 'ppl'
    (bits per byte per text)."""
    n = len(items)
    if task.kind == 'mc_partial':
        S = d2l.score_choices(
            model, tokenizer, [p for it in items for p in it['prompts']],
            [[it['continuation']] for it in items for _ in it['prompts']],
            task.norm, batch_size=suite.batch_size)[:, 0]
        preds = [int(s.argmax()) for s in S.split(
            [len(it['prompts']) for it in items])]
        return ([float(p == it['gold']) for p, it in zip(preds, items)],
                preds, [None] * n)
    if task.kind == 'code':
        sampling = dict(task.sampling or {})  # None: greedy
        if sampling:
            sampling.setdefault('strategy', 'sample')
        gens = d2l.generate(
            model, tokenizer, [it['prompt'] for it in items],
            task.max_new_tokens, chat=task.chat,
            enable_thinking=task.enable_thinking, stop=list(task.stop),
            batch_size=suite.batch_size, **sampling,
            seed=[d2l.item_seed(seed, task.id, it['id']) for it in items])
        return ([float(d2l.run_python_tests(it['prompt'] + g.text,
                                            it['test']))
                 for g, it in zip(gens, items)], [None] * n,
                [g.finish_reason for g in gens])
    if task.kind == 'ppl':  # the mean of texts, not per byte
        return ([d2l.bits_per_byte(model, tokenizer, [it['text']])
                 for it in items], [None] * n, [None] * n)
    raise ValueError(f'unknown kind: {task.kind!r}')

@d2l.add_to_class(d2l.EvalResult)  #@save
def macro(self, n_boot=10000, *, seed=0, alpha=0.05):
    """The mean of the task means, with a bootstrap that resamples items
    (or clusters) within each task."""
    reps = [d2l._bootstrap_replicates(df.score, n_boot,
                                      self._clusters(t, df.cluster),
                                      [seed, k])
            for k, (t, df) in enumerate(self.per_item.groupby('task'))]
    means = np.mean([m for _, m, _ in reps], axis=0)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return d2l.BootstrapResult(
        float(np.mean([y.mean() for y, _, _ in reps])),
        (float(lo), float(hi)), float(means.std(ddof=1)), None,
        sum(len(y) for y, _, _ in reps), len(reps), n_boot)
```

```{.python .input #eval-suites-lib-judge}
%%tab pytorch
#@save
def _judge_cut(tokenizer, text, i, name, limit=1500):
    """text cut to its first limit tokens, with a warning naming item i."""
    import warnings
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text
    warnings.warn(f'judge_pairwise: item {i}: {name} cut from {len(ids)} '
                  f'to {limit} tokens')
    return tokenizer.decode(ids[:limit])

#@save
def _judge_logodds(judge, tokenizer, texts, *, batch_size=8,
                   max_batch_tokens=None):
    """(len(texts), 3) float64: log P(A), log P(B) and the log-odds of the
    token 'A' over 'B' as the next token after each rendered prompt.
    Identical prompts run once; prompts are sorted by length, left-padded
    and batched (at most batch_size rows and, if given, max_batch_tokens
    tokens including padding). The log-odds are recomputed in fp32 from
    the input of the judge's output head, since bf16 logits are rounded
    (steps of 0.125 near 20); a head that does not give the logits (a
    soft-capped one) falls back to the logits as returned, with a
    warning."""
    import warnings
    letters = [tokenizer.encode(c, add_special_tokens=False) for c in 'AB']
    if any(len(ids) != 1 for ids in letters) or letters[0] == letters[1]:
        raise ValueError(f'A and B are not two single tokens: {letters}')
    AB = [letters[0][0], letters[1][0]]
    unique = list(dict.fromkeys(texts))
    ids = [tokenizer(t, add_special_tokens=False)['input_ids']
           for t in unique]
    order = sorted(range(len(ids)), key=lambda r: (-len(ids[r]), r))
    batches = [[]]
    for r in order:  # longest first, so a batch's first row sets its length
        b = batches[-1]
        if b and (len(b) == batch_size or max_batch_tokens is not None and
                  len(ids[b[0]]) * (len(b) + 1) > max_batch_tokens):
            batches.append([])
        batches[-1].append(r)
    pad = tokenizer.pad_token_id
    pad = tokenizer.eos_token_id if pad is None else pad
    device = next(judge.parameters()).device
    head = getattr(judge, 'get_output_embeddings', lambda: None)()
    seen, warned = [], False
    hook = (head.register_forward_hook(lambda m, x, y: seen.append(x[0]))
            if isinstance(head, nn.Linear) else None)
    lp = torch.empty(len(ids), 3, dtype=torch.float64)
    try:
        for b in batches if ids else []:
            T = len(ids[b[0]])
            X = torch.tensor([[pad] * (T - len(ids[r])) + ids[r] for r in b],
                             device=device)
            mask = torch.tensor([[0] * (T - len(ids[r])) + [1] * len(ids[r])
                                 for r in b], device=device)
            pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
            seen.clear()
            with torch.inference_mode():
                z = judge(input_ids=X, attention_mask=mask, position_ids=pos,
                          use_cache=False, logits_to_keep=1).logits[:, -1]
                zab = z[:, AB].float()
                if len(seen) == 1:
                    h = seen[0][:, -1].float()
                    w = head.weight[AB].to(h.device, torch.float32)
                    exact = h @ w.T if head.bias is None else (
                        h @ w.T + head.bias[AB].to(h.device, torch.float32))
                    exact = exact.to(zab.device)
                    if torch.allclose(exact, zab, rtol=2**-7, atol=1e-3):
                        zab = exact
                    elif not warned:  # a head that is not the last step
                        warned = True
                        warnings.warn('judge_pairwise: the output head does '
                                      'not give the logits; the log-odds '
                                      'use the logits as returned')
            lp[b, :2] = z.float().log_softmax(-1)[:, AB].double().cpu()
            lp[b, 2] = (zab[:, 0] - zab[:, 1]).double().cpu()
    finally:
        if hook is not None:
            hook.remove()
    row = {t: k for k, t in enumerate(unique)}
    return lp[[row[t] for t in texts]] if texts else lp
```

```{.python .input #eval-suites-lib-rerun}
%%tab pytorch
#@save
def _lm_frozen_key(key):
    """The registered artifact a regeneration of key is checked against:
    key itself if registered, else its highest registered earlier version
    (lm-x-v1 for lm-x-v2), else None."""
    if key in d2l.DATA_HUB:
        return key
    stem, _, v = key.rpartition('-v')
    if not v.isdigit():
        return None
    older = [k for k in d2l.DATA_HUB if k.rpartition('-v')[0] == stem and
             k.rpartition('-v')[2].isdigit() and int(k.rpartition('-v')[2])
             < int(v)]
    return max(older, key=lambda k: int(k.rpartition('-v')[2]),
               default=None)

#@save
def _lm_read_table(path):
    """A parquet file or JSON lines as a DataFrame."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_json(path, lines=True, dtype=False)

#@save
def lm_save_table(name, df, on, key, rerun_dir=os.path.join(
        '..', 'data', 'lm', 'rerun')):
    """Write a regenerated table to rerun_dir/name (parquet, or JSON lines
    for .jsonl) with its rows sorted by the columns on, print its sha1,
    and print its agreement with the registered artifact it replaces (see
    _lm_frozen_key): per column, the largest absolute difference of a
    float column, else the share of equal values. Returns the path."""
    df = df.sort_values(on, ignore_index=True)
    if name.endswith('.parquet'):
        write = lambda path: df.to_parquet(path, index=False)
    else:  # pandas converts the numpy values
        write = lambda path: d2l.write_jsonl(path, json.loads(df.to_json(
            orient='records', double_precision=15)))
    path = d2l.lm_save_rerun(name, write, rerun_dir)
    frozen = _lm_frozen_key(key)
    if frozen is None:
        print(f'{key}: no earlier version is registered')
        return path
    m = df.merge(_lm_read_table(d2l.download_keyed(frozen)), on=on,
                 suffixes=('', '_frozen'))
    agree = []
    for c in df.columns.difference(on):
        a, b = m[c], m.get(c + '_frozen')
        if b is None:
            continue
        if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
            agree.append(f'{c} max |diff| {np.nanmax(np.abs(a - b)):.2g}')
        else:
            missing = (a.isna() & b.isna()).to_numpy(bool)  # equal too
            same = np.mean(missing | (a.astype(object).astype(str) == b.astype(
                object).astype(str)).to_numpy(bool))
            agree.append(f'{c} equal {same:.3f}')
    print(f'against {frozen}: {len(m)} of {len(df)} rows matched')
    for k in range(0, len(agree), 3):
        print('    ' + '; '.join(agree[k:k + 3]))
    return path
```
