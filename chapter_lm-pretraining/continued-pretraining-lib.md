# Continued Pretraining: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is
never rendered or executed as a notebook: `make lib` only scans it for
`#@save` blocks, and `gen_notebooks.py` excludes it via `LIB_ONLY_FILES`.
It holds the library code of :numref:`sec_lm_continued_pretraining`
that the section explains in a sentence but does not show
(docs/code-guide.md 7.1):

- the section's registries: committed results (`PRETRAIN_ARTIFACTS`) and
  the long regeneration jobs (`PRETRAIN_JOBS`), each the page's call
  written a second time;
- the bookkeeping of `MixStream` (its state, iteration and window reads);
- the data: the book's train and held-out sections (`book_split`), the
  token shards of the two tiers (`cpt_data`; T1: the hosted R2-tokenized
  shards; T2: the local Qwen-tokenized shards, whose summary is the hosted
  manifest);
- per-document losses (`_doc_nll`), the evaluation sets and the suite;
- the arm driver `cpt_arms` (one resumable job folder per arm, the
  re-warmed schedule, fresh or restored AdamW moments, the evaluation
  grid, the equal-domain-token evaluation, the merge of the arms with the
  base's numbers), the D2 probe and the arms' plots;
- the frontier test behind the replay claim (`frontier_excess`);
- the SmolLM3 trajectory (`smollm3_trajectory`) and its plot.

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble inside its functions (`transformers`, `huggingface_hub`,
`fcntl`, `subprocess`, `unicodedata`).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_continued_pretraining`

## Registries

```{.python .input #continued-pretraining-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['cpt-qwen-manifest'] = (
    'lm-cpt-qwen-manifest-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['cpt-pes2o-12m'] = ('lm-pes2o-12m-v1', 'shards')
d2l.PRETRAIN_ARTIFACTS['cpt-pes2o-val'] = ('lm-pes2o-val-v1', 'shards')
d2l.PRETRAIN_ARTIFACTS['cpt-fwedu-replay-12m'] = (
    'lm-fwedu-replay-12m-v1', 'shards')
d2l.PRETRAIN_ARTIFACTS['cpt-t1'] = (
    'lm-cpt-t1-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['cpt-runs'] = (
    'lm-cpt-runs-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['smollm3-trajectory'] = (
    'lm-smollm3-trajectory-v1', 'json-table')
```

The long jobs that `tools/lm_regen.py` runs (one card each): the four Qwen
arms, their evaluation and merge, the six SmolLM3 checkpoints, and the T1
arms (which also run as a notebook regeneration, about 30 minutes).

```{.python .input #continued-pretraining-lib-jobs}
%%tab pytorch
#@save
d2l.PRETRAIN_JOBS['cpt-t1'] = ('cpt_arms', lambda: d2l._cpt_t1_call())
d2l.PRETRAIN_JOBS['cpt-qwen-A'] = ('cpt_arms', lambda: d2l._cpt_qwen_call())
d2l.PRETRAIN_JOBS['cpt-qwen-B'] = ('cpt_arms', lambda: d2l._cpt_qwen_call())
d2l.PRETRAIN_JOBS['cpt-qwen-C'] = ('cpt_arms', lambda: d2l._cpt_qwen_call())
d2l.PRETRAIN_JOBS['cpt-qwen-A2'] = ('cpt_arms',
                                    lambda: d2l._cpt_qwen_call())
d2l.PRETRAIN_JOBS['cpt-qwen-evals'] = ('cpt_arms',
                                       lambda: d2l._cpt_qwen_call())
d2l.PRETRAIN_JOBS['smollm3-0'] = ('smollm3_trajectory',
                                  lambda: d2l._smollm3_call())
d2l.PRETRAIN_JOBS['smollm3-1'] = ('smollm3_trajectory',
                                  lambda: d2l._smollm3_call())
d2l.PRETRAIN_JOBS['smollm3-2'] = ('smollm3_trajectory',
                                  lambda: d2l._smollm3_call())
d2l.PRETRAIN_JOBS['smollm3-3'] = ('smollm3_trajectory',
                                  lambda: d2l._smollm3_call())
d2l.PRETRAIN_JOBS['smollm3-4'] = ('smollm3_trajectory',
                                  lambda: d2l._smollm3_call())
d2l.PRETRAIN_JOBS['smollm3-5'] = ('smollm3_trajectory',
                                  lambda: d2l._smollm3_call())
```

## The Page's Calls

The configurations the page passes (the load check compares them with the
committed runs), the arm names of the Qwen regenerations, and the suite.

```{.python .input #continued-pretraining-lib-calls}
%%tab pytorch
#@save
CPT_TASKS = ('hellaswag', 'arc_easy', 'arc_challenge', 'piqa', 'sciq',
             'winogrande')

#@save
_CPT_QWEN_NAMES = {'A': 'cpt-qwen-A', 'B': 'cpt-qwen-B', 'C': 'cpt-qwen-C',
                   'A′': 'cpt-qwen-A2'}

#@save
def _cpt_qwen_call():
    """cpt_arms' arguments in the page's T2 cell."""
    arms = {'A': dict(peak_lr=1e-4, replay=0.0, data_seed=0),
            'B': dict(peak_lr=1e-4, replay=0.25, data_seed=0),
            'C': dict(peak_lr=1e-5, replay=0.0, data_seed=0),
            'A′': dict(peak_lr=1e-4, replay=0.0, data_seed=1)}
    return dict(arms=arms, base='Qwen/Qwen3-0.6B-Base', tokens=80_000_000,
                seq_len=2048, batch_tokens=262_144, warmup=10,
                final_frac=0.1)

def _cpt_t1_call():
    """cpt_arms' arguments in the page's T1 cell."""
    arms = {'A': dict(peak_lr=3e-4, replay=0.0, data_seed=0),
            'A′': dict(peak_lr=3e-4, replay=0.0, data_seed=1),
            'A″': dict(peak_lr=3e-4, replay=0.0, data_seed=2),
            'B': dict(peak_lr=3e-4, replay=0.25, data_seed=0),
            'cold': dict(peak_lr=None, replay=0.0, data_seed=0, warmup=0),
            'branch': dict(peak_lr=None, replay=0.0, data_seed=0, warmup=0,
                           start='plateau', schedule='linear')}
    return dict(arms=arms, base='r2', tokens=12_000_000, seq_len=1024,
                batch_tokens=65_536)

def cpt_qwen_runs():
    """The four Qwen3-0.6B-Base arms as the section's T2 cell computes
    them (cpt_arms); while they are pending, the T1 arms under a printed
    label."""
    return cpt_arms(**_cpt_qwen_call())
```

## Mixed Streams: State and Window Reads

`MixStream` (on the page) keeps its cursors as a table computed from the
seed alone; its state is the seed, the step and the layout of its streams,
so a resumed run rebuilds the same cursors. Stream k's j-th window is
entry j of its per-epoch permutation of all its windows (the permutation
`PackedStream` itself uses), so the windows a stream gives a mixture come
in the same order whatever the mixture's weights.

```{.python .input #continued-pretraining-lib-mix}
%%tab pytorch
#@save
def _epoch_perm(stream, e):
    """The permutation of all windows of stream in its epoch e (the one
    PackedStream's own steps take slices of), cached per stream."""
    key = (stream.seed, stream.epoch + int(e))
    cached = getattr(stream, '_perm', None)
    if cached is None or cached[0] != key:
        stream._perm = (key, np.random.default_rng(list(key)).permutation(
            stream.n_seq))
    return stream._perm[1]

def _stream_windows(stream, start, n):
    """Windows start, ..., start + n - 1 of stream's permutation order
    (epochs follow each other), as an int64 array (n, seq_len + 1)."""
    if n == 0:
        return np.empty((0, stream.seq_len + 1), dtype=np.int64)
    pos = np.arange(start, start + n, dtype=np.int64)
    idx = np.empty(n, dtype=np.int64)
    for e in np.unique(pos // stream.n_seq):
        sel = pos // stream.n_seq == e
        idx[sel] = _epoch_perm(stream, e)[pos[sel] % stream.n_seq]
    return d2l._windows(stream, idx)

def _window_batch(W, eos_id):
    """A PackedStream batch from windows W (n, T + 1)."""
    W = torch.from_numpy(np.ascontiguousarray(W))
    X, Y = W[:, :-1].contiguous(), W[:, 1:].clone()
    Y[X == eos_id] = -100  # the next token starts a document
    doc_ids, pos_ids = d2l.doc_ids_from_eos(X.int(), eos_id)
    return dict(input_ids=X, labels=Y, doc_ids=doc_ids, pos_ids=pos_ids)

def _mix_tokens(mix, step, streams):
    """Tokens the streams with the given indices gave before step."""
    return int(mix.cursors(step)[list(streams)].sum()) * mix.seq_len
```

```{.python .input #continued-pretraining-lib-mix-state}
%%tab pytorch
@d2l.add_to_class(d2l.MixStream)  #@save
def __iter__(self):
    while True:  # the cursor: step counts the batches consumed
        self.step += 1
        yield self.batch(self.step - 1)

@d2l.add_to_class(d2l.MixStream)  #@save
def state_dict(self):
    return dict(version=1, kind='mix', seed=self.seed, step=self.step,
                weights=[float(w) for w in self.weights],
                global_batch=self.global_batch, seq_len=self.seq_len,
                streams=[[s.layout_hash, s.seed, s.epoch]
                         for s in self.streams])

@d2l.add_to_class(d2l.MixStream)  #@save
def load_state_dict(self, state):
    mine = self.state_dict()
    for key in ('kind', 'weights', 'global_batch', 'seq_len', 'streams'):
        if state.get(key) != mine[key]:
            raise ValueError(f'{key}: {mine[key]} != {state.get(key)}')
    if state['seed'] != self.seed:  # the cursors follow from the seed
        self._cum = self._cum[:1]
    self.seed, self.step = state['seed'], state['step']
```

## The Data

The domain stream is peS2o (scientific prose) plus the book's train
sections, nine to one; replay is FineWeb-Edu file 005; the evaluation sets
are peS2o's validation split, FineWeb-Edu's validation shard and the
held-out book sections. T1 reads the hosted R2-tokenized shards (or, before
their upload, the builder's local output); T2 reads the local
Qwen-tokenized shards (`uint32`, never hosted), built by
`tools/build_pretraining_data.py --stage cpt-qwen` the first time a
regeneration needs them. The book's shards are built from the book corpus
on first use under either tokenizer (seconds; deterministic).

```{.python .input #continued-pretraining-lib-data}
%%tab pytorch
#@save
CptData = collections.namedtuple('CptData', ['table', 'shards', 'label'])

#@save
_CPT_STREAMS = (  # key, row label, source, licence
    ('pes2o', 'domain: peS2o', 'peS2o v2 train', 'ODC-By'),
    ('book', 'domain: book (train)', 'book sections', 'CC BY-SA 4.0'),
    ('replay', 'replay: FineWeb-Edu', 'FineWeb-Edu file 005', 'ODC-By'),
    ('val_domain', 'val: peS2o', 'peS2o v2 validation', 'ODC-By'),
    ('val_general', 'val: FineWeb-Edu', 'FineWeb-Edu validation',
     'ODC-By'),
    ('val_book', 'held-out book', 'every fifth section', 'CC BY-SA 4.0'))

#@save
_CPT_DIRS = {'t1': os.path.join('..', 'data', 'lm', 'cpt-t1-v1'),
             't2': os.path.join('..', 'data', 'lm', 'cpt-qwen-v1')}

#@save
_CPT_T1_KEYS = {'pes2o': 'cpt-pes2o-12m', 'val_domain': 'cpt-pes2o-val',
                'replay': 'cpt-fwedu-replay-12m'}

#@save
def book_split(every=5):
    """The book's sections, sorted by document id, as (train, held_out)
    lists of (doc_id, text): every fifth section is held out."""
    with d2l._quiet():
        corpus = d2l.book_corpus()
    docs = sorted(corpus.docs.values(), key=lambda d: d['doc_id'])
    pairs = [(d['doc_id'], d['text']) for d in docs]
    held = [p for i, p in enumerate(pairs) if i % every == every - 1]
    return [p for i, p in enumerate(pairs) if i % every != every - 1], held

def _cpt_tokenizer(tier):
    """The tokenizer of a tier: the reference tokenizer (T1) or Qwen3's."""
    with d2l._quiet():
        if tier == 't1':
            return d2l.r2_tokenizer()
        import transformers
        name = 'Qwen/Qwen3-0.6B-Base'
        return transformers.AutoTokenizer.from_pretrained(
            name, revision=d2l.HF_REVISIONS[name])

def _locked(path):
    """An exclusive file lock (a context manager) for a build that several
    processes may start at once."""
    import fcntl

    @contextlib.contextmanager
    def lock():
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    return lock()

def _book_shards(tier):
    """{'book': train sections, 'val_book': held-out sections} as
    TokenShards under the tier's tokenizer, built on first use."""
    tag = 'r2' if tier == 't1' else 'qwen'
    root = os.path.join('..', 'data', 'lm', f'cpt-book-{tag}-v1')
    out = {}
    with _locked(root + '.lock'):
        for key, docs in zip(('book', 'val_book'), book_split()):
            folder = os.path.join(root, key)
            if not os.path.exists(os.path.join(folder, 'manifest.json')):
                tmp = f'{folder}.{os.getpid()}.tmp'
                shutil.rmtree(tmp, ignore_errors=True)
                with d2l._quiet():
                    d2l.write_shards(docs, _cpt_tokenizer(tier), tmp,
                                     'train', shard_tokens=10 ** 9)
                os.makedirs(root, exist_ok=True)
                os.replace(tmp, folder)
            out[key] = d2l.TokenShards(folder, 'train')
    return out

def _cpt_t1_shards():
    """The T1 shards: the hosted ones, or the builder's local output before
    their upload."""
    out = {}
    for key, name in _CPT_T1_KEYS.items():
        local = os.path.join(_CPT_DIRS['t1'], key)
        if d2l.PRETRAIN_ARTIFACTS[name][0] in d2l.DATA_HUB:
            out[key] = d2l.load_artifact(name)
        elif os.path.exists(os.path.join(local, 'manifest.json')):
            out[key] = d2l.TokenShards(local, 'train')
        else:
            raise KeyError(f'{name} is not committed yet; build it with '
                           'tools/build_pretraining_data.py --stage cpt-t1')
    out['val_general'] = d2l.reference_shards('val')
    return dict(out, **_book_shards('t1'))

def _cpt_qwen_shards():
    """The T2 shards (Qwen tokens, local), built through the builder when
    missing; checked against the committed manifest when there is one."""
    import subprocess
    import sys
    keys = [k for k, *_ in _CPT_STREAMS if k not in ('book', 'val_book')]
    root = _CPT_DIRS['t2']
    have = lambda: all(os.path.exists(os.path.join(root, k,
                                                   'manifest.json'))
                       for k in keys)
    with _locked(root + '.lock'):
        if not have():
            repo = os.path.dirname(os.path.dirname(os.path.abspath(
                d2l.__file__)))
            subprocess.run([sys.executable, os.path.join(
                repo, 'tools', 'build_pretraining_data.py'), '--stage',
                'cpt-qwen'], check=True, cwd=repo, stdout=subprocess.DEVNULL)
    out = {k: d2l.TokenShards(os.path.join(root, k), 'train') for k in keys}
    if 'lm-cpt-qwen-manifest-v1' in d2l.DATA_HUB:
        layout = d2l.load_artifact('cpt-qwen-manifest').layout
        for k, s in out.items():
            if d2l._layout_hash(s) != layout[k]:
                raise ValueError(f'{k}: the local shards differ from the '
                                 'committed manifest; rebuild them')
    return dict(out, **_book_shards('t2'))

def qwen3_val_tokens(n=4, T=512):
    """n sequences of T Qwen3 tokens, each the start of one FineWeb-Edu
    validation document of at least T tokens, as a LongTensor (n, T)."""
    import gzip
    tok, out = _cpt_tokenizer('t2'), []
    root = d2l.reference_shards('val').root
    with gzip.open(os.path.join(root, 'val.jsonl.gz'), 'rt') as f:
        for line in f:
            ids = tok(json.loads(line)['text'])['input_ids']
            if len(ids) >= T:
                out.append(ids[:T])
            if len(out) == n:
                break
    return torch.tensor(out)

def _shard_table(shards):
    """stream, documents, tokens, source, licence of each stream."""
    rows = [(label, shards[k].n_docs, shards[k].n_tokens, src, lic)
            for k, label, src, lic in _CPT_STREAMS]
    return pd.DataFrame(rows, columns=['stream', 'documents', 'tokens',
                                       'source', 'licence'])

def _cpt_qwen_regenerating():
    names = set(_CPT_QWEN_NAMES.values()) | {'cpt-qwen-evals'}
    return bool(d2l._regenerating() & names)

def cpt_data(tier='t2'):
    """The arms' data: CptData(table, shards, label). table: per stream
    (domain peS2o, book train sections, replay FineWeb-Edu, the three
    evaluation sets) documents, tokens, source and licence. 't1': the
    R2-tokenized shards. 't2': in a normal run the committed manifest's
    table (no shard is opened, built or downloaded; shards is None); in a
    regeneration of the Qwen arms the local Qwen-tokenized shards."""
    if tier == 't1':
        shards = _cpt_t1_shards()
        return CptData(_shard_table(shards), shards, '')
    if tier != 't2':
        raise ValueError(f"tier is 't1' or 't2', not {tier!r}")
    if _cpt_qwen_regenerating():
        shards = _cpt_qwen_shards()
        return CptData(_shard_table(shards), shards, '')
    if 'lm-cpt-qwen-manifest-v1' in d2l.DATA_HUB:
        return CptData(d2l.load_artifact('cpt-qwen-manifest').table, None,
                       '')
    if d2l._regenerating():
        return d2l._show_label(CptData(pd.DataFrame(columns=[
            'stream', 'documents', 'tokens', 'source', 'licence']), None,
            'cpt-qwen-manifest pending'))
    raise KeyError('cpt-qwen-manifest is not committed yet')
```

## Losses per Document and the Suite

The evaluation reads the first `tokens` tokens of an evaluation set in
windows of the training length with document masks, as training does, and
adds each target's NLL to the document it belongs to (a document's EOS
belongs to it). A set's token-weighted loss is the sum of its documents'
NLL over the sum of their scored tokens.

```{.python .input #continued-pretraining-lib-losses}
%%tab pytorch
#@save
_CPT_EVAL_TOKENS = 2 ** 19  # the evaluations during training, per set

#@save
_CPT_FULL_TOKENS = 2 ** 21  # the full evaluation, per set (2.1M tokens)

#@save
_CPT_SETS = {'domain': 'val_domain', 'general': 'val_general',
             'book': 'val_book'}

#@save
def _token_nll(H, W, Y, chunk=2048):
    """Per-target NLL in fp32 from hidden states and the output weight,
    chunk positions at a time (0 where the target is ignored)."""
    h, y = H.flatten(0, -2), Y.flatten()
    out = torch.zeros(len(y), dtype=torch.float32, device=H.device)
    for i in range(0, len(y), chunk):
        logits = F.linear(h[i:i + chunk], W).float()
        out[i:i + chunk] = F.cross_entropy(logits, y[i:i + chunk],
                                           ignore_index=-100,
                                           reduction='none')
    return out.view(Y.shape)

@torch.no_grad()
def _doc_nll(model, shards, T, tokens=None, batch=8, chunk=2048):
    """Per-document summed NLL and scored targets (doc_id, sum_nll,
    n_tokens) on the first `tokens` tokens of shards (all if None), in
    windows of T tokens with document masks; the model in its own dtype
    under bf16 autocast on a GPU."""
    n_win = (shards.n_tokens - 1) // T
    if tokens is not None:  # tokens // T windows: full batches, one compile
        n_win = min(n_win, max(1, tokens // T))
    eos = shards.eos_id
    toks = shards.read(0, n_win * T + 1).astype(np.int64)
    doc = np.concatenate([[0], np.cumsum(toks == eos)[:-1]])
    dev = next(model.parameters()).device
    nll, cnt = np.zeros(doc[-1] + 1), np.zeros(doc[-1] + 1)
    was_training = model.training
    model.eval()
    for i in range(0, n_win, batch):
        rows = np.arange(i, min(i + batch, n_win))
        Wn = torch.from_numpy(np.stack([toks[j * T:(j + 1) * T + 1]
                                        for j in rows])).to(dev)
        X, Y = Wn[:, :-1], Wn[:, 1:].clone()
        Y[X == eos] = -100
        ctx = d2l.PackCtx(*d2l.doc_ids_from_eos(X.int(), eos))
        with torch.autocast(dev.type, torch.bfloat16,
                            enabled=dev.type == 'cuda'):
            H, Wt = model(X, ctx, return_hidden=True)
        loss = _token_nll(H, Wt, Y, chunk).cpu().numpy()
        valid = (Y != -100).cpu().numpy()
        d = doc[np.add.outer(rows * T, np.arange(1, T + 1))][valid]
        nll += np.bincount(d, weights=loss[valid], minlength=len(nll))
        cnt += np.bincount(d, minlength=len(cnt))
    model.train(was_training)
    keep = cnt > 0
    return pd.DataFrame(dict(doc_id=np.flatnonzero(keep), sum_nll=nll[keep],
                             n_tokens=cnt[keep].astype(np.int64)))

def _cpt_docs(model, shards, T, tokens):
    """Per-document rows (set, doc_id, sum_nll, n_tokens) of the three
    evaluation sets."""
    return pd.concat([_doc_nll(model, shards[k], T, tokens).assign(set=s)
                      for s, k in _CPT_SETS.items()], ignore_index=True)[
        ['set', 'doc_id', 'sum_nll', 'n_tokens']]

def _set_losses(docs):
    """{'<set>_loss': token-weighted loss} of per-document rows."""
    g = docs.groupby('set')[['sum_nll', 'n_tokens']].sum()
    return {f'{s}_loss': float(g.loc[s, 'sum_nll'] / g.loc[s, 'n_tokens'])
            for s in _CPT_SETS if s in g.index}

def suite_mean(items, by=('arm', 'step')):
    """The suite's mean accuracy (the mean over tasks of each task's mean
    score) per group of items."""
    by = list(by)
    per_task = items.groupby(by + ['task'])['score'].mean()
    return per_task.groupby(level=by).mean()

def _cpt_suite(tasks, n_max=None, n_task=None, name='cpt', batch_size=16):
    """An EvalSuite of tasks with at most n_max items each (all if None);
    n_task {task: n} sets single tasks; subsets seeded with 0."""
    out = []
    for t in tasks:
        task = d2l.EVAL_TASKS[t]
        n = (n_task or {}).get(t, n_max)
        if n is not None:
            with d2l._quiet():
                total = len(d2l._eval_load_rows(d2l._eval_canonical(task)))
            task = task._replace(n=min(n, total), subset_seed=0)
        out.append(task)
    with d2l._quiet():
        return d2l.EvalSuite(out, name=name, batch_size=batch_size)

def _suite_items(suite, model, tokenizer):
    """Per-item scores (task, item_id, score) of model on suite, under bf16
    autocast on a GPU (the model may hold fp32 master weights)."""
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with d2l._quiet(), torch.autocast(dev.type, torch.bfloat16,
                                      enabled=dev.type == 'cuda'):
        res = suite.run(model, tokenizer)
    model.train(was_training)
    return res.per_item[['task', 'item_id', 'score']].reset_index(drop=True)
```

## The Arms

Each arm trains in its own job folder under `../data/lm/rerun/ch21/cpt/`
with `_PretrainRun` (the state, logging and resume machinery of
`d2l.pretrain`): full states every 50 steps (deleted once the arm is done)
and the final weights in bf16. The step is `pretrain_step` on one GPU with
the loss chunk of a 151,936-token vocabulary (2,048 positions, so a chunk's
fp32 logits stay at 1.2 GB); the schedule is linear warmup, then a cosine
to `final_frac` of the peak (or, for the plateau branch, a linear decay to
zero). Interim evaluations (0.5M tokens per set, losses only) run every
three steps for the first 30 steps (the re-warm bump), then every 15 (T2)
or 5 (T1) steps. Arm A also runs the suite at step 15; the arms without
replay at the replay arm's peak get a full evaluation at the step where
they have seen as many domain tokens as the replay arm sees in its whole
run (equal-token accounting). The merge evaluates the base and every arm's
final weights on the full evaluation sets and the suite.

```{.python .input #continued-pretraining-lib-arms}
%%tab pytorch
#@save
CptRuns = collections.namedtuple(
    'CptRuns', ['table', 'logs', 'base_docs', 'docs', 'equal', 'items',
                'label'])

#@save
_CPT_COLUMNS = {
    'table': ['arm', 'peak_lr', 'replay', 'steps', 'domain_tokens',
              'domain_loss', 'general_loss', 'book_loss', 'suite', 'tok_s',
              'gib'],
    'logs': ['arm', 'step', 'tokens', 'domain_tokens', 'lr', 'loss',
             'grad_norm', 'domain_loss', 'general_loss', 'book_loss'],
    'base_docs': ['set', 'doc_id', 'sum_nll', 'n_tokens'],
    'docs': ['arm', 'set', 'doc_id', 'sum_nll', 'n_tokens'],
    'equal': ['arm', 'step', 'domain_tokens', 'domain_loss',
              'general_loss', 'book_loss'],
    'items': ['arm', 'step', 'task', 'item_id', 'score']}

#@save
_CPT_MICRO = {'t1': 16, 't2': 1}  # sequences per micro-batch (D2 probe:
# Qwen3-0.6B with fp32 master weights runs out of 24 GB at 2 x 2,048)

#@save
_CPT_SUITE_STEP = 15  # arm A's early look at the suite

#@save
def _arm_slug(arm):
    return {'A′': 'A2', 'A″': 'A3'}.get(arm, arm)

def _cpt_eval_steps(steps, tier, early=30):
    """Interim evaluation steps: every 3 up to `early`, then every 15 (T2)
    or 5 (T1), and the last."""
    every = 15 if tier == 't2' else 5
    return sorted(set(range(3, min(early, steps) + 1, 3)) | set(range(
        every, steps + 1, every)) | {steps})

def _cpt_configs(arms, base, tokens, seq_len, batch_tokens, warmup,
                 final_frac):
    """One JSON-able configuration per arm (defaults filled in)."""
    tier = 't2' if base.startswith('Qwen/') else 't1'
    steps, B = tokens // batch_tokens, batch_tokens // seq_len
    replay = {a.get('peak_lr'): a['replay'] for a in arms.values()
              if a.get('replay', 0) > 0}
    out = {}
    for name, a in arms.items():
        a = dict(dict(peak_lr=None, replay=0.0, data_seed=0, warmup=warmup,
                      start='final', schedule='cosine'), **a)
        r = replay.get(a['peak_lr']) if a['replay'] == 0 else None
        out[name] = dict(
            tier=tier, arm=name, base=base, steps=steps, seq_len=seq_len,
            global_batch=B, micro_batch=_CPT_MICRO[tier],
            final_frac=final_frac, z_loss=0.0 if tier == 't2' else 1e-4,
            equal_step=(round(steps * (1 - r)) if r and a['start'] ==
                        'final' else None),
            suite_step=_CPT_SUITE_STEP if name == 'A' else None, **a)
    return out

def _cpt_lr(step, cfg, peak):
    """The arm's learning rate at step (0-based)."""
    total, warmup = cfg['steps'], cfg['warmup']
    if step < warmup:
        return peak * (step + 1) / warmup
    if cfg['schedule'] == 'linear':  # the plateau branch: decay to zero
        return peak * (total - step) / (total - warmup)
    final = cfg['final_frac'] * peak
    t = (step - warmup) / max(1, total - warmup)
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * t))

def _cpt_step(model, optimizer, micro_batches, cfg, n_valid, chunk=2048):
    """pretrain_step on one GPU with the loss chunk of a large vocabulary."""
    nll = 0.0
    for mb in micro_batches:
        ctx = d2l.PackCtx(mb['doc_ids'], mb['pos_ids'])
        with torch.autocast(mb['input_ids'].device.type, torch.bfloat16,
                            enabled=cfg.dtype == 'bf16'):
            H, W = model(mb['input_ids'], ctx, return_hidden=True)
            loss, _, part = d2l.chunked_lm_loss_and_nll(
                H, W, mb['labels'], chunk=chunk, z_loss=cfg.z_loss)
        (loss / n_valid).backward()
        nll = nll + part.detach()
    params = [p for g in optimizer.param_groups for p in g['params']]
    norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip or math.inf)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=nll / n_valid, grad_norm=norm,
                clipped=norm > (cfg.clip or math.inf))

def _arch_of(model, T):
    """The Arch of a PackedGPT at sequence length T."""
    c = model.config()
    return d2l.Arch(c['num_blks'], c['num_hiddens'], c['num_heads'],
                    c['num_kv_heads'], c['head_dim'], c['ffn_hidden'],
                    c['vocab_size'], T, c['rope_theta'], c['qk_norm'],
                    c['tie_embeddings'])

def _cpt_r2_files():
    """(preset, final bf16 checkpoint, plateau state) of the T1 base: the
    reference run's (committed, else its job folder), else the t1 run's."""
    for preset, key, last, plateau in (
            ('r2', 'lm-r2-s6000-v1', 6000, 5100),
            ('t1', 'lm-r2t1-s1526-v1', 1526, 1297)):
        state_key = key.replace(f's{last}', f's{plateau}-state')
        job = os.path.join(d2l._job_root(), f'{preset}-s0')
        with d2l._quiet():
            ckpt = (d2l.download_keyed(key) if key in d2l.DATA_HUB else
                    os.path.join(job, 'ckpt', f's{last}.safetensors'))
            state = (d2l.download_keyed(state_key) if state_key in
                     d2l.DATA_HUB else d2l._state_path(job, plateau))
        if os.path.isdir(state):
            state = next(os.path.join(state, f) for f in sorted(
                os.listdir(state)) if f.endswith('.pt'))
        if os.path.exists(ckpt) and os.path.exists(state):
            return preset, ckpt, state
    raise FileNotFoundError('the T1 arms start from the reference run (or '
                            "the t1 run): neither one's final checkpoint "
                            'and plateau state are available')

def _cpt_base(cfg, dtype=torch.float32, device='cpu'):
    """(model, AdamW state or None, the base run's peak rate, label) of an
    arm's starting point: Qwen3-0.6B-Base, or the reference model's final
    checkpoint (fresh moments) or its plateau state (restored moments)."""
    if cfg['tier'] == 't2':
        with d2l._quiet():
            q = d2l.load_qwen3(cfg['base'], dtype=dtype, device=device)
        return d2l.PackedGPT.adopt(q), None, None, ''
    preset, ckpt, state_path = _cpt_r2_files()
    label = '' if preset == 'r2' else 't1 preset'
    if cfg.get('start') == 'plateau':
        state = torch.load(state_path, map_location='cpu',
                           weights_only=False)
        model = d2l.PackedGPT.from_arch(d2l.Arch(**state['cfg']['arch']))
        model.load_state_dict(state['model'])
        return (model.to(device, dtype), state['optim'],
                state['cfg']['peak_lr'], label)
    model = d2l._load_bf16(ckpt, dtype, device)
    return model, None, d2l.r2_config(preset).peak_lr, label

def _cpt_job(cfg):
    return os.path.join(d2l._job_root(), 'cpt',
                        f'{cfg["tier"]}-{_arm_slug(cfg["arm"])}')

def _cpt_train(model, mix, sets, cfg, job, optim_state=None, peak=None,
               tokenizer=None, suite=None, stop=None):
    """Train one arm (resumable in job; at most `stop` steps): the log
    rows, with the interim losses on the evaluation grid (unless
    cfg['evaluate'] is False); arm A's suite at its suite step and the
    equal-token per-document losses go to files in job."""
    dev = next(model.parameters()).device
    T = cfg['seq_len']
    pc = d2l.PretrainConfig(
        f'cpt-{_arm_slug(cfg["arm"])}', _arch_of(model, T), cfg['steps'],
        cfg['global_batch'], cfg['micro_batch'], peak, cfg['warmup'], 0,
        z_loss=cfg['z_loss'], compile=dev.type == 'cuda', probe_every=0,
        stats_every=0, val_every=0, ckpt_steps=(cfg['steps'],),
        state_every=50, seed=cfg['data_seed'], out_dir=job)
    run = d2l._PretrainRun(model, mix, pc, None, 'auto')
    if run.start == 0 and optim_state is not None:  # the plateau's moments
        run.optimizer.load_state_dict(optim_state)
    grid = (set(_cpt_eval_steps(cfg['steps'], cfg['tier']))
            if cfg.get('evaluate', True) else set())
    for seg in run.segments():
        for step in range(seg.start, min(seg.total, stop or seg.total)):
            lr = _cpt_lr(step, cfg, peak)
            for group in run.optimizer.param_groups:
                group['lr'] = lr
            t0 = time.perf_counter()
            batch, micro, n_valid = run.batch(step)
            t1 = time.perf_counter()
            out = _cpt_step(run.model, run.optimizer, micro, pc, n_valid)
            done = step + 1
            row = dict(step=done, tokens=done * run.step_tokens, lr=lr,
                       domain_tokens=_mix_tokens(mix, done, (0, 1)), **out)
            if done in grid:
                row.update(_set_losses(_cpt_docs(run.raw, sets, T,
                                                 _CPT_EVAL_TOKENS)))
            if done == cfg.get('equal_step'):
                d2l._write_result(os.path.join(job, 'equal.json.gz'),
                                  'json-table', _cpt_docs(
                                      run.raw, sets, T, _CPT_FULL_TOKENS),
                                  dict(config=None, label=''))
            if done == cfg.get('suite_step'):
                d2l._write_result(os.path.join(job, 'suite.json.gz'),
                                  'json-table', _suite_items(
                                      suite, run.raw, tokenizer),
                                  dict(config=None, label=''))
            run.record(seg, row, batch, None, t1 - t0, t0)
            if run.stopped(row):
                raise FloatingPointError(f'{job}: non-finite loss at '
                                         f'step {done}')
    return run.finish()

def _cpt_mix(shards, cfg):
    """The arm's MixStream: peS2o, book and replay windows in the shares
    (0.9 (1 - r), 0.1 (1 - r), r), data seed for streams and mixture."""
    T, B, seed, r = (cfg['seq_len'], cfg['global_batch'], cfg['data_seed'],
                     cfg['replay'])
    streams = [d2l.PackedStream(shards[k], T, B, seed=seed)
               for k in ('pes2o', 'book', 'replay')]
    return d2l.mix_streams(streams, (0.9 * (1 - r), 0.1 * (1 - r), r), B,
                           seed=seed)

def _cpt_suite_n(tier):
    return None if tier == 't2' else 2000  # T1: 2,000 items per task

def _cpt_arm_run(cfg):
    """Train one arm (or read its finished job folder): {log, equal,
    suite, peak_lr, gib, tok_s, label}."""
    job = _cpt_job(cfg)
    done = os.path.join(job, 'DONE')
    if not os.path.exists(done):
        os.makedirs(job, exist_ok=True)
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch._dynamo.reset()  # one process trains several arms: compile
        model, optim, old_peak, label = _cpt_base(cfg, device=dev)  # anew
        model.train()
        peak = cfg['peak_lr'] if cfg['peak_lr'] is not None else old_peak
        shards = cpt_data(cfg['tier']).shards
        tok = suite = None
        if cfg['suite_step']:
            tok, suite = _cpt_tokenizer(cfg['tier']), _cpt_suite(
                d2l.CPT_TASKS, n_max=_cpt_suite_n(cfg['tier']))
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(dev)
        with d2l._quiet():
            _cpt_train(model, _cpt_mix(shards, cfg), shards, cfg, job,
                       optim, peak, tok, suite)
        gib = (torch.cuda.max_memory_allocated(dev) / 2 ** 30
               if dev.type == 'cuda' else float('nan'))
        with open(done, 'w') as f:
            json.dump(dict(peak_lr=peak, gib=gib, label=label), f)
        shutil.rmtree(os.path.join(job, 'state'), ignore_errors=True)
        del model
        d2l.lm_free()
    with open(done) as f:
        info = json.load(f)
    log = d2l._read_job_logs(job)
    read = lambda name: (d2l._read_result(os.path.join(job, name),
                                          'json-table')
                         if os.path.exists(os.path.join(job, name))
                         else pd.DataFrame())
    steady = log.loc[log['step'] > 20, 'tok_s']
    return dict(log=log, equal=read('equal.json.gz'),
                suite=read('suite.json.gz'), peak_lr=info['peak_lr'],
                gib=info['gib'], tok_s=float(steady.median()),
                label=info['label'])

def _part(p):
    """A part as a dict, whether computed or read back from its file."""
    return p._asdict() if hasattr(p, '_asdict') else dict(p)

def _cpt_merge(parts, cfgs):
    """CptRuns from the arms' parts {arm: part}: the base and each arm's
    final weights on the full evaluation sets and the suite."""
    cfg0 = next(iter(cfgs.values()))
    tier, T = cfg0['tier'], cfg0['seq_len']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shards, tok = cpt_data(tier).shards, _cpt_tokenizer(tier)
    suite = _cpt_suite(d2l.CPT_TASKS, n_max=_cpt_suite_n(tier))
    torch._dynamo.reset()
    base, _, _, label = _cpt_base(dict(cfg0, start='final'),
                                  torch.bfloat16, dev)
    base_docs = _cpt_docs(base, shards, T, _CPT_FULL_TOKENS)
    interim = _set_losses(_cpt_docs(base, shards, T, _CPT_EVAL_TOKENS))
    logs = [pd.DataFrame([dict(arm='base', step=0, tokens=0,
                               domain_tokens=0, **interim)])]
    items = [_suite_items(suite, base, tok).assign(arm='base', step=0)]
    del base
    d2l.lm_free()
    rows = [dict(arm='base', steps=0, domain_tokens=0,
                 **_set_losses(base_docs))]
    docs, equal = [], []
    for arm, cfg in cfgs.items():
        p = _part(parts[arm])
        path = os.path.join(_cpt_job(cfg), 'ckpt',
                            f's{cfg["steps"]}.safetensors')
        model = d2l._load_bf16(path, device=dev)
        d = _cpt_docs(model, shards, T, _CPT_FULL_TOKENS)
        docs.append(d.assign(arm=arm))
        items.append(_suite_items(suite, model, tok).assign(
            arm=arm, step=cfg['steps']))
        del model
        d2l.lm_free()
        log = p['log']
        rows.append(dict(arm=arm, peak_lr=p['peak_lr'], replay=cfg['replay'],
                         steps=cfg['steps'], domain_tokens=int(
                             log['domain_tokens'].iloc[-1]),
                         tok_s=p['tok_s'], gib=p['gib'], **_set_losses(d)))
        logs.append(log.assign(arm=arm))
        if len(p['equal']):
            s = cfg['equal_step']
            equal.append(dict(arm=arm, step=s, domain_tokens=int(log.loc[
                log['step'] == s, 'domain_tokens'].iloc[0]),
                **_set_losses(p['equal'])))
        if len(p['suite']):
            items.append(p['suite'].assign(arm=arm, step=cfg['suite_step']))
    items = pd.concat(items, ignore_index=True)
    final = items[items['step'].isin([0, cfg0['steps']])]
    table = pd.DataFrame(rows)
    table['suite'] = table['arm'].map(suite_mean(final, by=('arm',)))
    C = _CPT_COLUMNS
    return CptRuns(table.reindex(columns=C['table']),
                   pd.concat(logs, ignore_index=True).reindex(
                       columns=C['logs']),
                   base_docs, pd.concat(docs, ignore_index=True)[C['docs']],
                   pd.DataFrame(equal, columns=C['equal']),
                   items[C['items']], label)

def cpt_arms(arms, base, tokens, seq_len, batch_tokens, warmup=10,
             final_frac=0.1):
    """Continue base on the domain stream (90% peS2o, 10% the book's train
    sections) with a replay share of FineWeb-Edu, one run per arm, and
    evaluate it: CptRuns(table, logs, base_docs, docs, equal, items,
    label). arms = {name: dict(peak_lr, replay, data_seed[, warmup, start,
    schedule])}; peak_lr None is the base run's own peak; start 'plateau'
    restores the reference run's plateau state (weights and AdamW moments)
    instead of its final checkpoint; schedule 'cosine' (to final_frac of
    the peak) or 'linear' (to zero). base 'Qwen/Qwen3-0.6B-Base' (T2: one
    regeneration per arm, cpt-qwen-A, -B, -C, -A2, merged by
    cpt-qwen-evals) or 'r2' (T1: the regeneration cpt-t1). Otherwise the
    committed result; while the Qwen arms are pending, the T1 arms under a
    printed label."""
    config = dict(arms=arms, base=base, tokens=tokens, seq_len=seq_len,
                  batch_tokens=batch_tokens, warmup=warmup,
                  final_frac=final_frac)
    cfgs = _cpt_configs(arms, base, tokens, seq_len, batch_tokens, warmup,
                        final_frac)
    if not base.startswith('Qwen/'):
        res = d2l._pretrain_result('cpt-t1', lambda: _cpt_merge(
            {a: _cpt_arm_run(c) for a, c in cfgs.items()}, cfgs), config)
    else:
        missing = set(arms) - set(_CPT_QWEN_NAMES)
        if missing:
            raise ValueError(f'no regeneration name for arms {missing}')
        parts = {_CPT_QWEN_NAMES[a]: (lambda c=c: _cpt_arm_run(c), c)
                 for a, c in cfgs.items()}
        names = {_CPT_QWEN_NAMES[a]: a for a in cfgs}
        res = d2l._pretrain_merged(
            'cpt-qwen-evals', parts, lambda r: _cpt_merge(
                {names[k]: v for k, v in r.items()}, cfgs), config,
            fallback='cpt-t1')
    return d2l._show_label(d2l._as_tuple(CptRuns, res, _CPT_COLUMNS))

def cpt_probe(micro_batches=(1, 2), steps=15, warmup=3):
    """The D2 probe of a T2 arm (arm A's configuration) on one GPU, per
    micro-batch size: peak GiB reserved, median seconds per step after
    compilation, tokens/s and the projected hours of a 305-step arm (no
    evaluations), as a DataFrame."""
    import tempfile
    rows = []
    dev = torch.device('cuda')
    for m in micro_batches:
        cfg = dict(_cpt_configs(**_cpt_qwen_call())['A'], micro_batch=m,
                   evaluate=False, equal_step=None, suite_step=None)
        torch._dynamo.reset()
        model = _cpt_base(cfg, device=dev)[0].train()
        shards = _cpt_qwen_shards()
        torch.cuda.reset_peak_memory_stats(dev)
        os.makedirs(d2l._job_root(), exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=d2l._job_root()) as job, \
                    d2l._quiet():
                log = _cpt_train(model, _cpt_mix(shards, cfg), shards, cfg,
                                 job, peak=cfg['peak_lr'],
                                 stop=steps + warmup)
            t = float(log['step_s'].iloc[warmup:].median())
        except torch.OutOfMemoryError:
            t = math.nan  # the row says so: gib is the card's limit
        B, T = cfg['global_batch'], cfg['seq_len']
        rows.append(dict(micro_batch=m, gib=torch.cuda.max_memory_reserved(
            dev) / 2 ** 30, step_s=t, tok_s=B * T / t,
                         arm_hours=cfg['steps'] * t / 3600))
        del model
        torch._dynamo.reset()
        d2l.lm_free()
    return pd.DataFrame(rows)
```

## Plots and the Frontier Test

```{.python .input #continued-pretraining-lib-plots}
%%tab pytorch
#@save
def plot_cpt_curves(runs, thin=('A′', 'A″')):
    """Domain, general and held-out book loss against tokens, one panel
    each, per arm (the arms in `thin` as thin lines), with the base's loss
    as a dashed line."""
    fig, axes = d2l.plt.subplots(1, 3, figsize=(8, 2.6))
    logs = runs.logs.dropna(subset=['domain_loss'])
    base = logs[logs['arm'] == 'base']
    arms = [a for a in dict.fromkeys(logs['arm']) if a != 'base']
    for ax, s in zip(axes, ('domain', 'general', 'book')):
        for i, arm in enumerate(arms):
            g = logs[logs['arm'] == arm]
            ax.plot(g['tokens'] / 1e6, g[f'{s}_loss'], color=f'C{i}',
                    lw=0.8 if arm in thin else 1.6, label=arm)
        if len(base):
            ax.axhline(base[f'{s}_loss'].iloc[0], color='black', ls='--',
                       lw=1)
        ax.set_xlabel('tokens (millions)')
        ax.set_title(f'{s} loss', fontsize=10)
    axes[0].set_ylabel('nats per token')
    axes[0].legend(fontsize=8)
    fig.tight_layout()

def frontier_excess(fg, arm='B', through='C'):
    """How far arm's domain gain lies above the line from the base (no
    gain, no forgetting) through arm `through`, at arm's forgetting, in
    nats per token: (excess, lo, hi), the 95% interval from the document
    resamples of forgetting_gain (fg.attrs['draws']); NaNs when an arm is
    missing."""
    draws = fg.attrs.get('draws', {})
    if not {arm, through} <= set(draws):
        return (math.nan,) * 3
    g, f = (lambda a: draws[a]['gain']), (lambda a: draws[a]['forget'])
    ex = g(arm) - g(through) * f(arm) / f(through)  # replicates, then point
    lo, hi = np.percentile(ex[:-1], [2.5, 97.5])
    return float(ex[-1]), float(lo), float(hi)

def plot_tradeoff(fg, through='C'):
    """Domain gain against forgetting with 95% intervals per arm, the base
    at the origin, and the line from the base through arm `through`."""
    fig, ax = d2l.plt.subplots(figsize=(4.2, 3.2))
    ax.plot([0], [0], 'ks')
    ax.annotate('base', (0, 0), xytext=(6, -10), textcoords='offset points',
                fontsize=9)
    for i, r in enumerate(fg.itertuples()):
        ax.errorbar(r.forget, r.gain, xerr=[[r.forget - r.forget_lo],
                                            [r.forget_hi - r.forget]],
                    yerr=[[r.gain - r.gain_lo], [r.gain_hi - r.gain]],
                    fmt='o', color=f'C{i}', capsize=2, label=r.arm)
    c = fg.set_index('arm').reindex([through]).iloc[0]
    if np.isfinite(c['forget']) and c['forget'] > 0:
        x = np.array([0, fg['forget_hi'].max()])
        ax.plot(x, x * c['gain'] / c['forget'], color='black', ls='--',
                lw=1)
    ax.axhline(0, color='black', lw=0.6)
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('forgetting (nats/token)')
    ax.set_ylabel('domain gain (nats/token)')
    ax.legend(fontsize=8)
    fig.tight_layout()
```

## The SmolLM3 Trajectory

Six base-model checkpoints of `HuggingFaceTB/SmolLM3-3B-checkpoints` (the
ends of the three pretraining stages, the two long-context stages and the
mid-training checkpoint), each scored with the suite (MMLU on a seeded
2,000-item subset, the other tasks in full) and with bits per byte on 500
documents of each evaluation set's texts (FineWeb-Edu validation, peS2o
validation) and on every held-out book section. Token counts: steps times
the global batch of the stage configurations (`smollm3-configs`: 192
replicas of 3 sequences of 4,096 tokens, 2,359,296 tokens per step; the
long-context stages keep 2,359,296 tokens per step over 20,000 steps
each); mid-training adds the blog's 140B tokens. One regeneration per
checkpoint (`smollm3-0` to `-5`), merged when all six exist. With `n`
given the suite is scored live on `n` items per task (seed 0) and nothing
is loaded.

```{.python .input #continued-pretraining-lib-smollm3}
%%tab pytorch
#@save
SmolTrajectory = collections.namedtuple(
    'SmolTrajectory', ['table', 'bpb', 'items', 'docs', 'label'])

#@save
SMOLLM3_TASKS = ('hellaswag', 'arc_easy', 'arc_challenge', 'piqa',
                 'winogrande', 'mmlu')

#@save
_SMOLLM3 = {  # name: (row label, tokens seen in billions)
    'stage1': ('stage 1 end', 3_440_000 * 2_359_296 / 1e9),
    'stage2': ('stage 2 end', 4_200_000 * 2_359_296 / 1e9),
    'stage3': ('stage 3 end', 4_720_000 * 2_359_296 / 1e9),
    'lc32k': ('LC 32K', 4_740_000 * 2_359_296 / 1e9),
    'lc64k': ('LC 64K', 4_760_000 * 2_359_296 / 1e9),
    'mid': ('mid-training', 4_760_000 * 2_359_296 / 1e9 + 140)}

#@save
_SMOLLM3_DOMAINS = ('web', 'science', 'book')

#@save
def _smollm3_call():
    return dict(names=('stage1', 'stage2', 'stage3', 'lc32k', 'lc64k',
                       'mid'))

def _bpb_texts(n=500):
    """{domain: texts}: the first n FineWeb-Edu validation documents, the
    first n of peS2o's validation file 0, and every held-out book
    section."""
    import gzip
    from huggingface_hub import hf_hub_download
    root = d2l.reference_shards('val').root
    with gzip.open(os.path.join(root, 'val.jsonl.gz'), 'rt') as f:
        web = [json.loads(line)['text'] for _, line in zip(range(n), f)]
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES['pes2o']
    with d2l._quiet():
        path = hf_hub_download(repo, 'data/v2/validation-00000-of-00002.'
                               'json.gz', repo_type='dataset', revision=rev)
    with gzip.open(path, 'rt') as f:
        science = [json.loads(line)['text'] for _, line in zip(range(n), f)]
    return dict(web=web, science=science,
                book=[t for _, t in book_split()[1]])

def _smollm3_point(name, n=None, texts=None):
    """One checkpoint: per-item suite scores, (with texts) per-document
    bits per byte and UTF-8 bytes, and its context length."""
    import unicodedata
    from transformers.utils import logging as hf_logging
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES[f'smollm3-{name}']
    bars = hf_logging.is_progress_bar_enabled()
    hf_logging.disable_progress_bar()  # a notebook shows them as widgets
    try:
        with d2l._quiet():
            model, tok = d2l.load_hf(repo, revision=rev)
    finally:
        if bars:
            hf_logging.enable_progress_bar()
    suite = _cpt_suite(SMOLLM3_TASKS, n_max=n, n_task=None if n else dict(
        mmlu=2000), name='smollm3', batch_size=8)
    with d2l._quiet():
        res = suite.run(model, tok)
    items = res.per_item[['task', 'item_id', 'score']].assign(
        checkpoint=_SMOLLM3[name][0])
    rows = []
    for dom, docs in (texts or {}).items():
        for i, t in enumerate(docs):
            with d2l._quiet():
                b = d2l.bits_per_byte(model, tok, [t], ctx_len=2048)
            rows.append(dict(checkpoint=_SMOLLM3[name][0], domain=dom, doc=i,
                             bpb=b, bytes=len(unicodedata.normalize(
                                 'NFC', t).encode())))
    ctx = int(model.config.max_position_embeddings)
    del model
    d2l.lm_free()
    return dict(items=items, docs=pd.DataFrame(rows), ctx=ctx)

def _smollm3_merge(points, names):
    """SmolTrajectory from {name: point}."""
    pts = {n: _part(points[n]) for n in names}
    items = pd.concat([pts[n]['items'] for n in names], ignore_index=True)
    acc = items.groupby(['checkpoint', 'task'], sort=False)['score'].mean(
        ).unstack('task')
    table = pd.DataFrame(dict(
        checkpoint=[_SMOLLM3[n][0] for n in names],
        tokens=[f'{_SMOLLM3[n][1] / 1e3:.1f}T' for n in names],
        ctx=[f'{pts[n]["ctx"] // 1024}K' for n in names]))
    table = table.join(acc[list(SMOLLM3_TASKS)], on='checkpoint')
    docs = pd.concat([pts[n]['docs'] for n in names], ignore_index=True)
    if len(docs):
        g = docs.assign(bits=docs['bpb'] * docs['bytes']).groupby(
            ['checkpoint', 'domain'], sort=False)[['bits', 'bytes']].sum()
        bpb = (g['bits'] / g['bytes']).unstack('domain')[list(
            _SMOLLM3_DOMAINS)].reindex(table['checkpoint']).reset_index()
    else:
        bpb = pd.DataFrame(columns=['checkpoint', *_SMOLLM3_DOMAINS])
    return SmolTrajectory(table, bpb, items, docs, '')

def smollm3_trajectory(names, n=None):
    """SmolLM3-3B's checkpoints `names` (of 'stage1', 'stage2', 'stage3',
    'lc32k', 'lc64k', 'mid'): SmolTrajectory(table, bpb, items, docs,
    label). table: checkpoint, tokens seen, context length and accuracy per
    task; bpb: bits per byte per domain (web, science, book). n=None: the
    committed full-set result (regenerations smollm3-0 to smollm3-5, one
    checkpoint each); n given: the suite scored live on n items per task
    in every run (no bits per byte)."""
    names = list(names)
    if n is not None:
        return _smollm3_merge({k: _smollm3_point(k, n=n) for k in names},
                              names)
    order = list(_smollm3_call()['names'])
    texts = {}

    def point(k):  # the texts are read once per process
        if 't' not in texts:
            texts['t'] = _bpb_texts()
        return _smollm3_point(k, texts=texts['t'])
    parts = {f'smollm3-{order.index(k)}': (
        lambda k=k: point(k), dict(name=k, pin=d2l.PRETRAIN_SOURCES[
            f'smollm3-{k}'][2])) for k in names}
    res = d2l._pretrain_merged(None, parts, lambda r: _smollm3_merge(
        {k: r[f'smollm3-{order.index(k)}'] for k in names}, names),
        dict(names=names))
    return d2l._show_label(d2l._as_tuple(SmolTrajectory, res, dict(
        table=['checkpoint', 'tokens', 'ctx', *SMOLLM3_TASKS],
        bpb=['checkpoint', *_SMOLLM3_DOMAINS],
        items=['task', 'item_id', 'score', 'checkpoint'],
        docs=['checkpoint', 'domain', 'doc', 'bpb', 'bytes'])))

def plot_capability_bpb(ref, traj, tasks=('hellaswag', 'arc_easy', 'piqa',
                                           'winogrande')):
    """Mean accuracy on the tasks both suites share against bits per byte
    on FineWeb-Edu validation text: the reference run's checkpoints (ref,
    a CheckpointEvals) and SmolLM3's (traj, web BPB)."""
    smol = traj.table.set_index('checkpoint')
    xs = [ref.table['bpb'], traj.bpb.set_index('checkpoint')['web'].reindex(
        smol.index)]
    ys = [ref.table[list(tasks)].mean(axis=1), smol[list(tasks)].mean(axis=1)]
    d2l.plot(xs, ys, 'bits per byte (FineWeb-Edu validation)',
             'mean accuracy (4 tasks)', legend=['reference model', 'SmolLM3-3B'],
             fmts=('o-', 's-'), figsize=(4.5, 3.2))

def plot_smollm3(traj):
    """Suite accuracy per task (left) and bits per byte per domain (right)
    across the checkpoints."""
    fig, (ax1, ax2) = d2l.plt.subplots(1, 2, figsize=(8, 3))
    x = np.arange(len(traj.table))
    for i, t in enumerate(SMOLLM3_TASKS):
        ax1.plot(x, traj.table[t], '-o', ms=3, color=f'C{i}', label=t)
    for i, dom in enumerate(_SMOLLM3_DOMAINS):
        ax2.plot(x, traj.bpb[dom], '-o', ms=3, color=f'C{i}', label=dom)
    for ax, what in ((ax1, 'accuracy'), (ax2, 'bits per byte')):
        ax.set_xticks(x)
        ax.set_xticklabels(traj.table['checkpoint'], rotation=30,
                           ha='right', fontsize=8)
        ax.set_ylabel(what)
        ax.legend(fontsize=7)
    fig.tight_layout()
```
