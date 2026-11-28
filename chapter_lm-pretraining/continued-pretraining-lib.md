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
- GSM8K in two formats (`score_gsm8k`: 8-shot completion and a ChatML
  turn with thinking) on a Transformers copy of an arm's weights;
- the arm driver `cpt_arms` (one resumable job folder per arm, the
  re-warmed schedule, fresh or restored AdamW moments, the evaluation
  grid, the equal-domain-token evaluation, the merge of the arms with the
  base's numbers), the D2 probe and the arms' plots;
- the frontier test behind the replay claim (`frontier_excess`: the
  replay arm against the chord between two no-replay arms) and the
  early general-loss gaps behind the cold-restart claims (`early_gaps`);
- the SmolLM3 trajectory (`smollm3_trajectory`: the cloze suite, bits per
  byte with no prefix token on four domains, calibration and GSM8K in two
  formats per checkpoint; the conditional GSM-Symbolic rung), the page's
  live GSM8K check (`smollm3_gsm8k`), the changes between stages
  (`stage_deltas`), the calibration table and the two plots.

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble inside its functions (`transformers`, `huggingface_hub`,
`fcntl`, `subprocess`, `unicodedata`).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_continued_pretraining`

## Registries

The hosted inputs (d2l-data R2): the summary of the Qwen-tokenized T2
data (the shards themselves are rebuilt by
`tools/build_pretraining_data.py --stage cpt-qwen-traces`, never hosted)
and the R2-tokenized T1 slices. The reasoning-trace corpus
(`lm-cpt-qwen-manifest-v3`, `lm-traces-12m-v1`, `lm-traces-val-v1`,
uploaded 2026-11-27; manifest-v3 replaced v2, the same shards with the
per-ending counts and the build's targets added) replaced the peS2o one (`lm-cpt-qwen-manifest-v1`,
`lm-pes2o-12m-v1`, `lm-pes2o-val-v1`, uploaded 2026-11-25), which no page
reads any more (the builder's check compares the rebuilt replay and general
validation shards with manifest-v1's layout hashes); the keys stay until
the section's close-out, which removes them for the pruning tool. `lm-fwedu-replay-12m-v1` serves both, and
21.5's T1 replay.

```{.python .input #continued-pretraining-lib-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-cpt-qwen-manifest-v1'] = (
    d2l.LM_DATA_URL + 'lm-cpt-qwen-manifest-v1.json.gz',
    '4669a1ba7b7fa4f8dbc6c15e8d91372b733330cc')
d2l.DATA_HUB['lm-pes2o-12m-v1'] = (
    d2l.LM_DATA_URL + 'lm-pes2o-12m-v1.tar',
    '8c9b514214da373453f155f2a0adfcc364e02ee0')
d2l.DATA_HUB['lm-pes2o-val-v1'] = (
    d2l.LM_DATA_URL + 'lm-pes2o-val-v1.tar',
    '0ef7a04d6f2e9b8b23567d4eb37ca5039caf5cf0')
d2l.DATA_HUB['lm-fwedu-replay-12m-v1'] = (
    d2l.LM_DATA_URL + 'lm-fwedu-replay-12m-v1.tar',
    '46232cacbe8b9eb30ca7ce6a2d3f7e14ac765c24')
d2l.DATA_HUB['lm-smollm3-trajectory-v1'] = (
    d2l.LM_DATA_URL + 'lm-smollm3-trajectory-v1.json.gz',
    '03aa35aca040cab3b127e3870b45c11b147b94b7')
d2l.DATA_HUB['lm-cpt-runs-v1'] = (
    d2l.LM_DATA_URL + 'lm-cpt-runs-v1.json.gz',
    '59dab5e3e95193643732a7d923f49a4efa1e6a58')
d2l.DATA_HUB['lm-cpt-t1-v1'] = (
    d2l.LM_DATA_URL + 'lm-cpt-t1-v1.json.gz',
    'ecc9890d0bac23b3548c8f2eaf5616f52b5d4e64')
d2l.DATA_HUB['lm-traces-12m-v1'] = (
    d2l.LM_DATA_URL + 'lm-traces-12m-v1.tar',
    '269d73e0166d62c5fcc428c0a1e080be4a87fcd9')
d2l.DATA_HUB['lm-traces-val-v1'] = (
    d2l.LM_DATA_URL + 'lm-traces-val-v1.tar',
    'c14411336f10e91ca619ab71c37e5bb2d7d07540')
d2l.DATA_HUB['lm-cpt-qwen-manifest-v3'] = (
    d2l.LM_DATA_URL + 'lm-cpt-qwen-manifest-v3.json.gz',
    '2c0d67094de0309473f81bfbae303105d210df4f')
```

The section's results. A regeneration run shows a result it has not
computed yet as pending (with the T1 arms as the stand-in for the Qwen
arms) instead of failing. The arms' results on the trace corpus are
`lm-cpt-runs-v2` and `lm-cpt-t1-v2` (their stored configurations name the
data they trained on, so a result from another corpus is refused); the
peS2o runs `lm-cpt-runs-v1` and `lm-cpt-t1-v1` stay registered until the
section's close-out, but no page reads them. The SmolLM3 trajectory is
`lm-smollm3-trajectory-v2` (bits per byte with no prefix token, the
trace domain, calibration, GSM8K); `lm-smollm3-trajectory-v1` stays
registered because version 2 copies its per-item cloze scores.

```{.python .input #continued-pretraining-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['cpt-qwen-manifest'] = (
    'lm-cpt-qwen-manifest-v3', 'json-table')
d2l.PRETRAIN_ARTIFACTS['cpt-traces-12m'] = ('lm-traces-12m-v1', 'shards')
d2l.PRETRAIN_ARTIFACTS['cpt-traces-val'] = ('lm-traces-val-v1', 'shards')
d2l.PRETRAIN_ARTIFACTS['cpt-fwedu-replay-12m'] = (
    'lm-fwedu-replay-12m-v1', 'shards')
d2l.PRETRAIN_ARTIFACTS['cpt-t1'] = (
    'lm-cpt-t1-v2', 'json-table')
d2l.PRETRAIN_ARTIFACTS['cpt-runs'] = (
    'lm-cpt-runs-v2', 'json-table')
d2l.PRETRAIN_ARTIFACTS['smollm3-trajectory'] = (
    'lm-smollm3-trajectory-v2', 'json-table')
```

The long jobs that `tools/lm_regen.py` runs (one card each): the four Qwen
arms, their evaluation and merge (`cpt-qwen-evals`: the full evaluation
sets, the suite and GSM8K in two formats for the base and every arm), the
six SmolLM3 checkpoints, and the T1 arms (which also run as a notebook
regeneration, about 75 minutes).

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
At T1, `cold-plateau` starts from the plateau state's weights with fresh
moments at the branch's rate and schedule, so it differs from `branch`
only in the moments (`moments='fresh'`).

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
            'cold-plateau': dict(peak_lr=None, replay=0.0, data_seed=0,
                                 warmup=0, start='plateau',
                                 schedule='linear', moments='fresh'),
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

The domain stream is reasoning traces: OpenThoughts3-1.2M at a pinned
revision, a pinned subset of its files (`PRETRAIN_SOURCE_FILES`), one
document per conversation (ChatML under Qwen3's tokenizer at T2, plain
`User: …\nAssistant: …` text under the reference tokenizer at T1), each
document's id the sha1 of its ChatML text. Replay is FineWeb-Edu file 005;
the evaluation sets are held-out traces (500 conversations whose questions
no training document asks; `val_domain` is their first 2.2M tokens of whole
conversations), FineWeb-Edu's validation shard and the held-out book
sections (the book is never trained). T1 reads the hosted R2-tokenized
shards (or, before their upload, the builder's local output); T2 reads the
local Qwen-tokenized shards (`uint32`, never hosted), built by
`tools/build_pretraining_data.py --stage cpt-qwen-traces` the first time a
regeneration needs them (it downloads 2.8 GB of parquet and takes about
4 minutes of CPU). Most traces stop at the dataset's generation cap: the
manifest counts each split's documents and tokens by ending (complete;
closed but long enough that the answer may be cut; never closing the
`<think>` block). The book is evaluation only: its held-out sections
(`shards['val_book']`) are built from the book corpus on first use under
either tokenizer (seconds; deterministic), and its train sections are not
in the shards at all.

The data enter each arm's job identity and each result's stored
configuration (`_cpt_data_id`): at T2 the committed manifest's layout
hashes of the four token streams (the local shards are checked against
them), at T1 the sha1 of the hosted tars. A folder or result from another
corpus is therefore refused, never reused. The book's sections are left
out of the identity: the book's text changes between builds, and it is
never trained.

```{.python .input #continued-pretraining-lib-data}
%%tab pytorch
#@save
CptData = collections.namedtuple('CptData', ['table', 'shards', 'label'])

#@save
_CPT_STREAMS = (  # key, row label, source, licence
    ('traces', 'domain: reasoning traces', 'OpenThoughts3-1.2M',
     'Apache-2.0'),
    ('replay', 'replay: FineWeb-Edu', 'FineWeb-Edu file 005', 'ODC-By'),
    ('val_domain', 'val: held-out traces', 'question-disjoint',
     'Apache-2.0'),
    ('val_general', 'val: FineWeb-Edu', 'FineWeb-Edu validation',
     'ODC-By'),
    ('val_book', 'held-out book', 'never trained', 'CC BY-SA 4.0'))

#@save
_CPT_DIRS = {'t1': os.path.join('..', 'data', 'lm', 'cpt-t1-traces-v1'),
             't2': os.path.join('..', 'data', 'lm', 'cpt-qwen-traces-v1')}

#@save
_CPT_T1_KEYS = {'traces': 'cpt-traces-12m', 'val_domain': 'cpt-traces-val',
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
    """{'val_book': the held-out sections} as TokenShards under the tier's
    tokenizer, built on first use (the train sections are never trained,
    so they are not built)."""
    tag = 'r2' if tier == 't1' else 'qwen'
    root = os.path.join('..', 'data', 'lm', f'cpt-book-{tag}-v1')
    out = {}
    with _locked(root + '.lock'):
        for key, docs in (('val_book', book_split()[1]),):
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
                           'tools/build_pretraining_data.py --stage '
                           'cpt-t1-traces')
    out['val_general'] = d2l.reference_shards('val')
    return dict(out, **_book_shards('t1'))

def _run_builder(stage):
    """Run tools/build_pretraining_data.py --stage stage (quietly)."""
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(d2l.__file__)))
    subprocess.run([sys.executable, os.path.join(
        repo, 'tools', 'build_pretraining_data.py'), '--stage', stage],
        check=True, cwd=repo, stdout=subprocess.DEVNULL)

def _cpt_manifest_committed():
    return d2l.PRETRAIN_ARTIFACTS['cpt-qwen-manifest'][0] in d2l.DATA_HUB

def _cpt_qwen_shards():
    """The T2 shards (Qwen tokens, local), built through the builder when
    missing; checked against the committed manifest when there is one."""
    keys = [k for k, *_ in _CPT_STREAMS if k != 'val_book']
    root = _CPT_DIRS['t2']
    have = lambda: all(os.path.exists(os.path.join(root, k,
                                                   'manifest.json'))
                       for k in keys)
    with _locked(root + '.lock'):
        if not have():
            _run_builder('cpt-qwen-traces')
    out = {k: d2l.TokenShards(os.path.join(root, k), 'train') for k in keys}
    if _cpt_manifest_committed():
        layout = d2l.load_artifact('cpt-qwen-manifest').layout
        for k, s in out.items():
            if d2l._layout_hash(s) != layout[k]:
                raise ValueError(f'{k}: the local shards differ from the '
                                 'committed manifest; rebuild them')
    return dict(out, **_book_shards('t2'))

def chatml(messages):
    """A conversation (dicts with role and content) as ChatML text: per
    message <|im_start|>role, a newline, the content verbatim, <|im_end|>
    and a newline. These are the ChatML markers of Qwen3 and SmolLM3, not
    their full chat templates (Qwen3's adds a newline after <think> and an
    empty think block to a reply without one)."""
    return ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n'
                   for m in messages)

def trace_conversations():
    """The 500 held-out reasoning-trace conversations of the section's
    corpus (OpenThoughts3-1.2M; no training document asks their
    questions), in id order: a DataFrame with doc_id, question_id, domain,
    source and messages (role, content). Most replies stop at the
    dataset's generation cap, many inside the <think> block. Local only,
    written by tools/build_pretraining_data.py --stage cpt-qwen-traces,
    which runs the first time they are needed; when the manifest is
    committed, the ids must be the ones it lists (ids_sha1)."""
    import gzip
    root = _CPT_DIRS['t2']
    path = os.path.join(root, 'heldout.jsonl.gz')
    with _locked(root + '.lock'):
        if not os.path.exists(path):
            _run_builder('cpt-qwen-traces')
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        conv = pd.DataFrame([json.loads(line) for line in f])
    if _cpt_manifest_committed():
        want = d2l.load_artifact('cpt-qwen-manifest').ids_sha1['heldout']
        got = hashlib.sha1(' '.join(conv['doc_id']).encode()).hexdigest()
        if got != want:
            raise ValueError(f'{path} holds other conversations than the '
                             'committed manifest lists; rebuild it')
    return conv

def trace_windows(tokenizer, T=2048, n=None, render=chatml):
    """One window per held-out conversation (the first n; all 500 by
    default): the first min(T, length) token ids of render(messages) under
    tokenizer (a Transformers tokenizer), as int64 arrays (a few
    conversations are shorter than 2,048 tokens). With the default ChatML
    rendering the chat markers must be the tokenizer's own special tokens,
    so a window reads as the model's chat format does."""
    conv = trace_conversations()
    conv = conv if n is None else conv.iloc[:n]
    if render is chatml:
        marks = tokenizer.convert_tokens_to_ids(['<|im_start|>',
                                                 '<|im_end|>'])
        if None in marks or tokenizer.unk_token_id in marks:
            raise ValueError('the tokenizer has no ChatML markers')
    ids = tokenizer([render(m) for m in conv['messages']],
                    add_special_tokens=False, verbose=False)['input_ids']
    return [np.asarray(x[:T], dtype=np.int64) for x in ids]

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

def _shard_table(shards, tier='t2'):
    """stream, documents, tokens, source, licence of each stream (the
    traces' source names their rendering: ChatML at T2, plain text at
    T1)."""
    form = {'t1': 'plain text', 't2': 'ChatML'}[tier]
    rows = [(label, shards[k].n_docs, shards[k].n_tokens,
             f'{src}, {form}' if k == 'traces' else src, lic)
            for k, label, src, lic in _CPT_STREAMS]
    return pd.DataFrame(rows, columns=['stream', 'documents', 'tokens',
                                       'source', 'licence'])

def _cpt_qwen_regenerating():
    names = set(_CPT_QWEN_NAMES.values()) | {'cpt-qwen-evals'}
    return bool(d2l._regenerating() & names)

def cpt_data(tier='t2'):
    """The arms' data: CptData(table, shards, label). table: per stream
    (domain reasoning traces, replay FineWeb-Edu, the three evaluation
    sets: held-out traces, FineWeb-Edu validation, held-out book)
    documents, tokens, source and licence. 't1': the R2-tokenized shards.
    't2': in a normal run the committed manifest's table (no shard is
    opened, built or downloaded; shards is None); in a regeneration of the
    Qwen arms the local Qwen-tokenized shards. shards: 'traces', 'replay',
    'val_domain', 'val_general', 'val_book' (the book is evaluation
    only)."""
    if tier == 't1':
        shards = _cpt_t1_shards()
        return CptData(_shard_table(shards, 't1'), shards, '')
    if tier != 't2':
        raise ValueError(f"tier is 't1' or 't2', not {tier!r}")
    if _cpt_qwen_regenerating():
        shards = _cpt_qwen_shards()
        return CptData(_shard_table(shards), shards, '')
    if _cpt_manifest_committed():
        return CptData(d2l.load_artifact('cpt-qwen-manifest').table, None,
                       '')
    if d2l._regenerating():
        return d2l._show_label(CptData(pd.DataFrame(columns=[
            'stream', 'documents', 'tokens', 'source', 'licence']), None,
            'cpt-qwen-manifest pending'))
    raise KeyError('cpt-qwen-manifest is not committed yet')

def _cpt_data_id(tier):
    """The identity of a tier's training and evaluation streams (traces,
    replay, val_domain, val_general; the book is left out): T2 the
    committed manifest's layout hashes (the local shards are checked
    against them when they are opened), T1 the sha1 of each hosted tar
    (the local shards' layout hash before its upload) and the reference
    model's validation shards' layout hash."""
    keys = ('traces', 'replay', 'val_domain', 'val_general')
    if tier == 't2':
        if not _cpt_manifest_committed():
            raise KeyError('cpt-qwen-manifest is not committed yet')
        layout = d2l.load_artifact('cpt-qwen-manifest').layout
        return {k: layout[k] for k in keys}
    if tier != 't1':
        raise ValueError(f"tier is 't1' or 't2', not {tier!r}")
    out = {}
    for k, name in _CPT_T1_KEYS.items():
        key = d2l.PRETRAIN_ARTIFACTS[name][0]
        out[k] = (d2l.DATA_HUB[key][1] if key in d2l.DATA_HUB else
                  d2l._layout_hash(d2l.TokenShards(
                      os.path.join(_CPT_DIRS['t1'], k), 'train')))
    out['val_general'] = d2l._layout_hash(d2l.reference_shards('val'))
    return {k: out[k] for k in keys}
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

## GSM8K in Two Formats

What the arms gained on a test set: 500 GSM8K test items (the seeded
subset chapter 18's tasks draw, seed 0), greedy, in two formats. The
completion format is chapter 18's `gsm8k_cot` task (eight written chains
of thought, 256 new tokens, cut at the next `Q:`), the format of a base
model. The chat format is one ChatML user turn and the assistant's
generation prompt, with no system turn and no prefilled `<think>`: the
format of the reasoning traces the arms train on. A chat reply gets 4,096
new tokens. Its answer is the content of the last `\boxed{}` after its
last `</think>`, else the number `extract_answer` finds there (the whole
reply when it opened no think block). A reply cut at the budget has no
answer, and neither has a reply that opened a think block and ended it
with `<|im_end|>` before closing it: that is the pre-registered score.
A second readout forces an answer from those replies (budget forcing,
Muennighoff et al. 2025, s1): the reply so far, a `</think>` if a think
block is open, and the cue `The final answer is $\boxed{`, continued
greedily for at most 32 tokens up to the closing brace. Replies that
answered on their own keep their score; the forced score puts a number
on every item at a cost of 32 tokens per cut reply, and the cut share
stays its own column (F21 review finding 1: traces averaging 15K tokens
teach thinking past 4,096, so the pre-registered score alone may measure
thinking length, not skill). Per item the table keeps the score, the
forced score, whether the reply was cut at the budget, whether it opened
and closed a think block, whether a completion reply wrote a ChatML or
think marker (the text before the first marker is scored, since the reply
has ended there), and the number of new tokens. The model's own file
also keeps each reply's text, the extracted answer, the forced
continuation and its answer, so the replies can be read and re-scored
without generating them again; the merged artifact holds only the
columns above. The settings of a cached file carry the scorer's version
(`_GSM8K_SCORER`, raised whenever the answer rules change), so a file
scored under older rules is refused, not reused.

Generation runs on a Transformers Qwen3 model, which batches prompts of
different lengths (left padding), with an arm's weights mapped back from
the book's layout (`_qwen3_hf_state` inverts `map_qwen3_state_dict`).
Qwen3-0.6B's key-value cache costs 28 layers x 8 heads x 128 x 2 x 2 B =
112 KiB per position, about 11 GiB at 24 chat replies of 4,096 + 300
positions; measured, batch 24 peaks at 13.1 GiB allocated, and batch 32
runs out of a 24 GiB card (the growing cache fragments the allocator, and
SDPA's grouped-query expansion copies each layer's keys and values; F21
review smoke). The forced answers prefill 4,400-token prompts and run 8 at
a time. Greedy decoding with left padding in bf16 is not batch-invariant,
so the batch sizes are part of each cached file's settings. One model's
two formats take up to about 1.5 hours on a 4090 when every chat reply
runs to the budget (the F22 pilot: 17 minutes per 100 chat items), so the
five models can be scored side by side on separate cards: each process
fills its model's cache file (`_cpt_gsm8k_all(cfgs, arms=())` for the
base, `arms=[arm]` for one arm once the base's file exists), and
`cpt-qwen-evals` reads them back.

```{.python .input #continued-pretraining-lib-gsm8k}
%%tab pytorch
#@save
_CPT_GSM8K = dict(n=500, seed=0, budget=dict(completion=256, chat=4096),
                  force=32)  # force: tokens after the forced-answer cue

#@save
_GSM8K_BATCH = dict(completion=32, chat=24, force=8)  # per batch

#@save
_GSM8K_CUE = 'The final answer is $\\boxed{'  # budget forcing (s1)

#@save
_CHAT_MARKS = ('<|im_start|>', '<|im_end|>', '<think>', '</think>')

#@save
_GSM8K_SCORER = 2  # raise when gsm8k_chat_answer, the box or marks change

#@save
_GSM8K_TEXT = ['text', 'pred', 'forced_pred', 'forced_text']  # local only

#@save
def gsm8k_items(n=500, seed=0):
    """n GSM8K test items (the seeded subset of chapter 18's tasks): dicts
    with id, question, reference and the 8-shot completion prompt of the
    gsm8k_cot task."""
    with d2l._quiet():
        suite = d2l.EvalSuite(['gsm8k_cot'], n=n, seed=seed)
    return suite.items('gsm8k_cot')

def gsm8k_chat_prompt(question):
    """One ChatML user turn and the assistant's generation prompt."""
    return (chatml([dict(role='user', content=question)])
            + '<|im_start|>assistant\n')

def _think_state(text):
    """(opened, closed): whether a reply opened a <think> block, and
    whether its last one is closed."""
    if '<think>' not in text:
        return False, '</think>' in text
    return True, '</think>' in text.split('<think>')[-1]

def gsm8k_chat_answer(text, finish_reason):
    """The answer of a chat reply, or None: a reply cut at the budget, or
    one that opened a think block and never closed it, has none; else,
    after the last </think> (in the whole reply when it did not think),
    the number in the last \\boxed{} if one holds a number, otherwise the
    number d2l.extract_answer finds there."""
    opened, closed = _think_state(text)
    if finish_reason == 'length' or (opened and not closed):
        return None
    tail = text.split('</think>')[-1]
    boxed = d2l.extract_answer(tail, 'boxed')
    return (boxed and d2l.extract_answer(boxed, 'gsm8k')) or \
        d2l.extract_answer(tail, 'gsm8k')

def gsm8k_force_prompt(prompt, text):
    """Budget forcing: the prompt, the reply so far, a closing </think>
    if a think block is open, and the answer cue."""
    opened, closed = _think_state(text)
    close = '\n</think>' if opened and not closed else ''
    return prompt + text + close + '\n\n' + _GSM8K_CUE

def score_gsm8k(model, tokenizer, items, formats=('completion', 'chat'),
                budget=None, batch_size=None, chat_prompt=None, force=None,
                end_token='<|endoftext|>'):
    """GSM8K replies of model (greedy) per item and format ('completion':
    the 8-shot prompt; 'chat': chat_prompt(question), by default
    gsm8k_chat_prompt): a DataFrame with format, item_id, score (exact
    match), forced (the score with budget forcing: a chat reply without an
    answer of its own is continued from gsm8k_force_prompt for at most
    `force` tokens, default _CPT_GSM8K['force'], 0 = off; else the score),
    cut (stopped at the budget), thinks and closed (opened and closed a
    think block), chat_tokens (wrote a ChatML or think marker),
    new_tokens, and the columns of _GSM8K_TEXT: the reply's text, its
    answer (pred), the forced answer and the forced continuation's text
    (None when not forced). budget is {format: n}, batch_size {format or
    'force': n}; end_token is the tokenizer's document end (Qwen3's
    <|endoftext|>, SmolLM3's <|end_of_text|>), which ends a completion,
    and with <|im_end|> a chat reply."""
    budget = {**_CPT_GSM8K['budget'], **(budget or {})}
    batch = {**_GSM8K_BATCH, **(batch_size or {})}
    force = _CPT_GSM8K['force'] if force is None else force
    chat_prompt = chat_prompt or gsm8k_chat_prompt
    num = lambda t: d2l.extract_answer(t, 'gsm8k')
    ids = {m: tokenizer.convert_tokens_to_ids(m) for m in
           (end_token, '<|im_end|>')}
    rows = []
    for fmt in formats:
        if fmt == 'completion':  # a ChatML end is text here, not a stop
            prompts, stop, eos = ([it['prompt'] for it in items],
                                  ['Q:'], [ids[end_token]])
        elif fmt == 'chat':
            prompts, stop, eos = ([chat_prompt(it['question'])
                                   for it in items], None, list(ids.values()))
        else:
            raise ValueError(f"format is 'completion' or 'chat', not {fmt!r}")
        gens = d2l.generate(model, tokenizer, prompts, budget[fmt],
                            eos_token_ids=eos, stop=stop,
                            batch_size=batch[fmt])
        if fmt == 'chat':
            preds = [gsm8k_chat_answer(g.text, g.finish_reason)
                     for g in gens]
            todo = [i for i, g in enumerate(gens) if force and (
                g.finish_reason == 'length'
                or _think_state(g.text) == (True, False))]  # no answer
            forced, more_text = list(preds), [None] * len(gens)
            if todo:  # up to the closing brace of the box
                more = d2l.generate(
                    model, tokenizer, [gsm8k_force_prompt(prompts[i],
                                                          gens[i].text)
                                       for i in todo], force,
                    eos_token_ids=eos, stop=['}'],
                    batch_size=batch['force'])
                for i, m in zip(todo, more):
                    forced[i] = num(m.text)
                    more_text[i] = m.text
        else:  # the reply has ended at its first chat marker
            preds = [num(g.text[:min(
                [g.text.find(m) for m in _CHAT_MARKS if m in g.text],
                default=None)]) for g in gens]
            forced, more_text = preds, [None] * len(gens)
        right = lambda p, it: float(p is not None and p == it['reference'])
        for it, g, pred, fp, mt in zip(items, gens, preds, forced, more_text):
            opened, closed = _think_state(g.text)
            rows.append(dict(format=fmt, item_id=it['id'],
                             score=right(pred, it), forced=right(fp, it),
                             cut=g.finish_reason == 'length', thinks=opened,
                             closed=closed, chat_tokens=any(
                                 m in g.text for m in _CHAT_MARKS),
                             new_tokens=len(g.token_ids), text=g.text,
                             pred=pred, forced_pred=fp, forced_text=mt))
    return pd.DataFrame(rows, columns=[c for c in _CPT_COLUMNS['gsm8k']
                                       if c != 'arm'] + _GSM8K_TEXT)

def _qwen3_hf_state(state, cfg):
    """A Transformers Qwen3 state dict from a Qwen3GPT or PackedGPT state
    dict (map_qwen3_state_dict inverted: names back, and the RoPE rows of
    q_proj and k_proj and the gains of q_norm and k_norm permuted back to
    Transformers' split halves); no lm_head when the embeddings are
    tied. A KeyError lists the tensors not consumed or not produced."""
    kw = d2l.qwen3_gpt_kwargs(cfg)
    d_h = kw['head_dim'] or kw['num_hiddens'] // kw['num_heads']
    inv = torch.argsort(d2l.qwen3_rope_permutation(d_h))
    back = lambda W: W.unflatten(0, (-1, d_h))[:, inv.to(W.device)].flatten(
        0, 1)
    keep = lambda W: W
    layer = {'norm1': ('input_layernorm', keep),
             'attention.W_q': ('self_attn.q_proj', back),
             'attention.W_k': ('self_attn.k_proj', back),
             'attention.W_v': ('self_attn.v_proj', keep),
             'attention.W_o': ('self_attn.o_proj', keep),
             'attention.q_norm': ('self_attn.q_norm', back),
             'attention.k_norm': ('self_attn.k_norm', back),
             'norm2': ('post_attention_layernorm', keep),
             'ffn.W_g': ('mlp.gate_proj', keep),
             'ffn.W_1': ('mlp.up_proj', keep),
             'ffn.W_2': ('mlp.down_proj', keep)}
    names = {'token_emb': ('model.embed_tokens', keep),
             'norm': ('model.norm', keep)}
    for i in range(kw['num_blks']):
        names.update({f'blks.{i}.{ours}': (f'model.layers.{i}.{hf}', f)
                      for ours, (hf, f) in layer.items()})
    if not kw['tie_embeddings']:
        names['lm_head'] = ('lm_head', keep)
    left, out = dict(state), {}
    missing = [n for n in names if n + '.weight' not in left]
    for ours, (hf, f) in names.items():
        if ours + '.weight' in left:
            out[hf + '.weight'] = f(left.pop(ours + '.weight'))
    if left or missing:
        raise KeyError(f'not consumed: {sorted(left)}; missing: {missing}')
    return out

def _load_into_hf(hf, path):
    """Load the weights of a PackedGPT bf16 file (an arm's final
    checkpoint) into the Transformers Qwen3 model hf of the same shape."""
    from safetensors.torch import load_file
    state = load_file(path, device=str(hf.device))
    hf_state = _qwen3_hf_state(state, hf.config)
    missing, unexpected = hf.load_state_dict(hf_state, strict=False)
    tied = {'lm_head.weight'} if hf.config.tie_word_embeddings else set()
    if unexpected or set(missing) - tied:
        raise KeyError(f'{path}: missing {missing}, unexpected {unexpected}')

def _cpt_hf_qwen(name='Qwen/Qwen3-0.6B-Base'):
    """(Transformers model, tokenizer) of the pinned base, bf16, quietly."""
    with d2l._quiet():
        return d2l.load_hf(name, revision=d2l.HF_REVISIONS[name],
                           dtype=torch.bfloat16)

def _cpt_gsm8k_one(folder, ident, model_fn, items):
    """score_gsm8k of the model model_fn() returns, kept in
    folder/gsm8k.json.gz under ident (read back when ident matches; a file
    of another model or setting is refused)."""
    path = os.path.join(folder, 'gsm8k.json.gz')
    if os.path.exists(path):
        res = d2l._read_result(path, 'json-table')
        d2l._check_config(res, ident, path)
        return res
    model, tok = model_fn()
    res = score_gsm8k(model, tok, items)
    os.makedirs(folder, exist_ok=True)
    d2l._write_result(path, 'json-table', res, dict(config=ident, label=''))
    return res

def _cpt_gsm8k_all(cfgs, arms=None):
    """GSM8K in both formats for the base and the final weights of every
    arm in arms (default: all of cfgs; T2), one Transformers model whose
    weights are swapped per arm; the settings of every cached file include
    the batch sizes (batched greedy decoding in bf16 is not
    batch-invariant) and the scorer's version. Processes on separate
    cards fill the same per-model files with arms=() (the base only) and
    arms=[arm], which the merge then reads back; the result keeps only the
    columns of _CPT_COLUMNS['gsm8k'] (no reply text)."""
    cfg0 = next(iter(cfgs.values()))
    arms = list(cfgs) if arms is None else list(arms)
    if set(arms) - set(cfgs):
        raise ValueError(f'no configuration for arms {set(arms) - set(cfgs)}')
    items = gsm8k_items(_CPT_GSM8K['n'], _CPT_GSM8K['seed'])
    settings = d2l._jsonable(dict(_CPT_GSM8K, batch=_GSM8K_BATCH,
                                  scorer=_GSM8K_SCORER))
    hf = {}

    def model(path=None):  # loaded once, on first need
        if not hf:
            hf['m'], hf['tok'] = _cpt_hf_qwen(cfg0['base'])
            hf['path'] = None
        if path != hf['path']:
            if path is None:
                raise RuntimeError('the base must be scored before an arm')
            _load_into_hf(hf['m'], path)
            hf['path'] = path
        return hf['m'], hf['tok']
    out = [_cpt_gsm8k_one(
        os.path.join(d2l._job_root(), 'cpt', 't2-base'), dict(
            settings=settings, base=_cpt_base_id(cfg0)), model,
        items).assign(arm='base')]
    for arm in arms:
        cfg = cfgs[arm]
        path = os.path.join(_cpt_job(cfg), 'ckpt',
                            f's{cfg["steps"]}.safetensors')
        out.append(_cpt_gsm8k_one(_cpt_job(cfg), dict(
            settings=settings, weights=_file_id(path)),
            lambda p=path: model(p), items).assign(arm=arm))
    hf.clear()
    d2l.lm_free()
    return pd.concat(out, ignore_index=True)[_CPT_COLUMNS['gsm8k']]

def gsm8k_tables(gsm):
    """The page's two GSM8K tables from runs.gsm8k, or None when it is
    empty (the T1 stand-in): per model the 8-shot, chat and forced-answer
    accuracies, the share of chat replies cut at the budget, the share
    that never close </think>, and the share of 8-shot replies writing
    chat markers; per arm the change against the base in points with the
    95% interval of the paired item bootstrap."""
    if not len(gsm):
        return None
    gsm = gsm.assign(open=gsm['thinks'] & ~gsm['closed'])
    by = gsm.groupby(['arm', 'format'], sort=False)[[
        'score', 'forced', 'cut', 'open', 'chat_tokens']].mean().unstack(
        'format')
    pct = lambda v, d=1: f'{100 * v:.{d}f}%'
    rates = pd.DataFrame({
        'model': by.index,
        '8-shot': by['score', 'completion'].map(pct),
        'chat': by['score', 'chat'].map(pct),
        'forced': by['forced', 'chat'].map(pct),
        'cut': by['cut', 'chat'].map(lambda v: pct(v, 0)),
        'no </think>': by['open', 'chat'].map(lambda v: pct(v, 0)),
        '8-shot chat tokens': by['chat_tokens', 'completion'].map(
            lambda v: pct(v, 0))})
    rows = []
    for arm in [a for a in by.index if a != 'base']:
        row = dict(arm=arm)
        for name, fmt, key in (('8-shot', 'completion', 'score'),
                               ('chat', 'chat', 'score'),
                               ('forced', 'chat', 'forced')):
            wide = gsm[gsm['format'] == fmt].pivot_table(
                index='item_id', columns='arm', values=key)
            r = d2l.paired_bootstrap(wide[arm], wide['base'])
            row[name] = (f'{100 * r.mean:+.1f} [{100 * r.ci[0]:+.1f}, '
                         f'{100 * r.ci[1]:+.1f}]')
        rows.append(row)
    return rates, pd.DataFrame(rows, columns=['arm', '8-shot', 'chat',
                                              'forced'])
```

## The Arms

Each arm trains in its own job folder under `../data/lm/rerun/ch21/cpt/`
with `_PretrainRun` (the state, logging and resume machinery of
`d2l.pretrain`): full states every 50 steps (deleted once the arm is done)
and the final weights in bf16. The step is `pretrain_step` on one GPU with
the loss chunk of a 151,936-token vocabulary (2,048 positions, so a chunk's
fp32 logits stay at 1.2 GB); the schedule is linear warmup, then a cosine
to `final_frac` of the peak (or, for the plateau branch, a linear decay to
zero). Interim evaluations (0.5M tokens per set, losses only) run at
step 0 (the training model before its first update, the baseline of the
re-warm bump), every three steps for the first 30 steps (the bump), then
every 15 (T2) or 5 (T1) steps. Arm A also runs the suite at step 15; the arms without
replay at the replay arm's peak get a full evaluation at the step where
they have seen as many domain tokens as the replay arm sees in its whole
run (equal-token accounting). The merge evaluates the base and every arm's
final weights on the full evaluation sets and the suite.

A job folder is named by tier and arm only, so it records what it was
trained from: `arm.json` (written when the arm starts) and `DONE` hold the
arm's configuration and the identity of its starting point (Qwen's pinned
revision, or the T1 preset with its checkpoint and plateau state), and a
folder trained from anything else is refused rather than reused. A T2 arm
keeps up to three full states of about 7 GB while it trains, so it starts
only with `_CPT_MIN_FREE_GB` free on the data disk: its own three states,
the third state of one arm already running, and a margin (arms run at most
two at a time on the 4090 box, whose data disk has under 100 GB free).

```{.python .input #continued-pretraining-lib-arms}
%%tab pytorch
#@save
CptRuns = collections.namedtuple(
    'CptRuns', ['table', 'logs', 'base_docs', 'docs', 'equal', 'items',
                'gsm8k', 'label'])

#@save
_CPT_COLUMNS = {
    'table': ['arm', 'peak_lr', 'replay', 'steps', 'domain_tokens',
              'domain_loss', 'general_loss', 'book_loss', 'suite',
              'gsm8k_completion', 'gsm8k_chat', 'gsm8k_forced', 'tok_s',
              'gib'],
    'logs': ['arm', 'step', 'tokens', 'domain_tokens', 'lr', 'loss',
             'grad_norm', 'domain_loss', 'general_loss', 'book_loss'],
    'base_docs': ['set', 'doc_id', 'sum_nll', 'n_tokens'],
    'docs': ['arm', 'set', 'doc_id', 'sum_nll', 'n_tokens'],
    'equal': ['arm', 'step', 'domain_tokens', 'domain_loss',
              'general_loss', 'book_loss'],
    'items': ['arm', 'step', 'task', 'item_id', 'score'],
    'gsm8k': ['arm', 'format', 'item_id', 'score', 'forced', 'cut',
              'thinks', 'closed', 'chat_tokens', 'new_tokens']}

#@save
_CPT_MICRO = {'t1': 16, 't2': 1}  # sequences per micro-batch (D2 probe:
# Qwen3-0.6B with fp32 master weights runs out of 24 GB at 2 x 2,048)

#@save
_CPT_SUITE_STEP = 15  # arm A's early look at the suite

#@save
_CPT_MIN_FREE_GB = {'t1': 10, 't2': 40}  # free disk before an arm starts

#@save
def _arm_slug(arm):
    return {'A′': 'A2', 'A″': 'A3'}.get(arm, arm)

def _cpt_eval_steps(steps, tier, early=30):
    """Interim evaluation steps: every 3 up to `early` (T1 also steps 1
    and 2, where the cold arms' first updates act), then every 15 (T2) or
    5 (T1), and the last."""
    every, first = (15, set()) if tier == 't2' else (5, {1, 2})
    return sorted({s for s in first if s <= steps} | set(range(
        3, min(early, steps) + 1, 3)) | set(range(
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
        a.setdefault('moments', 'restored' if a['start'] == 'plateau'
                     else 'fresh')
        if a['moments'] not in ('fresh', 'restored') or (
                a['moments'] == 'restored' and a['start'] != 'plateau'):
            raise ValueError(f'arm {name}: moments {a["moments"]!r}; only '
                             "the plateau state has moments to restore")
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
    """pretrain_step on one GPU with the loss chunk of a large vocabulary;
    it clips from the first step, so cfg.clip_from must be None."""
    if cfg.clip_from is not None:
        raise ValueError('_cpt_step clips from the first step; clip_from '
                         f'must be None, not {cfg.clip_from}')
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
    checkpoint (fresh moments) or its plateau state (its weights, with its
    moments restored unless cfg['moments'] is 'fresh')."""
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
        optim = (None if cfg.get('moments', 'restored') == 'fresh'
                 else state['optim'])
        return (model.to(device, dtype), optim, state['cfg']['peak_lr'],
                label)
    model = d2l._load_bf16(ckpt, dtype, device)
    return model, None, d2l.r2_config(preset).peak_lr, label

def _cpt_job(cfg):
    return os.path.join(d2l._job_root(), 'cpt',
                        f'{cfg["tier"]}-{_arm_slug(cfg["arm"])}')

def _file_id(path):
    """sha1 of a file's size and its first and last MiB: quick, and enough
    to tell two trainings of one architecture apart."""
    import hashlib
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, 'rb') as f:
        h.update(f.read(2 ** 20))
        f.seek(max(0, size - 2 ** 20))
        h.update(f.read(2 ** 20))
    return h.hexdigest()

def _cpt_base_id(cfg):
    """What an arm starts from: Qwen's pinned revision, or the T1 preset
    and the identity of its final checkpoint or plateau state."""
    if cfg['tier'] == 't2':
        return [cfg['base'], d2l.HF_REVISIONS[cfg['base']]]
    preset, ckpt, state = _cpt_r2_files()
    return [preset, _file_id(state if cfg.get('start') == 'plateau'
                             else ckpt)]

def _cpt_identity(cfg):
    """The record of an arm in its job folder: configuration, base and
    data (_cpt_data_id), so a folder trained on another corpus is
    refused."""
    return dict(config=d2l._jsonable(cfg), base=_cpt_base_id(cfg),
                data=_cpt_data_id(cfg['tier']))

def _cpt_check_job(job, ident, name='DONE'):
    """Whether job holds file name; raise if that file records another
    arm (configuration or starting point) than ident."""
    path = os.path.join(job, name)
    if not os.path.exists(path):
        return False
    with open(path) as f:
        stored = json.load(f)
    d = d2l._config_diff(ident, {k: stored.get(k) for k in ident})
    if d:
        raise ValueError(f'{job} holds another arm ({d[0]}: {d[2]!r}, this '
                         f'arm has {d[1]!r}); move the folder away to '
                         'retrain it')
    return True

def _cpt_check_disk(cfg):
    """Refuse to start an arm with less than _CPT_MIN_FREE_GB free."""
    path = os.path.abspath(d2l._job_root())
    while not os.path.exists(path):
        path = os.path.dirname(path)
    free = shutil.disk_usage(path).free / 2 ** 30
    need = _CPT_MIN_FREE_GB[cfg['tier']]
    if free < need:
        raise OSError(f'{free:.0f} GiB free under {path}; a {cfg["tier"]} '
                      f'arm starts with at least {need} GiB')

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
    step0 = os.path.join(job, 'step0.json')
    if grid and run.start == 0:  # the arm's own start: the bump's baseline
        losses = _set_losses(_cpt_docs(run.raw, sets, T, _CPT_EVAL_TOKENS))
        with open(step0 + '.tmp', 'w') as f:
            json.dump(losses, f)
        os.replace(step0 + '.tmp', step0)
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
                       domain_tokens=_mix_tokens(mix, done, (0,)), **out)
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
    """The arm's MixStream: reasoning-trace and replay windows in the
    shares (1 - r, r), the data seed for streams and mixture (the book is
    never trained). An arm sees each window at most once: a stream that
    the arm's cfg['steps'] would wrap is refused."""
    T, B, seed, r = (cfg['seq_len'], cfg['global_batch'], cfg['data_seed'],
                     cfg['replay'])
    keys = ('traces', 'replay')
    streams = [d2l.PackedStream(shards[k], T, B, seed=seed) for k in keys]
    mix = d2l.mix_streams(streams, (1 - r, r), B, seed=seed)
    for key, s, n in zip(keys, streams, mix.cursors(cfg['steps'])):
        if n > s.n_seq:
            raise ValueError(f'{key}: {cfg["steps"]} steps take {n} windows '
                             f'of {T} tokens, the stream has {s.n_seq}; an '
                             'arm sees its data once')
    return mix

def _cpt_suite_n(tier):
    return None if tier == 't2' else 2000  # T1: 2,000 items per task

def _cpt_arm_run(cfg):
    """Train one arm (or read its finished job folder): {log, equal,
    suite, peak_lr, gib, tok_s, label}."""
    job, ident = _cpt_job(cfg), _cpt_identity(cfg)
    done = os.path.join(job, 'DONE')
    if not _cpt_check_job(job, ident):
        _cpt_check_job(job, ident, 'arm.json')  # resume only the same arm
        _cpt_check_disk(cfg)
        os.makedirs(job, exist_ok=True)
        with open(os.path.join(job, 'arm.json'), 'w') as f:
            json.dump(ident, f)
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
            json.dump(dict(ident, peak_lr=peak, gib=gib, label=label), f)
        shutil.rmtree(os.path.join(job, 'state'), ignore_errors=True)
        del model
        d2l.lm_free()
    with open(done) as f:
        info = json.load(f)
    log = d2l._read_job_logs(job)
    step0 = os.path.join(job, 'step0.json')
    if os.path.exists(step0):  # the evaluation before the first update
        with open(step0) as f:
            log = pd.concat([pd.DataFrame([dict(
                step=0, tokens=0, domain_tokens=0, **json.load(f))]), log],
                            ignore_index=True)
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
    final weights on the full evaluation sets and the suite, and at T2 on
    GSM8K in two formats."""
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
        if not _cpt_check_job(_cpt_job(cfg), _cpt_identity(cfg)):
            raise FileNotFoundError(f'{_cpt_job(cfg)}: arm {arm} not done')
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
    gsm = (_cpt_gsm8k_all(cfgs) if tier == 't2'
           else pd.DataFrame(columns=C['gsm8k']))
    for col, fmt, key in (('gsm8k_completion', 'completion', 'score'),
                          ('gsm8k_chat', 'chat', 'score'),
                          ('gsm8k_forced', 'chat', 'forced')):
        acc = gsm[gsm['format'] == fmt].groupby('arm')[key].mean()
        table[col] = table['arm'].map(acc)
    return CptRuns(table.reindex(columns=C['table']),
                   pd.concat(logs, ignore_index=True).reindex(
                       columns=C['logs']),
                   base_docs, pd.concat(docs, ignore_index=True)[C['docs']],
                   pd.DataFrame(equal, columns=C['equal']),
                   items[C['items']], gsm, label)

def _cpt_config(arms, base, tokens, seq_len, batch_tokens, warmup=10,
                final_frac=0.1):
    """The configuration stored with a cpt_arms result (data included)."""
    tier = 't2' if base.startswith('Qwen/') else 't1'
    return dict(arms=arms, base=base, tokens=tokens, seq_len=seq_len,
                batch_tokens=batch_tokens, warmup=warmup,
                final_frac=final_frac, data=_cpt_data_id(tier))

def cpt_arms(arms, base, tokens, seq_len, batch_tokens, warmup=10,
             final_frac=0.1):
    """Continue base on reasoning traces with a replay share of
    FineWeb-Edu (the book is evaluated, never trained), one run per arm,
    and evaluate it: CptRuns(table, logs, base_docs, docs, equal, items,
    gsm8k, label). arms = {name: dict(peak_lr, replay, data_seed[, warmup,
    start, schedule, moments])}; peak_lr None is the base run's own peak;
    start 'plateau' starts from the reference run's plateau state instead
    of its final checkpoint, with its AdamW moments restored unless
    moments is 'fresh'; schedule 'cosine' (to final_frac of the peak) or
    'linear' (to zero). base 'Qwen/Qwen3-0.6B-Base' (T2: one regeneration
    per arm, cpt-qwen-A, -B, -C, -A2, merged with GSM8K by cpt-qwen-evals)
    or 'r2' (T1: the regeneration cpt-t1). Otherwise the committed result,
    whose stored configuration, data included, must match; while the Qwen
    arms are pending, the T1 arms under a printed label."""
    config = _cpt_config(arms, base, tokens, seq_len, batch_tokens,
                         warmup, final_frac)
    data = config['data']
    cfgs = _cpt_configs(arms, base, tokens, seq_len, batch_tokens, warmup,
                        final_frac)
    if not base.startswith('Qwen/'):
        res = d2l._pretrain_result('cpt-t1', lambda: _cpt_merge(
            {a: _cpt_arm_run(c) for a, c in cfgs.items()}, cfgs), config)
    else:
        missing = set(arms) - set(_CPT_QWEN_NAMES)
        if missing:
            raise ValueError(f'no regeneration name for arms {missing}')
        parts = {_CPT_QWEN_NAMES[a]: (lambda c=c: _cpt_arm_run(c),
                                      dict(c, data=data))
                 for a, c in cfgs.items()}
        names = {_CPT_QWEN_NAMES[a]: a for a in cfgs}
        res = d2l._pretrain_merged(
            'cpt-qwen-evals', parts, lambda r: _cpt_merge(
                {names[k]: v for k, v in r.items()}, cfgs), config,
            fallback=('cpt-t1', _cpt_config(**_cpt_t1_call())))
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
    """Held-out trace, general and held-out book loss against tokens, one
    panel each, per arm (the arms in `thin` as thin lines), with the base's
    loss as a dashed line."""
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
        ax.set_title(dict(domain='held-out traces', general='FineWeb-Edu',
                          book='held-out book')[s], fontsize=10)
    axes[0].set_ylabel('nats per token')
    if axes[0].get_legend_handles_labels()[0]:  # none while pending
        axes[0].legend(fontsize=8)
    fig.tight_layout()

#@save
Estimate = collections.namedtuple('Estimate', ['value', 'lo', 'hi', 'note',
                                               'ref'], defaults=('',))

#@save
def _estimate(x, note='', ref=''):
    """Estimate from replicates whose last entry is the point estimate."""
    if note:
        return Estimate(math.nan, math.nan, math.nan, note, ref)
    lo, hi = np.percentile(x[:-1], [2.5, 97.5])
    return Estimate(float(x[-1]), float(lo), float(hi), '', ref)

def frontier_excess(fg, arm='B', chord=('C', 'A')):
    """arm's domain gain minus the no-replay reference at arm's
    forgetting, in nats per token: an Estimate (value, lo, hi, note, ref)
    with the 95% interval from the document resamples of forgetting_gain
    (fg.attrs['draws']); ref names the reference. chord = (low, high) are
    two no-replay arms, low forgetting less. A no-replay frontier through
    the base (no gain, no forgetting) is concave, so between low's and
    high's forgetting the chord from low to high lies on or under it, and
    below low's forgetting so does the segment from the base to low: arm
    is measured against the chord, or against the segment if it forgets
    less than low (point estimates decide). Above high's forgetting
    nothing measured bounds the frontier, and at no forgetting (arm's
    point estimate <= 0) the segment's extension bounds it from above, not
    below: there the note says there is no verdict. A note also when an arm is missing, high does not forget more
    than low, or the reference's span of forgetting is not resolved from
    0 (its replicates would divide by about 0)."""
    low, high = chord
    draws = fg.attrs.get('draws', {})
    if not {arm, low, high} <= set(draws):
        missing = sorted({arm, low, high} - set(draws))
        return _estimate(None, f'needs arm {", ".join(missing)}: not in '
                         'this result')
    g, f = (lambda a: draws[a]['gain']), (lambda a: draws[a]['forget'])
    resolved = lambda x: np.percentile(x[:-1], 2.5) > 0
    if f(high)[-1] <= f(low)[-1]:
        return _estimate(None, f'{high} forgets no more than {low}: no '
                         'chord')
    if f(arm)[-1] <= 0:  # the segment's extension bounds from above there
        return _estimate(None, f'{arm} does not forget (outside the '
                         'measured no-replay range): no verdict')
    if f(arm)[-1] > f(high)[-1]:
        return _estimate(None, f"{arm}'s forgetting is outside the measured "
                         f"no-replay range (above {high}'s): no verdict")
    if f(arm)[-1] >= f(low)[-1]:
        span = f(high) - f(low)
        if not resolved(span):
            return _estimate(None, f'{high} minus {low} forgetting is not '
                             'resolved from 0: the chord is undefined')
        ref = g(low) + (g(high) - g(low)) * (f(arm) - f(low)) / span
        return _estimate(g(arm) - ref, ref=f'{low} → {high} chord')
    if not resolved(f(low)):
        return _estimate(None, f"{low}'s forgetting is not resolved from 0: "
                         'the segment is undefined')
    return _estimate(g(arm) - g(low) * f(arm) / f(low),
                     ref=f'base → {low} segment')

def replay_fractions(fg, arm='B', of='A'):
    """arm's domain gain and forgetting as fractions of arm `of`'s: two
    Estimates (kept, forgot) from the same document resamples (a paired
    ratio), or with a note when an arm is missing."""
    draws = fg.attrs.get('draws', {})
    if not {arm, of} <= set(draws):
        missing = sorted({arm, of} - set(draws))
        note = f'{arm} against {of}: needs arm {", ".join(missing)}'
        return _estimate(None, note), _estimate(None, note)
    return tuple(_estimate(draws[arm][k] / draws[of][k])
                 for k in ('gain', 'forget'))

def plot_tradeoff(fg, chord=('C', 'A')):
    """Domain gain against forgetting with 95% intervals per arm, the base
    at the origin, and the no-replay lower bound through the chord's arms
    (the segment from the base to the low arm, then the chord to the high
    arm), as a dashed line when both arms are there."""
    fig, ax = d2l.plt.subplots(figsize=(4.2, 3.2))
    ax.plot([0], [0], 'ks')
    ax.annotate('base', (0, 0), xytext=(6, -10), textcoords='offset points',
                fontsize=9)
    for i, r in enumerate(fg.itertuples()):
        ax.errorbar(r.forget, r.gain, xerr=[[r.forget - r.forget_lo],
                                            [r.forget_hi - r.forget]],
                    yerr=[[r.gain - r.gain_lo], [r.gain_hi - r.gain]],
                    fmt='o', color=f'C{i}', capsize=2, label=r.arm)
    ends = fg.set_index('arm').reindex(list(chord))
    if ends[['gain', 'forget']].notna().all().all():
        ax.plot([0, *ends['forget']], [0, *ends['gain']], color='black',
                ls='--', lw=1)
    ax.axhline(0, color='black', lw=0.6)
    ax.axvline(0, color='black', lw=0.6)
    ax.set_xlabel('forgetting (nats/token)')
    ax.set_ylabel('trace gain (nats/token)')
    if ax.get_legend_handles_labels()[0]:  # none while pending
        ax.legend(fontsize=8)
    fig.tight_layout()

def early_gaps(runs, pairs, seeds, within=40):
    """General-loss gaps over steps 1 to `within` (nats per token), at
    the evaluation steps the arms share: per pair (a, b) the largest
    a - b and its step; for `seeds` (one configuration, other data orders)
    the largest difference between any two of them and its step, the
    yardstick. Defined whether or not an arm has a transient (claims 8b
    and 8c, F21 review finding 3: rewarm_bump is NaN for a loss still
    rising at the window's end). A DataFrame comparison / gap / step; a
    comparison with an arm missing is left out."""
    logs = runs.logs.dropna(subset=['general_loss'])
    logs = logs[(logs['step'] > 0) & (logs['step'] <= within)]
    loss = logs.pivot_table(index='step', columns='arm',
                            values='general_loss')
    rows = []
    for a, b in pairs:
        d = (loss[a] - loss[b]).dropna() if {a, b} <= set(loss) else ()
        if len(d):
            rows.append(dict(comparison=f'{a} − {b}', gap=d.max(),
                             step=int(d.idxmax())))
    L = loss[[a for a in seeds if a in loss]].dropna()
    if L.shape[1] >= 2 and len(L):
        spread = L.max(axis=1) - L.min(axis=1)
        rows.append(dict(comparison=f"between {seeds[0]}'s seeds",
                         gap=spread.max(), step=int(spread.idxmax())))
    return pd.DataFrame(rows, columns=['comparison', 'gap', 'step'])
```

## The SmolLM3 Trajectory

Six base-model checkpoints of `HuggingFaceTB/SmolLM3-3B-checkpoints` (the
ends of the three pretraining stages, the two long-context stages and the
mid-training checkpoint), each read with four instruments. Token counts:
steps times the global batch of the stage configurations
(`smollm3-configs`: 192 replicas of 3 sequences of 4,096 tokens, 2,359,296
tokens per step; the long-context stages keep 2,359,296 tokens per step
over 20,000 steps each); mid-training adds the blog's 140B tokens.

- **Cloze suite** (the six tasks, MMLU on a seeded 2,000-item subset): the
  per-item scores of `lm-smollm3-trajectory-v1`, scored at the same pins
  with the same suite (version 2 changes neither), are copied; without
  that artifact, or once a pin differs from the one version 1 scored
  (`_SMOLLM3_V1_PINS`) or its MMLU subset is not `_SMOLLM3_CLOZE`'s, the
  suite is scored afresh.
- **Bits per byte with no prefix token** on 500 FineWeb-Edu validation
  documents, 500 peS2o validation documents, every held-out book section
  and the 500 held-out reasoning-trace conversations (`traces (seen)`:
  OpenThoughts3-1.2M, which SmolLM3's reasoning stage trained on, rendered
  with `chatml`, one window of the first 2,048 tokens each). SmolLM3
  trained with document masks and no BOS, so a document starts at
  position 0 and its first token, which has no context, goes unscored (and
  any next tokens up to the end of its character), as do the bytes it
  spells (`_scored_doc`). Version 1 put `<|begin_of_text|>` before each
  document instead, which the base checkpoints never saw in that place:
  it costs mid-training's first token 26.5 nats on average and inflates
  its book number by about 5% (S/investigate/smollm3/report.md, section
  3). Long documents are read in windows of 2,048 tokens advancing by
  1,024, each target scored once (`bits_per_byte`'s windows).
- **Calibration**, from the same forward passes: per document the loss at
  softmax temperatures `_SMOLLM3_TEMPS` (the logits divided by T), the
  entropy of the T = 1 distribution and the number of targets that are
  its top-1 token (`_logit_stats`).
- **GSM8K in two formats** (`score_gsm8k`): the 500 items and both
  formats of the Qwen arms (8-shot completion, budget 256; one ChatML user
  turn with thinking, budget 4,096, the answer after `</think>`, a cut
  reply or a think block never closed counts as no answer), greedy, with
  the forced-answer readout. The prompt is the same for all six
  checkpoints: the mid-training tokenizer's chat template is plain ChatML,
  which is `gsm8k_chat_prompt`, and the base checkpoints ship none.
  `<|end_of_text|>` ends a completion. Chat replies run 32 at a time:
  SmolLM3-3B's key-value cache is 36 layers x 4 heads x 128 x 2 x 2 B =
  72 KiB per position, about 10 GiB at 32 replies of 4,096 + 300
  positions, next to 5.8 GiB of weights (launch with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, F21 review).

Two tokenizers serve all six checkpoints, so every checkpoint scores the
same token ids: LC 64K's for the bits per byte and the cloze suite (it
reads `<code>` as text, which the base checkpoints saw), mid-training's
for generation (it decodes mid-training's own `<code>` tokens). The six
branches' tokenizers have the same vocabulary and the same ChatML and
think tokens; mid-training turns two reserved ids into `<code>` and
`</code>`.

One regeneration per checkpoint (`smollm3-0` to `-5`, about 85 minutes
each on a 4090: 16 GiB allocated at most, about 21 GiB reserved),
merged when all six exist. Each keeps its bits-per-byte rows and its
GSM8K replies (with their text, so they can be read and re-scored) in
`../data/lm/rerun/ch21/smollm3/<checkpoint>/`, so an interrupted job
resumes; the merged artifact holds no text. The bits-per-byte cache and
each part record the scoring version (`_BPB_SCORER`) and the identity of
the texts read (`_bpb_texts_id`: per domain their number and the SHA-1
of their NFC forms), so a cache or part of other texts or rules is
refused, and the merge checks every part against the texts it reads.
The table's `forced` column is the chat accuracy with forced answers
and `cut` the share of chat replies stopped at the budget; a cut reply
scores 0 in `chat`, so `chat / (1 - cut)` is the accuracy of the
replies that were not cut. A gate on the first 100 GSM8K items decides
the conditional rungs. The harder MATH-500 rung, run only if LC 64K or
mid-training scored above 85% under ChatML or the two within 5 points,
did not run on the gate's `chat` column (84% and 63%); with forced
answers mid-training is at 93% and the rule would fire, so the removal
holds only if the author's ruling on the chat budget (shared with the
Qwen arms' claim 13) keeps `chat` at 4,096 tokens; commit `4b08ffe6d`
holds the rung's code. `smollm3-symbolic`, a part of the same merge
once its name is in `PRETRAIN_RERUN_NAMES` (with `PRETRAIN_JOBS` and
`_RERUN_ARTIFACT` entries like `smollm3-0`'s), runs only if the full
GSM8K shows a chat gain of mid-training over LC 64K whose interval
excludes 0; it scores GSM-Symbolic (`gsm_symbolic_items`; chapter 18's
`gsm-symbolic-test`, CC BY-NC-ND 4.0: read at run time, no text kept in
the artifact) for LC 64K and mid-training, read by `symbolic_check`
(both chat gains of mid-training over LC 64K with paired intervals).
`cloze_task_deltas` gives the per-task cloze changes the page shows
only as thin lines. `smollm3_gsm8k` is the page's live check: two
checkpoints on the first n of the 500 items, both formats, no forced
answers.

```{.python .input #continued-pretraining-lib-smollm3}
%%tab pytorch
#@save
SmolTrajectory = collections.namedtuple(
    'SmolTrajectory', ['table', 'bpb', 'items', 'docs', 'answers', 'label'])

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
_SMOLLM3_DOMAINS = ('web', 'science', 'book', 'traces')

#@save
_SMOLLM3_TEMPS = (0.8, 1.0, 1.25, 1.5, 2.0)  # the calibration grid

#@save
_SMOLLM3_CTX = 2048  # window of the bits per byte; a trace's one window

#@save
_SMOLLM3_ANSWERS = {  # task: what score_gsm8k is given
    'gsm8k': dict(n=500, seed=0, budget=dict(completion=256, chat=4096),
                  force=32, batch=dict(completion=32, chat=32, force=8)),
    'gsm-symbolic': dict(instances=5, budget=dict(completion=256,
                                                  chat=4096),
                         force=32, batch=dict(completion=32, chat=32,
                                              force=8))}

#@save
_SMOLLM3_LIVE = dict(budget=dict(completion=256, chat=4096),
                     batch=dict(completion=25, chat=25))  # the page's check

#@save
_SMOLLM3_RUNGS = {  # conditional regeneration: (task, checkpoints)
    'smollm3-symbolic': ('gsm-symbolic', ('lc64k', 'mid'))}

#@save
_BPB_SCORER = 1  # raise when _scored_doc, _doc_stats or _logit_stats change

#@save
_SMOLLM3_CLOZE = dict(tasks=list(SMOLLM3_TASKS), n_task=dict(mmlu=2000),
                      subset_seed=0)  # the cloze suite (_cpt_suite's seed)

#@save
_SMOLLM3_V1_PINS = {  # the revisions lm-smollm3-trajectory-v1 scored
    'stage1': 'd07a5a83dd011f3f084e9d2f1b47f51e524ca8d4',
    'stage2': 'eebe6fccea5aa05cf48b0e630d10eeee97f92fc5',
    'stage3': '20e7817e636d6b1f63cffbee0f384721b1f4bb67',
    'lc32k': '24cc7e71eb21218315ef02c257b192cdd263b302',
    'lc64k': 'd4d10f417b85b7e385c163118183748bd67a34f6',
    'mid': '0485ec16b618f88f6dbb27b23dbdecca1d6a1cdd'}

#@save
def _smollm3_call():
    return dict(names=('stage1', 'stage2', 'stage3', 'lc32k', 'lc64k',
                       'mid'))

def _bpb_texts(n=500):
    """{domain: texts}: the first n FineWeb-Edu validation documents, the
    first n of peS2o's validation file 0, every held-out book section,
    and the held-out reasoning-trace conversations rendered with chatml
    (traces)."""
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
    traces = [chatml(m) for m in trace_conversations()['messages']]
    return dict(web=web, science=science,
                book=[t for _, t in book_split()[1]], traces=traces)

def _smollm3_tokenizer(name):
    """The pinned tokenizer of checkpoint name ('lc64k' or 'mid'), with
    the config's clean_up_tokenization_spaces=True turned off (a BPE
    tokenizer ignores it, and warns on every decode that does not say so)."""
    import transformers
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES[f'smollm3-{name}']
    with d2l._quiet():
        return transformers.AutoTokenizer.from_pretrained(
            repo, revision=rev, clean_up_tokenization_spaces=False)

def _smollm3_load(name):
    """Checkpoint name in bf16 on the GPU, quietly (no progress bars)."""
    from transformers.utils import logging as hf_logging
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES[f'smollm3-{name}']
    bars = hf_logging.is_progress_bar_enabled()
    hf_logging.disable_progress_bar()  # a notebook shows them as widgets
    try:
        with d2l._quiet():
            model, _ = d2l.load_hf(repo, revision=rev)
    finally:
        if bars:
            hf_logging.enable_progress_bar()
    return model

def _scored_doc(tokenizer, text, limit=None):
    """(ids, first, n_bytes) of a document read with no prefix token: its
    token ids (no special tokens added; at most the first `limit`, fewer
    when the limit falls inside a character), how many of them go
    unscored at the start (the first token, which has no context, and the
    next ones up to the end of the character it starts, as
    _lm_unscored_head), and the UTF-8 bytes of the text the scored tokens
    spell."""
    dec = lambda x: tokenizer.decode(x, clean_up_tokenization_spaces=False)
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    spelled = text
    if limit is not None and len(ids) > limit:
        ids = ids[:limit]
        while ids and not text.startswith(dec(ids)):  # a cut character
            ids = ids[:-1]
        spelled = dec(ids)
    first = 1
    while first < len(ids):
        head = dec(ids[:first])
        if text.startswith(head) or '\ufffd' not in head:
            break
        first += 1
    head = dec(ids[:first]) if ids else ''
    return ids, first, len(spelled.encode('utf-8')) - len(
        head.encode('utf-8'))

def _logit_stats(z, y, temps=_SMOLLM3_TEMPS):
    """Per position of fp32 logits z (n, V) with targets y (n,): the NLL of
    y at each temperature in temps (the logits divided by T before the
    softmax), the entropy of the T = 1 distribution, and 1 where y is its
    top-1 token: an (n, len(temps) + 2) tensor."""
    logp = F.log_softmax(z, -1)
    nll = [F.cross_entropy(z / t, y, reduction='none') for t in temps]
    return torch.stack(nll + [-(logp.exp() * logp).sum(-1),
                              (z.argmax(-1) == y).float()], 1)

@torch.no_grad()
def _doc_stats(model, docs, ctx_len=_SMOLLM3_CTX, stride=None,
               batch_size=4, temps=_SMOLLM3_TEMPS):
    """Per document (ids, first) of docs, summed over its scored targets
    ids[first:]: the NLL at each temperature, the entropy and the top-1
    count of _logit_stats, and the number of targets, as an array (n_docs,
    len(temps) + 3). A Transformers model with a plain linear output layer
    reads windows of ctx_len tokens advancing by stride (default
    ctx_len // 2), every target scored once, as bits_per_byte does with no
    boundary token; the scored positions are projected in fp32, 1,024 at
    a time."""
    stride = ctx_len // 2 if stride is None else stride
    windows = []
    for d, (seq, first) in enumerate(docs):
        seq, done = list(seq), first - 1  # targets seq[first:done + 1] done
        while done < len(seq) - 1:
            end = min(max(ctx_len, done + 1) if done == first - 1
                      else done + stride, len(seq) - 1)
            start = max(end - ctx_len, 0)
            windows.append((d, seq[start:end], seq[done + 1:end + 1]))
            done = end
    head = d2l._label_output_layer(model)
    if head is None:
        raise ValueError('needs a Transformers model with a linear head')
    W = head.weight.float()
    bias = None if head.bias is None else head.bias.float()
    dev = next(model.parameters()).device
    decoder = d2l._lm_unwrap(model).get_decoder()
    out = np.zeros((len(docs), len(temps) + 3))
    for s in range(0, len(windows), batch_size):
        w = windows[s:s + batch_size]
        T = max(len(x) for _, x, _ in w)
        X = torch.tensor([x + [0] * (T - len(x)) for _, x, _ in w],
                         device=dev)
        mask = (torch.arange(T, device=dev) < torch.tensor(
            [len(x) for _, x, _ in w], device=dev)[:, None]).long()
        H = decoder(input_ids=X, attention_mask=mask,
                    use_cache=False).last_hidden_state
        H = torch.cat([H[k, len(x) - len(ys):len(x)]
                       for k, (_, x, ys) in enumerate(w)])
        y = torch.tensor([t for _, _, ys in w for t in ys], device=dev)
        stats = torch.cat([_logit_stats(F.linear(
            H[i:i + 1024].float(), W, bias), y[i:i + 1024], temps)
            for i in range(0, len(y), 1024)]).cpu().numpy()
        doc = np.concatenate([[d] * len(ys) for d, _, ys in w])
        np.add.at(out, doc, np.concatenate(
            [stats, np.ones((len(doc), 1))], 1))
    return out

def _stat_columns(temps=_SMOLLM3_TEMPS):
    return [f'nll_t{t:g}' for t in temps] + ['entropy', 'top1', 'tokens']

def _smollm3_docs(model, tokenizer, texts):
    """Per document of texts {domain: texts} (each in NFC form): domain,
    doc, bytes, the sums of _doc_stats and bpb (the T = 1 NLL in bits per
    scored byte); a trace is its first _SMOLLM3_CTX tokens."""
    import unicodedata
    rows, docs = [], []
    for dom, ts in texts.items():
        for i, t in enumerate(ts):
            ids, first, n_bytes = _scored_doc(
                tokenizer, unicodedata.normalize('NFC', t),
                _SMOLLM3_CTX if dom == 'traces' else None)
            if len(ids) > first:
                rows.append(dict(domain=dom, doc=i, bytes=n_bytes))
                docs.append((ids, first))
    out = pd.DataFrame(rows).join(pd.DataFrame(_doc_stats(model, docs),
                                               columns=_stat_columns()))
    out['tokens'] = out['tokens'].astype(np.int64)
    out['bpb'] = out['nll_t1'] / (math.log(2) * out['bytes'])
    return out

def gsm_symbolic_items(instances=5):
    """GSM-Symbolic (apple/GSM-Symbolic, chapter 18's gsm-symbolic-test,
    CC BY-NC-ND 4.0: read at run time, never stored): instances 0 to
    instances - 1 of each of its 100 templates, as gsm8k_items (id
    'template-instance', question, reference, the 8-shot prompt of
    gsm8k_cot)."""
    with d2l._quiet():
        path = d2l.download_keyed('gsm-symbolic-test')
    with open(path, encoding='utf-8') as f:
        rows = sorted((json.loads(line) for line in f),
                      key=lambda r: (r['id'], r['instance']))
    return [dict(id=f'{r["id"]}-{r["instance"]}', question=r['question'],
                 reference=d2l.extract_answer(r['answer'], 'gsm8k',
                                              strict=True),
                 prompt=d2l.fewshot_prompt(d2l.GSM8K_COT_DEMOS,
                                           r['question'], 'qa'))
            for r in rows if r['instance'] < instances]

def _smollm3_items(task):
    """The items of a reasoning task of _SMOLLM3_ANSWERS."""
    cfg = _SMOLLM3_ANSWERS[task]
    if task == 'gsm8k':
        return gsm8k_items(cfg['n'], cfg['seed'])
    return gsm_symbolic_items(cfg['instances'])

def _smollm3_cached(name, what, ident, compute):
    """compute(), kept in ../data/lm/rerun/ch21/smollm3/<name>/<what>
    .json.gz under ident (read back when ident matches; a file of another
    setting is refused), so an interrupted regeneration resumes."""
    path = os.path.join(d2l._job_root(), 'smollm3', name,
                        f'{what}.json.gz')
    if os.path.exists(path):
        res = d2l._read_result(path, 'json-table')
        d2l._check_config(res, ident, path)
        return res
    res = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d2l._write_result(path, 'json-table', res, dict(config=ident, label=''))
    return res

def _smollm3_settings(task):
    """The settings a cached or merged answer file of task records."""
    return d2l._jsonable(dict(_SMOLLM3_ANSWERS[task], scorer=_GSM8K_SCORER,
                              tokenizer=d2l.PRETRAIN_SOURCES[
                                  'smollm3-mid'][2]))

def _smollm3_answers(model, name, task, tokenizer):
    """score_gsm8k of checkpoint name on task (its settings in
    _SMOLLM3_ANSWERS), cached per checkpoint, with task and checkpoint
    columns."""
    cfg = _SMOLLM3_ANSWERS[task]
    res = _smollm3_cached(name, task, dict(pin=d2l.PRETRAIN_SOURCES[
        f'smollm3-{name}'][2], settings=_smollm3_settings(task)),
        lambda: score_gsm8k(model(), tokenizer, _smollm3_items(task),
                            budget=cfg['budget'], batch_size=cfg['batch'],
                            force=cfg['force'],
                            end_token='<|end_of_text|>'))
    return res.assign(task=task, checkpoint=_SMOLLM3[name][0])

def _smollm3_cloze(name):
    """Checkpoint name's per-item cloze scores from
    lm-smollm3-trajectory-v1, or None (score the suite afresh) when that
    artifact is not registered, when the checkpoint's pin is no longer the
    one version 1 scored (_SMOLLM3_V1_PINS), when the suite's tasks are
    not _SMOLLM3_CLOZE's, or when a subset task does not have
    _SMOLLM3_CLOZE's item count (MMLU's seeded 2,000)."""
    key = 'lm-smollm3-trajectory-v1'
    if key not in d2l.DATA_HUB or d2l.PRETRAIN_SOURCES[
            f'smollm3-{name}'][2] != _SMOLLM3_V1_PINS[name]:
        return None
    with d2l._quiet():
        items = d2l._read_result(d2l.download_keyed(key), 'json-table').items
    got = items[items['checkpoint'] == _SMOLLM3[name][0]]
    counts = got['task'].value_counts()
    if set(got['task']) != set(_SMOLLM3_CLOZE['tasks']) or any(
            counts.get(t, 0) != n
            for t, n in _SMOLLM3_CLOZE['n_task'].items()):
        return None
    return got[['task', 'item_id', 'score', 'checkpoint']].reset_index(
        drop=True)

def _bpb_texts_id(texts):
    """{domain: {n, sha1}} of texts {domain: texts}: their number and the
    SHA-1 of their NFC forms, so that a cached or merged reading of other
    texts is refused."""
    import unicodedata
    out = {}
    for dom, ts in texts.items():
        h = hashlib.sha1()
        for t in ts:
            h.update(unicodedata.normalize('NFC', t).encode('utf-8'))
            h.update(b'\0')
        out[dom] = dict(n=len(ts), sha1=h.hexdigest())
    return out

def _smollm3_point(name, texts):
    """One checkpoint (regeneration smollm3-k): per-item cloze scores,
    per-document bits per byte and calibration sums (docs), GSM8K in both
    formats with the replies' text (answers), its context length and the
    identity of the texts read (_bpb_texts_id)."""
    tok, chat_tok = _smollm3_tokenizer('lc64k'), _smollm3_tokenizer('mid')
    held = {}

    def model():  # loaded on first need: cached parts need no weights
        if 'm' not in held:
            held['m'] = _smollm3_load(name)
        return held['m']
    pin = d2l.PRETRAIN_SOURCES[f'smollm3-{name}'][2]
    items = _smollm3_cloze(name)
    if items is None:  # no committed scores to copy: score the suite
        suite = _cpt_suite(_SMOLLM3_CLOZE['tasks'],
                           n_task=_SMOLLM3_CLOZE['n_task'], name='smollm3',
                           batch_size=8)
        with d2l._quiet():
            res = suite.run(model(), tok)
        items = res.per_item[['task', 'item_id', 'score']].assign(
            checkpoint=_SMOLLM3[name][0])
    text_id = _bpb_texts_id(texts)
    docs = _smollm3_cached(name, 'docs', dict(
        pin=pin, ctx=_SMOLLM3_CTX, temps=_SMOLLM3_TEMPS, scorer=_BPB_SCORER,
        tokenizer=d2l.PRETRAIN_SOURCES['smollm3-lc64k'][2], texts=text_id),
        lambda: _smollm3_docs(model(), tok, texts)).assign(
            checkpoint=_SMOLLM3[name][0])
    answers = _smollm3_answers(model, name, 'gsm8k', chat_tok)
    import transformers
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES[f'smollm3-{name}']
    with d2l._quiet():
        ctx = transformers.AutoConfig.from_pretrained(
            repo, revision=rev).max_position_embeddings
    held.clear()
    d2l.lm_free()
    return dict(items=items, docs=docs, answers=answers, ctx=int(ctx),
                texts=text_id)

def _smollm3_rung(task, names):
    """A conditional rung: task of _SMOLLM3_ANSWERS ('gsm-symbolic') for
    checkpoints names, in both formats."""
    chat_tok, out = _smollm3_tokenizer('mid'), []
    for name in names:
        held = {}

        def model(name=name):  # loaded on first need
            if 'm' not in held:
                held['m'] = _smollm3_load(name)
            return held['m']
        out.append(_smollm3_answers(model, name, task, chat_tok))
        held.clear()
        d2l.lm_free()
    return dict(answers=pd.concat(out, ignore_index=True))

def _smollm3_answer_columns():
    return ['checkpoint', 'task'] + _CPT_COLUMNS['gsm8k'][1:]

def _smollm3_merge(results, names, texts=None):
    """SmolTrajectory from {part: result} of smollm3_trajectory (no reply
    text): table per checkpoint (tokens seen, context length, cloze mean,
    GSM8K 8-shot and chat accuracy, chat accuracy with forced answers,
    the share of chat replies cut at the budget, bits per byte per
    domain), bpb per domain, the per-item cloze
    scores, per-document rows and every answer row. The parts must have
    read the same texts (their _bpb_texts_id), and those must be texts
    when it is given (the texts a regeneration reads now)."""
    order = list(_smollm3_call()['names'])
    pts = {n: _part(results[f'smollm3-{order.index(n)}']) for n in names}
    ids = {n: pts[n]['texts'] for n in names if pts[n].get('texts')}
    want = texts if texts is not None else next(iter(ids.values()), None)
    for n, got in ids.items():
        d = d2l._config_diff(d2l._jsonable(want), d2l._jsonable(got))
        if d:
            raise ValueError(f'part smollm3-{order.index(n)} read other '
                             f'texts: {d[0]} {d[2]!r} != {d[1]!r}')
    label = [_SMOLLM3[n][0] for n in names]
    items = pd.concat([pts[n]['items'] for n in names], ignore_index=True)
    answers = pd.concat([pts[n]['answers'] for n in names] + [
        _part(r)['answers'] for p, r in results.items()
        if p in _SMOLLM3_RUNGS], ignore_index=True)[_smollm3_answer_columns()]
    docs = pd.concat([pts[n]['docs'] for n in names], ignore_index=True)
    g = docs.assign(bits=docs['bpb'] * docs['bytes']).groupby(
        ['checkpoint', 'domain'], sort=False)[['bits', 'bytes']].sum()
    bpb = (g['bits'] / g['bytes']).unstack('domain').reindex(
        index=label, columns=list(_SMOLLM3_DOMAINS)).rename_axis(
        'checkpoint').reset_index().rename_axis(columns=None)
    gsm = answers[answers['task'] == 'gsm8k'].groupby(
        ['checkpoint', 'format'])[['score', 'forced', 'cut']].mean().unstack(
        'format')
    acc = lambda col, fmt: gsm[col, fmt].reindex(label).to_numpy() if (
        col, fmt) in gsm else np.full(len(label), np.nan)
    table = pd.DataFrame({
        'checkpoint': label,
        'tokens': [f'{_SMOLLM3[n][1] / 1e3:.1f}T' for n in names],
        'ctx': [f'{pts[n]["ctx"] // 1024}K' for n in names],
        'cloze': suite_mean(items, by=('checkpoint',)).reindex(
            label).to_numpy(),
        '8-shot': acc('score', 'completion'), 'chat': acc('score', 'chat'),
        'forced': acc('forced', 'chat'), 'cut': acc('cut', 'chat')})
    table = table.merge(bpb, on='checkpoint', how='left')
    return SmolTrajectory(table, bpb, items, docs, answers, '')

def smollm3_trajectory(names):
    """SmolLM3-3B's checkpoints `names` (of 'stage1', 'stage2', 'stage3',
    'lc32k', 'lc64k', 'mid'): SmolTrajectory(table, bpb, items, docs,
    answers, label). table: checkpoint, tokens seen, context length, cloze
    mean, GSM8K accuracy in the 8-shot and chat formats, chat accuracy
    with forced answers (a cut reply's answer forced, as score_gsm8k's
    forced), the share of chat replies cut at the budget (a cut reply
    scores 0 in chat, so chat / (1 - cut) is the accuracy of the replies
    not cut) and bits per byte (no prefix token) on
    web, science, book and traces (seen); bpb: those four columns; items:
    per-item cloze scores; docs: per document its bytes, scored tokens,
    summed NLL at each temperature of _SMOLLM3_TEMPS, entropy and top-1
    count; answers: per item and format GSM8K (and the conditional rung's
    GSM-Symbolic) scores. The committed full-set result
    (regenerations smollm3-0 to smollm3-5, one checkpoint each)."""
    names = list(names)
    order = list(_smollm3_call()['names'])
    held = {}

    def texts():  # read once per process, and only by a regeneration
        if 't' not in held:
            held['t'] = _bpb_texts()
        return held['t']
    bpb = dict(ctx=_SMOLLM3_CTX, temps=_SMOLLM3_TEMPS, scorer=_BPB_SCORER)
    parts = {f'smollm3-{order.index(k)}': (
        lambda k=k: _smollm3_point(k, texts()), dict(
            name=k, pin=d2l.PRETRAIN_SOURCES[f'smollm3-{k}'][2], **bpb,
            cloze=_SMOLLM3_CLOZE, gsm8k=_smollm3_settings('gsm8k')))
        for k in names}
    for rung, (task, ckpts) in _SMOLLM3_RUNGS.items():
        if rung in d2l.PRETRAIN_RERUN_NAMES:  # ruled in by the gate (F26)
            parts[rung] = (lambda task=task, ckpts=ckpts: _smollm3_rung(
                task, ckpts), dict(task=task, checkpoints=list(ckpts),
                                   settings=_smollm3_settings(task)))
    config = dict(names=names, **bpb, cloze=_SMOLLM3_CLOZE,
                  gsm8k=_smollm3_settings('gsm8k'),
                  rungs={r: _smollm3_settings(_SMOLLM3_RUNGS[r][0])
                         for r in parts if r in _SMOLLM3_RUNGS})
    res = d2l._pretrain_merged(None, parts, lambda r: _smollm3_merge(
        r, names, _bpb_texts_id(texts())), config)
    cols = ['checkpoint', *_SMOLLM3_DOMAINS]
    return d2l._show_label(d2l._as_tuple(SmolTrajectory, res, dict(
        table=['checkpoint', 'tokens', 'ctx', 'cloze', '8-shot', 'chat',
               'forced', 'cut', *_SMOLLM3_DOMAINS],
        bpb=cols, items=['task', 'item_id', 'score', 'checkpoint'],
        docs=['domain', 'doc', 'bytes', *_stat_columns(), 'bpb',
              'checkpoint'],
        answers=_smollm3_answer_columns())))

def smollm3_gsm8k(names, n=50):
    """GSM8K scored live for SmolLM3 checkpoints names on the first n of
    the 500 committed items, in both formats (greedy, the committed
    budgets, no forced answers), one checkpoint loaded at a time: a
    DataFrame checkpoint, format, item_id, score, cut and the other flags
    of score_gsm8k (no text)."""
    cfg = _SMOLLM3_ANSWERS['gsm8k']
    items = gsm8k_items(cfg['n'], cfg['seed'])[:n]
    tok, out = _smollm3_tokenizer('mid'), []
    for name in names:
        model = _smollm3_load(name)
        res = score_gsm8k(model, tok, items, budget=_SMOLLM3_LIVE['budget'],
                          batch_size=_SMOLLM3_LIVE['batch'], force=0,
                          end_token='<|end_of_text|>')
        out.append(res.drop(columns=_GSM8K_TEXT).assign(
            checkpoint=_SMOLLM3[name][0]))
        del model
        d2l.lm_free()
    res = pd.concat(out, ignore_index=True)
    return res[['checkpoint'] + [c for c in res.columns if c != 'checkpoint']]

def _paired_points(wide, a, b, n_boot, seed):
    """Mean and 95% interval of b - a in points over the rows of wide
    where both are present (paired items)."""
    import warnings
    keep = wide[[a, b]].notna().all(axis=1)
    if not keep.any():
        return math.nan, math.nan, math.nan
    with warnings.catch_warnings():  # a smoke run has under 100 items
        warnings.simplefilter('ignore')
        r = d2l.paired_bootstrap(wide[b][keep], wide[a][keep], n_boot,
                                 seed=seed)
    return 100 * r.mean, 100 * r.ci[0], 100 * r.ci[1]

#@save
_SMOLLM3_SHORT = {'stage 1 end': 'S1', 'stage 2 end': 'S2',
                  'stage 3 end': 'S3', 'LC 32K': 'LC32K', 'LC 64K': 'LC64K',
                  'mid-training': 'mid'}

#@save
def _transition(a, b):
    return f'{_SMOLLM3_SHORT.get(a, a)} → {_SMOLLM3_SHORT.get(b, b)}'

def _answer_wide(traj, task, fmt, value='score'):
    """Column value (score or forced) of task in format fmt: items x
    checkpoints."""
    a = traj.answers[(traj.answers['task'] == task)
                     & (traj.answers['format'] == fmt)]
    return a.pivot_table(index='item_id', columns='checkpoint',
                         values=value)

def answer_deltas(traj, task, pairs=None, n_boot=2000, seed=0):
    """Paired changes (points, items paired, 95% bootstrap interval) of
    task's 8-shot and chat accuracy between checkpoint pairs (a, b), by
    default each checkpoint holding task against the next one that does:
    columns transition, completion, chat, each
    with _lo and _hi."""
    cols = ['transition'] + [f'{k}{q}' for k in ('completion', 'chat')
                             for q in ('', '_lo', '_hi')]
    wide = {fmt: _answer_wide(traj, task, fmt)
            for fmt in ('completion', 'chat')}
    if pairs is None:
        held = set().union(*(w.columns for w in wide.values()))
        have = [c for c in traj.table['checkpoint'] if c in held]
        pairs = list(zip(have, have[1:]))
    rows = []
    for a, b in pairs:
        row = dict(transition=_transition(a, b))
        for fmt, t in wide.items():
            row[fmt], row[f'{fmt}_lo'], row[f'{fmt}_hi'] = (
                _paired_points(t, a, b, n_boot, seed)
                if {a, b} <= set(t.columns) else (math.nan,) * 3)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)

def symbolic_check(traj, n_boot=2000, seed=0):
    """The contamination check of the GSM-Symbolic rung: mid-training's
    chat gain over LC 64K (points, items paired, 95% interval) on GSM8K
    and on GSM-Symbolic, and the ratio of the second to the first (NaN
    while a task is missing); a ratio under 0.5 is the plan's sign that
    part of the GSM8K gain is memorized test items."""
    rows = []
    for task in ('gsm8k', 'gsm-symbolic'):
        r = answer_deltas(traj, task, [('LC 64K', 'mid-training')], n_boot,
                          seed).iloc[0]
        rows.append(dict(task=task, gain=r['chat'], lo=r['chat_lo'],
                         hi=r['chat_hi']))
    out = pd.DataFrame(rows)
    out['ratio'] = [math.nan, out['gain'][1] / out['gain'][0]
                    if out['gain'][0] else math.nan]
    return out

def cloze_task_deltas(traj, n_boot=2000, seed=0):
    """Per adjacent transition of traj and cloze task: the paired change
    of the task's accuracy in points (items paired) with its 95%
    bootstrap interval; columns transition, task, delta, lo, hi."""
    wide = traj.items.pivot_table(index=['task', 'item_id'],
                                  columns='checkpoint', values='score')
    names, rows = list(traj.table['checkpoint']), []
    for a, b in zip(names, names[1:]):
        for task in SMOLLM3_TASKS:
            if task not in wide.index.get_level_values('task'):
                continue
            d = _paired_points(wide.loc[task], a, b, n_boot, seed) if {
                a, b} <= set(wide.columns) else (math.nan,) * 3
            rows.append(dict(transition=_transition(a, b), task=task,
                             delta=d[0], lo=d[1], hi=d[2]))
    return pd.DataFrame(rows, columns=['transition', 'task', 'delta', 'lo',
                                       'hi'])

def stage_deltas(traj, n_boot=2000, seed=0):
    """Changes between adjacent checkpoints of traj, each with a 95%
    bootstrap interval: the cloze mean in points (every task weighs the
    same; items paired), GSM8K's 8-shot and chat accuracy and its chat
    accuracy with forced answers in points (items paired), and each
    domain's byte-weighted bits per byte in percent (documents paired).
    Columns transition, then cloze, gsm8k_completion, gsm8k_chat,
    gsm8k_forced, web, science, book and traces, each with _lo and
    _hi."""
    names = list(traj.table['checkpoint'])
    wide = traj.items.pivot_table(index=['task', 'item_id'],
                                  columns='checkpoint', values='score')
    tasks = wide.index.get_level_values('task')
    w = pd.Series(1 / tasks.map(tasks.value_counts()).to_numpy(),
                  index=wide.index)
    w = w / w.mean()  # every task weighs the same in the cloze mean
    answer = {f'gsm8k_{fmt}': _answer_wide(traj, 'gsm8k', fmt)
              for fmt in ('completion', 'chat')}
    answer['gsm8k_forced'] = _answer_wide(traj, 'gsm8k', 'chat', 'forced')
    docs = traj.docs.assign(bits=traj.docs['bpb'] * traj.docs['bytes'])
    keys = ['cloze', *answer, *_SMOLLM3_DOMAINS]
    rows = []
    for a, b in zip(names, names[1:]):
        row = dict(transition=_transition(a, b))
        row['cloze'], row['cloze_lo'], row['cloze_hi'] = _paired_points(
            wide.mul(w, axis=0), a, b, n_boot, seed)
        for k, t in answer.items():
            row[k], row[f'{k}_lo'], row[f'{k}_hi'] = (
                _paired_points(t, a, b, n_boot, seed)
                if {a, b} <= set(t.columns) else (math.nan,) * 3)
        for dom in _SMOLLM3_DOMAINS:
            d = docs[docs['domain'] == dom].pivot_table(
                index='doc', columns='checkpoint', values=['bits', 'bytes'])
            if not len(d) or not {a, b} <= set(d['bits'].columns):
                row.update(dict.fromkeys([dom, f'{dom}_lo', f'{dom}_hi'],
                                         math.nan))
                continue
            bits, n = d['bits'][[a, b]].to_numpy(), d['bytes'][a].to_numpy()
            idx = np.stack([*d2l.cluster_resample(np.arange(len(n)), n_boot,
                                                  seed), np.arange(len(n))])
            bpb = [bits[idx, k].sum(1) / n[idx].sum(1) for k in (0, 1)]
            est = _estimate(100 * (bpb[1] / bpb[0] - 1))
            row.update({dom: est.value, f'{dom}_lo': est.lo,
                        f'{dom}_hi': est.hi})
        rows.append(row)
    return pd.DataFrame(rows, columns=['transition'] + [
        f'{k}{q}' for k in keys for q in ('', '_lo', '_hi')])

def smollm3_calibration(traj):
    """Per checkpoint and domain of traj, token-weighted over the
    documents: loss (nats per token at T = 1), entropy, top-1 (accuracy),
    best T (the temperature of _SMOLLM3_TEMPS with the lowest loss), loss
    at best T, and recovered: the share of the loss's rise over the
    previous checkpoint that the best temperature takes back (NaN when
    the loss did not rise or the best temperature is 1)."""
    cols = ['checkpoint', 'domain', 'loss', 'entropy', 'top-1', 'best T',
            'loss at best T', 'recovered']
    if not len(traj.docs):
        return pd.DataFrame(columns=cols)
    stat = _stat_columns()
    g = traj.docs.groupby(['checkpoint', 'domain'], sort=False)[stat].sum()
    order = [(c, d) for c in traj.table['checkpoint']
             for d in _SMOLLM3_DOMAINS if (c, d) in g.index]
    g = g.loc[order]
    per = g[stat[:len(_SMOLLM3_TEMPS)]].div(g['tokens'], axis=0)
    best = per.to_numpy().argmin(1)
    out = pd.DataFrame(dict(
        checkpoint=[c for c, _ in order], domain=[d for _, d in order],
        loss=per['nll_t1'].to_numpy(),
        entropy=(g['entropy'] / g['tokens']).to_numpy(),
        top1=(g['top1'] / g['tokens']).to_numpy(),
        best=np.asarray(_SMOLLM3_TEMPS)[best],
        at_best=per.to_numpy()[np.arange(len(per)), best]))
    prev = dict(zip(traj.table['checkpoint'][1:], traj.table['checkpoint']))
    loss = out.set_index(['checkpoint', 'domain'])['loss']
    rise = np.array([loss[c, d] - loss.get((prev.get(c), d), math.nan)
                     for c, d in order])
    ok = (rise > 0) & (out['best'] != 1.0).to_numpy()
    out['recovered'] = math.nan
    out.loc[ok, 'recovered'] = (out['loss'] - out['at_best'])[ok] / rise[ok]
    out.columns = cols
    return out

def plot_smollm3(traj):
    """Across the checkpoints: accuracy (left; the cloze mean with its 95%
    interval over items, every task weighing the same, the six cloze
    tasks as thin lines, GSM8K 8-shot and chat) and bits per byte per
    domain (right)."""
    import warnings
    fig, (ax1, ax2) = d2l.plt.subplots(1, 2, figsize=(8, 3.2))
    t = traj.table
    x = np.arange(len(t))
    d2l.plot(x, [t['cloze'], t['8-shot'], t['chat']], None, 'accuracy',
             legend=['cloze mean', 'GSM8K 8-shot', 'GSM8K chat'],
             fmts=('o-', 's--', '^-.'), axes=ax1,
             figsize=(8, 3.2))
    per = traj.items.groupby(['checkpoint', 'task'])['score'].mean()
    for task in SMOLLM3_TASKS:
        ys = [per.get((c, task), math.nan) for c in t['checkpoint']]
        ax1.plot(x, ys, '-', color='C0', linewidth=0.6, alpha=0.35)
    lo, hi = [], []
    for c in t['checkpoint']:
        g = traj.items[traj.items['checkpoint'] == c]
        if not len(g):
            lo.append(math.nan), hi.append(math.nan)
            continue
        wt = 1 / g.groupby('task')['score'].transform('size')
        with warnings.catch_warnings():  # a smoke run has under 100 items
            warnings.simplefilter('ignore')
            r = d2l.bootstrap_ci(g['score'] * wt / wt.mean(), 2000)
        lo.append(r.ci[0]), hi.append(r.ci[1])
    ax1.errorbar(x, t['cloze'], yerr=np.abs(np.stack([
        t['cloze'] - lo, np.asarray(hi) - t['cloze']])), fmt='none',
        color='C0', capsize=3)
    d2l.plot(x, [traj.bpb[d] for d in _SMOLLM3_DOMAINS], None,
             'bits per byte',
             legend=['web', 'science', 'book', 'traces (seen)'],
             fmts=('o-', 's-', '^-', 'D--'), axes=ax2, figsize=(8, 3.2))
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(t['checkpoint'], rotation=30, ha='right',
                           fontsize=8)
        for text in (ax.get_legend().get_texts()
                     if ax.get_legend() is not None else ()):
            text.set_fontsize(7)
    fig.tight_layout()

def plot_smollm3_calibration(calib):
    """Entropy against loss (nats per token), one point per checkpoint
    and domain of smollm3_calibration's table, with the diagonal (a
    calibrated model's entropy equals its loss); the mid-training points
    are labelled."""
    if not len(calib):
        return
    doms = [d for d in _SMOLLM3_DOMAINS if d in set(calib['domain'])]
    sub = [calib[calib['domain'] == d] for d in doms]
    top = float(max(calib['loss'].max(), calib['entropy'].max())) * 1.05
    d2l.plot([s['loss'] for s in sub] + [[0, top]],
             [s['entropy'] for s in sub] + [[0, top]],
             'loss (nats per token)', 'entropy (nats per token)',
             legend=[*doms, 'entropy = loss'],
             fmts=[('o', 's', '^', 'D')[i] for i in range(len(doms))]
             + ['k:'], figsize=(4.2, 3.4))
    ax = d2l.plt.gca()
    for s in sub:
        for _, r in s[s['checkpoint'] == 'mid-training'].iterrows():
            ax.annotate('mid', (r['loss'], r['entropy']), fontsize=7,
                        xytext=(4, -8), textcoords='offset points')
```
