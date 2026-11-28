# Long Context: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is
never rendered or executed as a notebook: `make lib` only scans it for
`#@save` blocks, and `gen_notebooks.py` excludes it via `LIB_ONLY_FILES`.
It holds the library code of :numref:`sec_lm_long_context`
that the section explains in a sentence but does not show
(docs/code-guide.md 7.1):

- the section's registries: committed results (`PRETRAIN_ARTIFACTS`),
  the long regeneration jobs (`PRETRAIN_JOBS`), each the page's call
  written a second time;
- the data (`long_data`, `test_books`): the long half (PG-19 books and long
  peS2o papers), the short replay and the 100 held-out test books, whole,
  from which every evaluation cuts windows at seeded random offsets;
- the needle probe behind the page's `needle_windows`: the word pools
  (`needle_pools`), the insertion points, the scorer (`needle_recall`: one
  prefill, then each key's two-token query on the cached window), the
  oracle copier, the pilot's diagnostics (`needle_diagnostics`), the octave
  bins of gap (`recall_by_gap`), and the format comparison with the
  key-value template of the first build (cached candidate scorer for
  multi-token values);
- the loss measurements: per-target losses of whole and sliding windows,
  loss against the context given, repeat gaps, the rare-token set,
  window-resampled intervals, the FLOP count on the meta device;
- the evaluations: zero-shot (`lc_zeroshot`), the reference model
  Qwen3-0.6B-Base (`lc_reference`), the format table and reach during
  pretraining (`lc_probe_runs`), and the arm driver (`lc_arms`: one
  resumable job folder per arm and seed, the length stages, the periodic
  needle evaluation, one final evaluation per model kept in its job folder
  so that the models of a tier can be evaluated on separate cards, and the
  merge), the memory probe of the first build (`lc_memory_probe`), the
  section's plots, and the displays the page calls (`show_lc_*`: the
  result's label first, then a table at the precision the prose quotes).

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble inside its functions (`copy`, `hashlib`, `itertools`,
`tempfile`, `transformers`, `torch.utils.flop_counter`).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_long_context`

## Registries

The hosted inputs (d2l-data R2): the long half (PG-19 train books and long
peS2o papers, uploaded 2026-11-25) and the 100 PG-19 test books, whole
(`lm-pg19-test-v2`, uploaded 2026-11-28; `-v1` held the first 16,385
tokens of 64 books, too short for windows at random offsets), both
R2-tokenized shard sets built by `tools/build_pretraining_data.py --stage
lc-long` and `--stage pg19-test` (PG-19: Apache-2.0, public-domain texts;
peS2o: ODC-By 1.0; the attribution is in each manifest). The `-v1` result
keys of the first build stay registered until the pruning tool lists them;
no page reads them.

```{.python .input #long-context-lib-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-lc-long-60m-v1'] = (
    d2l.LM_DATA_URL + 'lm-lc-long-60m-v1.tar',
    '24ddc3182a630192b02dfaf9e8fb474f4554cd9f')
d2l.DATA_HUB['lm-pg19-test-v1'] = (
    d2l.LM_DATA_URL + 'lm-pg19-test-v1.tar',
    'a92784936e0d0da6a0623c0b9bd22d335b3a39df')
d2l.DATA_HUB['lm-pg19-test-v2'] = (
    d2l.LM_DATA_URL + 'lm-pg19-test-v2.tar',
    '974337dadd192fe4b08a97be65a89b6272e1249c')
d2l.DATA_HUB['lm-lc-t1-v1'] = (
    d2l.LM_DATA_URL + 'lm-lc-t1-v1.json.gz',
    '05a0b45828fbfe613d5b63e0eaa3e275b618f0c2')
d2l.DATA_HUB['lm-lc-evals-v1'] = (
    d2l.LM_DATA_URL + 'lm-lc-evals-v1.json.gz',
    '68aa67ec366d4b4618a79524d4909d1717ccb7f2')
d2l.DATA_HUB['lm-r2-16k-pi-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-16k-pi-v1.safetensors',
    '1cec7e5dcb3d7ff9f42d25835831eb556ac967c2')
d2l.DATA_HUB['lm-r2-16k-yarn-v1'] = (
    d2l.LM_DATA_URL + 'lm-r2-16k-yarn-v1.safetensors',
    'aa606cf42aca9bfea1169b40f99fa2148e0b9792')
```

The section's results and inputs. The evaluations of the needle probe are
new keys (`lm-lc-probe-v1`, `lm-lc-t1-v2`, `lm-lc-evals-v2`); a
regeneration run shows a result it has not computed yet as pending (the T2
cells with the T1 arms as their stand-in) instead of failing.

```{.python .input #long-context-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['lc-zeroshot'] = (  # the zero-shot part
    'lm-lc-evals-v2', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lc-reference'] = (  # the reference part
    'lm-lc-evals-v2', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lc-t1'] = (
    'lm-lc-t1-v2', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lc-evals'] = (
    'lm-lc-evals-v2', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lc-probe'] = (
    'lm-lc-probe-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lc-long'] = (
    'lm-lc-long-60m-v1', 'shards')
d2l.PRETRAIN_ARTIFACTS['pg19-test'] = (
    'lm-pg19-test-v2', 'shards')
```

The long jobs that `tools/lm_regen.py` runs (one card each): the zero-shot
evaluation, the reference model, the format table with reach during
pretraining, the T2 arms (seed 0 and the progressive arm are the first
build's runs, read from their finished job folders), their final
evaluation and merge (`--only t2-<arm>-s<seed>` or `--only t2-base`
evaluates one model, so the models can run on separate cards), and the T1
arms (also a notebook regeneration).

```{.python .input #long-context-lib-jobs}
%%tab pytorch
#@save
d2l.PRETRAIN_JOBS['lc-t1'] = ('lc_arms', lambda: d2l._lc_t1_call())
d2l.PRETRAIN_JOBS['lc-zeroshot'] = ('lc_zeroshot',
                                    lambda: d2l._lc_zeroshot_call())
d2l.PRETRAIN_JOBS['lc-reference'] = ('lc_reference',
                                     lambda: d2l._lc_reference_call())
d2l.PRETRAIN_JOBS['lc-probe'] = ('lc_probe_runs',
                                 lambda: d2l._lc_probe_call())
d2l.PRETRAIN_JOBS['lc-t2-pi'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-yarn'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-prog'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-pi-s1'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-pi-s2'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-yarn-s1'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-yarn-s2'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-evals'] = ('lc_arms', lambda: d2l._lc_t2_call())
```

## The Page's Calls

The configurations the page passes (the load check compares them with the
committed runs), the regeneration name of each T2 arm and seed, the suite,
and the evaluation of each tier: the trained length of the base, the
extended length, needle windows per length in the final, zero-shot and
periodic evaluations, the loss windows, and the contexts of the
loss-against-context curve.

```{.python .input #long-context-lib-calls}
%%tab pytorch
#@save
LC_TASKS = ('hellaswag', 'arc_easy', 'piqa', 'sciq', 'winogrande')

#@save
_LC_SCHEMES = {'none': dict(scheme='none', s=4),
               'PI': dict(scheme='pi', s=4),
               'NTK': dict(scheme='ntk', s=4),
               'YaRN': dict(scheme='yarn', s=4),
               'YaRN, mscale 1': dict(scheme='yarn', s=4, mscale=1.0)}

#@save
_LC_ARMS = {
    'PI': dict(scheme='pi', stages=((16_384, 4, 1.0),), seeds=(0, 1, 2)),
    'YaRN': dict(scheme='yarn', stages=((16_384, 4, 1.0),), seeds=(0, 1, 2)),
    'YaRN-prog': dict(scheme='yarn', stages=((8_192, 2, 0.4),
                                             (16_384, 4, 0.6)), seeds=(0,))}

#@save
_LC_ARMS_T1 = {
    'PI': dict(scheme='pi', stages=((4096, 4, 1.0),), seeds=(0, 1, 2, 3, 4)),
    'YaRN': dict(scheme='yarn', stages=((4096, 4, 1.0),),
                 seeds=(0, 1, 2, 3, 4)),
    'YaRN-prog': dict(scheme='yarn', stages=((2048, 2, 0.4),
                                             (4096, 4, 0.6)), seeds=(0,))}

#@save
_LC_T2_NAMES = {('PI', 0): 'lc-t2-pi', ('YaRN', 0): 'lc-t2-yarn',
                ('YaRN-prog', 0): 'lc-t2-prog', ('PI', 1): 'lc-t2-pi-s1',
                ('PI', 2): 'lc-t2-pi-s2', ('YaRN', 1): 'lc-t2-yarn-s1',
                ('YaRN', 2): 'lc-t2-yarn-s2'}

#@save
_LC_TIERS = {
    't2': dict(trained=4096, length=16_384, needles=500, zs_needles=250,
               loss_windows=64, contexts=(256, 1024, 4096), eval_needles=50,
               eval_windows=16),
    't1': dict(trained=1024, length=4096, needles=1000, zs_needles=250,
               loss_windows=64, contexts=(256, 1024), eval_needles=50,
               eval_windows=16)}

#@save
_LC_NEAR = (32, 512)  # the near gaps of every table

#@save
_LC_GROWTH = (256, 1024)  # the gaps of reach growth during pretraining

#@save
_LC_GROWTH_STEPS = (1000, 6000)  # r2's reach growth: from step to step

#@save
_LC_SHORT = 256  # the sliding context of the gain, the first context

#@save
_LC_REFERENCE = 'Qwen/Qwen3-0.6B-Base'  # ruling (d) of 2026-11-25

#@save
_LC_PROBE = dict(length=1024, items=200, depth=0.5, windows=500)

#@save
_LC_SUITE_N = 2000  # suite items per task (fewer where a task has fewer)

#@save
_LC_VAL_TOKENS = 2 ** 21  # the val loss at the trained length

#@save
_LC_MICRO = {'t1': 8, 't2': 1}  # sequences per micro-batch (D2 probe)

#@save
_LC_EVAL_EVERY = 64  # steps between the periodic evaluations

#@save
_LC_MIN_FREE_GB = {'t1': 5, 't2': 20}  # free disk before an arm starts

#@save
def _lc_zeroshot_call():
    """lc_zeroshot's arguments in the page's zero-shot cell."""
    return dict(base='r2', schemes=_LC_SCHEMES, lengths=(4096, 16384))

def _lc_reference_call():
    """lc_reference's arguments (the merge folds the result in)."""
    return dict(model_id=_LC_REFERENCE, lengths=(4096, 16384))

def _lc_probe_call():
    """lc_probe_runs' arguments in the page's format cell."""
    return dict(t1_steps=(509, 1017, 1526),
                r2_steps=(500, 1000, 2000, 3000, 4000, 6000))

def _lc_t2_call():
    """lc_arms' arguments in the page's T2 cell."""
    return dict(arms=_LC_ARMS, base='r2', tokens=100_000_000,
                seq_len=16_384, batch_tokens=262_144, peak_lr=1e-4,
                warmup=20, decay_frac=0.15, long_share=0.5)

def _lc_t1_call():
    """lc_arms' arguments in the page's T1 cell."""
    return dict(arms=_LC_ARMS_T1, base='t1', tokens=16_000_000,
                seq_len=4096, batch_tokens=65_536)
```

## The Data

The long half is the hosted `lm-lc-long-60m-v1` (PG-19 train books in file
order, then peS2o papers of at least 8,192 tokens; T1 reads its first
shard, PG-19 books only), the short replay FineWeb-Edu shard
`train-00032` of the 4B set (never consumed by the reference run; T1:
`lm-fwedu-replay-12m-v1`), and the 100 PG-19 test books, whole. The long
peS2o papers come from the validation file `validation-00001-of-00002`
(s2orc/valid) and serve here as training data; nothing in this section
evaluates on peS2o, and the held-out peS2o of continued pretraining is the
other validation file (s2ag/valid), so no evaluation text is trained on.

`test_books` cuts window `i` from a book (cycling over the books long
enough) at an offset drawn from a generator of its own, seeded by
`(seed, T, i)`: the first 16 of 64 windows are the same 16 windows, so the
periodic evaluations of an arm and the final one share books, and every
model of one tokenizer sees the same text. A model with another tokenizer
(the reference model) reads the books decoded from the R2 tokens and
encoded with its own tokenizer, at its own offsets.

```{.python .input #long-context-lib-data}
%%tab pytorch
#@save
LcData = collections.namedtuple('LcData', ['table', 'shards', 'label'])

#@save
_LC_LOCAL = {'lc-long': os.path.join('..', 'data', 'lm', 'lc-long-v1'),
             'pg19-test': os.path.join('..', 'data', 'lm', 'pg19-test-v2')}

#@save
_LC_BOOKS = {}  # tokenizer id: the test books in its tokens

#@save
def _lc_shards(name, max_tokens=None):
    """A shard set of this section: the hosted one, or the builder's local
    output before its upload; max_tokens keeps a prefix of shards."""
    key = d2l.PRETRAIN_ARTIFACTS[name][0]
    if key in d2l.DATA_HUB:
        root = d2l.load_artifact(name).root
    elif os.path.exists(os.path.join(_LC_LOCAL[name], 'manifest.json')):
        root = _LC_LOCAL[name]
    else:
        raise KeyError(f'{name} is not committed yet; build it with '
                       f'tools/build_pretraining_data.py --stage {name}')
    return d2l.TokenShards(root, 'train', max_tokens=max_tokens)

def _shard_subset(shards, keep):
    """A TokenShards over the shards with indices keep only."""
    import copy
    s = copy.copy(shards)
    s.shards = [shards.shards[k] for k in keep]
    s._docs = np.cumsum([0] + [x['n_docs'] for x in s.shards])
    s._toks = np.cumsum([0] + [x['n_tokens'] for x in s.shards])
    s.n_docs, s.n_tokens = int(s._docs[-1]), int(s._toks[-1])
    s.n_shards, s._mm, s._idx = len(s.shards), {}, {}
    return s

def _lc_replay(tier):
    """The short replay: 4B shard train-00032 (T2) or the T1 replay."""
    if tier == 't1':
        if 'lm-fwedu-replay-12m-v1' in d2l.DATA_HUB:
            return d2l.load_artifact('cpt-fwedu-replay-12m')
        return d2l.TokenShards(os.path.join(d2l._CPT_DIRS['t1'], 'replay'),
                               'train')
    with d2l._quiet():
        s = d2l.fwedu_shards('4b')
    k = next(i for i, x in enumerate(s.shards)
             if x['file'].endswith('train-00032'))
    return _shard_subset(s, [k])

def _lc_parts_table(shards):
    """part, documents, tokens, median length, source of each stream."""
    rows = []
    meta = pd.concat([shards['long'].meta(k)
                      for k in range(shards['long'].n_shards)])
    for part, name, src in (('pg19', 'PG-19 train books', 'PG-19 (PD)'),
                            ('pes2o', 'long peS2o papers',
                             'peS2o v2 (ODC-By)')):
        m = meta[meta['part'] == part]
        if len(m):
            rows.append((name, len(m), int(m['n_tokens'].sum()),
                         int(m['n_tokens'].median()), src))
    for key, name, src in (('short', 'short replay', 'FineWeb-Edu (ODC-By)'),
                           ('test', 'PG-19 test books', 'PG-19 (PD)')):
        s = shards[key]
        n = pd.concat([s.meta(k) for k in range(s.n_shards)])['n_tokens']
        rows.append((name, s.n_docs, s.n_tokens, int(n.median()), src))
    return pd.DataFrame(rows, columns=['part', 'documents', 'tokens',
                                       'median length', 'source'])

def long_data(tier='t2'):
    """The arms' data: LcData(table, shards, label). shards: 'long' (the
    long half; T1: its first shard), 'short' (the replay) and 'test' (the
    100 test books); table: per part documents, tokens, median length in
    tokens and source."""
    if tier not in ('t1', 't2'):
        raise ValueError(f"tier is 't1' or 't2', not {tier!r}")
    long = _lc_shards('lc-long', 8_000_000 if tier == 't1' else None)
    shards = dict(long=long, short=_lc_replay(tier),
                  test=_lc_shards('pg19-test'))
    return LcData(_lc_parts_table(shards), shards, '')

def _tok_id(tokenizer):
    """A short id of a tokenizer's vocabulary, cached on the object."""
    if '_d2l_tok_id' not in tokenizer.__dict__:
        import hashlib
        vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
        tokenizer.__dict__['_d2l_tok_id'] = hashlib.sha1(
            json.dumps(vocab).encode()).hexdigest()[:12]
    return tokenizer.__dict__['_d2l_tok_id']

def _encode(tokenizer, text):
    """Token ids of text (no special tokens), quietly (a book is longer
    than any tokenizer's nominal length)."""
    with d2l._quiet():
        return tokenizer(text, add_special_tokens=False)['input_ids']

def _test_book_tokens(tokenizer=None):
    """The test books (without their EOS) as int64 arrays in the tokens of
    tokenizer (default and the reference tokenizer: the stored R2 tokens;
    any other: the decoded text encoded with it), cached."""
    r2 = lc_tokenizer()
    key = _tok_id(tokenizer or r2)
    if key not in _LC_BOOKS:
        s = _lc_shards('pg19-test')
        books = [s.doc(i)[:-1].astype(np.int64) for i in range(s.n_docs)]
        if key != _tok_id(r2):
            books = [np.asarray(_encode(tokenizer, r2.decode(b.tolist())),
                                dtype=np.int64) for b in books]
        _LC_BOOKS[key] = books
    return _LC_BOOKS[key]

def test_books(n, T, seed=0, tokenizer=None):
    """n windows of T + 1 tokens of the PG-19 test books as a LongTensor
    (n, T + 1): window i from book i (cycling over the books longer than
    T + 1 tokens) at an offset seeded by (seed, T, i), in the tokens of
    tokenizer (default the reference tokenizer's)."""
    books = [b for b in _test_book_tokens(tokenizer) if len(b) > T + 1]
    out = []
    for i in range(n):
        b = books[i % len(books)]
        a = int(np.random.default_rng([seed, T, i]).integers(len(b) - T))
        out.append(b[a:a + T + 1])
    return torch.from_numpy(np.stack(out).astype(np.int64))

def _val_windows(T, tokens=None):
    """Windows (n, T + 1) of the FineWeb-Edu validation shard, the first
    tokens (default _LC_VAL_TOKENS) tokens, and its EOS id."""
    with d2l._quiet():
        s = d2l.fwedu_shards('val')
    n = (tokens or _LC_VAL_TOKENS) // T
    W = np.stack([s.read(i * T, T + 1) for i in range(n)]).astype(np.int64)
    return torch.from_numpy(W), s.eos_id
```

## The Needle Probe

The word pools (`needle_pools`) start from the reference tokenizer's
single-token words: tokens that decode to a space and 4 to 9 lowercase
letters, ranked 200 to 8,000 by their counts in the first 2^24 tokens of
the 100M shard. Even-ranked words are keys and must be one token on their
own (a key follows a line break, in a needle line and in the query);
odd-ranked words are values and must be one token with their leading
space. The same word list serves every tokenizer, filtered by that
tokenizer's tokens. `_needle_slots` draws the gaps log-uniform in
(32, L - 64], sorted so that the farthest needle comes first, snaps each
needle to the nearest line end of the window's text, and redraws until the
needles are at least 16 text tokens apart. The scorer `needle_recall`
prefills a window once (2,048 tokens at a time) and then runs each query's
two tokens on the cached window, writing them into the same two cache
slots every time, so no query sees another and the cache is never copied.
The diagnostics are the pilot's three checks of a miss (the oracle copier;
every pool word one token in the form the window uses; no key or value in
its window's text) and the uniqueness of each answer. The per-bin Wilson
intervals of `recall_by_gap` (and so the reach) treat the queries as
independent, which they are not quite: the eight queries of a window share
its text, and the windows come from 100 books (several per book at random
offsets, which can overlap at 16K). Differences of recall
(`recall_paired` for the same queries of two models or checkpoints,
`recall_unpaired` for queries of different windows) resample whole
windows instead.

```{.python .input #long-context-lib-needles}
%%tab pytorch
#@save
Needles = collections.namedtuple('Needles', ['prefix', 'keys', 'values',
                                             'gaps', 'nl'])

#@save
_LC_QUERY_COLUMNS = ['window', 'needle', 'gap', 'correct']

#@save
def lc_tokenizer(max_len=65_536):
    """The reference tokenizer with its nominal length (4,096) raised to
    max_len tokens."""
    tok = d2l.r2_tokenizer()
    tok.model_max_length = max_len
    return tok

def _needle_words(lo=200, hi=8000, tokens=2 ** 24):
    """Lowercase words of 4-9 letters that are one reference token with a
    leading space, in the order of their counts among the first tokens of
    the 100M shard, ranks lo to hi (cached)."""
    key = (lo, hi, tokens)
    if getattr(_needle_words, 'key', None) != key:
        with d2l._quiet():
            s = d2l.fwedu_shards('100m')
            tok = d2l.r2_tokenizer()
        counts = np.bincount(s.read(0, min(tokens, s.n_tokens)).astype(
            np.int64), minlength=len(tok))
        order = np.argsort(-counts, kind='stable')[lo:hi]
        word = re.compile(r' [a-z]{4,9}')
        texts = tok.batch_decode([[int(i)] for i in order])
        _needle_words.words = [t[1:] for t in texts if word.fullmatch(t)]
        _needle_words.key = key
    return _needle_words.words

def needle_pools(tokenizer, words=None):
    """(keys, values): disjoint lists of (word, token id) for the needles
    of tokenizer. Keys are the even-ranked words of words (default
    _needle_words()) that are one token on their own, values the odd-ranked
    ones that are one token with a leading space."""
    words = list(_needle_words() if words is None else words)
    cache = tokenizer.__dict__.setdefault('_d2l_needle_pools', {})
    if tuple(words) not in cache:
        forms = [w if i % 2 == 0 else ' ' + w for i, w in enumerate(words)]
        with d2l._quiet():
            ids = tokenizer(forms, add_special_tokens=False)['input_ids']
        pools = ([], [])
        for i, (w, x) in enumerate(zip(words, ids)):
            if len(x) == 1:
                pools[i % 2].append((w, int(x[0])))
        cache[tuple(words)] = pools
    return cache[tuple(words)]

def _line_ends(tokenizer):
    """A bool tensor over the vocabulary: the tokens whose text ends with a
    line break (cached on the tokenizer)."""
    if '_d2l_line_ends' not in tokenizer.__dict__:
        texts = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
        tokenizer.__dict__['_d2l_line_ends'] = torch.tensor(
            [t.endswith('\n') for t in texts])
    return tokenizer.__dict__['_d2l_line_ends']

def _text_words(tokenizer, ids):
    """The lowercase words of the text of ids."""
    text = tokenizer.decode([int(i) for i in ids]).lower()
    return set(re.findall(r'[a-z]+', text))

def _needle_pick(pool, words, text, k, generator):
    """k entries of pool drawn with generator, none whose word is in words
    or whose token occurs in text."""
    present = set(text.tolist())
    ok = [p for p in pool if p[0] not in words and p[1] not in present]
    if len(ok) < k:
        raise ValueError(f'{len(ok)} pool words left for {k} needles')
    return [ok[i] for i in torch.randperm(len(ok), generator=generator)[
        :k].tolist()]

def _needle_slots(ends, L, k, generator, sep=16, tries=10_000):
    """(gaps, at) of k needles in a window of L tokens whose text (the
    filler) has line ends ends (bool, one per text token): at[j] is the text
    token after which needle j goes (increasing, at least sep apart), and
    gaps[j] = L - 1 - v_j its gap, with v_j = at[j] + 3 j + 2 the position
    of its value; the gaps are log-uniform in (32, L - 64] before each
    needle is snapped to the nearest line end, and every snapped gap stays
    in that range."""
    where = torch.nonzero(ends).flatten()
    if not len(where):
        raise ValueError('the window has no line break')
    j = torch.arange(k)
    for _ in range(tries):
        gap = torch.empty(k, dtype=torch.float64).uniform_(
            math.log(33), math.log(L - 64), generator=generator).exp()
        target = (L - 1) - gap.sort(descending=True).values - 3 * j - 2
        hi = torch.searchsorted(where.double(), target).clamp(
            max=len(where) - 1)
        lo = (hi - 1).clamp(min=0)
        closer = (target - where[lo]).abs() <= (where[hi] - target).abs()
        at = torch.where(closer, where[lo], where[hi])
        gaps = (L - 1) - (at + 3 * j + 2)
        if ((at[1:] - at[:-1]) >= sep).all() and (gaps > 32).all() and (
                gaps <= L - 64).all():
            return gaps, at
    raise ValueError(f'no layout of {k} needles in {tries} draws')

def _needle_logits(model, needles, i, chunk=2048):
    """Logits (n_needles, vocabulary) at the query of each needle of window
    i: one prefill of the prefix into the KV cache (chunk tokens at a
    time), then each query '\\n<key>' on the cached prefix, always in the
    same two cache slots, so no query sees another."""
    dev = _device_of(model)
    P = needles.prefix[i].to(dev)
    T = len(P)
    ks, vs = model.init_cache(1, T + 2)
    for t in range(0, T, chunk):
        model.forward_cached(P[None, t:t + chunk], ks, vs, t)
    out = []
    for k in needles.keys[i].tolist():
        Q = torch.tensor([[needles.nl, k]], device=dev)
        out.append(model.forward_cached(Q, ks, vs, T)[0][0])
    return torch.stack(out)

def needle_recall(model, needles, chunk=2048):
    """Per query of needles: DataFrame(window, needle, gap, correct). The
    prediction is the value with the largest logit among the window's
    values at the query (_needle_logits; chance 1 / n_needles). bf16
    autocast on a GPU."""
    dev, rows = _device_of(model), []
    was_training = model.training
    model.eval()
    auto = torch.autocast(dev.type, torch.bfloat16,
                          enabled=dev.type == 'cuda')
    with torch.inference_mode(), auto:
        for i in range(len(needles.prefix)):
            logits = _needle_logits(model, needles, i, chunk)
            pick = logits[:, needles.values[i].to(dev)].argmax(-1).cpu()
            rows += [(i, j, int(needles.gaps[i, j]), int(p) == j)
                     for j, p in enumerate(pick)]
    model.train(was_training)
    return pd.DataFrame(rows, columns=_LC_QUERY_COLUMNS)

def needle_oracle(needles):
    """needle_recall's rows for the oracle copier: it predicts the token
    after the last earlier occurrence of the queried key."""
    rows = []
    for i, P in enumerate(needles.prefix):
        for j, k in enumerate(needles.keys[i].tolist()):
            at = (P[:-1] == k).nonzero().flatten()
            pred = int(P[at[-1] + 1]) if len(at) else -1
            rows.append((i, j, int(needles.gaps[i, j]),
                         pred == int(needles.values[i, j])))
    return pd.DataFrame(rows, columns=_LC_QUERY_COLUMNS)

def needle_diagnostics(tokenizer, needles):
    """The pilot's checks of a miss, as bools: 'oracle' (the oracle copier
    recalls every query of these windows), 'single' (every pool word is one
    token in the form the windows use: a key on its own and after a line
    break, '\\n<key>' encoding as the line break and the key; a value with
    its leading space on its own and after a key, '\\n<key> <value>'
    encoding as three tokens), 'absent' (no key or value of a window occurs
    in that window's text, as a word or as a token), and 'unique' (the
    needle tokens sit where the gaps say, each key once in its window, the
    values of a window distinct)."""
    keys, values = needle_pools(tokenizer)
    nl = _encode(tokenizer, '\n')
    k0 = keys[0] if keys else ('', None)
    with d2l._quiet():
        in_line = tokenizer(['\n' + w for w, _ in keys] + [
            f'\n{k0[0]} {w}' for w, _ in values],
            add_special_tokens=False)['input_ids']
    single = (all(_encode(tokenizer, w) == [i] for w, i in keys)
              and all(_encode(tokenizer, ' ' + w) == [i] for w, i in values)
              and all(x == nl + [i] for x, (_, i) in zip(in_line, keys))
              and all(x == nl + [k0[1], i] for x, (_, i) in zip(
                  in_line[len(keys):], values)))
    word = {i: w for w, i in keys + values}
    absent = unique = True
    for i, P in enumerate(needles.prefix):
        v = (len(P) + 1) - needles.gaps[i]  # the values' positions
        keep = torch.ones(len(P), dtype=torch.bool)
        keep[torch.cat([v - 1, v, v + 1])] = False
        text = P[keep]
        found, ids = _text_words(tokenizer, text), set(text.tolist())
        mine = needles.keys[i].tolist() + needles.values[i].tolist()
        absent &= all(word.get(t) not in found and t not in ids
                      for t in mine)
        unique &= bool((P[v - 1] == needles.keys[i]).all()
                       and (P[v] == needles.values[i]).all()
                       and (P[v + 1] == needles.nl).all())
        unique &= all(int((P == k).sum()) == 1 for k in mine[:len(v)])
        unique &= len(set(mine[len(v):])) == len(v)
    oracle = bool(needle_oracle(needles)['correct'].all())
    return dict(oracle=oracle, single=single, absent=absent, unique=unique)

def gap_bins(L):
    """The octave bins of gap (32, 64], (64, 128], ..., (L / 2, L]."""
    return [(32 * 2 ** k, 64 * 2 ** k) for k in range(int(math.log2(L //
                                                                      32)))]

def recall_by_gap(rec, L=None):
    """Recall per octave bin of gap (gap_bins(L); L defaults to the power
    of two above the largest gap): DataFrame(gap_lo, gap_hi, n, k, rate,
    lo, hi) with 95% Wilson bounds. rec has rows (gap, correct); correct
    may be a mean over seeds of the same query. The Wilson bounds treat
    the queries as independent: they do not account for the eight queries
    of a window sharing its text, nor for windows of one book overlapping
    (the differences below resample whole windows)."""
    L = L or 2 ** math.ceil(math.log2(max(64, int(rec['gap'].max()))))
    rows = []
    for a, b in gap_bins(L):
        x = rec[(rec['gap'] > a) & (rec['gap'] <= b)]['correct']
        k, n = float(x.sum()), len(x)
        rows.append((a, b, n, k, k / n if n else math.nan,
                     *d2l.wilson_ci(k, n)))
    return pd.DataFrame(rows, columns=['gap_lo', 'gap_hi', 'n', 'k', 'rate',
                                       'lo', 'hi'])

def recall_between(rec, lo, hi):
    """The share of the queries with a gap in (lo, hi] that are correct."""
    return float(rec[(rec['gap'] > lo) & (rec['gap'] <= hi)][
        'correct'].mean())

def _per_window(rec):
    """(correct, count) per window of query rows (window, correct)."""
    per = rec.groupby('window')['correct'].agg(['sum', 'count'])
    return (per['sum'].to_numpy(dtype=float),
            per['count'].to_numpy(dtype=float))

def recall_paired(a, b, on=('window', 'needle')):
    """Recall of the queries a minus the same queries b (rows matched on
    `on`, which starts with the window; correct may be a mean over seeds):
    (difference, lo, hi), the 95% interval from resampling whole windows."""
    m = a.merge(b, on=list(on), suffixes=('_a', '_b'))
    if not len(m):
        return (math.nan,) * 3
    d = m.assign(correct=m['correct_a'].astype(float)
                 - m['correct_b'].astype(float))
    return d2l.window_ci(*_per_window(d))

def recall_unpaired(a, b, n_boot=2000, seed=0):
    """Recall of the queries a minus that of other queries b (rows window,
    correct): (difference, lo, hi), the 95% interval from resampling the
    windows of a and of b independently."""
    if not len(a) or not len(b):
        return (math.nan,) * 3
    rng = np.random.default_rng(seed)
    point, boots = [], []
    for x in (a, b):
        s, c = _per_window(x)
        idx = rng.integers(len(s), size=(n_boot, len(s)))
        point.append(s.sum() / c.sum())
        boots.append(s[idx].sum(1) / c[idx].sum(1))
    lo, hi = np.quantile(boots[0] - boots[1], [0.025, 0.975])
    return float(point[0] - point[1]), float(lo), float(hi)
```

## The Format Comparison

The format table compares, on the same book windows of 1,024 tokens with
eight lines each (the queried line at mid depth of the text, the others at
random token positions), the key-value template of the first build
(`The value of <adjective> <noun> is <five digits>.`, queried by
`The value of <key> is`, candidates the eight values of several tokens
each) with the one-hop line `<key> <value>` of `needle_pools` words,
queried by `\n<key>`. Both are scored by summed log-probability of each
candidate with one prefill of the window (`_choice_ll`, the cache repeated
for the candidates); the oracle copies the tokens after the last earlier
occurrence of the query. Item i of the two formats is the same window: the
book, the offset, the lines' relative positions and the candidate order are
drawn from one generator for both formats, and only the lines from a
second, per-format one (the template's longer lines leave a shorter prefix
of the same text). The table's difference is therefore paired by item.

```{.python .input #long-context-lib-format}
%%tab pytorch
#@save
_RECALL_ADJ = (
    'amber', 'ancient', 'autumn', 'bitter', 'bold', 'brave', 'bright',
    'broken', 'calm', 'clever', 'cold', 'copper', 'crimson', 'dark', 'dry',
    'dusty', 'eager', 'early', 'empty', 'frozen', 'gentle', 'golden',
    'green', 'hidden', 'hollow', 'honest', 'little', 'lonely', 'lucky',
    'mighty', 'misty', 'narrow', 'noble', 'quiet', 'rapid', 'rough', 'round',
    'rusty', 'sharp', 'silent', 'silver', 'simple', 'small', 'smooth',
    'solid', 'steady', 'sweet', 'wild')

#@save
_RECALL_NOUN = (
    'anchor', 'apple', 'arrow', 'basket', 'bell', 'bridge', 'bucket',
    'candle', 'castle', 'cloud', 'coin', 'desert', 'door', 'eagle', 'engine',
    'feather', 'field', 'forest', 'garden', 'harbor', 'hammer', 'island',
    'jacket', 'kettle', 'ladder', 'lake', 'lantern', 'letter', 'meadow',
    'mirror', 'needle', 'orchard', 'pencil', 'pillow', 'planet', 'river',
    'rocket', 'saddle', 'shadow', 'shell', 'signal', 'spoon', 'stone',
    'temple', 'tower', 'valley', 'wagon', 'window')

#@save
def _format_lines(tokenizer, fmt, n_lines, words, text, generator):
    """(lines, query, candidates) of one format item as token lists."""
    nl = _encode(tokenizer, '\n')
    if fmt == 'template':
        adj, noun = _RECALL_ADJ, _RECALL_NOUN
        pick = torch.randperm(len(adj) * len(noun), generator=generator)[
            :n_lines].tolist()
        keys = [f'{adj[i // len(noun)]} {noun[i % len(noun)]}' for i in pick]
        values = [str(v + 10_000) for v in torch.randperm(
            90_000, generator=generator)[:n_lines].tolist()]
        head = _encode(tokenizer, f'The value of {keys[0]} is')
        conts = []
        for v in values:
            ids = _encode(tokenizer, f'The value of {keys[0]} is {v}')
            assert ids[:len(head)] == head, 'the query tokenizes apart'
            conts.append(ids[len(head):])
        lines = [nl + _encode(tokenizer, f'The value of {k} is {v}.') + nl
                 for k, v in zip(keys, values)]
        return lines, nl + head, conts
    if fmt != 'one-hop':
        raise ValueError(f"fmt is 'template' or 'one-hop', not {fmt!r}")
    key_pool, value_pool = needle_pools(tokenizer)
    keys = _needle_pick(key_pool, words, text, n_lines, generator)
    values = _needle_pick(value_pool, words, text, n_lines, generator)
    lines = [nl + [k, v] + nl for (_, k), (_, v) in zip(keys, values)]
    return lines, nl + [keys[0][1]], [[v] for _, v in values]

def _format_items(tokenizer, fmt, L, n, seed=0, n_lines=8, depth=0.5):
    """n items dict(ctx, query, conts, gold) of format fmt in windows of
    about L tokens: book text with n_lines lines inserted at token
    positions, the queried line (the first) at fraction depth of the text,
    the others at random, then the query; the candidates in a random
    order. The window (book, offset), the lines' relative positions in
    its text and the candidate order come from a generator that does not
    depend on fmt, so item i of each format is the same book window (the
    longer template lines leave a shorter prefix of the same text); only
    the lines themselves are drawn per format."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + L * 1009)
    gf = torch.Generator().manual_seed(seed * 1_000_003 + L * 1009 + 1
                                       + (fmt == 'template'))
    books = [torch.as_tensor(b) for b in _test_book_tokens(tokenizer)
             if len(b) > L]
    items = []
    for i in range(n):
        book = books[i % len(books)]
        a = int(torch.randint(len(book) - L, (1,), generator=g))
        u = torch.rand(n_lines - 1, generator=g, dtype=torch.float64)
        order = torch.randperm(n_lines, generator=g).tolist()
        words = _text_words(tokenizer, book[a:a + L])
        lines, query, conts = _format_lines(tokenizer, fmt, n_lines, words,
                                            book[a:a + L], gf)
        budget = L - sum(map(len, lines)) - len(query) - max(map(len, conts))
        text = book[a:a + budget].tolist()
        cuts = [round(depth * budget)] + (u * (budget + 1)).floor().long(
            ).clamp(max=budget).tolist()
        ids, prev = [], 0
        for j in sorted(range(n_lines), key=lambda j: (cuts[j], j)):
            ids += text[prev:cuts[j]] + lines[j]
            prev = cuts[j]
        items.append(dict(ctx=ids + text[prev:] + query, query=query,
                          conts=[conts[j] for j in order],
                          gold=order.index(0)))
    return items

def _format_oracle(item):
    """The candidate that continues the last earlier occurrence of the
    query in the context (-1 if none does)."""
    ctx, q = item['ctx'], item['query']
    for p in range(len(ctx) - 2 * len(q), -1, -1):
        if ctx[p:p + len(q)] == q:
            after = ctx[p + len(q):]
            for j, c in enumerate(item['conts']):
                if after[:len(c)] == c:
                    return j
            return -1
    return -1

def _choice_ll(model, ctx, conts, chunk=2048):
    """Summed log-probabilities of each continuation (token lists) after
    the context ids ctx: one cached prefill, the cache repeated for the
    candidates, one cached forward of every candidate's tokens."""
    dev = _device_of(model)
    m = max(map(len, conts))
    auto = torch.autocast(dev.type, torch.bfloat16,
                          enabled=dev.type == 'cuda')
    with torch.inference_mode(), auto:
        ks, vs = model.init_cache(1, len(ctx) - 1 + m)
        for t in range(0, len(ctx) - 1, chunk):
            model.forward_cached(torch.tensor(
                [ctx[t:min(t + chunk, len(ctx) - 1)]], device=dev), ks, vs, t)
        X = torch.tensor([[ctx[-1]] + c[:-1] + [0] * (m - len(c))
                          for c in conts], device=dev)
        C = len(conts)
        logits, _, _ = model.forward_cached(
            X, ks.repeat(1, C, 1, 1, 1), vs.repeat(1, C, 1, 1, 1),
            len(ctx) - 1, all_logits=True)
    lp = logits.float().log_softmax(-1)
    return [float(lp[j, torch.arange(len(c)), torch.tensor(c)].sum())
            for j, c in enumerate(conts)]
```

## Losses of Windows, Gaps and Intervals

Per-target losses of whole windows (`window_nlls`) and of a sliding short
context (`sliding_nlls`), the gap back to each token's previous
occurrence (`repeat_gaps`), the set of rare tokens the long-context gain
is measured on (`_rare_ids`: beyond rank 2,000 in the unigram counts of
the first 2^25 tokens of the 100M shard, decoding to at least three
letters), and intervals that resample whole windows (`window_ci`). The
loss against the context given scores the same targets (the second half
of each window) with a sliding context of c tokens (each target sees c / 2
to c tokens back), computed on the part of the window those targets need.

```{.python .input #long-context-lib-losses}
%%tab pytorch
#@save
def _device_of(model):
    import itertools
    for t in itertools.chain(model.parameters(), model.buffers()):
        return t.device
    return torch.device('cpu')

def window_nlls(model, windows, ctx=None, eos=None, batch=1, chunk=4096):
    """Per-target loss (n, T) in fp32 of windows (n, T + 1) under bf16
    autocast on a GPU: a PackedGPT through its hidden states (chunked), any
    other model through its logits. ctx: a PackCtx of the inputs; eos: the
    EOS id that ends documents (masks built from it, NaN after an EOS)."""
    W = torch.as_tensor(windows).long()
    dev, out = _device_of(model), []
    was_training = model.training
    model.eval()
    for i in range(0, len(W), batch):
        Wb = W[i:i + batch].to(dev)
        X, Y = Wb[:, :-1], Wb[:, 1:]
        c = None if ctx is None else d2l.PackCtx(
            ctx.doc_ids[i:i + batch].to(dev), ctx.pos_ids[i:i + batch].to(dev))
        if eos is not None:
            c = d2l.PackCtx(*d2l.doc_ids_from_eos(X.int(), eos))
        with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16,
                                             enabled=dev.type == 'cuda'):
            if isinstance(model, d2l.PackedGPT):
                H, Wt = model(X, c, return_hidden=True)
                nll = d2l._token_nll(H, Wt, Y, chunk)
            else:
                nll = F.cross_entropy(model(X).float().flatten(0, 1),
                                      Y.flatten(), reduction='none')
        nll = nll.view(Y.shape).float()
        if eos is not None:
            nll = nll.masked_fill(X == eos, float('nan'))
        out.append(nll.cpu())
    model.train(was_training)
    return torch.cat(out)

def sliding_nlls(model, windows, short_len):
    """Per-target loss (n, T) when every target sees only its last
    short_len / 2 to short_len tokens: windows of short_len tokens that
    advance by short_len / 2 (the first window scores all its targets)."""
    W = torch.as_tensor(windows).long()
    n, T, h = W.shape[0], W.shape[1] - 1, short_len // 2
    if T % h or T < short_len:
        raise ValueError(f'T = {T} is a multiple of short_len / 2 = {h}, '
                         'at least short_len')
    starts = list(range(0, T - short_len + 1, h))
    segs = torch.stack([W[:, a:a + short_len + 1] for a in starts], 1)
    nll = window_nlls(model, segs.flatten(0, 1),
                      batch=max(1, 16_384 // short_len)).view(
        n, len(starts), short_len)
    out = torch.empty(n, T)
    out[:, :short_len] = nll[:, 0]
    for k, a in enumerate(starts[1:], 1):
        out[:, a + h:a + short_len] = nll[:, k, h:]
    return out

def context_nlls(model, windows, c, full=None):
    """Per-target loss (n, T / 2) of the second half of windows (n, T + 1)
    when each target is given c / 2 to c tokens of context (sliding_nlls on
    the c tokens before the second half and the half itself); c = T: the
    whole window (full, if given, is window_nlls of the windows)."""
    W = torch.as_tensor(windows).long()
    T = W.shape[1] - 1
    h = T // 2
    if c >= T:
        return (window_nlls(model, W) if full is None else full)[:, h:]
    return sliding_nlls(model, W[:, h - c:], c)[:, c:]

def repeat_gaps(windows, rare_ids):
    """Per target (n, T): the distance back to the previous occurrence of
    the target token in its window when that token is in rare_ids, else 0
    (no earlier occurrence, or not rare)."""
    W = torch.as_tensor(windows).long()
    n, T1 = W.shape
    pos = torch.arange(T1).expand(n, T1)
    order = (W * T1 + pos).argsort(-1)  # by token, then by position
    tok = W.gather(1, order)
    same = tok[:, 1:] == tok[:, :-1]
    prev = torch.full_like(W, -1)
    prev.scatter_(1, order[:, 1:], torch.where(same, order[:, :-1], -1))
    gap = torch.where(prev >= 0, pos - prev, 0)
    rare = torch.isin(W, torch.as_tensor(rare_ids).long())
    return (gap * rare)[:, 1:]

def gain_sums(gain, gap, pairs, min_gap):
    """Per window and gap bucket (lo, hi]: summed gain and count of the
    targets with a gap in the bucket and at least min_gap."""
    sums, counts = [], []
    for lo, hi in pairs:
        hit = (gap > lo) & (gap <= hi) & (gap >= min_gap)
        sums.append(torch.where(hit, gain, 0.0).sum(1))
        counts.append(hit.sum(1))
    return torch.stack(sums, 1).double(), torch.stack(counts, 1).double()

def window_ci(sums, counts, n_boot=2000, seed=0):
    """(mean, lo, hi): the ratio sum(sums) / sum(counts) over windows and
    its 95% percentile interval from resampling whole windows."""
    s = np.asarray(sums, dtype=float)
    c = np.asarray(counts, dtype=float)
    if c.sum() == 0:
        return float('nan'), float('nan'), float('nan')
    idx = np.random.default_rng(seed).integers(len(s), size=(n_boot, len(s)))
    with np.errstate(invalid='ignore', divide='ignore'):
        means = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.nanquantile(means, [0.025, 0.975])
    return float(s.sum() / c.sum()), float(lo), float(hi)

def bucket_name(lo, hi):
    """'0-4K', '1.5-3K' or '256-512' for the positions (lo, hi]."""
    if hi >= 1024 and lo % 512 == 0 and hi % 512 == 0:
        return f'{lo / 1024:g}-{hi / 1024:g}K'
    return f'{lo}-{hi}'

def _length_name(L):
    """'16K', '1.5K' or '256' for a length in tokens."""
    return f'{L / 1024:g}K' if L >= 1024 and L % 512 == 0 else str(L)

def _rare_ids(rank=2000, min_letters=3, tokens=2 ** 25):
    """Token ids beyond rank in the unigram counts of the first tokens of
    the 100M shard (and seen there) that decode to at least min_letters
    letters (an optional leading space), cached."""
    key = (rank, min_letters, tokens)
    if getattr(_rare_ids, 'key', None) != key:
        with d2l._quiet():
            s = d2l.fwedu_shards('100m')
            tok = d2l.r2_tokenizer()
        counts = np.bincount(s.read(0, min(tokens, s.n_tokens)).astype(
            np.int64), minlength=len(tok))
        order = np.argsort(-counts, kind='stable')[rank:]
        word = re.compile(rf' ?[A-Za-z]{{{min_letters},}}')
        _rare_ids.ids = torch.tensor(sorted(
            int(i) for i in order if counts[i] > 0
            and word.fullmatch(tok.decode([int(i)]))))
        _rare_ids.key = key
    return _rare_ids.ids
```

## Two Models for the Gain Check

The page checks `long_context_gain` on a synthetic repeat with two models
that need no training: an induction head as a model (it predicts the token
that followed the previous occurrence of the current token) and a model
that ignores its context altogether.

```{.python .input #long-context-lib-oracles}
%%tab pytorch
#@save
class CopyOracle(nn.Module):
    """Logits that put `confidence` on the token that followed the
    previous occurrence of the current token (uniform where there is
    none): an induction head as a whole model."""
    def __init__(self, vocab_size, confidence=30.0):
        super().__init__()
        self.vocab_size, self.confidence = vocab_size, confidence

    def forward(self, X):
        n, T = X.shape
        idx = torch.arange(T, device=X.device)
        earlier = (X[:, :, None] == X[:, None, :]) & (idx[None, :] <
                                                     idx[:, None])
        last = torch.where(earlier, idx, -1).amax(-1)
        nxt = X.gather(1, last + 1)  # last + 1 <= t: no look-ahead
        logits = torch.zeros(n, T, self.vocab_size, device=X.device)
        hit = torch.where(last >= 0, self.confidence, 0.0)
        return logits.scatter_(2, nxt[..., None], hit[..., None])

#@save
class UnigramModel(nn.Module):
    """Uniform logits whatever the context."""
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, X):
        return torch.zeros(*X.shape, self.vocab_size, device=X.device)
```

## The FLOP Check and Transformers' YaRN

```{.python .input #long-context-lib-rule}
%%tab pytorch
#@save
def flop_counter_per_token(arch, T):
    """Training FLOPs per token that FlopCounterMode counts in one forward
    and backward pass of a PackedGPT of arch on one window of T tokens, on
    the meta device (nothing is allocated or computed)."""
    from torch.utils.flop_counter import FlopCounterMode
    with torch.device('meta'):
        model = d2l.PackedGPT.from_arch(arch, max_len=T, backend='sdpa')
        X = torch.zeros(1, T, dtype=torch.long)
    with FlopCounterMode(display=False) as counter:
        H, W = model(X, return_hidden=True)
        F.linear(H, W).float().sum().backward()
    return counter.get_total_flops() / T

def transformers_yarn(d_head, base=10000.0, s=4.0, L=4096, truncate=False):
    """(inverse frequencies, attention factor) of Transformers' YaRN
    initialization for a head width d_head trained at L, scaled by s."""
    import transformers
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    cfg = transformers.LlamaConfig(
        hidden_size=d_head, num_attention_heads=1, head_dim=d_head,
        max_position_embeddings=round(s * L), rope_parameters=dict(
            rope_type='yarn', rope_theta=base, factor=float(s),
            original_max_position_embeddings=L, truncate=truncate,
            beta_fast=32.0, beta_slow=1.0))
    inv, factor = ROPE_INIT_FUNCTIONS['yarn'](cfg, 'cpu')
    return inv.float(), float(factor)
```

## One Model's Evaluation

`_lc_eval` evaluates one model: on loss windows of T + 1 tokens, the loss
per position bucket of the trained length, the loss of the second half
against the context given (each tier's contexts and the whole window,
with the bytes of those targets for bits per byte), and the long-context
gain against a sliding context of 256 tokens in the octave bins of gap
from 256 (the needle's axis); the needle probe at each length; on request
the val loss at the trained length (per window, for paired differences)
and the suite. It returns per-window, per-query and per-item rows; the
summaries and intervals are computed from them, pooling seeds by window
or query.

```{.python .input #long-context-lib-eval}
%%tab pytorch
#@save
LcZeroShot = collections.namedtuple(
    'LcZeroShot', ['table', 'windows', 'needles', 'label'])

#@save
LcReference = collections.namedtuple(
    'LcReference', ['windows', 'needles', 'label'])

#@save
LcEval = collections.namedtuple('LcEval', ['windows', 'needles', 'items',
                                           'label'])

#@save
_LC_WINDOW_COLUMNS = ['measure', 'bucket', 'window', 'sum', 'n']

#@save
def _lc_gain_bins(T):
    """The gain's gap buckets: the octave bins of gap from 256 to T."""
    return [b for b in gap_bins(T) if b[0] >= _LC_SHORT]

def _lc_contexts(tier, T):
    """The contexts of the loss-against-context curve below T, then T."""
    return [c for c in _LC_TIERS[tier]['contexts'] if c < T] + [T]

def _lc_eval(model, tokenizer, tier, T, lengths, n_probe, n_windows, *,
             gain=True, contexts=True, val=False, suite=None):
    """Rows of one model (n_probe needle windows per length, n_windows
    loss windows of T + 1 tokens): windows (measure, bucket, window, sum, n) for
    'nll' (position buckets), 'ctx' (the second half per context; 'bytes'
    its bytes), 'gain' and 'val'; needles (length, window, needle, gap,
    correct); items (task, item_id, score) of the suite."""
    L0, rows = _LC_TIERS[tier]['trained'], []
    if n_windows:
        books = test_books(n_windows, T, tokenizer=tokenizer)
        nll = window_nlls(model, books)
        for lo in range(0, T, L0):
            rows += [('nll', bucket_name(lo, lo + L0), w,
                      float(nll[w, lo:lo + L0].sum()), L0)
                     for w in range(len(books))]
        short = sliding_nlls(model, books, _LC_SHORT) if gain else None
        if contexts:
            for c in _lc_contexts(tier, T):
                part = (short[:, T // 2:] if c == _LC_SHORT and gain else
                        context_nlls(model, books, c, nll))
                name = 'full' if c == T else _length_name(c)
                rows += [('ctx', name, w, float(part[w].sum()), T - T // 2)
                         for w in range(len(books))]
            rows += [('bytes', 'targets', w, float(len(tokenizer.decode(
                books[w, T // 2 + 1:].tolist()).encode('utf-8'))), T - T // 2)
                for w in range(len(books))]
        if gain:
            pairs = _lc_gain_bins(T)
            s, c = gain_sums(short - nll, repeat_gaps(books, _rare_ids()),
                             pairs, _LC_SHORT)
            for k, (lo, hi) in enumerate(pairs):
                rows += [('gain', bucket_name(lo, hi), w, float(s[w, k]),
                          float(c[w, k])) for w in range(len(books))]
    if val:
        W, eos = _val_windows(L0)
        v = window_nlls(model, W, eos=eos, batch=max(1, 8192 // L0))
        for w in range(len(W)):
            ok = ~torch.isnan(v[w])
            rows.append(('val', 'val', w, float(v[w][ok].sum()),
                         float(ok.sum())))
    queries = [needle_recall(model, needle_windows(tokenizer, L, n_probe))
               .assign(length=L) for L in lengths]
    items = (pd.DataFrame(columns=['task', 'item_id', 'score'])
             if suite is None else d2l._suite_items(suite, model, tokenizer))
    return (pd.DataFrame(rows, columns=_LC_WINDOW_COLUMNS),
            pd.concat(queries, ignore_index=True)[
                ['length'] + _LC_QUERY_COLUMNS], items)

def _lc_summary(windows, measure):
    """(model, bucket, value, lo, hi, n) of one measure: per model and
    bucket the ratio over windows (seeds pooled per window) with a
    window-resampled 95% interval."""
    w = windows[windows['measure'] == measure]
    g = w.groupby(['model', 'bucket', 'window'], sort=False)[
        ['sum', 'n']].sum().reset_index()
    rows = []
    for (model, bucket), x in g.groupby(['model', 'bucket'], sort=False):
        mean, lo, hi = window_ci(x['sum'], x['n'])
        rows.append((model, bucket, mean, lo, hi, int(x['n'].sum())))
    return pd.DataFrame(rows, columns=['model', 'bucket', 'value', 'lo',
                                       'hi', 'n'])

def _lc_position(windows):
    p = _lc_summary(windows, 'nll')
    return p.rename(columns={'value': 'nll'})[['model', 'bucket', 'nll',
                                               'lo', 'hi']]

def _lc_gain(windows):
    g = _lc_summary(windows, 'gain')
    return g.rename(columns={'value': 'gain'})[['model', 'bucket', 'gain',
                                                'lo', 'hi', 'n']]

def _lc_ctx_delta(windows, model, a, b='full'):
    """The loss of model's second-half targets with context b minus with
    context a (nats per token; window-resampled 95% interval, seeds
    averaged per window) and the same in bits per byte."""
    w = windows[(windows['model'] == model) & (windows['measure'].isin(
        ['ctx', 'bytes']))]
    per = w.groupby(['measure', 'bucket', 'window'])[['sum', 'n']].mean()
    if ('ctx', a, 0) not in per.index or ('ctx', b, 0) not in per.index:
        return (math.nan,) * 6
    d = (per.loc[('ctx', b)]['sum'] - per.loc[('ctx', a)]['sum']).to_numpy()
    n = per.loc[('ctx', b)]['n'].to_numpy()
    nb = per.loc[('bytes', 'targets')]['sum'].to_numpy() * math.log(2)
    return (*window_ci(d, n), *window_ci(d, nb))

def _lc_context(windows, models):
    """Per model and context: the loss of the second half minus its loss
    with the first context, nats per token and bits per byte, with
    window-resampled 95% intervals."""
    rows = []
    for m in models:
        names = list(dict.fromkeys(windows[(windows['model'] == m) & (
            windows['measure'] == 'ctx')]['bucket']))
        for c in names:
            rows.append((m, c, *_lc_ctx_delta(windows, m, names[0], c)))
    return pd.DataFrame(rows, columns=d2l._LC_COLUMNS['context'])

def _lc_recall(needles):
    """Per model, seed and length the recall by octave bin of gap; seed -1:
    the seeds pooled (each query once, its correctness averaged over the
    seeds, which share the windows)."""
    out = []
    for (m, L), x in needles.groupby(['model', 'length'], sort=False):
        for s, y in x.groupby('seed'):
            out.append(recall_by_gap(y, L).assign(model=m, seed=s, length=L))
        pooled = x.groupby(['window', 'needle', 'gap'])['correct'].mean()
        out.append(recall_by_gap(pooled.reset_index(), L).assign(
            model=m, seed=-1, length=L))
    if not out:
        return pd.DataFrame(columns=d2l._LC_COLUMNS['recall'])
    return pd.concat(out, ignore_index=True)[d2l._LC_COLUMNS['recall']]

def _lc_range(needles, model, L, lo, hi):
    """Recall of model at length L over the gaps (lo, hi], seeds averaged
    per query, with a window-resampled 95% interval: (rate, lo, hi)."""
    x = needles[(needles['model'] == model) & (needles['length'] == L)
                & (needles['gap'] > lo) & (needles['gap'] <= hi)]
    if not len(x):
        return (math.nan,) * 3
    q = x.groupby(['window', 'needle'])['correct'].mean().reset_index()
    per = q.groupby('window')['correct'].agg(['sum', 'count'])
    return window_ci(per['sum'], per['count'])

def _lc_set_scheme(model, scheme, s, trained, mscale=None):
    """Scaled RoPE on model: frequencies of scheme at scale s (base RoPE
    for 'none'), YaRN's attention scale for 'yarn' unless mscale is
    given; the model then accepts s times its trained length."""
    if mscale is None:
        mscale = d2l.yarn_mscale(s) if scheme == 'yarn' else 1.0
    inv = None if scheme == 'none' else d2l.rope_inv_freq(
        model.blks[0].attention.head_dim, model.rope_theta, scheme, s,
        trained)
    model.set_rope(inv, mscale, scheme, s)
    return model
```

## Zero-Shot and the Reference Model

`lc_zeroshot` evaluates the base model under each scheme without training
(250 needle windows at each length, 64 loss windows at the longest). The
regeneration `lc-zeroshot` writes its result as a part of
`lm-lc-evals-v2`, which the T2 merge folds in; a normal run reads it from
the committed evaluations (checking the page's configuration against the
stored one), and while those are pending shows the T1 zero-shot table that
`lc-t1` computes on the t1 checkpoint. `lc_reference` does the same for
Qwen3-0.6B-Base (loaded with `load_qwen3` and wrapped by `PackedGPT.adopt`;
its own tokenizer, pools and offsets; no gain, which needs the reference
tokenizer's rare tokens).

```{.python .input #long-context-lib-zeroshot}
%%tab pytorch
#@save
_LC_BASES = {'r2': ('lm-r2-s6000-v1', 'r2-s0', 6000),
             't1': ('lm-r2t1-s1526-v1', 't1-s0', 1526)}

#@save
def _lc_base_path(base):
    """The base checkpoint: committed, else the run's job folder."""
    key, job, step = _LC_BASES[base]
    if key in d2l.DATA_HUB:
        with d2l._quiet():
            return d2l.download_keyed(key)
    path = os.path.join(d2l._job_root(), job, 'ckpt', f's{step}.safetensors')
    if os.path.exists(path):
        return path
    raise KeyError(f'{key} is not committed yet')

def _lc_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _lc_tier(base):
    if base not in _LC_BASES:
        raise ValueError(f"base is 'r2' or 't1', not {base!r}")
    return 't2' if base == 'r2' else 't1'

def _lc_bucket_delta(windows, model, a, b):
    """The loss of model in position bucket b minus in bucket a on the same
    windows (nats per token; seeds averaged per window), with a 95%
    interval from resampling whole windows: (delta, lo, hi)."""
    w = windows[(windows['model'] == model) & (windows['measure'] == 'nll')
                & windows['bucket'].isin([a, b])]
    per = w.groupby(['bucket', 'window'])[['sum', 'n']].mean()
    have = set(per.index.get_level_values(0))
    if a not in have or b not in have:
        return (math.nan,) * 3
    d = per.loc[b]['sum'] - per.loc[a]['sum'].reindex(per.loc[b].index)
    return window_ci(d.to_numpy(), per.loc[b]['n'].to_numpy())

def _lc_zs_table(tier, windows, needles, names):
    """Per scheme at the longest length, then 'none' at the trained length
    (the control): the losses of the first and last position buckets and
    last minus first on the same windows, recall in the gap bin (128, 256]
    and in the top octave (L / 2, L], with intervals, and the reach."""
    L0, T = _LC_TIERS[tier]['trained'], int(needles['length'].max())
    pos, rec = _lc_position(windows), _lc_recall(needles)
    first, last = bucket_name(0, L0), bucket_name(T - L0, T)
    rows = []
    for name, L in [(n, T) for n in names] + [('none', L0)]:
        p = pos[pos['model'] == name].set_index('bucket')
        b = rec[(rec['model'] == name) & (rec['seed'] == 0)
                & (rec['length'] == L)].set_index('gap_lo')
        cell = lambda k: tuple(p.loc[k, ['nll', 'lo', 'hi']]) if (
            k in p.index and L == T) else (math.nan,) * 3
        delta = (_lc_bucket_delta(windows, name, first, last) if L == T
                 else (math.nan,) * 3)
        rows.append((name, L, *(cell(first) if L == T else
                                tuple(p.loc[first, ['nll', 'lo', 'hi']])),
                     *cell(last), *delta, *b.loc[128, ['rate', 'lo', 'hi']],
                     *b.loc[L // 2, ['rate', 'lo', 'hi']],
                     d2l.reach(b.reset_index())))
    return pd.DataFrame(rows, columns=d2l._LC_COLUMNS['zs_table'])

def _lc_zeroshot_compute(base, schemes, lengths):
    """LcZeroShot of base under each scheme (no training)."""
    tier = _lc_tier(base)
    spec, dev, tok = _LC_TIERS[tier], _lc_device(), lc_tokenizer()
    windows, needles = [], []
    for name, sc in schemes.items():
        model = d2l._load_bf16(_lc_base_path(base), device=dev)
        _lc_set_scheme(model, sc['scheme'], sc['s'], spec['trained'],
                       sc.get('mscale'))
        w, r, _ = _lc_eval(model, tok, tier, max(lengths), lengths,
                           spec['zs_needles'], spec['loss_windows'])
        windows.append(w.assign(model=name, seed=0))
        needles.append(r.assign(model=name, seed=0))
        del model
        d2l.lm_free()
    C = _LC_COLUMNS
    windows = pd.concat(windows, ignore_index=True)[C['windows']]
    needles = pd.concat(needles, ignore_index=True)[C['needles']]
    return LcZeroShot(_lc_zs_table(tier, windows, needles, list(schemes)),
                      windows, needles, '')

def _lc_part_path(part):
    """The file a part of lm-lc-evals-v2 (lc-zeroshot, lc-reference) is
    written to by its regeneration."""
    key = d2l.PRETRAIN_ARTIFACTS['lc-evals'][0]
    return os.path.join(d2l._rerun_dir(), f'{d2l._next_key(key)}.{part}'
                        '.json.gz')

def _lc_zs_as_tuple(res):
    """The zero-shot fields of a stored, pending or fallback result."""
    if isinstance(res, pd.DataFrame):  # pending: empty tables
        return d2l._as_tuple(LcZeroShot, res, {
            'table': _LC_COLUMNS['zs_table'],
            'windows': _LC_COLUMNS['windows'],
            'needles': _LC_COLUMNS['needles']})
    return LcZeroShot(res.zs_table, res.zs_windows, res.zs_needles,
                      res.label)

def _lc_part_result(part, compute, config, committed):
    """A part of lm-lc-evals-v2: computed and written when D2L_RERUN names
    it; else committed(result) of the committed lm-lc-evals-v2 after the
    configuration check (its field part's config); else pending."""
    rerun = d2l._regenerating()
    if part in rerun:
        res = compute()
        meta = dict(config=d2l._jsonable(config), label=d2l._label_of(res))
        d2l.lm_save_rerun(os.path.basename(_lc_part_path(part)),
                          lambda p: d2l._write_result(p, 'json-table', res,
                                                      meta),
                          rerun_dir=d2l._rerun_dir())
        return res
    key = d2l.PRETRAIN_ARTIFACTS['lc-evals'][0]
    if key in d2l.DATA_HUB:
        res = d2l.load_artifact('lc-evals')
        stored = getattr(res, {'lc-zeroshot': 'zs_config',
                               'lc-reference': 'ref_config'}[part])
        d = d2l._config_diff(d2l._jsonable(config), d2l._jsonable(stored))
        if d:
            raise ValueError(f'page configuration differs from the committed '
                             f'run {key}: {part}.{d[0]} {d[1]!r} != {d[2]!r}')
        return committed(res)
    return None

def lc_zeroshot(base, schemes, lengths):
    """The base model ('r2' or 't1') under each RoPE scheme without
    training: LcZeroShot(table, windows, needles, label). schemes =
    {name: dict(scheme, s[, mscale])}; table: per scheme at the longest
    length (and 'none' at the trained length) the losses of the first and
    last position buckets on 64 test windows, recall in the gap bins
    (128, 256] and (L / 2, L] (250 needle windows per length) and the
    reach. Regeneration lc-zeroshot; otherwise the zero-shot part of the
    committed lm-lc-evals-v2; while that is pending, the T1 zero-shot
    table, labelled so (show_lc_zeroshot prints the label)."""
    config = dict(base=base, schemes=schemes, lengths=lengths)
    res = _lc_part_result('lc-zeroshot', lambda: _lc_zeroshot_compute(
        base, schemes, lengths), config, _lc_zs_as_tuple)
    if res is not None:
        return res
    if d2l._regenerating() or 'lc-zeroshot' in d2l.PRETRAIN_PENDING:
        return _lc_zs_as_tuple(d2l._pending_result('lc-zeroshot', 'lc-t1'))
    raise KeyError('lc-zeroshot is not committed yet')

def _lc_load_reference(model_id):
    """(PackedGPT, tokenizer) of the pinned reference model on the GPU if
    there is one, bf16, quietly."""
    import transformers
    rev = d2l.HF_REVISIONS[model_id]
    with d2l._quiet():
        qwen = d2l.load_qwen3(model_id, rev, dtype=torch.bfloat16,
                              device=_lc_device())
        tok = transformers.AutoTokenizer.from_pretrained(model_id,
                                                         revision=rev)
    tok.model_max_length = 10 ** 7  # whole books are encoded once
    return d2l.PackedGPT.adopt(qwen).eval(), tok

def _lc_reference_compute(model_id, lengths):
    model, tok = _lc_load_reference(model_id)
    spec = _LC_TIERS['t2']
    w, r, _ = _lc_eval(model, tok, 't2', max(lengths), lengths,
                       spec['zs_needles'], spec['loss_windows'], gain=False)
    del model
    d2l.lm_free()
    name = model_id.split('/')[-1]
    return LcReference(w.assign(model=name, seed=0)[_LC_COLUMNS['windows']],
                       r.assign(model=name, seed=0)[_LC_COLUMNS['needles']],
                       '')

def lc_reference(model_id, lengths):
    """The reference model (Qwen3-0.6B-Base, trained to 32,768 tokens) on
    the needle probe at each length (250 windows) and the loss
    measurements at the longest (64 windows, its own tokenizer):
    LcReference(windows, needles, label). Regeneration lc-reference (a part
    of lm-lc-evals-v2 that the T2 merge folds in); otherwise the reference
    part of the committed lm-lc-evals-v2; pending while that is."""
    config = dict(model_id=model_id, revision=d2l.HF_REVISIONS[model_id],
                  lengths=lengths)
    res = _lc_part_result(
        'lc-reference', lambda: _lc_reference_compute(model_id, lengths),
        config, lambda r: LcReference(r.ref_windows, r.ref_needles, r.label))
    if res is not None:
        return res
    if d2l._regenerating() or 'lc-reference' in d2l.PRETRAIN_PENDING:
        return d2l._as_tuple(LcReference, d2l._pending_result(
            'lc-reference'), {'windows': _LC_COLUMNS['windows'],
                              'needles': _LC_COLUMNS['needles']})
    raise KeyError('lc-reference is not committed yet')

def _lc_part_for_merge(part, call, compute, as_tuple):
    """A part for the T2 merge: the file its regeneration wrote, or
    computed here when that regeneration was not run; and its config."""
    config = d2l._jsonable(dict(call, revision=d2l.HF_REVISIONS[
        call['model_id']]) if part == 'lc-reference' else call)
    path = _lc_part_path(part)
    if os.path.exists(path):
        res = d2l._read_result(path, 'json-table')
        d2l._check_config(res, config, os.path.basename(path))
        return as_tuple(res), config
    return compute(**call), config
```

## The Format Table and Reach during Pretraining

`lc_probe_runs` scores the format comparison (the oracle, the t1 and r2
final checkpoints and the reference model; 200 items of 1,024 tokens) and
the needle probe at the trained length of the committed checkpoints of
both runs (t1 at 1K, r2 at 4K; 500 windows each, the same windows for
every checkpoint of a run). Every checkpoint is read from its registered
key (`lm-r2t1-s<step>-v1`, `lm-r2-s<step>-v1`, sha1-checked on download),
never from a job folder or a local next version, so the regeneration
needs nothing that is not hosted; the keys and their sha1 are part of the
stored configuration, so a committed result made from other weights fails
the load check. The oracle copier must score 100% on each model's items
in both formats (Qwen3's are in its own tokens) before that model is
scored.

```{.python .input #long-context-lib-probe}
%%tab pytorch
#@save
LcProbe = collections.namedtuple(
    'LcProbe', ['format', 'reach_train', 'format_items', 'reach_queries',
                'label'])

#@save
_LC_PROBE_COLUMNS = {
    'format': ['model', 'format', 'n', 'k', 'rate', 'lo', 'hi'],
    'reach_train': ['model', 'step', 'tokens', 'gap_lo', 'gap_hi', 'n', 'k',
                    'rate', 'lo', 'hi'],
    'format_items': ['model', 'format', 'item', 'correct'],
    'reach_queries': ['model', 'step', 'tokens'] + _LC_QUERY_COLUMNS}

#@save
def _lc_probe_keys(t1_steps, r2_steps):
    """{key: sha1} of every checkpoint lc_probe_runs reads: the registered
    keys of the t1 and r2 runs, per run the format table's final checkpoint
    (_reference_key's default) first, then the steps; KeyError for a key
    not registered."""
    keys = [d2l._reference_key(p, s) for p, steps in (
        ('t1', (None, *t1_steps)), ('r2', (None, *r2_steps)))
            for s in steps]
    missing = [k for k in keys if k not in d2l.DATA_HUB]
    if missing:
        raise KeyError(f'not registered: {", ".join(missing)}')
    return {k: d2l.DATA_HUB[k][1] for k in dict.fromkeys(keys)}

def _lc_probe_model(preset, step, device):
    """The checkpoint of run preset at step (None: the final one) from its
    registered key (downloaded and sha1-checked on first use), in bf16 on
    device; never a
    job folder or a local next version (load_reference_model reads those
    during some regenerations)."""
    key = d2l._reference_key(preset, step)
    if key not in d2l.DATA_HUB:
        raise KeyError(f'{key} is not registered')
    with d2l._quiet():
        path = d2l.download_keyed(key)
    return d2l._load_bf16(path, torch.bfloat16, device)

def _lc_probe_compute(t1_steps, r2_steps):
    """LcProbe: the format table and reach during pretraining."""
    f, dev, tok = _LC_PROBE, _lc_device(), lc_tokenizer()
    _lc_probe_keys(t1_steps, r2_steps)  # every key registered, before work
    load = lambda preset: lambda: (_lc_probe_model(preset, None, dev), tok)
    models = [('oracle', lambda: (None, tok)), ('t1', load('t1')),
              ('r2', load('r2')), (_LC_REFERENCE.split('/')[-1],
                                   lambda: _lc_load_reference(_LC_REFERENCE))]
    rows = []
    for name, get in models:
        model, tk = get()
        for fmt in ('template', 'one-hop'):
            its = _format_items(tk, fmt, f['length'], f['items'])
            # the control, on every model's own items (Qwen3 has its own
            # tokenizer): each item is solvable by copying
            bad = sum(_format_oracle(it) != it['gold'] for it in its)
            assert not bad, (f'oracle copier misses {bad} {fmt} items '
                             f'({name})')
            for i, it in enumerate(its):
                pred = (_format_oracle(it) if model is None else int(
                    np.argmax(_choice_ll(model, it['ctx'], it['conts']))))
                rows.append((name, fmt, i, pred == it['gold']))
        del model
        d2l.lm_free()
    items = pd.DataFrame(rows, columns=_LC_PROBE_COLUMNS['format_items'])
    table = [(m, fmt, len(x), int(x.sum()), x.mean(),
              *d2l.wilson_ci(int(x.sum()), len(x)))
             for (m, fmt), x in items.groupby(['model', 'format'],
                                              sort=False)['correct']]
    queries, bins = [], []
    for preset, steps in (('t1', t1_steps), ('r2', r2_steps)):
        c = d2l.r2_config(preset)
        L = c.arch.T
        needles = needle_windows(tok, L, f['windows'])
        for step in steps:
            model = _lc_probe_model(preset, step, dev)
            q = needle_recall(model, needles).assign(
                model=preset, step=step, tokens=step * c.global_batch * L)
            queries.append(q)
            bins.append(recall_by_gap(q, L).assign(
                model=preset, step=step, tokens=step * c.global_batch * L))
            del model
            d2l.lm_free()
    P = _LC_PROBE_COLUMNS
    return LcProbe(pd.DataFrame(table, columns=P['format']),
                   pd.concat(bins, ignore_index=True)[P['reach_train']],
                   items, pd.concat(queries, ignore_index=True)[
                       P['reach_queries']], '')

def lc_probe_runs(t1_steps, r2_steps):
    """The format table (template against one-hop needle at 1K, 200 items,
    for the oracle copier, t1, r2 and Qwen3-0.6B-Base) and recall by gap
    at the trained length for the checkpoints of t1 (at 1K) and r2 (at 4K)
    at the steps given, 500 windows each: LcProbe(format, reach_train,
    format_items, reach_queries, label). Regeneration lc-probe (the
    checkpoints from their registered keys only); otherwise the committed
    lm-lc-probe-v1, whose stored configuration names the same keys and
    sha1; pending while that is."""
    config = dict(t1_steps=t1_steps, r2_steps=r2_steps, probe=_LC_PROBE,
                  reference=[_LC_REFERENCE, d2l.HF_REVISIONS[_LC_REFERENCE]],
                  checkpoints=_lc_probe_keys(t1_steps, r2_steps))
    res = d2l._pretrain_result('lc-probe', lambda: _lc_probe_compute(
        t1_steps, r2_steps), config)
    return d2l._as_tuple(LcProbe, res, _LC_PROBE_COLUMNS)
```

## The Arms

Each arm and seed trains in its own job folder under
`../data/lm/rerun/ch21/lc/` with `_PretrainRun` (the state, logging and
resume machinery of `d2l.pretrain`; full states every 50 steps, deleted
once the arm is done; the final weights in bf16 with the scheme, scale and
attention scale in their configuration). The step is `pretrain_step` on
one GPU (fp32 master weights, bf16 autocast, fresh AdamW moments); the
schedule is warmup, constant, and a linear decay to zero over the final
`decay_frac` of the steps. The data are the long half and the replay,
mixed by `mix_streams` in the shares `long_share` and `1 - long_share`
with the arm's seed (the same windows for every arm of a seed). A stage
with a shorter window cuts every window of the mixture into equal pieces
with their own document ids and positions (the same tokens per step and
one compiled shape), and the RoPE scale changes at the stage boundary,
which a resumed run derives from its step.

At T2, every 64 steps (and before the first update) a bf16 copy of the
model is evaluated: 50 needle windows at the arm's length, the loss by
position bucket and the gain on 16 test windows (the first 16 of the
final evaluation's 64). The evaluation draws no number from the training
generators and reads nothing from the stream (its windows have generators
of their own), so an arm trains the same with and without it; the flag
`evaluate` is therefore not part of an arm's identity, and a T2 arm of
seed 1 or 2 refuses to train if its configuration differs from seed 0's
(the first build's finished arm) in anything but the seed. Seed 0 and the
progressive arm were trained before this evaluation existed: their
folders hold the old timeline, which is not read. T1 arms train without
it. The final evaluation of each model (the base and every arm) is kept in
`evals.json.gz` in its job folder under the evaluation settings and the
weights' id, so that `lc-t2-evals --only <folder>` can evaluate the models
on separate cards and the merge reads them back.

```{.python .input #long-context-lib-arms}
%%tab pytorch
#@save
LcRuns = collections.namedtuple(
    'LcRuns', ['table', 'logs', 'position', 'context', 'recall', 'gain',
               'timeline', 'short', 'windows', 'needles', 'items',
               'zs_table', 'zs_windows', 'zs_needles', 'zs_config',
               'ref_windows', 'ref_needles', 'ref_config', 'tier', 'label'])

#@save
_LC_COLUMNS = {
    'table': ['arm', 'val', 'Δ val', 'NLL last', 'ctx', 'ctx lo', 'ctx hi',
              'near', 'far', 'reach', 'detectable'],
    'logs': ['arm', 'seed', 'step', 'tokens', 'stage', 'lr', 'loss',
             'grad_norm', 'tok_s'],
    'position': ['model', 'bucket', 'nll', 'lo', 'hi'],
    'context': ['model', 'context', 'nats', 'lo', 'hi', 'bpb', 'bpb_lo',
                'bpb_hi'],
    'recall': ['model', 'seed', 'length', 'gap_lo', 'gap_hi', 'n', 'k',
               'rate', 'lo', 'hi'],
    'gain': ['model', 'bucket', 'gain', 'lo', 'hi', 'n'],
    'timeline': ['arm', 'seed', 'step', 'tokens', 'length', 'nll_first',
                 'nll_last', 'near_k', 'near_n', 'near_lo', 'far_k', 'far_n',
                 'far_lo', 'gain', 'gain_lo'],
    'short': ['arm', 'Δ val', 'val lo', 'val hi', 'Δ suite', 'suite lo',
              'suite hi', 'Δ near', 'near lo', 'near hi'],
    'windows': ['model', 'seed', 'measure', 'bucket', 'window', 'sum', 'n'],
    'needles': ['model', 'seed', 'length', 'window', 'needle', 'gap',
                'correct'],
    'items': ['model', 'seed', 'task', 'item_id', 'score'],
    'zs_table': ['scheme', 'length', 'first', 'first lo', 'first hi',
                 'last', 'last lo', 'last hi', 'Δ', 'Δ lo', 'Δ hi', 'near', 'near lo', 'near hi',
                 'far', 'far lo', 'far hi', 'reach'],
    'zs_windows': ['model', 'seed', 'measure', 'bucket', 'window', 'sum',
                   'n'],
    'zs_needles': ['model', 'seed', 'length', 'window', 'needle', 'gap',
                   'correct'],
    'ref_windows': ['model', 'seed', 'measure', 'bucket', 'window', 'sum',
                    'n'],
    'ref_needles': ['model', 'seed', 'length', 'window', 'needle', 'gap',
                    'correct']}

#@save
_LC_STATE_EVERY = 50

#@save
_LC_NOT_IDENTITY = ('evaluate',)  # fields that do not change training

#@save
def _lc_slug(arm):
    return arm.lower().replace(' ', '-')

def _lc_configs(arms, base, tokens, seq_len, batch_tokens, peak_lr=1e-4,
                warmup=20, decay_frac=0.15, long_share=0.5):
    """One JSON-able configuration per (arm, seed): the stages (window T,
    scale s, steps), the schedule and the data shares; evaluate: whether
    the periodic needle evaluation runs (T2 only)."""
    tier = _lc_tier(base)
    out = {}
    for name, a in arms.items():
        stages = []
        for T, s, share in a['stages']:
            if seq_len % T or seq_len // T > 2:
                raise ValueError(f'stage window {T}: seq_len or half of it')
            stages.append(dict(T=T, s=s, steps=round(share * tokens /
                                                     batch_tokens)))
        steps = sum(st['steps'] for st in stages)
        for seed in a.get('seeds', (0,)):
            out[name, seed] = dict(
                tier=tier, arm=name, seed=seed, base=base,
                scheme=a['scheme'], mscale=a.get('mscale'), stages=stages,
                steps=steps, seq_len=seq_len,
                global_batch=batch_tokens // seq_len,
                micro_batch=_LC_MICRO[tier], peak_lr=peak_lr, warmup=warmup,
                decay=round(decay_frac * steps), long_share=long_share,
                trained_len=_LC_TIERS[tier]['trained'], z_loss=1e-4,
                eval_every=_LC_EVAL_EVERY, evaluate=tier == 't2')
    return out

def _lc_stage_of(cfg, step):
    """The index of the stage that step (0-based) belongs to."""
    end = 0
    for k, st in enumerate(cfg['stages']):
        end += st['steps']
        if step < end:
            return k
    return len(cfg['stages']) - 1

def _lc_set_stage(model, cfg, k):
    st = cfg['stages'][k]
    return _lc_set_scheme(model, cfg['scheme'], st['s'], cfg['trained_len'],
                          cfg['mscale'])

class _StagedStream:
    """A MixStream whose windows are cut into seq_len / T pieces in the
    stages with a shorter window T (document ids and positions recomputed
    per piece, so no piece attends to another): the same tokens per step
    and one compiled shape."""
    def __init__(self, mix, cfg, eos_id):
        self.mix, self.cfg, self.eos_id = mix, cfg, eos_id
        self.rank, self.world = mix.rank, mix.world

    @property
    def step(self):
        return self.mix.step

    @step.setter
    def step(self, value):
        self.mix.step = value

    def batch(self, step):
        b = self.mix.batch(step)
        T = self.cfg['stages'][_lc_stage_of(self.cfg, step)]['T']
        X = b['input_ids']
        n, k = X.shape[0], X.shape[1] // T
        if k == 1:
            return b
        doc, pos = d2l.doc_ids_from_eos(X.reshape(n * k, T).int(),
                                        self.eos_id)
        doc = doc.view(n, k, T).long()
        first = F.pad((doc.amax(-1) + 1)[:, :-1], (1, 0)).cumsum(-1)
        return dict(b, doc_ids=(doc + first[..., None]).view(n, -1).to(
            b['doc_ids'].dtype), pos_ids=pos.view(n, -1).to(
            b['pos_ids'].dtype))

    def state_dict(self):
        return self.mix.state_dict()

    def load_state_dict(self, state):
        self.mix.load_state_dict(state)

def _lc_stream(shards, cfg):
    """The arm's batches: the long half and the replay in the shares
    (long_share, 1 - long_share), the arm's seed for streams and mixture."""
    T, B, seed = cfg['seq_len'], cfg['global_batch'], cfg['seed']
    streams = [d2l.PackedStream(shards[k], T, B, seed=seed)
               for k in ('long', 'short')]
    mix = d2l.mix_streams(streams, (cfg['long_share'],
                                    1 - cfg['long_share']), B, seed=seed)
    return _StagedStream(mix, cfg, shards['long'].eos_id)

def _lc_job(cfg):
    return os.path.join(d2l._job_root(), 'lc', f'{cfg["tier"]}-'
                        f'{_lc_slug(cfg["arm"])}-s{cfg["seed"]}')

def _lc_identity(cfg):
    """The record of an arm in its job folder: configuration (without the
    fields that do not change training) and base."""
    keep = {k: v for k, v in cfg.items() if k not in _LC_NOT_IDENTITY}
    return dict(config=d2l._jsonable(keep),
                base=[cfg['base'], d2l._file_id(_lc_base_path(cfg['base']))])

def _lc_check_recipe(cfg):
    """Refuse an arm of seed > 0 whose configuration differs from its
    seed-0 arm (the finished folder's record, else the configuration seed 0
    would have) in anything but the seed and the evaluation flag: the new
    T2 seeds 1-2 and T1 seeds 3-4 train with the recipe of the first
    build's seeds."""
    zero = dict(cfg, seed=0)
    path = os.path.join(_lc_job(zero), 'arm.json')
    ref = _lc_identity(zero)
    if os.path.exists(path):
        with open(path) as f:
            ref = json.load(f)
    mine = _lc_identity(cfg)
    diff = sorted(k for k in set(mine['config']) | set(ref['config'])
                  if mine['config'].get(k) != ref['config'].get(k))
    if diff != ['seed'] or mine['base'] != ref['base']:
        raise ValueError(f'{cfg["arm"]} seed {cfg["seed"]} differs from seed '
                         f'0 in {diff or ["base"]}')

def _bf16_copy(model):
    import copy
    return copy.deepcopy(model).to(torch.bfloat16).eval()

def _lc_timeline_row(model, cfg, step, tokenizer):
    """The periodic evaluation of a bf16 copy of model after `step` steps:
    the losses of the first and last position buckets, near recall
    (32, 512] and far recall (beyond the trained length) with lower Wilson
    bounds, and the gain beyond the trained length with its lower bound,
    all at the window of the stage the model is in (`length`: a
    progressive arm's first stage is evaluated at its own window, with
    the scale it is trained at, never beyond it)."""
    spec, L0 = _LC_TIERS[cfg['tier']], cfg['trained_len']
    T = cfg['stages'][_lc_stage_of(cfg, max(step - 1, 0))]['T']
    ev = _bf16_copy(model)
    w, r, _ = _lc_eval(ev, tokenizer, cfg['tier'], T, (T,),
                       spec['eval_needles'], spec['eval_windows'],
                       contexts=False)
    del ev
    d2l.lm_free()
    ratio = lambda x: float(x['sum'].sum() / x['n'].sum())
    nll = w[w['measure'] == 'nll']
    far = [bucket_name(a, b) for a, b in _lc_gain_bins(T) if a >= L0]
    g = w[(w['measure'] == 'gain') & w['bucket'].isin(far)].groupby(
        'window')[['sum', 'n']].sum()
    gain, gain_lo, _ = window_ci(g['sum'], g['n'])
    out = dict(arm=cfg['arm'], seed=cfg['seed'], step=step,
               tokens=step * cfg['global_batch'] * cfg['seq_len'], length=T,
               nll_first=ratio(nll[nll['bucket'] == bucket_name(0, L0)]),
               nll_last=ratio(nll[nll['bucket'] == bucket_name(T - L0, T)]))
    for name, (a, b) in (('near', _LC_NEAR), ('far', (L0, T))):
        x = r[(r['gap'] > a) & (r['gap'] <= b)]['correct']
        out.update({f'{name}_k': int(x.sum()), f'{name}_n': len(x),
                    f'{name}_lo': d2l.wilson_ci(int(x.sum()), len(x))[0]})
    return dict(out, gain=gain, gain_lo=gain_lo, probe='needle')

def _lc_read_timeline(job):
    """The periodic needle evaluations in job (none for an arm trained
    before them: its rows have no 'probe' field)."""
    path = os.path.join(job, 'timeline.jsonl')
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [r for r in map(json.loads, f) if r.get('probe') ==
                    'needle']
    if not rows:
        return pd.DataFrame(columns=_LC_COLUMNS['timeline'])
    return pd.DataFrame(rows).drop_duplicates('step', keep='last')[
        _LC_COLUMNS['timeline']]

def _lc_train(model, stream, cfg, job, stop=None):
    """Train one arm (resumable in job; at most `stop` steps): the log
    rows; the periodic evaluations (when cfg['evaluate']) go to
    job/timeline.jsonl."""
    dev = _device_of(model)
    T = cfg['seq_len']
    pc = d2l.PretrainConfig(
        f'lc-{_lc_slug(cfg["arm"])}', d2l._arch_of(model, T), cfg['steps'],
        cfg['global_batch'], cfg['micro_batch'], cfg['peak_lr'],
        cfg['warmup'], cfg['decay'], z_loss=cfg['z_loss'],
        compile=dev.type == 'cuda', probe_every=0, stats_every=0,
        val_every=0, ckpt_steps=(cfg['steps'],),
        state_every=_LC_STATE_EVERY, seed=cfg['seed'], out_dir=job)
    run = d2l._PretrainRun(model, stream, pc, None, 'auto')
    stage = _lc_stage_of(cfg, run.start)
    _lc_set_stage(run.raw, cfg, stage)  # RoPE is not in the state
    evaluate = cfg.get('evaluate', False)
    tok = lc_tokenizer() if evaluate else None
    timeline = os.path.join(job, 'timeline.jsonl')

    def log_eval(step):
        row = _lc_timeline_row(run.raw, cfg, step, tok)
        with open(timeline, 'a') as f:
            f.write(json.dumps(d2l._jsonable(row)) + '\n')

    if evaluate and run.start == 0:
        log_eval(0)
    for seg in run.segments():
        for step in range(seg.start, min(seg.total, stop or seg.total)):
            if _lc_stage_of(cfg, step) != stage:  # the next stage's scale
                stage = _lc_stage_of(cfg, step)
                _lc_set_stage(run.raw, cfg, stage)
                torch._dynamo.reset()
                run.model = torch.compile(run.raw) if pc.compile else run.raw
            lr = d2l.wsd_lr(step, cfg['steps'], cfg['warmup'], cfg['decay'],
                            cfg['peak_lr'])
            for group in run.optimizer.param_groups:
                group['lr'] = lr
            t0 = time.perf_counter()
            batch, micro, n_valid = run.batch(step)
            t1 = time.perf_counter()
            out = d2l._cpt_step(run.model, run.optimizer, micro, pc, n_valid)
            done = step + 1
            row = dict(step=done, tokens=done * run.step_tokens, lr=lr,
                       stage=stage, **out)
            run.record(seg, row, batch, None, t1 - t0, t0)
            if run.stopped(row):
                raise FloatingPointError(f'{job}: non-finite loss at '
                                         f'step {done}')
            if evaluate and (done % cfg['eval_every'] == 0
                             or done == cfg['steps']):
                log_eval(done)
    return run.finish()

def _lc_check_disk(cfg):
    path = os.path.abspath(d2l._job_root())
    while not os.path.exists(path):
        path = os.path.dirname(path)
    free = shutil.disk_usage(path).free / 2 ** 30
    if free < _LC_MIN_FREE_GB[cfg['tier']]:
        raise OSError(f'{free:.0f} GiB free under {path}; a {cfg["tier"]} '
                      f'arm starts with at least '
                      f'{_LC_MIN_FREE_GB[cfg["tier"]]} GiB')

def _lc_arm_run(cfg):
    """Train one arm and seed (or read its finished job folder):
    {log, timeline, gib, tok_s, label}."""
    job, ident = _lc_job(cfg), _lc_identity(cfg)
    done = os.path.join(job, 'DONE')
    if not d2l._cpt_check_job(job, ident):
        d2l._cpt_check_job(job, ident, 'arm.json')  # resume the same arm
        if cfg['seed']:
            _lc_check_recipe(cfg)
        _lc_check_disk(cfg)
        os.makedirs(job, exist_ok=True)
        with open(os.path.join(job, 'arm.json'), 'w') as f:
            json.dump(ident, f)
        dev = _lc_device()
        torch._dynamo.reset()  # one process trains several arms
        model = d2l._load_bf16(_lc_base_path(cfg['base']), torch.float32,
                               dev).train()
        shards = long_data(cfg['tier']).shards
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(dev)
        with d2l._quiet():
            _lc_train(model, _lc_stream(shards, cfg), cfg, job)
        gib = (torch.cuda.max_memory_allocated(dev) / 2 ** 30
               if dev.type == 'cuda' else float('nan'))
        with open(done, 'w') as f:
            json.dump(dict(ident, gib=gib), f)
        shutil.rmtree(os.path.join(job, 'state'), ignore_errors=True)
        del model
        d2l.lm_free()
    with open(done) as f:
        info = json.load(f)
    log = d2l._read_job_logs(job).assign(arm=cfg['arm'], seed=cfg['seed'])
    steady = log.loc[log['step'] > 20, 'tok_s']
    return dict(log=log.reindex(columns=_LC_COLUMNS['logs']),
                timeline=_lc_read_timeline(job) if cfg.get('evaluate')
                else pd.DataFrame(columns=_LC_COLUMNS['timeline']),
                gib=info['gib'],
                tok_s=float(steady.median()) if len(steady) else math.nan,
                label='')

def _lc_eval_settings(tier, cfg):
    """What a final evaluation measures: the base (cfg None) at its trained
    length (needles, val loss, suite); an arm at the trained length, every
    stage's window and its own (needles), loss windows at its own length,
    val loss and suite."""
    spec = _LC_TIERS[tier]
    common = dict(needles=spec['needles'], val=_LC_VAL_TOKENS,
                  suite=[list(LC_TASKS), _LC_SUITE_N])
    if cfg is None:
        return dict(common, T=spec['trained'], lengths=[spec['trained']],
                    windows=0)
    lengths = sorted({spec['trained'], cfg['seq_len']}
                     | {st['T'] for st in cfg['stages']})
    return dict(common, T=cfg['seq_len'], lengths=lengths,
                windows=spec['loss_windows'],
                contexts=_lc_contexts(tier, cfg['seq_len']))

def _lc_model_job(tier, cfg):
    """(job folder, weights) of a final evaluation: the base or an arm."""
    if cfg is None:
        base = 'r2' if tier == 't2' else 't1'
        return (os.path.join(d2l._job_root(), 'lc', f'{tier}-base'),
                _lc_base_path(base))
    job = _lc_job(cfg)
    if not d2l._cpt_check_job(job, _lc_identity(cfg)):
        raise FileNotFoundError(f'{job}: the arm is not done')
    return job, os.path.join(job, 'ckpt', f's{cfg["steps"]}.safetensors')

def _lc_model_eval(tier, cfg):
    """LcEval of one model (the base when cfg is None), kept in its job
    folder's evals.json.gz under the settings and the weights' id."""
    job, weights = _lc_model_job(tier, cfg)
    s = _lc_eval_settings(tier, cfg)
    ident = d2l._jsonable(dict(s, weights=d2l._file_id(weights)))
    path = os.path.join(job, 'evals.json.gz')
    if os.path.exists(path):
        res = d2l._read_result(path, 'json-table')
        d2l._check_config(res, ident, path)
        return res
    model = d2l._load_bf16(weights, device=_lc_device())
    suite = d2l._cpt_suite(LC_TASKS, n_max=_LC_SUITE_N, name='lc')
    w, r, it = _lc_eval(model, lc_tokenizer(), tier, s['T'], s['lengths'],
                        s['needles'], s['windows'], val=True, suite=suite)
    del model
    d2l.lm_free()
    res = LcEval(w, r, it, '')
    os.makedirs(job, exist_ok=True)
    d2l._write_result(path, 'json-table', res, dict(config=ident, label=''))
    return res

def _lc_eval_one(cfgs, only):
    """lc-t2-evals --only: the final evaluation of the model whose job
    folder is named only ('t2-base' or an arm's)."""
    tier = next(iter(cfgs.values()))['tier']
    names = {f'{tier}-base': None, **{os.path.basename(_lc_job(c)): c
                                      for c in cfgs.values()}}
    if only not in names:
        raise ValueError(f'--only is one of {sorted(names)}, not {only!r}')
    return _lc_model_eval(tier, names[only])
```

## The Merge

The merge reads every model's final evaluation (computing the missing
ones), the zero-shot part and, at T2, the reference part, and computes the
tables the page shows. Zero-shot models are named `<scheme>, zero-shot`.
The base row of the table is the zero-shot PI model at the arms' length
with the base's own val loss; the reference model's row has recall only
(its losses are in its own tokens).

```{.python .input #long-context-lib-merge}
%%tab pytorch
#@save
def _suite_delta(a, b, n_boot=2000, seed=0):
    """Suite mean (the mean over tasks of each task's mean) of a minus b on
    the same items, with a 95% interval from resampling items within each
    task: (delta, lo, hi)."""
    m = a.merge(b, on=['task', 'item_id'], suffixes=('_a', '_b'))
    if not len(m):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    point, boots = [], []
    for _, t in m.groupby('task'):
        d = (t['score_a'] - t['score_b']).to_numpy(dtype=float)
        point.append(d.mean())
        boots.append(d[rng.integers(len(d), size=(n_boot, len(d)))].mean(1))
    lo, hi = np.quantile(np.mean(boots, 0), [0.025, 0.975])
    return float(np.mean(point)), float(lo), float(hi)

def _lc_short(tier, windows, items, needles, arms):
    """Per arm: the val loss at the trained length, the suite mean and near
    recall at the trained length minus the base's, with 95% intervals
    (paired by window, item and query; seeds averaged per window, item
    and query)."""
    L0 = _LC_TIERS[tier]['trained']
    val = windows[windows['measure'] == 'val'].groupby(
        ['model', 'window'])[['sum', 'n']].sum()
    per = (val['sum'] / val['n']).unstack('model')
    score = items.groupby(['model', 'task', 'item_id'])['score'].mean()
    near = needles[(needles['length'] == L0) & (needles['gap'] >
                                                _LC_NEAR[0])
                   & (needles['gap'] <= _LC_NEAR[1])]
    q = near.groupby(['model', 'window', 'needle'])['correct'].mean(
        ).unstack('model')
    rows = []
    for arm in arms:
        d, lo, hi = window_ci(per[arm] - per['base'], np.ones(len(per)))
        s = _suite_delta(score[arm].reset_index(),
                         score['base'].reset_index())
        dq = (q[arm] - q['base']).groupby('window').agg(['sum', 'count'])
        rows.append((arm, d, lo, hi, *s, *window_ci(dq['sum'],
                                                    dq['count'])))
    return pd.DataFrame(rows, columns=_LC_COLUMNS['short'])

def _lc_reach(recall, model, L, lo=0.5, strict=False):
    b = recall[(recall['model'] == model) & (recall['seed'] == -1)
               & (recall['length'] == L)]
    return d2l.reach(b, lo, strict) if len(b) else math.nan

def _lc_table(tier, T, windows, needles, recall, arms, ref):
    """Per model (the arms, the base as the zero-shot PI model, the
    reference): val loss at the trained length, the loss in the last
    position bucket, loss with the whole window minus with the trained
    length on the second half of each window (with its interval), near
    and far recall at T (seeds averaged per query), reach and detectable
    to at T."""
    L0 = _LC_TIERS[tier]['trained']
    val = windows[windows['measure'] == 'val'].groupby('model')[
        ['sum', 'n']].sum()
    val = val['sum'] / val['n']
    pos = _lc_position(windows)
    last = bucket_name(T - L0, T)
    rows = []
    for name, model in [(a, a) for a in arms] + [('base', 'PI, zero-shot')] + (
            [(ref, ref)] if ref else []):
        p = pos[(pos['model'] == model) & (pos['bucket'] == last)]['nll']
        v = val.get('base' if name == 'base' else model, math.nan)
        ctx = _lc_ctx_delta(windows, model, _length_name(L0))[:3]
        if model == ref:
            v, ctx, p = math.nan, (math.nan,) * 3, []
        rows.append((name, v, v - val.get('base', math.nan),
                     float(p.iloc[0]) if len(p) else math.nan, *ctx,
                     _lc_range(needles, model, T, *_LC_NEAR)[0],
                     _lc_range(needles, model, T, L0, T)[0],
                     _lc_reach(recall, model, T),
                     _lc_reach(recall, model, T, 0.25, strict=True)))
    return pd.DataFrame(rows, columns=_LC_COLUMNS['table'])

def _lc_merge(parts, cfgs, zs, zs_config, ref=None, ref_config=None):
    """LcRuns from the arms' parts {(arm, seed): part}, the zero-shot result
    and (T2) the reference: every model's final evaluation (read from its
    job folder, computed if missing), then the tables."""
    cfg0 = next(iter(cfgs.values()))
    tier, T, C = cfg0['tier'], cfg0['seq_len'], _LC_COLUMNS
    evals = [('base', 0, _lc_model_eval(tier, None))] + [
        (a, s, _lc_model_eval(tier, c)) for (a, s), c in cfgs.items()]
    cat = lambda xs, cols: pd.concat(xs, ignore_index=True)[cols]
    windows = cat([e.windows.assign(model=a, seed=s) for a, s, e in evals],
                  C['windows'])
    needles = cat([e.needles.assign(model=a, seed=s) for a, s, e in evals],
                  C['needles'])
    items = cat([e.items.assign(model=a, seed=s) for a, s, e in evals],
                C['items'])
    zero = lambda df: df.assign(model=df['model'] + ', zero-shot')
    ref_name = None if ref is None else ref.needles['model'].iloc[0]
    all_w = pd.concat([windows, zero(zs.windows)] + (
        [] if ref is None else [ref.windows]), ignore_index=True)
    all_n = pd.concat([needles, zero(zs.needles)] + (
        [] if ref is None else [ref.needles]), ignore_index=True)
    arms = list(dict.fromkeys(a for a, _ in cfgs))
    recall = _lc_recall(all_n)
    shown = ['PI, zero-shot', 'YaRN, zero-shot'] + arms
    position = _lc_position(all_w[all_w['model'].isin(shown)])
    context = _lc_context(all_w, shown + ([ref_name] if ref_name else []))
    table = _lc_table(tier, T, all_w, all_n, recall, arms, ref_name)
    parts = [parts[k] for k in cfgs]
    empty = lambda k: pd.DataFrame(columns=C[k])
    return LcRuns(
        table, cat([p['log'] for p in parts], C['logs']), position, context,
        recall, _lc_gain(windows), cat([empty('timeline')] + [
            p['timeline'] for p in parts if len(p['timeline'])],
            C['timeline']), _lc_short(tier, windows, items, needles, arms),
        windows, needles, items, zs.table, zs.windows, zs.needles, zs_config,
        empty('windows') if ref is None else ref.windows,
        empty('needles') if ref is None else ref.needles, ref_config, tier,
        '')

def _lc_t1_compute(cfgs):
    """The T1 campaign in one process: zero-shot on the t1 checkpoint at
    the T1 lengths, every arm and seed, and the merge."""
    spec = _LC_TIERS['t1']
    call = dict(_lc_zeroshot_call(), base='t1',
                lengths=(spec['trained'], spec['length']))
    zs = _lc_zeroshot_compute(**call)
    parts = {k: _lc_arm_run(c) for k, c in cfgs.items()}
    return _lc_merge(parts, cfgs, zs, d2l._jsonable(call))

def lc_arms(arms, base, tokens, seq_len, batch_tokens, peak_lr=1e-4,
            warmup=20, decay_frac=0.15, long_share=0.5):
    """Extend base ('r2' at 4K or 't1' at 1K) to seq_len, one run per arm
    and seed, and evaluate it: LcRuns(table, logs, position, context,
    recall, gain, timeline, short, windows, needles, items, zs_table,
    zs_windows, zs_needles, zs_config, ref_windows, ref_needles,
    ref_config, label). arms = {name: dict(scheme, stages, seeds[,
    mscale])}, stages a tuple of (window T, scale s, share of the tokens).
    base 'r2' (T2: one regeneration per arm and seed, _LC_T2_NAMES, merged
    by lc-t2-evals) or 't1' (the regeneration lc-t1). Otherwise the
    committed result; while the T2 arms are pending, the T1 arms, labelled
    so (every show_lc_* display prints the label)."""
    config = dict(arms=arms, base=base, tokens=tokens, seq_len=seq_len,
                  batch_tokens=batch_tokens, peak_lr=peak_lr, warmup=warmup,
                  decay_frac=decay_frac, long_share=long_share)
    cfgs = _lc_configs(arms, base, tokens, seq_len, batch_tokens, peak_lr,
                       warmup, decay_frac, long_share)
    if base == 't1':
        res = d2l._pretrain_result('lc-t1', lambda: _lc_t1_compute(cfgs),
                                   config)
        return d2l._as_tuple(LcRuns, res, _LC_COLUMNS)
    if set(cfgs) - set(_LC_T2_NAMES):
        raise ValueError(f'T2 arms and seeds are {sorted(_LC_T2_NAMES)}')
    only = os.environ.get('D2L_PRETRAIN_ONLY')
    if only and 'lc-t2-evals' in d2l._regenerating():
        _lc_eval_one(cfgs, only)  # one model; the merge reads it back
        return d2l._as_tuple(LcRuns, d2l._pending_result('lc-t2-evals'),
                             _LC_COLUMNS)
    parts = {_LC_T2_NAMES[k]: (lambda c=c: _lc_arm_run(c), c)
             for k, c in cfgs.items()}
    names = {_LC_T2_NAMES[k]: k for k in cfgs}

    def merge(results):
        zs, zs_config = _lc_part_for_merge(
            'lc-zeroshot', _lc_zeroshot_call(), _lc_zeroshot_compute,
            lambda r: LcZeroShot(r.table, r.windows, r.needles, ''))
        ref, ref_config = _lc_part_for_merge(
            'lc-reference', _lc_reference_call(), _lc_reference_compute,
            lambda r: LcReference(r.windows, r.needles, ''))
        return _lc_merge({names[k]: d2l._part(v) for k, v in
                          results.items()}, cfgs, zs, zs_config, ref,
                         ref_config)
    res = d2l._pretrain_merged('lc-t2-evals', parts, merge, config,
                               fallback='lc-t1')
    return d2l._as_tuple(LcRuns, res, _LC_COLUMNS)

def lc_memory_probe(micro_batches=(1, 2), steps=10, warmup=3):
    """The D2 probe of the T2 arms on one GPU: per configuration (the PI
    arm at each micro-batch size, and the progressive arm's first stage,
    8K pieces, at the first) peak GiB reserved, median seconds per step
    after compilation, tokens per second and the projected hours of the
    arm's steps (no evaluations), as a DataFrame."""
    import tempfile
    cfgs = _lc_configs(**_lc_t2_call())
    runs = [('PI', m) for m in micro_batches] + [('YaRN-prog',
                                                  micro_batches[0])]
    shards, dev, rows = long_data('t2').shards, torch.device('cuda'), []
    for arm, m in runs:
        cfg = dict(cfgs[arm, 0], micro_batch=m, evaluate=False)
        if arm == 'YaRN-prog':  # the first stage only
            cfg['stages'] = [dict(cfg['stages'][0], steps=cfg['steps'])]
        torch._dynamo.reset()
        model = d2l._load_bf16(_lc_base_path('r2'), torch.float32,
                               dev).train()
        torch.cuda.reset_peak_memory_stats(dev)
        os.makedirs(d2l._job_root(), exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=d2l._job_root()) as job, \
                    d2l._quiet():
                log = _lc_train(model, _lc_stream(shards, cfg), cfg, job,
                                stop=steps + warmup)
            t = float(log['step_s'].iloc[warmup:].median())
        except torch.OutOfMemoryError:
            t = math.nan  # the row says so: gib is the card's limit
        T = cfg['stages'][0]['T']
        rows.append(dict(arm=arm, window=T, micro_batch=m,
                         gib=torch.cuda.max_memory_reserved(dev) / 2 ** 30,
                         step_s=t, tok_s=cfg['global_batch'] *
                         cfg['seq_len'] / t,
                         arm_hours=cfgs[arm, 0]['steps'] * t / 3600))
        del model
        torch._dynamo.reset()
        d2l.lm_free()
    return pd.DataFrame(rows)
```

## Plots and the Scheme Difference

```{.python .input #long-context-lib-plots}
%%tab pytorch
#@save
def _lc_rules(axes, chance=1 / 8):
    """Chance (dashed) and the reach thresholds 0.5 and 0.25 (dotted)."""
    axes.axhline(chance, color='black', ls='--', lw=0.8)
    for y in (0.5, 0.25):
        axes.axhline(y, color='black', ls=':', lw=0.8)

def _lc_recall_axes(axes, curves, grey=(), title=None):
    """Recall against gap (log axis) on axes: curves {label: bins} (gap_lo,
    gap_hi, rate, lo, hi), each with its Wilson band; labels in grey drawn
    in grey."""
    for i, (label, b) in enumerate(curves.items()):
        x = np.sqrt(b['gap_lo'] * b['gap_hi']).to_numpy()
        color = '0.5' if label in grey else f'C{i % 10}'
        axes.plot(x, b['rate'], '-o', ms=3, color=color, label=label)
        axes.fill_between(x, b['lo'], b['hi'], color=color, alpha=0.15,
                          linewidth=0)
    _lc_rules(axes)
    axes.set_xscale('log', base=2)
    axes.set_ylim(0, 1.02)
    axes.set_xlabel('gap (tokens back)')
    if title:
        axes.set_title(title)

def plot_lc_train(logs, smooth=10):
    """Training loss (a running mean over smooth steps) against tokens for
    each arm's first seed, the progressive arm's stage switch marked."""
    logs = logs[logs['seed'] == 0]
    if not len(logs):
        return
    X, Y, names = [], [], []
    for arm, g in logs.groupby('arm', sort=False):
        X.append(g['tokens'].to_numpy() / 1e6)
        Y.append(g['loss'].rolling(smooth, min_periods=1).mean().to_numpy())
        names.append(arm)
    d2l.plot(X, Y, 'tokens (millions)', 'training loss (nats)',
             legend=names, figsize=(4.5, 2.8))
    for g in [g for _, g in logs.groupby('arm') if g['stage'].nunique() > 1]:
        switch = g.loc[g['stage'].diff() > 0, 'tokens'].iloc[0] / 1e6
        d2l.plt.gca().axvline(switch, color='black', ls=':', lw=1)

def plot_lc_loss(runs):
    """Left: loss by position bucket with 95% intervals (the zero-shot PI
    and YaRN models and the arms). Right: the loss of the second half of
    each window given c tokens of context minus given the first context,
    in bits per byte when the reference model is drawn (other tokenizer),
    else in nats per token."""
    fig, (a, b) = d2l.plt.subplots(1, 2, figsize=(8, 2.8))
    p = runs.position
    buckets = list(dict.fromkeys(p['bucket']))
    d2l.plot_intervals(buckets, {
        m: tuple(g.set_index('bucket').reindex(buckets)[k].to_numpy()
                 for k in ('nll', 'lo', 'hi'))
        for m, g in p.groupby('model', sort=False)}, 'position bucket',
        'loss (nats)', axes=a)
    c = runs.context
    bpb = len(runs.ref_needles) > 0
    cols = ('bpb', 'bpb_lo', 'bpb_hi') if bpb else ('nats', 'lo', 'hi')
    names = list(dict.fromkeys(c['context']))
    d2l.plot_intervals(names, {
        m: tuple(g.set_index('context').reindex(names)[k].to_numpy()
                 for k in cols)
        for m, g in c.groupby('model', sort=False)}, 'context given',
        f'Δ loss ({"bits per byte" if bpb else "nats"})', axes=b,
        reference=0.0)
    for ax in (a, b):
        if ax.get_legend():
            ax.get_legend().remove()
    b.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small')

def plot_lc_recall(runs):
    """Recall against gap with Wilson bands (seeds pooled): at the arms'
    length the zero-shot PI model, the arms and the reference (grey); at
    the trained length the base unscaled, the arms and the reference."""
    r = runs.recall[runs.recall['seed'] == -1]
    T, ref = int(r['length'].max()), None
    if len(runs.ref_needles):
        ref = runs.ref_needles['model'].iloc[0]
    L0 = min(L for L in r['length'].unique())
    arms = list(dict.fromkeys(runs.table['arm']))[:-2 if ref else -1]
    fig, axes = d2l.plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    for ax, L, first in ((axes[0], T, 'PI, zero-shot'), (axes[1], L0,
                                                          'base')):
        curves = {m: r[(r['model'] == m) & (r['length'] == L)]
                  for m in [first] + arms + ([ref] if ref else [])}
        _lc_recall_axes(ax, {m: b for m, b in curves.items() if len(b)},
                        grey=(ref,), title=f'window {_length_name(L)}')
    axes[0].set_ylabel('recall')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                   fontsize='small')

def plot_reach_train(probe):
    """Recall against gap (log axis, each octave bin at its geometric
    middle) at the trained length, one line per checkpoint from light
    (early) to dark (late), a panel per run (t1 at 1K, r2 at 4K), each
    with its legend beside it; chance dashed, the reach thresholds 0.5 and
    0.25 dotted."""
    runs = list(probe.reach_train.groupby('model', sort=False))
    fig, axes = d2l.plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 2.8),
                                 sharey=True, squeeze=False,
                                 layout='constrained')
    for ax, (m, x) in zip(axes[0], runs):
        by = [(step, b.sort_values('gap_lo')) for step, b in x.groupby('step')]
        g = [np.sqrt(b['gap_lo'] * b['gap_hi']).to_numpy() for _, b in by]
        names = [f'step {step:,}' for step, _ in by]
        d2l.plot(g, [b['rate'].to_numpy() for _, b in by],
                 'gap (tokens back)', 'recall' if ax is axes[0, 0] else '',
                 ylim=(0, 1.02), xscale='log', fmts=['-o'] * len(by),
                 figsize=(4.5 * len(runs), 2.8), axes=ax)
        for i, line in enumerate(ax.get_lines()):
            line.set_color(d2l.plt.cm.viridis_r(0.15 + 0.75 * i / max(
                1, len(by) - 1)))
            line.set_markersize(3)
        ax.legend(names, fontsize='x-small', loc='center left',
                  bbox_to_anchor=(1, 0.5))  # beside the curves, recoloured
        ax.set_xscale('log', base=2)
        _lc_rules(ax)
        ax.set_title(f'{m} at {_length_name(int(x["gap_hi"].max()))}')

def plot_lc_gain(gain):
    """Long-context gain against the gap bucket with 95% intervals."""
    if not len(gain):
        return
    buckets = list(dict.fromkeys(gain['bucket']))
    series = {m: tuple(x.set_index('bucket').reindex(buckets)[k].to_numpy()
                       for k in ('gain', 'lo', 'hi'))
              for m, x in gain.groupby('model', sort=False)}
    axes = d2l.plot_intervals(buckets, series,
                              'gap to the previous occurrence', 'gain (nats)',
                              reference=0.0)
    axes.legend(loc='upper left', bbox_to_anchor=(1.01, 1))

def scheme_difference(runs, a='YaRN', b='PI'):
    """a minus b per seed (both arms trained on the same data order), on
    the loss in the last position bucket, near recall (32, 512] and far
    recall (beyond the trained length) at the arms' length: a dict of
    (mean, lo, hi) per measure (a t interval over the seeds; nan below two
    seeds) and the seeds. Each difference pairs the two arms of one seed,
    so both arms must have the same seeds."""
    w, n = runs.windows, runs.needles
    L0 = int(n['length'].min())
    T = int(n['length'].max())
    last = bucket_name(T - L0, T)
    seeds = [sorted(set(n[n['model'] == m]['seed'])) for m in (a, b)]
    if seeds[0] != seeds[1]:
        raise ValueError(f'{a} has seeds {seeds[0]}, {b} {seeds[1]}: the '
                         'difference pairs the arms by seed')
    seeds = seeds[0]
    wl = w[(w['measure'] == 'nll') & (w['bucket'] == last)].groupby(
        ['model', 'seed'])[['sum', 'n']].sum()
    nll = wl['sum'] / wl['n']
    at = n[n['length'] == T]
    rate = lambda m, s, lo, hi: at[(at['model'] == m) & (at['seed'] == s) & (
        at['gap'] > lo) & (at['gap'] <= hi)]['correct'].mean()
    out = dict(seeds=seeds)
    for name, f in (('loss', lambda m, s: nll[m, s]),
                    ('near', lambda m, s: rate(m, s, *_LC_NEAR)),
                    ('far', lambda m, s: rate(m, s, L0, T))):
        d = [f(a, s) - f(b, s) for s in seeds]
        if len(d) < 2:
            out[name] = (float(np.mean(d)) if d else math.nan, math.nan,
                         math.nan)
        else:
            m, h = d2l.t_interval(d)
            out[name] = (m, m - h, m + h)
    return out
```

## What the Page Prints

Every display of a T2 result prints the result's label first, so the T1
stand-in (`T2 run pending; showing the T1 configuration`) never passes for
the 16K run, and prints one `[... pending]` line instead of an empty table.
The numbers carry the precision the prose quotes: losses at three
decimals (two in the zero-shot table), recall at two (500 windows give a
Wilson half-width of about 0.04 per octave bin), lengths in tokens.
`lc_first_steps` compares the periodic evaluations (the first 16 test
windows) with the unscaled base's loss on the same windows (the zero-shot
`none` model), since the mean of 16 windows differs from that of 64 by
about the margin itself.

```{.python .input #long-context-lib-show}
%%tab pytorch
#@save
def _lc_tier_of(runs):
    """'t1' or 't2' of a result; None for a pending one (no rows)."""
    return getattr(runs, 'tier', None) if len(runs.needles) else None

def _lc_head(res, table, what):
    """Print res's label (a stand-in or pending result) and, when table has
    no rows and res no label, '[what pending]'; whether to show table."""
    d2l._show_label(res)
    if not len(table) and not d2l._label_of(res):
        print(f'[{what} pending]')
    return bool(len(table))

def _num(x, digits, sign=False):
    """x at digits decimals (never '-0.00'), with its sign if sign; '-'
    for nan."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    return f'{round(float(x), digits) + 0:{"+" if sign else ""}.{digits}f}'

def _interval(x, lo, hi, digits, sign=False):
    """'x [lo, hi]' at digits decimals."""
    return (f'{_num(x, digits, sign)} [{_num(lo, digits, sign)}, '
            f'{_num(hi, digits, sign)}]')

def _pm(x, lo, hi, digits, sign=False):
    """'x±h' with h the half-width of [lo, hi]; '-' for nan."""
    if math.isnan(float(x)):
        return '-'
    return f'{_num(x, digits, sign)}±{_num((hi - lo) / 2, digits)}'

def _tokens(x):
    """A gap or length in tokens: '1,024'; '-' for nan."""
    return '-' if math.isnan(float(x)) else f'{int(x):,}'

def show_lc_format(probe):
    """The format table: recall at 1K of each model on the template and on
    the one-hop needle, with 95% Wilson intervals, and template minus
    one-hop paired by item (the same window in both formats; interval
    from resampling items), under the label."""
    if not _lc_head(probe, probe.format, 'lc-probe'):
        return
    rows = {}
    for r in probe.format.itertuples():
        rows.setdefault(r.model, {})[r.format] = _interval(r.rate, r.lo, r.hi,
                                                           2)
    it = probe.format_items.assign(window=probe.format_items['item'])
    out = []
    for m, v in rows.items():
        x = it[it['model'] == m]
        d = recall_paired(x[x['format'] == 'template'],
                          x[x['format'] == 'one-hop'], on=('window',))
        out.append([m, v.get('template', '-'), v.get('one-hop', '-'),
                    _interval(*d, 2, sign=True)])
    print(pd.DataFrame(out, columns=['model', 'template', 'one-hop',
                                     'template − one-hop']).to_string(
        index=False))

def _copying_steps(x):
    """(a, b): b the first checkpoint at which recall in (32, 64] is
    detectable (Wilson lower bound above 0.25); a the checkpoint before it
    when that one is at chance (its Wilson interval contains 1/8), else
    None; b None where no checkpoint is detectable."""
    b = x[x['gap_lo'] == 32].sort_values('step')
    ok = b['lo'] > 0.25
    if not ok.any():
        return None, None
    first = int(b['step'][ok].min())
    before = b[b['step'] < first]
    if not len(before):
        return None, first
    last = before.iloc[-1]
    chance = last['lo'] <= 1 / 8 <= last['hi']
    return (int(last['step']) if chance else None), first

def _reach_verdict(r):
    """'non-decreasing' if no reach falls below the one before it, 'non-
    decreasing within one bin' if none falls below half the largest one
    before it, else 'falling'."""
    if all(b >= a for a, b in zip(r, r[1:])):
        return 'non-decreasing'
    top = np.maximum.accumulate(r)
    return ('non-decreasing within one bin' if all(
        b >= a / 2 for a, b in zip(top, r[1:])) else 'falling')

def show_reach_train(probe):
    """When copying appears in each run (from a checkpoint at chance to one
    with recall in (32, 64] detectable); for the last run (r2) its reach
    at each checkpoint from _LC_GROWTH_STEPS[0] to _LC_GROWTH_STEPS[1], and
    its recall in _LC_GROWTH at the second minus at the first (paired by
    query, window-resampled 95% interval): claim 11 at fixed steps, the
    copying line alone follows the data; then the plot."""
    if not _lc_head(probe, probe.reach_train, 'lc-probe'):
        return
    parts = []
    for m, x in probe.reach_train.groupby('model', sort=False):
        a, b = _copying_steps(x)
        parts.append(f'between steps {a:,} and {b:,} ({m})' if a and b else
                     f'by step {b:,} ({m})' if b else
                     f'not by step {int(x["step"].max()):,} ({m})')
    print('copying appears ' + ' and '.join(parts))
    m, x = list(probe.reach_train.groupby('model', sort=False))[-1]
    steps = sorted(x['step'].unique())
    s0, s1 = _LC_GROWTH_STEPS
    if s0 not in steps or s1 not in steps:
        raise ValueError(f'{m} has no checkpoint at step {s0} or {s1}')
    shown = [s for s in steps if s0 <= s <= s1]
    r = [d2l.reach(x[x['step'] == s]) for s in shown]
    L = _length_name(int(x['gap_hi'].max()))
    print(f'{m} reach at {L} (steps {s0:,} → {s1:,}): '
          f'{", ".join(f"{v:,}" for v in r)}; {_reach_verdict(r)}')
    q = probe.reach_queries
    q = q[(q['model'] == m) & (q['gap'] > _LC_GROWTH[0])
          & (q['gap'] <= _LC_GROWTH[1])]
    d = recall_paired(q[q['step'] == s1], q[q['step'] == s0])
    gaps = f'{_LC_GROWTH[0]}-{_length_name(_LC_GROWTH[1])}'
    print(f'{m} recall {gaps} back, step {s1:,} − '
          f'step {s0:,}: {_interval(*d, 2, sign=True)} (paired by query)')
    plot_reach_train(probe)

def show_lc_zeroshot(zs):
    """The zero-shot table: per scheme at the longest length the losses of
    the first and last position buckets and last minus first on the same
    windows, recall in (128, 256] and in the top octave of gap (±: half the
    95% interval), and the reach; then the control line, the unscaled base
    at its trained length."""
    if not _lc_head(zs, zs.table, 'lc-zeroshot'):
        return
    t = zs.table
    T, L0 = int(t['length'].max()), int(t['length'].min())
    first, last = f'0-{_length_name(L0)}', bucket_name(T - L0, T)
    pm = lambda x, k, d=2, sign=False: [_pm(*r, d, sign) for r in zip(
        x[k], x[f'{k} lo'], x[f'{k} hi'])]
    x = t[t['length'] == T]
    print(pd.DataFrame({
        'scheme': x['scheme'], f'loss {first}': pm(x, 'first'),
        f'loss {last}': pm(x, 'last'),
        f'{last} − {first}': pm(x, 'Δ', sign=True),
        'recall 128-256': pm(x, 'near'),
        f'recall {bucket_name(T // 2, T)}': pm(x, 'far'),
        'reach': x['reach'].map(_tokens)}).to_string(index=False))
    for r in t[t['length'] != T].to_dict('records'):
        c = lambda k: _pm(r[k], r[f'{k} lo'], r[f'{k} hi'], 2)
        print(f'{r["scheme"]} at {_length_name(r["length"])} (control): loss'
              f' {first} {c("first")}; recall 128-256 {c("near")}, '
              f'{bucket_name(L0 // 2, L0)} {c("far")}; reach '
              f'{_tokens(r["reach"])}')

def _lc_seed_diff(runs, a, b, seed=0):
    """a minus b, both at one seed, at the arms' length: the loss in the
    last position bucket (paired by window), near and far recall (paired
    by query), each (difference, lo, hi) with a window-resampled 95%
    interval."""
    T, L0 = (_LC_TIERS[_lc_tier_of(runs)][k] for k in ('length', 'trained'))
    w, n = runs.windows, runs.needles
    last = bucket_name(T - L0, T)

    def nll(m):
        x = w[(w['model'] == m) & (w['seed'] == seed) & (w['measure'] ==
                                                         'nll')
              & (w['bucket'] == last)]
        return x.set_index('window')[['sum', 'n']]

    def queries(m, lo, hi):
        return n[(n['model'] == m) & (n['seed'] == seed) & (n['length'] == T)
                 & (n['gap'] > lo) & (n['gap'] <= hi)]

    la, lb = nll(a), nll(b).reindex(nll(a).index)
    loss = window_ci((la['sum'] - lb['sum']).to_numpy(), la['n'].to_numpy())
    return dict(loss=loss, **{k: recall_paired(queries(a, *g), queries(b, *g))
                              for k, g in (('near', _LC_NEAR),
                                           ('far', (L0, T)))})

def show_lc_table(runs):
    """The arms' table (seeds averaged, the number of seeds by each arm):
    val loss at the trained length and its change from the base, the loss
    in the last position bucket, the loss on the second half with the
    whole window minus with the trained length [95% CI], near and far
    recall at the arms' length, reach and detectable to (±: half the 95%
    interval); then each progressive arm minus its one-shot arm at seed 0
    of each, labelled single-seed."""
    if not _lc_head(runs, runs.table, 'lc-t2-evals'):
        return
    t = runs.table
    T, L0 = (_LC_TIERS[_lc_tier_of(runs)][k] for k in ('length', 'trained'))
    seeds = runs.needles.groupby('model')['seed'].nunique()
    name = lambda a: (f'{a} ({seeds[a]} seed{"s" if seeds[a] > 1 else ""})'
                      if a in seeds.index and a != 'base' else a)
    print(pd.DataFrame({
        'arm': t['arm'].map(name),
        f'val {_length_name(L0)} (Δ)': [
            '-' if math.isnan(v) else f'{_num(v, 3)} ({_num(d, 3, True)})'
            for v, d in zip(t['val'], t['Δ val'])],
        f'NLL {bucket_name(T - L0, T)}': t['NLL last'].map(
            lambda x: _num(x, 3)),
        f'{_length_name(T)} − {_length_name(L0)} ctx': [
            _pm(x, lo, hi, 3, True)
            for x, lo, hi in zip(t['ctx'], t['ctx lo'], t['ctx hi'])],
        'near': t['near'].map(lambda x: _num(x, 2)),
        'far': t['far'].map(lambda x: _num(x, 2)),
        'reach': t['reach'].map(_tokens),
        'detect.': t['detectable'].map(_tokens)}).to_string(index=False))
    arms = set(t['arm'])
    for a in t['arm']:
        b = a[:-len('-prog')] if a.endswith('-prog') else None
        if b not in arms:
            continue
        d = _lc_seed_diff(runs, a, b)
        print(f'{a} − {b}, seed 0 of each (single seed): loss '
              f'{bucket_name(T - L0, T)} {_interval(*d["loss"], 3, True)}\n'
              f'  near {_interval(*d["near"], 2, True)}, far '
              f'{_interval(*d["far"], 2, True)}')

def show_lc_loss(runs):
    """The two loss panels (by position; against the context given)."""
    if _lc_head(runs, runs.position, 'lc-t2-evals'):
        plot_lc_loss(runs)

def lc_reach_lines(runs, reach, seed=0):
    """'model: reach x, detectable to y' at the arms' length for each arm
    (seed given), the zero-shot PI model and the reference, with reach the
    page's function; an arm's line adds its far recall (beyond the trained
    length) minus the unscaled base's recall in the top octave of its
    trained window (window-resampled 95% interval, other windows)."""
    r = runs.recall
    T, L0 = (_LC_TIERS[_lc_tier_of(runs)][k] for k in ('length', 'trained'))
    n = runs.needles
    top = n[(n['model'] == 'base') & (n['seed'] == 0) & (n['length'] == L0)
            & (n['gap'] > L0 // 2) & (n['gap'] <= L0)]
    arms = set(runs.table['arm']) - {'base'}
    lines = []
    for m in dict.fromkeys(r['model']):
        b = r[(r['model'] == m) & (r['length'] == T) & (r['seed'] == seed)]
        if not len(b) or not (m in arms or m in (
                'PI, zero-shot', *set(runs.ref_needles['model']))):
            continue
        line = (f'{m}: reach {reach(b):,}, detectable to '
                f'{reach(b, 0.25, strict=True):,}')
        far = n[(n['model'] == m) & (n['seed'] == seed) & (n['length'] == T)
                & (n['gap'] > L0) & (n['gap'] <= T)]
        if m in arms and len(far) and len(top):
            line += ('; far − base top octave ' + _interval(
                *recall_unpaired(far, top), 2, sign=True))
        lines.append(line)
    return lines

def show_lc_recall(runs, reach):
    """Under the label: seed 0's reach and detectable range per model
    (reach the page's function; lc_reach_lines), the reference's recall
    per gap bin at the arms' length with 95% Wilson intervals, then the
    two recall panels."""
    if not _lc_head(runs, runs.recall, 'lc-t2-evals'):
        return
    print('\n'.join(lc_reach_lines(runs, reach)))
    r = runs.recall
    T = int(r['length'].max())
    for m in sorted(set(runs.ref_needles['model'])):
        b = r[(r['model'] == m) & (r['length'] == T) & (r['seed'] == 0)]
        cells = [f'{bucket_name(x.gap_lo, x.gap_hi)} '
                 f'{_interval(x.rate, x.lo, x.hi, 2)}' for x in
                 b.sort_values('gap_lo').itertuples()]
        lines = [f'{m} at {_length_name(T)} by gap:']
        for c in cells:  # wrapped between bins at 100 columns
            if len(lines[-1]) + len(c) + 3 > 100:
                lines[-1] += ','
                lines.append(' ')
            lines[-1] += (' ' if lines[-1].endswith(':') or lines[-1] == ' '
                          else ', ') + c
        print('\n'.join(lines))
    plot_lc_recall(runs)

def show_lc_gain(runs):
    """The long-context gain per gap bucket (nats; ±: half the 95%
    interval), as a table and a plot, under the label."""
    gain = runs.gain
    if not _lc_head(runs, gain, 'lc-t2-evals'):
        return
    cell = {(r.model, r.bucket): _pm(r.gain, r.lo, r.hi, 2)
            for r in gain.itertuples()}
    buckets = list(dict.fromkeys(gain['bucket']))
    print(pd.DataFrame(
        [[m] + [cell.get((m, b), '-') for b in buckets]
         for m in dict.fromkeys(gain['model']) if m != 'base'],
        columns=['arm'] + buckets).to_string(index=False))
    plot_lc_gain(gain[gain['model'] != 'base'])

def lc_first_steps(runs, margin):
    """Per arm and seed with a timeline: the tokens at the first periodic
    evaluation at which the loss in the last position bucket is within
    margin of the unscaled base's loss in its trained window on the same
    windows, near recall has a lower 95% Wilson bound of at least 0.5, far
    recall one above 0.25, and the gain beyond the trained length a lower
    bound above 0 ('not reached' if never), from the evaluations at the
    arms' length (a progressive arm's first-stage rows are at its shorter
    window); and the arms and seeds without a timeline."""
    spec = _LC_TIERS[_lc_tier_of(runs)]
    w = runs.zs_windows
    base = w[(w['model'] == 'none') & (w['measure'] == 'nll')
             & (w['bucket'] == bucket_name(0, spec['trained']))
             & (w['window'] < spec['eval_windows'])]
    ref = base['sum'].sum() / base['n'].sum()
    rows = []
    tl = runs.timeline[runs.timeline['length'] == spec['length']]
    for (arm, seed), t in tl.groupby(['arm', 'seed'], sort=False):
        first = lambda ok: (f'{t["tokens"][ok].min() / 1e6:.0f}M'
                            if ok.any() else 'not reached')
        rows.append((arm, seed, first(t['nll_last'] <= ref + margin),
                     first(t['near_lo'] >= 0.5), first(t['far_lo'] > 0.25),
                     first(t['gain_lo'] > 0)))
    have = {(a, s) for a, s, *_ in rows}
    missing = [f'{a} s{s}' for a, s in dict.fromkeys(zip(
        runs.logs['arm'], runs.logs['seed'])) if (a, s) not in have]
    return pd.DataFrame(rows, columns=['arm', 'seed', 'loss', 'near recall',
                                       'far recall', 'gain']), missing

def show_lc_first_steps(runs, margin):
    """lc_first_steps under the label: the table, then the arms without a
    timeline."""
    if not _lc_head(runs, runs.timeline, 'lc-t2-evals'):
        return
    table, missing = lc_first_steps(runs, margin)
    print(table.to_string(index=False))
    if missing:
        print('no timeline (trained before the periodic needle '
              f'evaluation): {", ".join(missing)}')

def show_lc_short(runs):
    """Per arm the val loss at the trained length, the suite mean and near
    recall at the trained length minus the base's, with paired 95%
    intervals, under the label."""
    if not _lc_head(runs, runs.short, 'lc-t2-evals'):
        return
    s = runs.short
    print(pd.DataFrame({
        'arm': s['arm'],
        'Δ val loss': [_interval(*x, 3, sign=True) for x in zip(
            s['Δ val'], s['val lo'], s['val hi'])],
        'Δ suite': [_interval(*x, 3, sign=True) for x in zip(
            s['Δ suite'], s['suite lo'], s['suite hi'])],
        'Δ near recall': [_interval(*x, 2, sign=True) for x in zip(
            s['Δ near'], s['near lo'], s['near hi'])]}).to_string(
        index=False))

def show_lc_seeds(t1, t2, a='YaRN', b='PI'):
    """The T1 table (as show_lc_table, without the context column; the
    number of seeds by each arm) with one more column, recall 32-64 back
    at the base's trained length (the base's row is its control, the arms'
    rows what extension left of it), then a minus b with a t interval over
    the seeds at T1 and at T2 ('T2: pending' while the T2 result is the T1
    stand-in or pending)."""
    if not _lc_head(t1, t1.table, 'lc-t1'):
        return
    t, r = t1.table, t1.recall
    T, L0 = (_LC_TIERS['t1'][k] for k in ('length', 'trained'))
    seeds = t1.needles.groupby('model')['seed'].nunique()
    name = lambda m: (f'{m} ({seeds[m]} seed{"s" if seeds[m] > 1 else ""})'
                      if m in seeds.index and m != 'base' else m)

    def short(m):  # recall 32-64 back at L0, seeds pooled per query
        c = r[(r['model'] == m) & (r['length'] == L0) & (r['gap_lo'] == 32)
              & (r['seed'] == -1)]
        return (_pm(*c[['rate', 'lo', 'hi']].iloc[0], 2) if len(c)
                else '-')
    print(pd.DataFrame({
        'arm': t['arm'].map(name),
        f'val {_length_name(L0)} (Δ)': [
            '-' if math.isnan(v) else f'{_num(v, 3)} ({_num(d, 3, True)})'
            for v, d in zip(t['val'], t['Δ val'])],
        f'NLL {bucket_name(T - L0, T)}': t['NLL last'].map(
            lambda x: _num(x, 3)),
        'near': t['near'].map(lambda x: _num(x, 2)),
        f'far at {_length_name(T)}': t['far'].map(lambda x: _num(x, 2)),
        'reach': t['reach'].map(_tokens),
        f'32-64 at {_length_name(L0)}': t['arm'].map(short)}).to_string(
        index=False))
    for tier, runs in (('T1', t1), ('T2', t2)):
        if _lc_tier_of(runs) != tier.lower() or d2l._label_of(runs):
            print(f'{tier}: pending')
            continue
        d = scheme_difference(runs, a, b)
        print(f'{a} − {b}, {tier} ({len(d["seeds"])} seeds): loss '
              f'{_interval(*d["loss"], 3, True)}\n  near '
              f'{_interval(*d["near"], 2, True)}, far '
              f'{_interval(*d["far"], 2, True)}')
```
