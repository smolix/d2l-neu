# Long Context: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is
never rendered or executed as a notebook: `make lib` only scans it for
`#@save` blocks, and `gen_notebooks.py` excludes it via `LIB_ONLY_FILES`.
It holds the library code of :numref:`sec_lm_long_context`
that the section explains in a sentence but does not show
(docs/code-guide.md 7.1):

- the section's registries: committed results (`PRETRAIN_ARTIFACTS`),
  the campaigns not yet run (`PRETRAIN_PENDING`) and the long regeneration
  jobs (`PRETRAIN_JOBS`), each the page's call written a second time;
- the data (`long_data`, `test_books`): the long half (PG-19 books and long
  peS2o papers), the short replay and the held-out test books;
- the measurement internals: per-target losses of whole and sliding
  windows, repeat gaps, the rare-token set, window-resampled intervals,
  the cached recall scorer (`score_recall`), the FLOP count on the meta
  device, and the effective-context rule;
- the zero-shot evaluation (`lc_zeroshot`) and the arm driver (`lc_arms`:
  one resumable job folder per arm and seed, the length stages, the
  periodic evaluations, the final evaluation and the merge), the D2 probe
  (`lc_probe`), the section's plots, and the displays the page calls
  (`show_lc_*`: the result's label first, then a table at the precision
  the prose quotes).

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble inside its functions (`copy`, `itertools`, `tempfile`,
`warnings`, `transformers`, `torch.utils.flop_counter`).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_long_context`

## Registries

The section's results and inputs. Until a campaign's result is committed
the page shows it as pending (the T2 cells with the T1 arms as their
stand-in) instead of failing; the two shard sets are read from the
builder's local output (`tools/build_pretraining_data.py --stage lc-long`,
`--stage pg19-test`) until their upload.

```{.python .input #long-context-lib-artifacts}
%%tab pytorch
#@save
_LC_PENDING = {'lc-t1', 'lc-zeroshot', 'lc-t2-evals'}
d2l.PRETRAIN_PENDING = d2l.PRETRAIN_PENDING | _LC_PENDING

#@save
d2l.PRETRAIN_ARTIFACTS['lc-zeroshot'] = (  # the zero-shot part
    'lm-lc-evals-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lc-t1'] = (
    'lm-lc-t1-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lc-evals'] = (
    'lm-lc-evals-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lc-long'] = (
    'lm-lc-long-60m-v1', 'shards')
d2l.PRETRAIN_ARTIFACTS['pg19-test'] = (
    'lm-pg19-test-v1', 'shards')
```

The long jobs that `tools/lm_regen.py` runs (one card each): the zero-shot
evaluation, the three T2 arms, their final evaluation and merge, and the
T1 arms (also a notebook regeneration, about 25 minutes).

```{.python .input #long-context-lib-jobs}
%%tab pytorch
#@save
d2l.PRETRAIN_JOBS['lc-t1'] = ('lc_arms', lambda: d2l._lc_t1_call())
d2l.PRETRAIN_JOBS['lc-zeroshot'] = ('lc_zeroshot',
                                    lambda: d2l._lc_zeroshot_call())
d2l.PRETRAIN_JOBS['lc-t2-pi'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-yarn'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-prog'] = ('lc_arms', lambda: d2l._lc_t2_call())
d2l.PRETRAIN_JOBS['lc-t2-evals'] = ('lc_arms', lambda: d2l._lc_t2_call())
```

## The Page's Calls

The configurations the page passes (the load check compares them with the
committed runs), the regeneration name of each T2 arm, the suite and the
evaluation grid of each tier: the trained length of the base, the lengths
of the recall grid, the test books and the items per grid cell of the
final, zero-shot and periodic evaluations.

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
_LC_ARMS = {'PI': dict(scheme='pi', stages=((16_384, 4, 1.0),)),
            'YaRN': dict(scheme='yarn', stages=((16_384, 4, 1.0),)),
            'YaRN-prog': dict(scheme='yarn', stages=((8_192, 2, 0.4),
                                                     (16_384, 4, 0.6)))}

#@save
_LC_ARMS_T1 = {
    'PI': dict(scheme='pi', stages=((4096, 4, 1.0),), seeds=(0, 1, 2)),
    'YaRN': dict(scheme='yarn', stages=((4096, 4, 1.0),), seeds=(0, 1, 2)),
    'YaRN-prog': dict(scheme='yarn', stages=((2048, 2, 0.4),
                                             (4096, 4, 0.6)), seeds=(0,))}

#@save
_LC_T2_NAMES = {'PI': 'lc-t2-pi', 'YaRN': 'lc-t2-yarn',
                'YaRN-prog': 'lc-t2-prog'}

#@save
_LC_TIERS = {
    't2': dict(trained=4096, lengths=(4096, 8192, 12288, 16384), books=64,
               n_items=100, zs_items=100, eval_books=16, eval_items=50),
    't1': dict(trained=1024, lengths=(1024, 2048, 3072, 4096), books=64,
               n_items=100, zs_items=100, eval_books=16, eval_items=50)}

#@save
LC_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)

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
    return dict(base='r2', schemes=_LC_SCHEMES, lengths=(4096, 8192, 16384))

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
`lm-fwedu-replay-12m-v1`), and the test books the first 16,385 tokens of
64 PG-19 test books (16,385 tokens and an EOS each). The long peS2o
papers come from the validation file `validation-00001-of-00002`
(s2orc/valid) and serve here as training data; nothing in this section
evaluates on peS2o, and the held-out peS2o of continued pretraining is the
other validation file (s2ag/valid), so no evaluation text is trained on.
Before their upload the two new shard sets are read from the builder's
local output.

```{.python .input #long-context-lib-data}
%%tab pytorch
#@save
LcData = collections.namedtuple('LcData', ['table', 'shards', 'label'])

#@save
_LC_LOCAL = {'lc-long': os.path.join('..', 'data', 'lm', 'lc-long-v1'),
             'pg19-test': os.path.join('..', 'data', 'lm', 'pg19-test-v1')}

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
    64 test books); table: per part documents, tokens, median length in
    tokens and source."""
    if tier not in ('t1', 't2'):
        raise ValueError(f"tier is 't1' or 't2', not {tier!r}")
    long = _lc_shards('lc-long', 8_000_000 if tier == 't1' else None)
    shards = dict(long=long, short=_lc_replay(tier),
                  test=_lc_shards('pg19-test'))
    return LcData(_lc_parts_table(shards), shards, '')

def test_books(n, T):
    """The first T + 1 tokens of the first n PG-19 test books, as a
    LongTensor (n, T + 1) (64 books of 16,385 tokens at most)."""
    s = _lc_shards('pg19-test')
    if n > s.n_docs or T + 1 > s.doc(0).shape[0] - 1:
        raise ValueError(f'{s.n_docs} test books of '
                         f'{s.doc(0).shape[0] - 1} tokens')
    return torch.from_numpy(np.stack([s.doc(i)[:T + 1].astype(np.int64)
                                      for i in range(n)]))

def _val_windows(T, tokens=None):
    """Windows (n, T + 1) of the FineWeb-Edu validation shard, the first
    tokens (default _LC_VAL_TOKENS) tokens, and its EOS id."""
    with d2l._quiet():
        s = d2l.fwedu_shards('val')
    n = (tokens or _LC_VAL_TOKENS) // T
    W = np.stack([s.read(i * T, T + 1) for i in range(n)]).astype(np.int64)
    return torch.from_numpy(W), s.eos_id
```

## Losses of Windows, Gaps and Intervals

Per-target losses of whole windows (`window_nlls`) and of a sliding short
context (`sliding_nlls`), the gap back to each token's previous
occurrence (`repeat_gaps`), the set of rare tokens the long-context gain
is measured on (`_rare_ids`: beyond rank 2,000 in the unigram counts of
the first 2^25 tokens of the 100M shard, decoding to at least three
letters), and intervals that resample whole windows (`window_ci`).

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

## Recall Items and the Cached Scorer

The keys of `make_recall_task` are two words, an adjective and a noun from
two lists of 48 (2,304 keys, more than the about 1,250 lines a 16K item
holds; at about 13 tokens a line they run out near 30K tokens, and
`recall_haystack` raises instead of returning a shorter item), the values
distinct five-digit numbers. Every (seed, length, depth) has a generator of
its own (`recall_generator`), so a grid cell's items do not depend on the
cells drawn before it. `lc_tokenizer` is the reference tokenizer with its
nominal length of 4,096 (which Transformers warns about on every longer
text) raised past the longest window. `score_recall` scores the items as
`score_choices` does (summed log-probability of each candidate value) but
runs the prompt once: one prefill into the KV cache (2,048 tokens at a
time), the cache repeated for the candidates, and one cached forward of
every candidate's tokens with all their logits.

```{.python .input #long-context-lib-recall}
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
RECALL_HEAD = 'Each line below gives the value of a key.\n'

#@save
def lc_tokenizer(max_len=65_536):
    """The reference tokenizer with its nominal length (4,096) raised to
    max_len tokens."""
    tok = d2l.r2_tokenizer()
    tok.model_max_length = max_len
    return tok

def recall_generator(seed, length, depth):
    """The random generator of the recall items of one (seed, length,
    depth)."""
    return torch.Generator().manual_seed(
        seed * 1_000_003 + length * 1009 + round(1000 * depth))

def _recall_lines(tokenizer, keys, values):
    """The haystack lines and their token counts."""
    lines = [f'The value of {k} is {v}.\n' for k, v in zip(keys, values)]
    return lines, [len(x) for x in d2l._lm_encode(tokenizer, lines)]

def recall_haystack(tokenizer, length, generator, n_pairs=None,
                    min_lines=8, value_tokens=6):
    """(keys, values, lines, start) of one recall item: distinct two-word
    keys and five-digit values, and as many haystack lines as fit length
    tokens after RECALL_HEAD with the query and a value (n_pairs lines if
    given, at least min_lines); start: the token offset of every line and
    of the query."""
    adj, noun = _RECALL_ADJ, _RECALL_NOUN
    pick = torch.randperm(len(adj) * len(noun), generator=generator).tolist()
    keys = [f'{adj[i // len(noun)]} {noun[i % len(noun)]}' for i in pick]
    values = (torch.randperm(90_000, generator=generator)[:len(keys)]
              + 10_000).tolist()
    n_max = min(len(keys), n_pairs or length // 10 + 1)
    lines, sizes = _recall_lines(tokenizer, keys[:n_max], values[:n_max])
    head = len(d2l._lm_encode(tokenizer, [RECALL_HEAD])[0])
    cum = np.cumsum(sizes)
    budget = length - head - max(sizes) - value_tokens
    n = n_pairs or int(np.searchsorted(cum, budget, side='right'))
    if n < min_lines or cum[n - 1] > budget:
        raise ValueError(f'{max(n, min_lines)} lines do not fit {length} '
                         'tokens')
    if n_pairs is None and n == n_max and budget - cum[-1] > max(sizes):
        raise ValueError(f'the {n_max:,} keys fill {head + cum[-1]:,} of '
                         f'{length:,} tokens')
    start = [head] + (head + cum[:n]).tolist()
    return keys[:n], values[:n], lines[:n], start

def _recall_ll(model, tokenizer, task, chunk=2048):
    """Summed log-probabilities (N, C) of every candidate of every item,
    with one prefill of the prompt per item (the cache repeated for the C
    candidates); equal to score_choices(..., norm='sum')."""
    dev = _device_of(model)
    ll = torch.full((len(task['prompts']), len(task['choices'][0])),
                    -math.inf, dtype=torch.float64)
    auto = torch.autocast(dev.type, torch.bfloat16,
                          enabled=dev.type == 'cuda')
    for i, (prompt, choices) in enumerate(zip(task['prompts'],
                                              task['choices'])):
        tail = prompt[prompt.rfind('\n') + 1:]  # the query line
        _, inputs, conts, _, _ = d2l._lm_continuations(
            tokenizer, [tail], [list(choices)])
        ctx = d2l._lm_encode(tokenizer, [prompt])[0]
        head = inputs[0][:len(inputs[0]) - len(conts[0]) + 1]
        assert ctx[-len(head):] == head, 'the query line tokenizes apart'
        m = max(map(len, conts))
        ks, vs = model.init_cache(1, len(ctx) - 1 + m)
        with auto:
            for t in range(0, len(ctx) - 1, chunk):
                X = torch.tensor([ctx[t:min(t + chunk, len(ctx) - 1)]],
                                 device=dev)
                model.forward_cached(X, ks, vs, t)
            X = torch.tensor([[ctx[-1]] + c[:-1] + [0] * (m - len(c))
                              for c in conts], device=dev)
            C = len(conts)
            logits, _, _ = model.forward_cached(
                X, ks.repeat(1, C, 1, 1, 1), vs.repeat(1, C, 1, 1, 1),
                len(ctx) - 1, all_logits=True)
        lp = logits.float().log_softmax(-1)
        for j, c in enumerate(conts):
            ll[i, j] = lp[j, torch.arange(len(c)), torch.tensor(c)].double(
                ).sum().cpu()
    return ll

def score_recall(model, tokenizer, task):
    """Per item (a bool array): whether the gold value scores highest
    among the candidates (summed log-probability, one prefill per item)."""
    ll = _recall_ll(model, tokenizer, task)
    return (ll.argmax(1).numpy() == np.asarray(task['gold']))

def _recall_task(tokenizer, length, depth, n_items, seed=0):
    """make_recall_task, cached on the tokenizer object (tokenizing
    16K-token items takes a while; every model is scored on the same
    items)."""
    cache = tokenizer.__dict__.setdefault('_d2l_recall_tasks', {})
    key = (length, depth, n_items, seed)
    if key not in cache:
        cache[key] = d2l.make_recall_task(tokenizer, length, depth,
                                          n_items=n_items, seed=seed)
    return cache[key]
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

## The Effective Context and the FLOP Check

```{.python .input #long-context-lib-rule}
%%tab pytorch
#@save
def effective_context(recall, gain, threshold=0.5):
    """Per model: the largest evaluated length L whose recall, pooled over
    depths, has a lower 95% Wilson bound of at least threshold and whose
    long-context gain in the gap bucket (L/2, L] has a lower 95% bound
    above 0; 0 where no length clears both. Seeds share the items, so an
    item counts once, with its recall averaged over the seeds."""
    out = {}
    for model, r in recall.groupby('model', sort=False):
        ok = [0]
        for L, rl in r.groupby('length'):
            item = rl.groupby(['depth', 'item'])['correct'].mean()
            lo = d2l.wilson_ci(float(item.sum()), len(item))[0]
            g = gain[(gain['model'] == model) & gain['rule'] & (
                gain['bucket'] == bucket_name(L // 2, L))]
            if lo >= threshold and len(g) and g['lo'].iloc[0] > 0:
                ok.append(int(L))
        out[model] = max(ok)
    return pd.Series(out, name='effective', dtype='int64')

def _length_name(L):
    """'16K', '1.5K' or '256' for a length in tokens."""
    return f'{L / 1024:g}K' if L >= 1024 and L % 512 == 0 else str(L)

def _context_name(L, lengths):
    return _length_name(L) if L else f'below {_length_name(min(lengths))}'

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

`_lc_eval` evaluates one model on the books (each window as long as the
longest evaluated length): the loss per position bucket of the trained
length, the long-context gain against a sliding context of a quarter of
the trained length (display buckets of doubling gaps from there, and the
rule buckets (L/2, L] of the effective-context rule), recall over the
length-by-depth grid, and on request the val loss at the trained length
(per window, for paired differences) and the suite. It returns per-window
and per-item rows; the summaries and intervals are computed from them,
pooling seeds by window or item.

```{.python .input #long-context-lib-eval}
%%tab pytorch
#@save
LcZeroShot = collections.namedtuple(
    'LcZeroShot', ['table', 'position', 'recall', 'gain', 'label'])

#@save
def _lc_gain_pairs(trained, T, lengths):
    """The display gap buckets (doubling from a quarter of the trained
    length to T) and the rule buckets (L/2, L], with a rule flag."""
    short, out = trained // 4, []
    hi = short
    while hi < T:
        out.append((hi, 2 * hi, False))
        hi *= 2
    return out + [(L // 2, L, True) for L in lengths]

def _lc_eval(model, tier, books, lengths, n_items, tokenizer, val=False,
             suite=None):
    """Rows of one model: windows (measure, bucket, window, sum, n) for
    'nll' (position buckets), 'gain', 'rule' and 'val'; recall (length,
    depth, item, correct); items (task, item_id, score) of the suite."""
    L0, T = _LC_TIERS[tier]['trained'], books.shape[1] - 1
    nll = window_nlls(model, books)
    rows = []
    for lo in range(0, T, L0):
        for w in range(len(books)):
            rows.append(('nll', bucket_name(lo, lo + L0), w,
                         float(nll[w, lo:lo + L0].sum()), L0))
    pairs = _lc_gain_pairs(L0, T, lengths)
    gain = sliding_nlls(model, books, L0 // 4) - nll
    s, c = gain_sums(gain, repeat_gaps(books, _rare_ids()),
                      [p[:2] for p in pairs], L0 // 4)
    for k, (lo, hi, rule) in enumerate(pairs):
        for w in range(len(books)):
            rows.append(('rule' if rule else 'gain', bucket_name(lo, hi), w,
                         float(s[w, k]), float(c[w, k])))
    if val:
        W, eos = _val_windows(L0)
        v = window_nlls(model, W, eos=eos, batch=max(1, 8192 // L0))
        for w in range(len(W)):
            ok = ~torch.isnan(v[w])
            rows.append(('val', 'val', w, float(v[w][ok].sum()),
                         float(ok.sum())))
    rec = []
    for L in lengths:
        for d in LC_DEPTHS:
            task = _recall_task(tokenizer, L, d, n_items)
            ok = score_recall(model, tokenizer, task)
            rec += [(L, d, i, bool(x)) for i, x in enumerate(ok)]
    items = (pd.DataFrame(columns=['task', 'item_id', 'score'])
             if suite is None else d2l._suite_items(suite, model, tokenizer))
    return (pd.DataFrame(rows, columns=['measure', 'bucket', 'window', 'sum',
                                        'n']),
            pd.DataFrame(rec, columns=['length', 'depth', 'item',
                                       'correct']), items)

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
    g = pd.concat([_lc_summary(windows, 'gain').assign(rule=False),
                   _lc_summary(windows, 'rule').assign(rule=True)],
                  ignore_index=True)
    return g.rename(columns={'value': 'gain'})[
        ['model', 'bucket', 'rule', 'gain', 'lo', 'hi', 'n']]

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

## Zero-Shot

`lc_zeroshot` evaluates the base model under each scheme without training
(64 books, recall at each length over five depths, 100 items each). The
regeneration `lc-zeroshot` writes its result as a part of
`lm-lc-evals-v1`, which the T2 merge folds in; a normal run reads it from
the committed evaluations (checking the page's configuration against the
stored one), and while those are pending shows the T1 zero-shot table that
`lc-t1` computes on the t1 checkpoint.

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

def lc_reference_model():
    """The reference model at step 6000 (trained at 4K) on the GPU if
    there is one; while it is pending, the t1 checkpoint (trained at 1K)
    under a printed label."""
    if _LC_BASES['r2'][0] in d2l.DATA_HUB:
        return d2l._load_bf16(_lc_base_path('r2'), device=_lc_device())
    model = d2l._load_bf16(_lc_base_path('t1'), device=_lc_device())
    print('[reference run pending; the t1 checkpoint, trained at 1K]')
    return model

def _lc_tier(base):
    if base not in _LC_BASES:
        raise ValueError(f"base is 'r2' or 't1', not {base!r}")
    return 't2' if base == 'r2' else 't1'

def _lc_zs_columns(trained, lengths):
    T = max(lengths)
    return (['scheme'] + [f'NLL {bucket_name(lo, lo + trained)}'
                          for lo in range(0, T, trained)]
            + [f'recall {_length_name(L)}' for L in lengths])

def _lc_zeroshot_compute(base, schemes, lengths):
    """LcZeroShot of base under each scheme (no training)."""
    tier = _lc_tier(base)
    spec, dev, tok = _LC_TIERS[tier], _lc_device(), lc_tokenizer()
    books = test_books(spec['books'], max(lengths))
    windows, recall = [], []
    for name, sc in schemes.items():
        model = d2l._load_bf16(_lc_base_path(base), device=dev)
        _lc_set_scheme(model, sc['scheme'], sc['s'], spec['trained'],
                       sc.get('mscale'))
        w, r, _ = _lc_eval(model, tier, books, lengths, spec['zs_items'],
                           tok)
        windows.append(w.assign(model=name, seed=0))
        recall.append(r.assign(model=name, seed=0))
        del model
        d2l.lm_free()
    windows = pd.concat(windows, ignore_index=True)
    recall = pd.concat(recall, ignore_index=True)
    position = _lc_position(windows)
    acc = recall.groupby(['model', 'length'], sort=False)['correct'].mean()
    rows = [[name] + list(position[position['model'] == name]['nll'])
            + [acc[name, L] for L in lengths] for name in schemes]
    table = pd.DataFrame(rows, columns=_lc_zs_columns(spec['trained'],
                                                      lengths))
    cols = d2l._LC_COLUMNS['recall']
    return LcZeroShot(table, position, recall[cols], _lc_gain(windows), '')

def _lc_zs_path():
    key = d2l.PRETRAIN_ARTIFACTS['lc-zeroshot'][0]
    return os.path.join(d2l._rerun_dir(),
                        f'{d2l._next_key(key)}.lc-zeroshot.json.gz')

def _lc_zs_as_tuple(res):
    """The zero-shot fields of a stored, pending or fallback result."""
    if isinstance(res, pd.DataFrame):  # pending: empty tables
        return d2l._as_tuple(LcZeroShot, res, {
            'table': [], 'position': _LC_COLUMNS['position'],
            'recall': _LC_COLUMNS['recall'], 'gain': _LC_COLUMNS['gain']})
    return LcZeroShot(res.zeroshot, res.zs_position, res.zs_recall,
                      res.zs_gain, res.label)

def lc_zeroshot(base, schemes, lengths):
    """The base model ('r2' or 't1') under each RoPE scheme without
    training: LcZeroShot(table, position, recall, gain, label). schemes =
    {name: dict(scheme, s[, mscale])}; table: per scheme the loss per
    position bucket of the trained length on 64 test books and recall at
    each length (five depths, 100 items each). Regeneration lc-zeroshot;
    otherwise the zero-shot part of the committed lm-lc-evals-v1; while
    that is pending, the T1 zero-shot table, labelled so (show_lc_zeroshot
    prints the label)."""
    config = d2l._jsonable(dict(base=base, schemes=schemes, lengths=lengths))
    rerun = d2l._regenerating()
    if 'lc-zeroshot' in rerun:
        res = _lc_zeroshot_compute(base, schemes, lengths)
        meta = dict(config=config, label=d2l._label_of(res))
        d2l.lm_save_rerun(os.path.basename(_lc_zs_path()),
                          lambda p: d2l._write_result(p, 'json-table', res,
                                                      meta),
                          rerun_dir=d2l._rerun_dir())
        return res
    key = d2l.PRETRAIN_ARTIFACTS['lc-zeroshot'][0]
    if key in d2l.DATA_HUB:
        res = d2l.load_artifact('lc-evals')
        d = d2l._config_diff(config, d2l._jsonable(res.zs_config))
        if d:
            raise ValueError(f'page configuration differs from the committed '
                             f'run {key}: zero-shot{d[0]} {d[1]!r} != '
                             f'{d[2]!r}')
        return _lc_zs_as_tuple(res)
    if rerun or 'lc-zeroshot' in d2l.PRETRAIN_PENDING:
        res = d2l._pending_result('lc-zeroshot', 'lc-t1')
        return _lc_zs_as_tuple(res)
    raise KeyError('lc-zeroshot is not committed yet')
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
which a resumed run derives from its step. Every 64 steps (and before the
first update) a bf16 copy is evaluated: loss by position bucket and
long-context gain on 16 test books, recall at the longest length over the
five depths (50 items each). The merge evaluates the base (base RoPE,
the long window accepted) and every arm's final weights on 64 books, the
full recall grid, the val loss at the trained length and the suite.

```{.python .input #long-context-lib-arms}
%%tab pytorch
#@save
LcRuns = collections.namedtuple(
    'LcRuns', ['table', 'logs', 'position', 'recall', 'gain', 'timeline',
               'short', 'windows', 'items', 'zeroshot', 'zs_position',
               'zs_recall', 'zs_gain', 'zs_config', 'label'])

#@save
_LC_COLUMNS = {
    'table': ['arm', 'val loss', 'Δ val', 'NLL 12-16K', 'recall 4K',
              'recall 16K', '95% lo', '95% hi', 'effective'],
    'logs': ['arm', 'seed', 'step', 'tokens', 'stage', 'lr', 'loss',
             'grad_norm', 'tok_s'],
    'position': ['model', 'bucket', 'nll', 'lo', 'hi'],
    'recall': ['model', 'seed', 'length', 'depth', 'item', 'correct'],
    'gain': ['model', 'bucket', 'rule', 'gain', 'lo', 'hi', 'n'],
    'timeline': ['arm', 'seed', 'step', 'tokens', 'nll_first', 'nll_last',
                 'recall_k', 'recall_n', 'recall_lo', 'gain', 'gain_lo'],
    'short': ['arm', 'Δ val', 'val lo', 'val hi', 'Δ suite', 'suite lo',
              'suite hi'],
    'windows': ['model', 'seed', 'measure', 'bucket', 'window', 'sum', 'n'],
    'items': ['model', 'seed', 'task', 'item_id', 'score'],
    'zeroshot': [], 'zs_position': ['model', 'bucket', 'nll', 'lo', 'hi'],
    'zs_recall': ['model', 'seed', 'length', 'depth', 'item', 'correct'],
    'zs_gain': ['model', 'bucket', 'rule', 'gain', 'lo', 'hi', 'n']}

#@save
_LC_STATE_EVERY = 50

#@save
def _lc_slug(arm):
    return arm.lower().replace(' ', '-')

def _lc_configs(arms, base, tokens, seq_len, batch_tokens, peak_lr, warmup,
                decay_frac, long_share):
    """One JSON-able configuration per (arm, seed): the stages (window T,
    scale s, steps), the schedule and the data shares."""
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
                eval_every=_LC_EVAL_EVERY)
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
    """The record of an arm in its job folder: configuration and base."""
    return dict(config=d2l._jsonable(cfg),
                base=[cfg['base'], d2l._file_id(_lc_base_path(cfg['base']))])

def _bf16_copy(model):
    import copy
    return copy.deepcopy(model).to(torch.bfloat16).eval()

def _lc_timeline_row(model, cfg, step, tokenizer):
    """The periodic evaluation of a bf16 copy of model: losses of the first
    and last position buckets, recall at the longest length (pooled over
    depths) with its lower Wilson bound, and the gain in the rule bucket
    of the longest length with its lower bound."""
    spec, T = _LC_TIERS[cfg['tier']], cfg['seq_len']
    ev = _bf16_copy(model)
    w, r, _ = _lc_eval(ev, cfg['tier'], test_books(spec['eval_books'], T),
                       (T,), spec['eval_items'], tokenizer)
    del ev
    d2l.lm_free()
    nll = w[w['measure'] == 'nll']
    first, last = nll['bucket'].iloc[0], nll['bucket'].iloc[-1]
    ratio = lambda x: float(x['sum'].sum() / x['n'].sum())
    rule = w[(w['measure'] == 'rule') & (w['bucket'] ==
                                         bucket_name(T // 2, T))]
    g, g_lo, _ = window_ci(rule['sum'], rule['n'])
    k, n = int(r['correct'].sum()), len(r)
    return dict(arm=cfg['arm'], seed=cfg['seed'], step=step,
                tokens=step * cfg['global_batch'] * T,
                nll_first=ratio(nll[nll['bucket'] == first]),
                nll_last=ratio(nll[nll['bucket'] == last]), recall_k=k,
                recall_n=n, recall_lo=d2l.wilson_ci(k, n)[0], gain=g,
                gain_lo=g_lo)

def _lc_read_timeline(job):
    path = os.path.join(job, 'timeline.jsonl')
    if not os.path.exists(path):
        return pd.DataFrame(columns=_LC_COLUMNS['timeline'])
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return pd.DataFrame(rows).drop_duplicates('step', keep='last')

def _lc_train(model, stream, cfg, job, stop=None):
    """Train one arm (resumable in job; at most `stop` steps): the log
    rows; the periodic evaluations (unless cfg['evaluate'] is False) go to
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
    evaluate = cfg.get('evaluate', True)
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
                timeline=_lc_read_timeline(job), gib=info['gib'],
                tok_s=float(steady.median()) if len(steady) else math.nan,
                label='')
```

## The Merge

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

def _lc_short(windows, items, arms):
    """Per arm: the val loss at the trained length and the suite mean
    minus the base's, with 95% intervals (paired by window and by item;
    seeds averaged per window and item)."""
    val = windows[windows['measure'] == 'val'].groupby(
        ['model', 'window'])[['sum', 'n']].sum()
    per = (val['sum'] / val['n']).unstack('model')
    score = items.groupby(['model', 'task', 'item_id'])['score'].mean()
    rows = []
    for arm in arms:
        d, lo, hi = window_ci(per[arm] - per['base'], np.ones(len(per)))
        s = _suite_delta(score[arm].reset_index(),
                         score['base'].reset_index())
        rows.append((arm, d, lo, hi, *s))
    return pd.DataFrame(rows, columns=_LC_COLUMNS['short'])

def _lc_table(tier, recall, position, windows, gain, arms):
    """Per model (the base, then each arm): val loss at the trained length
    and its difference from the base, the loss in the last position
    bucket, recall at the trained and the longest length (pooled over
    depths and seeds) with a 95% Wilson interval (one seed) or the range
    over seeds, and the effective context."""
    spec = _LC_TIERS[tier]
    L0, Lmax = spec['trained'], max(spec['lengths'])
    val = windows[windows['measure'] == 'val'].groupby('model')[
        ['sum', 'n']].sum()
    val = val['sum'] / val['n']
    eff = effective_context(recall, gain)
    last = position['bucket'].iloc[-1]
    multi = recall.groupby('model')['seed'].nunique().max() > 1
    names = ['min', 'max'] if multi else ['95% lo', '95% hi']
    rows = []
    for model in ['base'] + list(arms):
        r = recall[recall['model'] == model]
        at = lambda L: r[r['length'] == L]['correct']
        long = r[r['length'] == Lmax]
        if multi:
            per_seed = long.groupby('seed')['correct'].mean()
            lo, hi = float(per_seed.min()), float(per_seed.max())
        else:
            lo, hi = d2l.wilson_ci(int(long['correct'].sum()), len(long))
        p = position[(position['model'] == model) &
                     (position['bucket'] == last)]['nll']
        rows.append((model, val[model], val[model] - val['base'],
                     float(p.iloc[0]), at(L0).mean(), at(Lmax).mean(), lo,
                     hi, _context_name(int(eff.get(model, 0)),
                                       spec['lengths'])))
    return pd.DataFrame(rows, columns=[
        'arm', 'val loss', 'Δ val', f'NLL {last}',
        f'recall {_length_name(L0)}', f'recall {_length_name(Lmax)}',
        *names, 'effective'])

def _lc_merge(parts, cfgs, zs, zs_config):
    """LcRuns from the arms' parts {(arm, seed): part} and the zero-shot
    result: the base (base RoPE, the long window accepted) and every arm's
    final weights on the full evaluation."""
    cfg0 = next(iter(cfgs.values()))
    tier, T = cfg0['tier'], cfg0['seq_len']
    spec, dev, tok = _LC_TIERS[tier], _lc_device(), lc_tokenizer()
    books = test_books(spec['books'], T)
    suite = d2l._cpt_suite(LC_TASKS, n_max=_LC_SUITE_N, name='lc')
    models = [('base', 0, None)] + [(a, s, c) for (a, s), c in cfgs.items()]
    windows, recall, items = [], [], []
    for name, seed, cfg in models:
        if cfg is None:
            model = d2l._load_bf16(_lc_base_path(cfg0['base']), device=dev)
            _lc_set_scheme(model, 'none', T / spec['trained'],
                           spec['trained'])
        else:
            if not d2l._cpt_check_job(_lc_job(cfg), _lc_identity(cfg)):
                raise FileNotFoundError(f'{_lc_job(cfg)}: not done')
            model = d2l._load_bf16(os.path.join(
                _lc_job(cfg), 'ckpt', f's{cfg["steps"]}.safetensors'),
                device=dev)
        w, r, it = _lc_eval(model, tier, books, spec['lengths'],
                            spec['n_items'], tok, val=True, suite=suite)
        windows.append(w.assign(model=name, seed=seed))
        recall.append(r.assign(model=name, seed=seed))
        items.append(it.assign(model=name, seed=seed))
        del model
        d2l.lm_free()
    C = _LC_COLUMNS
    windows = pd.concat(windows, ignore_index=True)[C['windows']]
    recall = pd.concat(recall, ignore_index=True)[C['recall']]
    items = pd.concat(items, ignore_index=True)[C['items']]
    position, gain = _lc_position(windows), _lc_gain(windows)
    arms = list(dict.fromkeys(a for a, _ in cfgs))
    table = _lc_table(tier, recall, position, windows, gain, arms)
    zs_pos = zs.position[zs.position['model'].isin(['PI', 'YaRN'])]
    position = pd.concat([zs_pos.assign(model=zs_pos['model'] +
                                        ', zero-shot'), position],
                         ignore_index=True)
    parts = [parts[k] for k in cfgs]
    return LcRuns(
        table, pd.concat([p['log'] for p in parts], ignore_index=True)[
            C['logs']], position, recall, gain,
        pd.concat([p['timeline'] for p in parts], ignore_index=True)[
            C['timeline']], _lc_short(windows, items, arms), windows, items,
        zs.table, zs.position, zs.recall, zs.gain, zs_config, '')

def _lc_zs_for_merge():
    """The zero-shot part for the T2 merge: the file lc-zeroshot wrote, or
    computed here (about 30 minutes) when that regeneration was not run."""
    call = _lc_zeroshot_call()
    config = d2l._jsonable(call)
    path = _lc_zs_path()
    if os.path.exists(path):
        res = d2l._read_result(path, 'json-table')
        d2l._check_config(res, config, os.path.basename(path))
        return LcZeroShot(res.table, res.position, res.recall, res.gain,
                          ''), config
    return _lc_zeroshot_compute(**call), config

def _lc_t1_compute(cfgs):
    """The T1 campaign in one process: zero-shot on the t1 checkpoint at
    the T2 lengths over 4, every arm and seed, and the merge."""
    call = dict(_lc_zeroshot_call(), base='t1')
    call['lengths'] = tuple(L // 4 for L in call['lengths'])
    zs = _lc_zeroshot_compute(**call)
    parts = {k: _lc_arm_run(c) for k, c in cfgs.items()}
    return _lc_merge(parts, cfgs, zs, d2l._jsonable(call))

def lc_arms(arms, base, tokens, seq_len, batch_tokens, peak_lr=1e-4,
            warmup=20, decay_frac=0.15, long_share=0.5):
    """Extend base ('r2' at 4K or 't1' at 1K) to seq_len, one run per arm
    and seed, and evaluate it: LcRuns(table, logs, position, recall, gain,
    timeline, short, windows, items, zeroshot, zs_position, zs_recall,
    zs_gain, zs_config, label). arms = {name: dict(scheme, stages[,
    seeds, mscale])}, stages a tuple of (window T, scale s, share of the
    tokens). base 'r2' (T2: one regeneration per arm, lc-t2-pi, -yarn,
    -prog, merged by lc-t2-evals) or 't1' (the regeneration lc-t1).
    Otherwise the committed result; while the T2 arms are pending, the T1
    arms, labelled so (every show_lc_* display prints the label)."""
    config = dict(arms=arms, base=base, tokens=tokens, seq_len=seq_len,
                  batch_tokens=batch_tokens, peak_lr=peak_lr, warmup=warmup,
                  decay_frac=decay_frac, long_share=long_share)
    cfgs = _lc_configs(arms, base, tokens, seq_len, batch_tokens, peak_lr,
                       warmup, decay_frac, long_share)
    if base == 't1':
        res = d2l._pretrain_result('lc-t1', lambda: _lc_t1_compute(cfgs),
                                   config)
    else:
        if set(arms) - set(_LC_T2_NAMES) or any(s for _, s in cfgs):
            raise ValueError(f'T2 arms are {sorted(_LC_T2_NAMES)}, seed 0')
        parts = {_LC_T2_NAMES[a]: (lambda c=c: _lc_arm_run(c), c)
                 for (a, _), c in cfgs.items()}
        names = {_LC_T2_NAMES[a]: (a, s) for a, s in cfgs}

        def merge(results):
            zs, zs_config = _lc_zs_for_merge()
            return _lc_merge({names[k]: d2l._part(v) for k, v in
                              results.items()}, cfgs, zs, zs_config)
        res = d2l._pretrain_merged('lc-t2-evals', parts, merge, config,
                                   fallback='lc-t1')
    return d2l._as_tuple(LcRuns, res, _LC_COLUMNS)

def lc_probe(micro_batches=(1, 2), steps=10, warmup=3):
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
                         arm_hours=_lc_configs(**_lc_t2_call())[arm, 0][
                             'steps'] * t / 3600))
        del model
        torch._dynamo.reset()
        d2l.lm_free()
    return pd.DataFrame(rows)
```

## Plots and the Scheme Difference

```{.python .input #long-context-lib-plots}
%%tab pytorch
#@save
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

def plot_lc_position(position):
    """Loss by position bucket with 95% intervals, one series per model."""
    if not len(position):
        return
    buckets = list(dict.fromkeys(position['bucket']))
    series = {m: tuple(g.set_index('bucket').reindex(buckets)[k].to_numpy()
                       for k in ('nll', 'lo', 'hi'))
              for m, g in position.groupby('model', sort=False)}
    axes = d2l.plot_intervals(buckets, series, 'position bucket',
                              'loss (nats)')
    axes.legend(loc='upper left', bbox_to_anchor=(1.01, 1))

def plot_recall_grid(recall, models=('PI', 'YaRN'), threshold=0.5):
    """Recall over length (columns) and depth (rows) for each model on
    one 0-1 scale, with the threshold as a contour."""
    if not len(recall):
        return
    lengths = sorted(recall['length'].unique())
    depths = sorted(recall['depth'].unique())
    acc = recall.groupby(['model', 'length', 'depth'])['correct'].mean()
    M = np.array([[[acc.get((m, L, d), np.nan) for L in lengths]
                   for d in depths] for m in models])
    d2l.show_heatmaps(torch.tensor(M)[None], 'length', 'depth',
                      titles=list(models), figsize=(5, 2.5), cmap='Blues',
                      xticklabels=[_length_name(L) for L in lengths],
                      yticklabels=[f'{d:.1f}' for d in depths])
    for ax in d2l.plt.gcf().axes:  # recall on 0-1 in every panel and bar
        for image in ax.images:
            image.set_clim(0, 1)
    for ax, Z in zip(d2l.plt.gcf().axes, M):
        if np.nanmin(Z) < threshold < np.nanmax(Z):
            ax.contour(Z, levels=[threshold], colors='black',
                       linewidths=1.5)

def plot_lc_gain(gain):
    """Long-context gain against the gap bucket with 95% intervals."""
    g = gain[~gain['rule'].astype(bool)]
    if not len(g):
        return
    buckets = list(dict.fromkeys(g['bucket']))
    series = {m: tuple(x.set_index('bucket').reindex(buckets)[k].to_numpy()
                       for k in ('gain', 'lo', 'hi'))
              for m, x in g.groupby('model', sort=False)}
    axes = d2l.plot_intervals(buckets, series,
                              'gap to the previous occurrence', 'gain (nats)')
    axes.legend(loc='upper left', bbox_to_anchor=(1.01, 1))

def scheme_difference(runs, a='YaRN', b='PI'):
    """a minus b on recall at the longest length (pooled over depths) and
    on the loss in the last position bucket, with 95% intervals: over
    seeds (t interval) when both arms have two or more, else over the
    items and windows of the single run (paired resampling). A dict:
    recall, recall_lo, recall_hi, nll, nll_lo, nll_hi, method."""
    import warnings
    r, w = runs.recall, runs.windows
    Lmax = r['length'].max()
    last = w[w['measure'] == 'nll']['bucket'].iloc[-1]
    seeds = sorted(set(r[r['model'] == a]['seed'])
                   & set(r[r['model'] == b]['seed']))
    rl = r[r['length'] == Lmax]
    wl = w[(w['measure'] == 'nll') & (w['bucket'] == last)]
    if len(seeds) >= 2:
        acc = rl.groupby(['model', 'seed'])['correct'].mean()
        nll = wl.groupby(['model', 'seed'])[['sum', 'n']].sum()
        nll = nll['sum'] / nll['n']
        dr = [acc[a, s] - acc[b, s] for s in seeds]
        dn = [nll[a, s] - nll[b, s] for s in seeds]
        (mr, hr), (mn, hn) = d2l.t_interval(dr), d2l.t_interval(dn)
        return dict(recall=mr, recall_lo=mr - hr, recall_hi=mr + hr,
                    nll=mn, nll_lo=mn - hn, nll_hi=mn + hn,
                    method=f't interval, {len(seeds)} seeds')
    pick = lambda df, m, keys: df[df['model'] == m].sort_values(keys)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = d2l.paired_bootstrap(
            pick(rl, a, ['depth', 'item'])['correct'].to_numpy(float),
            pick(rl, b, ['depth', 'item'])['correct'].to_numpy(float),
            n_boot=2000)
    wa, wb = pick(wl, a, ['window']), pick(wl, b, ['window'])
    mn, lo, hi = window_ci(wa['sum'].to_numpy() / wa['n'].to_numpy()
                           - wb['sum'].to_numpy() / wb['n'].to_numpy(),
                           np.ones(len(wa)))
    return dict(recall=res.mean, recall_lo=res.ci[0], recall_hi=res.ci[1],
                nll=mn, nll_lo=lo, nll_hi=hi,
                method='single run, paired resampling')
```

## What the Page Prints

Every display of a T2 result prints the result's label first, so the T1
stand-in (`T2 run pending; showing the T1 configuration`) never passes for
the 16K run, and prints one `[... pending]` line instead of an empty table.
The numbers carry the precision the prose quotes: losses at three
decimals, recall at two (500 items per length give a Wilson half-width of
about 0.04), intervals as one `[lo, hi]` column. `lc_first_steps` compares
the periodic evaluations (the first `eval_books` test books) with the
base's loss on the same books, since the mean of 16 books differs from
that of 64 by about the margin itself.

```{.python .input #long-context-lib-show}
%%tab pytorch
#@save
def _lc_tier_of(runs):
    """'t1' or 't2' from the longest length of a result's recall grid;
    None for a pending result (no rows)."""
    if not len(runs.recall):
        return None
    L = int(runs.recall['length'].max())
    return next((t for t, s in _LC_TIERS.items() if max(s['lengths']) == L),
                None)

def _lc_head(res, table, what):
    """Print res's label (a stand-in or pending result) and, when table has
    no rows and res no label, '[what pending]'; whether to show table."""
    d2l._show_label(res)
    if not len(table) and not d2l._label_of(res):
        print(f'[{what} pending]')
    return bool(len(table))

def _num(x, digits, sign=False):
    """x at digits decimals (never '-0.00'), with its sign if sign."""
    return f'{round(float(x), digits) + 0:{"+" if sign else ""}.{digits}f}'

def _interval(x, lo, hi, digits, sign=False):
    """'x [lo, hi]' at digits decimals."""
    return (f'{_num(x, digits, sign)} [{_num(lo, digits, sign)}, '
            f'{_num(hi, digits, sign)}]')

def show_lc_zeroshot(zs):
    """The zero-shot table (loss per position bucket; recall per length
    with its 95% Wilson half-width) and the loss by position with 95%
    intervals, under the result's label."""
    if not _lc_head(zs, zs.table, 'lc-zeroshot'):
        return
    t = zs.table.copy()
    for c in [c for c in t.columns if c.startswith('NLL ')]:
        t[c] = t[c].map(lambda x: _num(x, 2))
    acc = zs.recall.groupby(['model', 'length'])['correct'].agg(
        ['sum', 'count'])
    for L in sorted(zs.recall['length'].unique()):
        cells = []
        for name in t['scheme']:
            k, n = (int(v) for v in acc.loc[(name, L)])
            lo, hi = d2l.wilson_ci(k, n)
            cells.append(f'{k / n:.2f} ± {(hi - lo) / 2:.2f}')
        t[f'recall {_length_name(L)}'] = cells
    print(t.to_string(index=False))
    plot_lc_position(zs.position)

def show_lc_table(runs):
    """The arms' table: seeds, val loss at the trained length and its change
    from the base, the loss in the last position bucket, recall at the
    trained and the longest length with its 95% interval (Wilson for one
    run, the range over seeds for several) and the effective context."""
    if not _lc_head(runs, runs.table, 'arms'):
        return
    t = runs.table
    multi = 'min' in t.columns
    lo, hi = ('min', 'max') if multi else ('95% lo', '95% hi')
    seeds = runs.recall.groupby('model')['seed'].nunique()
    out = pd.DataFrame({'arm': t['arm'],
                        'seeds': [int(seeds.get(a, 0)) for a in t['arm']]})
    for c in t.columns[1:4]:  # val loss, Δ val, last-bucket NLL
        out[c] = t[c].map(lambda x: _num(x, 3))
    for c in [c for c in t.columns if c.startswith('recall ')]:
        out[c] = t[c].map(lambda x: _num(x, 2))
    out['seed range' if multi else '95% CI'] = [
        f'[{_num(a, 2)}, {_num(b, 2)}]' for a, b in zip(t[lo], t[hi])]
    out['effective'] = t['effective']
    print(out.to_string(index=False))

def show_lc_position(runs):
    """The loss by position bucket of every model, under the label."""
    if _lc_head(runs, runs.position, 'position'):
        plot_lc_position(runs.position)

def show_lc_recall(runs, effective, threshold=0.5):
    """Per model the effective context and the worst depth at the longest
    length, then the recall grids of PI and YaRN, under the label."""
    if not _lc_head(runs, runs.recall, 'recall'):
        return
    r = runs.recall
    lengths = sorted(r['length'].unique())
    grid = r.groupby(['model', 'length', 'depth'])['correct'].mean()
    rows = []
    for model, L in effective.items():
        worst = grid[model, lengths[-1]]
        rows.append((model, _context_name(int(L), lengths),
                     f'{worst.idxmin():.1f} ({worst.min():.0%})'))
    print(pd.DataFrame(rows, columns=[
        'model', 'effective', f'worst depth at {_length_name(lengths[-1])}'
    ]).to_string(index=False))
    plot_recall_grid(r, models=('PI', 'YaRN'), threshold=threshold)

def show_lc_gain(runs):
    """The long-context gain per gap bucket with 95% intervals (nats), as a
    table and a plot, under the label."""
    gain = runs.gain[~runs.gain['rule'].astype(bool)]
    if not _lc_head(runs, gain, 'gain'):
        return
    cell = {(r.model, r.bucket): _interval(r.gain, r.lo, r.hi, 2)
            for r in gain.itertuples()}
    buckets = list(dict.fromkeys(gain['bucket']))
    print(pd.DataFrame(
        [[m] + [cell.get((m, b), '') for b in buckets]
         for m in dict.fromkeys(gain['model'])],
        columns=['model'] + [f'gap {b}' for b in buckets]).to_string(
        index=False))
    plot_lc_gain(runs.gain)

def lc_first_steps(runs, margin, threshold=0.5):
    """Per arm (its first seed): the first evaluated step at which the loss
    in the last position bucket is within margin of the base's loss in its
    trained window on the same books, recall at the longest length has a
    lower 95% bound of at least threshold, and the gain at the longest gaps
    a lower bound above 0 ('not reached' if never)."""
    spec = _LC_TIERS[_lc_tier_of(runs)]
    w = runs.windows
    base = w[(w['model'] == 'base') & (w['measure'] == 'nll')
             & (w['bucket'] == bucket_name(0, spec['trained']))
             & (w['window'] < spec['eval_books'])]
    ref = base['sum'].sum() / base['n'].sum()
    tl = runs.timeline[runs.timeline['seed'] == 0]
    rows = []
    for arm, t in tl.groupby('arm', sort=False):
        rows.append([arm] + [str(t['step'][ok].min()) if ok.any()
                             else 'not reached' for ok in (
            t['nll_last'] <= ref + margin, t['recall_lo'] >= threshold,
            t['gain_lo'] > 0)])
    return pd.DataFrame(rows, columns=['arm', 'perplexity', 'recall',
                                       'gain'])

def show_lc_first_steps(runs, margin, threshold=0.5):
    """lc_first_steps under the label."""
    if _lc_head(runs, runs.timeline, 'timeline'):
        print(lc_first_steps(runs, margin, threshold).to_string(index=False))

def show_lc_short(runs):
    """Per arm the val loss at the trained length and the suite mean minus
    the base's, with paired 95% intervals, under the label."""
    if not _lc_head(runs, runs.short, 'short-context cost'):
        return
    s = runs.short
    print(pd.DataFrame({
        'arm': s['arm'],
        'Δ val loss (nats)': [_interval(*x, 3, sign=True) for x in zip(
            s['Δ val'], s['val lo'], s['val hi'])],
        'Δ suite accuracy': [_interval(*x, 3, sign=True) for x in zip(
            s['Δ suite'], s['suite lo'], s['suite hi'])]}).to_string(
        index=False))

def show_scheme_difference(runs, tier, a='YaRN', b='PI'):
    """a minus b (scheme_difference) on recall at the longest length and on
    the loss in the last position bucket, in three lines; 'T2: pending'
    when runs is not a result of tier ('T1' or 'T2'), such as the T1
    stand-in for the T2 arms."""
    if _lc_tier_of(runs) != tier.lower():
        print(f'{tier}: pending')
        return
    d = scheme_difference(runs, a, b)
    L = _length_name(int(runs.recall['length'].max()))
    last = runs.windows[runs.windows['measure'] == 'nll']['bucket'].iloc[-1]
    print(f'{tier} ({d["method"]}): {a} - {b}')
    print(f'  recall at {L}: ' + _interval(
        d['recall'], d['recall_lo'], d['recall_hi'], 2, sign=True))
    print(f'  NLL {last} (nats): ' + _interval(
        d['nll'], d['nll_lo'], d['nll_hi'], 3, sign=True))
```
