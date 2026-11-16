# Data Mixtures: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_data_mixtures` explains in a sentence each but does not
show: the token pools of 20.4-20.6 (`TokenPool`, `load_pool`), the byte
length of each token for bits per byte (`token_byte_lengths`), and the
FineWeb-Edu `sample-10BT` file order of the chapter (`_FWEDU_FILES`), and
the internals of the proxy runs: the training stream (`_ProxyStream`: which
window of which pool each sequence reads), the learning-rate schedule
(`_lr_at`), the optimizer and model of a proxy, its evaluation in bits per
byte (`_proxy_eval`), checkpoints, the DoReMi step, the run design of a
campaign (`mix_run_specs`), the pools of a configuration (`mix_pools`), the
record a run leaves (`campaign_record`), the barycentric coordinates of the
simplex plot (`barycentric`) and the FLOP count of a run (`proxy_flops`).
`tools/run_mix_campaign.py` runs a campaign from these pieces.

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble inside its functions: a module-level import would join the
`_preamble` shard and move the freshness fingerprint of every notebook in the
book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_data_mixtures`

```{.python .input #mixtures-lib-files}
%%tab pytorch
#@save
_FWEDU_FILES = tuple(  # sample/10BT files read, in order; 005, 012, 013 are chapter 21's
    f'sample/10BT/{k:03d}_00000.parquet' for k in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11))
```

```{.python .input #mixtures-lib-pools}
%%tab pytorch
#@save
TokenPool = collections.namedtuple('TokenPool', 'ids meta')

#@save
def load_pool(path_or_key, name=None):
    """A token pool: a flat array of token ids (read-only memmap; every
    document ends in EOS) and its meta dict. path_or_key is a DATA_HUB key
    (a tar of <name>.bin files and their meta.json), a directory or a .bin
    file. meta.json describes one pool, or maps each .bin file name to its
    meta when a directory holds several (then name picks one)."""
    import pathlib
    p = path_or_key
    if isinstance(p, str) and p in d2l.DATA_HUB:
        tar = d2l.download_keyed(p)
        p = pathlib.Path(tar).with_suffix('')
        if not (p / '.extracted').exists():
            p.mkdir(exist_ok=True)
            d2l.extract(tar, str(p))
            (p / '.extracted').touch()
    p = pathlib.Path(p)
    if p.is_dir():
        bins = sorted(b for b in p.glob('*.bin')
                      if name is None or b.stem == name)
        if len(bins) != 1:
            raise ValueError(f'{p}: {len(bins)} pools match name={name!r} '
                             f'({[b.stem for b in p.glob("*.bin")]})')
        p = bins[0]
    with open(p.parent / 'meta.json') as f:
        meta = json.load(f)
    meta = meta.get(p.name, meta)
    ids = np.memmap(p, dtype=meta.get('dtype', 'uint16'), mode='r')
    if 'n_tokens' in meta and len(ids) != meta['n_tokens']:
        raise ValueError(f'{p}: {len(ids)} tokens, meta says '
                         f'{meta["n_tokens"]}')
    return d2l.TokenPool(ids, meta)

#@save
def token_byte_lengths(tokenizer):
    """np.ndarray of the UTF-8 bytes each token id stands for (0 for
    special tokens): the denominator of bits per byte. Takes a Transformers
    tokenizer, a tokenizers.Tokenizer or a tiktoken encoding."""
    if hasattr(tokenizer, 'decode_single_token_bytes'):  # tiktoken
        n = tokenizer.n_vocab
        special = set(tokenizer.special_tokens_set)
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            try:
                b = tokenizer.decode_single_token_bytes(i)
            except KeyError:  # an unused id
                continue
            if b.decode('utf-8', 'replace') not in special:
                out[i] = len(b)
        return out
    raw = getattr(tokenizer, 'backend_tokenizer', tokenizer)
    vocab = raw.get_vocab(with_added_tokens=True)
    special = {t.content for t in raw.get_added_tokens_decoder().values()
               if t.special}
    byte_level = 'ByteLevel' in raw.to_str()  # one character per byte
    out = np.zeros(max(vocab.values()) + 1, dtype=np.int64)
    for piece, i in vocab.items():
        if piece in special:
            continue
        if byte_level:
            out[i] = len(piece)
        elif re.fullmatch(r'<0x[0-9A-Fa-f]{2}>', piece):  # byte fallback
            out[i] = 1
        else:  # SentencePiece-style: U+2581 marks a space
            out[i] = len(piece.replace('▁', ' ').encode('utf-8'))
    return out
```

```{.python .input #mixtures-lib-stream}
%%tab pytorch
#@save
def _stage_weights(weights, stages, names, steps):
    """(steps, m) array: the mixture each step samples from. weights maps a
    domain to its weight from step 0; stages lists (fraction, weights)
    pairs, each in force from step steps - round((1 - fraction) * steps)
    on (so a stage at 0.8 starts with the decay of a WSD schedule whose
    decay_frac is 0.2)."""
    def vec(w):
        v = np.array([float(w.get(d, 0.0)) for d in names])
        bad = v.min() < 0 or math.fabs(v.sum() - 1) > 1e-9
        if set(w) - set(names) or bad:
            raise ValueError(f'weights {w} are not a mixture over {names}')
        return v
    out = np.tile(vec(weights), (steps, 1))
    for frac, w in sorted(stages or []):
        out[steps - round((1 - frac) * steps):] = vec(w)
    return out

#@save
def _proxy_alloc(step_weights, batch):
    """Domain of every sequence, in allocation order: sequence s goes to
    the domain whose count lags its target (the running sum of its
    weights) the most, ties to the lower index, so after n sequences each
    domain holds its target count to within one sequence. Also returns
    each sequence's ordinal among its domain's sequences."""
    m = step_weights.shape[1]
    target, count = [0.0] * m, [0] * m
    dom, ordinal = [], []
    for w in step_weights.tolist():
        for _ in range(batch):
            for i in range(m):
                target[i] += w[i]
            i = max(range(m), key=lambda j: (target[j] - count[j], -j))
            dom.append(i)
            ordinal.append(count[i])
            count[i] += 1
    return np.array(dom, dtype=np.int64), np.array(ordinal, dtype=np.int64)

#@save
class _ProxyStream:
    """The training stream of a proxy run: the domain and the window of
    every sequence of every batch. A pool of N tokens has (N - 1) //
    seq_len windows of seq_len + 1 tokens starting at multiples of
    seq_len; a domain reads its windows in epochs, each a fresh
    permutation seeded by (seed, the domain's name, epoch); the sequences
    of a batch are shuffled by (seed, step). The stream depends only on
    (seed, weights, stages, tokens) and the pools, and batch(step) keeps no
    state, so a resumed run reads exactly the batches it would have read.
    A domain that needs more than max_epochs passes over its pool raises
    before any batch is drawn."""
    def __init__(self, pools, weights, tokens, seq_len, batch, seed=0,
                 stages=None, max_epochs=1):
        import zlib
        self.names = list(pools)
        self.ids = [getattr(pools[d], 'ids', pools[d]) for d in self.names]
        self.seq_len, self.batch_size, self.seed = seq_len, batch, seed
        self.steps = tokens // (batch * seq_len)
        if self.steps < 1:
            raise ValueError(f'{tokens} tokens make no batch of {batch} x '
                             f'{seq_len}')
        self.windows = [(len(ids) - 1) // seq_len for ids in self.ids]
        self.weights = _stage_weights(weights, stages, self.names, self.steps)
        dom, ordinal = _proxy_alloc(self.weights, batch)
        for k in range(self.steps):  # shuffle the order within each batch
            sl = slice(k * batch, (k + 1) * batch)
            perm = np.random.default_rng([seed, 1, k]).permutation(batch)
            dom[sl], ordinal[sl] = dom[sl][perm], ordinal[sl][perm]
        self.dom, self.ordinal = dom, ordinal
        counts = np.bincount(dom, minlength=len(self.names))
        self.epochs, self.epoch_starts = {}, {}
        for i, d in enumerate(self.names):
            W = self.windows[i]
            if counts[i] and (W == 0 or counts[i] > max_epochs * W):
                raise ValueError(
                    f'{d}: the run reads {counts[i]:,} windows, the pool '
                    f'holds {W:,} (max_epochs={max_epochs}): the pool is '
                    'smaller than the run needs')
            self.epochs[d] = counts[i] / W if W else 0.0
            rows = np.flatnonzero((dom == i) & (ordinal % max(W, 1) == 0))
            self.epoch_starts[d] = sorted(int(r // batch) for r in rows)
        self._salt = [zlib.crc32(d.encode()) for d in self.names]
        self._perms = {}

    def window(self, i, ordinal):
        """Index of the window that domain i's ordinal-th sequence reads."""
        epoch, pos = divmod(int(ordinal), self.windows[i])
        if (i, epoch) not in self._perms:
            self._perms[i, epoch] = np.random.default_rng(
                [self.seed, 0, self._salt[i], epoch]).permutation(
                self.windows[i])
        return self._perms[i, epoch][pos]

    def batch(self, step):
        """Token windows of the step's batch, np.int64 (batch, seq_len +
        1), and the domain index of each row."""
        sl = slice(step * self.batch_size, (step + 1) * self.batch_size)
        L, rows = self.seq_len, []
        for i, o in zip(self.dom[sl], self.ordinal[sl]):
            j = self.window(i, o)
            rows.append(self.ids[i][j * L:j * L + L + 1])
        return np.stack(rows).astype(np.int64), self.dom[sl]
```

```{.python .input #mixtures-lib-train}
%%tab pytorch
#@save
def _lr_at(step, steps, cfg):
    """Learning-rate factor of a proxy run at step (0-based): a linear
    warmup over warmup_frac of the steps, then a cosine to final_lr_frac
    (schedule 'cosine') or a constant rate and a linear decay to
    final_lr_frac over the last decay_frac of the steps ('wsd')."""
    warm = max(1, round(cfg.warmup_frac * steps))
    if step < warm:
        return (step + 1) / warm
    f = cfg.final_lr_frac
    if cfg.schedule == 'cosine':
        p = min((step - warm + 1) / max(steps - warm, 1), 1.0)
        return f + (1 - f) * 0.5 * (1 + math.cos(math.pi * p))
    if cfg.schedule == 'wsd':
        start = steps - round(cfg.decay_frac * steps)
        if step < start:
            return 1.0
        return 1 - (1 - f) * min((step - start + 1) / (steps - start), 1.0)
    raise ValueError(f'unknown schedule {cfg.schedule!r}')

#@save
def _proxy_model(cfg, seed=0):
    """A fresh proxy d2l.GPT of the configuration (RoPE, RMSNorm, SwiGLU,
    tied embeddings), initialized on the CPU from the seed, so its weights
    do not depend on the device it then runs on."""
    torch.manual_seed(seed)
    return d2l.GPT(cfg.vocab_size, num_hiddens=cfg.num_hiddens,
                   num_heads=cfg.num_heads, num_blks=cfg.num_blks,
                   max_len=cfg.seq_len)

#@save
def _proxy_optimizer(model, cfg):
    """AdamW with the configuration's betas and weight decay on the weight
    matrices only (not on embeddings, norms or biases), as d2l.LMOptimizer;
    the learning rate is set before every step."""
    named = list(model.named_parameters())
    decay = [p for n, p in named if p.ndim == 2 and 'emb' not in n]
    rest = [p for n, p in named if p.ndim != 2 or 'emb' in n]
    return torch.optim.AdamW([{'params': decay,
                               'weight_decay': cfg.weight_decay},
                              {'params': rest, 'weight_decay': 0.0}],
                             lr=cfg.lr, betas=tuple(cfg.betas))

#@save
def _per_token_nll(model, X, cfg):
    """Negative log-likelihood in nats of every next token of the windows
    X (batch, seq_len + 1), (batch, seq_len), in float32 (or the float64 of
    a float64 model); bf16 autocast if the configuration says so."""
    with torch.autocast(X.device.type, dtype=torch.bfloat16,
                        enabled=cfg.dtype == 'bfloat16'):
        logits = model(X[:, :-1])
    if logits.dtype != torch.float64:
        logits = logits.float()
    return F.cross_entropy(logits.flatten(0, 1), X[:, 1:].flatten(),
                           reduction='none').view(X.shape[0], -1)

#@save
@torch.no_grad()
def _proxy_eval(model, ids, cfg, byte_lengths, n_windows=None, device=None):
    """Summed NLL (nats), scored tokens, the UTF-8 bytes they stand for and
    bits per byte of model on the windows of a validation pool (all of
    them, or the first n_windows)."""
    device = device or next(model.parameters()).device
    L = cfg.seq_len
    n = (len(ids) - 1) // L
    n = n if n_windows is None else min(n, n_windows)
    nll, n_bytes, was_training = 0.0, 0, model.training
    model.eval()
    for a in range(0, n, cfg.micro_batch):
        X = np.stack([ids[j * L:j * L + L + 1]
                      for j in range(a, min(n, a + cfg.micro_batch))])
        X = X.astype(np.int64)
        n_bytes += int(byte_lengths[X[:, 1:]].sum())
        nll += _per_token_nll(model, torch.from_numpy(X).to(device),
                              cfg).double().sum().item()
    model.train(was_training)
    return dict(nll=nll, n_tokens=n * L, n_bytes=n_bytes,
                bpb=d2l.bpb_from_nll(nll, n_bytes))

#@save
def _proxy_fingerprint(weights, stages, tokens, seed, cfg):
    """SHA-256 of what determines a proxy run, so a checkpoint is never
    resumed into another run."""
    spec = dict(weights=weights, stages=stages, tokens=tokens, seed=seed,
                cfg=cfg._asdict())
    return hashlib.sha256(json.dumps(spec, sort_keys=True,
                                     default=list).encode()).hexdigest()

#@save
def _proxy_save(path, model, opt, step, out, fingerprint):
    """Write a checkpoint atomically (a temporary file, then a rename): the
    model, the optimizer, the next step, the results so far and the RNG
    states. The data stream needs no cursor: batch(step) is stateless."""
    tmp = f'{path}.tmp'
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                    step=step, out=out, fingerprint=fingerprint,
                    rng=torch.get_rng_state(),
                    cuda_rng=(torch.cuda.get_rng_state_all()
                              if torch.cuda.is_available() else None)), tmp)
    os.replace(tmp, path)

#@save
def _proxy_load(path, model, opt, fingerprint):
    """Restore a checkpoint into model and opt; returns (step, out)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['fingerprint'] != fingerprint:
        raise ValueError(f'{path} belongs to another run')
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    torch.set_rng_state(ck['rng'])
    if ck['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    return ck['step'], ck['out']

#@save
def proxy_flops(cfg, tokens, n_params):
    """Training FLOP of a proxy run: 6 N D for the parameters plus
    12 L d T D for attention (the non-causal convention, which counts the
    masked half of the score matrix too)."""
    return (6 * n_params * tokens
            + 12 * cfg.num_blks * cfg.num_hiddens * cfg.seq_len * tokens)
```

```{.python .input #mixtures-lib-doremi}
%%tab pytorch
#@save
def _doremi_step(alpha, excess, eta, smooth):
    """DoReMi's domain-weight update: alpha_i exp(eta excess_i),
    normalized (the maximizer of <alpha, excess> - KL(alpha || alpha_t) /
    eta on the simplex), then mixed with the uniform weights."""
    a = np.log(np.asarray(alpha, float)) + eta * np.asarray(excess, float)
    a = np.exp(a - a.max())
    a = a / a.sum()
    return (1 - smooth) * a + smooth / len(a)

#@save
def _doremi_excess(nll_proxy, nll_ref):
    """Per-domain excess loss of DoReMi: the mean over a domain's tokens of
    max(loss of the proxy - loss of the reference, 0), clipped per token."""
    return torch.clamp(nll_proxy - nll_ref, min=0).mean().item()
```

```{.python .input #mixtures-lib-campaign}
%%tab pytorch
#@save
def barycentric(W):
    """2-D coordinates of three-domain mixtures (n, 3) in the triangle with
    the first domain at (0, 0), the second at (1, 0) and the third at the
    top."""
    W = np.asarray(W, float)
    return np.stack([W[:, 1] + 0.5 * W[:, 2], math.sqrt(3) / 2 * W[:, 2]], 1)

#@save
def mix_run_specs(c):
    """The run design of a MIX_CONFIGS entry c, in launch order: the
    learning-rate runs, the pure-domain runs, the training mixtures (the
    Dirichlet draws, then the fixed edges), the held-out mixtures, the
    uniform seeds (cosine and WSD), the DoReMi proxy and the
    three arms. Each spec is a dict(run_id, arm, weights, stages, seed,
    schedule, tokens, eval_at, depends); weights of None are resolved when
    the run starts (the regression's choice, DoReMi's weights)."""
    D = c['domains']
    uniform = [1 / len(D)] * len(D)
    def spec(run_id, arm, weights, seed=0, schedule='cosine', stages=None,
             tokens=c['tokens'], eval_at=c['eval_at'], depends=()):
        return dict(run_id=run_id, arm=arm, weights=weights, stages=stages,
                    seed=seed, schedule=schedule, tokens=tokens,
                    eval_at=list(eval_at), depends=list(depends))
    lr_ids = [f'lr-{lr:.0e}' for lr in c['lr_grid']]
    runs = [spec(i, 'lr', uniform, tokens=c['lr_tokens'], eval_at=(1.0,))
            for i in lr_ids]
    after_lr = dict(depends=lr_ids)
    runs += [spec(f'pure-{d}', 'pure', list(np.eye(len(D))[k]), **after_lr)
             for k, d in enumerate(D)]
    train = []
    for conc, n in c['train']:
        W = d2l.sample_mixtures(n, D, conc, c['design_seeds'][0],
                                c['max_weight'])
        train += [spec(f'train-a{conc:g}-{k:02d}'.replace('.', ''),
                       'train', list(w), **after_lr)
                  for k, w in enumerate(W)]
    train += [spec(f'train-edge-{k:02d}', 'train', [float(x) for x in w],
                   **after_lr) for k, w in enumerate(c.get('edges', ()))]
    runs += train
    W = d2l.sample_mixtures(c['held'], D, 1.0, c['design_seeds'][1],
                            c['max_weight'])
    runs += [spec(f'held-{k:02d}', 'held', list(w), **after_lr)
             for k, w in enumerate(W)]
    runs += [spec(f'uniform-cos-s{s}', 'uniform-cos', uniform, seed=s,
                  **after_lr) for s in c['cos_seeds']]
    arm = dict(schedule='wsd', eval_at=c['arm_eval_at'])
    runs += [spec(f'uniform-wsd-s{s}', 'uniform', uniform, seed=s, **arm,
                  **after_lr) for s in c['arm_seeds']]
    runs.append(spec('doremi-proxy', 'doremi-weights', uniform,
                     depends=lr_ids + ['uniform-wsd-s0']))
    picked = lr_ids + [r['run_id'] for r in train]
    for s in c['arm_seeds']:
        runs.append(spec(f'regsel-s{s}', 'regsel', None, seed=s, **arm,
                         depends=picked))
        runs.append(spec(f'doremi-s{s}', 'doremi', None, seed=s, **arm,
                         depends=lr_ids + ['doremi-proxy']))
        runs.append(spec(f'staged-s{s}', 'staged', uniform, seed=s, **arm,
                         stages=None, depends=picked))
    return runs

#@save
def mix_pools(c, root=os.path.join('..', 'data', 'lm', 'mix')):
    """(pools, val), each {domain: TokenPool}, of a MIX_CONFIGS entry c:
    the files tools/build_mix_pools.py wrote under root/<name>/ where they
    exist, else the registered keys of c['keys']. A domain with neither is
    left out."""
    out = ({}, {})
    for k, size in enumerate((c['pool'], 'val-' + c['val'])):
        for d in c['domains']:
            name = f'{d}-{size}'
            local = os.path.join(root, c['name'], name + '.bin')
            if os.path.exists(local):
                out[k][d] = d2l.load_pool(local)
            elif c['keys'].get(name) in d2l.DATA_HUB:
                out[k][d] = d2l.load_pool(c['keys'][name], name)
    return out

#@save
def campaign_record(spec, result, cfg):
    """The JSON record a finished proxy run leaves in a campaign file: its
    spec, its learning rate and train_proxy_lm's results, every array
    turned into a list."""
    rec = dict(spec, lr=cfg.lr, **result)
    return json.loads(json.dumps(rec, default=lambda x: (
        x.tolist() if hasattr(x, 'tolist') else float(x))))
```
