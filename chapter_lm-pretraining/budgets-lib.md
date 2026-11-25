# Budgets: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is
never rendered or executed as a notebook: `make lib` only scans it for
`#@save` blocks, and `gen_notebooks.py` excludes it via `LIB_ONLY_FILES`.
It holds the library code of :numref:`sec_lm_budgets`
that the section explains in a sentence but does not show
(docs/code-guide.md 7.1): 21.6's committed result and its regeneration
job, the ladder's points (or the labelled simulated ladder while the
ladder campaign is pending), the training rate read from a run's log, and
the decode-rate harness behind `d2l.decode_rates`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_budgets`

## Registries

The decode rates are measured on one whole RTX 4090 by
`python tools/lm_regen.py decode-rates` (under the GPU-lock helper; about
20 minutes). Until that result is registered the page shows the shapes and
the roofline, labelled pending.

```{.python .input #budgets-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['decode-rates'] = (
    'lm-decode-rates-v1', 'json-table')

#@save
d2l.PRETRAIN_PENDING = d2l.PRETRAIN_PENDING | {'decode-rates'}

#@save
DECODE_LIMIT_GIB = 22.5  # a configuration above this is infeasible

#@save
def _decode_call():
    """The page's decode_rates call (budgets.md, 'Measured Decode Rates')."""
    return dict(models=('r2', 'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B',
                        'Qwen/Qwen3-4B'),
                batches=(1, 8, 32, 128), ctx=1024, new_tokens=128)

#@save
d2l.PRETRAIN_JOBS['decode-rates'] = ('decode_rates', lambda: _decode_call())
```

## The Ladder's Points

The fit of the section reads the twelve finished points of 21.2's ladder
(four sizes at three budgets) and, for its sensitivity check, the plateau
points (the branches' undecayed starting states). While the ladder campaign
is pending, a simulated ladder stands in: the same twelve (N, D) points
with the losses of Hoffmann et al.'s fit plus Gaussian noise of the t1
runs' seed spread, labelled as such; no exponent is claimed from it.

```{.python .input #budgets-lib-ladder}
%%tab pytorch
#@save
LADDER_SIZES = ('ladder-25m', 'ladder-54m', 'ladder-88m', 'r2')

#@save
HOFFMANN_FIT = (1.69, 406.4, 410.7, 0.34, 0.28)  # E, A, B, alpha, beta
# Hoffmann et al. (2022), as in sec_scaling-laws; budgets.md's FITS too

#@save
SIMULATED_LADDER = ('ladder pending (campaign C1); '
                    'simulated at the Hoffmann fit')

#@save
SIM_SEED_SD = 0.01  # nats, while the t1 seeds are not committed

#@save
_LADDER_COLUMNS = ['size', 'N', 'tokens', 'tokens_per_param', 'val_loss',
                   'finished']

#@save
def _preset_params(preset):
    """Exact parameter count of a preset (built on the meta device)."""
    with torch.device('meta'):
        model = d2l.PackedGPT.from_arch(d2l.r2_config(preset).arch)
    return sum(p.numel() for p in model.parameters())

#@save
def _ladder_plan():
    """The planned points of the ladder: per size the main run's budget and
    the two branch budgets (finished), and the branches' starting states
    (plateau), as rows (size, N, tokens, finished)."""
    rows = []
    for size in LADDER_SIZES:
        cfg, N = d2l.r2_config(size), _preset_params(size)
        per_step = cfg.global_batch * cfg.arch.T
        rows.append((size, N, cfg.steps * per_step, True))
        for start, decay in d2l._preset_saves(size)['branches']:
            rows.append((size, N, (start + decay) * per_step, True))
            rows.append((size, N, start * per_step, False))
    return pd.DataFrame(rows, columns=['size', 'N', 'tokens', 'finished'])

#@save
def _seed_sd():
    """The t1 runs' seed standard deviation of the final validation loss
    when they are committed, else SIM_SEED_SD."""
    key = d2l.PRETRAIN_ARTIFACTS['t1-runs'][0]
    if key not in d2l.DATA_HUB:
        return SIM_SEED_SD
    log = d2l.load_artifact('t1-runs').dropna(subset=['val_loss'])
    final = log.groupby('seed')['val_loss'].last()
    return float(final.std(ddof=1)) if len(final) > 1 else SIM_SEED_SD

#@save
def _simulated_ladder(seed=0):
    """The twelve planned finished points with Hoffmann et al.'s losses plus
    Gaussian noise of the seed spread; no plateau points."""
    plan = _ladder_plan()
    pts = plan[plan['finished']].reset_index(drop=True)
    E, A, B, alpha, beta = HOFFMANN_FIT
    N, D = pts['N'].to_numpy(float), pts['tokens'].to_numpy(float)
    noise = np.random.default_rng(seed).normal(0, _seed_sd(), len(pts))
    pts['val_loss'] = E + A / N ** alpha + B / D ** beta + noise
    return pts

#@save
def ladder_points():
    """The ladder's points (21.2): per size the finished points of three
    budgets and the plateau points, with columns size, N (parameters),
    tokens, tokens_per_param, val_loss (nats) and finished; while the
    ladder campaign is pending, the simulated ladder, labelled."""
    key = d2l.PRETRAIN_ARTIFACTS['ladder-evals'][0]
    if key in d2l.DATA_HUB or d2l._regenerates('ladder-evals'):
        evals = d2l.load_artifact('ladder-evals')
        pts = evals.points.rename(columns={'P': 'N'})
        label = d2l._label_of(evals)
    elif ({'ladder', 'ladder-evals', 'reference-run'} & d2l.PRETRAIN_PENDING
          or d2l._regenerating()):
        pts = _simulated_ladder()
        label = SIMULATED_LADDER
    else:
        raise KeyError('ladder-evals is not committed yet')
    if label != SIMULATED_LADDER and pts['finished'].all():
        raise ValueError('the ladder has no plateau points')
    pts = pts.assign(tokens_per_param=pts['tokens'] / pts['N'],
                     rank=pts['size'].map(LADDER_SIZES.index))
    pts = pts.sort_values(['finished', 'rank', 'tokens'], ascending=[
        False, True, True]).reset_index(drop=True)[_LADDER_COLUMNS]
    return d2l._show_label(d2l._with_label(pts, label))

#@save
def show_ladder_label(points):
    """Print the label of a stand-in ladder (the simulated one while the
    ladder campaign is pending, or a smoke run), so that no number fitted
    to it passes for the campaign's; return whether it is simulated."""
    d2l._show_label(points)
    return d2l._label_of(points) == SIMULATED_LADDER
```

## The Training Rate

GPU-seconds per training token come from the reference run's log (or its
labelled stand-in, the t1 runs): the median over steps of the GPUs in use
divided by the global tokens per second. Scaling that rate to other sizes
in proportion to N keeps the measured MFU fixed. With no log committed, an
MFU of 0.40 is assumed on the 6N count and said so.

```{.python .input #budgets-lib-training-rate}
%%tab pytorch
#@save
TrainRate = collections.namedtuple('TrainRate', 'seconds N mfu label')

#@save
ASSUMED_MFU = 0.40  # used only while no training log is committed

#@save
def training_rate(log):
    """GPU-seconds per training token of the run that wrote log (the
    reference run or its labelled stand-in) and that run's N (the matrix
    parameters of 6ND): TrainRate(seconds, N, mfu, label); with an empty
    (pending) log, the reference model at ASSUMED_MFU, labelled."""
    peak = d2l.GPU_PEAK_BF16['NVIDIA GeForce RTX 4090']
    if len(log) and 'tok_s' in log:
        arch = d2l.Arch(**log.attrs['config']['runs'][0]['arch'])
        N = d2l.flops_per_token(arch) / 6
        seconds = float(log['world'].median() / log['tok_s'].median())
        return TrainRate(seconds, N, float(log['mfu'].median()),
                         d2l._label_of(log))
    N = d2l.flops_per_token(d2l.r2_config('r2').arch) / 6
    return d2l._show_label(TrainRate(
        6 * N / (ASSUMED_MFU * peak), N, ASSUMED_MFU,
        f'training log pending; MFU {ASSUMED_MFU:.2f} assumed'))
```

## The Decode-Rate Harness

`decode_rates` measures greedy decode on one GPU: for every model and batch
a key-value cache of `ctx + new_tokens` positions whose first `ctx` hold
random keys and values (the time of a step does not depend on them), then
`new_tokens` steps that each write one position and attend to the cache up
to it. One step is a static-shape function (the position is a device
tensor; the attention reads the whole buffer, the unwritten slots masked,
as two batched products with the query heads grouped by key-value head,
so the cache is read once), so it is
captured once as a CUDA graph and replayed, which removes the per-kernel
launch cost that eager decode of a small model pays on every step. Each
repeat restarts at position `ctx` and rewrites the same cache slots, which
is equivalent to a fresh copy of the prefilled cache. Median and quartiles
of milliseconds per step over three timed repeats after one warm-up; peak
memory from `torch.cuda.max_memory_reserved`; a configuration whose
estimate or peak exceeds `DECODE_LIMIT_GIB` is infeasible. For the Qwen3
rows, `d2l.hf_decode_ms` on the Hugging Face model at batch 1 is recorded
for reference (not printed): the Hugging Face eager rate, which the
harness must beat, or the run stops. The page compares the rates with the
roofline of the bytes a step reads, the whole buffer of `ctx + new_tokens`
slots.

```{.python .input #budgets-lib-decode}
%%tab pytorch
#@save
def _decode_config(name):
    """(layers, kv heads, head width, a meta-device model) of 'r2' or a
    pinned Qwen3 checkpoint (its config.json only, no weights)."""
    if name == 'r2':
        arch = d2l.r2_config('r2').arch
        with torch.device('meta'):
            model = d2l.PackedGPT.from_arch(arch)
        return arch.L, arch.H_kv, arch.d_h, model
    import huggingface_hub
    def get(**kwargs):
        return huggingface_hub.hf_hub_download(
            name, 'config.json', revision=d2l.HF_REVISIONS[name], **kwargs)

    with d2l._quiet():
        try:  # the cached copy first: a build host may be offline
            path = get(local_files_only=True)
        except Exception:  # noqa: BLE001 (not cached yet)
            path = get()
    with open(path) as f:
        cfg = json.load(f)
    with torch.device('meta'):
        model = d2l.Qwen3GPT(**d2l.qwen3_gpt_kwargs(cfg))
    head = cfg.get('head_dim') or cfg['hidden_size'] // cfg[
        'num_attention_heads']
    return (cfg['num_hidden_layers'], cfg['num_key_value_heads'], head,
            model)

#@save
def decode_shapes(models):
    """Per model: parameters and bf16 key-value bytes per cached token
    (2 L H_kv d_h 2), from the configurations alone."""
    rows = []
    for name in models:
        L, H_kv, d_h, model = _decode_config(name)
        rows.append(dict(model=name, params=sum(
            p.numel() for p in model.parameters()),
            kv_bytes=2 * L * H_kv * d_h * 2))
    return pd.DataFrame(rows)

#@save
def _decode_step(model, max_len):
    """One greedy decode step with static shapes: step(token, ks, vs, pos)
    writes the keys and values of token (batch, 1) at position pos (a
    0-dim device tensor) and attends to positions 0..pos (the slots after
    it masked); then token <- argmax and pos <- pos + 1, in place; returns
    the logits. The attention of the one query is two batched products
    over the cache with the query heads grouped by key-value head, so the
    cache is read once and never repeated per query head."""
    p, att0 = model.token_emb.weight, model.blks[0].attention
    d_h, H, H_kv = att0.head_dim, att0.num_heads, att0.num_kv_heads
    inv = (1.0 / float(model.rope_theta) ** (torch.arange(
        0, d_h, 2, dtype=torch.float32) / d_h)).to(p.device)
    slots = torch.arange(max_len, device=p.device)
    W = model.lm_head.weight if hasattr(model, 'lm_head') else p

    def step(token, ks, vs, pos):
        B = token.shape[0]
        angles = pos.float().view(1, 1) * inv
        tables = torch.cos(angles), torch.sin(angles)
        bias = torch.zeros(max_len, device=p.device).masked_fill(
            slots > pos, -math.inf)  # the slots not written yet
        H_ = model.token_emb(token)
        for i, blk in enumerate(model.blks):
            att = blk.attention
            q, k, v = att.qkv(blk.norm1(H_), tables=tables)
            ks[i].index_copy_(2, pos.view(1), k)
            vs[i].index_copy_(2, pos.view(1), v)
            q = q.reshape(B, H_kv, H // H_kv, d_h)  # heads by kv head
            s = (q @ ks[i].transpose(-1, -2)).float() / math.sqrt(d_h)
            a = torch.softmax(s + bias, -1).to(q.dtype)
            Y = (a @ vs[i]).reshape(B, 1, H * d_h)
            H_ = H_ + att.W_o(Y)
            H_ = H_ + blk.ffn(blk.norm2(H_))
        logits = F.linear(model.norm(H_[:, -1]), W)
        token.copy_(logits.argmax(-1, keepdim=True))
        pos.add_(1)
        return logits
    return step

#@save
def _decode_estimate(params, kv_bytes, batch, max_len, vocab):
    """GiB of the weights, the cache and the logits of a configuration."""
    return (2 * params + batch * max_len * kv_bytes
            + 6 * batch * vocab) / 2**30

#@save
def _graph(step, args):
    """step(*args) captured as a CUDA graph (after three warm-up calls on a
    side stream); returns its replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step(*args)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step(*args)
    return g.replay

#@save
def _time_decode(model, batch, ctx, new_tokens, warmup=1, repeats=3,
                 graph=True):
    """Milliseconds per decode step (median, first and third quartile over
    the repeats), each repeat new_tokens steps from position ctx."""
    max_len = ctx + new_tokens
    with torch.inference_mode():
        ks, vs = model.init_cache(batch, max_len)
        ks.normal_(), vs.normal_()  # a step's time does not depend on them
        token = torch.zeros(batch, 1, dtype=torch.long, device=ks.device)
        pos = torch.tensor(ctx, device=ks.device)
        step = _decode_step(model, max_len)
        run = (_graph(step, (token, ks, vs, pos)) if graph else
               lambda: step(token, ks, vs, pos))
        times = []
        for _ in range(warmup + repeats):
            pos.fill_(ctx)
            t = d2l.Benchmark(lambda: [run() for _ in range(new_tokens)],
                              0, 1).time
            times.append(1e3 * t / new_tokens)
    return np.percentile(times[warmup:], [50, 25, 75])

#@save
def _hf_check_ms(name, ctx, new_tokens):
    """d2l.hf_decode_ms at batch 1 on the Hugging Face model (median ms)."""
    with d2l._quiet():
        model, _ = d2l.load_hf(name)
    ids = torch.randint(0, model.config.vocab_size, (1, ctx),
                        device=model.device)
    out = d2l.hf_prefill(model, ids, ctx)[0]
    token = out.logits[:, -1:].argmax(-1)
    ms = d2l.hf_decode_ms(model, out.past_key_values, token,
                          steps=new_tokens, warmup=1, repeats=3)[0]
    del model, out
    d2l.lm_free()
    return float(ms)

#@save
def _decode_model(name):
    """'r2' (its architecture with random weights: decode time depends on
    the shapes only) or a pinned Qwen3 checkpoint, bf16 on the GPU."""
    with d2l._quiet():
        if name == 'r2':
            model = d2l.PackedGPT.from_arch(d2l.r2_config('r2').arch)
            return model.to('cuda', torch.bfloat16).eval()
        return d2l.load_qwen3(name, device='cuda')

#@save
def _decode_rates(models, batches, ctx, new_tokens):
    """Measure every (model, batch) on the current GPU."""
    rows = []
    for name, params, kv in decode_shapes(models).itertuples(index=False):
        model = _decode_model(name)
        vocab = model.token_emb.weight.shape[0]
        first = len(rows)
        for b in batches:
            row = dict(model=name, params=params, kv_bytes=kv, batch=b,
                       ms=math.nan, ms_q1=math.nan, ms_q3=math.nan,
                       peak_gib=math.nan, stack='eager + CUDA graph',
                       feasible=False, hf_ms=math.nan)
            if _decode_estimate(params, kv, b, ctx + new_tokens,
                                vocab) <= DECODE_LIMIT_GIB:
                try:
                    torch.cuda.reset_peak_memory_stats()
                    ms, q1, q3 = _time_decode(model, b, ctx, new_tokens)
                    peak = torch.cuda.max_memory_reserved() / 2**30
                    row.update(ms=ms, ms_q1=q1, ms_q3=q3, peak_gib=peak,
                               feasible=bool(peak <= DECODE_LIMIT_GIB))
                except torch.OutOfMemoryError:
                    pass
            d2l.lm_free()
            rows.append(row)
        del model
        d2l.lm_free()
        if name != 'r2':  # the Hugging Face eager rate at batch 1
            hf_ms = _hf_check_ms(name, ctx, new_tokens)
            rows[first]['hf_ms'] = hf_ms
            ms = rows[first]['ms']
            if not ms < hf_ms:  # NaN (a failed measurement) fails too
                raise RuntimeError(
                    f'{name}: the harness ({ms:.1f} ms/step) is not faster '
                    f'than Hugging Face eager decode ({hf_ms:.1f} ms/step)')
    table = pd.DataFrame(rows)
    table['tok_s'] = np.where(table['feasible'],
                              1e3 * table['batch'] / table['ms'], np.nan)
    return table[_DECODE_COLUMNS]

#@save
_DECODE_COLUMNS = ['model', 'params', 'kv_bytes', 'batch', 'tok_s', 'ms',
                   'ms_q1', 'ms_q3', 'peak_gib', 'stack', 'feasible',
                   'hf_ms']

#@save
def decode_rates(models, batches, ctx, new_tokens):
    """Greedy decode rates on one GPU: per model and batch the tokens per
    second (tok_s; NaN where infeasible), milliseconds per step with
    quartiles, peak GiB and the stack used, with each model's parameters
    and key-value bytes per token. Measured only when D2L_RERUN names
    'decode-rates' (one whole card); otherwise the committed result, or,
    while pending, the shapes with empty rates, labelled."""
    config = dict(models=list(models), batches=list(batches), ctx=ctx,
                  new_tokens=new_tokens)
    res = d2l._pretrain_result('decode-rates', lambda: _decode_rates(
        models, batches, ctx, new_tokens), config)
    if res.empty:  # pending: the shapes, every rate missing
        rows = [dict(r, batch=b) for r in decode_shapes(models).to_dict(
            'records') for b in batches]
        res = d2l._with_label(pd.DataFrame(rows).reindex(
            columns=_DECODE_COLUMNS), d2l._label_of(res))
    return d2l._show_label(res)
```

## Displays

The lifetime-cost and ratio-band plots, the decode table and plot, and the
ladder-fit plot of the page: formatting only; the page computes every
number they show.

```{.python .input #budgets-lib-displays}
%%tab pytorch
#@save
def _short(model):
    return model.split('/')[-1]

#@save
def fmt_power(x, digits=1, tex=False):
    """x as '10^9' or '7×10^11' at digits significant digits (tex=True:
    '10^{9}', '7\\times10^{11}' for math text); 0 as '0'."""
    if x == 0:
        return '0'
    mantissa, exponent = f'{x:.{digits - 1}e}'.split('e')
    power = f'10^{{{int(exponent)}}}' if tex else f'10^{int(exponent)}'
    if float(mantissa) == 1:
        return power
    return mantissa + (r'\times' if tex else '×') + power

#@save
def fmt_sig(x):
    """x (positive) with thousands separated, or at two significant digits
    below 10: '4,368', '28', '0.37'."""
    return f'{x:,.0f}' if x >= 10 else f'{x:.2g}'

#@save
def plot_lifetime_cost(N, costs, dots, crosses):
    """Lifetime FLOPs against N, one curve per demand in costs (demand ->
    costs at N; the last, the break-even demand, dotted black), a dot at
    each demand's optimum (dots: (N, cost) per curve) and a cross at the
    two models of equal cost (crosses: (N, cost) pairs)."""
    demands = list(costs)
    D = r'D_{\mathrm{inf}}'
    legend = [f'${D} = {fmt_power(x, tex=True)}$' for x in demands[:-1]]
    legend.append(f'break-even ${fmt_power(demands[-1], 2, tex=True)}$')
    d2l.plot(N, list(costs.values()), 'parameters N', 'lifetime FLOPs',
             xscale='log', yscale='log', legend=legend,
             fmts=('-',) * len(dots) + ('k:',), figsize=(4.5, 3),
             legend_loc='outside')
    ax = d2l.plt.gca()
    for i, (n, c) in enumerate(dots):
        ax.plot(n, c, 'o', color=f'C{i}')
    ax.plot(*zip(*crosses), 'kx')

#@save
def plot_ratio_band(demand, ratio):
    """The optimal D*/N* against demand under each fit (ratio: fit name ->
    values), with the band between the first two shaded."""
    d2l.plot(demand, list(ratio.values()),
             r'generated tokens $D_{\mathrm{inf}}$', 'optimal D*/N*',
             legend=list(ratio), xscale='log', yscale='log',
             figsize=(4.5, 3))
    d2l.plt.gca().fill_between(demand, *list(ratio.values())[:2],
                               color='gray', alpha=0.2)

#@save
def show_decode_table(rates):
    """Tokens per second per model and batch ('—' where infeasible or not
    measured), and the batch-1 rate over the roofline (column roofline)."""
    table = rates.pivot(index='model', columns='batch', values='tok_s')
    roof = rates[rates['batch'] == 1].set_index('model')['roofline']
    batches = list(table.columns)
    print(f'{"model":>11} ' + ' '.join(f'{f"batch {b}":>9}' for b in batches)
          + '   batch 1 ÷ roofline')
    for model in dict.fromkeys(rates['model']):
        tok = table.loc[model]
        cells = ' '.join(f'{v:9,.0f}' if v == v else f'{"—":>9}'
                         for v in tok)
        ratio = tok[batches[0]] / roof[model]
        print(f'{_short(model):>11} {cells}   ' + (
            f'{ratio:.2f}' if ratio == ratio else '—'))
    stacks = rates['stack'].dropna().unique()
    print('tokens per second; decode stack: '
          + (', '.join(stacks) if len(stacks) else 'not measured yet'))

#@save
def plot_decode(rates):
    """Measured (solid, dots) and roofline (dashed) tokens per second
    against the batch, one colour per model."""
    d2l.set_figsize((4.5, 3))
    ax = d2l.plt.gca()
    for i, (model, g) in enumerate(rates.groupby('model', sort=False)):
        ax.plot(g['batch'], g['roofline'], '--', color=f'C{i}')
        ax.plot(g['batch'], g['tok_s'], 'o-', color=f'C{i}',
                label=_short(model))
    ax.plot([], [], 'k--', label='roofline')
    ax.set_xscale('log', base=2), ax.set_yscale('log')
    batches = sorted(rates['batch'].unique())
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel('batch'), ax.set_ylabel('tokens per second')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

#@save
def plot_ladder_fit(points, law, withheld, interval):
    """Finished ladder points (loss against tokens, one colour per size),
    the fitted law per size, and the withheld point (open circle) with its
    prediction and interval."""
    d2l.set_figsize((4.5, 3))
    ax = d2l.plt.gca()
    D = np.logspace(np.log10(points['tokens'].min() / 1.5),
                    np.log10(points['tokens'].max() * 1.5), 100)
    for i, (size, g) in enumerate(points.groupby('size', sort=False)):
        mask = g.index != withheld
        ax.plot(D, law(g['N'].iloc[0], D), '-', color=f'C{i}', lw=1)
        ax.plot(g['tokens'][mask], g['val_loss'][mask], 'o', color=f'C{i}',
                label=size)
    w = points.loc[withheld]
    pred, lo, hi = interval
    ax.errorbar(w['tokens'] * 1.04, pred, yerr=[[pred - lo], [hi - pred]],
                fmt='s', color='black', capsize=3, markersize=4,
                label='prediction')
    ax.plot(w['tokens'], w['val_loss'], 'o', mfc='none', color='black',
            label='withheld')
    ax.set_xscale('log'), ax.set_xlabel('training tokens D')
    ax.set_ylabel('validation loss (nats)')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
```
