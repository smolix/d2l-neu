# Energy, Power and Cost: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_energy_cost` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1):

- the section's registry entries, one statement each (the price table, the
  box's power traces, the root clock sweep and chapter 23's measurements)
  and its regeneration job;
- the workloads a power trace runs (idle, CUDA-graph decode, a bf16 matrix
  product, a training step of the reference architecture) and the traced
  window (`_power_window`);
- the three measurements (`power_profile`, `decode_energy`,
  `train_energy`), each one `_hw_result('box-power', ...)` call with its
  display and live check;
- the readers and displays of the clock sweep (`hw_clock_sweep_log`,
  `plot_dvfs`, `cap_effect`) and the author's root sweep (`hw_root_sweep`,
  which only the campaign package's `box_clock_sweep.sh` runs);
- the price table (`hw_prices`, `rent_per_hour`), the MFU and goodput
  readers (`training_mfu`, `training_goodput`), the serving points
  (`frontier_points`, `prefill_seconds`), the owned-versus-rented inputs,
  interval and tornado (`owned_inputs`, `u_star_interval`, `tornado_order`,
  `plot_tornado`) and the dimensional check (`check_cost_units`);
- the regeneration driver `regen_power`.

Blocks are tagged `%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_energy_cost`

## Registries

The price table (`tools/lm_data/hw/prices.csv`, transcribed 2026-11-25 from
the providers' pages, each row with its source and date).

```{.python .input #energy-cost-lib-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-hw-prices-v1'] = (
    d2l.LM_DATA_URL + 'lm-hw-prices-v1.csv',
    'f01fbc697cecf0b679800cf396ef67c5591f3e65')
```

```{.python .input #energy-cost-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['prices'] = ('lm-hw-prices-v1', 'csv')
d2l.HW_ARTIFACTS['box-power'] = ('lm-hw-buildbox-power-v1', 'tar')
d2l.HW_CAMPAIGNS['box-clocksweep'] = (
    'lm-hw-buildbox-clocksweep-v1', 'tar', 'C-box-root',
    'RTX 4090 (this box, with root)')
d2l.HW_CAMPAIGNS['mfu-table'] = (
    'lm-mfu-table-v1', 'json', 'chapter 23', 'measured MFU per layout')
d2l.HW_CAMPAIGNS['par-goodput'] = (
    'lm-par-goodput-v1', 'json', 'chapter 23', 'goodput per layout')
```

```{.python .input #energy-cost-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-power'] = (
    'regen_power', lambda: dict(config='box'))
```

## Workloads under a Power Trace

A traced window runs one workload over and over: a warm-up of `warm_s`
seconds first (NVML's `power.draw` on Ada is an average over the last
second, so the window starts on the load it measures), then `seconds` (or
`steps`) under a 10 Hz trace (`BenchLog.power_trace`, the sampler of
:numref:`sec_lm_accelerator_design`), each step synchronized. The window's
rate is work per second of the loop; its energy per unit of work is the
trace's integral divided by the work done over the trace's span
(`energy_per_token`, on the page). NVML's clock-event reasons are kept per
reason as the fraction of samples that carry it, so the three processes of a
regeneration combine them as numbers.

```{.python .input #energy-cost-lib-loads}
%%tab pytorch
#@save
_HW_POWER = dict(model='Qwen/Qwen3-0.6B', L=512, micro_batch=2, hz=10,
                 warm_s=2.0, decode_s=5.0)

#@save
_HW_THROTTLE = (  # NVML clock-event reason bits and their display names
    (0x1, 'idle'), (0x2, 'app clocks'), (0x4, 'SW power cap'),
    (0x8, 'HW slowdown'), (0x10, 'sync boost'), (0x20, 'SW thermal'),
    (0x40, 'HW thermal'), (0x80, 'power brake'), (0x100, 'display clocks'))

#@save
def _hw_graph(fn, warmup=3):
    """fn() captured in a CUDA graph (after warm-up calls and one call on a
    side stream); returns the graph, whose replay() runs it."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g

#@save
def _decode_load(model, b, L):
    """(step, tokens per step): one CUDA-graph decode step of b sequences
    over a static cache holding L tokens (hw_decode_bench's step)."""
    token = torch.randint(model.config.vocab_size, (b, 1), device='cuda')
    with torch.inference_mode():
        cache, refill = d2l.hw_static_cache(model, b, L)
        position = torch.tensor([L], device='cuda')

        def step():  # the cache holds L tokens again: every replay is alike
            refill()
            model(token, past_key_values=cache, cache_position=position,
                  logits_to_keep=1)
        g = _hw_graph(step)
    return g.replay, b

#@save
def _random_micro_batch(b, T, vocab, device='cuda', seed=0):
    """b one-document windows of T random token ids >= 32 (the reserved
    ids excluded) and their shifted targets, as pretrain_step reads them."""
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(32, vocab, (b, T + 1), generator=gen)
    pos = torch.arange(T, dtype=torch.int32).expand(b, T).contiguous()
    batch = dict(input_ids=ids[:, :-1].contiguous(), labels=ids[:, 1:].clone(),
                 doc_ids=torch.zeros_like(pos), pos_ids=pos)
    return {k: v.to(device) for k, v in batch.items()}

#@save
def _train_load(arch, tokens_per_step, micro_batch):
    """(step, tokens per step): one pretrain_step of the reference
    architecture with random weights (fp32 master weights, bf16 autocast,
    fused AdamW, the chunked loss, eager) on micro-batches of micro_batch
    random sequences of arch.T tokens, accumulated to tokens_per_step."""
    per_micro = micro_batch * arch.T
    if tokens_per_step % per_micro:
        raise ValueError(f'{tokens_per_step} tokens do not split into '
                         f'micro-batches of {per_micro}')
    cfg = d2l.r2_config('r2', arch=arch, compile=False)
    torch.manual_seed(cfg.seed)
    model = d2l.PackedGPT.from_arch(arch).cuda()
    opt = torch.optim.AdamW(d2l.param_groups(model, cfg.wd), lr=1e-4,
                            betas=cfg.betas, eps=cfg.eps, fused=True)
    micro = [_random_micro_batch(micro_batch, arch.T, arch.vocab, seed=i)
             for i in range(tokens_per_step // per_micro)]

    def step():
        d2l.pretrain_step(model, opt, micro, cfg, tokens_per_step)
    return step, tokens_per_step

#@save
def _power_load(name):
    """(step, work per step, unit) of a named workload: 'idle', 'decode-b<b>'
    (Qwen3-0.6B, context _HW_POWER['L']), 'matmul-<n>' (bf16 n x n x n) or
    'train-<tokens>' (the reference architecture)."""
    kind, _, size = name.partition('-')
    if kind == 'idle':
        return (lambda: time.sleep(0.1)), 0, ''
    if kind == 'decode':
        model = d2l.hw_random_model(_HW_POWER['model'])
        step, b = _decode_load(model, int(size[1:]), _HW_POWER['L'])
        return step, b, 'token'
    if kind == 'matmul':
        n = int(size)
        return d2l.hw_matmul_op(n, n, n, 'bf16'), 2 * n**3, 'FLOP'
    if kind == 'train':
        step, tokens = _train_load(d2l.r2_config('r2').arch, int(size),
                                   _HW_POWER['micro_batch'])
        return step, tokens, 'token'
    raise ValueError(f'unknown workload {name!r}')

#@save
def _throttle_fractions(samples):
    """Fraction of the trace's samples that carry each clock-event reason,
    as thr_<name> entries."""
    marks = [s['throttle'] for s in samples if s.get('throttle') is not None]
    return {f'thr_{name.replace(" ", "_")}': (
                sum(bool(m & bit) for m in marks) / len(marks) if marks
                else math.nan) for bit, name in _HW_THROTTLE}

#@save
def _throttle_names(row, share=0.5):
    """The reasons a window's row carries in at least `share` of its
    samples, or 'none'."""
    names = [name for _, name in _HW_THROTTLE
             if row.get(f'thr_{name.replace(" ", "_")}', 0) >= share]
    return ', '.join(names) or 'none'

#@save
def _power_window(step, work, seconds=None, steps=None, warm_s=None,
                  hz=None):
    """Run step() (work units each) for warm_s seconds, then for `seconds`
    or `steps` under a power trace. A dict: rate (units/s), the trace's mean
    power (W, trapezoid), median SM and memory clocks (MHz) and
    temperature, energy per unit (J), the clock-event fractions, whether a
    foreign process was on the card, and the trace itself."""
    warm_s = _HW_POWER['warm_s'] if warm_s is None else warm_s
    hz = hz or _HW_POWER['hz']
    t = time.perf_counter()
    while time.perf_counter() - t < warm_s:
        step()
        torch.cuda.synchronize()
    foreign = _hw_contention()[0]
    with _hw_power_trace(hz) as samples:
        n, t0 = 0, time.perf_counter()
        while n < steps if steps else time.perf_counter() - t0 < seconds:
            step()
            torch.cuda.synchronize()
            n += 1
        elapsed = time.perf_counter() - t0
    foreign += _hw_contention()[0]
    rate = n * work / elapsed
    P = np.array([s['power_w'] for s in samples], dtype=float)
    ts = np.array([s['t'] for s in samples], dtype=float)
    span = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    return dict(
        rate=rate, seconds=elapsed, steps=n,
        power_w=(np.trapezoid(P, ts) / span if span else
                 float(np.mean(P)) if len(P) else math.nan),
        sm_mhz=_hw_median(samples, 'sm_mhz'),
        mem_mhz=_hw_median(samples, 'mem_mhz'),
        temp_c=_hw_median(samples, 'temp_c'),
        j_per_unit=(d2l.energy_per_token(samples, rate * span)
                    if work and span else math.nan),
        contended=float(bool(foreign)), **_throttle_fractions(samples),
        trace=dict(t=ts.round(3).tolist(), power_w=P.round(1).tolist(),
                   sm_mhz=[s.get('sm_mhz') for s in samples]))

#@save
def _quiet_window(step, work, **kwargs):
    """_power_window, repeated once if a foreign process shared the card
    (the second window is kept either way; its contended flag says so)."""
    w = _power_window(step, work, **kwargs)
    return _power_window(step, work, **kwargs) if w['contended'] else w

#@save
def _split_traces(rows, key):
    """rows without their traces, and the traces keyed by key(row)."""
    traces = {str(key(r)): r.pop('trace') for r in rows}
    return rows, traces

#@save
def _sweep_rows(kind, setting, workloads, seconds=10):
    """One row per workload at one sweep setting (a locked clock in MHz or
    a power cap in W): the traced window and the step time."""
    rows = []
    for name in workloads:
        step, work, unit = _power_load(name)
        w = _power_window(step, work, seconds=seconds)
        w.pop('trace')
        rows.append(dict(kind=kind, setting=setting, workload=name,
                         unit=unit, step_ms=1e3 * w['seconds'] / w['steps'],
                         **w))
        del step
        d2l.lm_free()
    return rows
```

## Power at Rest and at the Roof

```{.python .input #energy-cost-lib-profile}
%%tab pytorch
#@save
def _power_limits():
    """The card's power limit (W) and top SM clock (MHz)."""
    lim = _hw_gpu_limits()
    return dict(power_limit_w=lim.get('power_limit'),
                max_sm_mhz=lim.get('clocks_max_sm'))

#@save
def _profile_compute(workloads, seconds):
    rows = []
    for name in workloads:  # one workload on the card at a time
        step, work, unit = _power_load(name)
        rows.append(dict(workload=name, unit=unit, **_quiet_window(
            step, work, seconds=seconds)))
        del step
        d2l.lm_free()
    rows, traces = _split_traces(rows, lambda r: r['workload'])
    df = _hw_frame(rows)
    df.attrs.update(traces=traces, field='power.draw (NVML, 1 s average)',
                    hz=_HW_POWER['hz'], **_power_limits())
    return df

#@save
_HW_WORKLOAD_NAMES = {'idle': 'idle', 'decode': 'decode, b = {}',
                      'matmul': 'matmul {}^3', 'train': 'training, {} tok'}

#@save
def _workload_label(name):
    kind, _, size = name.partition('-')
    return _HW_WORKLOAD_NAMES[kind].format(size.lstrip('b'))

#@save
def _show_profile(prof):
    if prof.empty:
        print(prof.attrs.get('label', 'no profile'))
        return
    table = pd.DataFrame({
        'workload': [_workload_label(w) for w in prof.workload],
        'W': prof.power_w, 'SM MHz': prof.sm_mhz,
        'clock-event reasons': [_throttle_names(r) for r in
                                prof.to_dict('records')]})
    d2l.show_table(table, digits={'W': 0, 'SM MHz': 0}, width=79)
    a = prof.attrs
    print(f'{a["field"]} at {a["hz"]} Hz; limit {_hw_fmt(a["power_limit_w"])}'
          f' W, top SM {_hw_fmt(a["max_sm_mhz"])} MHz')

#@save
def _profile_live(prof, same, name='matmul-8192', seconds=3):
    row = prof[prof.workload == name]
    if not len(row):
        return 'not run (no committed matrix-product window)'
    step, work, _ = _power_load(name)
    w = _power_window(step, work, seconds=seconds)
    pts = [('W', row.power_w.iloc[0], w['power_w'], 0.02),
           ('SM MHz', row.sm_mhz.iloc[0], w['sm_mhz'], 0.02)]
    return _hw_agreement(pts, same, '')

#@save
def power_profile(workloads=('idle', 'decode-b1', 'matmul-8192'),
                  seconds=10):
    """Mean board power, median SM clock and the clock-event reasons of
    each workload over a traced window of `seconds` (after a warm-up), one
    workload on the card at a time; the traces in attrs['traces']."""
    workloads = list(workloads)
    return _hw_result(
        'box-power', lambda: _profile_compute(workloads, seconds),
        dict(workloads=workloads, seconds=seconds, load=_HW_POWER),
        entry='profile', live_check=_profile_live, show=_show_profile,
        live_label='3 s of the matrix product')
```

## Energy per Decode Token

```{.python .input #energy-cost-lib-decode}
%%tab pytorch
#@save
def _decode_energy_compute(model_id, batches, contexts, seconds):
    chip = d2l.chip_specs().loc['RTX 4090']  # the committed card's cut
    sm = d2l.ServingModel(_hw_model_config(model_id), chip)
    model, rows, cut = d2l.hw_random_model(model_id), [], []
    for L in contexts:
        for b in batches:
            if b > sm.max_batch(L):
                cut.append([b, L])
                continue
            torch.cuda.reset_peak_memory_stats()
            step, tokens = _decode_load(model, b, L)
            w = _quiet_window(step, tokens, seconds=seconds)
            rows.append(dict(b=b, L=L, peak_gib=_hw_peak(), **w))
            del step
            d2l.lm_free()
    del model
    d2l.lm_free()
    rows, traces = _split_traces(rows, lambda r: f'b={r["b"]},L={r["L"]}')
    df = _hw_frame(rows).rename(columns={'rate': 'tokens_per_s',
                                         'j_per_unit': 'j_per_token'})
    knee = {str(L): sm.weight_bytes() / (L * sm.kv_bytes_per_token())
            for L in contexts}  # the caches' bytes equal the weights'
    df.attrs.update(model=model_id, cut=cut, knee=knee, traces=traces,
                    field='power.draw (NVML, 1 s average)')
    return df

#@save
def _loglog_slope(x, y):
    if len(x) < 2:
        return math.nan
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])

#@save
KneeSlopes = collections.namedtuple('KneeSlopes', ['below', 'above'])

#@save
def decode_slopes(e, knee):
    """Slopes of log J/token on log b of one context's decode series at
    the batches up to the knee and at those from it on."""
    lo, hi = e[e.b <= knee], e[e.b >= knee]
    return KneeSlopes(_loglog_slope(lo.b, lo.j_per_token),
                      _loglog_slope(hi.b, hi.j_per_token))

#@save
def _plot_decode_energy(e):
    """J/token against b (log-log) per context with the predicted knees
    dotted, and mean power on a second axis (dashed)."""
    contexts = sorted(set(e.L))
    d2l.plot([e[e.L == L].b.tolist() for L in contexts],
             [e[e.L == L].j_per_token.tolist() for L in contexts],
             xlabel='batch b', ylabel='J per token', xscale='log',
             yscale='log', legend=[f'L = {L:,}' for L in contexts],
             fmts=[f'C{i}o-' for i in range(len(contexts))],
             figsize=(5, 3.5))
    ax = d2l.plt.gca()
    for i, L in enumerate(contexts):
        ax.axvline(e.attrs['knee'][str(int(L))], color=f'C{i}', ls=':',
                   lw=1)
    ax2 = ax.twinx()
    for i, L in enumerate(contexts):
        g = e[e.L == L]
        ax2.plot(g.b, g.power_w, f'C{i}s--', ms=3, lw=1)
    ax2.set_ylabel('board power (W), dashed')
    ax2.set_ylim(0, 1.15 * e.power_w.max())

#@save
def _show_decode_energy(e):
    if e.empty:
        print(e.attrs.get('label', 'no decode series'))
        return
    _plot_decode_energy(e)
    g = e[e.L == e.L.min()].set_index('b')
    L, lo, hi = int(e.L.min()), g.index.min(), min(64, g.index.max())
    print(f'L = {L:,}: {_hw_sig(g.j_per_token[lo])} J/token at b = {lo}, '
          f'{_hw_sig(g.j_per_token[hi])} at b = {hi}')
    print(f'power {_hw_sig(g.power_w[lo])} W at b = {lo}, '
          f'{_hw_sig(g.power_w[hi])} W at b = {hi} (x'
          f'{g.power_w[hi] / g.power_w[lo]:.1f}); tokens/s x'
          f'{g.tokens_per_s[hi] / g.tokens_per_s[lo]:.0f}')
    cut = ', '.join(f'({b}, {L:,})' for b, L in e.attrs['cut'])
    print(f'cut by capacity, (b, L): {cut or "none"}')

#@save
def _decode_live(e, same, batches=(1, 64), seconds=2):
    L = int(min(e.L))
    model, pts = d2l.hw_random_model(e.attrs['model']), []
    for b in batches:
        row = e[(e.L == L) & (e.b == b)]
        if not len(row):
            continue
        step, tokens = _decode_load(model, b, L)
        w = _power_window(step, tokens, seconds=seconds)
        pts.append((f'b={b}', row.j_per_token.iloc[0], w['j_per_unit'],
                    0.03))
        del step
        d2l.lm_free()
    del model
    d2l.lm_free()
    return _hw_agreement(pts, same, 'J/token') if pts else 'not run'

#@save
def decode_energy(cfg, batches, L, seconds=None):
    """Tokens/s, board power and J/token of CUDA-graph decode of the model
    of cfg (random weights) at each batch and context, each point a traced
    window; points whose cache does not fit 90% of an RTX 4090 are cut and
    listed. attrs['knee']: the predicted knee W / (L k_KV) per context."""
    model_id = cfg.name_or_path
    batches, contexts = [int(b) for b in batches], [int(x) for x in L]
    seconds = seconds or _HW_POWER['decode_s']
    return _hw_result(
        'box-power', lambda: _decode_energy_compute(model_id, batches,
                                                    contexts, seconds),
        dict(model=model_id, batches=batches, L=contexts, seconds=seconds,
             load=_HW_POWER),
        entry='decode', live_check=_decode_live, show=_show_decode_energy,
        live_label='J/token at b = 1 and 64')
```

## Energy per Training Token

```{.python .input #energy-cost-lib-train}
%%tab pytorch
#@save
def _train_energy_compute(arch, tokens_per_step, micro_batch, seconds):
    rows, peak = [], d2l.chip_specs().loc['RTX 4090'].p16_tflops * 1e12
    for tokens in tokens_per_step:
        torch.cuda.reset_peak_memory_stats()
        step, _ = _train_load(arch, tokens, micro_batch)
        for _ in range(3):  # first steps: allocator and kernel selection
            step()
        w = _quiet_window(step, tokens, seconds=seconds)
        rows.append(dict(tokens_per_step=tokens,
                         accum=tokens // (micro_batch * arch.T),
                         peak_gib=_hw_peak(), **w))
        del step
        d2l.lm_free()
    rows, traces = _split_traces(rows, lambda r: r['tokens_per_step'])
    df = _hw_frame(rows).rename(columns={'rate': 'tokens_per_s',
                                         'j_per_unit': 'j_per_token'})
    df['mfu'] = [d2l.mfu(r, arch, peak, T=arch.T, causal_half=True)
                 for r in df.tokens_per_s]
    df.attrs.update(arch=arch._asdict(), peak_flops=peak, traces=traces,
                    field='power.draw (NVML, 1 s average)')
    return df

#@save
def _show_train_energy(tr):
    if tr.empty:
        print(tr.attrs.get('label', 'no training series'))
        return
    table = pd.DataFrame({'tokens/step': tr.tokens_per_step,
                          'accum': tr.accum, 'tokens/s': tr.tokens_per_s,
                          'W': tr.power_w, 'J/token': tr.j_per_token,
                          'MFU': tr.mfu})
    d2l.show_table(table, digits={'tokens/s': 0, 'W': 0, 'J/token': 4,
                                  'MFU': 2}, width=79)
    print(f'MFU: of the {tr.attrs["peak_flops"] / 1e12:.0f} TFLOP/s bf16 '
          'datasheet peak, causal attention counted at half')

#@save
def _train_live(tr, same, steps=5):
    arch = d2l.Arch(**tr.attrs['arch'])
    row = tr.iloc[0]
    tokens = int(row.tokens_per_step)
    step, _ = _train_load(arch, tokens, _HW_POWER['micro_batch'])
    for _ in range(3):
        step()
    w = _power_window(step, tokens, steps=steps, warm_s=1.0)
    del step
    d2l.lm_free()
    return _hw_agreement([('tokens/s', row.tokens_per_s, w['rate'], 0.03),
                          ('W', row.power_w, w['power_w'], 0.03)], same, '')

#@save
def train_energy(arch, tokens_per_step, micro_batch, seconds=10):
    """Tokens/s, board power, J/token and MFU (causal attention halved, the
    RTX 4090's bf16 datasheet peak) of training steps of the reference
    architecture with random weights, one row per step size (micro-batches
    of micro_batch sequences, accumulated), each a traced window."""
    tokens_per_step = [int(t) for t in tokens_per_step]
    return _hw_result(
        'box-power', lambda: _train_energy_compute(
            arch, tokens_per_step, micro_batch, seconds),
        dict(arch=arch._asdict(), tokens_per_step=tokens_per_step,
             micro_batch=micro_batch, seconds=seconds),
        entry='train', live_check=_train_live, show=_show_train_energy,
        live_label='5 steps of 8,192 tokens')
```

## The Clock and Power-Cap Sweeps

The sweeps need root (`nvidia-smi -lgc` locks the SM clock, `-pl` sets the
power limit), so no notebook and no build runs them: the author runs the
campaign package's `box_clock_sweep.sh` with `sudo` on one card; it calls
`hw_root_sweep`, which restores the default clocks and limit on exit. Its
table (one row per setting and workload) becomes the campaign log
`box-clocksweep`; campaign C-T2 writes the same table for the A100
(`a100-clocksweep`). Until either is committed, the fit runs on the plan's
synthetic numbers: static power 80 W, 300 W at 1.4 GHz (k = 80 W/GHz^3),
throughput proportional to the clock.

```{.python .input #energy-cost-lib-sweep}
%%tab pytorch
#@save
_HW_SYNTHETIC_DVFS = dict(P_s=80.0, k=80.0,  # W, W/GHz^3 (the plan's case)
                          clocks_ghz=(0.6, 0.76, 0.92, 1.08, 1.24, 1.4))

#@save
def _synthetic_sweep():
    """The plan's synthetic clock sweep of a compute-bound workload: power
    P_s + k f^3 exactly, throughput proportional to f (1 at the top clock)."""
    s = _HW_SYNTHETIC_DVFS
    f = np.array(s['clocks_ghz'])
    df = pd.DataFrame(dict(kind='clock', setting=f * 1e3,
                           workload='matmul-8192', sm_mhz=f * 1e3,
                           power_w=s['P_s'] + s['k'] * f**3,
                           rate=f / f.max(), unit='relative'))
    df.attrs.update(source='synthetic: the plan\'s P_s = 80 W, k = 80 '
                    'W/GHz^3; no measurement yet', synthetic=True)
    return df

#@save
def hw_clock_sweep_log():
    """The clock and power-cap sweep to fit: the box's root sweep when
    committed, else campaign C-T2's A100 sweep, else the synthetic set;
    attrs['source'] says which, attrs['synthetic'] whether it is one."""
    for name in ('box-clocksweep', 'a100-clocksweep'):
        log = hw_campaign_log(name)
        if log:
            df = log['sweep']
            df.attrs.update(source=f'measured on {d2l.HW_CAMPAIGNS[name][3]}',
                            synthetic=False)
            return df
    return _synthetic_sweep()

#@save
def _cap_sweep(caps, workloads, seconds=10):
    """Rows (kind 'cap') of each workload under each power limit (W) on
    visible GPU 0, the default limit restored at the end (root)."""
    uuid = _hw_visible_uuids()[0]
    default = _hw_gpu_limits().get('power_default_limit')

    def limit(watts):
        if hw_run(['nvidia-smi', '-i', uuid, '-pl', str(int(watts))],
                  60).refused:
            raise PermissionError('setting the power limit needs root')
    rows = []
    try:
        for cap in caps:
            limit(cap)
            rows += _sweep_rows('cap', cap, workloads, seconds)
    finally:
        if default:
            limit(default)
    return rows

#@save
def hw_root_sweep(clocks=None, caps=(150, 225, 300, 375, 450),
                  workloads=('matmul-8192', 'decode-b1', 'train-8192')):
    """The author's root sweep on visible GPU 0 (clocks in MHz, by default
    six from 40% to 100% of the top clock; power caps in W): one table of
    both sweeps, written as the entry 'sweep' under D2L_HW_OUT for the
    campaign package to pack and upload. Needs root; restores the clocks
    and the limit on exit."""
    if clocks is None:
        top = _hw_gpu_limits()['clocks_max_sm']
        clocks = [int(15 * round(top * x / 15))
                  for x in np.linspace(0.4, 1.0, 6)]
    rows = d2l.hw_clock_sweep(clocks, workloads).to_dict('records')
    rows += _cap_sweep(caps, workloads)
    df = _hw_frame(rows)
    key = d2l.HW_CAMPAIGNS['box-clocksweep'][0]
    _hw_write(os.path.join(_hw_rerun_root(), key, _hw_entry_file('sweep')),
              df, dict(name='box-clocksweep', entry='sweep', env=_hw_env(),
                       config=dict(clocks=list(clocks), caps=list(caps),
                                   workloads=list(workloads))))
    return df

#@save
def plot_dvfs(fit):
    """Power (left) and throughput per watt relative to the top clock
    (right, dashed) against the SM clock: the sweep's points, the fitted
    model, f* dotted; prints the fit and the predicted against the measured
    energy minimum (or that the set is synthetic)."""
    f = np.linspace(min(fit.clocks) * 0.9, max(fit.clocks) * 1.05, 200)
    f_top = max(fit.clocks)
    power = lambda x: fit.P_s + fit.k * x**3
    eta_top = f_top / power(f_top)  # throughput per watt, throughput ~ f
    d2l.plot([f, fit.clocks], [power(f), fit.power],
             xlabel='SM clock (GHz)', ylabel='board power (W)',
             fmts=['C0-', 'C0o'], legend=['P_s + k f^3', 'sweep'],
             figsize=(5, 3.5))
    ax = d2l.plt.gca()
    ax.axvline(fit.f_star, color='k', ls=':', lw=1)
    ax2 = ax.twinx()
    ax2.plot(f, f / power(f) / eta_top, 'C1--')
    ax2.set_ylabel('work per joule / at the top clock')
    gain = fit.f_star / power(fit.f_star) / eta_top
    print(f'fit: P_s = {fit.P_s:.0f} W, k = {fit.k:.0f} W/GHz^3; f* = '
          f'{fit.f_star:.2f} GHz, P(f*) = {power(fit.f_star):.0f} W')
    print(f'at f*: {gain:.1f}x the work per joule and '
          f'{fit.f_star / f_top:.2f}x the throughput of {f_top:.2f} GHz')
    if not fit.synthetic:
        print(f'measured energy minimum at {fit.f_min:.2f} GHz, f* = '
              f'{fit.f_star:.2f} GHz (sweep step {fit.step:.2f} GHz)')
    print(fit.source)

#@save
def cap_effect(log):
    """The power-cap rows of a sweep log: training and batch-1 decode step
    times under each cap and their ratios to the uncapped (highest) limit;
    one line while no root sweep exists. Returns the table or None."""
    caps = log[log.kind == 'cap']
    if not len(caps):
        print('power-cap sweep pending (it needs root: the author\'s '
              'C-box-root script)')
        return None
    t = caps.pivot_table(index='setting', columns='workload',
                         values='step_ms').sort_index()
    train = next(c for c in t.columns if c.startswith('train'))
    decode = next(c for c in t.columns if c.startswith('decode'))
    top = t.index.max()
    table = pd.DataFrame({
        'cap W': t.index, 'train ms': t[train].values,
        'train ratio': (t[train] / t.loc[top, train]).values,
        'decode b=1 ms': t[decode].values,
        'decode ratio': (t[decode] / t.loc[top, decode]).values})
    table = table[table['cap W'] < top].reset_index(drop=True)
    d2l.show_table(table, digits={'cap W': 0, 'train ms': 0,
                                  'train ratio': 2, 'decode b=1 ms': 2,
                                  'decode ratio': 2}, width=79)
    return table
```

## Prices

`hw_prices` reads the dated price table and shows the rows the calculators
use: the rents by chip (median of the quotes, their number in parentheses),
the purchase prices and the owned-hardware assumptions, with each row's age
in days (marked above 90 days: a quote that old is refreshed before it is
quoted). The table's extra columns `chip` and `note` (beyond the plan's six)
name the `chip_specs` row a price belongs to and what exactly the page
listed.

```{.python .input #energy-cost-lib-prices}
%%tab pytorch
#@save
_HW_PRICE_COLUMNS = ('item', 'chip', 'value', 'unit', 'kind', 'source_url',
                     'retrieved', 'note')

#@save
def _check_prices(df):
    """The price table's contract: its columns, a known kind, an https
    source and an ISO date on every row, every chip a chip_specs row."""
    assert list(df.columns) == list(_HW_PRICE_COLUMNS), list(df.columns)
    specs = d2l.chip_specs()
    for r in df.itertuples():
        assert r.kind in ('quote', 'assumption'), r.item
        assert str(r.source_url).startswith('https://'), r.item
        assert len(str(r.retrieved)) == 10, r.item
        assert pd.isna(r.chip) or r.chip in specs.index, r.chip

#@save
def _price_age(df, today=None):
    """Days since each row was retrieved."""
    import datetime
    today = today or datetime.date.today()
    return [(today - datetime.date.fromisoformat(d)).days
            for d in df.retrieved]

#@save
def rent_per_hour(prices, chip):
    """Median on-demand price per GPU-hour of chip over the table's quotes."""
    rows = prices[prices.item.str.startswith('rent') & (prices.chip == chip)]
    if not len(rows):
        raise KeyError(f'no rental quote for {chip}')
    return float(rows.value.median())

#@save
def _price_rows(prices):
    """The displayed rows: one per rented chip (median, number of quotes),
    then the purchase prices and assumptions the calculators read."""
    rows = []
    rent = prices[prices.item.str.startswith('rent')]
    for chip, g in rent.groupby('chip', sort=False):
        rows.append(dict(item=f'rent {chip} ({len(g)})',
                         value=g.value.median(), unit=g.unit.iloc[0],
                         kind=g.kind.iloc[0], age=int(g.age.max())))
    used = prices[~prices.item.str.startswith('rent')
                  & (prices.item != 'buy card')]
    for r in used.itertuples():
        name = r.item if pd.isna(r.chip) else f'{r.item}, {r.chip}'
        rows.append(dict(item=name, value=r.value, unit=r.unit, kind=r.kind,
                         age=int(r.age)))
    return pd.DataFrame(rows)

#@save
def hw_prices():
    """The dated price table (one row per quote or assumption, with its age
    in days); prints the rows the cost calculators use."""
    prices = _hw_static('prices')
    _check_prices(prices)
    prices['age'] = _price_age(prices)
    table = _price_rows(prices)
    table['age (d)'] = [f'{a} (old)' if a > 90 else str(a)
                        for a in table.pop('age')]
    d2l.show_table(table, max_rows=12, width=79)
    return prices
```

## MFU and Goodput for the Cost of Training

```{.python .input #energy-cost-lib-mfu}
%%tab pytorch
#@save
def _reference_run_mfu(arch):
    """MFU of chapter 21's reference run on the box's four cards (the
    median of its logged steps after the first tenth), converted from
    chapter 21's count (the full attention term, its own peak table) to
    this chapter's (the causal half, chip_specs' peak); None while the log
    is not registered."""
    key = d2l.PRETRAIN_ARTIFACTS['reference-run-log'][0]
    if key not in d2l.DATA_HUB:
        return None
    log = d2l.load_artifact('reference-run-log').dropna(subset=['mfu'])
    steps = log[log.step > log.step.max() * 0.1]
    full = d2l.flops_per_token(arch, T=arch.T)
    half = d2l.flops_per_token(arch, T=arch.T, causal_half=True)
    peaks = (d2l.GPU_PEAK_BF16['NVIDIA GeForce RTX 4090']
             / (d2l.chip_specs().loc['RTX 4090'].p16_tflops * 1e12))
    return float(steps.mfu.median()) * half / full * peaks

#@save
def _box_qwen8b_mfu(cfg):
    """(MFU, s) of Qwen3-8B training on one RTX 4090 from 22.3's depth
    series: the step time of b sequences of s tokens is linear in the
    layers kept, so the line at the model's depth gives the full model's
    step time (an upper bound for a multi-GPU run: no communication)."""
    grid = hw_box_log('box-workloads', 'grid-box')
    tr = grid[(grid.workload == 'train') & (grid.model == 'Qwen/Qwen3-8B')]
    b, s = int(tr.b.max()), int(tr.L.max())
    tr = tr[(tr.b == b) & (tr.L == s)]
    slope, intercept = np.polyfit(tr.n_layers, tr.ms, 1)
    t_step = (intercept + slope * cfg.num_hidden_layers) / 1e3
    peak = d2l.chip_specs().loc['RTX 4090'].p16_tflops * 1e12
    return d2l.mfu(b * s / t_step, cfg, peak, T=s, causal_half=True), s

#@save
def training_mfu(chips=('A100 SXM 80 GB', 'H100 SXM', 'B200')):
    """One row per (model, chip): the MFU a training-cost estimate uses,
    its context T and its source. The reference model on the RTX 4090:
    chapter 21's four-card run when registered, else this section's
    one-card measurement (box-power). Qwen3-8B on each of chips: chapter
    23's mfu-table when registered (records with model, chip, mfu, T; the
    causal half counted), else 22.3's one-card Qwen3-8B MFU on the box, an
    upper bound."""
    arch = d2l.r2_config('r2').arch
    ref = _reference_run_mfu(arch)
    if ref is not None:
        rows = [dict(model='reference', chip='RTX 4090', mfu=ref, T=arch.T,
                     source='4 x RTX 4090, chapter 21')]
    else:
        tr = hw_box_log('box-power', 'train')
        rows = [dict(model='reference', chip='RTX 4090',
                     mfu=float(tr.mfu.max()), T=arch.T,
                     source='1 x RTX 4090, this section')]
    table = hw_campaign_log('mfu-table')
    measured = {(r['model'], r['chip']): r for r in table} if table else {}
    cfg, box = d2l.hw_config('Qwen/Qwen3-8B'), None
    for chip in chips:
        r = measured.get(('Qwen3-8B', chip))
        if r is not None:
            rows.append(dict(model='Qwen3-8B', chip=chip, mfu=r['mfu'],
                             T=r['T'], source='chapter 23, measured'))
            continue
        box = box or _box_qwen8b_mfu(cfg)
        rows.append(dict(model='Qwen3-8B', chip=chip, mfu=box[0], T=box[1],
                         source='1 x RTX 4090, upper bound'))
    return pd.DataFrame(rows)

#@save
def training_goodput(stated=0.9):
    """(g, source): chapter 23's measured goodput (par-goodput, a JSON with
    a 'goodput' value) when registered, else the stated value."""
    log = hw_campaign_log('par-goodput')
    if log:
        return float(log['goodput']), 'chapter 23, measured'
    return stated, 'stated; chapter 23 measures it'
```

## Serving Points

```{.python .input #energy-cost-lib-serve}
%%tab pytorch
#@save
def frontier_points(grid, model_id, L, batches):
    """Measured CUDA-graph decode points of 22.3's grid at context L: per
    batch, tokens/s per user (1 / step time) and per GPU (b / step time)."""
    d = grid[(grid.workload == 'decode') & (grid.series == 'grid')
             & (grid.model == model_id) & (grid.L == L)
             & grid.b.isin(list(batches))].sort_values('b')
    return pd.DataFrame({'b': d.b.values, 'user_tps': 1e3 / d.ms.values,
                         'gpu_tps': d.b.values * 1e3 / d.ms.values})

#@save
def prefill_seconds(grid, model_id, n):
    """Measured seconds to prefill one prompt of n tokens (22.3's grid)."""
    d = grid[(grid.workload == 'prefill') & (grid.model == model_id)
             & (grid.L == n)]
    return float(d.ms.iloc[0]) / 1e3
```

## Owned against Rented

```{.python .input #energy-cost-lib-owned}
%%tab pytorch
#@save
class OwnedInputs(dict):
    """The seven inputs of owned_vs_rented for one chip, with the kind
    ('quote' or 'assumption') of each in .kinds."""
    def __init__(self, values, kinds):
        super().__init__(values)
        self.kinds = dict(kinds)

#@save
def owned_inputs(prices, specs):
    """OwnedInputs of the box's card (a quarter of the four-card server,
    the owned-box operations cost) and of an H100 SXM (the quoted price
    per GPU, the datacentre operations cost), each at its chip's TDP."""
    def row(item, chip=None):
        r = prices[(prices.item == item) & (
            (prices.chip == chip) if chip else prices.chip.isna())]
        return float(r.value.iloc[0]), r.kind.iloc[0]
    shared = dict(life_h=row('life'), pue=row('PUE'),
                  elec_per_kwh=row('electricity'))
    out = {}
    for name, chip, capex, ops, n in (
            ('RTX 4090 (box)', 'RTX 4090', 'buy 4-card server',
             'ops owned box', 4),
            ('H100 SXM', 'H100 SXM', 'buy GPU', 'ops datacenter', 1)):
        cost, kind = row(capex, chip)
        x = dict(capex=(cost / n, kind),
                 power_kw=(float(specs.loc[chip].tdp_w) / 1e3, 'quote'),
                 ops_per_h=row(ops, chip),
                 rent_per_h=(rent_per_hour(prices, chip), 'quote'), **shared)
        out[name] = OwnedInputs({k: v[0] for k, v in x.items()},
                                {k: v[1] for k, v in x.items()})
    return out

#@save
def _u_star_scaled(x, scale):
    return d2l.owned_vs_rented(**{k: v * scale.get(k, 1.0)
                                  for k, v in x.items()})['u_star']

#@save
def u_star_interval(x, rel=0.3, draws=10_000, seed=0, q=(0.05, 0.95)):
    """The q-quantiles of u* when every input moves independently and
    uniformly within +-rel of its value (draws samples, seeded)."""
    rng = np.random.default_rng(seed)
    names = list(x)
    s = rng.uniform(1 - rel, 1 + rel, (draws, len(names)))
    u = [_u_star_scaled(x, dict(zip(names, row))) for row in s]
    return tuple(float(v) for v in np.quantile(u, q))

#@save
def tornado_order(x, rel=0.3):
    """(input, (u* at -rel, u* at +rel)) with each input moved alone,
    widest range first."""
    spans = {k: (_u_star_scaled(x, {k: 1 - rel}),
                 _u_star_scaled(x, {k: 1 + rel})) for k in x}
    return sorted(spans.items(),
                  key=lambda kv: -math.fabs(kv[1][1] - kv[1][0]))

#@save
def plot_tornado(inputs, rel=0.3):
    """One tornado per chip: the range of u* as each input moves by -rel
    to +rel alone, widest on top; assumption inputs hatched, u* at the
    inputs dashed."""
    fig, axes = d2l.plt.subplots(1, len(inputs), figsize=(7, 3))
    for ax, (chip, x) in zip(np.atleast_1d(axes), inputs.items()):
        order = tornado_order(x, rel)[::-1]  # widest on top
        for i, (k, (lo, hi)) in enumerate(order):
            quote = x.kinds[k] == 'quote'
            ax.barh(i, math.fabs(hi - lo), left=min(lo, hi),
                    edgecolor='C0', color='C0' if quote else 'white',
                    hatch='' if quote else '///')
        ax.axvline(d2l.owned_vs_rented(**x)['u_star'], color='k', ls='--',
                   lw=1)
        ax.set_yticks(range(len(order)), [k for k, _ in order])
        ax.set_xlabel('break-even utilization u*')
        ax.set_title(chip, fontsize=10)
    fig.tight_layout()
```

## Units

A dimensional check of the three cost functions: each argument becomes a
quantity that carries the exponents of its units (an hour counts as a time,
a kilowatt-hour as an energy), the functions run on them unchanged, and the
results must come out as dollars per token (the functions scale it by
10^6), dollars per time and a pure number. A sum of two quantities of
different units raises.

```{.python .input #energy-cost-lib-units}
%%tab pytorch
#@save
class _Dim:
    """Unit exponents only (usd, s, flop, token, joule): products add
    exponents, sums need equal ones; plain numbers are unitless."""
    def __init__(self, units=None):
        self.u = {k: v for k, v in (units or {}).items() if v}

    @staticmethod
    def _of(x):
        return x.u if isinstance(x, _Dim) else {}

    def __mul__(self, x):
        o = self._of(x)
        return _Dim({k: self.u.get(k, 0) + o.get(k, 0)
                     for k in set(self.u) | set(o)})
    __rmul__ = __mul__

    def __truediv__(self, x):
        o = self._of(x)
        return _Dim({k: self.u.get(k, 0) - o.get(k, 0)
                     for k in set(self.u) | set(o)})

    def __rtruediv__(self, x):
        return _Dim(self._of(x)) / self

    def __add__(self, x):
        if self._of(x) != self.u:
            raise TypeError(f'adding {self._of(x)} to {self.u}')
        return self
    __radd__ = __add__

#@save
def check_cost_units():
    """Run the cost functions on unit-carrying arguments and print the
    units of their results; raise if any is wrong."""
    D = lambda **u: _Dim(u)
    hour, usd, kwh = D(s=1), D(usd=1), D(joule=1)
    train = d2l.cost_per_mtok_train(usd / hour, D(flop=1, token=-1),
                                    D(flop=1, s=-1), 0.4, 0.9)
    serve = d2l.cost_per_mtok_serve(usd / hour, D(token=1, s=-1), 0.5)
    own = d2l.owned_vs_rented(usd, hour, kwh / hour, 1.5, usd / kwh,
                              usd / hour, usd / hour)
    assert train.u == serve.u == {'usd': 1, 'token': -1}, (train.u, serve.u)
    assert own['owned_per_used_h'].u == {'usd': 1, 's': -1}
    assert not _Dim._of(own['u_star'])
    print('units: USD per token (x 1e6) and per hour; u* a pure number')
```

## The Regeneration Driver

`tools/hw_regen.py box-power` calls this in each of its three processes on
one whole card: the page's three measurements on their full configuration,
each window also a BenchLog record (its trace stays in the entries' attrs).

```{.python .input #energy-cost-lib-driver}
%%tab pytorch
#@save
def regen_power(config='box'):
    """Driver of box-power: the energy-cost page's power profile, decode
    series and training series, one BenchLog record per window."""
    if not torch.cuda.is_available():
        raise RuntimeError('box-power needs a CUDA GPU')
    if config != 'box' or os.environ.get('D2L_HW_CONFIG'):
        raise ValueError('box-power has only the box configuration')
    folder = _hw_regen_dir('box-power')
    prof = d2l.power_profile(workloads=('idle', 'decode-b1', 'matmul-8192'),
                             seconds=10)
    e = d2l.decode_energy(d2l.hw_config('Qwen/Qwen3-0.6B'),
                          batches=2**np.arange(0, 8), L=(512, 2048))
    tr = d2l.train_energy(d2l.r2_config('r2').arch,
                          tokens_per_step=(8192, 32768), micro_batch=2)
    with d2l.BenchLog(os.path.join(folder, 'benchlog.jsonl'), 'torch',
                      _HW_TIERS[config]) as log:
        for cmd, df, keys in (('power_profile', prof, ('workload',)),
                              ('decode_energy', e, ('b', 'L')),
                              ('train_energy', tr, ('tokens_per_step',))):
            for rec in df.to_dict('records'):
                log.record(cmd, {k: rec[k] for k in keys}, rec)
```
