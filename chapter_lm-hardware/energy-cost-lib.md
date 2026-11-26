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
  box's power traces, the box's clock sweep, the A100 sweep's campaign
  configuration and chapter 23's measurements) and its regeneration jobs;
- the workloads a power trace runs (idle, CUDA-graph decode, a bf16 matrix
  product, a training step of the reference architecture) and the traced
  window (`_power_window`);
- the three measurements (`power_profile`, `decode_energy`,
  `train_energy`), each one `_hw_result('box-power', ...)` call with its
  display and live check;
- the clock sweep's configurations, its guards (`_clock_guard`, the 1 Hz
  `_ClockWatchdog`), the guarded `nvidia-smi` (`hw_smi_card`), the windows
  (`_sweep_rows`), the power-cap arm (`_cap_sweep`, off in every
  configuration), E4's edge rule (`_clock_edge`) and the readers and
  displays (`hw_clock_sweep_log`, `plot_dvfs`, `clock_effect`);
- the price table (`hw_prices`, `rent_per_hour`), the MFU and goodput
  readers (`training_mfu`, `training_goodput`), the serving points
  (`frontier_points`, `prefill_seconds`), the owned-versus-rented inputs,
  interval and tornado (`owned_inputs`, `u_star_interval`, `tornado_order`,
  `plot_tornado`) and the dimensional check (`check_cost_units`);
- the regeneration drivers `regen_power` and `hw_clock_regen` (which only
  `tools/hw_clock_sweep.sh` runs).

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
d2l.DATA_HUB['lm-hw-buildbox-power-v1'] = (
    d2l.LM_DATA_URL + 'lm-hw-buildbox-power-v1.tar',
    'bb1534b3ab70a875afa35916b673df4a30df420b')
```

```{.python .input #energy-cost-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['hw-prices'] = ('lm-hw-prices-v1', 'csv')
d2l.HW_ARTIFACTS['box-power'] = ('lm-hw-buildbox-power-v1', 'tar')
d2l.HW_ARTIFACTS['box-clocksweep'] = ('lm-hw-buildbox-clocksweep-v1', 'tar')
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
d2l.HW_JOBS['box-clocksweep'] = (
    'hw_clock_regen', lambda: dict(config='box'))
d2l.HW_CONFIG_LOGS[('box-clocksweep', 't2')] = 'a100-clocksweep'
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
def _hw_energy_j():
    """The first visible GPU's cumulative energy counter in J (NVML's
    nvmlDeviceGetTotalEnergyConsumption, Volta and later), or None where
    it cannot be read."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByUUID(_hw_visible_uuids()[0])
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1e3
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None

#@save
def _power_window(step, work, seconds=None, steps=None, warm_s=None,
                  hz=None):
    """Run step() (work units each) for warm_s seconds, then for `seconds`
    or `steps` under a power trace. A dict: rate (units/s), the median step
    time and its median absolute deviation (ms), the trace's mean power (W,
    trapezoid), median SM and memory clocks (MHz) and temperature, energy
    per unit (J) from the trace and, independently, from the card's energy
    counter, the clock-event fractions, whether a foreign process was on
    the card, and the trace itself."""
    warm_s = _HW_POWER['warm_s'] if warm_s is None else warm_s
    hz = hz or _HW_POWER['hz']
    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < warm_s:
        step()
        torch.cuda.synchronize()
    foreign = _hw_contention()[0]
    with _hw_power_trace(hz) as samples:
        E0, times, t0 = _hw_energy_j(), [], time.perf_counter()
        while (len(times) < steps if steps
               else time.perf_counter() - t0 < seconds):
            t = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        elapsed, E1 = time.perf_counter() - t0, _hw_energy_j()
    foreign += _hw_contention()[0]
    n, times = len(times), np.array(times)
    rate = n * work / elapsed
    P = np.array([s['power_w'] for s in samples], dtype=float)
    ts = np.array([s['t'] for s in samples], dtype=float)
    span = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    counted = work and E0 is not None and E1 is not None
    return dict(
        rate=rate, seconds=elapsed, steps=n,
        step_ms=1e3 * float(np.median(times)),
        step_mad_ms=1e3 * float(np.median(np.abs(times - np.median(times)))),
        power_w=(np.trapezoid(P, ts) / span if span else
                 float(np.mean(P)) if len(P) else math.nan),
        sm_mhz=_hw_median(samples, 'sm_mhz'),
        mem_mhz=_hw_median(samples, 'mem_mhz'),
        temp_c=_hw_median(samples, 'temp_c'),
        j_per_unit=(d2l.energy_per_token(samples, rate * span)
                    if work and span else math.nan),
        j_per_unit_counter=(E1 - E0) / (n * work) if counted else math.nan,
        contended=float(bool(foreign)), **_throttle_fractions(samples),
        trace=dict(t=ts.round(3).tolist(), power_w=P.round(1).tolist(),
                   sm_mhz=[s.get('sm_mhz') for s in samples]))

#@save
def _quiet_window(step, work, tries=3, **kwargs):
    """_power_window, repeated while a foreign process shared the card, up
    to tries windows; raises if every one was shared, so no contended
    window reaches a committed result."""
    for _ in range(tries):
        w = _power_window(step, work, **kwargs)
        if not w['contended']:
            return w
    raise RuntimeError(f'a foreign process shared the card in {tries} '
                       'windows in a row: regenerate when it is free')

#@save
def _split_traces(rows, key):
    """rows without their traces, and the traces keyed by key(row)."""
    traces = {str(key(r)): r.pop('trace') for r in rows}
    return rows, traces

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
_HW_WORKLOAD_NAMES = {'idle': 'idle (context open)',
                      'decode': 'decode, b = {}', 'matmul': 'matmul {}^3',
                      'train': 'training, {} tok'}

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
    print(f'{a["field"]} at {a["hz"]} Hz; power limit '
          f'{_hw_fmt(a["power_limit_w"])} W')

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
        live_label='matrix product, 3 s')
```

## Energy per Decode Token

```{.python .input #energy-cost-lib-decode}
%%tab pytorch
#@save
_HW_PER_TOKEN = {'rate': 'tokens_per_s', 'j_per_unit': 'j_per_token',
                 'j_per_unit_counter': 'j_per_token_counter'}

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
    df = _hw_frame(rows).rename(columns=_HW_PER_TOKEN)
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
             figsize=(5, 3.5), legend_loc='lower left')  # power: top right
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
        live_label='J/token at b = 1, 64')
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
    df = _hw_frame(rows).rename(columns=_HW_PER_TOKEN)
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
    gap = (tr.j_per_token_counter / tr.j_per_token - 1).abs().max()
    print(f'energy counter\'s J/token: within {gap:.1%} of the trace\'s '
          'W / (tokens/s)')

#@save
def _train_live(tr, same, seconds=3):
    arch = d2l.Arch(**tr.attrs['arch'])
    row = tr.iloc[0]
    tokens = int(row.tokens_per_step)
    step, _ = _train_load(arch, tokens, _HW_POWER['micro_batch'])
    for _ in range(3):
        step()
    w = _power_window(step, tokens, seconds=seconds, warm_s=1.0)
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
        live_label='training steps, 3 s')
```

## The Clock Sweep

The regeneration `box-clocksweep` locks one card's SM clock at each swept
setting (`nvidia-smi -lgc f,f`, ruling H1 of 2026-11-25), runs each workload
in a traced window and resets the clock (`-rgc`). Only
`tools/hw_clock_sweep.sh` runs it: under `gpu_run.sh` (the card's GPU lock,
no other user's process at the start), with `D2L_HW_SUDO=1`, so that `_HWSmi`
prefixes `sudo -n` to the clock commands, and with bash traps that reset the
card on any exit. The page's `hw_clock_sweep` shows the method (lock,
measure, `finally` reset); the guards live here:

- `_clock_guard`, before every lock and at each window's start and end: any
  compute process on the card but this one resets the card at once, the
  window is discarded, and the sweep waits in 60 s polls (at most 2 h, then
  fails with the card reset) and re-locks before it resumes;
- `_ClockWatchdog`, a thread that checks the card's compute processes once a
  second while the clock is locked: a foreign one resets the card within a
  second and flags the window, which `_sweep_rows` discards (H1: no other
  user's process runs at a locked clock for more than about a second); the
  two share one lock, so they never run `nvidia-smi` at the same time;
- `_sweep_rows` raises when the driver refused the setting, and discards a
  window whose SM clock ran more than 30 MHz above its lock (a lock that did
  not take: a locked card never runs above it, a power-capped one runs
  below), applies the lock again and measures it again; a dry run, which
  locks nothing, raises there, so it never yields lock rows;
- `hw_clock_regen` turns SIGTERM, SIGHUP and SIGINT into `sys.exit`, so a
  plain kill still runs every `finally` reset; the wrapper's traps and one
  `-rgc` after the job cover a `kill -9`.

The power-cap arm (`_cap_sweep`, `nvidia-smi -pl`) runs only when a
configuration's `caps` is not empty; ruling H1 covers clocks only, so every
configuration has `caps=()` until the author allows `-pl`. With the arm on,
`_HWSmi` carries the default limit and every reset restores it too.

E4's edge rule is pre-registered: if the matrix product's least energy per
FLOP is at the lowest lock, the same job sweeps one step lower (`edge`); the
first process decides, the others follow it, so the three processes' tables
combine row by row.

```{.python .input #energy-cost-lib-sweep}
%%tab pytorch
#@save
_HW_CLOCK_CONFIGS = {  # caps=(): no power limit is set (H1 covers clocks)
    'box': dict(clocks=tuple(range(600, 3001, 300)), edge=300,
                workloads=('matmul-8192', 'decode-b1', 'train-8192'),
                caps=()),  # with -pl allowed: (150, 225, 300, 375, 450)
    'smoke': dict(clocks=(1500,), edge=None,
                  workloads=('decode-b1', 'matmul-8192'), caps=()),
    't2': dict(clocks=tuple(range(360, 1411, 150)), edge=210,  # A100 PCIe
               workloads=('matmul-8192', 'decode-b1', 'train-8192'),
               caps=())}

#@save
_HW_GUARD = dict(poll_s=60, max_wait_s=7200, watch_s=1.0, tries=10,
                 lock_tol_mhz=30)  # a locked card never runs above its lock

#@save
_HW_SMI_CALLS = []  # every nvidia-smi setting of this process: the record

#@save
_HW_GUARD_EVENTS = []  # every foreign process the guards saw, every
                       # window that ran above its lock

#@save
def _card_pids(uuid):
    """PIDs of the compute processes on the card with this UUID (NVML,
    else nvidia-smi), or None when neither can be read."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            return {p.pid for p in
                    pynvml.nvmlDeviceGetComputeRunningProcesses(h)}
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        rows = _hw_smi('--query-compute-apps=gpu_uuid,pid')
        if not rows and not _hw_smi('--query-gpu=uuid'):
            return None  # nvidia-smi fails too
        return {int(r[1]) for r in rows
                if len(r) > 1 and r[0] == uuid and r[1].isdigit()}

#@save
def _card_foreign(smi):
    """Compute processes on smi's card other than this one ([-1] when they
    cannot be read: an unknown card counts as busy)."""
    pids = _card_pids(smi.uuid)
    if pids is None:
        return [-1]
    return sorted(p for p in pids if p != os.getpid())

#@save
def _clock_guard(smi, relock=None, dog=None):
    """True when smi's card holds no compute process but this one and the
    watchdog dog (if any) has not tripped. Else reset the card at once
    (smi.reset), wait in polls until it is free (at most max_wait_s, then
    raise with the card reset), apply relock (nvidia-smi arguments, the
    window's setting; a refusal raises), re-arm dog and return False: the
    caller discards its window. The dog's lock is held from the reset to
    the re-arm, so the watchdog never runs nvidia-smi at the same time."""
    if smi is None:
        return True
    pids = _card_foreign(smi)
    if not pids and not (dog is not None and dog.tripped):
        return True
    with dog.lock if dog is not None else contextlib.nullcontext():
        smi.reset()
        t0 = time.monotonic()
        if pids:  # (else the watchdog tripped and recorded its sighting)
            _HW_GUARD_EVENTS.append(dict(t=time.time(), by='guard',
                                         pids=pids))
        while pids:
            if time.monotonic() - t0 > _HW_GUARD['max_wait_s']:
                raise RuntimeError(f'a foreign process stayed on the card '
                                   f'for {_HW_GUARD["max_wait_s"] / 3600:g} '
                                   'h; the card is reset')
            time.sleep(_HW_GUARD['poll_s'])
            pids = _card_foreign(smi)
        if relock and smi(*relock).refused:
            raise PermissionError(f'the driver refused nvidia-smi '
                                  f'{" ".join(relock)} after a reset')
        if dog is not None:
            dog.rearm()
    return False

#@save
class _ClockWatchdog:
    """While a setting is applied: once every watch_s seconds the card's
    compute processes; a foreign one resets the card at once (smi.reset)
    and trips the dog, which then stays tripped (no further resets) until
    _clock_guard re-arms it on a free, re-locked card. trips counts the
    trips, so a window that saw one is known; lock serializes its check
    and reset with the guard's reset and re-lock. smi=None: a dog that
    never trips."""
    def __init__(self, smi):
        import threading
        self.smi, self.trips, self.lock = smi, 0, threading.Lock()
        self._armed, self._stop = threading.Event(), threading.Event()
        self._armed.set()
        self._thread = None
        if smi is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    @property
    def tripped(self):
        return not self._armed.is_set()

    def rearm(self):
        self._armed.set()

    def _run(self):
        while not self._stop.wait(_HW_GUARD['watch_s']):
            with self.lock:
                if not self._armed.is_set():
                    continue
                pids = _card_foreign(self.smi)
                if pids:
                    self.smi.reset()
                    self._armed.clear()
                    self.trips += 1
                    _HW_GUARD_EVENTS.append(dict(t=time.time(),
                                                 by='watchdog', pids=pids))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

#@save
def _sweep_relock(kind, setting):
    """The nvidia-smi arguments that apply a sweep row's setting (None for
    the default clocks)."""
    if kind == 'clock':
        return ('-lgc', f'{int(setting)},{int(setting)}')
    if kind == 'cap':
        return ('-pl', str(int(setting)))
    return None

#@save
def _clock_held(w, setting):
    """None when window w of a clock lock ran at or below its lock (MHz,
    within lock_tol_mhz; a power cap may hold it lower), else why not."""
    ran = w.get('sm_mhz')
    if ran is None or not np.isfinite(ran):
        return f'the clock as run was not read at the {setting:.0f} MHz lock'
    if ran > setting + _HW_GUARD['lock_tol_mhz']:
        return (f'the SM clock ran at {ran:.0f} MHz, above its '
                f'{setting:.0f} MHz lock')
    return None

#@save
def _sweep_rows(kind, setting, workloads, smi=None, seconds=10):
    """One row per workload at one sweep setting (kind 'clock': a locked
    clock in MHz, 'cap': a power limit in W, 'default': the default
    clocks): the traced window with its median step time and that time's
    median absolute deviation. With smi (the card's nvidia-smi), the
    guards run: a window that a foreign process shared, before, during or
    at its end, is discarded (column 'discarded' counts them) and measured
    again once the card is free and the setting re-applied. A refused
    setting raises; a clock window that ran above its lock is discarded
    and the lock applied again (a dry run, which locks nothing, raises)."""
    relock, rows = _sweep_relock(kind, setting), []
    with _ClockWatchdog(smi) as dog:
        for name in workloads:
            step, work, unit = _power_load(name)
            discarded = 0
            while True:
                _clock_guard(smi, relock, dog)
                if smi is not None and smi.last_refused:
                    raise PermissionError(f'the driver refused nvidia-smi '
                                          f'{" ".join(smi.last_refused)}')
                trips = dog.trips
                w = _power_window(step, work, seconds=seconds)
                why = 'shared the card'
                if (dog.trips == trips and not w['contended']
                        and _clock_guard(smi, relock, dog)):
                    held = _clock_held(w, setting) if kind == 'clock' else None
                    if held is None:
                        break
                    if os.environ.get('D2L_HW_DRYRUN') == '1':
                        raise RuntimeError(f'dry run: {held}; nothing was '
                                           'locked, so no lock row is kept')
                    why = held
                    _HW_GUARD_EVENTS.append(dict(t=time.time(), by='as run',
                                                 lock=setting,
                                                 sm_mhz=w.get('sm_mhz')))
                    if smi is not None:
                        with dog.lock:
                            if smi(*relock).refused:
                                raise PermissionError(
                                    'the driver refused nvidia-smi '
                                    f'{" ".join(relock)}')
                discarded += 1
                if discarded >= _HW_GUARD['tries']:
                    raise RuntimeError(f'{discarded} windows in a row were '
                                       f'discarded (last: {why}); regenerate '
                                       'when the card is free and locks')
            w.pop('trace')
            rows.append(dict(kind=kind, setting=setting, workload=name,
                             unit=unit, discarded=discarded, **w))
            del step
            d2l.lm_free()
    return rows

#@save
class _HWSmi:
    """nvidia-smi on one visible GPU, addressed by its UUID. smi(*args)
    applies a setting (-lgc, or -pl while the power-cap arm runs) after
    _clock_guard has found the card free, and returns the HWRun (a refusal
    is a result here, not an error, and last_refused names the setting
    until the next one is applied); smi.restore(*args) runs a reset and
    prints one line if it failed; smi.reset() puts the card back at once
    (-rgc, and the default power limit when default_limit is set, which
    only the cap arm does). With D2L_HW_SUDO=1 (set only by
    tools/hw_clock_sweep.sh) -lgc and -rgc, and -pl with the arm on, run
    as `sudo -n nvidia-smi`; -pl without the arm is refused here.
    D2L_HW_DRYRUN=1 prints each sudo command instead of running it."""
    def __init__(self, device=0, default_limit=None):
        if device != 0:
            raise ValueError('make the card visible GPU 0 '
                             '(CUDA_VISIBLE_DEVICES): the power trace reads '
                             'that one')
        self.uuid = _hw_visible_uuids()[device]
        self.default_limit, self.last_refused = default_limit, None

    def _run(self, *args):
        if args[0] == '-pl' and self.default_limit is None:
            raise PermissionError('setting a power limit: the power-cap arm '
                                  'is off in this configuration')
        cmd = ['nvidia-smi', '-i', self.uuid, *args]
        if (os.environ.get('D2L_HW_SUDO') == '1'
                and args[0] in ('-lgc', '-rgc', '-pl')):
            cmd = ['sudo', '-n', *cmd]
        if cmd[0] == 'sudo' and os.environ.get('D2L_HW_DRYRUN') == '1':
            print('+ ' + ' '.join(cmd), flush=True)
            run = HWRun(cmd, 0, '', '', 0.0, None)
        else:
            run = hw_run(cmd, 60, record=True)
        _HW_SMI_CALLS.append(dict(t=time.time(), args=list(args),
                                  refused=run.refused))
        if args[0] in ('-lgc', '-pl'):  # a setting: was it applied?
            self.last_refused = tuple(args) if run.refused else None
        return run

    def __call__(self, *args):
        if args[0] == '-lgc' or (args[0] == '-pl' and self.default_limit
                                 and float(args[1]) != self.default_limit):
            _clock_guard(self)
        return self._run(*args)

    def restore(self, *args):
        run = self._run(*args)
        if run.refused:
            print(f'nvidia-smi {" ".join(args)} failed ({run.refused}): '
                  'reset the card by hand')
        return run

    def reset(self):
        runs = [self.restore('-rgc')]
        if self.default_limit:
            runs.append(self.restore('-pl', f'{self.default_limit:g}'))
        return runs

#@save
def hw_smi_card(device=0, default_limit=None):
    """nvidia-smi on visible GPU device (which must be 0, the card the power
    trace reads), guarded (_HWSmi): smi(*args) applies a setting,
    smi.restore(*args) and smi.reset() reset loudly."""
    return _HWSmi(device, default_limit)

#@save
def _cap_sweep(caps, workloads, seconds=10):
    """Rows (kind 'cap') of each workload under each power limit (W) on
    visible GPU 0 at the default clocks, the default limit restored at the
    end and by every guard's reset. Refuses to start while the default
    limit is unknown (no way back)."""
    default = _hw_gpu_limits().get('power_default_limit')
    if not default:
        raise RuntimeError('the default power limit is unknown: no cap is '
                           'set without a way back')
    smi = hw_smi_card(0, default_limit=float(default))
    rows = []
    try:
        for cap in caps:
            if smi('-pl', str(int(cap))).refused:
                raise PermissionError('the driver refused the power limit')
            rows += _sweep_rows('cap', cap, workloads, smi, seconds)
    finally:
        smi.reset()
    return rows

#@save
def _least_energy_lock(log):
    """The locked clock (MHz) at which the matrix product used the least
    energy per unit of work (W / rate), or None without such rows."""
    s = log[(log.kind == 'clock') & log.workload.str.startswith('matmul')]
    if not len(s):
        return None
    return float(s.setting.iloc[int(np.argmin(s.power_w / s.rate))])

#@save
def _clock_edge(rows, clocks, edge):
    """E4's edge rule: [edge] when the matrix product's least energy per
    FLOP lies at the lowest of clocks, else []. Only the first process of
    a regeneration decides; the others repeat its choice (read from its
    entry), so their tables combine row by row."""
    if edge is None:
        return []
    proc = os.environ.get('D2L_HW_PROC')
    if proc not in (None, '0'):
        first = os.path.join(os.path.dirname(_hw_regen_dir('box-clocksweep')),
                             'proc0', _hw_entry_file('sweep'))
        done = _hw_read(first)[0]
        return [edge] if ((done.kind == 'clock')
                          & (done.setting == edge)).any() else []
    least = _least_energy_lock(pd.DataFrame(rows))
    return [edge] if least == min(clocks) else []

#@save
def hw_clock_sweep_log():
    """The committed clock sweep of the box (box-clocksweep, entry 'sweep'),
    with campaign C-T2's A100 sweep in attrs['a100'] once committed (else
    its CampaignPending) and the run in attrs['source']. The entry must be
    a box run of _HW_CLOCK_CONFIGS['box'] that locked clocks (no smoke, no
    dry run): else ValueError. Absent as every box result (hw_box_log)."""
    loaded = _hw_load('box-clocksweep', 'sweep')
    if loaded is None:
        log = hw_box_log('box-clocksweep', 'sweep')  # pending, or KeyError
    else:
        log, meta = loaded
        want = d2l._jsonable(dict(config='box', **_HW_CLOCK_CONFIGS['box']))
        d = d2l._config_diff(want, meta.get('config'))
        key = d2l.HW_ARTIFACTS['box-clocksweep'][0]
        if d:
            raise ValueError(f'page configuration differs from the committed '
                             f'run {key}: {d[0]} {d[1]!r} != {d[2]!r}')
        dry = bool(meta.get('dryrun') or log.attrs.get('dryrun'))
        if meta.get('run') != 'box' or dry:
            raise ValueError(f'{key} is not a box run that locked clocks '
                             f'(run {meta.get("run")!r}, dry run {dry})')
    a100 = hw_campaign_log('a100-clocksweep')
    log.attrs['a100'] = a100.get('sweep') if a100 else a100
    if len(log):
        n = log.attrs.get('n_proc', 1)
        log.attrs['source'] = (f'committed sweep on the {log.attrs["gpu"]} '
                               f'({n} process{"es" if n > 1 else ""})')
    return log

#@save
def plot_dvfs(fit):
    """Board power (left) and work per joule relative to the top lock
    (right, squares; the model dashed) against the SM clock each window
    ran at: the sweep's points, the fitted model, f* dotted, the least
    energy marked. Prints the fit, f* against the least-energy lock and
    the sweep step, the locks the product ran below (its power cap), and
    the source."""
    f = np.linspace(min(fit.clocks) * 0.9, max(fit.clocks) * 1.05, 200)
    power = lambda x: fit.P_s + fit.k * x**3
    top = int(np.argmax(fit.locks))
    eta = fit.rate / fit.power / (fit.rate[top] / fit.power[top])
    model = f / power(f) / (fit.clocks[top] / power(fit.clocks[top]))
    d2l.plot([f, fit.clocks], [power(f), fit.power],
             xlabel='SM clock as run (GHz)', ylabel='board power (W)',
             fmts=['C0-', 'C0o'], legend=['P_s + k f^3', 'power'],
             figsize=(5, 3.5), legend_loc='lower right')
    ax = d2l.plt.gca()
    ax.axvline(fit.f_star, color='k', ls=':', lw=1)
    ax2 = ax.twinx()
    ax2.plot(f, model, 'C1--', lw=1)
    ax2.plot(fit.clocks, eta, 'C1s', ms=4)
    i = int(np.argmax(eta))
    ax2.plot(fit.clocks[i], eta[i], 'k*', ms=10)  # the least energy
    ax2.set_ylim(0, 1.2 * max(model.max(), eta.max()))
    ax2.set_ylabel('work per joule (top lock = 1)')
    spread = (f'; power within {fit.spread:.1%} across processes'
              if np.isfinite(fit.spread) else '')
    print(f'fit to the clock as run: P_s = {fit.P_s:.0f} W, k = '
          f'{fit.k:.3g} W/GHz^3{spread}')
    where = (f'the lowest lock, {fit.f_min:.2f} GHz: at or below it'
             if fit.edge else f'the {fit.f_min:.2f} GHz lock')
    print(f'f* = {fit.f_star:.2f} GHz; least energy per FLOP at {where} '
          f'(step {fit.step:.2f} GHz)')
    below = [f'{a:.2f} -> {b:.2f}' for a, b in zip(fit.locks, fit.clocks)
             if b < a - 0.03]
    print('below its lock (power cap), GHz: ' + (', '.join(below) or 'none'))
    print(f'{fit.source}; a notebook never locks clocks')
    if not isinstance(fit.a100, pd.DataFrame):
        print(hw_status_line(fit.a100) or 'A100: no sweep')
    else:
        a = d2l.dvfs_fit(fit.a100)
        print(f'A100 (C-T2): P_s = {a.P_s:.0f} W, k = {a.k:.3g} W/GHz^3, '
              f'f* = {a.f_star:.2f} GHz, least energy at {a.f_min:.2f} GHz')

#@save
def _clock_column(log, kind, workload):
    rows = log[(log.kind == kind) & (log.workload == workload)]
    if not len(rows):
        raise ValueError(f'the sweep log has no {workload} rows of kind '
                         f'{kind!r}')
    return rows.set_index('setting').sort_index(ascending=False)

#@save
def _step_spread(log):
    """The between-process relative spread (max - min) / median of each
    row's step time (attrs['spread'] of _hw_combine, aligned on the
    combined table's rows; 0 where the processes agreed or ran once)."""
    s = (log.attrs.get('spread') or {}).get('step_ms')
    if s is None:
        return pd.Series(0.0, index=log.index)
    s = pd.Series(np.nan_to_num(np.asarray(s, float), nan=0.0))
    return s.reindex(log.index).fillna(0.0)

#@save
def _effect_table(log, kind, train, decode):
    """Per setting of kind (descending): the training step's clock as run,
    power and step time, the batch-1 decode step time, each time as a
    ratio x to the default-clock run, and its noise (R4: the lock and the
    default rows were not run back to back, the default clocks unlocked):
    the larger of 5 x the larger median absolute deviation over the
    default's step and either row's spread across processes."""
    log = log.assign(step_spread=_step_spread(log))
    t, d = _clock_column(log, kind, train), _clock_column(log, kind, decode)
    t0 = _clock_column(log, 'default', train).iloc[0]
    d0 = _clock_column(log, 'default', decode).iloc[0]
    noise = lambda g, g0: np.maximum(np.maximum(
        5 * np.maximum(g.step_mad_ms, g0.step_mad_ms) / g0.step_ms,
        g.step_spread), g0.step_spread)  # a Series on g's settings
    return pd.DataFrame({
        'clock MHz' if kind == 'clock' else 'cap W': t.index.astype(int),
        'train MHz': t.sm_mhz.values, 'train W': t.power_w.values,
        'train ms': t.step_ms.values,
        'train x': (t.step_ms / t0.step_ms).values,
        'train noise': noise(t, t0).values,
        'decode ms': d.step_ms.reindex(t.index).values,
        'decode x': (d.step_ms / d0.step_ms).reindex(t.index).values,
        'decode noise': noise(d, d0).reindex(t.index).values}), t0, d0

#@save
_HW_EFFECT_DIGITS = {'clock MHz': 0, 'cap W': 0, 'train MHz': 0,
                     'train W': 0, 'train ms': 1, 'train x': 3,
                     'train noise': 3, 'decode ms': 2, 'decode x': 3,
                     'decode noise': 3}

#@save
def clock_effect(log, train='train-8192', decode='decode-b1'):
    """What a lower SM clock costs (E5): the training step and batch-1
    decode at each locked clock against the default-clock run (and, if
    the power-cap arm ran, the same under each cap). attrs['decode_floor']:
    the lowest lock down to which every lock keeps decode within its
    noise (None if decode slows at every lowered clock; noise as in
    _effect_table)."""
    table, t0, d0 = _effect_table(log, 'clock', train, decode)
    d2l.show_table(table, digits=_HW_EFFECT_DIGITS)
    print(f'x: step time over the default clocks\' (training {t0.step_ms:.0f}'
          f' ms at {t0.sm_mhz:,.0f} MHz, decode {d0.step_ms:.2f} ms)')
    print('noise: the larger of 5 x the larger median absolute deviation '
          'and the spread across processes')
    floor = None
    for _, r in table.iterrows():  # from the top lock down
        if not np.isfinite(r['decode x']):
            raise ValueError(f'the sweep has no {decode} row at the '
                             f'{r["clock MHz"]:,.0f} MHz lock')
        if r['decode x'] - 1 > r['decode noise']:
            break
        floor = int(r['clock MHz'])
    table.attrs['decode_floor'] = floor
    if floor is None:
        print('decode at b = 1 slows beyond its noise at every lowered clock')
    else:
        r = table.set_index('clock MHz').loc[floor]
        print(f'decode at b = 1 within its noise down to {floor:,} MHz; the '
              f'training step there takes {r["train x"]:.2f}x as long')
    if (log.kind == 'cap').any():
        caps = _effect_table(log, 'cap', train, decode)[0]
        d2l.show_table(caps, digits=_HW_EFFECT_DIGITS)
        table.attrs['caps'] = caps
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
    prices = _hw_static('hw-prices')
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
_HW_STATED_MFU = 0.4  # Llama 3 reports 0.38-0.43 bf16 MFU on H100 (Table 4)

#@save
def training_mfu(chips=('A100 SXM 80 GB', 'H100 SXM', 'B200'),
                 stated=_HW_STATED_MFU):
    """One row per (model, chip): the MFU a training-cost estimate uses,
    its context T and its source. The reference model on the RTX 4090:
    chapter 21's four-card run when registered, else this section's
    one-card measurement (box-power, the median of its step sizes).
    Qwen3-8B on the RTX 4090: 22.3's one-card depth series, an upper bound
    (no communication). Qwen3-8B on each of chips: chapter 23's mfu-table
    when registered (records with model, chip, mfu, T; the causal half
    counted), else the stated value at the same context: an RTX 4090
    number is never attached to another chip."""
    arch = d2l.r2_config('r2').arch
    ref = _reference_run_mfu(arch)
    if ref is not None:
        rows = [dict(model='reference', chip='RTX 4090', mfu=ref, T=arch.T,
                     source='4 x RTX 4090, chapter 21')]
    else:
        tr = hw_box_log('box-power', 'train')
        rows = [dict(model='reference', chip='RTX 4090',
                     mfu=float(tr.mfu.median()), T=arch.T,
                     source='1 x RTX 4090, this section')]
    box, T = _box_qwen8b_mfu(d2l.hw_config('Qwen/Qwen3-8B'))
    rows.append(dict(model='Qwen3-8B', chip='RTX 4090', mfu=box, T=T,
                     source='1 x RTX 4090, upper bound'))
    table = hw_campaign_log('mfu-table')
    measured = {(r['model'], r['chip']): r for r in table} if table else {}
    for chip in chips:
        r = measured.get(('Qwen3-8B', chip))
        if r is not None:
            rows.append(dict(model='Qwen3-8B', chip=chip, mfu=r['mfu'],
                             T=r['T'], source='chapter 23, measured'))
        else:
            rows.append(dict(model='Qwen3-8B', chip=chip, mfu=stated, T=T,
                             source='stated; chapter 23 measures it'))
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
_OWNED_LABELS = dict(capex='purchase', life_h='life', power_kw='power',
                     pue='PUE', elec_per_kwh='electricity',
                     ops_per_h='operations', rent_per_h='rent')

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
        ax.set_yticks(range(len(order)), [_OWNED_LABELS[k] for k, _ in order])
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

## The Clock-Sweep Driver

`tools/hw_clock_sweep.sh <gpu>` runs `tools/hw_regen.py box-clocksweep` (three
processes, about 30 min on one card) with `D2L_HW_SUDO=1`; each process calls
`hw_clock_regen`, which writes one entry, `sweep`: the default-clock rows,
then one row per lock and workload (and per cap, if the arm is on), with the
record of every `nvidia-smi` setting and of every foreign process the guards
saw in its attrs. Before the three entries are combined,
`hw_clock_regen_prepare` moves each process's record into `attrs['procs']`
and sums each row's discarded windows. `--config smoke` (one lock) writes
only under `--out` and is never packed, nor is a dry run
(`D2L_HW_DRYRUN=1`); `--config t2` is campaign C-T2's A100 sweep.

```{.python .input #energy-cost-lib-the-clock-sweep-driver}
%%tab pytorch
#@save
def _clock_config_check(config):
    """Raise unless config is a clock-sweep configuration this process may
    run: 'box' and 't2' as tools/hw_regen.py sets D2L_HW_CONFIG, 'smoke'
    only into a folder of its own (--out)."""
    if config not in _HW_CLOCK_CONFIGS:
        raise ValueError(f'no clock-sweep configuration {config!r}; one of '
                         f'{sorted(_HW_CLOCK_CONFIGS)}')
    run = os.environ.get('D2L_HW_CONFIG') or 'box'
    if config == 'smoke':
        if run != 'box' or not os.environ.get('D2L_HW_OUT'):
            raise ValueError('the smoke configuration writes only into a '
                             'folder of its own (tools/hw_regen.py --config '
                             'smoke --out DIR)')
    elif config != run:
        raise ValueError(f'config {config!r} needs D2L_HW_CONFIG={config} '
                         '(tools/hw_regen.py --config)')

#@save
def _stop_on_signals():
    """SIGTERM, SIGHUP and SIGINT end this process through sys.exit, so every
    finally (the clock reset) runs; a second signal is ignored, so it cannot
    interrupt the reset."""
    import signal

    def stop(signum, frame):
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(s, signal.SIG_IGN)
        sys.exit(1)
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, stop)

#@save
def _clock_sweep_compute(cfg):
    """The sweep of one configuration on visible GPU 0: the default clocks,
    every lock of cfg (and E4's edge lock if the rule asks for it), then
    the power-cap arm if cfg['caps'] is not empty."""
    del _HW_SMI_CALLS[:], _HW_GUARD_EVENTS[:]
    smi, work = hw_smi_card(0), list(cfg['workloads'])
    if smi.restore('-rgc').refused:
        raise PermissionError('the driver refused to reset the SM clock: run '
                              'tools/hw_clock_sweep.sh')
    rows = _sweep_rows('default', math.nan, work, smi)
    rows += d2l.hw_clock_sweep(list(cfg['clocks']), work).to_dict('records')
    edge = _clock_edge(rows, cfg['clocks'], cfg['edge'])
    if edge:
        rows += d2l.hw_clock_sweep(edge, work).to_dict('records')
    if cfg['caps']:
        rows += _cap_sweep(cfg['caps'], work)
    df = _hw_frame(rows)
    df.attrs.update(edge=bool(edge), field='power.draw (NVML, 1 s average)',
                    hz=_HW_POWER['hz'], **_power_limits(),
                    dryrun=os.environ.get('D2L_HW_DRYRUN') == '1',
                    smi=list(_HW_SMI_CALLS), guard=list(_HW_GUARD_EVENTS))
    return df

#@save
def hw_clock_regen_prepare(folder, n):
    """Before tools/hw_regen.py combines processes proc0..proc{n-1}: every
    process's record (its nvidia-smi settings, guard events and power
    limits) goes into attrs['procs'] of each part, and each row's
    `discarded` becomes its sum over the processes, so the combined entry
    keeps all of them (the combination keeps one process's attrs)."""
    paths = [os.path.join(folder, f'proc{i}', _hw_entry_file('sweep'))
             for i in range(n)]
    parts = [d2l._hw_read(p) for p in paths]
    procs = [dict(smi=t.attrs.pop('smi', []), guard=t.attrs.pop('guard', []),
                  **{k: t.attrs.get(k) for k in ('power_limit_w',
                                                 'max_sm_mhz')})
             for t, _ in parts]
    same = all(len(t) == len(parts[0][0]) for t, _ in parts)
    total = (np.sum([t.discarded.to_numpy() for t, _ in parts], axis=0)
             if same else None)
    for (t, meta), path in zip(parts, paths):
        t.attrs['procs'] = procs
        if total is not None:  # (else the combination refuses the parts)
            t['discarded'] = total
        d2l._hw_write(path, t, meta)

#@save
def hw_clock_regen(config='box'):
    """Driver of box-clocksweep (only tools/hw_clock_sweep.sh runs it,
    through tools/hw_regen.py, with D2L_HW_SUDO=1): the entry 'sweep' of
    _clock_sweep_compute on the configuration's clocks and workloads;
    config 't2' is campaign C-T2's A100 sweep and runs only on that chip,
    'smoke' one lock into a folder of its own."""
    if not torch.cuda.is_available():
        raise RuntimeError('box-clocksweep needs a CUDA GPU')
    _clock_config_check(config)
    _hw_check_hardware('box-clocksweep', 't2' if config == 't2' else 'box')
    _stop_on_signals()
    cfg = _HW_CLOCK_CONFIGS[config]
    return _hw_result('box-clocksweep', lambda: _clock_sweep_compute(cfg),
                      dict(config=config, **cfg), entry='sweep')
```
