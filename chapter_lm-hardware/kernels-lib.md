# Kernels, Tiles and Fusion: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`. It is in `LIB_ONLY_EARLY_FILES` (emitted before the
numbered chapters): the Triton kernels of :numref:`sec_lm_kernels` are shown
on the page unchanged, with `@triton.jit` and `tl.constexpr` evaluated when
`d2l.torch` is imported, so `triton` and `tl` must be bound first. On a
machine without Triton (macOS) both are bound to an inert stand-in: the
kernels are defined, and launching one raises
`RuntimeError('Triton is not installed')`. It holds no registry entry: the
section's measurements are entries of 22.1's `box-microbench` result.

It also holds what :numref:`sec_lm_kernels` explains in a sentence but does
not show (docs/code-guide.md 7.1): the SDPA flash wrapper (`sdpa_causal`),
the attention inputs, shapes and FP32 reference (`attention_inputs`,
`attention_shapes`, `attention_error`, `check_close`), the timed sweeps whose
committed result the page reads (`time_attention`, `time_fns`,
`hw_shape_sweep`, `hw_wave_sweep`, `flash_sweep`; each one `_hw_result` call
with entry `three-ways`, `rmsnorm`, `shape-sweep`, `wave` or `flash-sweep`),
the compiled-kernel readers (`triton_spills`, `inductor_kernels`), the
displays and live checks, and `_regen_kernels`, the section's part of the
`box-microbench` regeneration. Blocks are tagged `%%tab pytorch`; they run
only when called, so they may use names defined later in `d2l.torch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_kernels`

## The Triton Binding

```{.python .input #kernels-lib-no-triton}
%%tab pytorch
#@save
class _TritonMissing:
    """A kernel defined without Triton: launching it raises."""
    def __getitem__(self, grid):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError('Triton is not installed')

#@save
class _NoTriton:
    """Stands in for triton and triton.language without Triton: decorators
    (jit, autotune, heuristics) give kernels that raise when launched, and
    every other attribute (tl.constexpr, tl.float32, triton.Config) is
    inert."""
    def __getattr__(self, name):
        return _NoTriton()

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _TritonMissing()  # used as a decorator
        return _NoTriton()

    def jit(self, fn=None, **kwargs):
        return _TritonMissing() if fn is not None else self

#@save
def _triton_modules():
    """(triton, triton.language), or two stand-ins without Triton."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return _NoTriton(), _NoTriton()
    return triton, tl
```

```{.python .input #kernels-lib-bind}
%%tab pytorch
#@save
triton = _triton_modules()[0]
tl = _triton_modules()[1]
```

## Attention Inputs, Shapes and the Library Kernel

The page's attention shapes are `{name: (configuration, batch)}`, the
configuration an `Arch` or a Transformers config. SDPA runs only its flash
backend: a shape that backend refuses raises instead of falling back to the
math kernel, so "within 2x of the library kernel" always compares against the
flash kernel.

```{.python .input #kernels-lib-inputs}
%%tab pytorch
#@save
def _attn_dims(cfg):
    """(H, H_kv, d_h) of an Arch or a Transformers config."""
    L, d, H, H_kv, d_h, f, V, tie = d2l._arch_fields(cfg)
    return H, H_kv, d_h

#@save
def attention_inputs(cfg, batch, seq, dtype=torch.bfloat16, seed=0,
                     device='cuda'):
    """Random q (batch, H, seq, d_h) and k, v (batch, H_kv, seq, d_h) of a
    configuration, seeded."""
    H, H_kv, d_h = _attn_dims(cfg)
    g = torch.Generator(device=device).manual_seed(seed)
    make = lambda h: torch.randn(batch, h, seq, d_h, dtype=dtype,
                                 device=device, generator=g)
    return make(H), make(H_kv), make(H_kv)

#@save
def sdpa_causal(q, k, v):
    """Causal attention through SDPA's FLASH_ATTENTION backend only (GQA
    when k and v have fewer heads than q); a shape that backend refuses
    raises, never a silent fallback to the math kernel."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=q.shape[1] != k.shape[1])

#@save
def attention_shapes(shapes, seq, l2_bytes, b=2):
    """Print each attention shape at seq tokens: batch, query and key-value
    heads, head dimension, and the K and V bytes of one key-value head (the
    bytes every query tile of a fused kernel re-reads) against an L2 of
    l2_bytes."""
    rows = {}
    for name, (cfg, batch) in shapes.items():
        H, H_kv, d_h = _attn_dims(cfg)
        kv = 2 * seq * d_h * b  # K and V of one head
        rows[name] = dict(B=batch, H=H, H_kv=H_kv, d_h=d_h,
                          kv_mib=kv / 2**20, l2=f'{kv / l2_bytes:.1%}')
    print(f'L2 of this GPU: {l2_bytes / 2**20:g} MiB')
    show_table(pd.DataFrame.from_dict(rows, orient='index'),
               digits={'kv_mib': 0}, headers={
                   'kv_mib': 'K+V per head (MiB)', 'l2': 'of L2'})

#@save
def check_close(actual, expected, atol):
    """Print the largest absolute difference and raise beyond atol."""
    err = float((actual.float() - expected.float()).abs().max())
    print(f'max abs error {err:.1e} (tolerance {atol:g})')
    if not err <= atol:
        raise AssertionError(f'max abs error {err:.3g} > {atol:g}')

#@save
def _attention_fp32(q, k, v, start, chunk):
    """FP32 causal attention of the queries start:start + chunk of one
    batch element (q (H, N, d), k and v (H, N, d) with GQA already
    expanded)."""
    Q = q[:, start:start + chunk].float()
    a = Q @ k.float().transpose(-2, -1) / math.sqrt(q.shape[-1])
    rows = torch.arange(start, start + Q.shape[1], device=q.device)[:, None]
    cols = torch.arange(k.shape[1], device=q.device)
    a.masked_fill_(cols > rows, float('-inf'))
    return torch.softmax(a, dim=-1) @ v.float()

#@save
def attention_error(fn, shape, seq, name='', band=2e-2, chunk=1024,
                    seed=0):
    """Largest |fn(q, k, v) - FP32 reference| over the largest |reference|
    for one attention shape (config, batch) at seq tokens; the reference
    runs in query chunks (an FP32 score matrix of 32 heads x 4,096^2 is
    2.1 GB). Prints one line and raises beyond band."""
    cfg, batch = shape
    q, k, v = attention_inputs(cfg, batch, seq, seed=seed)
    with d2l._quiet():  # a first Triton launch compiles
        out = fn(q, k, v)
    G = q.shape[1] // k.shape[1]
    err = peak = 0.0
    for i in range(batch):
        K, V = k[i].repeat_interleave(G, 0), v[i].repeat_interleave(G, 0)
        for start in range(0, seq, chunk):
            ref = _attention_fp32(q[i], K, V, start, chunk)
            got = out[i, :, start:start + chunk].float()
            err = max(err, float((got - ref).abs().max()))
            peak = max(peak, float(ref.abs().max()))
    print(f'{name}: max |error| / max |output| = {err / peak:.1e} '
          f'(band {band:g})')
    if not err / peak <= band:
        raise AssertionError(f'{name}: relative error {err / peak:.3g}')
```

## Timing Arms and Reading Kernel Names

```{.python .input #kernels-lib-arms}
%%tab pytorch
#@save
def _short_kernel(name, width=60):
    """A CUDA kernel's name without return type, namespaces, template and
    arguments (a CUTLASS kernel by its template argument), at most width
    characters."""
    m = re.match(r'(?:void )?cutlass::Kernel\d*<(\w+)', name)
    s = m.group(1) if m else re.split(r'[<(]', re.sub(r'^void ', '', name))[0]
    s = s.split('::')[-1]
    return s if len(s) <= width else s[:width - 3] + '...'

#@save
class _NoKernelsRecorded(RuntimeError):
    """The profiler did not record a whole call (torch 2.11's profiler can
    miss the first kernels of every trace after a cold Triton compile)."""

#@save
def _kernel_launches(fn, leads=(2, 8, 32)):
    """{CUDA kernel name: (launches, device ms)} of one call of fn, after a
    warm-up call. The call is profiled between marker kernels: torch 2.11's
    profiler can miss the first kernels of every trace (seen after a cold
    Triton compile of Inductor's attention kernels), so a trace counts only
    when it kept at least one of the leading markers and the trailing one;
    each retry leads with more markers. _NoKernelsRecorded when no trace
    did."""
    fn()
    torch.cuda.synchronize()
    cuda = torch.profiler.ProfilerActivity.CUDA
    for lead in leads:
        with d2l._quiet(), torch.profiler.profile(activities=[cuda],
                                                 acc_events=True) as prof:
            for _ in range(lead):
                torch.cuda._sleep(1000)  # leading markers (spin_kernel)
            fn()
            torch.cuda._sleep(1000)  # the trailing marker
            torch.cuda.synchronize()
        ev = sorted((e for e in prof.events()
                     if e.device_type == torch.autograd.DeviceType.CUDA),
                    key=lambda e: e.time_range.start)
        mark = ['spin_kernel' in e.name for e in ev]
        i = mark.index(False) if False in mark else len(ev)
        if 0 < i < len(ev) - 1 and mark[-1] and not any(mark[i:-1]):
            launches = {}
            for e in ev[i:-1]:
                c, ms = launches.get(e.name, (0, 0.0))
                launches[e.name] = (c + 1,
                                    ms + e.time_range.elapsed_us() / 1e3)
            return launches
    raise _NoKernelsRecorded('the profiler did not record a whole call; '
                             'restart the kernel and run the cell again')

#@save
def _arm_bench(fn, args, repeats):
    """ms per call of fn(*args) (median of repeats, CUDA events, no graph)
    and its MAD."""
    ms, mad = _hw_stats(_hw_time(lambda: fn(*args), repeats, graph=False))
    return dict(ms=ms, mad=mad)

#@save
def _time_arms(arms, args, flops=None, nbytes=None, repeats=10, warm_s=0.5):
    """One row per arm ({name: fn}) called on args: ms, MAD, TFLOP/s for
    flops and GB/s for nbytes per call, the SM clock and power while it ran
    (_hw_steady)."""
    rows = []
    for name, fn in arms.items():
        with d2l._quiet():
            fn(*args)  # compile (torch.compile, Triton) outside the timing
        r = _hw_steady(lambda: _arm_bench(fn, args, repeats), warm_s)
        row = dict(arm=name, **r)
        if flops:
            row['tflops'] = flops / r['ms'] / 1e9
        if nbytes:
            row['gbs'] = nbytes / r['ms'] / 1e6
        rows.append(row)
        torch.cuda.empty_cache()
    return _hw_frame(rows)

#@save
def _arm_launches(arms, args):
    """{arm: (kernel launches per call, its longest kernel's name)}, read
    with the profiler on this GPU (deterministic for the pinned software,
    so never committed); None, which drops the column, without a GPU,
    while box-microbench is regenerated, or when the profiler records no
    kernel (_NoKernelsRecorded)."""
    if not torch.cuda.is_available() or 'box-microbench' in _hw_regenerating():
        return None
    out = {}
    for name, fn in arms.items():
        try:
            with d2l._quiet():
                launches = _kernel_launches(lambda: fn(*args))
        except _NoKernelsRecorded:
            return None
        top = max(launches, key=lambda n: launches[n][1])
        out[name] = (sum(c for c, _ in launches.values()), _short_kernel(top))
    return out

#@save
def _arms_live(r, same, arms, args, names, col, unit):
    """Live line of a timed-arms table: the arms in names re-timed once,
    their col scaled by committed ms / live ms."""
    pts = []
    for i, name in enumerate(r.arm):
        if name in names and name in arms:
            live = _time_arms({name: arms[name]}, args, warm_s=0.25)
            pts.append((name, r[col].iloc[i],
                        r[col].iloc[i] * r.ms.iloc[i] / live.ms.iloc[0],
                        _hw_noise(r, i, 'ms')))
    return _hw_agreement(pts, same, unit)
```

## Attention Three Ways and the RMSNorm Kernels

```{.python .input #kernels-lib-time}
%%tab pytorch
#@save
def _attn_flops(q):
    """Useful FLOPs of causal attention of q (B, H, N, d): half of the two
    products' 4 B H N^2 d."""
    B, H, N, d = q.shape
    return 2 * B * H * N * N * d

#@save
def _show_attention(r, launches=None):
    """The timed arms; with launches ({arm: (count, kernel)}, read live),
    the kernel launches per call and the kernel SDPA ran."""
    if r.empty:
        return
    t = r.set_index('arm')[['ms', 'tflops']]
    if launches:
        t['kernels'] = [launches[a][0] for a in t.index]
    show_table(t, headers={'tflops': 'TFLOP/s*',
                           'kernels': 'kernels per call'})
    print('* useful causal FLOPs per call: 2 B H N^2 d_h')
    if launches and 'SDPA' in launches:
        print(f"SDPA ran {launches['SDPA'][1]} (the FLASH_ATTENTION backend)")

#@save
def time_attention(arms, q, k, v, repeats=10, live=('compiled', 'SDPA')):
    """Milliseconds, TFLOP/s (useful causal FLOPs) and kernel launches per
    call of each causal attention arm ({name: fn(q, k, v)}), each after a
    warm-up, with the SM clock and power while it ran; the live check
    re-times the arms named in live."""
    config = dict(arms=list(arms), q=list(q.shape), kv=list(k.shape),
                  dtype=str(q.dtype), repeats=repeats)
    compute = lambda: _time_arms(arms, (q, k, v), flops=_attn_flops(q),
                                 repeats=repeats)
    return _hw_result(
        'box-microbench', compute, config, entry='three-ways',
        live_check=lambda r, same: _arms_live(r, same, arms, (q, k, v), live,
                                              'tflops', 'TFLOP/s'),
        show=lambda r: _show_attention(r, _arm_launches(arms, (q, k, v))),
        live_label='attention')

#@save
def _dram_read_gbs():
    """The DRAM read plateau of the committed bandwidth ladder (GB/s), or
    None while it is absent."""
    try:
        ladder = hw_box_log('box-microbench', 'ladder')
    except (KeyError, OSError):
        return None
    return None if ladder.empty else _ladder_plateaus(ladder)[1]

#@save
def _show_fns(r, launches=None):
    """Microseconds and GB/s per arm against the committed DRAM read rate;
    with launches (read live), the kernels per call."""
    if r.empty:
        return
    t = r.set_index('arm')
    t['us'] = t.ms * 1e3
    cols, headers = ['us', 'gbs'], {'us': 'us per call', 'gbs': 'GB/s'}
    dram = _dram_read_gbs()
    if dram:
        t['of_dram'] = [f'{x / dram:.0%}' for x in t.gbs]
        cols.append('of_dram')
        headers['of_dram'] = f'of DRAM read ({_hw_sig(dram)} GB/s)'
    if launches:
        t['kernels'] = [launches[a][0] for a in t.index]
        cols.append('kernels')
    show_table(t[cols], digits={'us': 0, 'gbs': 0}, headers=headers)

#@save
def time_fns(fns, *args, nbytes, repeats=20, entry='rmsnorm'):
    """Microseconds and GB/s (nbytes moved per call) of each function
    ({name: fn(*args)}) on args, each after a warm-up, against the DRAM
    read rate of the committed bandwidth ladder; the committed result is
    the entry named entry (the RMSNorm kernels of the page)."""
    config = dict(arms=list(fns), shapes=[list(a.shape) for a in args],
                  dtypes=[str(a.dtype) for a in args], nbytes=int(nbytes),
                  repeats=repeats)
    compute = lambda: _time_arms(fns, args, nbytes=nbytes, repeats=repeats)
    return _hw_result(
        'box-microbench', compute, config, entry=entry,
        live_check=lambda r, same: _arms_live(r, same, fns, args, tuple(fns),
                                              'gbs', 'GB/s'),
        show=lambda r: _show_fns(r, _arm_launches(fns, args)),
        live_label='RMSNorm')
```

## The Shape Sweep and the Saw-Tooth

```{.python .input #kernels-lib-shapes}
%%tab pytorch
#@save
def _shape_sweep_compute(products, M, repeats):
    rows = []
    for name, n, k in products:
        for m in sorted(M, reverse=True):  # largest first: a warm card
            if _hw_fits(4 * (m * k + n * k) + 2 * m * n):
                rows.append(dict(name=name, M=m, N=n, K=k, **_hw_steady(
                    lambda: d2l.hw_matmul_bench(m, n, k, 'bf16', repeats),
                    warm_s=0.25)))
                torch.cuda.empty_cache()
    order = [p[0] for p in products]
    rows.sort(key=lambda r: (order.index(r['name']), r['M']))
    return _hw_frame(rows)

#@save
def _shape_live(r, same):
    """The first product at three M, and the sweep's slowest point."""
    rows = r[r.name == r.name.iloc[0]]
    picks = [rows.index[j] for j in (0, len(rows) // 2, len(rows) - 1)]
    slowest = r.index[int(np.argmin(r.tflops.to_numpy()))]
    pts = []
    for i in picks + ([slowest] if slowest not in picks else []):
        m, n, k = int(r.M[i]), int(r.N[i]), int(r.K[i])
        live = d2l.hw_matmul_bench(m, n, k, 'bf16', repeats=5)['tflops']
        pts.append((f'{r.name[i]} M={m:,}', r.tflops[i], live,
                    _hw_noise(r, i, 'tflops')))
    return _hw_agreement(pts, same, 'TFLOP/s')

#@save
def hw_shape_sweep(cfg, M, repeats=20):
    """Achieved BF16 TFLOP/s of each weight product of a configuration
    (gemm_shapes) at every token count M (hw_matmul_bench, each point after
    a short warm-up, with the SM clock and power while it ran)."""
    table = gemm_shapes(cfg, 1)[['name', 'N', 'K']]
    products = [(n, int(N), int(K))
                for n, N, K in table.itertuples(index=False)]
    M = [int(m) for m in M]
    config = dict(products=products, M=M, dtype='bf16', repeats=repeats)
    return _hw_result(
        'box-microbench', lambda: _shape_sweep_compute(products, M, repeats),
        config, entry='shape-sweep', live_check=_shape_live,
        live_label='matmul shapes')

#@save
def _show_slowest(sweep, dip=0.85):
    """The sweep's slowest point against its median; when it lies below
    dip times the median and this is the GPU the sweep ran on, the cuBLAS
    kernel there and at the same product's faster neighbour in M (read
    live: cuBLAS picks a kernel per shape)."""
    i = int(np.argmin(sweep.tflops.to_numpy()))
    slow, median = sweep.iloc[i], float(sweep.tflops.median())
    print(f"slowest point: {slow['name']} at M = {int(slow.M):,}, "
          f'{slow.tflops:.0f} TFLOP/s (median of the sweep {median:.0f})')
    if (slow.tflops >= dip * median or not torch.cuda.is_available()
            or 'box-microbench' in _hw_regenerating()
            or _gpu_short(torch.cuda.get_device_name(0))
            != sweep.attrs.get('gpu')):
        return
    part = sweep[sweep.name == slow['name']].sort_values('M')
    j = list(part.index).index(sweep.index[i])
    near = [part.iloc[x] for x in (j - 1, j + 1) if 0 <= x < len(part)]
    rows = [slow, max(near, key=lambda r: r.tflops)] if near else [slow]
    try:
        names = [_gemm_tile(int(r.M), int(r.N), int(r.K))[0] for r in rows]
    except _NoKernelsRecorded:
        return
    print(f"cuBLAS kernels of the {slow['name']} product on this GPU:")
    for r, name in zip(rows, names):
        print(f'  M = {int(r.M):,}: {_short_kernel(name, 62)}')

#@save
def plot_shape_sweep(sweep, specs, reference=None):
    """TFLOP/s against M (log2 x) for each weight product, with the
    datasheet BF16 peak of each measured GPU's chip-table row and the boost
    clock it assumes; the A100 campaign's series dotted when its log is
    committed (else one status line); the slowest point (_show_slowest)."""
    if sweep.empty:
        return
    logs, status = [sweep], ''
    if isinstance(reference, CampaignPending):
        status = hw_status_line(reference)
    elif reference and 'shape-sweep' in reference:
        logs.append(reference['shape-sweep'])
    X, Y, fmts, legend = [], [], [], []
    for j, df in enumerate(logs):
        for i, name in enumerate(dict.fromkeys(sweep.name)):
            part = df[df.name == name]
            X.append(part.M.to_numpy())
            Y.append(part.tflops.to_numpy())
            fmts.append(f'C{i}' + ('-o' if j == 0 else ':'))
            legend.append(name if j == 0 else f'{name} ({df.attrs["gpu"]})')
    d2l.plot(X, Y, 'tokens M', 'TFLOP/s (BF16)', fmts=fmts, xscale='log',
             figsize=(4.5, 3))
    ax = d2l.plt.gca()
    ax.set_xscale('log', base=2)
    handles = ax.get_lines()[:len(Y)]
    for j, df in enumerate(logs):
        chip = _hw_chip_row(df.attrs.get('gpu'), df.attrs.get('mem_bytes'),
                            specs)
        if chip:
            peak = specs.loc[chip, 'p16_tflops']
            handles.append(ax.axhline(peak, color='k', ls='--' if j == 0
                                      else ':', lw=1))
            legend.append(f'datasheet {peak:,.0f} TFLOP/s'
                          + (f' ({chip})' if len(logs) > 1 else ''))
    ax.set_ylim(0, None)
    ax.legend(handles, legend, fontsize=8, loc='center left',
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    _show_slowest(sweep)
    chip = _hw_chip_row(sweep.attrs.get('gpu'), sweep.attrs.get('mem_bytes'),
                        specs)
    clock = sweep.sm_mhz.dropna() if 'sm_mhz' in sweep else []
    if chip in _HW_BOOST_MHZ and len(clock):
        print(f'datasheet line: {_HW_BOOST_MHZ[chip] / 1000:.2f} GHz boost '
              f'clock; SM clock in the sweep {clock.min() / 1000:.2f}-'
              f'{clock.max() / 1000:.2f} GHz')
    if status:
        print(status)

#@save
def _gemm_tile(m, n, k):
    """The cuBLAS kernel a bf16 (m x k) @ (k x n) product runs and the tile
    its name encodes (the first 'AxB' in it), e.g. ('..._128x128_...',
    '128 x 128'); _NoKernelsRecorded when the profiler saw none."""
    fn = d2l.hw_matmul_op(m, n, k, 'bf16')
    launches = _kernel_launches(fn)
    name = _short_kernel(max(launches, key=lambda s: launches[s][1]), 200)
    tile = re.search(r'(\d+)x(\d+)', name)
    return name, (f'{tile[1]} x {tile[2]}' if tile else '')

#@save
def _wave_tiles(M, K, N, tile_every=16):
    """{N: the tile of the cuBLAS kernel} at every tile_every-th N, read
    live (None as for _arm_launches)."""
    if not torch.cuda.is_available() or 'box-microbench' in _hw_regenerating():
        return None
    try:
        return {n: _gemm_tile(M, n, K)[1] for n in list(N)[::tile_every]}
    except _NoKernelsRecorded:
        return None

#@save
@triton.jit
def _tile_matmul_kernel(A, B, C, M, N, K, BLOCK_M: tl.constexpr,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """One BLOCK_M x BLOCK_N tile of C = A B per program (row-major)."""
    i, j = tl.program_id(0), tl.program_id(1)
    a = tl.make_block_ptr(A, (M, K), (K, 1), (i * BLOCK_M, 0),
                          (BLOCK_M, BLOCK_K), (1, 0))
    b = tl.make_block_ptr(B, (K, N), (N, 1), (0, j * BLOCK_N),
                          (BLOCK_K, BLOCK_N), (1, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(0, K, BLOCK_K):
        acc += tl.dot(tl.load(a, boundary_check=(0, 1)),
                      tl.load(b, boundary_check=(0, 1)))
        a, b = tl.advance(a, (0, BLOCK_K)), tl.advance(b, (BLOCK_K, 0))
    c = tl.make_block_ptr(C, (M, N), (N, 1), (i * BLOCK_M, j * BLOCK_N),
                          (BLOCK_M, BLOCK_N), (1, 0))
    tl.store(c, acc.to(C.dtype.element_ty), boundary_check=(0, 1))

#@save
def tile_matmul(a, b, tile=(128, 128), block_k=64, num_warps=8,
                num_stages=3):
    """a @ b for contiguous 2-D a and b with one fixed tile of the output
    per program, whatever the shape (cuBLAS picks a kernel per shape)."""
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, tile[0]), triton.cdiv(N, tile[1]))
    _tile_matmul_kernel[grid](a, b, c, M, N, K, *tile, block_k,
                              num_warps=num_warps, num_stages=num_stages)
    return c

#@save
_WAVE_TILE = (128, 128)  # the fixed tile of the Triton arm

#@save
def _wave_bench(arm, m, n, k, repeats):
    """TFLOP/s, ms and MAD of a BF16 m x n x k product: cuBLAS
    (hw_matmul_bench) or tile_matmul with _WAVE_TILE."""
    if arm == 'cuBLAS':
        return d2l.hw_matmul_bench(m, n, k, 'bf16', repeats)
    g = torch.Generator(device='cuda').manual_seed(0)
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16, generator=g)
    b = torch.randn(k, n, device='cuda', dtype=torch.bfloat16, generator=g)
    ms, mad = d2l._hw_stats(d2l._hw_time(
        lambda: tile_matmul(a, b, _WAVE_TILE), repeats))
    return dict(tflops=2 * m * n * k / ms / 1e9, ms=ms, mad=mad)

#@save
def _wave_arm_name():
    return 'Triton {} x {}'.format(*_WAVE_TILE)

#@save
def _wave_compute(M, K, N, repeats):
    _hw_steady(lambda: d2l.hw_matmul_bench(M, max(N), K, 'bf16', 5),
               warm_s=2.0)  # the card at its loaded clock before the sweep
    rows = [dict(arm=arm, N=n, **_hw_steady(
        lambda: _wave_bench(arm, M, n, K, repeats), warm_s=0.1))
        for arm in ('cuBLAS', _wave_arm_name()) for n in N]
    df = _hw_frame(rows)
    df.attrs.update(M=M, K=K)
    return df

#@save
def _sawtooth_period(N, y, min_ratio=4.0):
    """The period (in N) of the strongest periodic component of y over the
    evenly spaced N, after removing a linear trend: the periodogram's peak
    over periods from two steps to half the range, refined to a quarter
    step within one step of it (a period that is not a multiple of the
    step is not aliased to one); None when the peak is below min_ratio
    times the median power (no clear saw-tooth)."""
    N, y = np.asarray(N, float), np.asarray(y, float)
    r = y - np.polyval(np.polyfit(N, y, 1), N)
    step = N[1] - N[0]
    power = lambda ps: np.array([
        np.abs(np.sum(r * np.exp(-2j * np.pi * N / p)))**2 for p in ps])
    periods = np.arange(2 * step, (N[-1] - N[0]) / 2 + step, step)
    coarse = power(periods)
    best = int(np.argmax(coarse))
    if coarse[best] < min_ratio * np.median(coarse):
        return None
    fine = periods[best] + np.arange(-4, 5) * step / 4
    fine = fine[fine >= 2 * step]
    return float(fine[int(np.argmax(power(fine)))])

#@save
def _show_wave(wave, tiles=None):
    """The fixed-tile arm's measured saw-tooth period against one wave of
    tiles, n_SM T_m T_n / M columns, and the TFLOP/s range of each arm;
    with tiles ({N: tile}, read live) the tiles cuBLAS's kernels encode."""
    if wave.empty:
        return
    M, n_sm, (tm, tn) = wave.attrs['M'], wave.attrs['n_sm'], _WAVE_TILE
    for arm, r in wave.groupby('arm', sort=False):
        line = f'{arm}: {r.tflops.min():.0f}-{r.tflops.max():.0f} TFLOP/s'
        if arm != 'cuBLAS':
            p = _sawtooth_period(r.N, r.tflops)
            line += (', no clear saw-tooth' if p is None else
                     f', saw-tooth period {p:,.0f} columns (measured)')
        print(line)
    print(f'one wave: n_SM T_m T_n / M = {n_sm} x {tm} x {tn} / {M:,} = '
          f'{n_sm * tm * tn / M:,.0f} columns')
    if tiles:
        print('cuBLAS tiles by kernel name: ' + ', '.join(
            dict.fromkeys(t for t in tiles.values() if t)))

#@save
def _wave_live(r, same):
    """The fixed-tile arm's points from one before its largest drop to
    two after it (fewer at the ends of the sweep)."""
    r = r[r.arm != 'cuBLAS'].reset_index(drop=True)
    i = int(np.argmax(-np.diff(r.tflops.to_numpy())))  # the largest drop
    picks = [j for j in range(i - 1, i + 3) if 0 <= j < len(r)]
    M, K = r.attrs['M'], r.attrs['K']
    pts = [(f'N={int(r.N[j]):,}', r.tflops[j], _wave_bench(
        r.arm[j], M, int(r.N[j]), K, repeats=10)['tflops'],
        _hw_noise(r, j, 'tflops')) for j in picks]
    return _hw_agreement(pts, same, 'TFLOP/s')

#@save
def hw_wave_sweep(M, K, N, repeats=10):
    """BF16 TFLOP/s of an M x N x K product for every N (after one warm-up
    of the card), with cuBLAS and with tile_matmul's one fixed tile."""
    N = [int(n) for n in N]
    config = dict(M=M, K=K, N=N, dtype='bf16', repeats=repeats,
                  tile=list(_WAVE_TILE))
    return _hw_result(
        'box-microbench', lambda: _wave_compute(M, K, N, repeats), config,
        entry='wave', live_check=_wave_live,
        show=lambda w: _show_wave(w, _wave_tiles(M, K, N)),
        live_label='saw-tooth')

#@save
def plot_wave(wave):
    """TFLOP/s against N of the wave sweep, one line per arm."""
    if wave.empty:
        return
    arms = list(dict.fromkeys(wave.arm))
    parts = [wave[wave.arm == a] for a in arms]
    d2l.plot(parts[0].N.to_numpy(), [p.tflops.to_numpy() for p in parts],
             'N (columns)', 'TFLOP/s (BF16)', legend=arms, figsize=(4.5, 2.5))
```

## The Flash Sweep and the Compiled Kernels

```{.python .input #kernels-lib-flash}
%%tab pytorch
#@save
def _flash_arms(block_q, block_k, num_warps, num_stages):
    """{label: flash_attention_triton with that configuration}."""
    import functools
    return {f'{bq} x {bk}, {nw} warps': functools.partial(
                d2l.flash_attention_triton, block_q=bq, block_k=bk,
                num_warps=nw, num_stages=num_stages)
            for bq in block_q for bk in block_k for nw in num_warps}

#@save
def _resource_error(e):
    """'shared memory 128 KiB > 99 KiB' from Triton's OutOfResources."""
    return (f'{e.name} {e.required / 2**10:g} KiB > {e.limit / 2**10:g} KiB'
            if hasattr(e, 'required') else type(e).__name__)

#@save
def _flash_compute(shapes, seq, block_q, block_k, num_warps, num_stages,
                   repeats):
    rows = []
    for shape, (cfg, batch) in shapes.items():
        q, k, v = attention_inputs(cfg, batch, seq)
        H, H_kv, d_h = _attn_dims(cfg)
        base = dict(shape=shape, H=H, H_kv=H_kv, d_h=d_h)
        arms = dict(_flash_arms(block_q, block_k, num_warps, num_stages),
                    SDPA=sdpa_causal, compiled=torch.compile(
                        d2l.chunked_causal_attention, dynamic=False))
        for label, fn in arms.items():
            kw = getattr(fn, 'keywords', {})  # 0: not a Triton configuration
            tiles = {key: kw.get(key, 0)
                     for key in ('block_q', 'block_k', 'num_warps')}
            try:
                r = _time_arms({label: fn}, (q, k, v), flops=_attn_flops(q),
                               repeats=repeats, warm_s=0.25)
                row = dict(r.iloc[0].to_dict(), error='')
            except Exception as e:  # a tile that does not fit on chip
                if type(e).__name__ != 'OutOfResources':
                    raise
                row = dict(arm=label, ms=math.nan, mad=math.nan,
                           sm_mhz=math.nan, power_w=math.nan,
                           tflops=math.nan, error=_resource_error(e))
            rows.append(dict(base, **tiles, **row))
        del q, k, v
        torch.cuda.empty_cache()
    df = _hw_frame(rows)
    df.attrs.update(seq=seq, num_stages=num_stages)
    return df

#@save
def _flash_summary(r, best=3, worst=1):
    """Per shape: the best and worst Triton configurations, SDPA and the
    compiled chunked attention, with TFLOP/s and the fraction of SDPA."""
    out = []
    for shape, part in r.groupby('shape', sort=False):
        ok = part[(part.error == '') & (part.block_q > 0)].sort_values(
            'tflops', ascending=False)
        sdpa = float(part[part.arm == 'SDPA'].tflops.iloc[0])
        keep = pd.concat([ok.head(best), ok.tail(worst),
                          part[part.block_q == 0]]).drop_duplicates('arm')
        for _, row in keep.iterrows():
            out.append(dict(shape=shape, arm=row.arm, tflops=row.tflops,
                            of_sdpa=f'{row.tflops / sdpa:.0%}'))
    return pd.DataFrame(out)

#@save
def _show_flash(r):
    if r.empty:
        return
    show_table(_flash_summary(r).set_index(['shape', 'arm']), headers={
        'tflops': 'TFLOP/s', 'of_sdpa': 'of SDPA'})
    import textwrap
    failed = r[r.error != '']
    for err, part in failed.groupby('error', sort=False):
        print(textwrap.fill('did not compile: ' + ', '.join(
            f'{s} {a}' for s, a in zip(part['shape'], part.arm))
            + f' ({err})', 79, subsequent_indent='  '))

#@save
def _flash_live(r, same, shapes, seq):
    pts = []
    for shape, (cfg, batch) in shapes.items():
        part = r[(r['shape'] == shape) & (r.error == '')]
        tri = part[part.block_q > 0]
        best = tri.index[int(np.argmax(tri.tflops.to_numpy()))]
        sdpa = part.index[list(part.arm).index('SDPA')]
        q, k, v = attention_inputs(cfg, batch, seq)
        arms = dict(_flash_arms(
            (int(r.block_q[best]),), (int(r.block_k[best]),),
            (int(r.num_warps[best]),), r.attrs.get('num_stages', 2)),
            SDPA=sdpa_causal)
        for i, (label, fn) in zip((best, sdpa), arms.items()):
            live = _time_arms({label: fn}, (q, k, v), warm_s=0.25)
            pts.append((f'{shape} {label}', r.tflops[i],
                        r.tflops[i] * r.ms[i] / live.ms.iloc[0],
                        _hw_noise(r, i, 'ms')))
    return _hw_agreement(pts, same, 'TFLOP/s')

#@save
def flash_sweep(shapes, seq, block_q, block_k, num_warps, num_stages=2,
                repeats=10):
    """TFLOP/s (useful causal FLOPs) of flash_attention_triton for every
    (block_q, block_k, num_warps) at each attention shape, against SDPA's
    flash kernel and the compiled chunked attention; a configuration that
    does not fit the GPU's on-chip memory is recorded with the reason."""
    dims = {name: [*_attn_dims(cfg), batch]
            for name, (cfg, batch) in shapes.items()}
    config = dict(shapes=dims, seq=seq, block_q=list(block_q),
                  block_k=list(block_k), num_warps=list(num_warps),
                  num_stages=num_stages, repeats=repeats)
    compute = lambda: _flash_compute(shapes, seq, block_q, block_k,
                                     num_warps, num_stages, repeats)
    return _hw_result(
        'box-microbench', compute, config, entry='flash-sweep',
        live_check=lambda r, same: _flash_live(r, same, shapes, seq),
        show=_show_flash, live_label='attention kernels')

#@save
def flash_sweep_table(log):
    """The flash sweep of a campaign log, or its one status line while the
    log is not committed."""
    if isinstance(log, CampaignPending):
        print(hw_status_line(log))
    elif log and 'flash-sweep' in log:
        _show_flash(log['flash-sweep'])

#@save
def _compiled_metadata(kernel):
    """(constexpr values by name, num_warps, num_stages, registers per
    thread, spilled 32-bit words per thread) of every compiled
    specialization of a Triton kernel on the current GPU."""
    dev = torch.cuda.current_device()
    out = []
    for ck in kernel.device_caches[dev][0].values():
        ck._init_handles()
        consts = {kernel.arg_names[key[0]]: val
                  for key, val in ck.src.constants.items()}
        out.append((consts, ck.metadata.num_warps, ck.metadata.num_stages,
                    ck.n_regs, ck.n_spills))
    return out

#@save
def triton_spills(launcher, flash):
    """Registers per thread and spilled registers (32-bit words of local
    memory per thread) of every Triton configuration of a flash sweep,
    read from the compiled kernels' metadata (deterministic for the pinned
    Triton), next to each configuration's committed TFLOP/s when this GPU
    is the one the sweep ran on."""
    stages = flash.attrs.get('num_stages', 2)
    kernels = [launcher.__globals__[n] for n in launcher.__code__.co_names
               if isinstance(launcher.__globals__.get(n),
                             triton.runtime.JITFunction)]
    tri = flash[flash.block_q > 0]
    rows = {}
    for (shape, arm), row in tri.set_index(['shape', 'arm']).iterrows():
        H, H_kv, d = int(row.H), int(row.H_kv), int(row.d_h)
        n = 2 * max(int(row.block_q), int(row.block_k))
        q = torch.zeros(1, H, n, d, dtype=torch.bfloat16, device='cuda')
        kv = torch.zeros(1, H_kv, n, d, dtype=torch.bfloat16, device='cuda')
        try:
            with d2l._quiet():
                launcher(q, kv, kv, causal=True, block_q=int(row.block_q),
                         block_k=int(row.block_k),
                         num_warps=int(row.num_warps), num_stages=stages)
        except Exception as e:
            if type(e).__name__ != 'OutOfResources':
                raise
        regs = spills = math.nan
        for consts, nw, ns, n_regs, n_spills in (
                m for k in kernels for m in _compiled_metadata(k)):
            if (consts.get('BLOCK_Q') == row.block_q
                    and consts.get('BLOCK_K') == row.block_k
                    and consts.get('D') == d
                    and consts.get('GROUP') == H // H_kv
                    and consts.get('CAUSAL') is True
                    and nw == row.num_warps and ns == stages):
                regs, spills = n_regs, n_spills
        tag = shape.split('-')[-1] if '-' in shape else shape[:3]
        label = f'{int(row.block_q)}x{int(row.block_k)} w{int(row.num_warps)}'
        rows.setdefault(label, {}).update({
            f'TF/s {tag}': row.tflops, f'regs {tag}': regs,
            f'spills {tag}': spills})
    t = pd.DataFrame.from_dict(rows, orient='index')
    committed = flash.attrs.get('gpu')
    here = _gpu_short(torch.cuda.get_device_name(0))
    if committed != here:  # never committed timings next to this GPU's
        t = t[[c for c in t.columns if not c.startswith('TF')]]
    show_table(t, digits={c: 0 for c in t.columns if not c.startswith('TF')})
    print('regs: registers per thread; spills: 32-bit words per thread in '
          'local memory')
    if committed != here:
        print(f'(TF/s omitted: the sweep ran on {committed}, not on {here})')

#@save
_SOFTMAX_OPS = ('masked_fill', 'amax', 'maximum', 'sub', 'exp', 'sum', 'div',
                'mul')  # the operations of the online softmax, in its order

#@save
def _kernel_kind(name):
    """A CUDA kernel's kind in a few words: a library GEMM by the tile its
    name encodes; an Inductor kernel (named after the operations it fuses)
    by kind and the softmax operations among them; else its short name."""
    m = re.match(r'triton_(\w{3})_fused_(.*?)(_\d+)?$', name)
    if m:
        kind = {'poi': 'pointwise', 'per': 'reduction',
                'red': 'reduction'}.get(m[1], m[1])
        ops = [op for op in _SOFTMAX_OPS if f'_{op}_' in f'_{m[2]}_']
        return f'Inductor {kind}: ' + ', '.join(ops)
    tile = re.search(r'_(\d+)x(\d+)_', name)
    if 'gemm' in name and tile:
        return f'library GEMM, tile {tile[1]} x {tile[2]}'
    return 'library ' + _short_kernel(name, 50)

#@save
def inductor_kernels(fn, *args):
    """The CUDA kernels one call of a compiled function launches, most
    launched first: library GEMMs by tile, and the Triton kernels Inductor
    generated by kind (pointwise, reduction) and the softmax operations
    each one fuses (read from its name); kernels of the same kind on one
    line."""
    with d2l._quiet():
        fn(*args)  # compile
    launches = _kernel_launches(lambda: fn(*args))
    total = sum(c for c, _ in launches.values())
    print(f'{total} launches per call, {len(launches)} distinct kernels:')
    kinds = collections.Counter()
    for name, (c, _) in launches.items():
        kinds[_kernel_kind(name)] += c
    for text, c in kinds.most_common():
        print(f'{c:3d} x {text}')
```

## The Section's Part of the Regeneration

`regen_microbench` (22.1's driver of `box-microbench`) calls this after its
own entries: the page's calls on the full configuration.

```{.python .input #kernels-lib-regen}
%%tab pytorch
#@save
def _regen_kernels():
    """The kernels section's measurements (the page's calls): entries
    three-ways, shape-sweep, wave, rmsnorm and flash-sweep; returns
    (command, table) pairs for the benchmark log."""
    R = d2l.r2_config('r2').arch
    cfg_8b = d2l.hw_config('Qwen/Qwen3-8B')
    shapes = {'reference': (R, 8), 'Qwen3-8B': (cfg_8b, 1)}
    seq = 4096
    q, k, v = attention_inputs(R, batch=8, seq=seq)
    ways = time_attention({
        'chunked': d2l.chunked_causal_attention,
        'compiled': torch.compile(d2l.chunked_causal_attention),
        'SDPA': sdpa_causal}, q, k, v)
    del q, k, v
    torch.cuda.empty_cache()
    wave = hw_wave_sweep(M=4096, K=4096, N=range(1024, 8193, 64))
    d = 4096
    x = torch.randn(8 * 4096, d, device='cuda')
    w = torch.randn(d, device='cuda')
    norms = time_fns({
        'Triton': d2l.triton_rmsnorm, 'formula': d2l.rms_norm_formula,
        'formula, compiled': torch.compile(d2l.rms_norm_formula),
        'F.rms_norm': lambda x, w: F.rms_norm(x, (d,), w, eps=1e-6)},
        x, w, nbytes=2 * x.nbytes)
    del x, w
    torch.cuda.empty_cache()
    flash = flash_sweep(shapes, seq, block_q=(64, 128), block_k=(32, 64, 128),
                        num_warps=(4, 8))
    sweep = hw_shape_sweep(cfg_8b, M=2**np.arange(9, 15))
    return [('time_attention', ways), ('hw_shape_sweep', sweep),
            ('hw_wave_sweep', wave), ('time_fns', norms),
            ('flash_sweep', flash)]
```
