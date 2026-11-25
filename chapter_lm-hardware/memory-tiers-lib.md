# Memory and Storage Tiers: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_memory_tiers` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1): the
section's registry entries, one statement each (the tier table, the box's
storage sweeps and the campaign log C-T3), its regeneration job and the
campaign configuration; `HW_OFFLOAD_CONFIG` and the other sweep
configurations; the machine facts (host memory, GPU count, PCIe link state,
host bridges); the sweeps behind the page's calls (host links, the four
cards at once, host DRAM, fio, the object store, the CPU optimizer), the
offloaded run, the shard read; scratch-file handling and the disk check;
fits and displays; the driver `regen_storage`. Blocks are tagged
`%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_memory_tiers`

## Registries

```{.python .input #memory-tiers-lib-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-hw-tiers-v1'] = (
    d2l.LM_DATA_URL + 'lm-hw-tiers-v1.csv',
    '039aaa0d909e4d22fe1a6be28dd7cb00f2b2f050')
```

```{.python .input #memory-tiers-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['tiers'] = ('lm-hw-tiers-v1', 'csv')
d2l.HW_ARTIFACTS['box-storage'] = ('lm-hw-buildbox-storage-v1', 'tar')
d2l.HW_CAMPAIGNS['node-storage'] = (
    'lm-hw-node-storage-v1', 'tar', 'C-T3', 'one 8 x A100 40 GB node')
```

```{.python .input #memory-tiers-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-storage'] = (
    'regen_storage', lambda: dict(config='box'))
d2l.HW_CONFIG_LOGS[('box-storage', 't3')] = 'node-storage'
d2l._HW_TIERS['t3'] = 'T3'
```

## Configurations

The offloaded run's model is a Qwen3-style configuration of 0.82 billion
parameters whose tied embedding does not dominate; its token counts bracket
`max(T_link, T_cpu)` by about 8x on each side, from the infra step's and the
smoke run's measurements (`P_eff` 177 TFLOP/s, pinned link 25 GB/s per
direction, CPU update 0.40 ns per parameter: `T_link` 2,400 and `T_cpu`
11,800 tokens), in multiples of one micro-batch (two sequences of 2,048).
The fio sweep is the blueprint's (block sizes 4 KiB to 4 MiB, sequential and
random reads, queue depth 1 and 64) with 10 s runs after a 2 s ramp; the
object-store probe reads ranges of one registered file of the book's bucket.

```{.python .input #memory-tiers-lib-configs}
%%tab pytorch
#@save
HW_OFFLOAD_CONFIG = {
    'box': dict(d_model=2048, n_layers=16, n_heads=16, n_kv_heads=8,
                head_dim=128, d_ff=5632, vocab=32768, seq_len=2048,
                micro_seqs=2, chunk=2048, cpu_threads=32, warmup=2, steps=5,
                tokens=[4096, 8192, 12288, 16384, 24576, 32768, 65536,
                        98304])}

#@save
_HW_FIO = {'box': dict(file_gib=16, bs=[2**12, 2**14, 2**16, 2**18, 2**20,
                                        2**22],
                       rw=['read', 'randread'], qd=[[1, 1], [64, 4]],
                       runtime=10, ramp=2)}

#@save
_HW_OBJECT = {'box': dict(sizes=[2**16, 2**18, 2**20, 2**22, 2**24, 2**26],
                          repeats=3, streams=8, stream_bytes=2**24)}

#@save
_HW_PROBE_KEY = 'lm-fwedu-shards-100m-v1'  # the object-store probe's file

#@save
_HW_SCRATCH = os.path.join('..', 'data', 'hw-scratch')  # fio's files

#@save
_PCIE_GTS = {3: 8, 4: 16, 5: 32}  # GT/s per lane; 128b/130b encoding
```

## The Machine

```{.python .input #memory-tiers-lib-machine}
%%tab pytorch
#@save
def host_memory_bytes():
    """Physical memory of this host in bytes."""
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

#@save
def host_gpu_count():
    """GPUs on this host, whichever this process may use (nvidia-smi)."""
    return len(d2l._hw_smi('--query-gpu=index'))

#@save
def pcie_ceiling(gen=4, lanes=16):
    """Bytes per second one direction of a PCIe link carries: lanes x
    GT/s x 128/130 (Gen 3 to 5 encoding) / 8 bits."""
    return lanes * _PCIE_GTS[gen] * 1e9 * 128 / 130 / 8

#@save
@contextlib.contextmanager
def _numa_bind(node):
    """Run the block on the CPUs of NUMA node `node` (None: unchanged), so
    buffers it allocates and first touches live on that node."""
    if node is None:
        yield
        return
    with open(f'/sys/devices/system/node/node{node}/cpulist') as f:
        cpus = set()
        for part in f.read().strip().split(','):
            a, _, b = part.partition('-')
            cpus.update(range(int(a), int(b or a) + 1))
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, old)

#@save
def _pcie_link():
    """(generation, width) of this process's current GPU's link now."""
    uuid = d2l._hw_visible_uuids()[torch.cuda.current_device()]
    row = d2l._hw_smi('--query-gpu=pcie.link.gen.current,'
                      'pcie.link.width.current', '-i', uuid)
    if not row:
        return None, None
    return tuple(int(x) if x.isdigit() else None for x in row[0][:2])

#@save
def _pcie_under_load(seconds=1.0):
    """The link's generation and width at rest and while pinned copies of
    256 MiB keep it busy (an idle link drops to Gen 1 to save power)."""
    import threading
    host = torch.ones(2**28, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty_like(host, device='cuda')
    idle, stop = _pcie_link(), threading.Event()

    def copy():
        with torch.cuda.stream(torch.cuda.Stream()):
            while not stop.is_set():
                gpu.copy_(host, non_blocking=True)
                torch.cuda.current_stream().synchronize()
    thread = threading.Thread(target=copy)
    thread.start()
    time.sleep(seconds / 2)
    busy = _pcie_link()
    stop.set()
    thread.join()
    return dict(pcie_gen_idle=idle[0], pcie_width_idle=idle[1],
                pcie_gen=busy[0], pcie_width=busy[1])

#@save
def _topo_relations(text):
    """{(i, j): relation} between GPUs from `nvidia-smi topo -m` (PHB: the
    two share a host bridge; NODE: different bridges of one NUMA node; SYS:
    different sockets; NV#: NVLink)."""
    rows = {}
    for line in re.sub(r'\x1b\[[0-9;]*m', '', text).splitlines():
        cells = line.split()
        if cells and re.fullmatch(r'GPU\d+', cells[0]):
            i = int(cells[0][3:])
            for j, rel in enumerate(cells[1:]):
                if not re.fullmatch(r'X|PHB|PIX|PXB|NODE|SYS|NV\d+', rel):
                    break
                rows[(i, j)] = rel
    return rows

#@save
def _bridge_groups(n, relations):
    """The groups of the four-card question: one card, a pair behind one
    host bridge, a pair across bridges (when the topology has both), all
    n cards."""
    groups = [('one card', (0,))]
    same = [j for j in range(1, n)
            if relations.get((0, j)) in ('PHB', 'PIX', 'PXB')]
    other = [j for j in range(1, n)
             if relations.get((0, j)) in ('NODE', 'SYS')]
    if same:
        groups.append(('two cards, one host bridge', (0, same[0])))
    if other:
        groups.append(('two cards, two host bridges', (0, other[0])))
    if n > 2:
        groups.append((f'all {n} cards', tuple(range(n))))
    return groups

#@save
def _row(df, **match):
    """Index of the first row of df whose columns equal match."""
    hit = np.ones(len(df), dtype=bool)
    for k, v in match.items():
        hit &= (df[k] == v).to_numpy()
    return int(np.flatnonzero(hit)[0])

#@save
def _spread(df, col, i, default=0.0):
    """Relative spread between a regeneration's processes of row i's col
    (default when the processes agreed or there was one)."""
    s = df.attrs.get('spread', {}).get(col)
    return s[i] if s else default
```

## The Host Link

```{.python .input #memory-tiers-lib-host-link}
%%tab pytorch
#@save
def _h2d_live(r, same):
    pts = []
    for d in ('h2d', 'd2h'):
        i = _row(r, mode='pinned', direction=d, bytes=2**28)
        pts.append((f'pinned {d} 256 MiB', r.gbs.iloc[i],
                    d2l.hw_h2d_bench(2**28, True, d, repeats=5)['gbs'],
                    d2l._hw_noise(r, i, 'gbs')))
    return d2l._hw_agreement(pts, same, 'GB/s')

#@save
def hw_h2d_sweep(sizes, pinned=(True, False), directions=('h2d', 'd2h')):
    """hw_h2d_bench at every size (bytes), pinned and pageable, host to
    device and back, on this process's first GPU; attrs hold the PCIe
    generation and width at rest and under load (read with nvidia-smi
    during the sweep)."""
    sizes, pinned = [int(s) for s in sizes], [bool(p) for p in pinned]
    directions = list(directions)

    def compute():
        rows = [dict(mode='pinned' if p else 'pageable', direction=d,
                     bytes=s, **d2l.hw_h2d_bench(s, p, d))
                for p in pinned for d in directions for s in sizes]
        df = d2l._hw_frame(rows)
        df.attrs.update(_pcie_under_load())
        return df
    return d2l._hw_result('box-storage', compute,
                          dict(sizes=sizes, pinned=pinned,
                               directions=directions), entry='h2d',
                          live_check=_h2d_live, live_label='host link')

#@save
def fit_h2d(h2d, n_boot=200, seed=0):
    """alpha (s), beta (bytes/s) and a bootstrap 95% interval of beta per
    (mode, direction) of a host-link sweep (fit_alpha_beta on resampled
    points, seed fixed)."""
    rng, rows = np.random.default_rng(seed), {}
    for (mode, d), g in h2d.groupby(['mode', 'direction'], sort=False):
        s, t = g.bytes.to_numpy(float), g.ms.to_numpy() / 1e3
        alpha, beta, _ = d2l.fit_alpha_beta(s, t)
        boots = []
        while len(boots) < n_boot:
            k = rng.integers(0, len(s), len(s))
            if len(np.unique(s[k])) >= 3:
                boots.append(d2l.fit_alpha_beta(s[k], t[k])[1])
        rows[(mode, d)] = dict(alpha=alpha, beta=beta,
                               beta_lo=np.percentile(boots, 2.5),
                               beta_hi=np.percentile(boots, 97.5))
    return pd.DataFrame.from_dict(rows, orient='index')

#@save
def plot_h2d(h2d, links, pcie_gen=4, lanes=16):
    """GB/s against bytes per copy for each mode and direction, with the
    PCIe per-direction ceiling dashed; prints the fitted alpha and beta of
    pinned h2d and d2h and pageable h2d and the link state during the
    sweep."""
    ceiling = pcie_ceiling(pcie_gen, lanes) / 1e9
    X, Y, legend = [], [], []
    for (mode, d), g in h2d.groupby(['mode', 'direction'], sort=False):
        X.append(g.bytes.to_numpy())
        Y.append(g.gbs.to_numpy())
        legend.append(f'{mode} {d}')
    x = np.sort(h2d.bytes.unique())
    X.append(x)
    Y.append(np.full(len(x), ceiling))
    legend.append(f'PCIe {pcie_gen}.0 x{lanes}, one direction')
    d2l.plot(X, Y, 'bytes per copy', 'bandwidth (GB/s)', legend=legend,
             xscale='log', fmts=('C0-', 'C1-', 'C0:', 'C1:', 'k--'),
             figsize=(4.5, 3))
    d2l.plt.gca().set_xscale('log', base=2)
    for mode, d in (('pinned', 'h2d'), ('pinned', 'd2h'),
                    ('pageable', 'h2d')):
        r = links.loc[(mode, d)]
        print(f'{mode} {d}: alpha {r.alpha * 1e6:.0f} us, beta '
              f'{r.beta / 1e9:.1f} GB/s (95% CI {r.beta_lo / 1e9:.1f}-'
              f'{r.beta_hi / 1e9:.1f}), {r.beta / 1e9 / ceiling:.0%} of '
              f'{ceiling:.1f}')
    a = h2d.attrs
    print(f"link during the sweep: Gen {a.get('pcie_gen')} "
          f"x{a.get('pcie_width')} (at rest: Gen {a.get('pcie_gen_idle')})")
```

## The Four Cards at Once

All cards copy 256 MiB from their own pinned buffers at the same moment
(threads released together); the aggregate is the bytes moved over the
wall time from the first start to the last finish, the median of five
rounds. `nvbandwidth` measures each card alone with its copy engines, a
cross-check of the single-card figure.

```{.python .input #memory-tiers-lib-concurrent}
%%tab pytorch
#@save
def _concurrent_once(bufs, devices, direction, copies):
    """Aggregate GB/s of `copies` copies per device, all devices at once."""
    import threading
    barrier, spans = threading.Barrier(len(devices)), {}

    def work(i):
        host, gpu = bufs[i]
        src, dst = (host, gpu) if direction == 'h2d' else (gpu, host)
        with torch.cuda.device(i), torch.cuda.stream(torch.cuda.Stream()):
            dst.copy_(src, non_blocking=True)  # warm
            torch.cuda.current_stream().synchronize()
            barrier.wait()
            t = time.perf_counter()
            for _ in range(copies):
                dst.copy_(src, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            spans[i] = (t, time.perf_counter())
    threads = [threading.Thread(target=work, args=(i,)) for i in devices]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = (max(b for _, b in spans.values())
            - min(a for a, _ in spans.values()))
    return len(devices) * copies * bufs[devices[0]][0].numel() / wall / 1e9

#@save
def _nvbandwidth_cards():
    """Per-card GB/s of nvbandwidth's copy-engine h2d and d2h tests (each
    card alone), or None without the tool."""
    exe = d2l.hw_bin('nvbandwidth')
    if exe is None:
        return None
    run = d2l.hw_run([exe, '-b', '512', '-i', '5', '-t',
                      'host_to_device_memcpy_ce', 'device_to_host_memcpy_ce'],
                     timeout_s=600)
    parsed = d2l.parse_nvbandwidth(run.stdout)
    return {d: [float(x) for x in np.ravel(parsed[f'{d}_memcpy_ce'])]
            for d in ('host_to_device', 'device_to_host')
            if f'{d}_memcpy_ce' in parsed}

#@save
def _concurrent_compute(nbytes=2**28, copies=8, repeats=5):
    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError('the four-card measurement needs every card')
    topo = d2l.hw_run(['nvidia-smi', 'topo', '-m'], timeout_s=60).stdout
    groups = _bridge_groups(n, _topo_relations(topo))
    bufs = {i: (torch.ones(nbytes, dtype=torch.uint8, pin_memory=True),
                torch.empty(nbytes, dtype=torch.uint8, device=i))
            for i in range(n)}
    rows = []
    for label, devices in groups:
        for d in ('h2d', 'd2h'):
            gbs = [_concurrent_once(bufs, devices, d, copies)
                   for _ in range(repeats)]
            med, mad = d2l._hw_stats(gbs)
            rows.append(dict(group=label, n_cards=len(devices), direction=d,
                             gbs=med, mad_gbs=mad))
    df = d2l._hw_frame(rows)
    df.attrs.update(nbytes=nbytes, nvbandwidth=_nvbandwidth_cards())
    return df

#@save
def _show_concurrent(agg):
    one = {d: agg.gbs[(agg.n_cards == 1) & (agg.direction == d)].iloc[0]
           for d in ('h2d', 'd2h')}
    for label, g in agg.groupby('group', sort=False):
        h, b = (g[g.direction == d].iloc[0] for d in ('h2d', 'd2h'))
        k = int(h.n_cards)
        ratio = '' if k == 1 else (f' ({h.gbs / (k * one["h2d"]):.0%} of '
                                   f'{k} x one card)')
        print(f'{label}: h2d {h.gbs:.1f}, d2h {b.gbs:.1f} GB/s{ratio}')
    nvb = agg.attrs.get('nvbandwidth')
    if nvb and nvb.get('host_to_device'):
        h = np.median(nvb['host_to_device'])
        print(f'nvbandwidth, each card alone (copy engines): h2d {h:.1f} '
              f'GB/s')

#@save
def _concurrent_live(r, same):
    i = _row(r, n_cards=1, direction='h2d')
    live = d2l.hw_h2d_bench(2**28, True, 'h2d', repeats=5)['gbs']
    return d2l._hw_agreement([('one card h2d', r.gbs.iloc[i], live,
                               r.mad_gbs.iloc[i] / r.gbs.iloc[i])], same,
                             'GB/s')

#@save
def hw_h2d_concurrent(config='box'):
    """Pinned host-to-device and device-to-host copies on one card, on two
    cards behind one host bridge and behind two, and on every card at
    once: aggregate GB/s (it needs every card of the host; a normal run
    shows the committed result and re-measures one card)."""
    return d2l._hw_result('box-storage', _concurrent_compute,
                          dict(config=config), entry='concurrent',
                          live_check=_concurrent_live,
                          show=_show_concurrent, live_label='one card')
```

## Host Memory

```{.python .input #memory-tiers-lib-dram}
%%tab pytorch
#@save
@contextlib.contextmanager
def _cpu_threads(n):
    """Run the block with n intra-op CPU threads."""
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)

#@save
def _host_read(threads, nbytes, repeats=5):
    """GB/s of a summing read of nbytes of host memory on threads cores
    (the buffer is written first, so every page is mapped)."""
    x = torch.ones(nbytes // 4, dtype=torch.float32)
    with _cpu_threads(threads):
        x.sum()
        t = []
        for _ in range(repeats):
            start = time.perf_counter()
            x.sum()
            t.append(time.perf_counter() - start)
    med, mad = d2l._hw_stats(t)
    return dict(gbs=nbytes / med / 1e9, mad=mad / med, threads=threads,
                nbytes=nbytes)

#@save
def _tier_spec(tier, column='bw_gbs'):
    """A column of the tier table's row (the registered table)."""
    t = d2l._hw_static('tiers').set_index('tier')
    return float(t.loc[tier, column])

#@save
def _show_dram(r):
    spec = _tier_spec('host_dram')
    print(f"host DRAM read, {r['threads']} threads: {r['gbs']:.0f} GB/s, "
          f"{r['gbs'] / spec:.0%} of {spec:.1f} (8 channels x 25.6 GB/s)")

#@save
def hw_host_mem_bench(threads=32, nbytes=2**33):
    """Read bandwidth of host memory (GB/s) with `threads` cores summing a
    buffer of nbytes (8 GiB, far beyond the caches)."""
    def live(r, same):
        x = _host_read(threads, 2**32, repeats=3)
        return d2l._hw_agreement([('4 GiB', r['gbs'], x['gbs'], r['mad'])],
                                 same, 'GB/s')
    return d2l._hw_result('box-storage',
                          lambda: _host_read(threads, nbytes),
                          dict(threads=threads, nbytes=nbytes), entry='dram',
                          live_check=live, show=_show_dram,
                          live_label='host DRAM')
```

## Drives

fio reads a file of its own in `../data/hw-scratch/` with `O_DIRECT` (no page
cache) through io_uring (the box has no libaio headers). A sweep writes the
file once, so every read hits flash, reads it at each block size, pattern
and queue depth, and deletes it. fio's `disk_util` reports the sectors each
drive under the file system read, which shows RAID 1 serving reads from
both mirrors.

```{.python .input #memory-tiers-lib-fio}
%%tab pytorch
#@save
def _scratch(nbytes, free_gb):
    """A path in the scratch folder for a file of nbytes, or None when the
    disk has less than free_gb GB free."""
    os.makedirs(_HW_SCRATCH, exist_ok=True)
    if shutil.disk_usage(_HW_SCRATCH).free < free_gb * 1e9:
        return None
    return os.path.join(_HW_SCRATCH,
                        f'fio-{nbytes >> 20}m-{os.getpid()}.dat')

#@save
def _fio_cmd(path, size, rw, bs, qd=1, jobs=1, runtime=10, ramp=2):
    """fio's command for one run on a file of size bytes: each job its own
    slice of the file, O_DIRECT through io_uring, JSON out (runtime None:
    one pass over the slices)."""
    part = size // jobs
    cmd = [d2l.hw_bin('fio'), '--name=bench', f'--filename={path}',
           f'--size={part}', f'--offset_increment={part}', f'--rw={rw}',
           f'--bs={bs}', f'--iodepth={qd}', f'--numjobs={jobs}',
           '--ioengine=io_uring', '--direct=1', '--group_reporting',
           '--output-format=json']
    if runtime:
        cmd += ['--time_based', f'--runtime={runtime}', f'--ramp_time={ramp}']
    return cmd

#@save
def _fio_runs(path, size, points, runtime, ramp):
    """Write the file once, then one fio run per (rw, bs, qd, jobs) point:
    rows of parse_fio plus the share of the bytes the less busy drive
    read (0.5 when RAID 1 splits reads evenly)."""
    run = d2l.hw_run(_fio_cmd(path, size, 'write', 2**20, 32, 1, None, None),
                     timeout_s=900)
    if run.refused:
        raise RuntimeError(f'fio write: {run.refused}: {run.stderr[-500:]}')
    rows = []
    for rw, bs, qd, jobs in points:
        run = d2l.hw_run(_fio_cmd(path, size, rw, bs, qd, jobs, runtime,
                                  ramp), timeout_s=runtime + ramp + 120)
        if run.refused:
            raise RuntimeError(f'fio: {run.refused}: {run.stderr[-500:]}')
        r = d2l.parse_fio(run.stdout)
        drives = r.pop('drive_bytes')
        total = sum(drives.values())
        r['mirror_share'] = (min(drives.values()) / total
                             if len(drives) > 1 and total else math.nan)
        rows.append(r)
    return pd.DataFrame(rows)

#@save
def _fio_compute(spec):
    size = spec['file_gib'] * 2**30
    if d2l.hw_bin('fio') is None:
        raise RuntimeError('fio is not built: run tools/build_hw_tools.sh')
    path = _scratch(size, free_gb=40)
    if path is None:
        raise RuntimeError('less than 40 GB free for the 16 GiB fio file')
    points = [(rw, bs, qd, jobs) for rw in spec['rw']
              for qd, jobs in spec['qd'] for bs in spec['bs']]
    try:
        return _fio_runs(path, size, points, spec['runtime'], spec['ramp'])
    finally:
        if os.path.exists(path):
            os.remove(path)

#@save
def _fio_knee(fio):
    """(beta in bytes/s, IOPS, knee block size beta / IOPS): sequential
    read at depth, random 4 KiB read at depth."""
    deep = fio[fio.qd > 1]
    beta = deep[deep.rw == 'read'].bw_gbs.max() * 1e9
    small = deep[(deep.rw == 'randread') & (deep.bs == deep.bs.min())]
    iops = small.iops.max()
    return beta, iops, beta / iops

#@save
def _show_fio(fio):
    sheet = _tier_spec('nvme')
    beta, iops, knee = _fio_knee(fio)
    _, axes = d2l.plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, col, label in ((axes[0], 'bw_gbs', 'bandwidth (GB/s)'),
                           (axes[1], 'iops', 'reads per second')):
        X, Y, legend, fmts = [], [], [], []
        for (rw, qd, jobs), g in fio.groupby(['rw', 'qd', 'jobs'],
                                             sort=False):
            X.append(g.bs.to_numpy())
            Y.append(g[col].to_numpy())
            legend.append(f'{rw}, QD {qd}' + (f' x {jobs} jobs' if jobs > 1
                                              else ''))
            fmts.append(('C0' if rw == 'read' else 'C1')
                        + ('-' if qd > 1 else '--'))
        d2l.plot(X, Y, 'block size (bytes)', label, xscale='log',
                 yscale='log', fmts=fmts, axes=ax)
        ax.set_xscale('log', base=2)
        ax.axvline(knee, color='k', ls=':', lw=1)
    axes[0].axhline(sheet, color='k', ls='--', lw=1)
    axes[1].legend(axes[0].get_lines(),  # one legend, in the empty corner
                   legend + ['knee beta / IOPS', 'datasheet, one drive'],
                   fontsize=7, loc='upper right')
    big = fio[(fio.rw == 'read') & (fio.qd > 1)].sort_values('bs').iloc[-1]
    print(f'sequential read, {d2l._hw_bytes(int(big.bs))} at QD {big.qd} x '
          f'{big.jobs}: {big.bw_gbs:.1f} GB/s, {big.bw_gbs / sheet:.1f}x one '
          f"drive's {sheet:.2f}")
    print(f'RAID 1: the less busy drive read {big.mirror_share:.0%} of the '
          f'bytes')
    print(f'knee: beta / IOPS = {beta / 1e9:.1f} GB/s / {iops / 1e3:.0f}K '
          f'reads/s = {_size(knee)}')

#@save
def _fio_live(r, same):
    if d2l.hw_bin('fio') is None:
        return 'skipped: fio is not built (tools/build_hw_tools.sh)'
    size = 2**31
    path = _scratch(size, free_gb=10)
    if path is None:
        return 'skipped: less than 10 GB of disk free'
    picks = [('read', 2**20, 64, 4), ('randread', 2**12, 64, 4)]
    try:
        live = _fio_runs(path, size, picks, runtime=10, ramp=2)
    finally:
        if os.path.exists(path):
            os.remove(path)
    pts = []
    for (rw, bs, qd, jobs), x in zip(picks, live.bw_gbs):
        i = _row(r, rw=rw, bs=bs, qd=qd, jobs=jobs)
        pts.append((f'{rw} {d2l._hw_bytes(bs)}', r.bw_gbs.iloc[i], x,
                    _spread(r, 'bw_gbs', i)))
    return d2l._hw_agreement(pts, same, 'GB/s')

#@save
def hw_fio_sweep(config='box'):
    """fio on a 16 GiB file of this machine's drives (O_DIRECT, io_uring):
    sequential and random reads at block sizes 4 KiB to 4 MiB and queue
    depths 1 and 64 x 4 jobs, with each drive's share of the bytes; the
    live check reads a 2 GiB file at two points (skipped without fio or
    with less than 10 GB free)."""
    spec = _HW_FIO[config]
    return d2l._hw_result('box-storage', lambda: _fio_compute(spec),
                          dict(config=config, **spec), entry='fio',
                          live_check=_fio_live, show=_show_fio,
                          live_label='fio')
```

## The Object Store

Range GETs of one registered file of the book's bucket over the public
internet. The time to the first byte includes the connection (DNS, TCP,
TLS), as for a loader that opens each shard afresh; one request before the
timed ones warms the resolver and the CDN.

```{.python .input #memory-tiers-lib-object}
%%tab pytorch
#@save
def _range_get(url, offset, nbytes, timeout=120):
    """(seconds to the first byte, seconds to the last) of one range GET."""
    import urllib.request
    req = urllib.request.Request(url, headers={
        'Range': f'bytes={offset}-{offset + nbytes - 1}',
        'User-Agent': 'd2l-bench'})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read(1)
        first = time.perf_counter() - t
        n = 1 + len(r.read())
    if n != nbytes:
        raise IOError(f'range GET returned {n} bytes, not {nbytes}')
    return first, time.perf_counter() - t

#@save
def _parallel_gets(url, streams, nbytes, offset=0):
    """Aggregate bytes/s of `streams` concurrent range GETs of nbytes."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(streams) as ex:
        t = time.perf_counter()
        list(ex.map(lambda i: _range_get(url, offset + i * nbytes, nbytes),
                    range(streams)))
        return streams * nbytes / (time.perf_counter() - t)

#@save
def _object_size(url):
    """Bytes of the object at url (a HEAD request)."""
    import urllib.request
    req = urllib.request.Request(url, method='HEAD',
                                 headers={'User-Agent': 'd2l-bench'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])

#@save
def _object_compute(spec):
    url = d2l.DATA_HUB[_HW_PROBE_KEY][0]
    size, reps = _object_size(url), spec['repeats']
    _range_get(url, 0, 2**16)  # warm the resolver and the CDN
    rows = []
    for rep in range(reps):  # each repeat reads elsewhere in the object
        for s in spec['sizes']:
            first, total = _range_get(url, rep * (size - s) // reps, s)
            rows.append(dict(stream='single', bytes=s, rep=rep,
                             first_s=first, seconds=total))
        n, b = spec['streams'], spec['stream_bytes']
        bps = _parallel_gets(url, n, b,
                             offset=rep * (size - n * b) // reps)
        rows.append(dict(stream=f'{n} streams', bytes=n * b, rep=rep,
                         first_s=math.nan, seconds=n * b / bps))
    df = pd.DataFrame(rows)
    df.attrs['streams'] = spec['streams']
    return df

#@save
def object_fit(obj):
    """From an object-store table: alpha (s) and beta (bytes/s) fitted to
    the single stream, its median time to the first byte (s) and the
    aggregate bytes/s of the concurrent streams."""
    one, many = obj[obj.stream == 'single'], obj[obj.stream != 'single']
    alpha, beta, _ = d2l.fit_alpha_beta(one.bytes, one.seconds)
    return dict(alpha=alpha, beta=beta, first_s=float(one.first_s.median()),
                beta_streams=float(np.median(many.bytes / many.seconds)),
                streams=obj.attrs['streams'])

#@save
def _show_object(obj):
    a = object_fit(obj)
    print(f"first byte: {a['first_s'] * 1e3:.0f} ms measured, alpha "
          f"{a['alpha'] * 1e3:.0f} ms fitted")
    print(f"bandwidth: {a['beta'] / 1e6:.0f} MB/s one stream, "
          f"{a['beta_streams'] / 1e6:.0f} MB/s over {a['streams']} streams")

#@save
def _object_live(r, same):
    url = d2l.DATA_HUB[_HW_PROBE_KEY][0]
    _range_get(url, 0, 2**16)
    pts = []
    for s in (2**16, 2**24):
        t = r[(r.stream == 'single') & (r.bytes == s)].seconds.median()
        live = _range_get(url, 0, s)[1]
        pts.append((d2l._hw_bytes(s), s / t / 1e6, s / live / 1e6, 0.2))
    return d2l._hw_agreement(pts, same, 'MB/s', floor=0.5)

#@save
def hw_object_store(config='box'):
    """Range GETs of 64 KiB to 64 MiB from the book's object store, one
    stream (three repeats), and eight concurrent 16 MiB streams: the time
    to the first byte of each (object_fit fits alpha and beta)."""
    spec = _HW_OBJECT[config]
    return d2l._hw_result('box-storage', lambda: _object_compute(spec),
                          dict(config=config, **spec), entry='object',
                          live_check=_object_live, show=_show_object,
                          live_label='object store')
```

## Tiers, Measured

```{.python .input #memory-tiers-lib-tiers}
%%tab pytorch
#@save
def tier_measurements(links, agg, dram, fio, obj):
    """The measured bandwidth (GB/s) and latency (ns) of each tier of the
    tier table this machine has: the pinned host link (one card, and all
    cards at once as pcie_host_all), host DRAM, the drives (sequential read
    at depth; 4 KiB random read at depth 1) and the object store (one
    stream)."""
    pin = links.loc[('pinned', 'h2d')]
    every = agg[agg.direction == 'h2d'].sort_values('n_cards').iloc[-1]
    rand = fio[(fio.rw == 'randread') & (fio.qd == 1)].sort_values('bs')
    o = object_fit(obj)
    rows = [('pcie_host', pin.beta / 1e9, pin.alpha * 1e9),
            ('pcie_host_all', every.gbs, math.nan),
            ('host_dram', dram['gbs'], math.nan),
            ('nvme', fio[fio.rw == 'read'].bw_gbs.max(),
             rand.lat_mean_us.iloc[0] * 1e3),
            ('object_store', o['beta'] / 1e9, o['first_s'] * 1e9)]
    df = pd.DataFrame(rows, columns=['tier', 'measured_bw_gbs',
                                     'measured_latency_ns']).set_index('tier')
    df.attrs['n_cards'] = int(every.n_cards)
    return df

#@save
def plot_tiers(tiers, measured):
    """Datasheet bandwidth of each tier (bars, GB/s on a log axis, as in
    fig_bandwidth_ladder) with this machine's measurements (dots); returns
    the rows that have both a datasheet figure and a measurement:
    datasheet, measured, their ratio."""
    t = tiers.set_index('tier')
    k = measured.attrs.get('n_cards', 1)
    spec = dict(t.bw_gbs, pcie_host_all=k * t.bw_gbs['pcie_host'])
    names = dict(t.name, pcie_host_all=f'host links, {k} cards at once')
    at = list(t.index).index('pcie_host') + 1
    order = list(t.index[:at]) + ['pcie_host_all'] + list(t.index[at:])
    y = np.arange(len(order))[::-1]
    d2l.set_figsize((5, 3.6))
    ax = d2l.plt.gca()
    ax.barh(y, [spec[n] for n in order], color='C0', alpha=0.5,
            label='datasheet')
    ax.plot([measured.measured_bw_gbs.get(n, math.nan) for n in order], y,
            'o', color='C1', label='measured here')
    ax.set_yticks(y, [names[n] for n in order], fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('bandwidth (GB/s)')
    ax.legend(fontsize=8, loc='lower right')
    df = pd.DataFrame([(names[n], spec[n], measured.measured_bw_gbs[n])
                       for n in order if n in measured.index
                       and np.isfinite(spec[n])],
                      columns=['tier', 'datasheet', 'measured'])
    df['fraction'] = df.measured / df.datasheet
    return df.set_index('tier')

#@save
def _seconds(t):
    """0.000012 -> '12 us', 0.07 -> '70 ms' (NaN -> '-')."""
    if not np.isfinite(t):
        return '-'
    for unit, scale in (('s', 1), ('ms', 1e-3), ('us', 1e-6), ('ns', 1e-9)):
        if t >= scale:
            return f'{d2l._hw_sig(t / scale)} {unit}'
    return f'{t:.1e} s'

#@save
def _size(n):
    """Bytes as the prose quotes them: '230 KB', '38 MB' (NaN -> '-')."""
    if not np.isfinite(n):
        return '-'
    for unit, scale in (('PB', 1e15), ('TB', 1e12), ('GB', 1e9), ('MB', 1e6),
                        ('KB', 1e3)):
        if n >= scale:
            return f'{d2l._hw_sig(n / scale)} {unit}'
    return f'{n:.0f} B'

#@save
def show_tiers(tiers):
    """Print the tier table: bandwidth, latency, capacity per node and cost
    rank per tier (the example devices, sources and dates stay in the
    frame), and the measured columns when the frame has them."""
    cols = {'GB/s': [_tier_num(b) for b in tiers.bw_gbs],
            'latency': [_seconds(x * 1e-9) for x in tiers.latency_ns],
            'per node': [_size(c * 1e9) for c in tiers.capacity_gb_node],
            'cost': [f'{c:.0f}' if np.isfinite(c) else '-'
                     for c in tiers.cost_rank]}
    if 'measured_bw_gbs' in tiers:
        cols['measured GB/s'] = [_tier_num(b) for b in tiers.measured_bw_gbs]
    d2l.show_table(pd.DataFrame(cols, index=list(tiers.name)))

#@save
def _tier_num(x):
    """A table number at three significant digits, '-' when absent."""
    return d2l._hw_sig(x, 3) if np.isfinite(x) else '-'

#@save
def prefix_words(n):
    """A crossover prefix length as the table prints it: 'any' (from one
    token), 'never' (inf), else the count."""
    if not np.isfinite(n):
        return 'never'
    return 'any' if n <= 1 else f'{n:,.0f} tokens'

#@save
def shard_rule(links, fio, obj):
    """alpha, beta and s_90 = 9 alpha beta (the read size that gets 90% of
    the bandwidth) for the pinned host link, the drives (one reader at
    queue depth 1, a request taking 1 / IOPS) and the object store (one
    stream); prints them and returns the table (seconds, bytes/s,
    bytes)."""
    one = fio[(fio.rw == 'read') & (fio.qd == 1) & (fio.jobs == 1)]
    a_nvme, b_nvme, _ = d2l.fit_alpha_beta(one.bs, 1 / one.iops)
    pin, o = links.loc[('pinned', 'h2d')], object_fit(obj)
    df = pd.DataFrame(
        [('host link, pinned', pin.alpha, pin.beta),
         ('NVMe, one reader', a_nvme, b_nvme),
         ('object store, one stream', o['alpha'], o['beta'])],
        columns=['tier', 'alpha', 'beta']).set_index('tier')
    df['s_90'] = 9 * df.alpha * df.beta
    d2l.show_table(pd.DataFrame({
        'alpha': [_seconds(a) for a in df.alpha],
        'beta (GB/s)': [d2l._hw_sig(b / 1e9) for b in df.beta],
        's_90': [_size(s) for s in df.s_90]}, index=df.index))
    return df
```

## The CPU Optimizer

The CPU side of the offloaded step, measured alone: per tensor, the bf16
gradient (as the GPU sends it) into an fp32 buffer, a fused AdamW step on
the fp32 master weights and moments, and the updated master weights back to
bf16 for the trip up. That is 40 bytes of host-memory traffic per
parameter; the offloaded run does exactly this.

```{.python .input #memory-tiers-lib-adam}
%%tab pytorch
#@save
class _CPUAdam:
    """fp32 master weights, gradients and AdamW moments in pinned host
    memory for tensors shaped like `shapes` (initialized from `init`, else
    zero), bf16 gradient and weight buffers in pinned memory, one fused
    AdamW per tensor, so a tensor can be updated as soon as its gradient
    arrives."""
    def __init__(self, shapes, init=None, lr=1e-5):
        self.master = [(init[i].detach().float().cpu() if init is not None
                        else torch.zeros(s)).pin_memory()
                       for i, s in enumerate(shapes)]
        self.g16 = [torch.zeros(s, dtype=torch.bfloat16, pin_memory=True)
                    for s in shapes]
        self.w16 = [torch.zeros(s, dtype=torch.bfloat16, pin_memory=True)
                    for s in shapes]
        for m in self.master:
            m.grad = torch.zeros_like(m)
        self.opts = [torch.optim.AdamW([m], lr=lr, fused=True)
                     for m in self.master]

    def update(self, i):
        """Tensor i: bf16 gradient in, AdamW step, bf16 weights out."""
        self.master[i].grad.copy_(self.g16[i])
        self.opts[i].step()
        self.w16[i].copy_(self.master[i])

#@save
def _adam_seconds(n_params, threads, repeats=5, size=2**22):
    """Seconds per parameter of _CPUAdam.update over n_params parameters
    in tensors of `size` (after one warm-up pass)."""
    n = max(1, n_params // size)
    adam = _CPUAdam([(size,)] * n)
    with _cpu_threads(threads):
        t = []
        for _ in range(repeats + 1):
            start = time.perf_counter()
            for i in range(n):
                adam.update(i)
            t.append(time.perf_counter() - start)
    return float(np.median(t[1:])) / (n * size)

#@save
def hw_cpu_adam_bench(threads=32, n_params=10**8):
    """s_adam: seconds per parameter of the CPU update of an offloaded
    step (bf16 gradient in, fused AdamW in fp32, bf16 weights out) on
    `threads` cores; the live check repeats it with three timings."""
    def live(r, same):
        x = _adam_seconds(n_params, threads, repeats=3)
        return d2l._hw_agreement([('AdamW', r * 1e9, x * 1e9, 0.05)], same,
                                 'ns per parameter')
    return d2l._hw_result('box-storage',
                          lambda: _adam_seconds(n_params, threads),
                          dict(threads=threads, n_params=n_params),
                          entry='cpu-adam', live_check=live,
                          live_label='CPU AdamW')
```

## The Offloaded Run

The toy follows ZeRO-Offload's delayed parameter update. bf16 weights and
gradients live on the GPU, fp32 master weights and AdamW moments in pinned
host memory (`_CPUAdam`). A step runs its micro-batches with gradient
accumulation and `chunked_lm_loss`. On the last micro-batch a hook
(`register_post_accumulate_grad_hook`) sends each finished bf16 gradient
down on a side stream and queues it for a CPU worker thread; the worker
updates that tensor (`_CPUAdam.update`) and sends its bf16 weights up on a
second side stream into one of two GPU staging copies of the weights. The
GPU meanwhile runs the next step on the weights it has, so the update of
step k has the whole of step k + 1 to finish; step k + 2 waits for it and
copies it in (one-step staleness). Arms: `offload` (all of it), `compute`
(the same micro-batches, no transfer, no update); alone, once: `transfer`
(2N bytes down and 2N up at the same time) and `cpu` (the update of every
tensor).

```{.python .input #memory-tiers-lib-offload}
%%tab pytorch
#@save
def _offload_model(c, seed=0):
    """The run's Qwen3-style model: random bf16 weights on the GPU."""
    import transformers
    cfg = transformers.Qwen3Config(
        hidden_size=c['d_model'], num_hidden_layers=c['n_layers'],
        num_attention_heads=c['n_heads'], num_key_value_heads=c['n_kv_heads'],
        head_dim=c['head_dim'], intermediate_size=c['d_ff'],
        vocab_size=c['vocab'], tie_word_embeddings=True,
        max_position_embeddings=2 * c['seq_len'])
    torch.manual_seed(seed)
    with d2l._quiet(), torch.device('cuda'):
        model = transformers.AutoModelForCausalLM.from_config(
            cfg, dtype=torch.bfloat16, attn_implementation='sdpa')
    return model.train()

#@save
class _Offloader:
    """The delayed parameter update of ZeRO-Offload for `model` (see the
    text above): hooks send gradients down on the last micro-batch, a CPU
    thread updates each tensor as it lands and sends its weights up into
    one of two staging copies; apply() at a step's start installs the
    update of two steps back."""
    def __init__(self, model, threads):
        import queue
        import threading
        self.params = list(model.parameters())
        self.adam = _CPUAdam([p.shape for p in self.params], self.params)
        self.stage = [[torch.empty_like(p) for p in self.params]
                      for _ in range(2)]
        self.down, self.up = torch.cuda.Stream(), torch.cuda.Stream()
        self.queue, self.pending = queue.Queue(), []
        self.last, self.n = False, 0
        self.index = {id(p): i for i, p in enumerate(self.params)}
        self.hooks = [p.register_post_accumulate_grad_hook(self._hook)
                      for p in self.params]
        self.worker = threading.Thread(target=self._work, args=(threads,),
                                       daemon=True)
        self.worker.start()

    def _hook(self, p):
        if not self.last:  # accumulate until the step's last micro-batch
            return
        i, g = self.index[id(p)], p.grad
        self.down.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.down):
            self.adam.g16[i].copy_(g, non_blocking=True)
            landed = torch.cuda.Event()
            landed.record()
        g.record_stream(self.down)  # keep it until the copy is done
        self.queue.put(('grad', i, landed))

    def _work(self, threads):
        torch.set_num_threads(threads)
        slot = 0
        while True:
            item = self.queue.get()
            if item is None:
                return
            kind = item[0]
            if kind == 'slot':  # the staging copy this update fills
                slot = item[1]
            elif kind == 'done':
                item[1].set()
            else:
                _, i, landed = item
                landed.synchronize()
                self.adam.update(i)
                with torch.cuda.stream(self.up):
                    self.stage[slot][i].copy_(self.adam.w16[i],
                                              non_blocking=True)

    def begin(self):
        """Before the step's last micro-batch: its gradients go down."""
        import threading
        self.slot, self.last = self.n % 2, True
        self.done, self.n = threading.Event(), self.n + 1
        self.queue.put(('slot', self.slot))

    def end(self):
        self.last = False
        self.queue.put(('done', self.done))
        self.pending.append((self.done, self.slot))

    def apply(self):
        """At a step's start: install the update of two steps back."""
        if len(self.pending) < 2:
            return
        done, slot = self.pending.pop(0)
        done.wait()
        torch.cuda.current_stream().wait_stream(self.up)
        with torch.no_grad():
            torch._foreach_copy_(self.params, self.stage[slot])

    def drain(self):
        for done, _ in self.pending:
            done.wait()
        self.pending.clear()
        torch.cuda.synchronize()

    def close(self):
        self.drain()
        self.queue.put(None)
        self.worker.join()
        for h in self.hooks:
            h.remove()
```

```{.python .input #memory-tiers-lib-offload-run}
%%tab pytorch
#@save
def _offload_steps(model, off, tokens, c, steps, warmup):
    """Median seconds per step of `tokens` tokens (micro-batches of
    micro_seqs sequences), offloaded when off is given, else compute only."""
    n_micro = tokens // (c['micro_seqs'] * c['seq_len'])
    ids = torch.randint(c['vocab'], (c['micro_seqs'], c['seq_len'] + 1),
                        device='cuda')
    times = []
    for _ in range(warmup + steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        if off:
            off.apply()
        for k in range(n_micro):
            if off and k == n_micro - 1:
                off.begin()
            hidden = model.model(input_ids=ids[:, :-1]).last_hidden_state
            loss, n = d2l.chunked_lm_loss(hidden, model.lm_head.weight,
                                          ids[:, 1:], chunk=c['chunk'])
            (loss / n).backward()
        if off:
            off.end()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    if off:
        off.drain()
    return float(np.median(times[warmup:]))

#@save
def _transfer_seconds(off, repeats=5):
    """2N bytes down and 2N bytes up at once (the link alone)."""
    t = []
    for _ in range(repeats + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.stream(off.down):
            for i, p in enumerate(off.params):
                off.adam.g16[i].copy_(p.detach(), non_blocking=True)
        with torch.cuda.stream(off.up):
            for i, p in enumerate(off.params):
                off.stage[0][i].copy_(off.adam.w16[i], non_blocking=True)
        torch.cuda.synchronize()
        t.append(time.perf_counter() - start)
    return float(np.median(t[1:]))

#@save
def _cpu_seconds(off, threads, repeats=3):
    """The CPU update of every tensor, alone."""
    t = []
    with _cpu_threads(threads):
        for _ in range(repeats + 1):
            start = time.perf_counter()
            for i in range(len(off.params)):
                off.adam.update(i)
            t.append(time.perf_counter() - start)
    return float(np.median(t[1:]))

#@save
def _offload_compute(c, tokens=None, arms=('compute', 'offload'),
                     alone=True, steps=None):
    torch.cuda.reset_peak_memory_stats()
    model = _offload_model(c)
    off = _Offloader(model, c['cpu_threads'])
    N = sum(p.numel() for p in off.params)
    rows = []
    try:
        for T in tokens or c['tokens']:
            for arm in arms:
                s = _offload_steps(model, off if arm == 'offload' else None,
                                   T, c, steps or c['steps'], c['warmup'])
                rows.append(dict(tokens=T, arm=arm, seconds=s))
        if alone:  # each once, independent of the tokens per step
            rows.append(dict(tokens=0, arm='transfer',
                             seconds=_transfer_seconds(off)))
            rows.append(dict(tokens=0, arm='cpu', seconds=_cpu_seconds(
                off, c['cpu_threads'])))
    finally:
        off.close()
    df = d2l._hw_frame(rows)
    df.attrs.update(N=N, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    del model, off
    torch.cuda.empty_cache()
    return df

#@save
def _offload_live(r, same, c):
    picks = [min(c['tokens']), max(c['tokens'])]
    live = _offload_compute(c, picks, arms=('offload',), alone=False,
                            steps=3)
    pts = []
    for T, s in zip(live.tokens, live.seconds):
        i = _row(r, tokens=T, arm='offload')
        pts.append((f'{T:,} tokens', r.seconds.iloc[i] * 1e3, s * 1e3,
                    _spread(r, 'seconds', i, 0.05)))
    return d2l._hw_agreement(pts, same, 'ms per step')

#@save
def hw_offload_run(config='box'):
    """The toy offloaded run of HW_OFFLOAD_CONFIG[config]: seconds per step
    against tokens per step, offloaded and compute only, the transfer-only
    and CPU-only times (rows with tokens 0), the parameter count N and the
    peak GPU memory (attrs); the live check repeats the smallest and
    largest token counts, offloaded."""
    c = HW_OFFLOAD_CONFIG[config]
    return d2l._hw_result('box-storage', lambda: _offload_compute(c),
                          dict(config=config, **c), entry='offload',
                          live_check=lambda r, same: _offload_live(r, same,
                                                                   c),
                          live_label='offloaded run')

#@save
def offload_crossover(run):
    """The measured crossover (tokens per step): where the stalled level
    (the offloaded step at the fewest tokens) meets the overlapped line
    (the offloaded step at the most tokens, proportional to tokens)."""
    off = run[run.arm == 'offload'].sort_values('tokens')
    slope = off.seconds.iloc[-1] / off.tokens.iloc[-1]
    return off.seconds.iloc[0] / slope

#@save
def plot_offload(run, T_link, T_cpu, p_eff=None):
    """Step time against tokens per step (log-log): offloaded and compute
    only measured, the link alone and the CPU update alone (flat), and the
    two predicted thresholds; prints the measured crossover against
    max(T_link, T_cpu) and, given the P_eff those used, against the same
    threshold at the rate the compute-only arm trains at."""
    alone = run[run.tokens == 0].set_index('arm').seconds
    comp = run[run.arm == 'compute'].sort_values('tokens')
    off = run[run.arm == 'offload'].sort_values('tokens')
    T = comp.tokens.to_numpy()
    d2l.plot([T] * 4, [off.seconds.to_numpy(), comp.seconds.to_numpy(),
                       np.full(len(T), alone['transfer']),
                       np.full(len(T), alone['cpu'])],
             'tokens per step', 'seconds per step',
             legend=['offloaded', 'compute only', 'link alone (2N each way)',
                     'CPU update alone'],
             xscale='log', yscale='log', fmts=('C1-o', 'C0-', 'k--', 'k:'),
             figsize=(4.5, 3))
    ax = d2l.plt.gca()
    ax.axvline(T_link, color='C2', ls='-.', lw=1)
    ax.axvline(T_cpu, color='C2', ls='--', lw=1)
    measured, predicted = offload_crossover(run), max(T_link, T_cpu)
    print(f'crossover: measured {measured:,.0f}, predicted max(T_link, '
          f'T_cpu) {predicted:,.0f} tokens ({measured / predicted:.2f}x)')
    if p_eff:
        p_train = 6 * run.attrs['N'] * T[-1] / comp.seconds.iloc[-1]
        rescaled = predicted * p_train / p_eff
        print(f"at the run's own {p_train / 1e12:.0f} TFLOP/s the "
              f'threshold is {rescaled:,.0f} tokens '
              f'({measured / rescaled:.2f}x)')
```

## Shards and the Loader

```{.python .input #memory-tiers-lib-shards}
%%tab pytorch
#@save
def _odirect_read(path, nbytes, chunk=2**23):
    """(seconds, bytes) to read the first nbytes of path with O_DIRECT (the
    page cache neither serves nor keeps it) in chunk-sized requests."""
    import mmap
    buf = mmap.mmap(-1, chunk)  # page aligned, as O_DIRECT requires
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        t, done = time.perf_counter(), 0
        while done < nbytes:
            n = os.preadv(fd, [buf], done)
            if n <= 0:
                break
            done += n
        return time.perf_counter() - t, done
    finally:
        os.close(fd)
        buf.close()

#@save
def _loader_rate(shards, tokens, seq_len=2048, batch=16):
    """Tokens per second PackedStream delivers on one core."""
    stream = d2l.PackedStream(shards, seq_len, batch)
    n = max(1, tokens // (seq_len * batch))
    with _cpu_threads(1):
        stream.batch(0)
        t = time.perf_counter()
        for s in range(1, n + 1):
            stream.batch(s % stream.steps_per_epoch)
        return n * seq_len * batch / (time.perf_counter() - t)

#@save
def _shard_compute(shards, tokens, repeats=3):
    path = shards._path(0, 'bin')
    want = min(int(tokens) * shards.dtype.itemsize, os.path.getsize(path))
    want -= want % 4096
    reads = [_odirect_read(path, want) for _ in range(repeats)]
    seconds = float(np.median([t for t, _ in reads]))
    return dict(bytes=reads[0][1], drive_bps=reads[0][1] / seconds,
                loader_tps=_loader_rate(shards, tokens),
                itemsize=shards.dtype.itemsize)

#@save
def _show_shard(r):
    print(f"O_DIRECT read of the shard ({r['bytes'] / 1e6:.0f} MB, 8 MiB "
          f"requests): {r['drive_bps'] / 1e9:.1f} GB/s")
    print(f"PackedStream on one core: {r['loader_tps'] / 1e6:.0f}M tokens/s "
          f"(windows of 2,048 tokens, {r['itemsize']} bytes each)")

#@save
def hw_shard_read(shards, tokens=2**26):
    """The read side of data loading: the first tokens of the first shard
    read with O_DIRECT (bytes/s), and the tokens/s PackedStream delivers
    from it on one core; the live check uses 2^24 tokens."""
    def live(r, same):
        x = _shard_compute(shards, 2**24, repeats=1)
        return d2l._hw_agreement(
            [('drive', r['drive_bps'] / 1e9, x['drive_bps'] / 1e9, 0.1),
             ('loader', r['loader_tps'] / 1e9, x['loader_tps'] / 1e9, 0.1)],
            same, 'G per s')
    return d2l._hw_result('box-storage',
                          lambda: _shard_compute(shards, tokens),
                          dict(layout=d2l._layout_hash(shards),
                               tokens=int(tokens)),
                          entry='shard-read', live_check=live,
                          show=_show_shard, live_label='shards')
```

## The Regeneration Driver

`tools/hw_regen.py box-storage` runs this in each of its three processes with
every card of the box (the four-card measurement) on a quiet box. It makes
the page's calls on the full configuration, then writes the measured tiers
as an entry (`tiers`) and as `tiers.json` in the blueprint's schema (read by
the serving chapters; unmeasured tiers omitted), and one BenchLog record per
measured point.

```{.python .input #memory-tiers-lib-driver}
%%tab pytorch
#@save
def _storage_check(config):
    if not torch.cuda.is_available():
        raise RuntimeError('box-storage needs a CUDA GPU')
    if (os.environ.get('D2L_HW_CONFIG') or 'box') != config:
        raise ValueError(f'config {config!r} needs D2L_HW_CONFIG={config} '
                         '(tools/hw_regen.py --config)')
    name = torch.cuda.get_device_name(0)
    if config == 't3' and ('A100' not in name
                           or torch.cuda.device_count() < 8):
        raise RuntimeError(f'config t3 measures an 8 x A100 node, not '
                           f'{torch.cuda.device_count()} x {name}')

#@save
def _tiers_json(links, fio, obj):
    """tiers.json records (blueprint schema): measured tiers only."""
    recs = []
    for (mode, d), r in links.iterrows():
        recs.append(dict(tier=f'{d}_{mode}', bw_gbs=r.beta / 1e9,
                         bw_ci95=[r.beta_lo / 1e9, r.beta_hi / 1e9],
                         latency_us=r.alpha * 1e6, latency_p99_us=None,
                         n_repeats=10, tool='torch', note='fit_alpha_beta'))
    seq = fio[fio.rw == 'read'].sort_values('bw_gbs').iloc[-1]
    recs.append(dict(tier='nvme_seq_read', bw_gbs=float(seq.bw_gbs),
                     bw_ci95=None, latency_us=float(seq.lat_mean_us),
                     latency_p99_us=float(seq.lat_p99_us), n_repeats=1,
                     tool='fio', note=f'bs {int(seq.bs)}, qd {seq.qd} x '
                     f'{seq.jobs}, O_DIRECT, RAID 1'))
    o = object_fit(obj)
    recs.append(dict(tier='object_store', bw_gbs=o['beta'] / 1e9,
                     bw_ci95=None, latency_us=o['first_s'] * 1e6,
                     latency_p99_us=None, n_repeats=len(obj),
                     tool='urllib range GET', note='public internet'))
    return recs

#@save
def regen_storage(config='box'):
    """Driver of box-storage: the memory-tiers page's calls on the full
    configuration, the measured tiers (entry and tiers.json), and a
    BenchLog record per measured point."""
    _storage_check(config)
    folder = d2l._hw_regen_dir('box-storage')
    os.makedirs(folder, exist_ok=True)
    with d2l.BenchLog(os.path.join(folder, 'benchlog.jsonl'), 'torch',
                      d2l._HW_TIERS[config]) as log:
        h2d = hw_h2d_sweep(sizes=2**np.arange(12, 31, 2),
                           pinned=(True, False), directions=('h2d', 'd2h'))
        agg = hw_h2d_concurrent(config)
        dram = hw_host_mem_bench(threads=32)
        fio = hw_fio_sweep(config)
        obj = hw_object_store(config)
        s_adam = hw_cpu_adam_bench(threads=32)
        run = hw_offload_run(config)
        read = hw_shard_read(d2l.fwedu_shards('100m'), tokens=2**26)
        links = fit_h2d(h2d)
        measured = tier_measurements(links, agg, dram, fio, obj)
        d2l._hw_result('box-storage', lambda: measured, {}, entry='tiers')
        with open(os.path.join(folder, 'tiers.json'), 'w') as f:
            json.dump(_tiers_json(links, fio, obj), f, indent=1)
        for cmd, df in (('hw_h2d_bench', h2d), ('concurrent copies', agg),
                        ('fio', fio), ('range GET', obj),
                        ('offloaded step', run)):
            for rec in df.to_dict('records'):
                log.record(cmd, {k: rec[k] for k in (
                    'mode', 'direction', 'bytes', 'group', 'rw', 'bs', 'qd',
                    'jobs', 'stream', 'tokens', 'arm') if k in rec}, rec)
        for cmd, rec in (('host read', dram), ('cpu adamw',
                                                dict(s_adam=s_adam)),
                         ('shard read', read)):
            log.record(cmd, {}, rec)
```
