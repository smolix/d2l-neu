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
d2l.DATA_HUB['lm-hw-buildbox-storage-v1'] = (
    d2l.LM_DATA_URL + 'lm-hw-buildbox-storage-v1.tar',
    '9b17b56e6491970add49b436659a297f39b12e56')
```

```{.python .input #memory-tiers-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['hw-tiers'] = ('lm-hw-tiers-v1', 'csv')
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
parameters whose tied embedding does not dominate. Its token counts, 1,024
to 131,072 in factors of 2, bracket the toy's own predicted crossover (the
compute-only arm trains at about 88 TFLOP/s and the update takes about
0.35 s on 8 threads: about 6,000 tokens) by about 6x below and 20x above;
below 4,096 tokens a step is one sequence of that length, above it
micro-batches of two sequences of 2,048. The optimizer runs on 8 threads
(`cpu_threads`; it is bound by host memory, so more threads barely speed it
up) and, in the `offload-32` arm, on 32 (`contention_threads`).
The fio sweep is the blueprint's (block sizes 4 KiB to 4 MiB, sequential and
random reads, queue depth 1 and 64) with 10 s runs after a 2 s ramp; the
object-store probe reads ranges of one registered file of the book's bucket.

```{.python .input #memory-tiers-lib-configs}
%%tab pytorch
#@save
HW_OFFLOAD_CONFIG = {
    'box': dict(d_model=2048, n_layers=16, n_heads=16, n_kv_heads=8,
                head_dim=128, d_ff=5632, vocab=32768, seq_len=2048,
                micro_seqs=2, chunk=2048, cpu_threads=8,
                contention_threads=32, warmup=2, steps=5,
                tokens=[2**k for k in range(10, 18)])}

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

#@save
_HW_DIMMS = {'box': dict(channels=8, kind='DDR4', mt_s=3200, read=False)}
```

The box's host memory: EDAC sysfs (no root) shows 8 channels of 2 ranks
of registered DDR4, 256 GB on one controller, but not the speed the DIMMs
run at, which `dmidecode` reads with root only. Two ranks or two DIMMs per
channel can run below the processor's 3200 MT/s, so `read=False` marks the
speed as assumed; once the configured speed has been read, `mt_s` holds it,
`read` is True and the datasheet value follows from it.

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
    """(generation, width) of this process's current GPU's link now
    (nvidia-smi)."""
    uuid = d2l._hw_visible_uuids()[torch.cuda.current_device()]
    row = d2l._hw_smi('--query-gpu=pcie.link.gen.current,'
                      'pcie.link.width.current', '-i', uuid)
    if not row:
        return None, None
    return tuple(int(x) if x.isdigit() else None for x in row[0][:2])

#@save
_PCIE_GEN_OF_GTS = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}

#@save
def _pcie_link_sysfs():
    """(generation, width) of this process's current GPU's link read from
    sysfs: no CUDA work, so reading it does not wake the link."""
    uuid = d2l._hw_visible_uuids()[torch.cuda.current_device()]
    row = d2l._hw_smi('--query-gpu=pci.bus_id', '-i', uuid)
    if not row:
        return None, None
    bus = row[0][0].lower()[-12:]  # 00000000:81:00.0 -> 0000:81:00.0
    folder = f'/sys/bus/pci/devices/{bus}'
    try:
        with open(f'{folder}/current_link_speed') as f:
            gts = float(f.read().split()[0])
        with open(f'{folder}/current_link_width') as f:
            width = int(f.read().strip())
    except (OSError, ValueError):
        return None, None
    return _PCIE_GEN_OF_GTS.get(gts), width

#@save
def _pcie_at_rest(wait_s=20.0, poll_s=0.5):
    """The link at rest: the lowest generation sysfs shows within wait_s
    of polling (a link that just carried traffic needs a few seconds to
    drop to its idle speed)."""
    best, end = _pcie_link_sysfs(), time.time() + wait_s
    while best[0] is not None and best[0] > 1 and time.time() < end:
        time.sleep(poll_s)
        now = _pcie_link_sysfs()
        if now[0] is not None and now[0] < best[0]:
            best = now
    return best

#@save
def _pcie_under_load(seconds=1.0):
    """The link's generation and width while pinned copies of 256 MiB keep
    it busy (nvidia-smi during the copies)."""
    import threading
    host = torch.ones(2**28, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty_like(host, device='cuda')
    stop = threading.Event()

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
    return busy

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

## The Tier Table and the Tools' Parsers

`tier_table` loads the sourced table (`lm-hw-tiers-v1`) and prints it; with
a campaign log it joins the measured columns, and while the log is pending
it prints one status line. The three parsers turn each tool's output into
numbers; `tools/test_lm_hardware.py` checks them on committed samples
(nvbandwidth and fio from this box, gdsio from NVIDIA's GDS documentation).

```{.python .input #memory-tiers-lib-tier-table}
%%tab pytorch
#@save
def tier_table(log=None):
    """The memory and storage tiers, one row each: datasheet bandwidth
    (GB/s), latency (ns), capacity per node (GB), an ordinal cost rank (1 =
    cheapest per byte), example device, source and date. Given a campaign
    log, its measured columns are joined and shown too; while the log is
    pending, one status line."""
    tiers = d2l._hw_static('hw-tiers')
    if isinstance(log, d2l.CampaignPending):
        print(d2l.hw_status_line(log))
        return tiers
    if log is not None:
        tiers = tiers.join(log['tiers'], on='tier')
    show_tiers(tiers)
    return tiers

#@save
def parse_nvbandwidth(text):
    """{test case: matrix of GB/s (rows: sources, columns: targets; NaN
    where not measured), test case + '_sum': SUM line, test case + '_cv':
    COEFFICIENT_OF_VARIATION line} from nvbandwidth's output."""
    out, lines = {}, text.splitlines()
    for k, line in enumerate(lines):
        m = re.match(r'Running (\S+)\.$', line.strip())
        if m:
            rows = []
            for row in lines[k + 3:]:  # after the title and the column row
                if not row.strip():
                    break
                rows.append([float(v) if v != 'N/A' else np.nan
                             for v in row.split()[1:]])
            out[m.group(1)] = np.array(rows)
        m = re.match(r'(SUM|COEFFICIENT_OF_VARIATION) (\S+) (\S+)', line)
        if m:
            suffix = '_sum' if m.group(1) == 'SUM' else '_cv'
            out[m.group(2) + suffix] = float(m.group(3))
    return out

#@save
def parse_fio(json_text):
    """One fio run (--output-format=json, --group_reporting): read GB/s,
    IOPS, mean and p99 latency (us), block size, pattern, queue depth,
    jobs, and drive_bytes, the bytes each physical drive read."""
    doc = json.loads(json_text[json_text.index('{'):])
    job = doc['jobs'][0]
    opt, r = job['job options'], job['read']
    to_bytes = {'k': 2**10, 'm': 2**20, 'g': 2**30}
    bs = opt['bs'].lower()
    drives = {d['name']: d['read_sectors'] * 512 for d in doc.get(
        'disk_util', []) if re.fullmatch(r'nvme\d+n\d+|sd[a-z]+', d['name'])}
    return dict(rw=opt['rw'], bs=int(bs[:-1]) * to_bytes[bs[-1]]
                if bs[-1] in to_bytes else int(bs),
                qd=int(opt.get('iodepth', 1)), jobs=int(opt.get('numjobs', 1)),
                bw_gbs=r['bw_bytes'] / 1e9, iops=r['iops'],
                lat_mean_us=r['lat_ns']['mean'] / 1e3,
                lat_p99_us=r['clat_ns']['percentile']['99.000000'] / 1e3,
                drive_bytes=drives)

#@save
def parse_gdsio(text):
    """The IoType lines of gdsio's output as a table: I/O type, transfer
    type (GPUD: GPUDirect; CPU_GPU: through a bounce buffer; CPUONLY),
    threads, I/O size (bytes), GiB/s and mean latency (us)."""
    pattern = (r'IoType: (\w+) XferType: (\w+) Threads: (\d+) .*?'
               r'IOSize: (\d+)\(KiB\) Throughput: ([\d.]+) GiB/sec, '
               r'Avg_Latency: ([\d.]+) usecs')
    rows = [dict(io=m[0], xfer=m[1], threads=int(m[2]),
                 io_size=int(m[3]) * 2**10, gib_s=float(m[4]),
                 lat_us=float(m[5]))
            for m in re.findall(pattern, text)]
    return pd.DataFrame(rows)
```

## The Host Link

`hw_h2d_bench` times back-to-back copies between two CUDA events; the
buffers come from `_copy_buffers`. A pinned buffer from `pin_memory` lives in
4 KiB pages; the `pinned-2M` mode maps an anonymous buffer in 2 MiB
transparent huge pages (`madvise`, no root needed) and registers it with
`cudaHostRegister`. This box's IOMMU translates every DMA (`DMA-FQ`), so the
page size decides how many translations a large copy needs. The sweep reads
the link's state at rest from sysfs before its first copy (no CUDA work, so
the read does not wake the link) and under load with nvidia-smi.

```{.python .input #memory-tiers-lib-host-link}
%%tab pytorch
#@save
def _copies_per_timing(nbytes, target_s=5e-3, max_copies=50,
                       nominal_bps=25e9):
    """Back-to-back copies per timed round: enough for about target_s at a
    nominal link rate plus 10 us per copy, at least 1, at most
    max_copies."""
    per_copy = 1e-5 + nbytes / nominal_bps
    return int(min(max_copies, max(1, math.ceil(target_s / per_copy))))

#@save
def _huge_pinned(nbytes):
    """A pinned uint8 host tensor of nbytes in 2 MiB transparent huge pages
    (an anonymous mapping, madvise(MADV_HUGEPAGE), first touched, then
    registered with cudaHostRegister; unregistered when it is freed)."""
    import mmap
    import weakref
    huge = 2**21
    mm = mmap.mmap(-1, nbytes + huge,
                   flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    mm.madvise(mmap.MADV_HUGEPAGE)
    whole = torch.frombuffer(mm, dtype=torch.uint8)
    skip = -whole.data_ptr() % huge  # start on a 2 MiB boundary
    host = whole[skip:skip + nbytes]
    host.fill_(1)  # touch every page, so the kernel maps it now
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostRegister(host.data_ptr(), nbytes, 0)
    if int(err) != 0:
        raise RuntimeError(f'cudaHostRegister failed: {err}')
    weakref.finalize(host, cudart.cudaHostUnregister, host.data_ptr())
    return host

#@save
def _huge_fraction(t):
    """Fraction of host tensor t's bytes the kernel maps in huge pages
    (AnonHugePages of its mapping in /proc/self/smaps)."""
    ptr = t.data_ptr()
    with open('/proc/self/smaps') as f:
        blocks = re.split(r'\n(?=[0-9a-f]+-[0-9a-f]+ )', f.read())
    for block in blocks:
        lo, hi = (int(x, 16) for x in block.split()[0].split('-'))
        if lo <= ptr < hi:
            m = re.search(r'AnonHugePages:\s+(\d+) kB', block)
            return min(1.0, int(m.group(1)) * 1024 / t.numel()) if m else 0.0
    return 0.0

#@save
def _copy_buffers(nbytes, pinned=True, numa=None, device=None,
                  huge_pages=False):
    """(host, gpu) uint8 buffers of nbytes for hw_h2d_bench: the host one
    pinned or pageable, allocated and first touched on NUMA node numa if
    given, in 2 MiB pages if huge_pages (pinned only); the GPU one on
    device."""
    if huge_pages and not pinned:
        raise ValueError('2 MiB pages are measured pinned')
    device = torch.device(device or 'cuda')
    with _numa_bind(numa):
        host = (_huge_pinned(nbytes) if huge_pages else
                torch.ones(nbytes, dtype=torch.uint8, pin_memory=pinned))
    return host, torch.empty(nbytes, dtype=torch.uint8, device=device)

#@save
_H2D_MODES = {'pinned': dict(pinned=True),
              'pinned-2M': dict(pinned=True, huge_pages=True),
              'pageable': dict(pinned=False)}

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
def hw_h2d_sweep(sizes, modes=('pinned', 'pinned-2M', 'pageable'),
                 directions=('h2d', 'd2h')):
    """hw_h2d_bench at every size (bytes) for each mode (pinned in 4 KiB
    pages, pinned in 2 MiB pages, pageable) and direction, on this
    process's first GPU; attrs hold the PCIe generation and width at rest
    (sysfs, before the first copy) and under load (nvidia-smi), and the
    fraction of the 2 MiB-page buffer the kernel mapped in huge pages."""
    sizes, modes = [int(s) for s in sizes], list(modes)
    directions = list(directions)

    def compute():
        idle = _pcie_at_rest()
        rows = [dict(mode=m, direction=d, bytes=s,
                     **d2l.hw_h2d_bench(s, direction=d, **_H2D_MODES[m]))
                for m in modes for d in directions for s in sizes]
        df = d2l._hw_frame(rows)
        busy = _pcie_under_load()
        df.attrs.update(pcie_gen_idle=idle[0], pcie_width_idle=idle[1],
                        pcie_gen=busy[0], pcie_width=busy[1])
        if 'pinned-2M' in modes:
            probe = _huge_pinned(2**26)
            df.attrs['huge_fraction'] = _huge_fraction(probe)
        return df
    return d2l._hw_result('box-storage', compute,
                          dict(sizes=sizes, modes=modes,
                               directions=directions), entry='h2d',
                          live_check=_h2d_live, live_label='host link')

#@save
def fit_h2d(h2d, alpha_below=2**20, plateau_from=2**28):
    """Per (mode, direction) of a host-link sweep: alpha (s) from
    fit_alpha_beta on the copies of at most alpha_below bytes, beta
    (bytes/s) as the median rate of the copies of at least plateau_from
    bytes with their range (beta_lo, beta_hi), and beta_fit, the slope of
    one alpha-beta fit over every size (for comparison: these data are not
    alpha-beta shaped)."""
    rows = {}
    for (mode, d), g in h2d.groupby(['mode', 'direction'], sort=False):
        s, t = g.bytes.to_numpy(float), g.ms.to_numpy() / 1e3
        small, big = s <= alpha_below, s >= plateau_from
        alpha, _, _ = d2l.fit_alpha_beta(s[small], t[small])
        rate = s[big] / t[big]
        rows[(mode, d)] = dict(alpha=alpha, beta=float(np.median(rate)),
                               beta_lo=rate.min(), beta_hi=rate.max(),
                               beta_fit=d2l.fit_alpha_beta(s, t)[1])
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.attrs.update(alpha_below=int(alpha_below),
                    plateau_from=int(plateau_from))
    return df

#@save
def _h2d_label(mode, d):
    return {'pinned': 'pinned', 'pinned-2M': 'pinned, 2 MiB pages',
            'pageable': 'pageable'}[mode] + f' {d}'

#@save
def plot_h2d(h2d, links, pcie_gen=4, lanes=16):
    """GB/s against bytes per copy for each mode and direction, with the
    PCIe per-direction ceiling dashed; prints alpha and the plateau beta of
    pinned h2d and d2h (the median, and the range over the plateau's
    copies: not an interval), the 2 MiB-page plateaus, pageable h2d and the
    link state at rest and during the sweep."""
    ceiling = pcie_ceiling(pcie_gen, lanes) / 1e9
    colors = {'pinned': 'C0', 'pinned-2M': 'C2', 'pageable': 'C1'}
    X, Y, legend, fmts = [], [], [], []
    for (mode, d), g in h2d.groupby(['mode', 'direction'], sort=False):
        X.append(g.bytes.to_numpy())
        Y.append(g.gbs.to_numpy())
        legend.append(_h2d_label(mode, d))
        fmts.append(colors.get(mode, 'C3') + ('-' if d == 'h2d' else ':'))
    x = np.sort(h2d.bytes.unique())
    X.append(x)
    Y.append(np.full(len(x), ceiling))
    legend.append(f'PCIe {pcie_gen}.0 x{lanes}, one direction')
    d2l.plot(X, Y, 'bytes per copy', 'bandwidth (GB/s)', xscale='log',
             fmts=fmts + ['k--'], figsize=(4.5, 3))
    ax = d2l.plt.gca()
    ax.set_xscale('log', base=2)
    ax.legend(legend, fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    big = d2l._hw_bytes(links.attrs.get('plateau_from', int(x[-1])))

    def line(mode, d):
        r = links.loc[(mode, d)]
        lo, hi = r.beta_lo / 1e9, r.beta_hi / 1e9
        k = 1 if round(lo) == round(hi) else 0  # never 'range 15-15'
        print(f'{mode} {d}: alpha {r.alpha * 1e6:.0f} us, beta '
              f'{r.beta / 1e9:.0f} GB/s (range {lo:.{k}f}-{hi:.{k}f}), '
              f'{r.beta / 1e9 / ceiling:.0%} of {ceiling:.1f}')
    print(f'plateau beta: the median of the copies of {big} and more, and '
          'their range')
    line('pinned', 'h2d')
    line('pinned', 'd2h')
    if ('pinned-2M', 'h2d') in links.index:
        h, b = (links.loc[('pinned-2M', d)].beta / 1e9
                for d in ('h2d', 'd2h'))
        print(f'pinned, 2 MiB pages: h2d {h:.0f} GB/s ({h / ceiling:.0%}), '
              f'd2h {b:.0f} GB/s ({b / ceiling:.0%})')
    line('pageable', 'h2d')
    a = h2d.attrs
    print(f"link: Gen {a.get('pcie_gen')} x{a.get('pcie_width')} during the "
          f"sweep, Gen {a.get('pcie_gen_idle')} x{a.get('pcie_width_idle')} "
          f"at rest")
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
    relations = _topo_relations(topo)
    groups = _bridge_groups(n, relations)
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
    df.attrs.update(nbytes=nbytes, nvbandwidth=_nvbandwidth_cards(),
                    bridge_cards=_bridge_cards(n, relations))
    return df

#@save
def _bridge_cards(n, relations):
    """Cards behind each host bridge, largest first: GPUs i and j share
    one when `nvidia-smi topo -m` relates them by PHB, PIX or PXB."""
    root = list(range(n))

    def find(i):
        while root[i] != i:
            i = root[i]
        return i
    for (i, j), rel in relations.items():
        if i < n and j < n and rel in ('PHB', 'PIX', 'PXB'):
            root[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    return sorted((roots.count(r) for r in set(roots)), reverse=True)

#@save
def _show_concurrent(agg):
    one = {d: agg.gbs[(agg.n_cards == 1) & (agg.direction == d)].iloc[0]
           for d in ('h2d', 'd2h')}
    for label, g in agg.groupby('group', sort=False):
        h, b = (g[g.direction == d].iloc[0] for d in ('h2d', 'd2h'))
        k = int(h.n_cards)
        ratio = '' if k == 1 else (f' ({h.gbs / (k * one["h2d"]):.0%} of '
                                   f'{k} x one card)')
        print(f'{label}: h2d {h.gbs:.0f}, d2h {b.gbs:.0f} GB/s{ratio}')
    bridge = _per_bridge(agg)
    if bridge:
        print(bridge)
    nvb = agg.attrs.get('nvbandwidth') or {}
    rates = [f'{d} {np.median(nvb[k]):.0f}' for d, k in (
        ('h2d', 'host_to_device'), ('d2h', 'device_to_host')) if nvb.get(k)]
    if rates:
        print(f"nvbandwidth, each card alone (copy engines, 512 MiB): "
              f"{', '.join(rates)} GB/s")

#@save
def _per_bridge(agg):
    """One line: the h2d rate of the pair behind one host bridge as the
    bridge's rate, and every card at once against the bridges times that
    rate; None without both groups or unless every bridge holds two cards
    (only then is the pair the whole bridge)."""
    h2d = agg[agg.direction == 'h2d'].set_index('group').gbs
    all_n = agg[(agg.direction == 'h2d') & (agg.n_cards > 2)]
    if 'two cards, one host bridge' not in h2d.index or all_n.empty:
        return None
    x, n = h2d['two cards, one host bridge'], int(all_n.n_cards.iloc[0])
    # box-storage v1 predates the field; the box has two cards behind each
    # of its two bridges (`nvidia-smi topo -m`, box-p2p's relation matrix)
    cards = agg.attrs.get('bridge_cards') or [2] * (n // 2) + [1] * (n % 2)
    if sum(cards) != n or set(cards) != {2}:
        return None
    total, bridges = float(all_n.gbs.iloc[0]), len(cards)
    return (f'per host bridge: about {x:.0f} GB/s h2d (two cards behind it); '
            f'all {n}: {total / (bridges * x):.0%} of {bridges} x {x:.0f}')

#@save
def _concurrent_live(r, same):
    """One card again, measured as the committed run measured it (eight
    back-to-back copies of the same size, wall clock, median of five)."""
    i = _row(r, n_cards=1, direction='h2d')
    nbytes = int(r.attrs.get('nbytes', 2**28))
    bufs = {0: (torch.ones(nbytes, dtype=torch.uint8, pin_memory=True),
                torch.empty(nbytes, dtype=torch.uint8, device=0))}
    live, _ = d2l._hw_stats([_concurrent_once(bufs, (0,), 'h2d', 8)
                             for _ in range(5)])
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

The page measures host memory with a torch bench: STREAM's operations (a
summing read, a copy, the triad) as tensor operations, bytes counted as
STREAM counts them. Its numbers are labelled "torch bench": STREAM's kernel
names (Copy, Triad) belong to numbers of the compiled STREAM binary only:
STREAM's licence reserves its name for conforming runs of its code. The binary
(`tools/build_hw_tools.sh ... stream`; `parse_stream` reads its output)
checked the bench on this box: its Triad matches the bench's triad, so the
committed numbers stand. Its Copy is higher, because the compiler turns
the copy loop into `memcpy`, whose non-temporal stores skip the
write-allocate read.

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
def _stream_kernels(threads, nbytes, repeats=5):
    """GB/s of the torch bench's kernels on host memory with `threads`
    intra-op threads, three fp32 arrays a, b, c of nbytes each, bytes
    counted as STREAM counts them (read plus written, no write-allocate
    traffic): a summing read of b, the copy a = b and the triad
    a = b + 3 c; with the relative MAD of each."""
    n = nbytes // 4
    a, b, c = (torch.ones(n) for _ in range(3))  # written: every page mapped
    kernels = {'read': (lambda: b.sum(), 4 * n),
               'copy': (lambda: a.copy_(b), 8 * n),
               'triad': (lambda: torch.add(b, c, alpha=3.0, out=a), 12 * n)}
    out = dict(threads=threads, nbytes=nbytes)
    with _cpu_threads(threads):
        for name, (fn, moved) in kernels.items():
            fn()
            t = []
            for _ in range(repeats):
                start = time.perf_counter()
                fn()
                t.append(time.perf_counter() - start)
            med, mad = d2l._hw_stats(t)
            out[name], out[name + '_mad'] = moved / med / 1e9, mad / med
    return out

#@save
def _stream_best(thread_counts, nbytes):
    """Each torch-bench kernel's best rate over the thread counts (the
    achievable rate: memory-bound kernels peak below all cores), with the
    thread count that reached it."""
    runs = [_stream_kernels(k, nbytes) for k in thread_counts]
    out = dict(nbytes=nbytes, by_threads={str(r['threads']): {
        k: r[k] for k in ('read', 'copy', 'triad')} for r in runs})
    for k in ('read', 'copy', 'triad'):
        best = max(runs, key=lambda r: r[k])
        out[k], out[k + '_mad'] = best[k], best[k + '_mad']
        out[k + '_threads'] = best['threads']
    return out

#@save
def _tier_spec(tier, column='bw_gbs'):
    """A column of the tier table's row (the registered table)."""
    t = d2l._hw_static('hw-tiers').set_index('tier')
    return float(t.loc[tier, column])

#@save
def _dram_sheet(machine='box'):
    """Host DRAM's datasheet bandwidth (GB/s: channels x MT/s x 8 bytes),
    the channel count and the DIMM speed with how it is known, as the
    table prints it ('DDR4-3200, assumed' while the configured speed is
    unread). An assumed speed must give the tier table's figure."""
    d = _HW_DIMMS[machine]
    gbs = d['channels'] * d['mt_s'] * 8 / 1e3
    if not d['read'] and not math.isclose(gbs, _tier_spec('host_dram'),
                                          rel_tol=1e-3):
        raise ValueError(f"{gbs} GB/s is not the tier table's host DRAM")
    how = 'configured' if d['read'] else 'assumed'
    return gbs, d['channels'], f"{d['kind']}-{d['mt_s']}, {how}"

#@save
def _show_dram(r):
    # The box's bench: hw_host_mem_bench reads box-storage only.
    spec, channels, speed = _dram_sheet('box')
    moved = r['triad'] * 4 / 3  # with the read that allocates a's line
    print(f"host DRAM, torch bench: read {r['read']:.0f}, copy "
          f"{r['copy']:.0f}, triad {r['triad']:.0f} GB/s "
          f"(triad at {int(r['triad_threads'])} threads)")
    print(f"{channels} x {speed}: {spec:.1f} GB/s; triad "
          f"{r['triad'] / spec:.0%}, with write-allocate {moved:.0f} "
          f"({moved / spec:.0%})")

#@save
def parse_stream(text):
    """The kernel table of a STREAM run (stream.c 5.10): each kernel's best
    rate (copy, scale, add, triad; GB/s, bytes counted as STREAM counts
    them), the array size (elements), the thread count and whether the
    run validated."""
    rates = {m[0].lower(): float(m[1]) / 1e3 for m in re.findall(
        r'^(Copy|Scale|Add|Triad):\s+([\d.]+)', text, re.M)}
    if len(rates) != 4:
        raise ValueError('not a STREAM kernel table')
    size = re.search(r'Array size = (\d+)', text)
    threads = re.search(r'Number of Threads counted = (\d+)', text)
    return dict(rates, elements=int(size[1]) if size else None,
                threads=int(threads[1]) if threads else 1,
                validates='Solution Validates' in text)

#@save
def hw_host_mem_bench(threads=(8, 16, 32), nbytes=2**31):
    """The torch bench on host memory: read, copy and triad (GB/s, bytes
    counted as STREAM counts them), arrays of nbytes (2 GiB, 16 times the
    L3 cache), the best of the thread counts for each kernel; the live
    check repeats the triad at its best thread count on 1 GiB arrays."""
    threads = [int(k) for k in threads]

    def live(r, same):
        x = _stream_kernels(int(r['triad_threads']), 2**30, repeats=3)
        return d2l._hw_agreement([('triad', r['triad'], x['triad'],
                                   r['triad_mad'])], same, 'GB/s')
    return d2l._hw_result('box-storage',
                          lambda: _stream_best(threads, nbytes),
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
    rows of parse_fio plus the number of drives that served the reads and
    the share of the bytes the less busy one read (0.5 when RAID 1 splits
    reads evenly)."""
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
        r['n_drives'] = len(drives)
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
def _fio_best(fio):
    """The drives' bandwidth row: the sequential read at depth with the
    highest rate (every display and tiers.json use this one)."""
    deep = fio[(fio.rw == 'read') & (fio.qd > 1)]
    return deep.loc[deep.bw_gbs.idxmax()]

#@save
def _fio_latency(fio):
    """The drives' latency row: one 4 KiB random read at a time (QD 1, one
    job), the time one request takes, not queueing."""
    one = fio[(fio.rw == 'randread') & (fio.qd == 1) & (fio.jobs == 1)]
    return one.loc[one.bs.idxmin()]

#@save
def _fio_knee(fio):
    """(beta in bytes/s, IOPS, knee block size beta / IOPS): the best
    sequential read at depth, random reads of the smallest block at
    depth."""
    beta = _fio_best(fio).bw_gbs * 1e9
    deep = fio[(fio.rw == 'randread') & (fio.qd > 1)]
    iops = deep[deep.bs == deep.bs.min()].iops.max()
    return beta, iops, beta / iops

#@save
def _show_fio(fio):
    sheet = _tier_spec('nvme')
    beta, iops, knee = _fio_knee(fio)
    _, axes = d2l.plt.subplots(1, 2, figsize=(7, 2.8))
    d2l.plt.subplots_adjust(wspace=0.4)
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
    best = _fio_best(fio)
    n = int(best['n_drives']) if best.get('n_drives', 0) > 0 else 1
    axes[0].axhline(n * sheet, color='k', ls='--', lw=1)
    axes[1].legend(axes[0].get_lines(), legend + [
        'knee beta / IOPS', f'datasheet, {n} drive' + ('s' if n > 1 else '')],
        fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))
    share = best.bw_gbs / (n * sheet)
    print(f'sequential read, {d2l._hw_bytes(int(best.bs))} at QD {best.qd} '
          f'x {best.jobs}: {best.bw_gbs:.1f} GB/s, {share:.0%} of {n} x '
          f'{sheet:.2f} (RAID 1)')
    print(f'RAID 1: the less busy drive read {best.mirror_share:.0%} of the '
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
def _gpu_tiers(micro):
    """This box's GPU tiers from sec_lm_accelerator_design's committed
    box-microbench result (a dict of its entries): rows (tier, GB/s, ns)
    for shared memory (the L1 read kernel; the latency of a working set
    that fits L1), L2 (the best read that fits it) and GDDR (the median
    read of at least eight times L2)."""
    ladder, lat, smem = micro['ladder'], micro['latency'], micro['smem']
    l2 = ladder.attrs['l2_bytes']
    read = ladder[ladder.kind == 'read']
    l1 = lat.attrs.get('l1_bytes') or 2**17
    return [('smem', smem['tbs'] * 1e3,
             lat[lat.bytes <= l1 // 2].ns_per_load.median()),
            ('l2', read[read.bytes <= l2].gbs.max(),
             lat[(lat.bytes > l1) & (lat.bytes <= l2 // 2)]
             .ns_per_load.median()),
            ('gddr', read[read.bytes >= 8 * l2].gbs.median(),
             lat[lat.bytes >= 2 * l2].ns_per_load.median())]

#@save
def tier_measurements(links, agg, dram, fio, obj, gpu_tiers=True):
    """The measured bandwidth (GB/s) and latency (ns) of each tier of the
    tier table this machine has: the GPU's shared memory, L2 and GDDR
    (sec_lm_accelerator_design's committed result, when it exists), the
    pinned host link (plateau beta and fitted alpha; all cards at once as
    pcie_host_all), host DRAM (the torch bench's read), the drives (best
    sequential read at depth; one 4 KiB random read at a time) and the
    object store (one stream; time to the first byte). attrs: the number
    of cards and of drives. gpu_tiers=False leaves out the GPU rows (the
    regeneration stores this machine's storage measurements only)."""
    rows = []
    if gpu_tiers:
        try:
            with d2l._quiet():
                rows = _gpu_tiers(d2l.hw_box_log('box-microbench'))
        except (KeyError, TypeError):  # not committed (or pending) yet
            rows = []
    pin = links.loc[('pinned', 'h2d')]
    every = agg[agg.direction == 'h2d'].sort_values('n_cards').iloc[-1]
    best, one, o = _fio_best(fio), _fio_latency(fio), object_fit(obj)
    rows += [('pcie_host', pin.beta / 1e9, pin.alpha * 1e9),
             ('pcie_host_all', every.gbs, math.nan),
             ('host_dram', dram['read'], math.nan),
             ('nvme', best.bw_gbs, one.lat_mean_us * 1e3),
             ('object_store', o['beta'] / 1e9, o['first_s'] * 1e9)]
    df = pd.DataFrame(rows, columns=['tier', 'measured_bw_gbs',
                                     'measured_latency_ns']).set_index('tier')
    df.attrs.update(n_cards=int(every.n_cards),
                    n_drives=int(best.get('n_drives', 1) or 1))
    return df

#@save
def plot_tiers(tiers, measured, machine='box'):
    """Datasheet bandwidth and latency of each tier (bars on log axes, as
    in fig_bandwidth_ladder and fig_latency_ladder) with this machine's
    measurements (dots); returns the rows where a datasheet figure states
    what was measured: datasheet, measured, their ratio and a note (the
    host links of all cards are drawn against n x the ceiling, not
    compared; RAID 1's pair against n x one drive; host DRAM's read
    against the channels' peak of `machine`, whose DIMM speed the note
    says is assumed or configured)."""
    t = tiers.set_index('tier')
    k, n = measured.attrs.get('n_cards', 1), measured.attrs.get('n_drives', 1)
    dram, _, dram_note = _dram_sheet(machine)
    spec = dict(t.bw_gbs, pcie_host_all=k * t.bw_gbs['pcie_host'],
                nvme=n * t.bw_gbs['nvme'],  # the drives this machine reads
                host_dram=dram)
    names = dict(t.name, pcie_host_all=f'host links, {k} cards')
    if n > 1:
        names['nvme'] = f"{t.name['nvme']}, {n} in RAID 1"
    rows = dict(names, host_dram=f"{names['host_dram']}, read (torch bench)")
    lat = dict(t.latency_ns, pcie_host_all=math.nan)
    at = list(t.index).index('pcie_host') + 1
    order = list(t.index[:at]) + ['pcie_host_all'] + list(t.index[at:])
    y = np.arange(len(order))[::-1]
    _, axes = d2l.plt.subplots(1, 2, figsize=(7, 3.6), sharey=True)
    for ax, sheet, col, label in (
            (axes[0], spec, 'measured_bw_gbs', 'bandwidth (GB/s)'),
            (axes[1], lat, 'measured_latency_ns', 'latency (ns)')):
        ax.barh(y, [sheet[r] for r in order], color='C0', alpha=0.5,
                label='datasheet or ladder')
        ax.plot([measured[col].get(r, math.nan) for r in order], y, 'o',
                color='C1', label='measured here')
        ax.set_xscale('log')
        ax.set_xlabel(label)
    axes[0].set_yticks(y, [names[r] for r in order], fontsize=8)
    axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
    df = pd.DataFrame([(rows[r], spec[r], measured.measured_bw_gbs[r],
                        dram_note if r == 'host_dram' else '')
                       for r in ('gddr', 'host_dram', 'pcie_host', 'nvme')
                       if r in measured.index and np.isfinite(spec[r])],
                      columns=['tier', 'datasheet', 'measured', 'note'])
    df.insert(3, 'fraction', df.measured / df.datasheet)
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
def _size(n, digits=2):
    """Bytes as the prose quotes them: '230 KB', '38 MB' (NaN -> '-')."""
    if not np.isfinite(n):
        return '-'
    for unit, scale in (('PB', 1e15), ('TB', 1e12), ('GB', 1e9), ('MB', 1e6),
                        ('KB', 1e3)):
        if n >= scale:
            return f'{d2l._hw_sig(n / scale, digits)} {unit}'
    return f'{n:.0f} B'

#@save
def show_tiers(tiers):
    """Print the tier table: bandwidth, latency, capacity per node and cost
    rank per tier (the example devices, sources and dates stay in the
    frame), and the measured columns when the frame has them."""
    cols = {'GB/s': [_tier_num(b) for b in tiers.bw_gbs],
            'latency': [_seconds(x * 1e-9) for x in tiers.latency_ns],
            'per node': [_size(c * 1e9, 3) for c in tiers.capacity_gb_node],
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
def _flops_only_chip(p_eff):
    """A chip_specs-like row whose memory never binds: FLOP/s p_eff."""
    return pd.Series(dict(p16_tflops=p_eff / 1e12, p8_tflops=np.nan,
                          mem_bw_tbs=np.inf, mem_gb=np.inf))

#@save
def prefix_words(n, n_max=None):
    """A crossover prefix length as the table prints it: 'any' (from one
    token), 'never' (inf; 'never within' the n_max-token context when
    given), else the count at two significant digits."""
    if not np.isfinite(n):
        return 'never' + (f' within {n_max:,}' if n_max else '')
    return 'any' if n <= 1 else f'{d2l._hw_sig(n)} tokens'

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
bf16 for the trip up. That is 40 bytes of operands per parameter; the
offloaded run does exactly this. The update is bound by host memory: 8
threads run it at about 85% of the speed of 32, and the offloaded run uses
8, because 32 optimizer threads slow the thread that feeds the GPU (the
`offload-32` arm shows it).

```{.python .input #memory-tiers-lib-adam}
%%tab pytorch
#@save
class _CPUAdam:
    """fp32 master weights, gradients and AdamW moments in pinned host
    memory for tensors shaped like `shapes` (initialized from `init`, else
    zero), bf16 gradient and weight buffers in pinned memory, `slots` sets
    of them (an offloaded step writes one set while the update of the step
    before still reads the other), one fused AdamW per tensor, so a tensor
    can be updated as soon as its gradient arrives."""
    def __init__(self, shapes, init=None, lr=1e-5, slots=1):
        self.master = [(init[i].detach().float().cpu() if init is not None
                        else torch.zeros(s)).pin_memory()
                       for i, s in enumerate(shapes)]
        self.g16, self.w16 = ([[torch.zeros(s, dtype=torch.bfloat16,
                                            pin_memory=True) for s in shapes]
                               for _ in range(slots)] for _ in range(2))
        for m in self.master:
            m.grad = torch.zeros_like(m)
        self.opts = [torch.optim.AdamW([m], lr=lr, fused=True)
                     for m in self.master]

    def update(self, i, slot=0):
        """Tensor i: bf16 gradient of `slot` in, AdamW step, bf16 weights
        out into `slot`."""
        self.master[i].grad.copy_(self.g16[slot][i])
        self.opts[i].step()
        self.w16[slot][i].copy_(self.master[i])

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
def hw_cpu_adam_bench(threads=8, n_params=10**8):
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
host memory (`_CPUAdam`, two slots of bf16 buffers). A step runs its
micro-batches with gradient accumulation and `chunked_lm_loss`. On the last
micro-batch a hook (`register_post_accumulate_grad_hook`) sends each
finished bf16 gradient down on a side stream into the step's slot and
queues it for a CPU worker thread; the worker updates that tensor
(`_CPUAdam.update`) and sends its bf16 weights up on a second side stream
into the same slot's GPU staging copy. The GPU meanwhile runs the next step
on the weights it has, so the update of step k has the whole of step k + 1
to finish; step k + 2 waits for it and copies it in (one-step staleness).
Each buffer of a slot is reused two steps later, after that wait, so no
copy overwrites a buffer the worker or an upload still reads.

Arms, at every token count: `compute` (the same micro-batches, no transfer,
no update), `offload` (the update on `cpu_threads` threads) and
`offload-32` (on `contention_threads`); alone, once: `transfer` (2N bytes
down and 2N up at the same time) and `cpu` (the update of every tensor on
`cpu_threads`). Each row also holds the GPU's busy time per step (CUDA
events from the step's first kernel to its last backward) and, offloaded,
the worker's update time per step.

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
    text above): hooks send gradients down into the step's slot on the last
    micro-batch, a CPU thread (`threads` intra-op threads) updates each
    tensor as it lands and sends its weights up into the slot's staging
    copy; apply() at a step's start installs the update of two steps
    back."""
    def __init__(self, model, threads):
        import queue
        import threading
        self.params = list(model.parameters())
        self.adam = _CPUAdam([p.shape for p in self.params], self.params,
                             slots=2)
        self.stage = [[torch.empty_like(p) for p in self.params]
                      for _ in range(2)]
        self.down, self.up = torch.cuda.Stream(), torch.cuda.Stream()
        self.queue, self.pending = queue.Queue(), []
        self.last, self.n, self.threads = False, 0, threads
        self.update_s = []  # the worker's seconds per step's update
        self.index = {id(p): i for i, p in enumerate(self.params)}
        self.hooks = [p.register_post_accumulate_grad_hook(self._hook)
                      for p in self.params]
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def _hook(self, p):
        if not self.last:  # accumulate until the step's last micro-batch
            return
        i, g = self.index[id(p)], p.grad
        self.down.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.down):
            self.adam.g16[self.slot][i].copy_(g, non_blocking=True)
            landed = torch.cuda.Event()
            landed.record()
        g.record_stream(self.down)  # keep it until the copy is done
        self.queue.put(('grad', i, landed))

    def _work(self):
        slot, start = 0, None
        while True:
            item = self.queue.get()
            if item is None:
                return
            kind = item[0]
            if kind == 'slot':  # a step's update begins: its slot, threads
                slot, start = item[1], None
                torch.set_num_threads(item[2])
            elif kind == 'done':
                if start is not None:
                    self.update_s.append(time.perf_counter() - start)
                item[1].set()
            else:
                _, i, landed = item
                landed.synchronize()
                start = start or time.perf_counter()
                self.adam.update(i, slot)
                with torch.cuda.stream(self.up):
                    self.stage[slot][i].copy_(self.adam.w16[slot][i],
                                              non_blocking=True)

    def begin(self):
        """Before the step's last micro-batch: its gradients go down."""
        import threading
        self.slot, self.last = self.n % 2, True
        self.done, self.n = threading.Event(), self.n + 1
        self.queue.put(('slot', self.slot, self.threads))

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
def _micro_shape(tokens, c):
    """(sequences, length, micro-batches) of a step of `tokens` tokens:
    micro-batches of micro_seqs sequences of seq_len, or one shorter
    sequence below seq_len tokens."""
    length = min(c['seq_len'], tokens)
    seqs = max(1, min(c['micro_seqs'], tokens // length))
    return seqs, length, max(1, tokens // (seqs * length))

#@save
def _offload_steps(model, off, tokens, c, steps, warmup):
    """Medians over the timed steps of `tokens` tokens: seconds per step,
    the GPU's busy seconds (CUDA events from the first kernel to the last
    backward) and, offloaded (off given), the worker's update seconds."""
    seqs, length, n_micro = _micro_shape(tokens, c)
    ids = torch.randint(c['vocab'], (seqs, length + 1), device='cuda')
    first, last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times, busy = [], []
    for _ in range(warmup + steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        if off:
            off.apply()
        first.record()
        for k in range(n_micro):
            if off and k == n_micro - 1:
                off.begin()
            hidden = model.model(input_ids=ids[:, :-1]).last_hidden_state
            loss, n = d2l.chunked_lm_loss(hidden, model.lm_head.weight,
                                          ids[:, 1:], chunk=c['chunk'])
            (loss / n).backward()
        last.record()
        if off:
            off.end()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        busy.append(first.elapsed_time(last) / 1e3)
    update = math.nan
    if off:
        off.drain()
        update = float(np.median(off.update_s[-steps:]))
    return dict(seconds=float(np.median(times[warmup:])),
                gpu_s=float(np.median(busy[warmup:])), update_s=update)

#@save
def _transfer_seconds(off, repeats=5):
    """2N bytes down and 2N bytes up at once (the link alone)."""
    t = []
    for _ in range(repeats + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.stream(off.down):
            for i, p in enumerate(off.params):
                off.adam.g16[0][i].copy_(p.detach(), non_blocking=True)
        with torch.cuda.stream(off.up):
            for i, p in enumerate(off.params):
                off.stage[0][i].copy_(off.adam.w16[0][i], non_blocking=True)
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
def _offload_compute(c, tokens=None, arms=('compute', 'offload',
                                           'offload-32'),
                     alone=True, steps=None):
    torch.cuda.reset_peak_memory_stats()
    model = _offload_model(c)
    off = _Offloader(model, c['cpu_threads'])
    threads = {'offload': c['cpu_threads'],
               'offload-32': c['contention_threads']}
    N = sum(p.numel() for p in off.params)
    rows, grid = [], tokens or c['tokens']
    try:
        for arm in threads:  # untimed: the optimizer state and first faults
            off.threads = threads[arm]
            _offload_steps(model, off, grid[0], c, 1, c['warmup'])
        for T in grid:
            for arm in arms:
                off.threads = threads.get(arm, c['cpu_threads'])
                r = _offload_steps(model, off if arm in threads else None,
                                   T, c, steps or c['steps'], c['warmup'])
                rows.append(dict(tokens=T, arm=arm, **r))
        if alone:  # each once, independent of the tokens per step
            rows.append(dict(tokens=0, arm='transfer',
                             seconds=_transfer_seconds(off)))
            rows.append(dict(tokens=0, arm='cpu', seconds=_cpu_seconds(
                off, c['cpu_threads'])))
    finally:
        off.close()
    df = d2l._hw_frame(rows)
    df.attrs.update(N=N, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                    cpu_threads=c['cpu_threads'],
                    contention_threads=c['contention_threads'])
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
    (and the GPU's busy seconds, the worker's update seconds) against
    tokens per step for the arms compute, offload and offload-32, the
    transfer-only and CPU-only times (rows with tokens 0), the parameter
    count N and the peak GPU memory (attrs); the live check repeats the
    smallest and largest token counts, offloaded."""
    c = HW_OFFLOAD_CONFIG[config]
    return d2l._hw_result('box-storage', lambda: _offload_compute(c),
                          dict(config=config, **c), entry='offload',
                          live_check=lambda r, same: _offload_live(r, same,
                                                                   c),
                          live_label='offloaded run')

#@save
def offload_prediction(run):
    """The toy's own threshold: the run's training rate (6 N T over the
    compute-only step at the most tokens), its link-alone time, the CPU
    update's time in the run (the worker's median over the offloaded
    steps; t_cpu_alone: measured alone), T_link and T_cpu at that rate and
    the predicted crossover max(t_link, t_cpu) / (compute seconds per
    token)."""
    alone = run[run.tokens == 0].set_index('arm').seconds
    comp = run[run.arm == 'compute'].sort_values('tokens').iloc[-1]
    per_token = comp.seconds / comp.tokens
    t_link = float(alone['transfer'])
    t_cpu = float(run[run.arm == 'offload'].update_s.median())
    return dict(p_train=6 * run.attrs['N'] / per_token, t_link=t_link,
                t_cpu=t_cpu, t_cpu_alone=float(alone['cpu']),
                T_link=t_link / per_token, T_cpu=t_cpu / per_token,
                predicted=max(t_link, t_cpu) / per_token)

#@save
def offload_crossover(run, arm='offload'):
    """The measured crossover (tokens per step) of an offloaded arm: the
    corner T0 of the least-squares fit of log step time by
    log(a max(1, T / T0)), a stalled level a that turns into a line through
    the origin at T0."""
    off = run[run.arm == arm].sort_values('tokens')
    T, y = off.tokens.to_numpy(float), np.log(off.seconds.to_numpy())
    best = (math.inf, math.nan)
    for T0 in np.geomspace(T[0] / 2, T[-1] * 2, 2000):
        shape = np.log(np.maximum(1, T / T0))
        log_a = np.mean(y - shape)
        best = min(best, (float(np.sum((y - shape - log_a)**2)), T0))
    return best[1]

#@save
def plot_offload(run):
    """Step time against tokens per step (log-log): offloaded on the
    optimizer's threads and on 32, compute only, the link alone and the CPU
    update in the run (flat), and the toy's own thresholds; prints its
    training rate and times, the predicted and measured crossover, the
    excess over compute only above it and the GPU's busy time beside the
    optimizer's threads and beside 32."""
    pred = offload_prediction(run)
    comp = run[run.arm == 'compute'].sort_values('tokens')
    T = comp.tokens.to_numpy()
    series = [run[run.arm == a].sort_values('tokens').seconds.to_numpy()
              for a in ('offload', 'offload-32')]
    d2l.plot([T] * 5, series + [comp.seconds.to_numpy(),
                                np.full(len(T), pred['t_link']),
                                np.full(len(T), pred['t_cpu'])],
             'tokens per step', 'seconds per step', xscale='log',
             yscale='log', fmts=('C1-o', 'C3-.', 'C0-', 'k--', 'k:'),
             figsize=(4.5, 3))
    ax = d2l.plt.gca()
    for x, ls in ((pred['T_link'], '-.'), (pred['T_cpu'], '--')):
        ax.axvline(x, color='C2', ls=ls, lw=1)
    k = run.attrs.get('contention_threads', 32)
    ax.legend([f"offloaded, {run.attrs.get('cpu_threads')} CPU threads",
               f'offloaded, {k} CPU threads', 'compute only',
               'link alone (2N each way)', 'CPU update in the run',
               'T_link of the run', 'T_cpu of the run'],
              fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    measured = offload_crossover(run)
    over = run[(run.arm == 'offload') & (run.tokens >= 2 * measured)]
    base = comp.set_index('tokens').seconds
    excess = np.median(over.seconds.to_numpy()
                       - base[over.tokens].to_numpy()) if len(over) else 0.0
    print(f"the run trains at {pred['p_train'] / 1e12:.0f} TFLOP/s; the link "
          f"alone takes {pred['t_link']:.2f} s per step")
    print(f"CPU update: {pred['t_cpu_alone']:.2f} s alone, {pred['t_cpu']:.2f}"
          f" s during the offloaded steps")
    ratio = measured / pred['predicted']
    print(f"crossover: predicted {d2l._hw_sig(pred['predicted'])}, measured "
          f"{d2l._hw_sig(measured)} tokens ({ratio:.1f}x)")
    print(f'above 2x the crossover the offloaded step exceeds compute only '
          f'by {excess:.2f} s')
    at = int(T[T >= measured][0]) if (T >= measured).any() else int(T[-1])
    busy = run[run.tokens == at].set_index('arm').gpu_s
    print(f'GPU busy at {at:,} tokens: {busy["compute"]:.2f} s alone, '
          f'{busy["offload"]:.2f} s ({run.attrs.get("cpu_threads")} CPU '
          f'threads), {busy["offload-32"]:.2f} s ({k})')
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
def _tiers_json(links, fio, obj, repeats=10):
    """tiers.json records (blueprint schema): measured tiers only. Host
    links: the plateau beta with its range over the large copies and
    alpha, `repeats` timings per point; drives: bandwidth of the best
    sequential read at depth, latency of one 4 KiB random read at a time
    (not queueing time); object store: one stream, time to the first
    byte."""
    n_proc = max(1, int(fio.attrs.get('n_proc', 1)))
    recs = []
    for (mode, d), r in links.iterrows():
        recs.append(dict(tier=f'{d}_{mode}', bw_gbs=r.beta / 1e9,
                         bw_ci95=None, bw_range=[r.beta_lo / 1e9,
                                                 r.beta_hi / 1e9],
                         latency_us=r.alpha * 1e6, latency_p99_us=None,
                         n_repeats=repeats * n_proc, tool='torch',
                         note='beta: median of the copies of 256 MiB and '
                              'more; alpha: fit_alpha_beta up to 1 MiB'))
    best, one = _fio_best(fio), _fio_latency(fio)
    recs.append(dict(tier='nvme_seq_read', bw_gbs=float(best.bw_gbs),
                     bw_ci95=None, latency_us=float(one.lat_mean_us),
                     latency_p99_us=float(one.lat_p99_us), n_repeats=n_proc,
                     tool='fio', note=f'bandwidth: bs {int(best.bs)}, qd '
                     f'{best.qd} x {best.jobs}; latency: 4 KiB random reads '
                     f'at qd 1; O_DIRECT, RAID 1 of {int(best.n_drives)}'))
    o = object_fit(obj)
    recs.append(dict(tier='object_store', bw_gbs=o['beta'] / 1e9,
                     bw_ci95=None, latency_us=o['first_s'] * 1e6,
                     latency_p99_us=None,
                     n_repeats=int((obj.stream == 'single').sum()),
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
        h2d = hw_h2d_sweep(sizes=2**np.arange(12, 31),
                           modes=('pinned', 'pinned-2M', 'pageable'),
                           directions=('h2d', 'd2h'))
        agg = hw_h2d_concurrent(config)
        dram = hw_host_mem_bench(threads=(8, 16, 32))
        fio = hw_fio_sweep(config)
        obj = hw_object_store(config)
        s_adam = hw_cpu_adam_bench(threads=8)
        run = hw_offload_run(config)
        read = hw_shard_read(d2l.fwedu_shards('100m'), tokens=2**26)
        links = fit_h2d(h2d, alpha_below=2**20, plateau_from=2**28)
        measured = tier_measurements(links, agg, dram, fio, obj,
                                     gpu_tiers=False)
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
        for cmd, rec in (('stream kernels', dram), ('cpu adamw',
                                                dict(s_adam=s_adam)),
                         ('shard read', read)):
            log.record(cmd, {}, rec)
```
