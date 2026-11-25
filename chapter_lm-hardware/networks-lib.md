# Scale-Up Domains and Scale-Out Networks: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_networks` that the section explains
in a sentence but does not show (docs/code-guide.md 7.1): the section's
registry entries, one statement each (the systems table, the box's loopback
sweeps and the campaign log C-T4; `DATA_HUB` holds the systems table, the
box-net entry follows its upload), its regeneration job and configuration;
the systems table loader `systems_table` with its short column names
`SYSTEMS_SHORT`; the loopback sweeps (`loopback_sweep`: gloo on CPU
processes, NCCL forced onto its socket transport over loopback) with their
live checks and NCCL's transport record; the quoted latencies (with
sources), `link_betas` and `transport_ladder`; the fat tree's max-flow
check; the perftest parser and fit of the two-node campaign; the displays
of the axis ratios and the layout calculator (`show_ratios`,
`show_layout_fit`, `show_epsilon`, `plot_layout_fit`); the driver
`regen_net`. Blocks are tagged `%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_networks`

## Registries

```{.python .input #networks-lib-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-hw-systems-v1'] = (
    d2l.LM_DATA_URL + 'lm-hw-systems-v1.json',
    '4ef586baa5a2e50dd5dc2d063f5b5a62b84270f5')
```

```{.python .input #networks-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['systems'] = ('lm-hw-systems-v1', 'json')
d2l.HW_ARTIFACTS['box-net'] = ('lm-hw-buildbox-net-v1', 'tar')
d2l.HW_CAMPAIGNS['cluster-nccl'] = (
    'lm-hw-cluster-nccl-v1', 'tar', 'C-T4',
    'two 8 x A100 nodes with RDMA')
```

```{.python .input #networks-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-net'] = (
    'regen_net', lambda: dict(config='box'))
```

## The Systems Table

`systems_table` reads the sourced table (`tools/lm_data/hw/systems.json`,
transcribed 2026-11-25; every row names its page, and numbers that only the
MLSS PDF supplies are marked "as reported"). Vendors quote scale-up bandwidth
per chip with both directions added; the loader halves it (`up_gbs_dir`),
except on a torus, where a collective along one axis uses that axis's two
links. The box's row is measured: its rate is the all-reduce bus bandwidth on
all four cards from :numref:`sec_lm_server_paths`'s committed
`alpha_beta.json`.

```{.python .input #networks-lib-systems}
%%tab pytorch
#@save
SYSTEMS_SHORT = {'domain_chips': 'chips/dom', 'up_gbs_dir': 'up GB/s/dir',
                 'domain_mem_tb': 'dom. TB', 'scaleout_gbs_per_chip':
                 'out GB/s/chip', 'rack_kw': 'rack kW'}  # columns shown

#@save
def systems_table():
    """The systems table (one row per system, indexed by name): chips per
    domain, scale-up GB/s per chip as quoted (scaleup_dir says whether
    both directions are added) and per direction (up_gbs_dir), HBM per chip
    and per domain, scale-out GB/s per chip and direction, dense bf16
    TFLOP/s, power, kind (quoted, measured, spec), source and date."""
    df = pd.DataFrame(d2l._hw_static('systems')['rows']).set_index('system')
    assert df.source_url.notna().all() and df.retrieved.notna().all()
    for name in df.index[df.measured.notna()]:
        df.loc[name, 'scaleup_gbs_per_chip'] = (
            d2l.box_alpha_beta()['beta'] / 1e9)  # sec_lm_server_paths
    up = df.scaleup_gbs_per_chip.astype(float)
    df['up_gbs_dir'] = up.where(df.scaleup_dir != 'bidir', up / 2)
    torus = df.scaleup_topology == 'torus'
    df.loc[torus, 'up_gbs_dir'] = 2 * df.link_gbs_per_dir[torus]  # one axis
    df['domain_mem_tb'] = df.domain_chips * df.hbm_gb_per_chip / 1e3
    return df
```

## Loopback Sweeps

The software floor of the box (the section's "not a datacenter network"):
`loopback_sweep(backend='gloo', ...)` times `all_reduce` of float32 CPU
tensors in 2 and 4 processes over TCP on the loopback interface (the
torch.distributed sweep of :numref:`sec_lm_server_paths` through torchrun,
`CUDA_VISIBLE_DEVICES` empty); `backend='nccl'` runs nccl-tests'
`all_reduce_perf` on the first two cards with the environment that forces
NCCL onto its socket transport, and records the transport NCCL reports. The
result is the same long table as `nccl_sweeps` (`op, g, path, algo, env`
and nccl-tests' columns, `path='loopback'`), so `fit_collectives` fits it:
alpha from the messages up to 64 KiB, beta from those of 4 MiB and more.

```{.python .input #networks-lib-configs}
%%tab pytorch
#@save
_HW_NET = {  # the page's calls, by backend: ranks, sizes, iterations
    'gloo': dict(ranks=(2, 4), env='', min_bytes=64, max_bytes=2**24,
                 iters=20, warmup=5),
    'nccl': dict(ranks=(2,), env='NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 '
                 'NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo', min_bytes=8,
                 max_bytes=2**28, iters=20, warmup=5)}

#@save
_HW_NET_REGIMES = dict(alpha_below=2**16, plateau_from=2**22)  # bytes

#@save
_HW_NET_LIVE = dict(sizes=(2**10, 2**17, 2**24), timeout_s=60)
```

```{.python .input #networks-lib-loopback}
%%tab pytorch
#@save
def _gloo_env():
    """CPU processes on the loopback interface, a few threads each."""
    return dict(CUDA_VISIBLE_DEVICES='', GLOO_SOCKET_IFNAME='lo',
                OMP_NUM_THREADS='4')

#@save
def _gloo_sweep(k, sizes, iters, warmup, timeout_s=None):
    """(rows, HWRun) of the gloo all-reduce sweep in k CPU processes."""
    timeout_s = timeout_s or d2l._sweep_timeout(
        max(sizes), iters, warmup, min_rate=0.2e9)
    return d2l._torchrun_sweep('all_reduce', k, sizes, iters, warmup,
                               _gloo_env(), timeout_s, backend='gloo')

#@save
def _net_info(k, env):
    """NCCL's account of one small all-reduce on the first k cards with
    env: its version and transports ('NET/Socket/0', ...)."""
    exe = d2l.hw_bin('all_reduce_perf')
    if exe is None:
        return dict(version='.'.join(map(str, torch.cuda.nccl.version())),
                    transports=[])
    child = dict(d2l._gpu_env(tuple(range(k)), env), NCCL_DEBUG='INFO')
    run = d2l.hw_run([exe, '-b', 8, '-e', 8, '-g', k, '-n', 1, '-w', 1,
                      '-c', 0], timeout_s=120,
                     env=d2l._nccl_tests_env(child))
    lines = [d2l._hw_scrub(l[l.index('NCCL INFO'):])
             for l in run.stdout.splitlines() if 'NCCL INFO' in l]
    ver = next((re.search(r'NCCL version (\S+)', l).group(1) for l in lines
                if 'NCCL version' in l), None)
    via = sorted({l.split(' via ')[1].split()[0] for l in lines
                  if ' via ' in l})
    return dict(version=ver, transports=via)

#@save
def _loopback_compute(backend, ranks, env, min_bytes, max_bytes, iters,
                      warmup):
    sizes = [int(min_bytes * 2**i) for i in range(
        int(round(math.log2(max_bytes / min_bytes))) + 1)]
    frames, info = [], {}
    for k in ranks:
        if backend == 'gloo':
            rows, run = _gloo_sweep(k, sizes, iters, warmup)
            df = d2l._rows_from_times(rows, 'all_reduce', k)
            df.attrs.update(tool='torch.distributed (gloo)',
                            refused=run.refused)
        else:
            df = d2l.nccl_sweep('all_reduce', tuple(range(k)), env=env,
                                min_bytes=min_bytes, max_bytes=max_bytes,
                                iters=iters, warmup=warmup)
            info[k] = _net_info(k, env)
        if df.attrs.get('refused') or df.empty:
            raise RuntimeError(f'{backend} sweep on {k} ranks: '
                               f"{df.attrs.get('refused') or 'no rows'}")
        frames.append(df.assign(op='all_reduce', g=k, path='loopback',
                                algo='default', env=''))
    out = pd.concat(frames, ignore_index=True)
    out.attrs.update(tool=frames[0].attrs.get('tool'), backend=backend,
                     env=env, info=info, **_HW_NET_REGIMES)
    if info:
        out.attrs['nccl'] = next(iter(info.values()))['version']
    return out

#@save
def _loopback_name(r):
    if r.attrs.get('backend') != 'nccl':
        return 'gloo over TCP'
    via = sorted({t for i in r.attrs.get('info', {}).values()
                  for t in i['transports']})
    if all(t.startswith('NET/Socket') for t in via):  # the intended path
        return 'NCCL over sockets'
    return f"NCCL over {', '.join(via)}"

#@save
def _ci(lo, hi, scale):
    sig = d2l._hw_sig
    return '' if np.isnan(lo) else f' [{sig(lo * scale)}-{sig(hi * scale)}]'

#@save
def _show_loopback(r):
    """One line per number of ranks: alpha (us) and beta (GB/s, bus
    bandwidth) with their 95% intervals over the regeneration's
    processes (none with one process)."""
    ab = d2l.fit_collectives(r)
    if ab.empty:
        print(r.attrs.get('label', 'pending'))
        return
    unit = 'GPUs' if r.attrs.get('backend') == 'nccl' else 'ranks'
    for (_, g, _), x in ab.iterrows():
        print(f'{_loopback_name(r)}, {g} {unit}: alpha '
              f'{d2l._hw_sig(x.alpha * 1e6)} us'
              f'{_ci(x.alpha_lo, x.alpha_hi, 1e6)}, beta '
              f'{d2l._hw_sig(x.beta / 1e9)} GB/s'
              f'{_ci(x.beta_lo, x.beta_hi, 1e-9)}')

#@save
def _loopback_live_us(backend, env, size):
    """The live time (us) of one all-reduce of size bytes on 2 ranks, or
    None when the run was refused."""
    if backend == 'gloo':
        rows, _ = _gloo_sweep(2, [size], 20, 5, _HW_NET_LIVE['timeout_s'])
        return rows[0]['time_us'] if rows else None
    df = d2l.nccl_sweep('all_reduce', (0, 1), env=env, min_bytes=size,
                        max_bytes=size, timeout_s=_HW_NET_LIVE['timeout_s'])
    return None if df.empty else float(df.oop_us.iloc[0])

#@save
def _loopback_live(r, same):
    """The live check of a loopback sweep: 2 ranks at three sizes, one
    short line (the cell names the backend)."""
    backend = r.attrs.get('backend')
    if backend == 'nccl' and torch.cuda.device_count() < 2:
        return 'needs two GPUs: the committed box run is shown'
    pts = []
    for size in _HW_NET_LIVE['sizes']:
        hit = np.flatnonzero(((r.g == 2) & (r['size'] == size)).to_numpy())
        if not len(hit):
            continue
        live = _loopback_live_us(backend, r.attrs.get('env', ''), size)
        if live is None:
            return f'{d2l._hw_bytes(size)}: not run'
        i = int(hit[0])
        pts.append((d2l._hw_bytes(size), float(r.oop_us.iloc[i]), live,
                    d2l._spread(r, 'oop_us', i, 0.05)))
    if not same:
        return d2l._hw_agreement(pts, same, 'us')
    # one short line: loopback times move with the host's load
    worst = max(math.fabs(p[2] - p[1]) / p[1] for p in pts)
    bad = any(math.fabs(p[2] - p[1]) / p[1] > max(5 * p[3], 0.25)
              for p in pts)
    return (f'{len(pts)} sizes on 2 ranks, largest difference {worst:.0%}'
            + (' (differs)' if bad else ''))

#@save
def loopback_sweep(backend='gloo', ranks=(2, 4), env='', max_bytes=2**24):
    """The all-reduce time over loopback on this host: backend 'gloo' (CPU
    processes over TCP) on each number of ranks, or 'nccl' with env (the
    variables that force its socket transport) on the first cards; the
    long sweep table of nccl_sweeps (path 'loopback'). Committed in
    box-net; a normal run prints the fits and re-times 2 ranks at three
    sizes."""
    spec = _HW_NET[backend]
    config = dict(backend=backend, ranks=list(ranks), env=env,
                  min_bytes=spec['min_bytes'], max_bytes=max_bytes,
                  iters=spec['iters'], warmup=spec['warmup'])
    compute = lambda: _loopback_compute(backend, tuple(ranks), env,
                                        spec['min_bytes'], max_bytes,
                                        spec['iters'], spec['warmup'])
    return d2l._hw_result('box-net', compute, config, entry=backend,
                          live_check=_loopback_live, show=_show_loopback,
                          live_label='')
```

## Quoted Latencies and the Transport Ladder

The two quoted rows the measured floor is set against: an RDMA write round
trip between two NICs and one NVLink hop, both from the latency ladder of the
MLSS efficiency lecture (slide 33), with the bandwidth of one NDR port (400
Gb/s, NVIDIA's InfiniBand adapter pages) and of NVLink 3 per direction (the
A100 datasheet, as in the links table).

```{.python .input #networks-lib-ladder}
%%tab pytorch
#@save
def quoted_latencies():
    """Quoted latency (us) and bandwidth (GB/s per direction) of an RDMA
    NIC and an NVLink hop, with what the latency is and the sources."""
    mlss = 'https://alex.smola.org/posts/45-mlss-efficiency/main.pdf'
    return pd.DataFrame([
        dict(transport='RDMA, one NDR port', alpha_us=5.0, beta_gbs=50.0,
             alpha_of='round trip', source='quoted',
             source_url=mlss + ' (slide 33); https://www.nvidia.com/en-us/'
             'networking/infiniband-adapters/'),
        dict(transport='NVLink 3, one hop', alpha_us=1.5, beta_gbs=300.0,
             alpha_of='one hop', source='quoted',
             source_url=mlss + ' (slide 33); https://www.nvidia.com/content/'
             'dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-'
             'us-nvidia-1758950-r4-web.pdf')])

#@save
def link_betas(systems, box):
    """Bytes/s per direction of the three links the ratios compare: NVLink
    and the NIC of an 8 x A100 SXM node, and the box's host-staged path
    (box: a box_alpha_beta record)."""
    node = systems.loc['8 x A100 SXM 40 GB']
    return {'NVLink (A100)': node.up_gbs_dir * 1e9,
            'NIC (A100, HDR)': node.scaleout_gbs_per_chip * 1e9,
            'host-staged (box)': box['beta']}

#@save
def transport_ladder(floor, sock, box, quoted):
    """Print and return, per transport, alpha (us), beta (GB/s per
    direction), what alpha is (a whole all-reduce, a round trip or one
    hop) and whether it was measured here or quoted; floor and sock are
    loopback sweeps, box a box_alpha_beta record, quoted the
    quoted_latencies table."""
    rows = []
    for name, sweep, unit in (('gloo', floor, 'ranks'),
                              ('NCCL sockets', sock, 'GPUs')):
        for (_, g, _), x in d2l.fit_collectives(sweep).iterrows():
            rows.append(dict(transport=f'{name}, {g} {unit}',
                             alpha_us=x.alpha * 1e6, beta_gbs=x.beta / 1e9,
                             alpha_of='all-reduce', source='measured'))
    rows.append(dict(transport=f"NCCL shm, {box['g']} GPUs",
                     alpha_us=box['alpha'] * 1e6, beta_gbs=box['beta'] / 1e9,
                     alpha_of='all-reduce', source='measured'))
    rows += quoted.drop(columns='source_url').to_dict('records')
    df = pd.DataFrame(rows).set_index('transport')
    d2l.show_table(df, headers={'alpha_us': 'alpha (us)', 'beta_gbs':
                                'beta (GB/s)', 'alpha_of': 'alpha of'},
                   width=79)
    return df
```

## The Fat Tree's Cut

```{.python .input #networks-lib-flow}
%%tab pytorch
#@save
def cross_leaf_flow(ft, beta):
    """Max-flow (bytes/s) from the endpoints under the first edge switch
    to every endpoint under another one, each link carrying beta per
    direction (networkx)."""
    import networkx as nx
    G = nx.DiGraph()
    for u, v in ft['graph'].edges:
        G.add_edge(u, v, capacity=beta)
        G.add_edge(v, u, capacity=beta)
    first = ft['edge_switches'][0]
    for ep, leaf in ft['attach'].items():
        if leaf == first:
            G.add_edge('source', ep)  # no capacity attribute: unlimited
        else:
            G.add_edge(ep, 'sink')
    return nx.maximum_flow_value(G, 'source', 'sink')
```

## The Two-Node Campaign (C-T4)

`parse_perftest` reads the result table of perftest's `ib_write_bw` (with
`--report_gbits`) and `ib_write_lat`; the tests check it on a hand-written
sample of that format (labelled as such), since the box has no RDMA device.
`fit_perftest` turns the campaign's files (one bandwidth and one latency file
per rail) into one alpha-beta row per rail: alpha the median typical one-way
latency of the smallest messages, beta the median average bandwidth of the
messages of 1 MiB and more.

```{.python .input #networks-lib-perftest}
%%tab pytorch
#@save
_PERFTEST_HEADERS = {
    'bw': ('#bytes', '#iterations', 'BW peak[Gb/sec]', 'BW average[Gb/sec]',
           'MsgRate[Mpps]'),
    'lat': ('#bytes', '#iterations', 't_min[usec]', 't_max[usec]',
            't_typical[usec]', 't_avg[usec]', 't_stdev[usec]',
            '99% percentile[usec]', '99.9% percentile[usec]')}

#@save
_PERFTEST_COLUMNS = {
    'bw': ['bytes', 'iterations', 'bw_peak_gbits', 'bw_avg_gbits',
           'msg_rate_mpps'],
    'lat': ['bytes', 'iterations', 't_min_us', 't_max_us', 't_typical_us',
            't_avg_us', 't_stdev_us', 't_p99_us', 't_p999_us']}

#@save
def parse_perftest(text, kind='bw'):
    """The result table of ib_write_bw --report_gbits (kind='bw') or
    ib_write_lat ('lat') as a DataFrame; raises ValueError when the header
    is not the one this parser was written for."""
    lines = text.splitlines()
    head = next((i for i, l in enumerate(lines) if '#bytes' in l), None)
    want = _PERFTEST_HEADERS[kind]
    if head is None or not all(h in lines[head] for h in want):
        raise ValueError(f'not an ib_write_{kind} table of the expected '
                         f'format (header {want})')
    rows = []
    for l in lines[head + 1:]:
        parts = l.split()
        if not parts or not parts[0].isdigit():
            if rows:
                break
            continue
        rows.append([float(p) for p in parts[:len(want)]])
    df = pd.DataFrame(rows, columns=_PERFTEST_COLUMNS[kind])
    df[['bytes', 'iterations']] = df[['bytes', 'iterations']].astype(int)
    return df

#@save
def fit_perftest(log, small=64, large=2**20):
    """alpha (s, one way) and beta (bytes/s) per rail from a campaign log
    holding entry 'perftest' ({rail: {'bw': text, 'lat': text}}), in
    fit_collectives' layout (show_fits prints it); for a pending log an
    empty table whose attrs['status'] is its status line."""
    if isinstance(log, d2l.CampaignPending):
        out = d2l._hw_pending(log.name)
        out.attrs['status'] = d2l.hw_status_line(log)
        return out
    rows = {}
    for rail, files in sorted(log['perftest'].items()):
        bw = parse_perftest(files['bw'], 'bw')
        lat = parse_perftest(files['lat'], 'lat')
        alpha = float(lat.t_typical_us[lat.bytes <= small].median()) / 1e6
        beta = float(bw.bw_avg_gbits[bw.bytes >= large].median()) * 1e9 / 8
        rows[('ib_write', 1, rail)] = dict(
            alpha=alpha, alpha_lo=np.nan, alpha_hi=np.nan, beta=beta,
            beta_lo=np.nan, beta_hi=np.nan, n_half=alpha * beta,
            n_points=len(bw))
    out = pd.DataFrame.from_dict(rows, orient='index')
    out.index = pd.MultiIndex.from_tuples(out.index, names=['op', 'g', 'path'])
    return out
```

## Displays of the Ratios and the Layout Calculator

`state_per_chip` is the memory side of `layout_fit`: the 16 bytes per
parameter of mixed-precision training (`TrainStepModel.state['mixed16']`),
the dense part sharded over tensor and pipeline parallelism, the experts
also over expert parallelism; activations are not counted.

```{.python .input #networks-lib-displays}
%%tab pytorch
#@save
def state_per_chip(cfg, system, layout):
    """The layout_fit attributes of a layout's memory: system and model
    names, the training state per chip (GB, mem_gb) and the chip's HBM
    (GB, hbm_gb)."""
    m = d2l.ServingModel(cfg, d2l.chip_specs().loc[system.chip])
    E, _, d_ff, n_moe = m.experts()
    N = m.n_params()[0]
    N_exp = n_moe * E * 3 * cfg.hidden_size * d_ff if E > 1 else 0
    t, e, p = layout['t'], layout['e'], layout['p']
    state = d2l.TrainStepModel.state['mixed16'] * (
        (N - N_exp) / (t * p) + N_exp / (t * p * e))
    name = getattr(cfg, 'name_or_path', '') or ''
    return dict(system=system.name, short=system.short,
                model=name.split('-', 1)[-1], mem_gb=state / 1e9,
                hbm_gb=float(system.hbm_gb_per_chip))

#@save
def show_ratios(rho):
    """The table of communication-to-compute ratios (rows: axis and model,
    columns: links) at two significant digits."""
    d2l.show_table(rho.map(lambda x: d2l._hw_sig(x, 2)), width=79)

#@save
def _fit_cell(fit, link):
    """The axes of one layout on one link: those that fit, and in
    parentheses those over epsilon; '-' when none lands there, 'n/a' when
    the system's rates are not stated."""
    on = fit[fit.link == link]
    if on.empty:
        return '-'
    if on.ratio.isna().all():
        return 'n/a'
    return ' '.join(a if ok else f'({a})' for a, ok in zip(on.index, on.fit))

#@save
def show_layout_fit(fits):
    """One row per system, three columns per model: the axes on the
    scale-up link and on the scale-out link (in parentheses: over epsilon;
    n/a: a rate the system's sources do not state) and the training state
    per chip (GB); then the layouts whose state exceeds the chip's
    memory."""
    rows, over = {}, []
    for f in fits:
        a = f.attrs
        row = rows.setdefault(a['short'], {})
        row[f"{a['model']} up"] = _fit_cell(f, 'scale-up')
        row[f"{a['model']} out"] = _fit_cell(f, 'scale-out')
        row[f"{a['model']} GB"] = a['mem_gb']
        if a['mem_gb'] > a['hbm_gb']:
            over.append(f"{a['short']} {a['model']} "
                        f"({a['mem_gb']:.0f} > {a['hbm_gb']:.0f} GB)")
    d2l.show_table(pd.DataFrame.from_dict(rows, orient='index'),
                   digits=1, width=79)
    print('state over the chip\'s memory: ' + (', '.join(over) or 'none'))

#@save
def show_epsilon(fits, eps_values):
    """Per epsilon, the verdicts that differ from those at the fits' own
    epsilon: one line each, naming up to two (system, model, axis)."""
    for eps in eps_values:
        flips = [f"{f.attrs['short']} {f.attrs['model']} {axis}"
                 for f in fits for axis, x in f.iterrows()
                 if np.isfinite(x.ratio) and bool(x.ratio <= eps) != x.fit]
        text = '; '.join(flips[:2]) + (f'; +{len(flips) - 2} more'
                                       if len(flips) > 2 else '')
        print(f'eps = {eps:g}: {len(flips)} changed'
              + (f': {text}' if flips else ''))

#@save
_AXIS_COLORS = {'tp': 'C0', 'ep': 'C1', 'pp': 'C2', 'dp': 'C3'}

#@save
def plot_layout_fit(fits):
    """Each axis of each layout as a point: x the ridge P_eff / beta of the
    link it lands on (FLOP per byte), y its ratio after overlap; filled
    circles on the scale-up link, open squares on the scale-out link. A
    ratio is proportional to the ridge, so the points of one axis and
    model lie on one line; the dashed line is epsilon."""
    d2l.set_figsize((6, 3.4))
    _, ax = d2l.plt.subplots()
    for f in fits:
        for axis, x in f.dropna(subset=['ratio']).iterrows():
            c, up = _AXIS_COLORS[axis], x.link == 'scale-up'
            ax.scatter(x.ridge, x.ratio, s=28, edgecolors=c,
                       facecolors=c if up else 'none',
                       marker='o' if up else 's')
    eps = fits[0].attrs['eps']
    ax.axhline(eps, color='black', linestyle='--', linewidth=1,
               label=f'eps = {eps:g}')
    for axis, c in _AXIS_COLORS.items():
        ax.scatter([], [], color=c, label=axis)
    ax.scatter([], [], facecolors='none', edgecolors='black', marker='s',
               label='scale-out')
    ax.set(xscale='log', yscale='log', xlabel='link ridge (FLOP/byte)',
           ylabel='communication / compute')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
```

## Driver

`regen_net` is the regeneration of box-net: the gloo sweeps (CPU) and the
NCCL socket sweep on the first two cards, with both cards' locks held and no
other job on the host (`tools/hw_regen.py box-net` under `gpu_run.sh 0,1`,
three processes). `regen_net_finish` keeps every process's times with the
combined sweeps, from which `fit_collectives` draws its intervals.

```{.python .input #networks-lib-driver}
%%tab pytorch
#@save
def regen_net(config='box'):
    """Driver of box-net: the page's two loopback sweeps."""
    if config != 'box':
        raise ValueError('box-net has no campaign configuration: the '
                         'two-node campaign C-T4 runs perftest')
    if torch.cuda.device_count() < 2:
        raise RuntimeError('box-net needs two GPUs (lock both)')
    floor = loopback_sweep(backend='gloo', ranks=_HW_NET['gloo']['ranks'],
                           max_bytes=_HW_NET['gloo']['max_bytes'])
    sock = loopback_sweep(backend='nccl', ranks=_HW_NET['nccl']['ranks'],
                          env=_HW_NET['nccl']['env'],
                          max_bytes=_HW_NET['nccl']['max_bytes'])
    return floor, sock

#@save
def regen_net_finish(folder):
    """After tools/hw_regen.py combined the processes: every process's
    sweep times kept with each combined sweep (attrs['proc_us'])."""
    for entry in ('gloo', 'nccl'):
        path = os.path.join(folder, f'{entry}.json.gz')
        table, meta = d2l._hw_read(path)
        procs = sorted(glob.glob(os.path.join(folder, 'proc*',
                                              f'{entry}.json.gz')))
        table.attrs['proc_us'] = [d2l._hw_read(p)[0].oop_us.tolist()
                                  for p in procs]
        d2l._hw_write(path, table, meta)
```
