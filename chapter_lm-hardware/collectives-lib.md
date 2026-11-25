# Collective Communication at Scale: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_collectives` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1): the
section's registry entries, one statement each (the box's forced algorithm
and protocol sweeps; 22.7 also reads the C-T4 log registered by 22.6; the
`DATA_HUB` entry follows the upload) and its regeneration job; the
configurations; the simulator's topology; NCCL's own account of a run
(`nccl_info`); the forced-pair sweeps with their drop reasons
(`nccl_algo_proto_sweeps`), their plot and fits (`plot_algo_crossover`,
`fit_pairs`); the tuner check (`tuner_regret`); the opening's two symptoms
(`coll_symptoms`); the hierarchical benchmark and its torchrun script
(`hier_vs_flat`); and the driver `regen_coll`. Blocks are tagged
`%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_collectives`

## Registries

```{.python .input #collectives-lib-entries}
%%tab pytorch
#@save
d2l.HW_ARTIFACTS['box-coll'] = ('lm-hw-buildbox-coll-v1', 'tar')
```

```{.python .input #collectives-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-coll'] = (
    'regen_coll', lambda: dict(config='box'))
```

## Configurations

The box configuration is the page's calls on the full box: all-reduce on all
four cards from 8 B to 256 MiB in factors of 2 (5 warm-up and 20 timed
iterations), once with NCCL's own choice and once for every forced pair of
`NCCL_ALGO` in {Ring, Tree} and `NCCL_PROTO` in {LL, LL128, Simple}, each
with a timeout; the hierarchical benchmark on the 2 x 2 split (the NCCL pairs
0-1 and 2-3, each behind one host bridge on the box, the gloo pairs 0-2 and
1-3 across the bridges) at 1 MiB to 256 MiB. The live checks re-time three
pairs and one hierarchical size. Campaign C-T4 (two 8 x A100 nodes with
RDMA, shared with :numref:`sec_lm_networks`) runs the same pairs plus NVLS
and CollNet where the fabric has them, on 16 ranks to 4 GiB; it needs a
multi-node launcher (one rank per GPU under `srun` or `mpirun`), so its
configuration is here and its launch is the campaign package's.

```{.python .input #collectives-lib-configs}
%%tab pytorch
#@save
_HW_COLL = {
    'box': dict(
        g=4, pairs=[('default', 'default'), ('Ring', 'LL'),
                    ('Ring', 'LL128'), ('Ring', 'Simple'), ('Tree', 'LL'),
                    ('Tree', 'LL128'), ('Tree', 'Simple')],
        min_bytes=8, max_bytes=2**28, iters=20, warmup=5, timeout_s=120,
        alpha_below=2**16, plateau_from=2**22,
        intra=[[0, 1], [2, 3]], inter=[[0, 2], [1, 3]],
        hier_sizes=[2**20, 2**22, 2**24, 2**26, 2**28], hier_iters=10,
        hier_warmup=3, hier_timeout_s=600, check_bytes=2**20),
    't4': dict(
        g=16, pairs=[('default', 'default')] + [
            (a, p) for a in ('Ring', 'Tree', 'NVLS', 'CollNet')
            for p in ('LL', 'LL128', 'Simple')],
        min_bytes=8, max_bytes=2**32, iters=50, warmup=10, timeout_s=900,
        alpha_below=2**16, plateau_from=2**28, launcher='campaign')}

#@save
_HW_COLL_LIVE = dict(pairs=[('default', 'default'), ('Ring', 'LL'),
                            ('Ring', 'Simple')],
                     sizes=(2**12, 2**20, 2**28), hier_size=2**26,
                     timeout_s=120)

#@save
def _pair_name(algo, proto):
    """'default' or 'Ring/LL'."""
    return 'default' if algo == 'default' else f'{algo}/{proto}'
```

## The Simulator's Topology

```{.python .input #collectives-lib-topology}
%%tab pytorch
#@save
def sim_topology(G, alpha, beta):
    """The network sim_collective runs on: G ranks, each with one
    full-duplex port of beta bytes/s per direction to a non-blocking
    switch, and alpha seconds of latency per message."""
    if G < 2 or alpha < 0 or beta <= 0:
        raise ValueError('need G >= 2, alpha >= 0 and beta > 0')
    return dict(G=int(G), alpha=float(alpha), beta=float(beta))
```

## NCCL's Account of a Run

`NCCL_DEBUG=INFO` makes NCCL describe a communicator as it builds it: its
version, the transport between each pair of ranks, the channels (each with
its own ring order and tree), and, with the `TUNING` subsystem, the tuner's
table of latency and bandwidth per (algorithm, protocol) and the pair it
picks for each call. The lines go to a file (`NCCL_DEBUG_FILE`), never into
the benchmark's table; `_parse_info` digests them, `_show_info` prints the
digest in at most eight lines.

```{.python .input #collectives-lib-info}
%%tab pytorch
#@save
def _info_run(gpus, env='', args=(), timeout_s=120):
    """(HWRun, INFO text) of one nccl-tests all_reduce_perf run on this
    process's GPUs `gpus` with NCCL_DEBUG=INFO written to files (not into
    the table), the variables of env added; the binary's arguments
    `args`. A refusal is returned, never raised. Without the binary,
    (None, '')."""
    exe = d2l.hw_bin('all_reduce_perf')
    if exe is None:
        return None, ''
    with tempfile.TemporaryDirectory() as tmp:
        child = dict(d2l._gpu_env(gpus, env), NCCL_DEBUG='INFO',
                     NCCL_DEBUG_SUBSYS='INIT,ENV,GRAPH,TUNING',
                     NCCL_DEBUG_FILE=os.path.join(tmp, 'nccl.%p.log'))
        run = d2l.hw_run([exe, '-g', len(gpus), *args], timeout_s=timeout_s,
                         env=d2l._nccl_tests_env(child), record=True)
        text = ''
        for log in sorted(glob.glob(os.path.join(tmp, 'nccl.*.log'))):
            with open(log, errors='replace') as f:
                text += f.read()
    lines = [d2l._hw_scrub(l[l.index('NCCL '):]) for l in text.splitlines()
             if 'NCCL ' in l]  # no host:pid:tid prefix
    return run, '\n'.join(lines)

#@save
_NCCL_ALGOS = ('Tree', 'Ring', 'CollNetDirect', 'CollNetChain', 'NVLS',
               'NVLSTree', 'PAT')  # NCCL's enum order

#@save
_NCCL_PROTOS = ('LL', 'LL128', 'Simple')

#@save
def _nccl_name(word, names):
    """NCCL's name of an algorithm or protocol printed as a name in any
    case ('RING') or as its enum number ('1')."""
    if word.isdigit():
        return names[int(word)] if int(word) < len(names) else word
    return next((n for n in names if n.lower() == word.lower()), word)

#@save
def _tuner_table(text, func='AllReduce'):
    """The tuner's latency (us) and bandwidth (GB/s) of func per
    (algorithm, protocol) from the INFO lines of NCCL_DEBUG_SUBSYS=TUNING
    (rows of algorithms, a row of protocols under the first, a row of
    'latency/bandwidth' cells per function under each row of algorithms):
    {('Ring', 'LL'): (lat, bw), ...}; {} when the table is absent."""
    algos, protos, out = [], [], {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split('|')]
        head = cells[0].split()[-1] if cells[0].split() else ''
        words = [w for c in cells[1:] for w in c.split()]
        if head == 'Algorithm':
            algos = [_nccl_name(w, _NCCL_ALGOS) for w in words]
        elif head == 'Protocol':
            protos = [_nccl_name(w, _NCCL_PROTOS) for w in words]
        elif head == func and algos and protos:
            per = len(set(protos))  # protocols per algorithm
            nums = re.findall(r'(-?[\d.]+)\s*/\s*(-?[\d.]+)',
                              '|'.join(cells[1:]))
            for i, (lat, bw) in enumerate(nums[:per * len(algos)]):
                out[(algos[i // per], protos[i % len(protos)])] = (
                    float(lat), float(bw))
    return out

#@save
def _choices(text, func='AllReduce'):
    """{bytes: 'Algo/proto'} that NCCL reported choosing for func, from
    INFO lines '<func>: <n> Bytes -> Algo <a> proto <p> ...' (names or
    enum numbers; {} when NCCL does not print them)."""
    out = {}
    pat = re.compile(rf'{func}: (\d+) Bytes -> Algo (\w+) proto (\w+)')
    for m in pat.finditer(text):
        out[int(m.group(1))] = (f'{_nccl_name(m.group(2), _NCCL_ALGOS)}/'
                                f'{_nccl_name(m.group(3), _NCCL_PROTOS)}')
    return out

#@save
def _parse_info(text):
    """A digest of NCCL's INFO lines: version, ranks and nodes, the
    transports between ranks, the number of channels, each channel's ring
    order, the trees (parent of each rank per channel), the tuner's
    all-reduce table and the choice per message size."""
    ver = re.search(r'NCCL version (\S+)', text)
    comm = re.search(r'nRanks (\d+) nNodes (\d+)', text)
    via = sorted({m.group(1) for m in re.finditer(
        r'\] -> \d+\[\d+\] (?:\[\w+\] )?via (\S+)', text)})
    rings = {}
    for m in re.finditer(r'Channel (\d+)/(\d+) :((?:[ \t]+\d+)+)[ \t]*$',
                         text, re.M):
        rings[int(m.group(1))] = [int(x) for x in m.group(3).split()]
    nchan = re.search(r'(\d+) coll channels', text)
    trees = {}  # channel -> {rank: parent}
    for m in re.finditer(r'Trees((?: \[\d+\] [-\d/]+->\d+->-?\d+)+)', text):
        for c, rank, up in re.findall(r'\[(\d+)\] [-\d/]+->(\d+)->(-?\d+)',
                                      m.group(1)):
            trees.setdefault(int(c), {})[int(rank)] = int(up)
    return dict(version=ver.group(1).split('+')[0] if ver else None,
                n_ranks=int(comm.group(1)) if comm else None,
                n_nodes=int(comm.group(2)) if comm else None,
                transports=via, rings=rings,
                channels=int(nchan.group(1)) if nchan else len(rings),
                trees=trees, tuner=_tuner_table(text), choices=_choices(text))

#@save
def _chain(parent):
    """'3 -> 2 -> 1 -> 0 (a chain)' when the tree {rank: parent, -1 at the
    root} is a chain, else its edges."""
    kids = collections.Counter(p for p in parent.values() if p >= 0)
    leaves = [r for r in parent if r not in kids]
    if len(leaves) == 1 and max(kids.values(), default=0) <= 1:
        path, r = [], leaves[0]
        while r >= 0 and len(path) <= len(parent):
            path.append(r)
            r = parent.get(r, -1)
        return ' -> '.join(map(str, path)) + ' (a chain to the root)'
    return ', '.join(f'{r} -> {p}' for r, p in sorted(parent.items()))

#@save
def _ranges(choices):
    """'Ring/LL up to 64 KiB, Ring/Simple from 128 KiB' from {bytes: pair}."""
    runs = []
    for size in sorted(choices):
        if runs and runs[-1][0] == choices[size]:
            runs[-1][2] = size
        else:
            runs.append([choices[size], size, size])
    if len(runs) == 1:
        return f'{runs[0][0]} at every size'
    parts = []
    for i, (pair, lo, hi) in enumerate(runs):
        if i == 0:
            parts.append(f'{pair} up to {d2l._hw_bytes(hi)}')
        elif i == len(runs) - 1:
            parts.append(f'{pair} from {d2l._hw_bytes(lo)}')
        else:
            parts.append(f'{pair} {d2l._hw_bytes(lo)}-{d2l._hw_bytes(hi)}')
    return ', '.join(parts)

#@save
def _show_info(d, width=79):
    """At most eight lines: version and ranks, transports, channels and
    ring order, the distinct trees (two at most), the tuner's all-reduce
    model of each pair it may use and the pairs it disabled, the choice
    per size."""
    import textwrap
    if not isinstance(d, dict) or not d.get('version'):
        label = (d.get('label') if isinstance(d, dict) else
                 getattr(d, 'attrs', {}).get('label'))
        print(label or 'NCCL INFO not recorded')
        return
    say = lambda s: print(textwrap.fill(s, width, subsequent_indent='  '))
    say(f"NCCL {d['version']}: {d['n_ranks']} ranks on {d['n_nodes']} "
        f"node{'s' if (d['n_nodes'] or 1) > 1 else ''}; transport "
        f"between ranks: {', '.join(d['transports']) or 'not reported'}")
    if d['rings']:
        orders = sorted({' '.join(map(str, r)) for r in d['rings'].values()})
        say(f"{d['channels']} channels; ring order {'; '.join(orders)}")
    shown = []
    for c in sorted(d['trees'], key=int):
        text = _chain({int(k): v for k, v in d['trees'][c].items()})
        if text not in shown and len(shown) < 2:
            shown.append(text)
            say(f'tree of channel {c}: {text}')
    tab = {tuple(k.split('/')) if isinstance(k, str) else tuple(k): v
           for k, v in d['tuner'].items()}
    tab = {k: v for k, v in tab.items() if k[0] in ('Ring', 'Tree')}
    if tab:
        on = [f'{a}/{p} {lat:.0f} us + S / {bw:.1f} GB/s'
              for (a, p), (lat, bw) in sorted(tab.items()) if bw > 0]
        off = [f'{a}/{p}' for (a, p), (_, bw) in sorted(tab.items())
               if bw <= 0]
        say('tuner model of an all-reduce: ' + ', '.join(on)
            + (f"; disabled: {', '.join(off)}" if off else ''))
    if d['choices']:
        chosen = {int(k): v for k, v in d['choices'].items()}
        say(f"tuner's choice: {_ranges(chosen)}")

#@save
def _info_compute(gpus, max_bytes=2**28):
    """NCCL's digest of all-reduces on gpus from 8 B to max_bytes (factor
    4, one timed iteration each)."""
    run, text = _info_run(gpus, args=('-b', 8, '-e', max_bytes, '-f', 4,
                                      '-w', 1, '-n', 1, '-c', 0))
    if run is None:
        return dict(version=None, label='no nccl-tests binary: NCCL INFO '
                    'not recorded')
    if run.refused:
        raise RuntimeError(f'NCCL INFO run: {run.refused}')
    d = _parse_info(text)
    d['tuner'] = {f'{a}/{p}': v for (a, p), v in d['tuner'].items()}
    return d

#@save
def nccl_info(gpus=(0, 1, 2, 3)):
    """What NCCL builds on this machine's GPUs `gpus` for an all-reduce
    (NCCL_DEBUG=INFO, digested in at most eight lines): live with two or
    more GPUs (it depends only on the machine and the software), else the
    committed box header. Committed as box-coll's entry 'info'."""
    gpus = tuple(gpus)
    if 'box-coll' in d2l._hw_regenerating():
        return d2l._hw_result('box-coll', lambda: _info_compute(gpus),
                              dict(gpus=list(gpus)), entry='info',
                              show=_show_info)
    if torch.cuda.device_count() >= 2 and len(gpus) >= 2:
        with d2l._quiet():
            d = _info_compute(gpus)
        _show_info(d)
        return d
    print('needs two GPUs: the committed box header is shown')
    d = d2l.hw_box_log('box-coll', 'info')
    _show_info(d)
    return d
```

## Forced Pairs

A forced pair is the all-reduce sweep with `NCCL_ALGO` and `NCCL_PROTO` set.
NCCL may refuse a pair (an error), hang, return wrong results or quietly run
another pair, so each pair first runs a short check with `NCCL_DEBUG=INFO`
and nccl-tests' result check (`-c 1`, one iteration per size, factor 4):
a refusal, a timeout, a wrong result or an INFO choice other than the forced
pair drops the pair with its reason ('refused', 'timed out', 'wrong
results', 'ignored: ran Ring/Simple'). Then the timed sweep runs without
INFO (its lines would cost host time inside the timed loop). The long table
has every pair at every size (NaN times for a dropped pair), the pair in
`algo` (`'default'`, `'Ring/LL'`, ...), its protocol in `proto`, the drop
reason in `status` (empty for a kept pair) and what NCCL reported choosing
at each size in `used`, so that `fit_collectives` fits any one pair.

```{.python .input #collectives-lib-pairs}
%%tab pytorch
#@save
def _pair_env(algo, proto):
    return '' if algo == 'default' else f'NCCL_ALGO={algo} NCCL_PROTO={proto}'

#@save
def _sizes(lo, hi, factor=2):
    n = int(round(math.log(hi / lo, factor)))
    return [int(lo * factor**i) for i in range(n + 1)]

#@save
def _pair_check(gpus, algo, proto, spec):
    """(reason or '', {bytes: pair NCCL ran}) of one forced pair: a short
    run with NCCL_DEBUG=INFO and the result check."""
    run, text = _info_run(gpus, _pair_env(algo, proto),
                          ('-b', spec['min_bytes'], '-e', spec['max_bytes'],
                           '-f', 4, '-w', 1, '-n', 1, '-c', 1),
                          spec['timeout_s'])
    if run is None:  # no nccl-tests: nothing to check against
        return '', {}
    if run.refused:
        return ('timed out' if run.refused.startswith('timeout')
                else 'refused'), {}
    table = d2l.parse_nccl_tests(run.stdout)
    if (table[['oop_wrong', 'ip_wrong']].fillna(0) > 0).any().any():
        return 'wrong results', {}
    used = _choices(text)
    want = _pair_name(algo, proto).lower()
    other = sorted({u for u in used.values()
                    if algo != 'default' and u.lower() != want})
    if other:
        return f"ignored: ran {', '.join(other)}", used
    return '', used

#@save
def _pair_sweep(gpus, algo, proto, spec, timeout_s=None, sizes=None):
    """The timed sweep of one pair (factor 2 between sizes): nccl-tests'
    table (the torch.distributed fallback without the binary); attrs
    refused."""
    sizes = sizes or (spec['min_bytes'], spec['max_bytes'])
    env = _pair_env(algo, proto)
    exe = d2l.hw_bin('all_reduce_perf')
    timeout_s = timeout_s or spec['timeout_s']
    if exe is not None:
        run = d2l.hw_run([exe, '-b', sizes[0], '-e', sizes[-1], '-f', 2,
                          '-g', len(gpus), '-w', spec['warmup'], '-n',
                          spec['iters'], '-c', 0], timeout_s=timeout_s,
                         env=d2l._nccl_tests_env(d2l._gpu_env(gpus, env)),
                         record=True)
        df = (d2l.parse_nccl_tests(run.stdout) if not run.refused else
              pd.DataFrame(columns=list(d2l._NCCL_COLUMNS)))
    else:
        rows, run = d2l._torchrun_sweep(
            'all_reduce', len(gpus), _sizes(sizes[0], sizes[-1]),
            spec['iters'], spec['warmup'], d2l._gpu_env(gpus, env),
            timeout_s)
        df = d2l._rows_from_times(rows, 'all_reduce', len(gpus))
    df.attrs['refused'] = run.refused
    return df

#@save
def _pairs_compute(config):
    spec, gpus = _HW_COLL[config], tuple(range(_HW_COLL[config]['g']))
    sizes = _sizes(spec['min_bytes'], spec['max_bytes'])
    frames, dropped = [], {}
    for algo, proto in spec['pairs']:
        name = _pair_name(algo, proto)
        reason, used = _pair_check(gpus, algo, proto, spec)
        df = None
        if not reason:
            df = _pair_sweep(gpus, algo, proto, spec)
            if df.attrs['refused']:
                reason = ('timed out' if df.attrs['refused'].startswith(
                    'timeout') else 'refused')
        if reason:
            dropped[name] = reason
            df = pd.DataFrame({'size': sizes})
        df = df.set_index('size').reindex(sizes).reset_index()
        df['used'] = [used.get(s, '') for s in sizes]
        frames.append(df.assign(op='all_reduce', g=len(gpus), path='all',
                                algo=name, proto=proto, env='',
                                status=reason))
    out = pd.concat(frames, ignore_index=True)
    out.attrs.update(dropped=dropped, alpha_below=spec['alpha_below'],
                     plateau_from=spec['plateau_from'],
                     nccl='.'.join(map(str, torch.cuda.nccl.version())),
                     tool='nccl-tests' if d2l.hw_bin('all_reduce_perf')
                     else 'torch.distributed')
    return out

#@save
def _show_pairs(r):
    """The lines naming the dropped pairs (at most two, by reason)."""
    if r.empty:
        print(r.attrs.get('label', 'pending'))
        return
    by = collections.defaultdict(list)
    for name, reason in r.attrs.get('dropped', {}).items():
        by[reason].append(name)
    for reason, names in sorted(by.items())[:2]:
        print(f"dropped ({reason}): {', '.join(names)}")

#@save
def _pairs_live(r, same):
    """Re-time the full sweeps of the default, Ring/LL and Ring/Simple on
    all cards and compare three sizes (with two or three cards: g = 2, then
    each pair's bus bandwidth at the largest size, uncompared): one
    line."""
    n = torch.cuda.device_count()
    if n < 2:
        return 'needs two GPUs: the committed box run is shown'
    spec = _HW_COLL['box']
    g = spec['g'] if n >= spec['g'] else 2
    compare = same and g == int(r.g.iloc[0])
    pts, busbw, top = [], [], _HW_COLL_LIVE['sizes'][-1]
    for algo, proto in _HW_COLL_LIVE['pairs']:
        name = _pair_name(algo, proto)
        if name in r.attrs.get('dropped', {}):
            continue
        df = _pair_sweep(tuple(range(g)), algo, proto, spec,
                         _HW_COLL_LIVE['timeout_s'])  # the full sweep
        if df.attrs['refused']:
            return f"{name}: {df.attrs['refused']}; committed run shown"
        t = lambda s: float(df.oop_us[df['size'] == s].iloc[0])
        phi = d2l.nccl_busbw_factor('all_reduce', g)
        busbw.append(f'{name} {phi * top / t(top) / 1e3:.1f}')  # GB/s
        for s in _HW_COLL_LIVE['sizes']:
            i = d2l._row(r, algo=name, size=s)
            pts.append((f'{name} {d2l._hw_bytes(s)}', float(r.oop_us.iloc[i]),
                        t(s), d2l._spread(r, 'oop_us', i, 0.05)))
    if not compare:
        return (f'{g} GPUs, busbw at {d2l._hw_bytes(top)} (GB/s):\n  '
                + ', '.join(busbw))
    return d2l._hw_agreement(pts, same, 'us')

#@save
def nccl_algo_proto_sweeps(config='box'):
    """The all-reduce on all cards of the box with NCCL's default choice
    and with every forced pair of NCCL_ALGO x NCCL_PROTO (the configuration
    _HW_COLL[config]): one long table (algo holds the pair, status the
    reason a pair was dropped), committed in box-coll. A normal run prints
    the dropped pairs and re-times three pairs."""
    spec = _HW_COLL[config]
    if spec.get('launcher') == 'campaign':
        raise ValueError(f'{config} runs on several nodes through the '
                         'campaign launcher, not here')
    config_doc = dict(config=config, g=spec['g'],
                      pairs=[list(p) for p in spec['pairs']],
                      min_bytes=spec['min_bytes'], max_bytes=spec['max_bytes'],
                      iters=spec['iters'], warmup=spec['warmup'])
    return d2l._hw_result('box-coll', lambda: _pairs_compute(config),
                          config_doc, entry='pairs', live_check=_pairs_live,
                          show=_show_pairs, live_label='three pairs')
```

## The Crossover Plot, the Fits and the Tuner

```{.python .input #collectives-lib-tuner}
%%tab pytorch
#@save
def _kept(pairs):
    """{pair: (sizes, times in s)} of the pairs that were not dropped."""
    out = {}
    for name, x in pairs[pairs.status == ''].groupby('algo', sort=False):
        x = x[x['size'] > 0].dropna(subset=['oop_us'])
        out[name] = (x['size'].to_numpy(float), x.oop_us.to_numpy() / 1e6)
    return out

#@save
def _crossing(s, t1, t2):
    """The size where t1 - t2 changes sign from negative (t1 faster at
    small sizes) to positive, by interpolation of log time over log size;
    NaN when it does not."""
    d = np.log(t1) - np.log(t2)
    for i in range(len(d) - 1):
        if d[i] < 0 <= d[i + 1]:
            x = np.log(s[i]) + (np.log(s[i + 1]) - np.log(s[i])) * (
                -d[i] / (d[i + 1] - d[i]))
            return float(np.exp(x))
    return float('nan')

#@save
def pair_time(pairs, pair, size):
    """Seconds of one all-reduce of size bytes with pair ('default',
    'Ring/LL', ...) in a forced-pair table."""
    return float(pairs.oop_us.iloc[d2l._row(pairs, algo=pair, size=size)]
                 ) / 1e6

#@save
def plot_algo_crossover(pairs, small='Ring/LL', large='Ring/Simple'):
    """Time of one all-reduce of each kept forced pair relative to NCCL's
    default (bold black at 1) against message size; a dotted line at the
    measured crossover of small and large."""
    kept = _kept(pairs)
    if 'default' not in kept:
        print(pairs.attrs.get('label', 'no forced pairs'))
        return
    s, base = kept['default']
    names = [n for n in kept if n != 'default']
    rel = [np.interp(s, *kept[n]) / base for n in names]
    d2l.plot([s] * (len(names) + 1), rel + [np.ones_like(s)],
             xlabel='message size (bytes)',
             ylabel="time / NCCL default's", legend=names + ['default'],
             xscale='log', figsize=(5, 3), legend_loc='outside',
             fmts=[f'C{i}-' for i in range(len(names))] + ['k-'])
    ax = d2l.plt.gca()
    for line in ax.get_lines():
        line.set_linewidth(1.0)
    ax.get_lines()[len(names)].set_linewidth(2.5)
    if small in kept and large in kept:
        x = _crossing(kept[small][0], kept[small][1], kept[large][1])
        if np.isfinite(x):
            ax.axvline(x, color='k', linestyle=':', linewidth=1)

#@save
def fit_pairs(pairs, small='Ring/LL', large='Ring/Simple'):
    """alpha (s) and beta (bytes/s, bus bandwidth) per kept pair of a
    forced-pair table (fit_collectives' two regimes, 95% intervals over
    the regeneration's processes), indexed by (algo, proto); attrs
    'crossover': the measured and the predicted size at which large
    overtakes small (t = alpha + phi S / beta of each fit). A campaign log
    gives its 'pairs' entry's fits, or while it is pending an empty table
    whose attrs['status'] is its status line."""
    if isinstance(pairs, d2l.CampaignPending):
        return d2l.fit_collectives(pairs)
    if isinstance(pairs, dict):
        pairs = pairs['pairs']
    if pairs.empty:
        return d2l._hw_pending('box-coll')
    rows, kept = [], _kept(pairs)
    for name in kept:
        ab = d2l.fit_collectives(pairs, algo=name)
        r = ab.iloc[0].to_dict()
        r['algo'], r['proto'] = (('default', 'default') if name == 'default'
                                 else name.split('/'))
        rows.append(r)
    out = pd.DataFrame(rows).set_index(['algo', 'proto'])
    out.attrs.update(n_proc=ab.attrs.get('n_proc', 1),
                     g=int(pairs.g.iloc[0]))
    if small in kept and large in kept:
        phi = d2l.nccl_busbw_factor('all_reduce', out.attrs['g'])
        a, b = (out.loc[tuple(n.split('/'))] for n in (small, large))
        pred = (b.alpha - a.alpha) / (phi * (1 / a.beta - 1 / b.beta))
        s, t1 = kept[small]
        out.attrs['crossover'] = dict(
            small=small, large=large,
            measured=_crossing(s, t1, kept[large][1]), predicted=float(pred))
    return out

#@save
def show_pair_fits(fits):
    """At most seven rows: algorithm, protocol, alpha (us), beta (GB/s)
    with its 95% interval; one line with the measured and the predicted
    crossover. For a pending log its status line."""
    if fits.empty:
        print(fits.attrs.get('status') or fits.attrs.get('label', 'no fits'))
        return
    ci = lambda lo, hi: ('-' if np.isnan(lo) else
                         f'{lo / 1e9:.2f}-{hi / 1e9:.2f}')
    d2l.show_table(pd.DataFrame({
        'algo': [i[0] for i in fits.index],
        'proto': [i[1] for i in fits.index],
        'alpha (us)': fits.alpha.to_numpy() * 1e6,
        'beta (GB/s)': fits.beta.to_numpy() / 1e9,
        '95% interval': [ci(lo, hi) for lo, hi in zip(fits.beta_lo,
                                                      fits.beta_hi)]}),
        digits={'alpha (us)': 0, 'beta (GB/s)': 2}, max_rows=7, width=79)
    x = fits.attrs.get('crossover')
    if x:
        size = lambda b: _approx_bytes(b) if np.isfinite(b) else 'none'
        print(f"{x['large']} overtakes {x['small']} at {size(x['measured'])}"
              f" measured, {size(x['predicted'])} predicted by the fits")

#@save
def _approx_bytes(b):
    """73159.2 -> '71 KiB': two significant digits."""
    for unit, size in (('GiB', 2**30), ('MiB', 2**20), ('KiB', 2**10)):
        if b >= size:
            return f'{d2l._hw_sig(b / size)} {unit}'
    return f'{d2l._hw_sig(b)} B'

#@save
def _proc_times(pairs, name):
    """(sizes, per-process times in s, shape processes x sizes) of one
    pair; one row when the table has no per-process times."""
    hit = ((pairs.algo == name) & (pairs['size'] > 0)).to_numpy()
    procs = pairs.attrs.get('proc_us') or [pairs.oop_us.tolist()]
    t = np.asarray(procs, dtype=float)[:, hit] / 1e6
    return pairs['size'].to_numpy()[hit], t

#@save
def tuner_regret(pairs, k=3, floor=0.02):
    """Per size: NCCL's default time, the best kept forced pair and its
    time, the excess (default / best - 1) and that excess in noise units
    (the larger median absolute deviation between the regeneration's
    processes of the two times, at least floor of the time). Prints the
    largest excess and how many sizes are within k noise units."""
    kept = _kept(pairs)
    names = [n for n in kept if n != 'default']
    if 'default' not in kept or not names:
        print(pairs.attrs.get('label', 'no default or no forced pair'))
        return pd.DataFrame()
    s, t_def = _proc_times(pairs, 'default')
    forced = {n: _proc_times(pairs, n)[1] for n in names}
    med = lambda t: np.median(t, axis=0)
    mad = lambda t: np.median(np.abs(t - med(t)), axis=0)
    best = [min(names, key=lambda n: med(forced[n])[i]) for i in range(len(s))]
    t_best = np.array([med(forced[b])[i] for i, b in enumerate(best)])
    noise = np.maximum.reduce([mad(t_def), np.array(
        [mad(forced[b])[i] for i, b in enumerate(best)]), floor * t_best])
    out = pd.DataFrame(dict(size=s, default_s=med(t_def), best=best,
                            best_s=t_best))
    out['excess'] = out.default_s / out.best_s - 1
    out['units'] = (out.default_s - out.best_s) / noise
    i = int(out.units.idxmax())
    print(f'largest excess of the default: {out.excess[i]:.0%} '
          f'({out.units[i]:.1f} units) at '
          f'{d2l._hw_bytes(int(out["size"][i]))} over {out.best[i]}')
    n_proc = len(forced[names[0]])
    print(f'within {k} units at {int((out.units <= k).sum())} of '
          f'{len(out)} sizes (unit: MAD of {n_proc} run'
          f'{"s" if n_proc > 1 else ""}, at least {floor:.0%})')
    return out

#@save
def coll_symptoms(log, ab, small=2**12, large=2**28):
    """The opening's two lines from the committed box-coll result (log:
    hw_box_log('box-coll')) and sec_lm_server_paths' all-reduce fit ab
    (box_alpha_beta()): a small all-reduce against that alpha-beta line;
    the default's bus bandwidth at a large size against the best forced
    pair's."""
    pairs = log.get('pairs') if isinstance(log, dict) else log
    if pairs is None or pairs.empty:
        print(getattr(pairs, 'attrs', {}).get('label', 'box-coll pending'))
        return
    g = int(pairs.g.iloc[0])
    phi = d2l.nccl_busbw_factor('all_reduce', g)
    t = pair_time(pairs, 'default', small)
    line = ab['alpha'] + phi * small / ab['beta']
    print(f'{d2l._hw_bytes(small)} all-reduce on {g} GPUs: {t * 1e6:.0f} us '
          f'measured, {line * 1e6:.0f} us on the alpha-beta line')
    busbw = lambda n: phi * large / pair_time(pairs, n, large) / 1e9
    best = max((n for n in _kept(pairs) if n != 'default'), key=busbw)
    print(f"{d2l._hw_bytes(large)}: default {busbw('default'):.2f} GB/s "
          f'busbw, best forced pair {best} {busbw(best):.2f} GB/s')
```

## The Hierarchical Benchmark

`hier_vs_flat` runs a `torchrun` script in four processes, one GPU each: the
world group on NCCL, the intra groups (pairs 0-1 and 2-3) on NCCL, the inter
groups (pairs 0-2 and 1-3) and a second world group on gloo over the
loopback interface. At each size it times, as the median of per-iteration
maxima over the ranks, the flat all-reduce on the gloo world (the CUDA
tensor through a pinned host copy, as the page's `hier_allreduce` does for
its gloo stage), the flat all-reduce on the NCCL world, the page's
`hier_allreduce` (its source is written next to the script, so the benchmark
runs the page's function), and its two stages alone: the NCCL pairs on S
bytes and the gloo pairs on S / 2. It checks `hier_allreduce` once against
a CPU sum of every rank's input (`rtol=1e-5`). The same script in `cpu` mode
runs four gloo processes on the CPU (every group gloo), the check alone: the
T1 path on a machine without four GPUs, labelled not a datacenter network.
The predictions use the committed fits of the two earlier sections: the NCCL
pair and the four-card all-reduce of :numref:`sec_lm_server_paths`, the
gloo loopback fits of :numref:`sec_lm_networks`.

```{.python .input #collectives-lib-hier}
%%tab pytorch
#@save
_HIER_SCRIPT = '''
import json, os, sys, time, torch, torch.distributed as dist
mode, sizes, iters, warmup, check = (sys.argv[1], json.loads(sys.argv[2]),
                                     int(sys.argv[3]), int(sys.argv[4]),
                                     int(sys.argv[5]))
intra_ranks, inter_ranks = json.loads(sys.argv[6]), json.loads(sys.argv[7])
exec(open(sys.argv[8]).read())  # hier_allreduce, as on the page
rank, k = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
if mode == 'cuda':
    dev = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(dev)
    dist.init_process_group('nccl', device_id=dev)
    sync = torch.cuda.synchronize
else:
    dev = torch.device('cpu')
    dist.init_process_group('gloo')
    sync = lambda: None
intra, _ = dist.new_subgroups_by_enumeration(intra_ranks)
inter, _ = dist.new_subgroups_by_enumeration(inter_ranks, backend='gloo')
gloo_world = dist.new_group(backend='gloo')
def via_host(x, group):
    if x.is_cuda:
        h = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        h.copy_(x)
        dist.all_reduce(h, group=group)
        x.copy_(h)
    else:
        dist.all_reduce(x, group=group)
g = len(intra_ranks[0])
variants = {
    'flat gloo': lambda x: via_host(x, gloo_world),
    'flat NCCL': lambda x: dist.all_reduce(x),
    'hierarchical': lambda x: hier_allreduce(x, intra, inter),
    'intra stage': lambda x: dist.all_reduce(x, group=intra),
    'inter stage': lambda x: via_host(x[:x.numel() // g], inter)}
if check:
    n = check // 4
    gen = lambda r: torch.Generator().manual_seed(r)
    ref = sum(torch.randn(n, generator=gen(r)) for r in range(k))
    x = torch.randn(n, generator=gen(rank)).to(dev)
    hier_allreduce(x, intra, inter)
    sync()
    err = float(((x.cpu() - ref).abs() / ref.abs().max()).max())
    ok = bool(torch.allclose(x.cpu(), ref, rtol=1e-5, atol=1e-6))
    if rank == 0:
        print(json.dumps(dict(check=check, max_rel_err=err, ok=ok)),
              flush=True)
for size in sizes:
    for name, op in variants.items():
        x = torch.ones(size // 4, device=dev)
        for _ in range(warmup):
            op(x)
        ts = []
        for _ in range(iters):
            sync()
            dist.barrier()
            t = time.perf_counter()
            op(x)
            sync()
            ts.append(time.perf_counter() - t)
        ts = torch.tensor(ts, device=dev)
        dist.all_reduce(ts, op=dist.ReduceOp.MAX)
        ts = ts.cpu()
        med = ts.median()
        if rank == 0:
            print(json.dumps(dict(variant=name, size=size,
                                  us=float(med) * 1e6,
                                  mad_us=float((ts - med).abs().median())
                                  * 1e6)), flush=True)
dist.destroy_process_group()
'''

#@save
def _hier_run(mode, sizes, iters, warmup, check, intra, inter, timeout_s,
              gpus=(0, 1, 2, 3)):
    """(timing rows, check record, HWRun) of _HIER_SCRIPT in four torchrun
    processes ('cuda': one GPU each; 'cpu': gloo on the CPU)."""
    import inspect
    with tempfile.TemporaryDirectory() as tmp:
        path, fn = (os.path.join(tmp, f) for f in ('hier.py', 'fn.py'))
        with open(path, 'w') as f:
            f.write(_HIER_SCRIPT)
        with open(fn, 'w') as f:
            f.write(inspect.getsource(d2l.hier_allreduce))
        env = dict(GLOO_SOCKET_IFNAME='lo', OMP_NUM_THREADS='4',
                   NCCL_DEBUG='WARN')
        if mode == 'cuda':
            env['CUDA_VISIBLE_DEVICES'] = ','.join(
                d2l._hw_visible_uuids()[i] for i in gpus)
        else:
            env['CUDA_VISIBLE_DEVICES'] = ''
        launcher = os.path.join(os.path.dirname(sys.executable), 'torchrun')
        try:
            run = d2l.hw_run(
                [launcher, '--standalone', '--nproc-per-node=4', '--log-dir',
                 tmp, '--redirects', '3', path, mode, json.dumps(sizes),
                 iters, warmup, check, json.dumps(intra), json.dumps(inter),
                 fn], timeout_s=timeout_s, env=env)
        finally:
            d2l._kill_workers(path)
        logs = glob.glob(os.path.join(tmp, '*', 'attempt_0', '0',
                                      'stdout.log'))
        lines = []
        if logs:
            with open(logs[0]) as f:
                lines = f.read().splitlines()
    recs = [json.loads(l) for l in lines if l.startswith('{')]
    rows = [r for r in recs if 'variant' in r]
    check_rec = next((r for r in recs if 'check' in r), None)
    return rows, check_rec, run

#@save
def _hier_compute(config):
    spec = _HW_COLL[config]
    rows, check, run = _hier_run(
        'cuda', spec['hier_sizes'], spec['hier_iters'], spec['hier_warmup'],
        spec['check_bytes'], spec['intra'], spec['inter'],
        spec['hier_timeout_s'])
    if run.refused or not rows or check is None:
        raise RuntimeError('hierarchical benchmark: '
                           f'{run.refused or "no rows"}\n{run.stderr[-2000:]}')
    out = pd.DataFrame(rows)
    rel = d2l._smi_relations()
    out.attrs.update(check=check, intra=spec['intra'], inter=spec['inter'],
                     intra_relation=rel(*spec['intra'][0][:2]),
                     inter_relation=rel(*spec['inter'][0][:2]))
    return out

#@save
_HIER_ROWS = ('flat gloo', 'flat NCCL', 'hierarchical', 'intra stage',
              'inter stage')

#@save
def _hier_predict(r):
    """(predicted seconds per row of r, intra/inter bandwidth ratio) from
    the committed fits of sec_lm_server_paths (NCCL on the intra pair's
    relation and on all four cards) and sec_lm_networks (gloo over loopback
    on 2 and 4 ranks); None while any of them is absent."""
    try:
        pair = d2l.box_alpha_beta(g=2, path=r.attrs['intra_relation'])
        four = d2l.box_alpha_beta(g=4)
        gloo = d2l.fit_collectives(d2l.hw_box_log('box-net', 'gloo'))
    except (KeyError, RuntimeError):
        return None
    if gloo.empty:
        return None
    g2, g4 = (gloo.loc[('all_reduce', k, 'loopback')] for k in (2, 4))
    a_in, b_in = pair['alpha'] / 2, pair['beta']  # per ring step
    ring = d2l.ring_allreduce_time
    model = {
        'flat gloo': lambda S: ring(S, 4, g4.alpha_step, g4.beta),
        'flat NCCL': lambda S: ring(S, 4, four['alpha'] / 6, four['beta']),
        'hierarchical': lambda S: d2l.hier_allreduce_time(
            S, 2, 2, a_in, b_in, g2.alpha_step, g2.beta, rails=2),
        'intra stage': lambda S: ring(S, 2, a_in, b_in),
        'inter stage': lambda S: ring(S / 2, 2, g2.alpha_step, g2.beta)}
    pred = np.array([model[v](s) for v, s in zip(r.variant, r['size'])])
    return pred, b_in / g2.beta

#@save
def _show_hier(r, sizes=(2**24, 2**28)):
    """At most five rows: each variant's time (ms) at two sizes, measured
    and predicted by the closed forms; one line with the hierarchical gain
    over the flat gloo baseline, measured and predicted; one with the
    intra/inter bandwidth ratio, measured from the two stages alone and
    from the fits."""
    if r.empty:
        print(r.attrs.get('label', 'pending'))
        return
    p = None if 'box-coll' in d2l._hw_regenerating() else _hier_predict(r)
    t = r.assign(model=p[0] if p else np.nan)
    cols = {}
    for s in sizes:
        x = t[t['size'] == s].set_index('variant').reindex(list(_HIER_ROWS))
        cols[(d2l._hw_bytes(s), 'ms')] = x.us.to_numpy() / 1e3
        if p:
            cols[(d2l._hw_bytes(s), 'model')] = x.model.to_numpy() * 1e3
    d2l.show_table(pd.DataFrame(cols, index=list(_HIER_ROWS)), digits=1,
                   width=79)
    big = t[t['size'] == max(sizes)].set_index('variant')
    text = (f"at {d2l._hw_bytes(max(sizes))}: hierarchical "
            f"{big.us['flat gloo'] / big.us['hierarchical']:.1f}x faster "
            f'than flat gloo')
    if p:
        text += (f" ({big.model['flat gloo'] / big.model['hierarchical']:.1f}"
                 f'x by the model)')
    print(text)
    # a 2-rank ring moves S per rank in the intra stage, S / 2 in the inter
    ratio = 2 * big.us['inter stage'] / big.us['intra stage']
    print(f'intra/inter bandwidth: {ratio:.1f}x measured from the stages'
          + (f', {p[1]:.1f}x from the fits' if p else ''))

#@save
def _check_line(rec, where):
    if rec is None:
        return f'{where}: correctness check not run'
    verdict = '=' if rec['ok'] else 'DIFFERS FROM'
    return (f"{where}: hier_allreduce {verdict} CPU sum (rtol 1e-5, "
            f"max error {rec['max_rel_err']:.0e})")

#@save
def hier_vs_flat(config='box'):
    """The hand-written hierarchical all-reduce (NCCL inside the pairs 0-1
    and 2-3, gloo over loopback across them) against the flat all-reduce on
    gloo and on NCCL, and its two stages alone, from 1 MiB to 256 MiB on
    four GPUs: a long table variant, size, us, mad_us (committed in
    box-coll) and model_us, the closed forms' prediction from the earlier
    sections' fits (NaN while they are absent). A normal run re-times 64
    MiB on four GPUs and checks hier_allreduce against a CPU sum (on four
    CPU gloo processes without four GPUs)."""
    spec, checks = _HW_COLL[config], []
    def live(r, same):
        rows, check, run = _hier_run(
            'cuda', [_HW_COLL_LIVE['hier_size']], 5, 2, spec['check_bytes'],
            spec['intra'], spec['inter'], _HW_COLL_LIVE['timeout_s'])
        checks.append(check)
        if run.refused or not rows:
            return f'{run.refused or "no rows"}; committed run shown'
        got = {x['variant']: x['us'] for x in rows}
        pts = []
        for v in ('flat gloo', 'flat NCCL', 'hierarchical'):
            i = d2l._row(r, variant=v, size=_HW_COLL_LIVE['hier_size'])
            pts.append((v, float(r.us.iloc[i]), got[v],
                        d2l._spread(r, 'us', i, 0.05)))
        return d2l._hw_agreement(pts, same, 'us')
    four = torch.cuda.device_count() >= 4
    r = d2l._hw_result(
        'box-coll', lambda: _hier_compute(config),
        dict(config=config, sizes=spec['hier_sizes'],
             iters=spec['hier_iters'], warmup=spec['hier_warmup'],
             intra=spec['intra'], inter=spec['inter']),
        entry='hier', live_check=live if four else None, show=_show_hier,
        live_label='64 MiB')
    if 'box-coll' in d2l._hw_regenerating() or r.empty:
        return r
    p = _hier_predict(r)
    r = r.assign(model_us=p[0] * 1e6 if p else np.nan)
    if four:
        print(_check_line(checks[0] if checks else None, 'live, 4 GPUs'))
    else:
        print('needs four GPUs: committed run shown; the check runs on the '
              'CPU (gloo)')
        with d2l._quiet():
            _, check, _ = _hier_run('cpu', [], 1, 0, spec['check_bytes'],
                                    spec['intra'], spec['inter'],
                                    _HW_COLL_LIVE['timeout_s'])
        print(_check_line(check, '4 CPU processes'))
    return r
```

## The Regeneration Driver

`regen_coll` is the regeneration of box-coll: NCCL's INFO digest, the forced
pairs and the hierarchical benchmark, on all four cards of a quiet box
(`tools/hw_regen.py box-coll` under `gpu_run.sh 0,1,2,3`, three processes).
`regen_coll_finish` keeps every process's times with the combined tables
(`attrs['proc_us']`, from which `fit_collectives` and `tuner_regret` draw
their noise) and writes `alpha_beta.json`: one record per kept pair in the
schema of :numref:`sec_lm_server_paths` with `algo` and `proto` filled
(`None` for NCCL's default, the row chapter 23 reads).

```{.python .input #collectives-lib-driver}
%%tab pytorch
#@save
def _coll_check(config):
    spec = _HW_COLL[config]
    if spec.get('launcher') == 'campaign':
        raise ValueError(f'{config} runs on several nodes through the '
                         'campaign launcher, not tools/hw_regen.py')
    n = torch.cuda.device_count()
    host = len(d2l._hw_smi('--query-gpu=index'))
    if n < spec['g'] or n != host:
        raise RuntimeError(f"box-coll needs every GPU of the host and at "
                           f"least {spec['g']} ({host} on the host, {n} "
                           'visible): lock them all')

#@save
def regen_coll(config='box'):
    """Driver of box-coll: NCCL's INFO digest on all cards, the forced
    pairs and the hierarchical benchmark."""
    _coll_check(config)
    nccl_info(gpus=tuple(range(_HW_COLL[config]['g'])))
    pairs = nccl_algo_proto_sweeps(config)
    hier_vs_flat(config)
    return pairs

#@save
def _coll_records(fits, info):
    """alpha_beta.json rows of the kept pairs (sec_lm_server_paths' schema;
    algo and proto None for NCCL's default)."""
    num = lambda x: None if np.isnan(x) else float(x)
    via = ' '.join((info or {}).get('transports', []))
    link = ('nvlink' if 'NVL' in via else 'pcie_p2p' if 'P2P' in via
            else 'pcie_host_staged')
    recs = []
    for (algo, proto), r in fits.iterrows():
        recs.append(dict(
            scope='intra_node', link=link, op='all_reduce',
            g=int(fits.attrs['g']), algo=None if algo == 'default' else algo,
            proto=None if proto == 'default' else proto,
            transport=d2l._transport(info), path='all', env=None,
            alpha_s=float(r.alpha), beta_Bps=float(r.beta),
            alpha_ci95=[num(r.alpha_lo), num(r.alpha_hi)],
            beta_ci95=[num(r.beta_lo), num(r.beta_hi)],
            n_half_bytes=float(r.n_half), resid_rel=float(r.resid_rel),
            n_points=int(r.n_points)))
    return recs

#@save
def regen_coll_finish(folder):
    """After tools/hw_regen.py combined the processes: every process's
    times with the combined tables, and alpha_beta.json."""
    for entry, col in (('pairs', 'oop_us'), ('hier', 'us')):
        path = os.path.join(folder, f'{entry}.json.gz')
        table, meta = d2l._hw_read(path)
        procs = sorted(glob.glob(os.path.join(folder, 'proc*',
                                              f'{entry}.json.gz')))
        table.attrs['proc_us'] = [d2l._hw_read(p)[0][col].tolist()
                                  for p in procs]
        d2l._hw_write(path, table, meta)
    pairs, _ = d2l._hw_read(os.path.join(folder, 'pairs.json.gz'))
    info, _ = d2l._hw_read(os.path.join(folder, 'info.json.gz'))
    recs = _coll_records(fit_pairs(pairs), info)
    with open(os.path.join(folder, 'alpha_beta.json'), 'w') as f:
        json.dump(recs, f, indent=1)
    return recs
```
