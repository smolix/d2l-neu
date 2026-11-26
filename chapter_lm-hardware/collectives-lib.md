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
configurations; the simulator's topology and its claim line
(`sim_summary`); the all-to-all's closed form
(`a2a_inter_node_time`, for chapter 23); NCCL's own account of a run
(`nccl_info`); the forced-pair sweeps with their drop reasons
(`nccl_algo_proto_sweeps`), their plot and fits (`plot_algo_crossover`,
`fit_pairs`); the tuner check (`tuner_regret`); the opening's two symptoms
(`coll_symptoms`); the hierarchical benchmark and its torchrun script
(`hier_vs_flat`); and the driver `regen_coll` with its hooks
`regen_coll_prepare` and `regen_coll_finish`. Blocks are tagged
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

The box result, committed with the book (tools/hw_regen.py writes it).

```{.python .input #collectives-lib-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-hw-buildbox-coll-v1'] = (
    d2l.LM_DATA_URL + 'lm-hw-buildbox-coll-v1.tar',
    '2d0e6e2ec405653039d1e00c942b4f014b8e7c98')
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
pairs (compared at 256 B, 16 MiB and 256 MiB, sizes on flat parts of the
curves, away from the steps where NCCL changes its channel count) and one
hierarchical size. Campaign C-T4 (two 8 x A100 nodes with
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
                     sizes=(2**8, 2**24, 2**28), hier_size=2**26,
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

#@save
def sim_summary(err):
    """Claim C1 in one sentence, read from the simulator's table (G,
    algo, the signed error in % of the simulation against the closed form
    at each 'f S_b (%)' column): the ring's largest error, the tree's
    largest excess over the closed form, its largest error far from S_b
    (the first and last columns) and its largest deficit at S_b, with the
    G where it occurs."""
    import textwrap
    cols = [c for c in err.columns if str(c).endswith('S_b (%)')]
    ring, tree = err[err.algo == 'ring'], err[err.algo == 'tree']
    r = float(ring[cols].abs().max().max())
    over = float(tree[cols].max().max())
    far = float(tree[[cols[0], cols[-1]]].abs().max().max())
    at = tree['1 S_b (%)']
    i = at.idxmin()
    text = (f"ring: {'exact' if r == 0 else f'within {r:.1f}%'}; tree: "
            + (f'exceeds the closed form by at most {over:.1f}%'
               if over > 0 else 'never above the closed form')
            + f', within {far:.1f}% of it far from S_b, '
            + (f'up to {-at[i]:.1f}% below it at S_b (G = {tree.G[i]})'
               if at[i] < 0 else 'never below it at S_b'))
    print(textwrap.fill(text, 79, subsequent_indent='  '))
```

## The All-to-All across Nodes

The section's text derives the cross-node all-to-all in a sentence (a share
g / G of each GPU's bytes stays in its node, the rest crosses its NIC, PXN
first moves a byte to the GPU on the destination's rail); this page computes
nothing with it, chapter 23 composes it with the other time functions.

```{.python .input #collectives-lib-a2a}
%%tab pytorch
#@save
def a2a_inter_node_time(S, n, g, a_out, b_out, a_in, b_in):
    """Seconds of an all-to-all of S bytes per GPU across n nodes of g
    GPUs, one NIC per GPU (PXN routes each byte to the GPU on its
    destination's rail first): the share g / G stays in the node, the rest
    crosses the GPU's own NIC, and the two run at the same time."""
    G = n * g
    inside = (g - 1) * a_in + S * (g - 1) / (G * b_in)
    across = (n - 1) * a_out + S * (G - g) / (G * b_out)
    return max(inside, across)
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
    model of each pair it may use (its algorithm bandwidth converted to
    nccl-tests' bus bandwidth, the convention of the fits) and the pairs it
    disabled, the choice per size."""
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
    if tab:  # NCCL's bandwidths are algorithm bandwidths: shown as busbw
        phi = d2l.nccl_busbw_factor('all_reduce', max(d['n_ranks'] or 2, 2))
        on = [f'{a}/{p} {lat:.0f} us, {phi * bw:.1f} GB/s'
              for (a, p), (lat, bw) in sorted(tab.items()) if bw > 0]
        off = [f'{a}/{p}' for (a, p), (_, bw) in sorted(tab.items())
               if bw <= 0]
        say('tuner model of an all-reduce (latency, busbw as nccl-tests '
            'reports it): ' + ', '.join(on)
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
and nccl-tests' result check (`-c 1`, one iteration per size, factor 2):
a refusal, a timeout, a wrong result, an INFO choice other than the forced
pair, or no INFO choice at all drops the pair with its reason ('refused',
'timed out', 'wrong results', 'ignored: ran Ring/Simple', 'unverified').
The check is one iteration: it shows that a pair runs and reports itself,
not that it is safe; whether NCCL's tuner enables a pair is a separate fact,
its table, kept in `attrs['tuner']` from the default's check run. Then the
timed sweep runs without INFO (its lines would cost host time inside the
timed loop). The long table has every pair at every size (NaN times for a
dropped pair), the pair in `algo` (`'default'`, `'Ring/LL'`, ...), its
protocol in `proto`, the drop reason in `status` (empty for a kept pair)
and what NCCL reported choosing at each size in `used`, so that
`fit_collectives` fits any one pair.

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
    """(reason or '', {bytes: pair NCCL ran}, the tuner's table) of one
    pair: a short run with NCCL_DEBUG=INFO and the result check at every
    size of the sweep. A forced pair whose run reports no choice at all is
    'unverified' (an NCCL that stopped printing its choice lines must not
    keep every forced pair unchecked)."""
    run, text = _info_run(gpus, _pair_env(algo, proto),
                          ('-b', spec['min_bytes'], '-e', spec['max_bytes'],
                           '-f', 2, '-w', 1, '-n', 1, '-c', 1),
                          spec['timeout_s'])
    forced = algo != 'default'
    if run is None:  # no nccl-tests: no INFO run to check against
        return ('unverified' if forced else ''), {}, {}
    if run.refused:
        return ('timed out' if run.refused.startswith('timeout')
                else 'refused'), {}, {}
    table = d2l.parse_nccl_tests(run.stdout)
    if (table[['oop_wrong', 'ip_wrong']].fillna(0) > 0).any().any():
        return 'wrong results', {}, {}
    used, tuner = _choices(text), _tuner_table(text)
    want = _pair_name(algo, proto).lower()
    other = sorted({u for u in used.values()
                    if forced and u.lower() != want})
    if other:
        return f"ignored: ran {', '.join(other)}", used, tuner
    if forced and not used:
        return 'unverified', used, tuner
    return '', used, tuner

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
    frames, dropped, tuner = [], {}, {}
    for algo, proto in spec['pairs']:
        name = _pair_name(algo, proto)
        reason, used, table = _pair_check(gpus, algo, proto, spec)
        if algo == 'default':  # the tuner's own table, unforced
            tuner = {f'{a}/{p}': v for (a, p), v in table.items()}
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
    out.attrs.update(dropped=dropped, tuner=tuner,
                     alpha_below=spec['alpha_below'],
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
    r = d2l._hw_result('box-coll', lambda: _pairs_compute(config),
                       config_doc, entry='pairs', live_check=_pairs_live,
                       show=_show_pairs, live_label='three pairs')
    if not torch.cuda.is_available() and not r.empty and (
            'box-coll' not in d2l._hw_regenerating()):
        print('no GPU: the committed box run is shown')
    return r
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
def _tuner_off(pairs):
    """The pairs NCCL's own tuner disables on the measured machine
    (bandwidth 0 in its table, attrs['tuner']): they run when forced and
    pass a one-iteration check, but NCCL does not consider them safe
    there. Empty when the table was not recorded."""
    table = pairs.attrs.get('tuner') or {}
    return {name for name, (_, bw) in table.items() if bw <= 0}

#@save
def _say(text, width=79):
    """Print text wrapped at width, continuation lines indented."""
    import textwrap
    print(textwrap.fill(text, width, subsequent_indent='  '))

#@save
def _fmt_s(t):
    """Seconds as '36 us', '167 us', '1.07 ms' or '169 ms'."""
    return f'{t * 1e6:.0f} us' if t < 1e-3 else f'{t * 1e3:.3g} ms'

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
    default (bold black at 1) against message size; a pair NCCL's tuner
    disables is dashed and labelled so; a dotted line at the measured
    crossover of small and large."""
    kept, off = _kept(pairs), _tuner_off(pairs)
    if 'default' not in kept:
        print(pairs.attrs.get('label', 'no forced pairs'))
        return
    s, base = kept['default']
    names = [n for n in kept if n != 'default']
    rel = [np.interp(s, *kept[n]) / base for n in names]
    d2l.plot([s] * (len(names) + 1), rel + [np.ones_like(s)],
             xlabel='message size (bytes)',
             ylabel="time / NCCL default's",
             legend=[n + (' (off)' if n in off else '') for n in names]
             + ['default'], xscale='log', figsize=(5, 3),
             legend_loc='outside',
             fmts=[f"C{i}{'--' if n in off else '-'}"
                   for i, n in enumerate(names)] + ['k-'])
    ax = d2l.plt.gca()
    for line in ax.get_lines():
        line.set_linewidth(1.0)
    ax.get_lines()[len(names)].set_linewidth(2.5)
    if small in kept and large in kept:
        x = _crossing(kept[small][0], kept[small][1], kept[large][1])
        if np.isfinite(x):
            ax.axvline(x, color='k', linestyle=':', linewidth=1)

#@save
def fit_pairs(pairs, small='Ring/LL', large='Ring/Simple',
              window=(2**14, 2**20)):
    """alpha (s) and beta (bytes/s, bus bandwidth) per kept pair of a
    forced-pair table (fit_collectives' two regimes, 95% intervals over
    the regeneration's processes), indexed by (algo, proto); attrs
    'crossover': the measured and the predicted size at which large
    overtakes small (t = alpha + phi S / beta of each fit; NaN with the
    reason in 'why' when the fits predict none) and, for the sizes in
    window, each fit against its measured times and the two fitted lines
    against each other ('resid', 'gap'), 'off': the pairs NCCL's tuner
    disables. A campaign log gives its 'pairs' entry's fits, or
    while it is pending an empty table whose attrs['status'] is its status
    line."""
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
                     g=int(pairs.g.iloc[0]), off=sorted(_tuner_off(pairs)))
    if small in kept and large in kept:
        phi = d2l.nccl_busbw_factor('all_reduce', out.attrs['g'])
        a, b = (out.loc[tuple(n.split('/'))] for n in (small, large))
        why = (f"{small}'s fitted bandwidth is not below {large}'s"
               if a.beta >= b.beta else
               f"{small}'s fitted alpha is not below {large}'s"
               if a.alpha >= b.alpha else '')
        pred = float('nan') if why else (b.alpha - a.alpha) / (
            phi * (1 / a.beta - 1 / b.beta))
        s, t1 = kept[small]
        out.attrs['crossover'] = dict(
            small=small, large=large, why=why,
            measured=_crossing(s, t1, kept[large][1]), predicted=float(pred),
            **_pair_residuals(kept, a, b, small, large, phi, window))
    return out

#@save
def _pair_residuals(kept, a, b, small, large, phi, window):
    """How well the two alpha-beta lines t = alpha + phi S / beta (fits a
    of small, b of large) describe the sizes in window (lo, hi bytes):
    each fit against its measured times (fit / measured - 1, smallest and
    largest), and the large pair's fitted line against the small pair's
    at both ends of the window; {} without sizes there."""
    lo, hi = window
    line = lambda r, x: r.alpha + phi * x / r.beta
    resid = {}
    for name, r in ((small, a), (large, b)):
        x, t = kept[name]
        m = (x >= lo) & (x <= hi)
        if not m.any():
            return {}
        e = line(r, x[m]) / t[m] - 1
        resid[name] = (float(e.min()), float(e.max()))
    x = kept[small][0]
    x = x[(x >= lo) & (x <= hi)][[0, -1]]
    gap = line(b, x) / line(a, x) - 1
    return dict(window=(int(x[0]), int(x[1])), resid=resid,
                gap=(float(gap[0]), float(gap[1])))

#@save
def show_pair_fits(fits):
    """At most seven rows: algorithm, protocol ('(off)' when NCCL's tuner
    disables the pair), alpha (us), beta (GB/s) and beta's 95% interval;
    one line with the measured and the predicted crossover, and one with
    the two fits' errors near it. For a pending log its status line."""
    if fits.empty:
        print(fits.attrs.get('status') or fits.attrs.get('label', 'no fits'))
        return
    off = set(fits.attrs.get('off', []))
    ci = lambda lo, hi: ('-' if np.isnan(lo) else
                         f'{lo / 1e9:.2f}-{hi / 1e9:.2f}')
    d2l.show_table(pd.DataFrame({
        'algo': [i[0] for i in fits.index],
        'proto': [i[1] + (' (off)' if '/'.join(i) in off else '')
                  for i in fits.index],
        'alpha (us)': fits.alpha.to_numpy() * 1e6,
        'beta (GB/s)': fits.beta.to_numpy() / 1e9,
        'beta 95%': [ci(lo, hi) for lo, hi in zip(fits.beta_lo,
                                                  fits.beta_hi)]}),
        digits={'alpha (us)': 0, 'beta (GB/s)': 2}, max_rows=7, width=79)
    x = fits.attrs.get('crossover')
    if x:
        size = lambda b: _approx_bytes(b) if np.isfinite(b) else 'none'
        _say(f"{x['large']} overtakes {x['small']} at {size(x['measured'])} "
             + (f"measured; no crossover predicted: {x['why']}" if x['why']
                else f"measured, {size(x['predicted'])} predicted by the "
                'fits'))
        if x.get('resid'):
            lo, hi = (d2l._hw_bytes(b) for b in x['window'])
            pct = lambda e: f'{100 * e:+.0f}%'
            fits_vs = ', '.join(f'{n} {pct(e0)} to {pct(e1)}'
                                for n, (e0, e1) in x['resid'].items())
            _say(f'fit against measured, {lo} to {hi}: {fits_vs}; '
                 f"{x['large']}'s fit against {x['small']}'s: "
                 f"{pct(x['gap'][0])} at {lo}, {pct(x['gap'][1])} at {hi}")

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
def _med_mad(pairs, name):
    """(sizes, median time in s, median absolute deviation in s) of one
    pair over the regeneration's processes (MAD 0 with one process)."""
    s, t = _proc_times(pairs, name)
    med = np.median(t, axis=0)
    return s, med, np.median(np.abs(t - med), axis=0)

#@save
def protocol_order(pairs, small='Ring/LL', large='Ring/Simple', k=3,
                   floor=0.02):
    """Claim C2's first half: small is faster at the smallest size, large
    at the largest. One line per size with both times and the gap in
    noise units (the larger MAD of the two times over the processes, at
    least floor of the faster time); returns the two gaps in noise units,
    positive when the expected pair is faster. A gap within k units is
    called within noise."""
    if small not in _kept(pairs) or large not in _kept(pairs):
        print(f'{small} or {large} dropped: no protocol order')
        return np.array([])
    stats, units = {n: _med_mad(pairs, n) for n in (small, large)}, []
    for i, fast, slow in ((0, small, large), (-1, large, small)):
        (s, t_f, m_f), (_, t_s, m_s) = stats[fast], stats[slow]
        noise = max(m_f[i], m_s[i], floor * min(t_f[i], t_s[i]))
        units.append((t_s[i] - t_f[i]) / noise)
        verdict = (f'{fast} faster' if units[-1] > k else
                   f'{slow} faster' if units[-1] < -k else 'within noise')
        _say(f'{d2l._hw_bytes(int(s[i]))}: {fast} {_fmt_s(t_f[i])}, {slow} '
             f'{_fmt_s(t_s[i])}: {verdict} ({units[-1]:+.1f} noise units)')
    return np.array(units)

#@save
def _regret_table(pairs, floor=0.02):
    """Per size at which NCCL reported its choice ('own', in the default's
    `used` column): the default's time, the forced copy of its own choice,
    the best other forced pair the tuner enables and its time; 'cost', the
    forced copy over the default minus 1 (what forcing changes besides the
    pair); 'gain', the time the best pair saves against the forced own
    choice (both forced: the comparison is not confounded by forcing);
    'units', that gap over the noise unit, the largest of the MADs of the
    two forced times, the cost of forcing and floor of the time."""
    off = _tuner_off(pairs)
    on = [n for n in _kept(pairs) if n != 'default' and n not in off]
    used = pairs[pairs.algo == 'default'].set_index('size')['used']
    s, t_def, _ = _med_mad(pairs, 'default')
    stats = {n: _med_mad(pairs, n)[1:] for n in on}
    rows = []
    for i, size in enumerate(s):
        own = used.get(size, '')
        rivals = [n for n in on if n != own]
        if own not in stats or not rivals:
            continue
        best = min(rivals, key=lambda n: stats[n][0][i])
        (t_own, m_own), (t_best, m_best) = (
            (stats[n][0][i], stats[n][1][i]) for n in (own, best))
        noise = max(m_own, m_best, math.fabs(t_def[i] - t_own),
                    floor * t_best)
        rows.append(dict(size=int(size), own=own, default_s=t_def[i],
                         own_s=t_own, best=best, best_s=t_best,
                         cost=t_own / t_def[i] - 1, gain=1 - t_best / t_own,
                         units=(t_own - t_best) / noise))
    return pd.DataFrame(rows)

#@save
def _tuner_switch(pairs, small, large):
    """Bytes at which NCCL's own table (attrs['tuner'], latency in us,
    algorithm bandwidth in GB/s) switches from small to large; NaN when
    the table lacks either or predicts no switch."""
    table = pairs.attrs.get('tuner') or {}
    if small not in table or large not in table:
        return float('nan')
    (lat_s, bw_s), (lat_l, bw_l) = table[small], table[large]
    if min(bw_s, bw_l) <= 0 or bw_s >= bw_l or lat_s >= lat_l:
        return float('nan')
    return (lat_l - lat_s) * 1e3 / (1 / bw_s - 1 / bw_l)  # GB/s = 1e3 B/us

#@save
def tuner_regret(pairs, k=3, floor=0.02):
    """Is NCCL's choice right? Claim C3, tested forced against forced
    (_regret_table): the cost of forcing NCCL's own choice; the largest
    gain of another enabled pair over it, in noise units; the number of
    sizes within k units; the size at which the tuner's table switches
    between its small- and large-message choices against where the forced
    pairs cross. Returns the table."""
    r = _regret_table(pairs, floor)
    if r.empty:
        print(pairs.attrs.get('label', "no NCCL choice or no forced pair"))
        return r
    c = r.cost.abs()
    _say(f"cost of forcing NCCL's own choice: median {c.median():.0%}, up to "
         f"{c.max():.0%} at {d2l._hw_bytes(int(r['size'][c.idxmax()]))}")
    x = r.loc[r.units.idxmax()]
    if x.units > k:
        _say(f'largest gain over NCCL\'s choice, both forced: {x.best} '
             f'{x.gain:.0%} faster than {x.own} at '
             f"{d2l._hw_bytes(int(x['size']))} ({x.units:.1f} noise units)")
    else:
        _say(f"no forced pair beats NCCL's choice by more than {k} noise "
             'units')
    off = sorted(_tuner_off(pairs) & set(_kept(pairs)))
    n_proc = len(pairs.attrs.get('proc_us') or [0])
    _say(f'within {k} noise units at {int((r.units <= k).sum())} of '
         f'{len(r)} sizes (unit: the largest of the MAD of {n_proc} run'
         f"{'s' if n_proc > 1 else ''}, the cost of forcing and {floor:.0%})"
         + (f"; {', '.join(off)} left out: off in NCCL's tuner" if off
            else ''))
    small, large = r.own.iloc[0], r.own.iloc[-1]
    if small != large:
        kept = _kept(pairs)
        cross = _crossing(kept[small][0], kept[small][1], kept[large][1])
        _say(f"NCCL's table switches {small} -> {large} at "
             f'{_approx_bytes(_tuner_switch(pairs, small, large))}; '
             f'forced, they cross at {_approx_bytes(cross)}')
    return r

#@save
def coll_symptoms(log, ab, k=3):
    """The opening's two lines from the committed box-coll result (log:
    hw_box_log('box-coll')) and sec_lm_server_paths' all-reduce fit ab
    (box_alpha_beta()): the size at which a forced pair beats NCCL's own
    choice by the most noise units (_regret_table), with both times; the
    bus bandwidth NCCL's tuner assumes for its large-message choice
    against the fitted one."""
    pairs = log.get('pairs') if isinstance(log, dict) else log
    if pairs is None or pairs.empty:
        print(getattr(pairs, 'attrs', {}).get('label', 'box-coll pending'))
        return
    g, r = int(pairs.g.iloc[0]), _regret_table(pairs)
    x = None if r.empty else r.loc[r.units.idxmax()]
    if x is not None and x.units > k:
        _say(f"{d2l._hw_bytes(int(x['size']))} on {g} GPUs: NCCL's choice "
             f'{x.own} {_fmt_s(x.default_s)}, forced {x.best} '
             f'{_fmt_s(x.best_s)}')
    else:
        print(f"all-reduce on {g} GPUs: NCCL's choice is within noise of "
              'every forced pair')
    table, used = pairs.attrs.get('tuner') or {}, r.own.iloc[-1] if len(
        r) else ''
    if table.get(used, (0, 0))[1] > 0:
        phi = d2l.nccl_busbw_factor('all_reduce', g)
        print(f"NCCL's model of {used}: {phi * table[used][1]:.0f} GB/s "
              f"busbw; the measured line: {ab['beta'] / 1e9:.1f} GB/s")
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
    """{'pred': predicted seconds per row of r, 'ratio': the 2-rank
    intra/inter bandwidth ratio of the fits, 'gloo4', 'shm': the inputs of
    the C4 gate} from the committed fits of sec_lm_server_paths (NCCL on
    the intra pair's relation and on all four cards) and sec_lm_networks
    (gloo over loopback on 2 and 4 ranks); None while any is absent."""
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
    return dict(pred=pred, ratio=b_in / g2.beta, gloo4=g4, shm=four)

#@save
def _c4_gate(gloo4, shm):
    """The gate of claim C4, fixed before the box-net regeneration (ch22
    spec log, "The 22.6 gate rule"): C4 stands, gloo over loopback being
    the emulated slow network, iff the whole 95% interval of gloo's
    4-rank beta lies below half the beta of NCCL's shared-memory path on
    the four cards; with one process (no interval) iff gloo's beta lies
    below a third. (stands, the line stating inputs, ratios and branch)."""
    beta, hi, b_shm = gloo4.beta, gloo4.beta_hi, shm['beta']
    if np.isfinite(hi):
        stands = hi < b_shm / 2
        ci, rule = f' (95% {gloo4.beta_lo / 1e9:.2f}-{hi / 1e9:.2f})', (
            "gloo's interval is below half" if stands else
            "gloo's interval is not below half")
    else:
        stands, ci = beta < b_shm / 3, ' (one run)'
        rule = ('one run, gloo is below a third' if stands else
                'one run, gloo is not below a third')
    branch = ('C4 stands' if stands else
              'C4 takes its fallback (gloo is slower by latency only)')
    return stands, (
        f"C4 gate: NCCL's shared memory {b_shm / 1e9:.2f} GB/s busbw, gloo "
        f'on 4 ranks {beta / 1e9:.2f}{ci}: {b_shm / beta:.1f}x the '
        f"bandwidth, 1/{gloo4.alpha / shm['alpha']:.0f} the latency; {rule}, "
        f'so {branch}')

#@save
def _show_hier(r, sizes=(2**24, 2**28)):
    """At most five rows: each variant's time (ms) at two sizes, measured
    and predicted by the closed forms; one line with the hierarchical gain
    over the flat gloo baseline, measured and predicted; one with the
    intra/inter bandwidth ratio of the 2-rank stages, measured and from
    the fits; the C4 gate's line (_c4_gate)."""
    if r.empty:
        print(r.attrs.get('label', 'pending'))
        return
    regen = 'box-coll' in d2l._hw_regenerating()
    p = None if regen else _hier_predict(r)
    t = r.assign(model=p['pred'] if p else np.nan)
    cols = {}
    for s in sizes:
        x = t[t['size'] == s].set_index('variant').reindex(list(_HIER_ROWS))
        cols[(d2l._hw_bytes(s), 'ms')] = x.us.to_numpy() / 1e3
        if p:
            cols[(d2l._hw_bytes(s), 'model')] = x.model.to_numpy() * 1e3
    d2l.show_table(pd.DataFrame(cols, index=list(_HIER_ROWS)), digits=1,
                   width=79)
    big = t[t['size'] == max(sizes)].set_index('variant')
    gain = lambda c: big[c]['flat gloo'] / big[c]['hierarchical']
    _say(f'at {d2l._hw_bytes(max(sizes))}: hierarchical {gain("us"):.1f}x '
         'faster than flat gloo' + (
             f' ({gain("model"):.1f}x by the model)' if p else '' if regen
             else ' (no model: the box-p2p and box-net fits are absent)'))
    # a 2-rank ring moves S per rank in the intra stage, S / 2 in the inter
    ratio = 2 * big.us['inter stage'] / big.us['intra stage']
    print(f'2-rank stages: intra/inter bandwidth {ratio:.1f}x measured'
          + (f", {p['ratio']:.1f}x from the fits" if p else ''))
    if p:
        _say(_c4_gate(p['gloo4'], p['shm'])[1])

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
    r = r.assign(model_us=p['pred'] * 1e6 if p else np.nan)
    if four:
        print(_check_line(checks[0] if checks else None, 'live, 4 GPUs'))
    else:
        _say('needs four GPUs: committed run shown; the check runs on 4 CPU '
             'processes, gloo over loopback (not a datacenter network)')
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
Before the processes are combined, `regen_coll_prepare` makes their
forced-pair tables agree: a pair dropped in any process is dropped in all
(with the reason and in how many processes), and the pair NCCL reported at
a size is the one most processes saw, so that one flaky pair does not abort
the combination. `regen_coll_finish` keeps every process's times with the
combined tables
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
def regen_coll_prepare(folder, n):
    """Before tools/hw_regen.py combines processes proc0..proc{n-1}: a pair
    dropped in any process is dropped in every one (times NaN, status and
    attrs['dropped'] '<reason> in k of n runs' when k < n), and each size's
    `used` is the value most processes reported."""
    paths = [os.path.join(folder, f'proc{i}', 'pairs.json.gz')
             for i in range(n)]
    parts = [d2l._hw_read(p) for p in paths]
    why = collections.defaultdict(list)
    for t, _ in parts:
        for name, reason in t.attrs.get('dropped', {}).items():
            why[name].append(reason)
    dropped = {name: r[0] + (f' in {len(r)} of {n} runs' if len(r) < n
                             else '') for name, r in why.items()}
    keys = ['size', 'op', 'g', 'path', 'algo', 'proto', 'env', 'status',
            'used']
    used = [collections.Counter(u).most_common(1)[0][0]
            for u in zip(*(t.used.tolist() for t, _ in parts))]
    for (t, meta), path in zip(parts, paths):
        t['used'] = used
        for name, reason in dropped.items():
            hit = t.algo == name
            t.loc[hit, [c for c in t.columns if c not in keys]] = np.nan
            t.loc[hit, ['status', 'used']] = [reason, '']
        t.attrs['dropped'] = dropped
        d2l._hw_write(path, t, meta)

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
