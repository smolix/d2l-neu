# Data Paths inside a Server: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_server_paths` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1): the
section's registry entries, one statement each (the link table, the box's
topology and NCCL sweeps, the campaign logs C-T3 and C-T2pair), its
regeneration job and the campaign configurations; the captures of the
machine (`nvidia-smi topo`, `numactl`, `lspci`) and the two documentation
samples of other nodes; the parsers of `nvidia-smi topo`,
`p2pBandwidthLatencyTest` and nccl-tests; the link table; the topology
graph and its displays; the GPU-to-GPU matrix and the concurrent pairs;
the NCCL sweeps (nccl-tests, or a torch.distributed fallback through
torchrun with its own timeout), the NCCL transport digest, the copy-engine
microbenchmark; the alpha-beta fits per collective, `alpha_beta.json` and
the box's alpha-beta reader; the driver `regen_p2p`. Blocks are tagged
`%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_server_paths`

## Registries

```{.python .input #server-paths-lib-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-hw-links-v1'] = (
    d2l.LM_DATA_URL + 'lm-hw-links-v1.csv',
    '1ef216a1dd49b7a136e191d2a1d71ab9899d0b66')
```

```{.python .input #server-paths-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['links'] = ('lm-hw-links-v1', 'csv')
d2l.HW_ARTIFACTS['box-p2p'] = ('lm-hw-buildbox-p2p-v1', 'tar')
d2l.HW_CAMPAIGNS['node-p2p'] = (
    'lm-hw-node-p2p-v1', 'tar', 'C-T3', 'one 8 x A100 40 GB node')
d2l.HW_CAMPAIGNS['t2pair-p2p'] = (
    'lm-hw-t2pair-p2p-v1', 'tar', 'C-T2pair',
    'two A100 40 GB with an NVLink bridge')
```

```{.python .input #server-paths-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-p2p'] = (
    'regen_p2p', lambda: dict(config='box'))
d2l.HW_CONFIG_LOGS[('box-p2p', 't3')] = 'node-p2p'
d2l.HW_CONFIG_LOGS[('box-p2p', 't2pair')] = 't2pair-p2p'
```

## Configurations

A sweep is one nccl-tests run (8 B to 256 MiB in factors of 2, 5 warm-up
and 20 timed iterations, `-c 0`) of one operation on one placement of GPUs:
`pairs` is one pair per relation class of `nvidia-smi topo -m` that the
machine has (on the box a pair behind one host bridge, `PHB`, and a pair
across the two bridges, `NODE`), `all` every GPU, an integer the first that
many GPUs. The box measures all-reduce and all-to-all on every placement,
all-gather and reduce-scatter on all four cards, and all-reduce again with
`NCCL_ALGO=Ring`. Campaign C-T3 adds g = 4 and 8, `NCCL_ALGO=Tree` and the
`NCCL_P2P_DISABLE=1` emulation of a host-staged node; C-T2pair runs the
bridged pair with and without peer-to-peer and with the copy-engine knob.

```{.python .input #server-paths-lib-configs}
%%tab pytorch
#@save
_HW_P2P = {
    'box': dict(series=[
        ('all_reduce', 'pairs', 'default', ''),
        ('all_reduce', 'all', 'default', ''),
        ('alltoall', 'pairs', 'default', ''),
        ('alltoall', 'all', 'default', ''),
        ('all_gather', 'all', 'default', ''),
        ('reduce_scatter', 'all', 'default', ''),
        ('all_reduce', 'pairs', 'Ring', ''),
        ('all_reduce', 'all', 'Ring', '')],
        max_bytes=2**28, iters=20, warmup=5, copy_engine=True),
    't3': dict(series=[
        (op, place, 'default', '') for op in ('all_reduce', 'alltoall')
        for place in ('pairs', 4, 'all')] + [
        ('all_gather', 'all', 'default', ''),
        ('reduce_scatter', 'all', 'default', ''),
        ('all_reduce', 'all', 'Ring', ''),
        ('all_reduce', 'all', 'Tree', ''),
        ('all_reduce', 'all', 'default', 'NCCL_P2P_DISABLE=1'),
        ('alltoall', 'all', 'default', 'NCCL_P2P_DISABLE=1')],
        max_bytes=2**32, iters=50, warmup=10, copy_engine=False),
    't2pair': dict(series=[
        ('all_reduce', 'pairs', 'default', ''),
        ('alltoall', 'pairs', 'default', ''),
        ('all_reduce', 'pairs', 'default', 'NCCL_P2P_DISABLE=1'),
        ('alltoall', 'pairs', 'default', 'NCCL_P2P_DISABLE=1'),
        ('all_reduce', 'pairs', 'default',
         'NCCL_P2P_DISABLE=1 NCCL_SHM_USE_CUDA_MEMCPY=1')],
        max_bytes=2**32, iters=50, warmup=10, copy_engine=False)}

#@save
_HW_P2P_TIMEOUTS = dict(p2p_test=300, capture=60, info=120)  # seconds
```

## Captures of the Machine

`hw_capture(name)` runs one read-only command and returns its text (scrubbed
of the host name and addresses); where the command is missing, it returns
the box's committed capture and says so in one line. Two further names are
documentation samples of other machines, committed with the box result and
labelled "sample, not measured here": an 8 x A100 SXM node behind NVSwitch
(nccl-tests issue 149) and a workstation with two NVLink-bridged pairs of
A100 80 GB PCIe cards (JAX issue 11478).

```{.python .input #server-paths-lib-captures}
%%tab pytorch
#@save
_HW_CAPTURES = {
    'topo': ['nvidia-smi', 'topo', '-m'],
    'topo-p2p': ['nvidia-smi', 'topo', '-p2p', 'r'],
    'numactl': ['numactl', '-H'],
    'lspci': ['lspci', '-tv'],
    'pcie': ['nvidia-smi', '--query-gpu=index,pci.bus_id,pcie.link.gen.max,'
             'pcie.link.width.max', '--format=csv']}

#@save
_HW_SAMPLES = {  # name: (file in tools/lm_data/hw/samples, source)
    'nvswitch-node': ('doc_nvswitch_node_topo_m.txt',
                      'https://github.com/NVIDIA/nccl-tests/issues/149'),
    'bridged-pair-node': ('doc_bridged_pair_topo_m.txt',
                          'https://github.com/jax-ml/jax/issues/11478')}

#@save
_HW_SAMPLE_DIR = os.path.join('..', 'tools', 'lm_data', 'hw', 'samples')

#@save
def _hw_scrub(text):
    """text without this host's name, IPv4 addresses or home paths."""
    import platform
    host = platform.node()
    for name in {host, host.split('.')[0]} - {''}:
        text = re.sub(rf'(?<![A-Za-z.-]){re.escape(name)}(?![\w.-])',
                      '<host>', text)
    text = re.sub(r'\b\d{1,3}(\.\d{1,3}){3}\b', '<ip>', text)
    return re.sub(r'/home/[^/\s]+', '<path>', text)

#@save
def _hw_capture_now(name):
    """Text of capture `name` taken now (a sample: read from the repository),
    or None where the command or file is missing or the command refused."""
    if name in _HW_SAMPLES:
        path = os.path.join(_HW_SAMPLE_DIR, _HW_SAMPLES[name][0])
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8') as f:
            return f.read()
    cmd = _HW_CAPTURES[name]
    if shutil.which(cmd[0]) is None:
        return None
    run = d2l.hw_run(cmd, timeout_s=_HW_P2P_TIMEOUTS['capture'])
    return None if run.refused else _hw_scrub(run.stdout)

#@save
def _hw_capture_all():
    """Every capture and sample (the box result's captures entry)."""
    return {name: _hw_capture_now(name) or ''
            for name in (*_HW_CAPTURES, *_HW_SAMPLES)}

#@save
def hw_capture(name):
    """The text of a machine capture ('topo', 'topo-p2p', 'numactl',
    'lspci', 'pcie'), taken now, or of a documentation sample
    ('nvswitch-node', 'bridged-pair-node'). Without the command (and for a
    sample) the committed text of the box result is returned, with a
    one-line note for a missing command."""
    if name not in _HW_CAPTURES and name not in _HW_SAMPLES:
        raise KeyError(f'unknown capture {name!r}')
    if name in _HW_CAPTURES:
        text = _hw_capture_now(name)
        if text:
            return text
    elif 'box-p2p' in d2l._hw_regenerating():
        return _hw_capture_now(name)
    stored = d2l.hw_box_log('box-p2p', 'captures')
    if name in _HW_CAPTURES:
        print(f'{_HW_CAPTURES[name][0]} is not available here: the box\'s '
              'committed capture')
    return stored.get(name, '') if isinstance(stored, dict) else ''
```

## Parsers

```{.python .input #server-paths-lib-parsers}
%%tab pytorch
#@save
TopoMatrix = collections.namedtuple(
    'TopoMatrix', ['labels', 'link', 'cpu_affinity', 'numa_affinity',
                   'legend'])

#@save
_TOPO_CELL = re.compile(r'X|OK|CNS|GNS|TNS|NS|U|SOC|SYS|NODE|PHB|PXB|PIX|'
                        r'NV\d+')

#@save
def parse_topo_matrix(text):
    """The matrix of `nvidia-smi topo -m` or `topo -p2p r|w|n` as a
    TopoMatrix: labels (GPU0..., NIC0... or mlx5_0...), link[i][j] (the
    cell: X, NV12, PHB, NODE, SYS, ...; OK, CNS, ... for -p2p), cpu_affinity
    and numa_affinity per label (None where the row has none) and the
    legend ({code: meaning}). ANSI escapes and carriage returns are
    stripped; labels are read from the row heads, so a header whose names
    run together still parses."""
    text = re.sub(r'\x1b\[[0-9;]*m', '', text).replace('\r', '')
    rows = []
    for line in text.splitlines():
        cells = line.split()
        if (len(cells) > 1 and not cells[0].endswith(':')
                and _TOPO_CELL.fullmatch(cells[1])
                and not _TOPO_CELL.fullmatch(cells[0])):
            rows.append(cells)
    if not rows:
        raise ValueError('no topology rows found')
    labels, n = [r[0] for r in rows], len(rows)
    link, cpu, numa = [], [], []
    for r in rows:
        cells = r[1:1 + n]
        if len(cells) != n or not all(_TOPO_CELL.fullmatch(c) for c in cells):
            raise ValueError(f'row {r[0]} does not have {n} link cells')
        link.append(cells)
        rest = r[1 + n:]
        cpu.append(rest[0] if rest else None)
        numa.append(rest[1] if len(rest) > 1 and rest[1] != 'N/A' else None)
    legend = {}
    for line in text.split('Legend:', 1)[-1].splitlines():
        m = re.match(r'\s+(\S+)\s+=\s+(.*\S)', line)
        if m:
            legend[m.group(1)] = m.group(2)
    return TopoMatrix(labels, link, cpu, numa, legend)

#@save
def parse_p2p(text):
    """The matrices of cuda-samples' p2pBandwidthLatencyTest keyed by their
    titles ('Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)', ...;
    the two blocks under a latency title get the suffix ' GPU' and ' CPU'),
    each an n x n array."""
    out, lines, i = {}, text.splitlines(), 0
    is_row = lambda l: bool(l.split()) and l.split()[0].isdigit()
    while i < len(lines):
        title = lines[i].strip()
        i += 1
        if not title.endswith(('Matrix (GB/s)', 'Matrix (us)')):
            continue
        while i < len(lines):
            head = lines[i].split()
            if head and head[0] in ('D\\D', 'GPU', 'CPU'):
                rows, i = [], i + 1
                while i < len(lines) and is_row(lines[i]):
                    rows.append([float(x) for x in lines[i].split()[1:]])
                    i += 1
                key = title if head[0] == 'D\\D' else f'{title} {head[0]}'
                out[key] = np.array(rows)
                if head[0] in ('D\\D', 'CPU'):
                    break
            elif lines[i].strip().endswith(('(GB/s)', '(us)')):
                break
            else:
                i += 1
    return out

#@save
_NCCL_COLUMNS = ('size', 'count', 'dtype', 'redop', 'root', 'oop_us',
                 'oop_algbw', 'oop_busbw', 'oop_wrong', 'ip_us', 'ip_algbw',
                 'ip_busbw', 'ip_wrong')

#@save
def parse_nccl_tests(text):
    """The table of one nccl-tests run (v2.21/v2.22 column layout: size,
    count, type, redop, root, then time, algbw, busbw and #wrong out of
    place and in place) as a DataFrame with the columns _NCCL_COLUMNS;
    '#' lines are skipped, #wrong is NaN when the run did not check
    (-c 0). attrs: avg_busbw (GB/s) and the tool version. Another layout
    raises ValueError."""
    head = [l.split() for l in text.splitlines() if l.startswith('#')]
    if not any('out-of-place' in h and 'in-place' in h for h in head) or \
            not any(h[1:6] == ['size', 'count', 'type', 'redop', 'root']
                    for h in head):
        raise ValueError('not the nccl-tests v2.21/v2.22 column layout')
    num = lambda x: float('nan') if x == 'N/A' else float(x)
    rows = []
    for line in text.splitlines():
        cells = line.split()
        if line.startswith('#') or len(cells) != 13 or not cells[0].isdigit():
            continue
        rows.append([int(cells[0]), int(cells[1]), cells[2], cells[3],
                     int(cells[4]), *map(num, cells[5:13])])
    df = pd.DataFrame(rows, columns=list(_NCCL_COLUMNS))
    m = re.search(r'Avg bus bandwidth\s*:\s*([\d.]+)', text)
    v = re.search(r'nccl-tests version (\S+)', text)
    df.attrs.update(avg_busbw=float(m.group(1)) if m else None,
                    version=v.group(1) if v else None)
    return df
```

## The Link Table

`links_table` reads the sourced table (`lm-hw-links-v1`) and computes the
PCIe rows from lanes, rate and encoding: Gen 3 to 5 carry 128 bits of data
in every 130 on the wire, Gen 6's FLIT overhead is left out (its row says
so). Rates per direction; the vendors' NVLink figures are both directions
together, which the second column keeps.

```{.python .input #server-paths-lib-links}
%%tab pytorch
#@save
def _link_name(row):
    """'PCIe Gen 4 x16', 'NVLink 3 (A100 SXM)', ..."""
    if row['link'] == 'PCIe':
        return f"PCIe {row['generation']} x{int(row['lanes'])}"
    return f"{row['link']} {row['generation']}"

#@save
def links_table(show=True):
    """The link classes of a server (the sourced table) with GB/s per
    direction and both directions together, the PCIe rows computed (lanes x
    GT/s x 128/130 / 8 for Gen 3 to 5, raw for Gen 6), indexed by name;
    printed (name and the two rates) when show."""
    df = d2l._hw_static('links').copy()
    pcie = df['link'] == 'PCIe'
    coded = np.where(df['encoding'] == '128b/130b', 128 / 130, 1.0)
    df.loc[pcie, 'gbs_per_dir'] = (df.lanes * df.rate_gts / 8 * coded)[pcie]
    df.loc[pcie, 'gbs_bidir'] = 2 * df.loc[pcie, 'gbs_per_dir']
    df.index = [_link_name(r) for _, r in df.iterrows()]
    if show:
        d2l.show_table(df[['gbs_per_dir', 'gbs_bidir']],
                       digits={'gbs_per_dir': 1, 'gbs_bidir': 0}, width=79,
                       headers={'gbs_per_dir': 'GB/s per direction',
                                'gbs_bidir': 'both directions'})
    return df
```

## The Topology as a Graph and Its Displays

`topo_graph` turns the matrix into a graph with the PCIe structure that the
relation codes imply: devices joined by `PIX` or `PXB` hang below one PCIe
switch, devices joined by `PHB` (or closer) below one host bridge, devices
joined by `NODE` (or closer) on one CPU socket, named by its NUMA node, and
`SYS` joins sockets. A GPU pair marked `NV#` is joined by # NVLinks; when
every pair of GPUs is, the GPUs hang off an NVSwitch fabric instead. Rates
come from the links table (one PCIe generation for every PCIe edge, the
named NVLink generation per link); the links inside the CPU (host bridge to
socket, socket to socket) have no datasheet rate and are left to the
measurement.

```{.python .input #server-paths-lib-graph}
%%tab pytorch
#@save
_TOPO_LEVELS = (('PCIe switch', {'PIX', 'PXB'}),
                ('host bridge', {'PIX', 'PXB', 'PHB'}),
                ('CPU', {'PIX', 'PXB', 'PHB', 'NODE'}))

#@save
def _topo_components(n, close):
    """Groups of 0..n-1 joined by the pairs (i, j) with close(i, j)."""
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if close(i, j):
                parent[root(i)] = root(j)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values())

#@save
def topo_graph(topo, links, nvlink='NVLink 3 (A100 SXM)',
               bridge='NVLink bridge 3 (A100 PCIe)', pcie='PCIe Gen 4 x16'):
    """A networkx graph of the machine that `topo` (parse_topo_matrix of
    `nvidia-smi topo -m`) describes: device nodes (GPUs, NICs); PCIe
    switch, host bridge and CPU nodes inferred from the relation codes; an
    NVSwitch node when every GPU pair is NVLinked (row nvlink of `links`),
    else NVLink bridges between the pairs (row bridge). Edge attributes:
    link (the links_table row) and gbs (GB/s per direction, NaN where no
    datasheet gives one)."""
    import networkx as nx
    lab, rel, n = topo.labels, topo.link, len(topo.labels)
    gpus = [i for i in range(n) if lab[i].startswith('GPU')]
    pcie_gbs = float(links.loc[pcie, 'gbs_per_dir'])
    G = nx.Graph()
    G.add_nodes_from(lab)
    nv_pairs = [(i, j) for i in gpus for j in gpus
                if i < j and rel[i][j].startswith('NV')]
    fabric = len(gpus) > 1 and len(nv_pairs) == len(gpus) * (
        len(gpus) - 1) // 2
    row = links.loc[nvlink if fabric else bridge]
    per_link = float(row.gbs_per_dir / row.lanes)  # GB/s per NVLink, one way
    for i, j in nv_pairs:
        lanes = int(rel[i][j][2:])
        if fabric:  # every pair NVLinked: each GPU's links go to NVSwitch
            G.add_edge(lab[i], 'NVSwitch', link=row.name,
                       gbs=lanes * per_link)
            G.add_edge(lab[j], 'NVSwitch', link=row.name,
                       gbs=lanes * per_link)
        else:
            G.add_edge(lab[i], lab[j], link=row.name, gbs=lanes * per_link)
    below = {i: lab[i] for i in range(n)}  # the node each device hangs from
    for name, codes in _TOPO_LEVELS:
        groups = _topo_components(n, lambda i, j: rel[i][j] in codes)
        for gi, grp in enumerate(groups):
            if len(grp) == 1 and name != 'CPU':
                continue
            numa = next((topo.numa_affinity[i] for i in grp
                         if topo.numa_affinity[i] is not None), gi)
            node = f'CPU {numa}' if name == 'CPU' else f'{name} {gi}'
            for top in sorted({below[i] for i in grp}):
                host = name == 'CPU' and top.startswith('host bridge')
                G.add_edge(top, node, link='host fabric' if host else pcie,
                           gbs=float('nan') if host else pcie_gbs)
            for i in grp:
                below[i] = node
    cpus = sorted({below[i] for i in range(n)})
    for a, b in zip(cpus, cpus[1:]):
        G.add_edge(a, b, link='inter-socket', gbs=float('nan'))
    return G

#@save
def show_edges(G):
    """The graph's edges as a table: from, to, link class, GB/s per
    direction (empty where no datasheet gives one)."""
    rows = [dict(a=a, b=b, link=d['link'], gbs=d['gbs'])
            for a, b, d in G.edges(data=True)]
    d2l.show_table(pd.DataFrame(rows), digits={'gbs': 1}, width=79,
                   headers={'a': 'from', 'b': 'to',
                            'gbs': 'GB/s per direction'})

#@save
def _pair_counts(topo):
    """'28 NV12' style counts of the relation of every GPU pair."""
    gpus = [i for i, l in enumerate(topo.labels) if l.startswith('GPU')]
    counts = {}
    for a in gpus:
        for b in gpus:
            if a < b:
                c = topo.link[a][b]
                counts[c] = counts.get(c, 0) + 1
    return ', '.join(f'{v} {k}' for k, v in sorted(
        counts.items(), key=lambda kv: (-kv[1], kv[0])))

#@save
def show_topo(topo, p2p=None, samples=()):
    """The GPU rows of a topology matrix beside the P2P read matrix (if
    given), one line on CPU and NUMA affinity, and one line per
    documentation sample (an hw_capture name): its GPUs, NICs and the
    relation of every GPU pair."""
    gpus = [i for i, l in enumerate(topo.labels) if l.startswith('GPU')]
    names = [topo.labels[i] for i in gpus]
    w = max(5, *(len(topo.link[i][j]) + 1 for i in gpus for j in gpus))
    pw = 4 if p2p is None else max(4, *(len(c) + 1 for r in p2p.link
                                        for c in r))
    fmt = lambda cells, width: ''.join(f'{c:>{width}}' for c in cells)
    gap = '' if p2p is None else '   '
    print(f'{"":5}{"link (topo -m)":>{w * len(gpus)}}' + (
        '' if p2p is None else f'{gap}{"P2P read":>{pw * len(gpus)}}'))
    print(f'{"":5}{fmt(names, w)}' + (
        '' if p2p is None else gap + fmt([n[3:] for n in names], pw)))
    for i in gpus:
        right = '' if p2p is None else gap + fmt(
            [p2p.link[i][j] for j in gpus], pw)
        print(f'{topo.labels[i]:5}{fmt([topo.link[i][j] for j in gpus], w)}'
              + right)
    cpu = {topo.cpu_affinity[i] for i in gpus}
    numa = {topo.numa_affinity[i] for i in gpus}
    if len(cpu) == 1 and len(numa) == 1:
        print(f'every GPU: CPUs {cpu.pop()}, NUMA node {numa.pop()}')
    else:
        print('NUMA node: ' + ', '.join(
            f'{topo.labels[i]} {topo.numa_affinity[i]}' for i in gpus))
    if samples:
        print('documentation samples, not measured here:')
    for name in samples:
        s = parse_topo_matrix(hw_capture(name))
        n_gpu = sum(l.startswith('GPU') for l in s.labels)
        nics = len(s.labels) - n_gpu
        print(f'  {name}: {n_gpu} GPUs' + (f', {nics} NICs' if nics else '')
              + '; GPU pairs ' + _pair_counts(s))
```

## The GPU-to-GPU Matrix

Each ordered pair of GPUs copies a 256 MiB buffer with `copy_` (the driver
stages it through host memory, since peer-to-peer is refused) and a 4-byte
buffer for the latency: wall clock from before the copies until both cards
have synchronized, the median of five rounds. The concurrent pairs copy at
the same moment from their own streams (threads released together); their
aggregate is the bytes moved over the wall time from the first start to
the last finish. `p2pBandwidthLatencyTest` runs as a cross-check where it
was built.

```{.python .input #server-paths-lib-matrix}
%%tab pytorch
#@save
def _smi_relations():
    """relation(i, j) between this process's GPUs i and j (torch indices),
    read from `nvidia-smi topo -m` through the GPUs' UUIDs."""
    topo = parse_topo_matrix(_hw_capture_now('topo') or '')
    smi = {u: int(i) for i, u in d2l._hw_smi('--query-gpu=index,uuid')}
    index = [topo.labels.index(f'GPU{smi[u]}')
             for u in d2l._hw_visible_uuids()]
    return lambda i, j: topo.link[index[i]][index[j]]

#@save
def _placements(rel, n):
    """{path: GPUs}: GPU 0 with its first partner in each relation class
    (PHB, NODE, NV12, ...) and, with more than two GPUs, 'all'."""
    out = {}
    for j in range(1, n):
        out.setdefault(rel(0, j), (0, j))
    if n > 2:
        out['all'] = tuple(range(n))
    return out

#@save
_NEAR = ('NV', 'PIX', 'PXB', 'PHB')  # relations below one host bridge

#@save
def _bridge_pairs(rel, n):
    """The groups of GPUs behind one host bridge (or closer), and the
    concurrent copy patterns: two pairs each behind its own bridge, two
    pairs across the bridges (both sources behind one bridge, both
    targets behind the other) and one card sending to two."""
    groups = _topo_components(n, lambda i, j: rel(i, j).startswith(_NEAR))
    pats = {}
    if len(groups) >= 2 and len(groups[0]) >= 2 and len(groups[1]) >= 2:
        (a, b), (c, d) = groups[0][:2], groups[1][:2]
        pats['two pairs, one bridge each'] = [(a, b), (c, d)]
        pats['two pairs across the bridges'] = [(a, c), (b, d)]
        pats['one card to two'] = [(a, b), (a, c)]
    return [tuple(g) for g in groups], pats

#@save
def _sync_pair(i, j):
    torch.cuda.synchronize(i)
    torch.cuda.synchronize(j)

#@save
def _staged_copy_s(i, j, nbytes, copies, repeats=5):
    """(median, MAD) seconds per copy_ of nbytes from GPU i to GPU j (wall
    clock, both cards synchronized)."""
    src = torch.ones(nbytes, dtype=torch.uint8, device=i)
    dst = torch.empty(nbytes, dtype=torch.uint8, device=j)
    dst.copy_(src)  # warm: the first copy sets up the staging path
    _sync_pair(i, j)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(copies):
            dst.copy_(src)
        _sync_pair(i, j)
        times.append((time.perf_counter() - t) / copies)
    return d2l._hw_stats(times)

#@save
def _copies_for(nbytes, target_s=0.05, rate=20e9):
    """Back-to-back copies of nbytes for about target_s at rate bytes/s."""
    return max(1, int(target_s * rate / nbytes))

#@save
def _concurrent_staged(pairs, nbytes, repeats=5):
    """(median, MAD) aggregate GB/s of the pairs (src, dst) copying at
    once, each from its own stream."""
    import threading
    copies = _copies_for(nbytes)
    bufs = {p: (torch.ones(nbytes, dtype=torch.uint8, device=p[0]),
                torch.empty(nbytes, dtype=torch.uint8, device=p[1]))
            for p in pairs}

    def once():
        barrier, spans = threading.Barrier(len(pairs)), {}

        def work(p):
            src, dst = bufs[p]
            with torch.cuda.device(p[0]), torch.cuda.stream(
                    torch.cuda.Stream(p[0])):
                dst.copy_(src)
                _sync_pair(*p)
                barrier.wait()
                t = time.perf_counter()
                for _ in range(copies):
                    dst.copy_(src)
                _sync_pair(*p)
                spans[p] = (t, time.perf_counter())
        threads = [threading.Thread(target=work, args=(p,)) for p in pairs]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = (max(b for _, b in spans.values())
                - min(a for a, _ in spans.values()))
        return len(pairs) * copies * nbytes / wall / 1e9
    return d2l._hw_stats([once() for _ in range(repeats)])

#@save
def _p2p_test():
    """(parsed matrices, raw text) of p2pBandwidthLatencyTest, or ({}, '')
    without the binary."""
    exe = d2l.hw_bin('p2pBandwidthLatencyTest')
    if exe is None:
        return {}, ''
    run = d2l.hw_run([exe], timeout_s=_HW_P2P_TIMEOUTS['p2p_test'])
    return (parse_p2p(run.stdout) if not run.refused else {},
            _hw_scrub(run.stdout))

#@save
def _gpu_matrix_compute(nbytes, lat_bytes=4):
    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError('the GPU matrix needs two GPUs or more')
    rel = _smi_relations()
    gbs, mad = np.full((n, n), np.nan), np.full((n, n), np.nan)
    lat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                t, m = _staged_copy_s(i, j, nbytes, _copies_for(nbytes))
                gbs[i, j], mad[i, j] = nbytes / t / 1e9, m / t
                lat[i, j] = _staged_copy_s(i, j, lat_bytes, 100)[0] * 1e6
    groups, pats = _bridge_pairs(rel, n)
    conc = pd.DataFrame([
        dict(pattern=k, pairs=', '.join(f'{a}->{b}' for a, b in v),
             gbs=s[0], mad_gbs=s[1])
        for k, v in pats.items() for s in [_concurrent_staged(v, nbytes)]])
    peer = np.array([[int(i != j and torch.cuda.can_device_access_peer(i, j))
                      for j in range(n)] for i in range(n)])
    p2p, raw = _p2p_test()
    return dict(gbs=gbs, mad_rel=mad, lat_us=lat, peer=peer,
                relation=[[rel(i, j) if i != j else 'X' for j in range(n)]
                          for i in range(n)],
                groups=groups, concurrent=conc, p2p_test=p2p,
                p2p_test_raw=raw, nbytes=nbytes, lat_bytes=lat_bytes)

#@save
def _offdiag(m):
    m = np.asarray(m, dtype=float)
    return m[~np.eye(len(m), dtype=bool)]

#@save
def _show_gpu_matrix(r):
    g, lat = _offdiag(r['gbs']), _offdiag(r['lat_us'])
    peer = 'refused' if not _offdiag(r['peer']).any() else 'available'
    print(f"peer access between the GPUs: {peer}")
    print(f"{d2l._hw_bytes(r['nbytes'])} copies GPU to GPU: "
          f"{g.min():.0f}-{g.max():.0f} GB/s (median {np.median(g):.0f}); "
          f"{r['lat_bytes']} B copies: {lat.min():.0f}-{lat.max():.0f} us")
    for _, c in r['concurrent'].iterrows():
        print(f'{c.pattern} ({c.pairs}): {c.gbs:.0f} GB/s together')
    test = r['p2p_test'].get('Unidirectional P2P=Disabled Bandwidth Matrix '
                             '(GB/s)')
    if test is not None:
        t = _offdiag(test)
        print(f'p2pBandwidthLatencyTest (P2P disabled): {t.min():.0f}-'
              f'{t.max():.0f} GB/s')

#@save
def _gpu_matrix_live(r, same):
    n = torch.cuda.device_count()
    if n < 2:
        return 'needs two GPUs: the committed box run is shown'
    rel = _smi_relations()
    pairs = [(0, 1)] + [(0, j) for j in range(2, n)
                        if rel(0, j) != rel(0, 1)][:1]
    nbytes = int(r['nbytes'])
    pts = []
    for i, j in pairs:
        t = _staged_copy_s(i, j, nbytes, _copies_for(nbytes))[0]
        pts.append((f'{i}->{j}', r['gbs'][i][j], nbytes / t / 1e9,
                    r['mad_rel'][i][j]))
    return d2l._hw_agreement(pts, same, 'GB/s')

#@save
def hw_gpu_matrix(nbytes=2**28):
    """GB/s of nbytes copies and microseconds of 4-byte copies between
    every ordered pair of GPUs (copy_ between devices; without peer access
    the driver stages them through host memory), the concurrent pairs, the
    bridge groups and p2pBandwidthLatencyTest's matrices. It needs every
    GPU of the host; a normal run shows the committed box result and
    re-measures two pairs."""
    return d2l._hw_result('box-p2p', lambda: _gpu_matrix_compute(nbytes),
                          dict(nbytes=nbytes), entry='gpu-matrix',
                          live_check=_gpu_matrix_live,
                          show=_show_gpu_matrix, live_label='two pairs')
```

## NCCL Sweeps

`nccl_sweep` is one measurement: an nccl-tests binary on the chosen GPUs
(selected by UUID, so the indices are this process's) or, without the
binary, the same sizes through `torch.distributed` in `torchrun` processes,
with the same table columns and a timeout of its own. nccl-tests' first
line names the tool and the NCCL headers it was built against; the library
it runs is the torch wheel's (`libnccl.so.2` first on the library path).
NCCL's own account of a run (`NCCL_DEBUG=INFO`) is taken in a separate
small run, since its lines interleave with the table.

```{.python .input #server-paths-lib-nccl}
%%tab pytorch
#@save
def _gpu_env(gpus, env=''):
    """The environment of a child process on this process's GPUs `gpus`
    (by UUID) with the variables of env ('A=1 B=2')."""
    uuids = d2l._hw_visible_uuids()
    out = dict(CUDA_VISIBLE_DEVICES=','.join(uuids[i] for i in gpus),
               NCCL_DEBUG='WARN')
    out.update(kv.split('=', 1) for kv in env.split())
    return out

#@save
def _sweep_timeout(max_bytes, iters, warmup, min_rate=0.5e9):
    """Five times the expected seconds of a sweep up to max_bytes (sizes
    doubling, so their sum is about 2 max_bytes, at min_rate bytes/s,
    plus a few seconds of setup), at least 60 s."""
    expected = 5 + (iters + warmup) * 2 * max_bytes / min_rate
    return max(60.0, 5 * expected)

#@save
_TORCH_SWEEP = '''
import json, os, sys, time, torch, torch.distributed as dist
op, sizes, iters, warmup, backend = (sys.argv[1], json.loads(sys.argv[2]),
                                     int(sys.argv[3]), int(sys.argv[4]),
                                     sys.argv[5])
k = int(os.environ['WORLD_SIZE'])
if backend == 'nccl':
    dev = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(dev)
    dist.init_process_group(backend, device_id=dev)
else:
    dev = torch.device('cpu')
    dist.init_process_group(backend)
sync = torch.cuda.synchronize if backend == 'nccl' else (lambda: None)
for size in sizes:
    n = max(size // 4 // k, 1) * k  # float32 elements, a multiple of k
    x, y = torch.ones(n, device=dev), torch.empty(n, device=dev)
    part = torch.empty(n // k, device=dev)
    run = {'all_reduce': lambda: dist.all_reduce(x),
           'all_gather': lambda: dist.all_gather_into_tensor(y, part),
           'reduce_scatter': lambda: dist.reduce_scatter_tensor(part, x),
           'alltoall': lambda: dist.all_to_all_single(y, x)}[op]
    for _ in range(warmup):
        run()
    sync()
    dist.barrier()
    t = time.perf_counter()
    for _ in range(iters):
        run()
    sync()
    t = torch.tensor([(time.perf_counter() - t) / iters], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(json.dumps(dict(size=4 * n, time_us=t.item() * 1e6)),
              flush=True)
dist.destroy_process_group()
'''

#@save
def _kill_workers(script):
    """SIGKILL every process of this user running script: torchrun starts
    its workers in sessions of their own, so killing torchrun's process
    group at a timeout leaves them running."""
    import signal
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                args = f.read().split(b'\0')
            if script.encode() in args and os.stat(f'/proc/{pid}').st_uid \
                    == os.getuid():
                os.kill(int(pid), signal.SIGKILL)
        except (OSError, ValueError):
            pass

#@save
def _torchrun_sweep(op, k, sizes, iters, warmup, env, timeout_s,
                    backend='nccl'):
    """(rows, HWRun) of the torch.distributed sweep in k torchrun
    processes: d2l.torchrun's command line through hw_run, so that it has
    a timeout of its own; the workers' output goes to files (--redirects
    3), and a timeout kills them too."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'sweep.py')
        with open(path, 'w') as f:
            f.write(_TORCH_SWEEP)
        launcher = os.path.join(os.path.dirname(sys.executable), 'torchrun')
        try:
            run = d2l.hw_run(
                [launcher, '--standalone', f'--nproc-per-node={k}',
                 '--log-dir', tmp, '--redirects', '3', path, op,
                 json.dumps(list(sizes)), iters, warmup, backend],
                timeout_s=timeout_s, env=env)
        finally:
            _kill_workers(path)
        logs = glob.glob(os.path.join(tmp, '*', 'attempt_0', '0',
                                      'stdout.log'))
        lines = open(logs[0]).read().splitlines() if logs else []
    return [json.loads(l) for l in lines if l.startswith('{')], run

#@save
def _rows_from_times(rows, op, k):
    """nccl-tests columns from (size, time_us) rows (out of place only)."""
    df = pd.DataFrame(rows, columns=['size', 'time_us'])
    out = pd.DataFrame({c: np.nan for c in _NCCL_COLUMNS}, index=df.index)
    out['size'], out['count'] = df['size'], df['size'] // 4
    out['dtype'], out['redop'], out['root'] = 'float', 'sum', -1
    out['oop_us'] = df.time_us
    out['oop_algbw'] = df['size'] / df.time_us / 1e3
    out['oop_busbw'] = out.oop_algbw * d2l.nccl_busbw_factor(op, k)
    return out

#@save
def nccl_sweep(op, gpus, env='', max_bytes=2**28, timeout_s=None,
               min_bytes=8, iters=20, warmup=5, algo='default'):
    """One sweep of the collective op ('all_reduce', 'all_gather',
    'reduce_scatter', 'alltoall') on this process's GPUs `gpus`, message
    sizes min_bytes to max_bytes in factors of 2: nccl-tests' table
    (parse_nccl_tests columns), or the torch.distributed fallback's with
    the same columns. env adds variables ('NCCL_P2P_DISABLE=1'); algo
    other than 'default' sets NCCL_ALGO. attrs: tool, cmd, env, raw output
    and refused (why) for a run that timed out or failed, whose table is
    empty."""
    timeout_s = timeout_s or _sweep_timeout(max_bytes, iters, warmup)
    if algo != 'default':
        env = f'{env} NCCL_ALGO={algo}'.strip()
    child = _gpu_env(gpus, env)
    k, exe = len(gpus), d2l.hw_bin(f'{op}_perf')
    if exe is not None:
        cmd = [exe, '-b', min_bytes, '-e', max_bytes, '-f', 2, '-g', k,
               '-w', warmup, '-n', iters, '-c', 0]
        run = d2l.hw_run(cmd, timeout_s=timeout_s,
                         env=d2l._nccl_tests_env(child))
        df = (parse_nccl_tests(run.stdout) if not run.refused
              else pd.DataFrame(columns=list(_NCCL_COLUMNS)))
        tool = f"nccl-tests {df.attrs.get('version')}"
    else:
        sizes = [int(min_bytes * 2**i) for i in range(
            int(round(math.log2(max_bytes / min_bytes))) + 1)]
        rows, run = _torchrun_sweep(op, k, sizes, iters, warmup, child,
                                    timeout_s)
        df, tool = _rows_from_times(rows, op, k), 'torch.distributed'
    df.attrs.update(tool=tool, env=env, refused=run.refused,
                    cmd=' '.join(os.path.basename(str(c)) for c in run.cmd),
                    raw=_hw_scrub(run.stdout))
    return df

#@save
def _nccl_info(gpus):
    """NCCL's account of one small all-reduce on gpus (NCCL_DEBUG=INFO):
    its version and the transports between the cards ('SHM/direct/direct',
    'P2P/CUMEM', ...), with the transport lines, scrubbed."""
    exe = d2l.hw_bin('all_reduce_perf')
    if exe is None:
        return dict(version='.'.join(map(str, torch.cuda.nccl.version())),
                    transports=[], lines=[])
    env = dict(_gpu_env(gpus), NCCL_DEBUG='INFO')
    run = d2l.hw_run([exe, '-b', 8, '-e', 8, '-g', len(gpus), '-n', 1,
                      '-w', 1, '-c', 0], timeout_s=_HW_P2P_TIMEOUTS['info'],
                     env=d2l._nccl_tests_env(env))
    lines = [_hw_scrub(l[l.index('NCCL INFO'):])  # no host:pid:tid prefix
             for l in run.stdout.splitlines()
             if 'NCCL INFO' in l and (' via ' in l or 'NCCL version' in l)]
    ver = next((re.search(r'NCCL version (\S+)', l).group(1) for l in lines
                if 'NCCL version' in l), None)
    via = sorted({l.split(' via ')[1].strip() for l in lines if ' via ' in l})
    return dict(version=ver, transports=via, lines=lines[:40])

#@save
def _write_raw(name, text):
    """Keep a tool's raw output with a box-p2p regeneration (raw/<name>.txt
    in its folder); nothing outside one."""
    if 'box-p2p' not in d2l._hw_regenerating():
        return
    folder = os.path.join(d2l._hw_regen_dir('box-p2p'), 'raw')
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r'[^A-Za-z0-9_.=-]+', '_', name)
    with open(os.path.join(folder, f'{safe}.txt'), 'w') as f:
        f.write(text)

#@save
def _p2p_check(config):
    if (os.environ.get('D2L_HW_CONFIG') or 'box') != config:
        raise ValueError(f'config {config!r} needs D2L_HW_CONFIG={config} '
                         '(tools/hw_regen.py --config)')
    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if n else ''
    need = {'box': 2, 't3': 8, 't2pair': 2}[config]
    if n < need or (config != 'box' and 'A100' not in name):
        raise RuntimeError(f'config {config} needs {need} GPUs'
                           + ('' if config == 'box' else ' (A100)')
                           + f', not {n} x {name}')

#@save
def _nccl_compute(config):
    spec, n = _HW_P2P[config], torch.cuda.device_count()
    places = _placements(_smi_relations(), n)
    frames = []
    for op, place, algo, env in spec['series']:
        if place == 'pairs':
            todo = {p: g for p, g in places.items() if p != 'all'}
        elif place == 'all':
            todo = {'all': tuple(range(n))}
        else:
            todo = {f'first {place}': tuple(range(place))}
        for path, gpus in todo.items():
            df = nccl_sweep(op, gpus, env=env, algo=algo,
                            max_bytes=spec['max_bytes'], iters=spec['iters'],
                            warmup=spec['warmup'])
            _write_raw(f'{op}-g{len(gpus)}-{path}-{algo}'
                       + (f'-{env}' if env else ''), df.attrs['raw'])
            frames.append(df.assign(op=op, g=len(gpus), path=path,
                                    algo=algo, env=env,
                                    gpus=','.join(map(str, gpus))))
    out = pd.concat(frames, ignore_index=True)
    info = {path: _nccl_info(gpus) for path, gpus in places.items()}
    out.attrs.update(tool=frames[0].attrs['tool'], info=info,
                     nccl=next(iter(info.values()))['version'],
                     max_bytes=spec['max_bytes'])
    return out

#@save
def _show_nccl(nccl):
    via = sorted({t for i in nccl.attrs.get('info', {}).values()
                  for t in i['transports']})
    version = str(nccl.attrs.get('nccl')).split('+')[0]  # no '+cuda12.9'
    print(f"{nccl.attrs.get('tool')}, NCCL {version}; transport between "
          f"the cards: {', '.join(via) or 'not recorded'}")

#@save
def _busbw_at(nccl, op, path, size, algo='default', env=''):
    """(busbw GB/s, relative spread between processes) of one row."""
    hit = ((nccl.op == op) & (nccl.path == path) & (nccl.algo == algo)
           & (nccl.env == env) & (nccl['size'] == size))
    i = int(np.flatnonzero(hit.to_numpy())[0])
    return float(nccl.oop_busbw.iloc[i]), d2l._spread(nccl, 'oop_busbw', i,
                                                      0.05)

#@save
def _nccl_live(r, same):
    n = torch.cuda.device_count()
    if n < 2:
        return 'needs two GPUs: the committed box run is shown'
    places, size = _placements(_smi_relations(), n), int(r.attrs['max_bytes'])
    pts = []
    for path in (next(iter(places)), 'all'):
        if path not in places or path not in set(r.path):
            continue
        gpus = places[path]
        if len(gpus) != int(r.g[r.path == path].iloc[0]):
            continue  # another machine: not the committed placement
        live = nccl_sweep('all_reduce', gpus, max_bytes=size)
        if live.empty:
            return f"all_reduce g = {len(gpus)}: {live.attrs['refused']}"
        committed, noise = _busbw_at(r, 'all_reduce', path, size)
        pts.append((f'g = {len(gpus)} {path}', committed,
                    float(live.oop_busbw.iloc[-1]), noise))
    return (d2l._hw_agreement(pts, same, 'GB/s busbw') if pts else
            'no placement of the committed run on this machine')

#@save
def nccl_sweeps(config='box'):
    """The NCCL sweeps of one configuration of _HW_P2P (the box: all-reduce
    and all-to-all on a pair behind one host bridge, a pair across the two
    bridges and all cards; all-gather and reduce-scatter on all cards;
    all-reduce with NCCL_ALGO=Ring), with NCCL's transports. One long
    table: op, g, path, algo, env, gpus and nccl-tests' columns. It needs
    every GPU of the host; a normal run shows the committed box result and
    re-runs the all-reduce on one pair and on all cards."""
    spec = _HW_P2P[config]
    return d2l._hw_result('box-p2p', lambda: _nccl_compute(config),
                          dict(config=config, series=spec['series'],
                               max_bytes=spec['max_bytes'],
                               iters=spec['iters'], warmup=spec['warmup']),
                          entry='nccl', live_check=_nccl_live,
                          show=_show_nccl, live_label='all-reduce')

#@save
def _host_link_beta(log):
    """The pinned host-to-device plateau (bytes/s) of sec_lm_memory_tiers'
    committed host-link sweep, or None while it is absent."""
    h2d = log.get('h2d') if isinstance(log, dict) else None
    if h2d is None or h2d.empty:
        return None
    return float(d2l.fit_h2d(h2d).loc[('pinned', 'h2d'), 'beta'])

#@save
_OP_NAME = {'all_reduce': 'all-reduce', 'alltoall': 'all-to-all',
            'all_gather': 'all-gather', 'reduce_scatter': 'reduce-scatter'}

#@save
def plot_nccl(nccl, host_link=None):
    """Bus bandwidth against message size, NCCL's default algorithm: the
    all-reduce on every placement and the all-to-all on all cards; dashed,
    one pinned host-to-device copy (host_link: sec_lm_memory_tiers'
    committed result, a line saying so while it is absent)."""
    d = nccl[(nccl.algo == 'default') & (nccl.env == '') & (nccl['size'] > 0)]
    X, Y, legend = [], [], []
    for op in ('all_reduce', 'alltoall'):
        for (path, g), s in d[d.op == op].groupby(['path', 'g'], sort=False):
            if op != 'all_reduce' and path != 'all':
                continue
            X.append(s['size'].to_numpy())  # busbw from the times: the
            Y.append(s['size'].to_numpy() / s.oop_us.to_numpy() / 1e3
                     * d2l.nccl_busbw_factor(op, g))  # printed 0.00 is 0
            legend.append(f"{_OP_NAME[op]}, g = {g}"
                          + ('' if path == 'all' else f', {path}'))
    beta = _host_link_beta(host_link)
    if beta is None:
        print('pinned host copy: sec_lm_memory_tiers\' result is not '
              'committed yet')
    else:
        X.append(np.array([X[0][0], X[0][-1]]))
        Y.append(np.array([beta, beta]) / 1e9)
        legend.append('pinned host copy')
    d2l.plot(X, Y, xlabel='message size (bytes)',
             ylabel='bus bandwidth (GB/s)', legend=legend, xscale='log',
             yscale='log', figsize=(5, 2.8), legend_loc='outside',
             fmts=('-', 'm-', 'g-', 'r-.', 'k--'))
```

## Fits per Collective

`fit_collectives` fits `t = alpha + S / B` to each sweep (sizes above zero)
with `fit_alpha_beta`, then reports alpha, the collective's fixed cost, and
beta = phi(op, k) B, the rate each GPU's link carries (bus bandwidth's
convention, phi = `nccl_busbw_factor`), with 95% intervals from 2,000
bootstrap draws over the sizes (seed 0; each draw one weighted
least-squares solve), the half-bandwidth message size `n_1/2 = alpha B`,
the median relative residual and `alpha_step`, alpha over the ring's number
of steps, so that `coll_time(op, S, k, alpha_step, beta)` reproduces the
fit.

```{.python .input #server-paths-lib-fits}
%%tab pytorch
#@save
def _ring_steps(op, k):
    """Messages each GPU sends in a ring (direct for all-to-all)."""
    return 2 * (k - 1) if op.replace('_', '') == 'allreduce' else k - 1

#@save
def _wls_draws(s, t, draws, rng):
    """(alpha, 1/B) of the weighted least-squares fit (weights 1/t^2) of
    t = alpha + s/B to `draws` resamples of the points, all at once."""
    idx = rng.integers(0, len(s), size=(draws, len(s)))
    S, T = s[idx], t[idx]
    W = 1 / T**2
    a, b, c = W.sum(1), (W * S).sum(1), (W * S * S).sum(1)
    y0, y1 = (W * T).sum(1), (W * S * T).sum(1)
    det = a * c - b * b
    ok = det > 1e-9 * a * c  # a resample with one distinct size is singular
    return ((c * y0 - b * y1) / det)[ok], ((a * y1 - b * y0) / det)[ok]

#@save
def fit_collectives(nccl, algo='default', env='', draws=2000, seed=0):
    """alpha (s) and beta (bytes/s per link, bus-bandwidth convention) with
    95% bootstrap intervals, n_half (message bytes), alpha_step, the
    median relative residual and the number of points, per (op, g, path)
    of a sweep table (nccl_sweeps), of a campaign log holding one
    ('nccl'), or for a pending campaign log an empty table whose
    attrs['status'] is the log's status line."""
    if isinstance(nccl, d2l.CampaignPending):
        out = d2l._hw_pending(nccl.name)
        out.attrs['status'] = d2l.hw_status_line(nccl)
        return out
    if isinstance(nccl, dict):
        nccl = nccl['nccl']
    if nccl.empty:
        return d2l._hw_pending('nccl')
    rng, rows = np.random.default_rng(seed), {}
    d = nccl[(nccl.algo == algo) & (nccl.env == env) & (nccl['size'] > 0)]
    for (op, g, path), x in d.groupby(['op', 'g', 'path'], sort=False):
        s, t = x['size'].to_numpy(float), x.oop_us.to_numpy() / 1e6
        alpha, B, _ = d2l.fit_alpha_beta(s, t)
        phi = d2l.nccl_busbw_factor(op, int(g))
        a_b, inv_b = _wls_draws(s, t, draws, rng)
        rows[(op, int(g), path)] = dict(
            alpha=alpha, alpha_lo=np.percentile(a_b, 2.5),
            alpha_hi=np.percentile(a_b, 97.5), beta=phi * B,
            beta_lo=phi / np.percentile(inv_b, 97.5),
            beta_hi=phi / np.percentile(inv_b, 2.5), n_half=alpha * B,
            alpha_step=alpha / _ring_steps(op, int(g)),
            resid_rel=float(np.median(np.abs(t - alpha - s / B) / t)),
            n_points=len(s), algo=algo, env=env)
    out = pd.DataFrame.from_dict(rows, orient='index')
    out.index = pd.MultiIndex.from_tuples(out.index, names=['op', 'g', 'path'])
    out = out.sort_index()
    out.attrs.update(tool=nccl.attrs.get('tool'), nccl=nccl.attrs.get('nccl'),
                     draws=draws, seed=seed)
    return out

#@save
def show_fits(ab, max_rows=10):
    """The fits as the text reads them: op, g, path, alpha (us), beta
    (GB/s) with its 95% interval, n_1/2 (KiB); for a pending log its
    status line."""
    if ab.empty:
        print(ab.attrs.get('status') or ab.attrs.get('label', 'no fits'))
        return
    show = pd.DataFrame({
        'op': [_OP_NAME.get(i[0], i[0]) for i in ab.index],
        'g': [i[1] for i in ab.index],
        'path': [i[2] for i in ab.index],
        'alpha (us)': ab.alpha.to_numpy() * 1e6,
        'beta (GB/s)': ab.beta.to_numpy() / 1e9,
        '95% interval': [f'{lo / 1e9:.2f}-{hi / 1e9:.2f}' for lo, hi in
                         zip(ab.beta_lo, ab.beta_hi)],
        'n_1/2 (KiB)': ab.n_half.to_numpy() / 2**10})
    d2l.show_table(show, digits={'alpha (us)': 0, 'beta (GB/s)': 2,
                                 'n_1/2 (KiB)': 0}, max_rows=max_rows,
                   width=79)

#@save
_A2A_OP = {'alltoall': 'all_to_all'}  # alpha_beta.json's operation names

#@save
def _transport(info):
    """'shm', 'p2p', 'nvls' or 'net' from an _nccl_info record."""
    via = ' '.join((info or {}).get('transports', []))
    for key, name in (('NVLS', 'nvls'), ('P2P', 'p2p'), ('NET', 'net'),
                      ('SHM', 'shm')):
        if key in via:
            return name
    return None

#@save
def alpha_beta_records(ab, info=None, scope='intra_node',
                       link='pcie_host_staged'):
    """The rows of alpha_beta.json (01_interfaces schema plus proto,
    transport, path and env): alpha_s, beta_Bps per link (bus-bandwidth
    convention), 95% intervals, n_half_bytes, resid_rel, n_points; info
    ({path: _nccl_info record}) gives each path's transport."""
    info, recs = info or {}, []
    for (op, g, path), r in ab.iterrows():
        recs.append(dict(
            scope=scope, link=link, op=_A2A_OP.get(op, op), g=int(g),
            algo=None if r.algo == 'default' else r.algo, proto=None,
            transport=_transport(info.get(path) or info.get('all')),
            path=path, env=r.env or None, alpha_s=float(r.alpha),
            beta_Bps=float(r.beta),
            alpha_ci95=[float(r.alpha_lo), float(r.alpha_hi)],
            beta_ci95=[float(r.beta_lo), float(r.beta_hi)],
            n_half_bytes=float(r.n_half), resid_rel=float(r.resid_rel),
            n_points=int(r.n_points)))
    return recs

#@save
def box_alpha_beta(op='all_reduce', g=None, path=None, algo=None):
    """One row of the box's committed alpha_beta.json (sec_lm_server_paths):
    by default the all-reduce on the most GPUs with NCCL's default
    algorithm. The record's fields plus 'alpha' (s) and 'beta' (bytes/s
    per link)."""
    file = d2l._hw_box_file('box-p2p', 'alpha_beta.json')
    if file is None or not os.path.exists(file):
        raise KeyError('box-p2p has no alpha_beta.json yet')
    with open(file, encoding='utf-8') as f:
        recs = json.load(f)
    op = _A2A_OP.get(op, op)
    hits = [r for r in recs if r['op'] == op and r['algo'] == algo
            and r.get('env') is None and (g is None or r['g'] == g)
            and (path is None or r.get('path') == path)]
    if not hits:
        raise KeyError(f'no alpha-beta row for {op}, g = {g}, {path}')
    r = max(hits, key=lambda r: r['g'])
    return dict(r, alpha=r['alpha_s'], beta=r['beta_Bps'])
```

## The Copy-Engine Knob

The microbenchmark of the workaround (ruling H2): the same 256 MiB
all-reduce on two cards, alternating without and with
`NCCL_SHM_USE_CUDA_MEMCPY=1` (NCCL's shared-memory transport then moves the
data with the copy engines instead of SM loads and stores across PCIe),
three times each. It is never set in a training job: at the torch 2.11 pin
it deadlocks DDP.

```{.python .input #server-paths-lib-copy-engine}
%%tab pytorch
#@save
_KNOB = 'NCCL_SHM_USE_CUDA_MEMCPY=1'

#@save
def _copy_engine_compute(gpus, nbytes, repeats=3):
    arms = {'default': [], 'knob': []}
    for _ in range(repeats):
        for arm, env in (('default', ''), ('knob', _KNOB)):
            df = nccl_sweep('all_reduce', gpus, env=env, min_bytes=nbytes,
                            max_bytes=nbytes, timeout_s=60)
            _write_raw(f'copy-engine-{arm}', df.attrs['raw'])
            arms[arm].append(float(df.oop_busbw.iloc[-1]))
    med = {a: float(np.median(v)) for a, v in arms.items()}
    return dict(gpus=list(gpus), nbytes=nbytes, busbw=arms,
                default=med['default'], knob=med['knob'],
                ratio=med['knob'] / med['default'])

#@save
def _show_copy_engine(r):
    print(f"all-reduce of {d2l._hw_bytes(r['nbytes'])} on GPUs "
          f"{r['gpus'][0]} and {r['gpus'][1]}: {r['default']:.1f} GB/s "
          f"busbw by default")
    print(f"with {_KNOB}: {r['knob']:.1f} GB/s ({r['ratio']:.1f}x)")

#@save
def _copy_engine_live(r, same):
    gpus = tuple(r['gpus'])
    if torch.cuda.device_count() <= max(gpus):
        return f'needs GPUs {gpus[0]} and {gpus[1]}: the box run is shown'
    pts = []
    for arm, env in (('default', ''), ('knob', _KNOB)):
        df = nccl_sweep('all_reduce', gpus, env=env, min_bytes=r['nbytes'],
                        max_bytes=r['nbytes'], timeout_s=60)
        if df.empty:
            return f"{arm}: {df.attrs['refused']}"
        v = np.asarray(r['busbw'][arm], dtype=float)
        pts.append((arm, r[arm], float(df.oop_busbw.iloc[-1]),
                    float(np.median(np.abs(v - np.median(v))) / np.median(v))))
    return d2l._hw_agreement(pts, same, 'GB/s busbw')

#@save
def copy_engine_effect(gpus=(0, 1), nbytes=2**28):
    """busbw of one all-reduce of nbytes on two GPUs, NCCL's default
    against NCCL_SHM_USE_CUDA_MEMCPY=1 (a bare microbenchmark, never a
    training job): the median of three alternating runs each, and their
    ratio."""
    return d2l._hw_result(
        'box-p2p', lambda: _copy_engine_compute(gpus, nbytes),
        dict(gpus=list(gpus), nbytes=nbytes), entry='copy-engine',
        live_check=_copy_engine_live, show=_show_copy_engine,
        live_label='both arms')
```

## The Regeneration Driver

`tools/hw_regen.py box-p2p` runs `regen_p2p` in each of its three processes
with every card of the box locked (a quiet box: nothing else on any card).
It makes the page's calls on the full configuration and keeps the raw
outputs; after the processes are combined, `regen_p2p_finish` writes the
captures as text files and `alpha_beta.json` from the combined sweeps.

```{.python .input #server-paths-lib-driver}
%%tab pytorch
#@save
def regen_p2p(config='box'):
    """Driver of box-p2p (and of the campaigns C-T3 and C-T2pair with
    config 't3' and 't2pair'): the captures, the GPU matrix, the NCCL
    sweeps and, on the box, the copy-engine microbenchmark."""
    _p2p_check(config)
    d2l._hw_result('box-p2p', _hw_capture_all, {}, entry='captures')
    hw_gpu_matrix(nbytes=2**28)
    nccl = nccl_sweeps(config)
    if _HW_P2P[config]['copy_engine']:
        places = _placements(_smi_relations(), torch.cuda.device_count())
        copy_engine_effect(gpus=next(iter(places.values())), nbytes=2**28)
    return nccl

#@save
_CAPTURE_FILES = {'topo': 'topo.txt', 'topo-p2p': 'topo-p2p.txt',
                  'numactl': 'numactl.txt', 'lspci': 'lspci-tv.txt',
                  'pcie': 'pcie.csv'}

#@save
def regen_p2p_finish(folder):
    """After tools/hw_regen.py combined the processes: the captures as
    text files, p2pBandwidthLatencyTest's output, and alpha_beta.json (the
    fits of the combined sweeps, every algorithm and environment)."""
    caps, _ = d2l._hw_read(os.path.join(folder, 'captures.json.gz'))
    for name, file in _CAPTURE_FILES.items():
        with open(os.path.join(folder, file), 'w') as f:
            f.write(caps.get(name, ''))
    mat, _ = d2l._hw_read(os.path.join(folder, 'gpu-matrix.json.gz'))
    with open(os.path.join(folder, 'p2pBandwidthLatencyTest.txt'), 'w') as f:
        f.write(mat.get('p2p_test_raw', ''))
    nccl, _ = d2l._hw_read(os.path.join(folder, 'nccl.json.gz'))
    recs = []
    for (algo, env), _ in nccl.groupby(['algo', 'env'], sort=False):
        recs += alpha_beta_records(fit_collectives(nccl, algo=algo, env=env),
                                   info=nccl.attrs.get('info'))
    with open(os.path.join(folder, 'alpha_beta.json'), 'w') as f:
        json.dump(recs, f, indent=1)
    return recs
```
