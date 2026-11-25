# Accelerator Design: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, and listed first of
chapter 22's lib files so its empty registries exist before the other lib
files add to them). It holds the measurement plumbing of the whole chapter
and the parts of :numref:`sec_lm_accelerator_design` that the section explains
in a sentence but does not show (docs/code-guide.md 7.1):

- the chapter's registries, each filled one statement per entry by the
  section that owns the entry: box regenerations (`HW_ARTIFACTS`, keyed),
  campaign logs produced elsewhere (`HW_CAMPAIGNS`, keyed), static tables
  (`HW_STATIC`, keyed), the regeneration jobs (`HW_JOBS`, one shard, read
  only by `tools/hw_regen.py`), the campaign configurations of a
  regeneration (`HW_CONFIG_LOGS`, the log a `--config` run writes), the
  results the author declared pending
  (`HW_PENDING`), the chapter's new Hub pins (`HW_REVISIONS`) and the
  configurations built in code (`HW_SHAPES`);
- the one place that decides between measuring and loading a result
  (`_hw_result`), the readers of committed results (`hw_box_log`,
  `hw_campaign_log`, `hw_registered`, `hw_status_line`, `_hw_static`) and
  their storage format;
- the measurement environment (`_hw_env`: versions, a hashed host name,
  GPU UUID prefixes, foreign processes on the cards);
- the tool runner (`hw_bin`, `hw_run`, with per-call timeouts), the model
  configurations at their pins (`hw_config`) and the table display
  (`show_table`);
- the accelerator-design section's registry entries.

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble inside its functions. No function body names an artifact
key or a short name it does not itself read (the freshness scan follows
string literals to registry entries).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_accelerator_design`

## Registries

```{.python .input #accelerator-design-lib-registries}
%%tab pytorch
#@save
HW_ARTIFACTS = {}  # box regeneration name: (DATA_HUB key, kind)

#@save
HW_CAMPAIGNS = {}  # log name: (DATA_HUB key, kind, campaign, hardware)

#@save
HW_STATIC = {}  # static table name: (DATA_HUB key, kind)

#@save
HW_JOBS = {}  # regeneration name: (driver name in d2l, zero-argument kwargs)

#@save
HW_CONFIG_LOGS = {}  # (regeneration name, config): the campaign log it writes

#@save
HW_PENDING = frozenset()  # box results the author declared pending

#@save
HW_REVISIONS = {}  # Hub model id: pinned revision (chapter 22's new pins)

#@save
HW_SHAPES = {}  # name: a model configuration built in code (no download)
```

Chapter 22's Hub pins (Apache-2.0, licence field checked 2026-11-25). The
Qwen3-0.6B, 1.7B and 8B pins are chapter 18's (`HF_REVISIONS`).

```{.python .input #accelerator-design-lib-revisions}
%%tab pytorch
#@save
d2l.HW_REVISIONS['Qwen/Qwen3-30B-A3B'] = (
    'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39')
d2l.HW_REVISIONS['Qwen/Qwen3-235B-A22B'] = (
    '8efa61729e24bd65b1d152b5ab5409052aa80e65')
```

```{.python .input #accelerator-design-lib-shape}
%%tab pytorch
#@save
class HWShape(dict):
    """A model configuration as a dict with attribute access and to_dict,
    so it reads like a Transformers configuration (config_memory,
    flops_per_token) without a download."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def to_dict(self):
        return dict(self)
```

The "70B configuration" of the chapter (decision Q17): Qwen3's architecture at
Llama-3-70B's shape, 70.6B parameters.

```{.python .input #accelerator-design-lib-shapes}
%%tab pytorch
#@save
d2l.HW_SHAPES['dense-70b'] = d2l.HWShape(
    model_type='qwen3', architectures=['Qwen3ForCausalLM'],
    hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
    num_key_value_heads=8, head_dim=128, intermediate_size=28672,
    vocab_size=128256, tie_word_embeddings=False,
    max_position_embeddings=131072, rope_theta=500000.0,
    rms_norm_eps=1e-5, hidden_act='silu', attention_bias=False,
    torch_dtype='bfloat16')
```

The accelerator-design section's entries: its regeneration, its static table
and the A100 campaign (C-T2) logs. The `DATA_HUB` entries follow the uploads.

```{.python .input #accelerator-design-lib-entries}
%%tab pytorch
#@save
d2l.HW_ARTIFACTS['box-microbench'] = ('lm-hw-buildbox-microbench-v1', 'tar')
d2l.HW_STATIC['chip-specs'] = ('lm-hw-chip-specs-v1', 'csv')
d2l.HW_CAMPAIGNS['a100-microbench'] = (
    'lm-hw-a100-microbench-v1', 'tar', 'C-T2', 'A100 PCIe 40 GB')
d2l.HW_CAMPAIGNS['a100-clocksweep'] = (
    'lm-hw-a100-clocksweep-v1', 'tar', 'C-T2', 'A100 PCIe 40 GB')
```

```{.python .input #accelerator-design-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-microbench'] = (
    'regen_microbench', lambda: dict(config='box'))
```

The same driver on the A100 of campaign C-T2 (`tools/hw_regen.py
box-microbench --config t2`) writes the campaign's log instead of the box
result.

```{.python .input #accelerator-design-lib-config-logs}
%%tab pytorch
#@save
d2l.HW_CONFIG_LOGS[('box-microbench', 't2')] = 'a100-microbench'
```

## Storage of Committed Results

A regeneration writes each entry of a result as `<entry>.json.gz` in a folder
named after the next version of its key (`../data/lm/rerun/ch22/<key>/`);
`tools/hw_regen.py` packs the folder into `<key>.tar` (kind `tar`) or the one
table into `<key>.jsonl.gz` (kind `jsonl`).

```{.python .input #accelerator-design-lib-io}
%%tab pytorch
#@save
_HW_EXT = {'tar': '.tar', 'jsonl': '.jsonl.gz', 'csv': '.csv',
           'json': '.json'}

#@save
def _hw_rerun_root():
    """Where regenerations write (relative to a chapter folder); D2L_HW_OUT
    overrides it (the campaign launcher)."""
    return os.environ.get('D2L_HW_OUT') or os.path.join(
        '..', 'data', 'lm', 'rerun', 'ch22')

#@save
def _hw_encode(x):
    """x as JSON values; DataFrames, arrays, tuples and namedtuples tagged
    so that _hw_decode restores their type."""
    if isinstance(x, pd.DataFrame):
        df, index = x, None
        if not isinstance(df.index, pd.RangeIndex) or df.index.name:
            index = [n if n is not None else f'level_{i}'
                     for i, n in enumerate(df.index.names)]
            df = df.copy()
            df.index.names = index
            df = df.reset_index()
        doc = json.loads(df.to_json(orient='split', index=False,
                                    double_precision=15, date_format='iso'))
        return {'__frame__': dict(columns=doc['columns'], data=doc['data'],
                                  index=index,
                                  attrs=_hw_encode(dict(x.attrs)))}
    if isinstance(x, pd.Series):
        return _hw_encode(x.to_frame())
    if hasattr(x, '_asdict') and hasattr(x, '_fields'):
        return {'__namedtuple__': type(x).__name__,
                'fields': {k: _hw_encode(v) for k, v in x._asdict().items()}}
    if isinstance(x, np.ndarray):
        return {'__array__': x.tolist(), 'dtype': str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): _hw_encode(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return {'__tuple__': [_hw_encode(v) for v in x]}
    if isinstance(x, list):
        return [_hw_encode(v) for v in x]
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, torch.Tensor):
        return {'__array__': x.detach().cpu().tolist(),
                'dtype': str(x.dtype).replace('torch.', '')}
    return x

#@save
def _hw_decode(x):
    """The inverse of _hw_encode."""
    if isinstance(x, list):
        return [_hw_decode(v) for v in x]
    if not isinstance(x, dict):
        return x
    if '__frame__' in x:
        f = x['__frame__']
        df = pd.DataFrame(f['data'], columns=f['columns'])
        if f['index']:
            df = df.set_index(f['index'])
            df.index.names = [None if n.startswith('level_') else n
                              for n in f['index']]
        df.attrs.update(_hw_decode(f['attrs']))
        return df
    if '__namedtuple__' in x:
        fields = {k: _hw_decode(v) for k, v in x['fields'].items()}
        cls = getattr(d2l, x['__namedtuple__'], None)
        if not (isinstance(cls, type) and getattr(cls, '_fields', None)
                == tuple(fields)):
            cls = collections.namedtuple(x['__namedtuple__'], list(fields))
        return cls(**fields)
    if '__array__' in x:
        return np.array(x['__array__'], dtype=x['dtype'])
    if '__tuple__' in x:
        return tuple(_hw_decode(v) for v in x['__tuple__'])
    return {k: _hw_decode(v) for k, v in x.items()}

#@save
def _hw_write(path, result, meta):
    """Write one entry (result with meta = {config, env, label, ...}) as
    gzipped JSON, through a temporary file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        json.dump(dict(meta=_hw_encode(meta), result=_hw_encode(result)), f)
    os.replace(tmp, path)

#@save
def _hw_read(path):
    """(result, meta) of an entry written by _hw_write."""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        doc = json.load(f)
    return _hw_decode(doc['result']), _hw_decode(doc['meta'])

#@save
def _hw_entry_file(entry):
    return f'{entry or "result"}.json.gz'

#@save
def _hw_unpack(path, kind):
    """The folder of a registered tar (extracted once, next to it), or the
    one-table folder of a jsonl artifact (converted once)."""
    folder = os.path.join(os.path.dirname(path), 'extracted')
    done = os.path.join(folder, '.complete')
    if os.path.exists(done):
        return folder
    os.makedirs(folder, exist_ok=True)
    if kind == 'tar':
        with tarfile.open(path) as t:
            t.extractall(folder, filter='data')
    elif kind == 'jsonl':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            recs = [json.loads(line) for line in f]
        meta = _hw_decode(recs.pop(0)['_meta'])
        _hw_write(os.path.join(folder, _hw_entry_file(None)),
                  pd.DataFrame(recs), meta)
    else:
        raise ValueError(f'kind {kind!r} has no entries')
    open(done, 'w').close()
    return folder

#@save
def _hw_pack(folder, kind, out):
    """Pack a regeneration folder into its artifact file out (tar: every
    file under folder; jsonl: the one table, meta first)."""
    tmp = f'{out}.{os.getpid()}.tmp'
    if kind == 'tar':
        with tarfile.open(tmp, 'w') as t:
            for root, _, files in sorted(os.walk(folder)):
                for name in sorted(files):
                    p = os.path.join(root, name)
                    info = t.gettarinfo(p, os.path.relpath(p, folder))
                    info.mtime, info.uid, info.gid = 0, 0, 0
                    info.uname = info.gname = ''
                    with open(p, 'rb') as f:
                        t.addfile(info, f)
    elif kind == 'jsonl':
        result, meta = _hw_read(os.path.join(folder, _hw_entry_file(None)))
        lines = [json.dumps({'_meta': _hw_encode(meta)})] + [
            json.dumps(_hw_encode(rec)) for rec in result.to_dict('records')]
        with open(tmp, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb',
                                                   mtime=0) as f:
            f.write(('\n'.join(lines) + '\n').encode('utf-8'))
    else:
        raise ValueError(f'cannot pack kind {kind!r}')
    os.replace(tmp, out)
    return out
```

A regeneration runs its driver in three processes, one after the other
(`tools/hw_regen.py`); the headline of every measured number is the median of
the per-process medians, and a table keeps the spread between the processes
(`attrs['spread']`), the run-to-run noise that a gap between two measured
numbers has to exceed.

```{.python .input #accelerator-design-lib-combine}
%%tab pytorch
#@save
def _hw_combine(results):
    """One result from the same result of several processes: numbers that
    differ become their median (elementwise in tables, arrays, lists and
    the fields of dicts and namedtuples); everything else must agree. A
    table records, per numeric column that differs, the relative spread
    (max - min) / median of each row in attrs['spread']."""
    first = results[0]
    if isinstance(first, pd.DataFrame):
        for r in results[1:]:
            if not (isinstance(r, pd.DataFrame) and r.shape == first.shape
                    and list(r.columns) == list(first.columns)
                    and r.index.equals(first.index)):
                raise ValueError('tables differ in shape between processes')
        out, spread = first.copy(), {}
        for c in first.columns:
            cols = [r[c].to_numpy() for r in results]
            if all(pd.Series(x).equals(pd.Series(cols[0])) for x in cols):
                continue
            if (pd.api.types.is_numeric_dtype(first[c])
                    and not pd.api.types.is_bool_dtype(first[c])):
                x = np.stack(cols).astype(float)
                out[c] = med = np.median(x, axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    spread[c] = ((x.max(0) - x.min(0)) / np.abs(med)).tolist()
            else:
                raise ValueError(f'column {c} differs between processes')
        out.attrs = dict(first.attrs, n_proc=len(results), spread=spread)
        return out
    if hasattr(first, '_fields'):
        return type(first)(*[_hw_combine([getattr(r, f) for r in results])
                             for f in first._fields])
    if isinstance(first, dict):
        return {k: _hw_combine([r[k] for r in results]) for k in first}
    if isinstance(first, (list, tuple)) and all(
            len(r) == len(first) for r in results):
        return type(first)(_hw_combine([r[i] for r in results])
                           for i in range(len(first)))
    if isinstance(first, np.ndarray) and first.dtype.kind in 'iuf':
        return np.median(np.stack(results).astype(float), axis=0)
    if isinstance(first, (int, float, np.integer, np.floating)) and not (
            isinstance(first, (bool, np.bool_))):
        same = all(r == first for r in results)
        return first if same else float(np.median(results))
    return first
```

## The Measurement Environment

```{.python .input #accelerator-design-lib-env}
%%tab pytorch
#@save
def _hw_smi(*query):
    """Rows of `nvidia-smi --query-...` as lists of strings ([] if
    nvidia-smi is absent or fails)."""
    try:
        out = subprocess.run(['nvidia-smi', *query, '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return [[c.strip() for c in line.split(',')]
            for line in out.stdout.splitlines() if line.strip()]

#@save
def _hw_visible_uuids():
    """UUIDs of the GPUs this process sees (CUDA_VISIBLE_DEVICES applied)."""
    if not torch.cuda.is_available():
        return []
    return ['GPU-' + str(torch.cuda.get_device_properties(i).uuid)
            for i in range(torch.cuda.device_count())]

#@save
def _hw_compute_procs():
    """(gpu uuid, pid, foreign) for every compute process on the host; a
    process is foreign when another user owns it (or its owner cannot be
    read)."""
    procs = []
    for row in _hw_smi('--query-compute-apps=gpu_uuid,pid'):
        if len(row) < 2 or not row[1].isdigit():
            continue
        try:
            with open(f'/proc/{row[1]}/status') as f:
                uid = next(int(line.split()[1]) for line in f
                           if line.startswith('Uid:'))
            foreign = uid != os.getuid()
        except (OSError, StopIteration, ValueError):
            foreign = True
        procs.append((row[0], int(row[1]), foreign))
    return procs

#@save
def _hw_contention():
    """(foreign processes on this process's GPUs, compute processes of
    anyone else on the other GPUs of the host)."""
    mine = set(_hw_visible_uuids())
    procs = [p for p in _hw_compute_procs() if p[1] != os.getpid()]
    foreign = sum(1 for u, _, f in procs if f and u in mine)
    neighbours = sum(1 for u, _, _ in procs if u not in mine)
    return foreign, neighbours

#@save
def _hw_env():
    """The software and hardware a measurement ran on: versions, the GPU
    model and count, 8-character GPU UUID prefixes and a sha1 prefix of the
    host name (never the name itself)."""
    import platform
    env = dict(host=hashlib.sha1(platform.node().encode()).hexdigest()[:8],
               python=platform.python_version(), torch=torch.__version__,
               cuda=torch.version.cuda)
    try:
        import triton
        env['triton'] = triton.__version__
    except ImportError:
        env['triton'] = None
    if torch.cuda.is_available():
        env['gpu'] = torch.cuda.get_device_name(0)
        env['n_gpu'] = torch.cuda.device_count()
        env['uuids'] = [u[4:12] for u in _hw_visible_uuids()]
        env['nccl'] = '.'.join(map(str, torch.cuda.nccl.version()))
        rows = _hw_smi('--query-gpu=driver_version')
        env['driver'] = rows[0][0] if rows else None
    else:
        env['gpu'], env['n_gpu'] = None, 0
    try:
        git = subprocess.run(['git', 'rev-parse', '--short=12', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        env['git'] = git.stdout.strip() if git.returncode == 0 else None
    except OSError:
        env['git'] = None
    return env

#@save
def _gpu_short(name):
    """'NVIDIA GeForce RTX 4090' -> 'RTX 4090'."""
    for prefix in ('NVIDIA GeForce ', 'NVIDIA '):
        if name and name.startswith(prefix):
            return name[len(prefix):]
    return name
```

## Committed Results: Measure or Load

```{.python .input #accelerator-design-lib-results}
%%tab pytorch
#@save
def _hw_regenerating():
    """The chapter's regeneration names that D2L_RERUN lists. D2L_RERUN=1
    (every regeneration of the chapter, four-card jobs and a 61 GB download
    among them) is refused."""
    if os.environ.get('D2L_RERUN', '').strip() == '1':
        raise ValueError('D2L_RERUN=1 would run every regeneration of this '
                         'chapter; run tools/hw_regen.py with one of: '
                         + ', '.join(d2l.HW_ARTIFACTS))
    return d2l.lm_rerun(tuple(d2l.HW_ARTIFACTS))

#@save
def _hw_regen_key(name):
    """(key, kind) a regeneration of name writes: its box artifact, or with
    D2L_HW_CONFIG (tools/hw_regen.py --config) the campaign log that
    HW_CONFIG_LOGS names for that configuration."""
    config = os.environ.get('D2L_HW_CONFIG')
    if not config:
        return d2l.HW_ARTIFACTS[name]
    log = d2l.HW_CONFIG_LOGS.get((name, config))
    if log is None:
        raise KeyError(f'{name} has no campaign configuration {config!r}')
    return d2l.HW_CAMPAIGNS[log][:2]

#@save
def _hw_regen_dir(name):
    """The folder a regeneration of name writes (the next version of its
    key, _hw_regen_key); D2L_HW_PROC (set by tools/hw_regen.py) selects a
    process's own subfolder."""
    key, _ = _hw_regen_key(name)
    folder = os.path.join(_hw_rerun_root(), d2l._next_key(key))
    proc = os.environ.get('D2L_HW_PROC')
    return os.path.join(folder, f'proc{proc}') if proc else folder

#@save
def _hw_folder(name):
    """The folder holding name's committed entries: the registered
    artifact, else (in a regeneration run) the folder an earlier
    regeneration wrote, else None. A tools/hw_regen.py process
    (D2L_HW_PROC set) that reads another, unregistered result raises: an
    unchecked or stale folder must not end up inside a new artifact."""
    key, kind = d2l.HW_ARTIFACTS[name]
    if key in d2l.DATA_HUB:
        with d2l._quiet():
            path = d2l.download_keyed(key)
        return _hw_unpack(path, kind)
    regen = _hw_regenerating()
    if (os.environ.get('D2L_HW_PROC') is not None and name not in regen
            and name not in d2l.HW_PENDING):
        raise RuntimeError(f'{name} is not registered; register it before a '
                           'regeneration that reads it')
    if regen:
        folder = os.path.join(_hw_rerun_root(), d2l._next_key(key))
        if os.path.isdir(folder):
            return folder
    return None

#@save
def _hw_pending(name, empty=None):
    """The labelled stand-in for a result that is not there yet: empty()
    if given, else an empty DataFrame, labelled '<name> pending'."""
    result = empty() if empty is not None else pd.DataFrame()
    if hasattr(result, 'attrs'):
        result.attrs['label'] = f'{name} pending'
    return result

#@save
def _hw_load(name, entry=None):
    """(result, meta) of a committed entry, or None while it is absent."""
    folder = _hw_folder(name)
    if folder is None:
        return None
    path = os.path.join(folder, _hw_entry_file(entry))
    if not os.path.exists(path):
        raise KeyError(f'{name} has no entry {entry!r}')
    return _hw_read(path)

#@save
def _hw_live(live_check, result, meta, label=''):
    """Print the one agreement line of a live check, naming what it checks
    (label): compared when this GPU is the committed run's GPU, else this
    GPU's values uncompared. A live check never fails a run and never
    changes the committed numbers."""
    if live_check is None or not torch.cuda.is_available():
        return
    gpu = torch.cuda.get_device_name(0)
    same = gpu == (meta.get('env') or {}).get('gpu')
    try:
        with d2l._quiet():
            text = live_check(result, same)
    except Exception as e:  # report, never fail the page
        text = f'not run ({type(e).__name__})'
    what = f' ({label})' if label else ''
    print(f'live {"check" if same else "values"}{what} on {_gpu_short(gpu)}: '
          f'{text}')

#@save
def _hw_result(name, compute, config, *, entry=None, live_check=None,
               show=None, empty=None, live_label=''):
    """The result of regeneration name (an HW_ARTIFACTS name; entry names
    one part of a multi-part result) for the call's configuration (a
    JSON-able dict of its arguments).

    - D2L_RERUN names it: compute() runs and its result is written, with
      config and the measurement environment, under the next version of the
      key (tools/hw_regen.py runs this; never a notebook).
    - Else the committed entry, whose stored configuration must equal
      config (ValueError naming the first difference), then show(result)
      and the one line of live_check(result, same_gpu), headed by
      live_label.
    - Else, in a regeneration run or when the author declared name pending
      (HW_PENDING), the labelled pending result (_hw_pending); else
      KeyError."""
    assert name in d2l.HW_ARTIFACTS, name
    config, rerun = d2l._jsonable(config), _hw_regenerating()
    if name in rerun:
        f0, n0 = _hw_contention()
        t = time.time()
        result = compute()
        f1, n1 = _hw_contention()
        env = dict(_hw_env(), contended=bool(f0 or f1),
                   neighbours_busy=bool(n0 or n1),
                   seconds=round(time.time() - t, 1))
        meta = dict(name=name, entry=entry, config=config, env=env,
                    run=os.environ.get('D2L_HW_CONFIG') or 'box',
                    label=getattr(result, 'attrs', {}).get('label', ''))
        _hw_write(os.path.join(_hw_regen_dir(name), _hw_entry_file(entry)),
                  result, meta)
        if show is not None:
            show(result)
        return result
    loaded = _hw_load(name, entry)
    if loaded is not None:
        result, meta = loaded
        d = d2l._config_diff(config, meta.get('config'))
        if d:
            key = d2l.HW_ARTIFACTS[name][0]
            raise ValueError(f'page configuration differs from the committed '
                             f'run {key}: {d[0]} {d[1]!r} != {d[2]!r}')
        if show is not None:
            show(result)
        _hw_live(live_check, result, meta, live_label)
        return result
    if rerun or name in d2l.HW_PENDING:
        return _hw_pending(name, empty)
    raise KeyError(f'{name} is not committed yet: run tools/hw_regen.py '
                   f'{name}')

#@save
def hw_box_log(name, entry=None):
    """A committed box result (an HW_ARTIFACTS name) for a section that
    reads another section's measurement: the entry's result, or with
    entry=None a dict of every entry of a multi-part result. No
    configuration check and no live check; absent as in _hw_result."""
    assert name in d2l.HW_ARTIFACTS, name
    folder = _hw_folder(name)
    if folder is None:
        if _hw_regenerating() or name in d2l.HW_PENDING:
            return _hw_pending(name)
        raise KeyError(f'{name} is not committed yet: run tools/hw_regen.py '
                       f'{name}')
    path = os.path.join(folder, _hw_entry_file(entry))
    if entry is not None or os.path.exists(path):
        return _hw_read(path)[0]
    return {f[:-len('.json.gz')]: _hw_read(os.path.join(folder, f))[0]
            for f in sorted(os.listdir(folder)) if f.endswith('.json.gz')}

#@save
def _hw_box_file(name, filename):
    """Local path of a raw file (topo.txt, alpha_beta.json, ...) stored
    with a committed box result, or None while the result is absent."""
    folder = _hw_folder(name)
    return None if folder is None else os.path.join(folder, filename)

#@save
def hw_registered(name):
    """True when the artifact of a chapter name (box result, campaign log
    or static table) is registered in DATA_HUB."""
    for registry in (d2l.HW_ARTIFACTS, d2l.HW_CAMPAIGNS, d2l.HW_STATIC):
        if name in registry:
            return registry[name][0] in d2l.DATA_HUB
    raise KeyError(f'{name} is not a chapter-22 name')
```

## Campaign Logs and Static Tables

```{.python .input #accelerator-design-lib-campaigns}
%%tab pytorch
#@save
class CampaignPending(collections.namedtuple(
        'CampaignPending', ['name', 'campaign', 'hardware'])):
    """What hw_campaign_log returns while a campaign's log is not
    registered: false in a test, and it names what is missing."""
    def __bool__(self):
        return False

#@save
def hw_campaign_log(name):
    """The committed log of a campaign run elsewhere (an HW_CAMPAIGNS
    name): the dict of its entries (tar), its table (jsonl) or its JSON;
    while the log is not registered, a CampaignPending (false), which is
    the normal state of a campaign log, not an error."""
    if name not in d2l.HW_CAMPAIGNS:
        raise KeyError(f'unknown campaign log {name!r}; one of '
                       f'{sorted(d2l.HW_CAMPAIGNS)}')
    key, kind, campaign, hardware = d2l.HW_CAMPAIGNS[name]
    if key not in d2l.DATA_HUB:
        return CampaignPending(name, campaign, hardware)
    with d2l._quiet():
        path = d2l.download_keyed(key)
    if kind == 'json':
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    folder = _hw_unpack(path, kind)
    entries = {f[:-len('.json.gz')]: _hw_read(os.path.join(folder, f))[0]
               for f in sorted(os.listdir(folder)) if f.endswith('.json.gz')}
    return entries.get('result', entries) if kind == 'jsonl' else entries

#@save
def hw_status_line(log):
    """The one line a display prints for a campaign log that is not
    committed yet (empty for a committed log)."""
    if not isinstance(log, CampaignPending):
        return ''
    if log.campaign.startswith('chapter'):
        return f'{log.name}: {log.campaign}\'s measurement, not committed yet'
    return f'{log.hardware}: campaign {log.campaign} not run yet'

#@save
def _hw_static(name):
    """A registered static table of the chapter (HW_STATIC): a DataFrame
    (csv), a dict (json) or a folder (tar)."""
    if name not in d2l.HW_STATIC:
        raise KeyError(f'unknown table {name!r}')
    key, kind = d2l.HW_STATIC[name]
    if key not in d2l.DATA_HUB:
        raise KeyError(f'{name} is not committed yet')
    with d2l._quiet():
        path = d2l.download_keyed(key)
    if kind == 'csv':
        return pd.read_csv(path)
    if kind == 'json':
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    folder = os.path.join(os.path.dirname(path), 'extracted')
    if not os.path.isdir(folder):
        with tarfile.open(path) as t:
            t.extractall(folder, filter='data')
    return folder
```

## Configurations, Tools and Tables

```{.python .input #accelerator-design-lib-config}
%%tab pytorch
#@save
def hw_config(model_id):
    """The text configuration of a Hub model at its pinned revision
    (HF_REVISIONS, else HW_REVISIONS): configuration JSON only, no
    weights."""
    revision = d2l.HF_REVISIONS.get(model_id) or d2l.HW_REVISIONS.get(
        model_id)
    if revision is None:
        raise KeyError(f'{model_id} is not pinned')
    from transformers import AutoConfig
    with d2l._quiet():
        return AutoConfig.from_pretrained(model_id, revision=revision)
```

```{.python .input #accelerator-design-lib-tools}
%%tab pytorch
#@save
HWRun = collections.namedtuple(
    'HWRun', ['cmd', 'returncode', 'stdout', 'stderr', 'seconds', 'refused'])

#@save
def hw_bin(tool):
    """Path of a benchmark tool built by tools/build_hw_tools.sh (D2L_HW_BIN,
    else ../data/hw-tools/bin), else on PATH, else None."""
    folder = os.environ.get('D2L_HW_BIN') or os.path.join(
        '..', 'data', 'hw-tools', 'bin')
    return shutil.which(tool, path=folder) or shutil.which(tool)

#@save
def _nccl_tests_env(env=None):
    """The environment for the nccl-tests binaries: the torch wheel's NCCL
    (libnccl.so.2) first on the library path."""
    import nvidia.nccl
    lib = os.path.join(list(nvidia.nccl.__path__)[0], 'lib')
    env = dict(env or {})
    path = env.get('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH', ''))
    env['LD_LIBRARY_PATH'] = lib + (os.pathsep + path if path else '')
    return env

#@save
def hw_run(cmd, timeout_s, env=None):
    """Run a command (a list) with a timeout and return an HWRun record.
    On timeout the whole process group is killed and the record says why
    (refused='timeout after N s'); a missing binary or a non-zero exit is
    a refusal too. In a regeneration run a refusal raises instead (a
    timeout fails the run). env adds variables to the current ones."""
    t = time.time()
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
            env=dict(os.environ, **(env or {})))
    except OSError as e:
        run = HWRun(list(cmd), None, '', str(e), 0.0,
                    f'cannot start: {type(e).__name__}')
    else:
        try:
            out, err = proc.communicate(timeout=timeout_s)
            refused = (None if proc.returncode == 0 else
                       f'exit status {proc.returncode}')
        except subprocess.TimeoutExpired:
            d2l._kill_session(proc)
            out, err = proc.communicate()
            refused = f'timeout after {timeout_s:g} s'
        run = HWRun(list(cmd), proc.returncode, out, err,
                    round(time.time() - t, 2), refused)
    if run.refused and _hw_regenerating():
        raise RuntimeError(f'{os.path.basename(str(cmd[0]))}: {run.refused}'
                           f'\n{run.stderr[-2000:]}')
    return run
```

```{.python .input #accelerator-design-lib-table}
%%tab pytorch
#@save
def _hw_fmt(x, digits=None):
    """A number as the prose quotes it: digits decimals if given, else 3
    significant digits, thousands separated (integers, and whole floats
    from 100 up, exact); NaN as an empty cell."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(
            x, (int, float, np.integer, np.floating)):
        return str(x)
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return '' if np.isnan(x) else str(x)
    if digits is not None:
        return f'{x:,.{digits}f}'
    if isinstance(x, (int, np.integer)) or (float(x).is_integer()
                                            and math.fabs(x) >= 100):
        return f'{int(round(x)):,}'
    mag = math.floor(math.log10(math.fabs(x))) if x else 0
    if mag >= 3:  # 533,907.3 -> 534,000
        return f'{int(round(x, 2 - mag)):,}'
    return f'{x:,.{max(0, 2 - mag)}f}'

#@save
def show_table(df, digits=None, max_rows=15, headers=None, width=100):
    """Print df as an aligned table: numbers rounded (digits: one int for
    every column or a dict per column; default 3 significant digits),
    columns renamed by headers, at most max_rows rows (a last line counts
    the rest). A line wider than width characters raises: shorten the
    headers instead."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    shown = df.head(max_rows)
    cols = {}
    for c in shown.columns:
        d = digits.get(c) if isinstance(digits, dict) else digits
        cols[(headers or {}).get(c, c)] = [_hw_fmt(v, d) for v in shown[c]]
    table = pd.DataFrame(cols, index=shown.index)
    table.index.names = [None] * table.index.nlevels  # no name line
    show_index = not isinstance(df.index, pd.RangeIndex)
    text = table.to_string(index=show_index, justify='right')
    if len(df) > max_rows:
        text += f'\n... {len(df) - max_rows} more rows'
    wide = max(len(line) for line in text.splitlines())
    if wide > width:
        raise ValueError(f'table is {wide} characters wide (> {width}): '
                         'pass shorter headers or fewer columns')
    print(text)
```

## The Specification Table

`chip_specs` (on the page) reads the registered table and checks it here: every
row has its source and retrieval date, and the four chips of the book's
`tab_gpu_specs` match that table after its rounding.

```{.python .input #accelerator-design-lib-check-specs}
%%tab pytorch
#@save
_HW_BOOK_TABLE = {  # tab_gpu_specs of sec_hardware, as printed (strings keep its rounding)
    'H100 SXM': dict(mem_gb='80', mem_bw_tbs='3.35', p16_tflops='989',
                     p8_tflops='1979', p4_tflops=None, tdp_w='700',
                     ridge16='295', link='450'),
    'B200': dict(mem_gb='180', mem_bw_tbs='8', p16_tflops='2250',
                 p8_tflops='4500', p4_tflops='9000', tdp_w='1000',
                 ridge16='281', link='900'),
    'RTX 4090': dict(mem_gb='24', mem_bw_tbs='1.0', p16_tflops='165',
                     p8_tflops='330', p4_tflops=None, tdp_w='450',
                     ridge16='164', link='32'),
    'RTX 5090': dict(mem_gb='32', mem_bw_tbs='1.8', p16_tflops='210',
                     p8_tflops='419', p4_tflops='1676', tdp_w='575',
                     ridge16='117', link='64')}

#@save
_HW_SPEC_COLUMNS = (
    'vendor', 'kind', 'year', 'mem_type', 'mem_gb', 'mem_bw_tbs', 'sram_mb',
    'p16_tflops', 'p8_tflops', 'p4_tflops', 'tdp_w', 'scaleup_gbs',
    'scaleup_note', 'scaleup_domain', 'scaleout_gbs', 'sparsity_note',
    'source_url', 'retrieved', 'notes', 'status', 'tf32_tflops',
    'fp32_tflops')

#@save
def _book_round(x, printed):
    """x rounded as the book prints `printed` (its number of decimals)."""
    decimals = len(printed.split('.')[1]) if '.' in printed else 0
    return round(float(x), decimals) == float(printed)

#@save
def _check_specs(df):
    """Assert the table's contract: the 01_interfaces columns (plus status,
    tf32_tflops, fp32_tflops), a source and a retrieval date per row, a
    known kind and status, and the book's four rows reproduced (the ridge
    to +-1 FLOP/byte, links as half the bidirectional scale-up figure)."""
    missing = [c for c in _HW_SPEC_COLUMNS if c not in df.columns]
    assert not missing, f'chip table lacks columns {missing}'
    assert df.index.name == 'chip' and df.index.is_unique
    for chip, row in df.iterrows():
        assert isinstance(row.source_url, str) and row.source_url.startswith(
            'https://'), f'{chip}: no source_url'
        assert isinstance(row.retrieved, str) and len(row.retrieved) == 10, (
            f'{chip}: no retrieval date')
        assert row.kind in ('gpu', 'tpu', 'asic', 'wafer', 'lpu'), chip
        assert row.status in ('shipping', 'announced'), chip
    for chip, book in _HW_BOOK_TABLE.items():
        row = df.loc[chip]
        for col in ('mem_gb', 'mem_bw_tbs', 'p16_tflops', 'p8_tflops',
                    'p4_tflops', 'tdp_w'):
            if book[col] is None:
                assert np.isnan(row[col]), f'{chip} {col}: book has none'
            else:
                assert _book_round(row[col], book[col]), (
                    f'{chip} {col}: {row[col]} != book {book[col]}')
        ridge = row.p16_tflops / row.mem_bw_tbs
        assert math.fabs(ridge - float(book['ridge16'])) <= 1, f'{chip} ridge'
        assert _book_round(row.scaleup_gbs / 2, book['link']), f'{chip} link'
```

The section's static table (transcribed 2026-11-25; `tools/lm_data/hw/chip_specs.csv`).

```{.python .input #accelerator-design-lib-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-hw-chip-specs-v1'] = (
    d2l.LM_DATA_URL + 'lm-hw-chip-specs-v1.csv',
    '173e1685760ac66e99376ce195fe667f83a15dcb')
```

## Timing on One GPU

```{.python .input #accelerator-design-lib-timer}
%%tab pytorch
#@save
def _hw_time(fn, repeats=20, graph=True, target_ms=2.0, max_inner=100,
             warmup=3):
    """Milliseconds per call of fn() on the current GPU, repeats values:
    warm-up calls, then inner calls per timing (enough for target_ms, at
    most max_inner), captured in one CUDA graph when graph (no launch
    cost between them), timed with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    stop.record()
    stop.synchronize()
    inner = int(min(max_inner, max(1, math.ceil(
        target_ms / max(start.elapsed_time(stop), 1e-4)))))
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
        run = g.replay
        run()
    else:
        def run():
            for _ in range(inner):
                fn()
    times = []
    for _ in range(repeats):
        start.record()
        run()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) / inner)
    return np.array(times)

#@save
def _hw_stats(t):
    """(median, median absolute deviation) of the times t."""
    med = float(np.median(t))
    return med, float(np.median(np.abs(np.asarray(t) - med)))

#@save
@contextlib.contextmanager
def _hw_tf32(on):
    """FP32 matrix products on TF32 tensor cores (on) or in full FP32."""
    old = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.set_float32_matmul_precision('high' if on else 'highest')
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old[0]
        torch.set_float32_matmul_precision(old[1])

#@save
def _hw_fits(nbytes, fraction=0.5):
    """True when nbytes fit in fraction of this GPU's memory (the ladders
    skip larger working sets on a smaller card)."""
    return nbytes <= fraction * torch.cuda.get_device_properties(0).total_memory

#@save
_HW_L1_KIB = {  # unified L1/shared-memory SRAM per SM by compute capability
    (8, 0): 192, (8, 6): 128, (8, 9): 128, (9, 0): 256, (10, 0): 256,
    (12, 0): 128}  # NVIDIA Ampere, Ada, Hopper and Blackwell tuning guides

#@save
def _hw_frame(rows):
    """A result table with the GPU it was measured on in its attrs: name,
    memory, L2, SMs, shared memory and the unified L1/shared SRAM per SM."""
    df = pd.DataFrame(rows)
    p = torch.cuda.get_device_properties(0)
    l1 = _HW_L1_KIB.get((p.major, p.minor))
    df.attrs.update(gpu=_gpu_short(p.name), mem_bytes=int(p.total_memory),
                    l2_bytes=int(p.L2_cache_size),
                    n_sm=int(p.multi_processor_count),
                    smem_per_sm=int(p.shared_memory_per_multiprocessor),
                    l1_bytes=l1 * 2**10 if l1 else None)
    return df

#@save
def _hw_steady(bench, warm_s=0.5, hz=100):
    """bench() (a dict) repeated for warm_s seconds, then once more for the
    result under a power trace: that result plus the median SM clock (MHz)
    and board power (W) sampled while it ran (NaN without samples; NVML
    averages power over about a second, which the warm-up fills with the
    same load). The warm-up brings the card to the clock and power it holds
    under this load, so the timing and its clock come from one state."""
    t = time.time()
    while time.time() - t < warm_s:
        bench()
    with _hw_power_trace(hz) as samples:
        r = bench()
    return dict(r, sm_mhz=_hw_median(samples, 'sm_mhz'),
                power_w=_hw_median(samples, 'power_w'))

#@save
def _hw_median(samples, key):
    """Median of key over the trace samples that have it (NaN if none)."""
    values = [x[key] for x in samples if x.get(key) is not None]
    return float(np.median(values)) if values else math.nan
```

## The Measurement Environment, Logs and Power Traces

```{.python .input #accelerator-design-lib-gpu-query}
%%tab pytorch
#@save
_HW_GPU_FIELDS = ('clocks.max.sm', 'clocks.max.mem', 'power.limit',
                  'power.default_limit', 'power.min_limit', 'power.max_limit',
                  'pcie.link.gen.max', 'pcie.link.width.max',
                  'pcie.link.gen.current', 'pcie.link.width.current')

#@save
def _hw_number(s):
    try:
        x = float(s)
    except (TypeError, ValueError):
        return None
    return int(x) if x.is_integer() else x

#@save
def _hw_gpu_limits():
    """Limits of the first visible GPU (nvidia-smi, by UUID): clocks, power
    limits, PCIe link; its SM count and the host's GPU count."""
    if not torch.cuda.is_available():
        return {}
    uuid = _hw_visible_uuids()[0]
    rows = _hw_smi('--query-gpu=' + ','.join(_HW_GPU_FIELDS), '-i', uuid)
    vals = [x.split()[0] if x.split() else x
            for x in (rows[0] if rows else [None] * len(_HW_GPU_FIELDS))]
    out = {f.replace('.', '_'): _hw_number(v)
           for f, v in zip(_HW_GPU_FIELDS, vals)}
    out['n_sm'] = torch.cuda.get_device_properties(0).multi_processor_count
    out['n_host'] = len(_hw_smi('--query-gpu=uuid'))
    return out

#@save
def _hw_pack_raw(raw):
    """Raw tool output as gzip + base64 text, so a parser can be fixed and
    rerun on the stored output later."""
    import base64
    data = raw.encode() if isinstance(raw, str) else bytes(raw)
    return base64.b64encode(gzip.compress(data, mtime=0)).decode('ascii')

#@save
def _hw_unpack_raw(text):
    import base64
    return gzip.decompress(base64.b64decode(text)).decode()

#@save
def _hw_nvml_sampler(hz, samples, stop):
    """Append NVML samples of the first visible GPU at hz until stop is set
    (False if pynvml cannot be used)."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return False
    def run():
        h = pynvml.nvmlDeviceGetHandleByUUID(_hw_visible_uuids()[0])
        t0 = time.time()
        while not stop.is_set():
            samples.append(dict(
                t=round(time.time() - t0, 3),
                power_w=pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                sm_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                mem_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                temp_c=pynvml.nvmlDeviceGetTemperature(
                    h, pynvml.NVML_TEMPERATURE_GPU),
                throttle=int(
                    pynvml.nvmlDeviceGetCurrentClocksEventReasons(h))))
            stop.wait(1 / hz)
        pynvml.nvmlShutdown()
    import threading
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread

#@save
@contextlib.contextmanager
def _hw_power_trace(hz=10):
    """Samples (t, power_w, sm_mhz, mem_mhz, temp_c, throttle) of the first
    visible GPU at hz during the block: pynvml, else an `nvidia-smi -lms`
    subprocess, else none (an empty list)."""
    import threading
    samples, stop = [], threading.Event()
    thread = (_hw_nvml_sampler(hz, samples, stop)
              if torch.cuda.is_available() else False)
    proc = None
    if thread is False and torch.cuda.is_available():
        try:
            proc = subprocess.Popen(
                ['nvidia-smi', '-i', _hw_visible_uuids()[0],
                 '--query-gpu=power.draw,clocks.sm,clocks.mem,'
                 'temperature.gpu,clocks_event_reasons.active',
                 '--format=csv,noheader,nounits', f'-lms={int(1000 / hz)}'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                start_new_session=True)
        except OSError:
            proc = None
    try:
        yield samples
    finally:
        stop.set()
        if thread:
            thread.join(timeout=5)
        if proc is not None:
            d2l._kill_session(proc)
            out = proc.communicate()[0]
            for i, line in enumerate(out.splitlines()):
                v = [x.strip() for x in line.split(',')]
                if len(v) == 5:
                    samples.append(dict(
                        t=round(i / hz, 3), power_w=_hw_number(v[0]),
                        sm_mhz=_hw_number(v[1]), mem_mhz=_hw_number(v[2]),
                        temp_c=_hw_number(v[3]), throttle=int(v[4], 16)
                        if v[4].startswith('0x') else None))
```

## Two Kernels in Triton: L1/Shared-Memory Bandwidth and the Pointer Chase

The build box's system CUDA toolkit cannot build `load_inline` extensions
against this torch without a second CUDA runtime (infra toolchain check), so
both kernels are Triton. Triton has no explicit shared-memory load; on Ada,
L1 and shared memory are one 128 KB SRAM per SM with the same 128 bytes per
clock, so the bandwidth kernel re-reads a small, L1-resident block per
program at a different offset each iteration (no load can be hoisted).

```{.python .input #accelerator-design-lib-kernels}
%%tab pytorch
#@save
@triton.jit
def _l1_read_kernel(x_ptr, out_ptr, iters, WORK: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(iters):
        acc += tl.load(x_ptr + pid * WORK + (offs + i * BLOCK) % WORK,
                       cache_modifier='.ca')
    tl.store(out_ptr + pid * BLOCK + offs, acc)

#@save
@triton.jit
def _chase_kernel(next_ptr, pos_ptr, steps):
    j = tl.load(pos_ptr)  # continue where the last launch stopped
    for _ in range(steps):
        j = tl.load(next_ptr + j)  # each address depends on the last load
    tl.store(pos_ptr, j)

#@save
@triton.jit
def _read_kernel(x_ptr, out_ptr, n, passes, BLOCK: tl.constexpr):
    pid, grid = tl.program_id(0), tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in range(passes):
        for start in range(pid * BLOCK, n, grid * BLOCK):
            i = start + offs
            acc += tl.load(x_ptr + i, mask=i < n, other=0.0,
                           cache_modifier='.cg')  # L2 only, never L1
    tl.store(out_ptr + pid * BLOCK + offs, acc)

#@save
def read_passes(x, passes, block=2048, programs_per_sm=4, min_block=128):
    """Read the fp32 tensor x passes times in one kernel (loads cached in
    L2, not L1), so a working set that fits L2 is served from L2 after the
    first pass and launch cost is paid once; returns the per-program sums.
    A small x gets smaller blocks (down to min_block elements), so that
    every SM still runs programs_per_sm programs: the small end of the
    ladder then measures the memory system, not a half-empty grid."""
    n_sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    programs = n_sm * programs_per_sm
    while block > min_block and triton.cdiv(x.numel(), block) < programs:
        block //= 2
    grid = (max(1, min(programs, triton.cdiv(x.numel(), block))),)
    out = torch.empty(grid[0] * block, device=x.device)
    _read_kernel[grid](x, out, x.numel(), passes, BLOCK=block,
                       num_warps=max(1, min(8, block // 256)))
    return out

#@save
_HW_L1_BYTES_PER_CLK = 128  # per SM per clock, Ada and Ampere whitepapers

#@save
def _smem_compute(programs_per_sm=8, work=2048, block=1024, iters=8192,
                  repeats=20):
    x = torch.rand(torch.cuda.get_device_properties(0).multi_processor_count
                   * programs_per_sm * work, device='cuda')
    grid = (x.numel() // work,)
    out = torch.empty(grid[0] * block, device='cuda')
    fn = lambda: _l1_read_kernel[grid](x, out, iters, WORK=work, BLOCK=block,
                                       num_warps=4)
    with d2l._quiet():  # the first call compiles
        fn()
    r = _hw_steady(lambda: dict(zip(('ms', 'mad'), _hw_stats(
        _hw_time(fn, repeats, graph=False, max_inner=1)))))
    p = torch.cuda.get_device_properties(0)
    moved = grid[0] * iters * block * 4
    peak = p.multi_processor_count * _HW_L1_BYTES_PER_CLK * r['sm_mhz'] / 1e6
    return dict(r, gpu=_gpu_short(p.name), mem_bytes=int(p.total_memory),
                tbs=moved / r['ms'] / 1e9, n_sm=p.multi_processor_count,
                peak_tbs=peak)

#@save
def _show_smem(r):
    if not r:
        return
    print(f"L1 loads: {_hw_sig(r['tbs'])} TB/s measured; bound "
          f"{r['n_sm']} SMs x {_HW_L1_BYTES_PER_CLK} B/clk x "
          f"{_hw_sig(r['sm_mhz'] / 1000, 3)} GHz = {_hw_sig(r['peak_tbs'])} "
          'TB/s')

#@save
def _smem_live(r, same):
    live = _smem_compute(repeats=5)
    return _hw_agreement([('L1', r.get('tbs'), live['tbs'], r.get('mad', 0)
                           / r.get('ms', 1))], same, 'TB/s')

#@save
def hw_smem_bench():
    """L1 load bandwidth of every SM at once (TB/s), with the bound
    n_SM x 128 B/clk x the median SM clock sampled while the kernel ran."""
    return _hw_result('box-microbench', _smem_compute, dict(kernel='l1-read'),
                      entry='smem', live_check=_smem_live, show=_show_smem,
                      empty=dict, live_label='L1')

#@save
def _chase_next(nbytes, line=128, seed=0):
    """The pointer-chase array over nbytes: one int32 per line bytes, a
    random cycle through all of them (each entry the next one's index)."""
    n_lines, stride = nbytes // line, line // 4
    g = torch.Generator(device='cuda').manual_seed(seed)
    perm = torch.randperm(n_lines, device='cuda', generator=g)
    nxt = torch.zeros(nbytes // 4, dtype=torch.int32, device='cuda')
    nxt[perm * stride] = (torch.roll(perm, -1) * stride).int()
    return nxt  # index 0 (line 0) is on the cycle: the chase starts there

#@save
def _chase_ns(nbytes, steps=16384, repeats=5, warm_lines=2**20):
    """ns per load of the chase over nbytes: one warm-up walk over the whole
    cycle (up to warm_lines lines), then timed walks of steps loads, each
    continuing where the last stopped (a cycle larger than the L2 keeps
    missing it)."""
    nxt = _chase_next(nbytes)
    pos = torch.zeros(1, dtype=torch.int32, device='cuda')
    with d2l._quiet():  # the first launch compiles
        _chase_kernel[(1,)](nxt, pos, min(nbytes // 128, warm_lines),
                            num_warps=1)
    fn = lambda: _chase_kernel[(1,)](nxt, pos, steps, num_warps=1)
    t = _hw_time(fn, repeats, graph=False, max_inner=1, warmup=0)
    ms, mad = _hw_stats(t)
    return dict(bytes=nbytes, ns_per_load=ms * 1e6 / steps,
                mad=mad * 1e6 / steps)

#@save
def _latency_live(r, same):
    picks = r.iloc[[0, len(r) // 2, -1]] if len(r) else r
    pairs = [(f"{_hw_bytes(b)}", ns, _chase_ns(int(b))['ns_per_load'],
              mad / ns) for b, ns, mad in zip(picks.bytes, picks.ns_per_load,
                                               picks.mad)]
    return _hw_agreement(pairs, same, 'ns')

#@save
def _latency_plateaus(lat):
    """(L1, L2, DRAM) ns per dependent load: medians over working sets up to
    a quarter of the L1, from 1 MiB to a quarter of the L2, and from four
    times the L2 up."""
    l1 = lat.attrs.get('l1_bytes') or 128 * 2**10
    l2 = lat.attrs.get('l2_bytes') or 2**25
    b, ns = lat.bytes, lat.ns_per_load
    return tuple(float(ns[m].median()) if m.any() else math.nan for m in (
        b <= l1 / 4, (b >= 2**20) & (b <= l2 / 4), b >= 4 * l2))

#@save
def _show_latency(lat):
    if lat.empty:
        return
    l1, l2, dram = _latency_plateaus(lat)
    print(f'latency per dependent load: L1 {_hw_sig(l1)} ns, '
          f'L2 {_hw_sig(l2)} ns, DRAM {_hw_sig(dram)} ns')

#@save
def hw_latency_bench(sizes):
    """Nanoseconds per dependent load (one thread chasing a random cycle)
    for working sets of each size in bytes: L1, L2 and DRAM latency."""
    sizes = [int(s) for s in sizes]
    compute = lambda: _hw_frame([_chase_ns(s) for s in sizes if _hw_fits(s)])
    return _hw_result('box-microbench', compute, dict(sizes=sizes),
                      entry='latency', live_check=_latency_live,
                      show=_show_latency, live_label='latency')
```

## Displays and Live Checks

```{.python .input #accelerator-design-lib-display}
%%tab pytorch
#@save
def _hw_sig(x, digits=2):
    """x at `digits` significant digits, thousands separated."""
    if x is None or not np.isfinite(x):
        return 'n/a'
    if x == 0:
        return '0'
    mag = math.floor(math.log10(math.fabs(x)))
    r = round(x, digits - 1 - mag)
    return f'{r:,.{max(0, digits - 1 - mag)}f}'

#@save
def _hw_bytes(n):
    """4096 -> '4 KiB', 2**30 -> '1 GiB'."""
    for unit, size in (('GiB', 2**30), ('MiB', 2**20), ('KiB', 2**10)):
        if n >= size:
            return f'{n / size:g} {unit}'
    return f'{n} B'

#@save
def _hw_agreement(points, same, unit, floor=0.1, k=5):
    """The live line: on the committed run's GPU, the largest relative
    difference of (label, committed, live, committed relative noise)
    points, and 'differs' for any beyond max(k x noise, floor): series not
    run back to back with unlocked clocks move by several percent with the
    card's temperature; on another GPU, the live values."""
    if not same:
        return ', '.join(f'{p[0]} {_hw_sig(p[2])}' for p in points) + f' {unit}'
    diffs = [(p[0], p[1], p[2], math.fabs(p[2] - p[1]) / p[1]) for p in points]
    worst = max(d[3] for d in diffs)
    bad = [d for d, p in zip(diffs, points) if d[3] > max(k * p[3], floor)]
    text = (f'{len(points)} point{"s" if len(points) > 1 else ""}, largest '
            f'difference {worst:.0%}')
    if bad:
        text += '; differs: ' + ', '.join(
            f'{d[0]} {_hw_sig(d[2])} vs {_hw_sig(d[1])} {unit}' for d in bad)
    return text

#@save
def _hw_noise(df, i, col):
    """Relative noise of row i's col in a committed table: the larger of
    its within-process MAD (over the timed repeats) and its spread between
    the regeneration's processes."""
    row = df.iloc[i]
    spread = df.attrs.get('spread', {}).get(col)
    return max(row.mad / row.ms, spread[i] if spread else 0.0)

#@save
def _ladder_live(r, same):
    picks = [s for s in (2**26, 2**30, 2**32) if _hw_fits(s)]
    pts = []
    for s in picks:
        hit = np.flatnonzero((r.kind == 'read').to_numpy()
                             & (r.bytes == s).to_numpy())
        if len(hit):
            i = int(hit[0])
            pts.append((_hw_bytes(s), r.gbs.iloc[i], d2l.hw_copy_bench(
                s, 'read', repeats=5)['gbs'], _hw_noise(r, i, 'gbs')))
    return _hw_agreement(pts, same, 'GB/s')

#@save
def _dtype_live(r, same):
    n = int(r.attrs.get('n', 8192))
    fp8 = torch.cuda.get_device_capability() >= (8, 9)  # FP8 tensor cores
    pts = [(f, r.tflops.iloc[i],
            d2l.hw_matmul_bench(n, n, n, f, repeats=5)['tflops'],
            _hw_noise(r, i, 'tflops'))
           for i, f in enumerate(r.format)
           if f == 'bf16' or (f == 'fp8' and fp8)]
    return _hw_agreement(pts, same, 'TFLOP/s')

#@save
_HW_DATASHEET = {'fp32': 'fp32_tflops', 'tf32': 'tf32_tflops',
                 'bf16': 'p16_tflops', 'fp16': 'p16_tflops',
                 'fp8': 'p8_tflops'}

#@save
_HW_BOOST_MHZ = {  # the boost clock the vendor's peak figures assume
    'RTX 4090': 2520,  # NVIDIA RTX 4090 product page
    'A100 PCIe 40 GB': 1410}  # NVIDIA A100 datasheet

#@save
def _hw_chip_row(gpu, mem_bytes, specs):
    """The chip-table row of a measured GPU: the longest row name inside
    the device name (letters and digits, case ignored) whose capacity is
    within 10% of the device's memory (mem_bytes, unchecked if None);
    None when no row fits (an A100 SXM 80 GB is not the A100 PCIe 40 GB)."""
    if not gpu:
        return None
    key = lambda name: re.sub(r'[^a-z0-9]', '', name.lower())
    for chip in sorted((c for c in specs.index if key(c) in key(gpu)),
                       key=len, reverse=True):
        gib = specs.loc[chip, 'mem_gb']
        if mem_bytes is None or math.fabs(mem_bytes / 2**30 / gib - 1) <= 0.1:
            return chip
    return None

#@save
def _show_dtypes(r):
    """The format ladder against the datasheet, with the spread between
    processes and the SM clock and board power while each format ran."""
    if r.empty:
        return
    specs = d2l.chip_specs()
    chip = _hw_chip_row(r.attrs.get('gpu'), r.attrs.get('mem_bytes'), specs)
    sheet = [specs.loc[chip, _HW_DATASHEET[f]] if chip else math.nan
             for f in r.format]
    cols = {'format': r.format, 'TFLOP/s': [_hw_sig(x) for x in r.tflops]}
    spread = r.attrs.get('spread', {}).get('tflops')
    if spread:
        cols['spread'] = [f'{100 * x:.1f}%' for x in spread]
    cols.update({'datasheet': sheet, 'fraction': [
        f'{m / s:.0%}' if s == s else '' for m, s in zip(r.tflops, sheet)]})
    if 'sm_mhz' in r:
        cols['SM GHz'] = [_hw_sig(x / 1000, 3) for x in r.sm_mhz]
        cols['W'] = [_hw_sig(x, 3) for x in r.power_w]
    show_table(pd.DataFrame(cols))
    if chip in _HW_BOOST_MHZ:
        print(f'datasheet peaks assume the {_HW_BOOST_MHZ[chip] / 1000:.2f} '
              'GHz boost clock')

#@save
def _ladder_plateaus(ladder):
    """(L2 read, DRAM read, DRAM copy) in GB/s: medians over 4 MiB to half
    the L2 and over working sets of at least 1 GiB."""
    l2 = ladder.attrs.get('l2_bytes', 0)
    read, copy = ladder[ladder.kind == 'read'], ladder[ladder.kind == 'copy']
    in_l2 = read[(read.bytes >= 2**22) & (read.bytes <= l2 / 2)]
    return (float(in_l2.gbs.median()),
            float(read[read.bytes >= 2**30].gbs.median()),
            float(copy[copy.bytes >= 2**30].gbs.median()))

#@save
def _show_ladder(ladder):
    """The ladder's three plateaus, the DRAM read against the datasheet."""
    if ladder.empty:
        return
    specs = d2l.chip_specs()
    chip = _hw_chip_row(ladder.attrs.get('gpu'), ladder.attrs.get('mem_bytes'),
                        specs)
    l2 = ladder.attrs.get('l2_bytes', 0)
    l2_gbs, dram, copy = _ladder_plateaus(ladder)
    print(f'L2 read (4 MiB to {l2 / 2**21:g} MiB): {_hw_sig(l2_gbs)} GB/s, '
          f'{l2_gbs / dram:.1f}x the DRAM read')
    line = f'DRAM read (1-4 GiB): {_hw_sig(dram)} GB/s'
    if chip:
        sheet = specs.loc[chip, 'mem_bw_tbs'] * 1000
        line += f', {dram / sheet:.0%} of the datasheet {sheet:,.0f} GB/s'
    print(line)
    print(f'DRAM copy (1-4 GiB, read + write counted): {_hw_sig(copy)} GB/s')

#@save
def plot_ladder(ladder, specs, reference=None):
    """Plot read and copy bandwidth against the working set (log2 x, log
    y) with the datasheet DRAM line and the L2 edge (half to all of the L2,
    shaded), plus the A100 campaign's series when its log is committed and
    measured on the chip it names (else one line says why not)."""
    logs, status = [ladder], ''
    if isinstance(reference, CampaignPending):
        status = hw_status_line(reference).replace('not run yet', 'pending')
    elif reference:
        ref = reference['ladder']
        if _hw_chip_row(ref.attrs.get('gpu'), ref.attrs.get('mem_bytes'),
                        specs):
            logs.append(ref)
        else:
            status = (f"reference measured on {ref.attrs.get('gpu')}, a GPU "
                      'without a chip-table row: not plotted')
    X, Y, legend, fmts = [], [], [], []
    for i, df in enumerate(logs):
        for kind, label, colour in (('read', 'read', 'C0'),
                                    ('copy', 'copy (read + write)', 'C1')):
            part = df[df.kind == kind]
            X.append(part.bytes.to_numpy())
            Y.append(part.gbs.to_numpy())
            fmts.append(colour + ('-' if i == 0 else ':'))
            gpu = df.attrs.get('gpu')
            legend.append(f'{gpu}: {label}' if len(logs) > 1 else label)
    d2l.plot(X, Y, 'working set (bytes)', 'bandwidth (GB/s)', fmts=fmts,
             xscale='log', yscale='log', figsize=(4.5, 3))
    ax = d2l.plt.gca()
    ax.set_xscale('log', base=2)
    handles = ax.get_lines()[:len(Y)]
    chip = _hw_chip_row(ladder.attrs.get('gpu'), ladder.attrs.get('mem_bytes'),
                        specs)
    if chip:
        sheet = specs.loc[chip, 'mem_bw_tbs'] * 1000
        handles.append(ax.axhline(sheet, color='k', ls='--', lw=1))
        legend.append(f'datasheet {sheet:,.0f} GB/s')
    l2 = ladder.attrs.get('l2_bytes', 0)
    if l2:
        handles.append(ax.axvspan(l2 / 2, l2, color='0.9', zorder=0))
        legend.append(f'L2 edge ({l2 / 2**21:g}-{l2 / 2**20:g} MiB)')
    ax.legend(handles, legend, fontsize=8, loc='center left',
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    if status:
        print(status)

#@save
def plot_latency(lat):
    """Nanoseconds per dependent load against the working set (log2 x, log
    y), with the unified L1/shared SRAM per SM and the L2 capacity marked."""
    d2l.plot(lat.bytes.to_numpy(), lat.ns_per_load.to_numpy(),
             'working set (bytes)', 'latency (ns per load)', xscale='log',
             yscale='log', figsize=(4.5, 3))
    ax = d2l.plt.gca()
    ax.set_xscale('log', base=2)
    for size, label in ((lat.attrs.get('l1_bytes'), 'L1 per SM'),
                        (lat.attrs.get('l2_bytes'), 'L2')):
        if size:  # labels at the bottom, clear of the plateaus
            ax.axvline(size, color='k', ls=':', lw=1)
            ax.text(size * 1.15, ax.get_ylim()[0] * 1.08,
                    f'{label} ({_hw_bytes(size)})', va='bottom', fontsize=9,
                    bbox=dict(fc='white', ec='none', pad=1))

#@save
def _place_labels(ax, points, names, radii, fontsize=7):
    """Annotate each point with its name at the first of eight offsets
    (outside its marker of the given radius, in points) whose text box
    overlaps no earlier label and no other point."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    marks = [ax.transData.transform(p) for p in points]
    placed = []
    unit = [(1, 0.5), (1, -1.4), (-1, 0.5), (-1, -1.4), (0, 1.2),
            (0, -2.2), (1.4, -0.5), (-1.4, -0.5)]
    for (x, y), name, own, r in zip(points, names, marks, radii):
        offsets = [(dx * (r + 3), dy * (r + 3)) for dx, dy in unit]
        for dx, dy in offsets:
            text = ax.annotate(name, (x, y), xytext=(dx, dy),
                               textcoords='offset points', fontsize=fontsize,
                               ha='left' if dx >= 0 else 'right')
            box = text.get_window_extent(renderer).expanded(1.05, 1.1)
            hit = any(box.overlaps(b) for b in placed) or any(
                box.contains(*m) for m in marks if m is not own)
            if not hit:
                break
            text.remove()
        else:
            text = ax.annotate(name, (x, y), xytext=offsets[0],
                               textcoords='offset points', fontsize=fontsize)
            box = text.get_window_extent(renderer)
        placed.append(box)

#@save
def plot_chip_plane(ratios):
    """The design plane: ridge point (FLOP/byte, dense 16-bit) against
    sweep time C / beta (ms), marker area by memory capacity, colour by
    kind; chips without a dense 16-bit figure (or without a bandwidth) sit
    in the strip below, on a log axis of their own."""
    fig = d2l.plt.figure(figsize=(6.5, 5))
    ax, strip = fig.subplots(2, 1, gridspec_kw=dict(height_ratios=[5, 1.3]))
    kinds = sorted(ratios.kind.unique())
    colour = {k: f'C{i}' for i, k in enumerate(kinds)}
    area = lambda df: 15 + 1.2 * df.mem_gb.clip(lower=1).to_numpy()
    full = ratios[ratios.ridge16.notna() & ratios.sweep_ms.notna()]
    side = ratios[ratios.ridge16.isna() & ratios.sweep_ms.notna()]
    for kind in kinds:
        part = full[full.kind == kind]
        ax.scatter(part.sweep_ms, part.ridge16, s=area(part),
                   color=colour[kind], alpha=0.6, label=kind, lw=0)
        part = side[side.kind == kind]
        strip.scatter(part.sweep_ms, np.zeros(len(part)), s=area(part),
                      color=colour[kind], alpha=0.6, lw=0)
    ax.set_xlabel('sweep time C / beta (ms)')
    ax.set_ylabel('ridge point (FLOP/byte)')
    ax.margins(0.12)
    handles = [d2l.plt.Line2D([], [], ls='', marker='o', alpha=0.6,
                              color=colour[k], label=k) for k in kinds]
    ax.legend(handles=handles, fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1), title='marker area:\nmemory',
              title_fontsize=8, frameon=False)
    ax.text(0.02, 0.97, 'more FLOP per byte: prefill, training',
            transform=ax.transAxes, va='top', fontsize=8, color='k')
    ax.text(0.02, 0.03, 'shorter sweep: decode', transform=ax.transAxes,
            va='bottom', fontsize=8, color='k')
    strip.set_xscale('log')
    strip.set_yticks([])
    strip.set_ylim(-1, 1.6)
    strip.margins(x=0.15)
    strip.set_xlabel('sweep time (ms), chips without a dense 16-bit figure')
    fig.tight_layout()
    radius = lambda df: np.sqrt(area(df) / np.pi)
    _place_labels(ax, list(zip(full.sweep_ms, full.ridge16)), list(full.index),
                  radius(full))
    _place_labels(strip, list(zip(side.sweep_ms, np.zeros(len(side)))),
                  list(side.index), radius(side))
    for a in (ax, strip):
        for spine in a.spines.values():
            spine.set_color('k')

#@save
def show_env(env):
    """The fingerprint as six labelled lines; one line without a GPU."""
    tools = (f"CUDA {env.get('cuda')}, torch {env.get('torch')}, Triton "
             f"{env.get('triton') or 'absent'}")
    if not env.get('gpu'):
        print(f'no CUDA GPU visible; {tools}')
        return
    v = lambda k, spec='': 'n/a' if env.get(k) is None else format(env[k],
                                                                    spec)
    print(f"GPU: {_gpu_short(env['gpu'])} ({v('n_gpu')} visible, "
          f"{env.get('n_host') or v('n_gpu')} on the host)")
    print(f"SMs per GPU: {v('n_sm')}")
    print(f"max SM clock: {v('clocks_max_sm', ',')} MHz; memory clock "
          f"{v('clocks_max_mem', ',')} MHz")
    print(f"power limit: {v('power_limit', 'g')} W (settable "
          f"{v('power_min_limit', 'g')}-{v('power_max_limit', 'g')} W)")
    print(f"PCIe: Gen {v('pcie_link_gen_max')} x{v('pcie_link_width_max')} "
          f"(max); Gen {v('pcie_link_gen_current')} "
          f"x{v('pcie_link_width_current')} now")
    print(f"driver {v('driver')}, NCCL {v('nccl')}, {tools}")

#@save
_HW_SHEET_ROWS = (  # quantity, datasheet column, (entry, key) of the measurement
    ('BF16 (TFLOP/s)', 'p16_tflops', ('dtypes', 'bf16')),
    ('FP16 (TFLOP/s)', 'p16_tflops', ('dtypes', 'fp16')),
    ('TF32 (TFLOP/s)', 'tf32_tflops', ('dtypes', 'tf32')),
    ('FP32 (TFLOP/s)', 'fp32_tflops', ('dtypes', 'fp32')),
    ('FP8 (TFLOP/s)', 'p8_tflops', ('dtypes', 'fp8')),
    ('DRAM read (GB/s)', 'mem_bw_tbs', ('ladder', 'dram')),
    ('L2 read (GB/s)', None, ('ladder', 'l2')),
    ('L1 loads (TB/s)*', None, ('smem', 'tbs')))

#@save
def _hw_measured(entries, entry, key):
    """One measured number of a {'ladder', 'dtypes', 'smem'} dict."""
    r = entries.get(entry)
    if r is None or len(r) == 0:
        return math.nan
    if entry == 'dtypes':
        rows = r.set_index('format')
        return float(rows.loc[key, 'tflops']) if key in rows.index else math.nan
    if entry == 'ladder':
        l2, dram, _ = _ladder_plateaus(r)
        return dram if key == 'dram' else l2
    return float(r.get(key, math.nan))

#@save
def _hw_entries(measured):
    """The {'ladder', 'dtypes', 'smem'} dict of datasheet_vs_measured's
    arguments; {} for a pending campaign log."""
    if len(measured) == 1:
        return {} if isinstance(measured[0], CampaignPending) else dict(
            measured[0])
    return dict(zip(('ladder', 'dtypes', 'smem'), measured))

#@save
def _hw_measured_gpu(entries):
    """(device name, memory in bytes) the entries were measured on."""
    for name in ('ladder', 'dtypes'):
        if name in entries and len(entries[name]):
            a = entries[name].attrs
            return a.get('gpu'), a.get('mem_bytes')
    smem = entries.get('smem') or {}
    return smem.get('gpu'), smem.get('mem_bytes')

#@save
def _hw_sheet_table(row, entries):
    """Datasheet, measured and their fraction for the eight quantities of
    the chip row; the L1 reference is the bound n_SM x 128 B/clk x the
    measured SM clock, not a datasheet figure."""
    rows = []
    for quantity, column, (entry, key) in _HW_SHEET_ROWS:
        sheet = math.nan if column is None else float(row[column])
        if column == 'mem_bw_tbs':
            sheet *= 1000
        if entry == 'smem' and entries.get('smem'):
            sheet = entries['smem'].get('peak_tbs', math.nan)
        rows.append((quantity, sheet, _hw_measured(entries, entry, key)))
    table = pd.DataFrame(rows, columns=['quantity', 'datasheet', 'measured']
                         ).set_index('quantity')
    table['fraction'] = table.measured / table.datasheet
    return table

#@save
def datasheet_vs_measured(row, *measured):
    """Print datasheet, measured and their fraction for the eight quantities
    of the chip row: measured is (ladder, dtypes, smem) or one committed
    campaign log (a dict of those entries). Measurements from a GPU that is
    not this row print one line instead; a pending campaign log prints its
    status and the datasheet column only."""
    entries = _hw_entries(measured)
    table = _hw_sheet_table(row, entries)
    if not entries:
        print(hw_status_line(measured[0]) + ' (datasheet values only)')
        show_table(table[['datasheet']].dropna())
        return
    gpu, mem = _hw_measured_gpu(entries)
    specs = d2l.chip_specs()
    if _hw_chip_row(gpu, mem, specs) != row.name:
        print(f'measured on {gpu} ({_hw_bytes(mem or 0)}), not the '
              f'{row.name}: no datasheet comparison')
        return
    show_table(pd.DataFrame({
        'datasheet': table.datasheet,
        'measured': [_hw_sig(x) for x in table.measured],
        'fraction': [f'{f:.0%}' if f == f else '' for f in table.fraction]
    }, index=table.index))
    smem = entries.get('smem') or {}
    if smem.get('peak_tbs') is not None and np.isfinite(smem['peak_tbs']):
        print(f"* bound, not a datasheet figure: {smem['n_sm']} SMs x "
              f"{_HW_L1_BYTES_PER_CLK} B/clk x "
              f"{_hw_sig(smem['sm_mhz'] / 1000, 3)} GHz")
```

## The Benchmark Log and the Two Ladders

Every measured point also goes into a log of records: the command, its
parameters and results, the raw tool output, the environment, and whether
another user's process shared the card. The ladders run the page's
`hw_copy_bench` and `hw_matmul_bench` over working sets and formats, each
point after a warm-up under a power trace; whether they measure or load the
committed run is `_hw_result`'s decision.

```{.python .input #accelerator-design-lib-benchlog}
%%tab pytorch
#@save
def env_fingerprint():
    """What a measurement ran on: software versions, the GPU model, count,
    SMs, clocks, power limits and PCIe link, sha1 prefixes of the host name
    and of `nvidia-smi topo -m`, the git commit (no names, no paths)."""
    env = _hw_env()
    env.update(_hw_gpu_limits())
    topo = hw_run(['nvidia-smi', 'topo', '-m'], timeout_s=60)
    env['topo_sha1'] = (None if topo.refused else
                        hashlib.sha1(topo.stdout.encode()).hexdigest()[:8])
    return env

#@save
class BenchLog:
    """An append-only JSONL log of microbenchmark records (tool, tier):
    each record holds the command, its parameters and results, the raw
    tool output (gzip + base64), the environment, and contended: a foreign
    compute process on the GPU when the log opened or at the record."""
    def __init__(self, path, tool, tier):
        self.path, self.tool, self.tier = path, tool, tier

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        self.env = d2l.env_fingerprint()
        self.foreign = d2l._hw_contention()[0]
        self.file = open(self.path, 'a', encoding='utf-8')
        return self

    def __exit__(self, *exc):
        self.file.close()

    def record(self, cmd, params, results, raw=''):
        rec = dict(ts=round(time.time(), 3), tool=self.tool, tier=self.tier,
                   cmd=cmd, params=params, results=results, env=self.env,
                   contended=bool(self.foreign or d2l._hw_contention()[0]),
                   raw=d2l._hw_pack_raw(raw))
        self.file.write(json.dumps(d2l._hw_encode(rec)) + '\n')
        self.file.flush()
        return rec

    @staticmethod
    def read(path):
        with open(path, encoding='utf-8') as f:
            recs = [d2l._hw_decode(json.loads(l)) for l in f if l.strip()]
        for r in recs:
            r['raw'] = d2l._hw_unpack_raw(r['raw'])
        return recs

    def power_trace(self, hz=10):
        """Samples (power, clocks, temperature, throttle reasons) at hz
        while the block runs (pynvml, else nvidia-smi)."""
        return d2l._hw_power_trace(hz)
```

```{.python .input #accelerator-design-lib-ladders}
%%tab pytorch
#@save
def hw_bandwidth_ladder(sizes, kinds=('read', 'copy')):
    """The bandwidth ladder: hw_copy_bench for each kind and working set
    (bytes; sizes too large for this GPU skipped), each point after a short
    warm-up, with the SM clock and board power while it ran."""
    sizes, kinds = [int(s) for s in sizes], list(kinds)
    def compute():  # largest first: a small point never starts on a cold card
        rows = [dict(bytes=s, kind=k, **_hw_steady(
                    lambda: d2l.hw_copy_bench(s, k)))
                for k in kinds for s in sorted(sizes, reverse=True)
                if _hw_fits(s * (2 if k == 'copy' else 1))]
        rows.sort(key=lambda r: (kinds.index(r['kind']), r['bytes']))
        return _hw_frame(rows)
    return _hw_result('box-microbench', compute,
                      dict(sizes=sizes, kinds=kinds), entry='ladder',
                      live_check=_ladder_live, show=_show_ladder,
                      live_label='read bandwidth')

#@save
def hw_dtype_ladder(n, formats):
    """Measured TFLOP/s of an n x n x n product in each format
    (hw_matmul_bench), each after a one-second warm-up, with the SM clock
    and board power while it ran, against the datasheet of the GPU."""
    formats = list(formats)
    def compute():
        df = _hw_frame([dict(format=f, **_hw_steady(
            lambda: d2l.hw_matmul_bench(n, n, n, f), warm_s=1.0))
            for f in formats])
        df.attrs['n'] = n
        return df
    return _hw_result('box-microbench', compute,
                      dict(n=n, formats=formats), entry='dtypes',
                      live_check=_dtype_live, show=_show_dtypes,
                      live_label='matmul')
```

## The Regeneration Driver

`tools/hw_regen.py box-microbench` calls this in each of its three processes.
It runs the page's calls on the full configuration, then the kernels
section's (`_regen_kernels`, in `kernels-lib.md`: entries three-ways,
shape-sweep, wave, rmsnorm, flash-sweep) and writes the fingerprint, `nvidia-smi topo -m` and one
BenchLog record per measured point next to the entries. With `--config t2`
the same calls run on the A100 of campaign C-T2 and write its log.

```{.python .input #accelerator-design-lib-driver}
%%tab pytorch
#@save
_HW_TIERS = {'box': 'T1', 't2': 'T2'}

#@save
def _hw_check_hardware(name, config):
    """Raise unless this GPU is the chip-table row of the campaign log that
    config writes (no check for the box run)."""
    if config == 'box':
        return
    hardware = d2l.HW_CAMPAIGNS[d2l.HW_CONFIG_LOGS[(name, config)]][3]
    p = torch.cuda.get_device_properties(0)
    chip = _hw_chip_row(_gpu_short(p.name), p.total_memory, d2l.chip_specs())
    if chip != hardware:
        raise RuntimeError(f'config {config} measures the {hardware}, not '
                           f'{p.name} ({p.total_memory / 2**30:.0f} GiB)')

#@save
def regen_microbench(config='box'):
    """Driver of box-microbench: the accelerator-design page's calls on the
    full configuration, each point also recorded in a BenchLog; config
    't2' is campaign C-T2 and runs only on the chip its log names."""
    if not torch.cuda.is_available():
        raise RuntimeError('box-microbench needs a CUDA GPU')
    if (os.environ.get('D2L_HW_CONFIG') or 'box') != config:
        raise ValueError(f'config {config!r} needs D2L_HW_CONFIG={config} '
                         '(tools/hw_regen.py --config)')
    _hw_check_hardware('box-microbench', config)
    folder = _hw_regen_dir('box-microbench')
    os.makedirs(folder, exist_ok=True)
    topo = hw_run(['nvidia-smi', 'topo', '-m'], timeout_s=60)
    with open(os.path.join(folder, 'topo.txt'), 'w') as f:
        f.write(topo.stdout)
    with d2l.BenchLog(os.path.join(folder, 'benchlog.jsonl'), 'torch',
                      _HW_TIERS[config]) as log:
        with open(os.path.join(folder, 'fingerprint.json'), 'w') as f:
            json.dump(_hw_encode(log.env), f, indent=1)
        ladder = d2l.hw_bandwidth_ladder(sizes=2**np.arange(18, 33),
                                         kinds=('read', 'copy'))
        fp8 = ('fp8',) if torch.cuda.get_device_capability() >= (8, 9) else ()
        dtypes = d2l.hw_dtype_ladder(  # no FP8 matrix units before Ada/Hopper
            n=8192, formats=('fp32', 'tf32', 'bf16', 'fp16') + fp8)
        smem = hw_smem_bench()
        lat = hw_latency_bench(sizes=2**np.arange(12, 31))
        kernels = _regen_kernels()  # sec_lm_kernels: kernels-lib.md
        for cmd, df in (('hw_copy_bench', ladder), ('hw_matmul_bench', dtypes),
                        ('pointer chase', lat), *kernels):
            for rec in df.to_dict('records'):
                log.record(cmd, {k: rec[k] for k in (
                    'bytes', 'kind', 'format', 'arm', 'name', 'M', 'N', 'K',
                    'shape') if k in rec}, rec)
        log.record('l1 read kernel', {}, smem)
```
