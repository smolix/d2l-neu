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
  only by `tools/hw_regen.py`), the results the author declared pending
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
- 22.1's registry entries.

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

22.1's entries: its regeneration, its static table and the A100 campaign
(C-T2) logs. The `DATA_HUB` entries follow the uploads.

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
the per-process medians.

```{.python .input #accelerator-design-lib-combine}
%%tab pytorch
#@save
def _hw_combine(results):
    """One result from the same result of several processes: numbers that
    differ become their median (elementwise in tables, arrays, lists and
    the fields of dicts and namedtuples); everything else must agree."""
    first = results[0]
    if isinstance(first, pd.DataFrame):
        for r in results[1:]:
            if not (isinstance(r, pd.DataFrame) and r.shape == first.shape
                    and list(r.columns) == list(first.columns)
                    and r.index.equals(first.index)):
                raise ValueError('tables differ in shape between processes')
        out = first.copy()
        for c in first.columns:
            cols = [r[c].to_numpy() for r in results]
            if all(pd.Series(x).equals(pd.Series(cols[0])) for x in cols):
                continue
            if (pd.api.types.is_numeric_dtype(first[c])
                    and not pd.api.types.is_bool_dtype(first[c])):
                out[c] = np.median(np.stack(cols).astype(float), axis=0)
            else:
                raise ValueError(f'column {c} differs between processes')
        out.attrs = dict(first.attrs, n_proc=len(results))
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
def _hw_regen_dir(name):
    """The folder a regeneration of name writes (the next version of its
    key); D2L_HW_PROC (set by tools/hw_regen.py) selects a process's own
    subfolder."""
    key, _ = d2l.HW_ARTIFACTS[name]
    folder = os.path.join(_hw_rerun_root(), d2l._next_key(key))
    proc = os.environ.get('D2L_HW_PROC')
    return os.path.join(folder, f'proc{proc}') if proc else folder

#@save
def _hw_folder(name):
    """The folder holding name's committed entries: the registered
    artifact, else (in a regeneration run) the folder an earlier
    regeneration wrote, else None."""
    key, kind = d2l.HW_ARTIFACTS[name]
    if key in d2l.DATA_HUB:
        with d2l._quiet():
            path = d2l.download_keyed(key)
        return _hw_unpack(path, kind)
    if _hw_regenerating():
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
def _hw_live(live_check, result, meta):
    """Print the one agreement line of a live check: compared when this GPU
    is the committed run's GPU, else this GPU's values uncompared. A live
    check never fails a run and never changes the committed numbers."""
    if live_check is None or not torch.cuda.is_available():
        return
    gpu = torch.cuda.get_device_name(0)
    same = gpu == (meta.get('env') or {}).get('gpu')
    try:
        with d2l._quiet():
            text = live_check(result, same)
    except Exception as e:  # report, never fail the page
        text = f'not run ({type(e).__name__})'
    print(f'live {"check" if same else "values"} on {_gpu_short(gpu)}: '
          f'{text}')

#@save
def _hw_result(name, compute, config, *, entry=None, live_check=None,
               show=None, empty=None):
    """The result of regeneration name (an HW_ARTIFACTS name; entry names
    one part of a multi-part result) for the call's configuration (a
    JSON-able dict of its arguments).

    - D2L_RERUN names it: compute() runs and its result is written, with
      config and the measurement environment, under the next version of the
      key (tools/hw_regen.py runs this; never a notebook).
    - Else the committed entry, whose stored configuration must equal
      config (ValueError naming the first difference), then show(result)
      and the one line of live_check(result, same_gpu).
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
        _hw_live(live_check, result, meta)
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
    significant digits, thousands separated; NaN as an empty cell."""
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
