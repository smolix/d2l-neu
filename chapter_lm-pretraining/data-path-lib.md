# The Pretraining Data Path: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_data_path` explains in a sentence each but does not show
(docs/code-guide.md 7.1):

- the chapter's registries: Hub pins (`PRETRAIN_SOURCES`), committed results
  (`PRETRAIN_ARTIFACTS`, one statement per name; a keyed registry, so a new
  or re-versioned artifact stales only the notebooks that read it), the
  long regeneration jobs (`PRETRAIN_JOBS`), the regeneration names
  (`PRETRAIN_RERUN_NAMES`) and the campaigns declared pending
  (`PRETRAIN_PENDING`);
- the one place that decides between computing and loading a result
  (`_pretrain_result`, `_pretrain_merged`), `load_artifact`, and `_quiet`,
  under which every fetch of the chapter runs;
- the attention internals of `PackedAttention`/`PackedGPT`: RoPE tables from
  per-document positions, the backend choice, the boolean and FlexAttention
  document masks, explicit-score attention, the model's configuration and
  the Qwen3 shape it adopts;
- `PackedStream` internals (layout hash, permutation slice, window reads);
- the naive-restart contrast of the opening (`reseeded_restart_overlap`);
- the DDP helpers shared with 21.2 and the small-sample t interval.

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble inside its functions (`torch.distributed`, FlexAttention,
`pyarrow`, `safetensors`, `scipy`).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_data_path`

## Registries

```{.python .input #data-path-lib-sources}
%%tab pytorch
#@save
PRETRAIN_SOURCES = {}  # name: (kind, Hub repository, revision, licence)
```

```{.python .input #data-path-lib-source-pins}
%%tab pytorch
#@save
d2l.PRETRAIN_SOURCES['smollm3-stage1'] = (
    'model', 'HuggingFaceTB/SmolLM3-3B-checkpoints',
    'd07a5a83dd011f3f084e9d2f1b47f51e524ca8d4', 'apache-2.0')
d2l.PRETRAIN_SOURCES['smollm3-stage2'] = (
    'model', 'HuggingFaceTB/SmolLM3-3B-checkpoints',
    'eebe6fccea5aa05cf48b0e630d10eeee97f92fc5', 'apache-2.0')
d2l.PRETRAIN_SOURCES['smollm3-stage3'] = (
    'model', 'HuggingFaceTB/SmolLM3-3B-checkpoints',
    '20e7817e636d6b1f63cffbee0f384721b1f4bb67', 'apache-2.0')
d2l.PRETRAIN_SOURCES['smollm3-lc32k'] = (
    'model', 'HuggingFaceTB/SmolLM3-3B-checkpoints',
    '24cc7e71eb21218315ef02c257b192cdd263b302', 'apache-2.0')
d2l.PRETRAIN_SOURCES['smollm3-lc64k'] = (
    'model', 'HuggingFaceTB/SmolLM3-3B-checkpoints',
    'd4d10f417b85b7e385c163118183748bd67a34f6', 'apache-2.0')
d2l.PRETRAIN_SOURCES['smollm3-mid'] = (
    'model', 'HuggingFaceTB/SmolLM3-3B-checkpoints',
    '0485ec16b618f88f6dbb27b23dbdecca1d6a1cdd', 'apache-2.0')
d2l.PRETRAIN_SOURCES['smollm3-configs'] = (
    'dataset', 'HuggingFaceTB/smollm3-configs',
    '8d27ec7c533ebf42931a8bf156e3a534b1ffac48', 'none stated (cited only)')
d2l.PRETRAIN_SOURCES['pes2o'] = (
    'dataset', 'allenai/peS2o',
    '636a503e44a3ca1b58e01fb61eab0825cd574de0', 'odc-by')
d2l.PRETRAIN_SOURCES['pg19'] = (
    'dataset', 'emozilla/pg19',
    'd2f3c93e25bc848bd62bec2aa27a6bec14f4faf4', 'apache-2.0')
```

```{.python .input #data-path-lib-registries}
%%tab pytorch
#@save
PRETRAIN_ARTIFACTS = {}  # load name: (DATA_HUB key, kind)

#@save
PRETRAIN_JOBS = {}  # regeneration name: (driver, zero-argument kwargs)

#@save
PRETRAIN_PENDING = frozenset()  # campaigns an author step declared pending

#@save
PRETRAIN_RERUN_NAMES = (  # every regeneration name of the chapter
    'pdp-throughput', 'pdp-mask-runs',
    't1-runs', 'inst-t1-none', 'inst-t1-qkclip', 'inst-t1-qknorm',
    'inst-t1-softcap', 'spike-replay', 'reference-run', 'ladder', 'inst-t2',
    'reference-evals', 'ladder-evals',
    'gemm-sweep', 'lowp-runs',
    'cpt-t1', 'cpt-qwen-A', 'cpt-qwen-B', 'cpt-qwen-C', 'cpt-qwen-A2',
    'cpt-qwen-evals', 'smollm3-0', 'smollm3-1', 'smollm3-2', 'smollm3-3',
    'smollm3-4', 'smollm3-5',
    'lc-t1', 'lc-zeroshot', 'lc-t2-pi', 'lc-t2-yarn', 'lc-t2-prog',
    'lc-t2-evals',
    'decode-rates')

#@save
_RERUN_ARTIFACT = {  # regeneration name: the load name of what it writes
    'pdp-throughput': 'packing-throughput',
    'pdp-mask-runs': 'packing-mask-runs',
    't1-runs': 't1-runs', 'inst-t1-none': 'instability-t1',
    'inst-t1-qkclip': 'instability-t1', 'inst-t1-qknorm': 'instability-t1',
    'inst-t1-softcap': 'instability-t1', 'spike-replay': 'spike-replay',
    'reference-run': 'reference-run-log', 'ladder': 'ladder-logs',
    'inst-t2': 'instability-t2', 'reference-evals': 'reference-evals',
    'ladder-evals': 'ladder-evals', 'gemm-sweep': 'gemm-sweep',
    'lowp-runs': 'lowp-runs', 'cpt-t1': 'cpt-t1', 'cpt-qwen-A': 'cpt-runs',
    'cpt-qwen-B': 'cpt-runs', 'cpt-qwen-C': 'cpt-runs',
    'cpt-qwen-A2': 'cpt-runs', 'cpt-qwen-evals': 'cpt-runs',
    **{f'smollm3-{i}': 'smollm3-trajectory' for i in range(6)},
    'lc-t1': 'lc-t1', 'lc-zeroshot': 'lc-zeroshot', 'lc-t2-pi': 'lc-evals',
    'lc-t2-yarn': 'lc-evals', 'lc-t2-prog': 'lc-evals',
    'lc-t2-evals': 'lc-evals', 'decode-rates': 'decode-rates'}
```

21.1's committed results (the `DATA_HUB` entries follow the uploads).

```{.python .input #data-path-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['packing-throughput'] = (
    'lm-pdp-throughput-v1', 'parquet')
d2l.PRETRAIN_ARTIFACTS['packing-mask-runs'] = (
    'lm-pdp-mask-runs-v1', 'parquet')
```

## Committed Results: Load or Regenerate

```{.python .input #data-path-lib-results}
%%tab pytorch
#@save
_ARTIFACT_EXT = {'jsonl': '.jsonl.gz', 'parquet': '.parquet',
                 'json-table': '.json.gz', 'safetensors': '.safetensors',
                 'tar': '.tar', 'shards': '.tar'}

#@save
def _rerun_dir():
    """Where regeneration runs write (relative to a chapter folder)."""
    return os.path.join('..', 'data', 'lm', 'rerun')

#@save
def _next_key(key):
    """The key a regeneration of key writes: key itself while it is not
    registered, else the next free -vN."""
    if key not in d2l.DATA_HUB:
        return key
    stem, v = key.rsplit('-v', 1)
    n = int(v) + 1
    while f'{stem}-v{n}' in d2l.DATA_HUB:
        n += 1
    return f'{stem}-v{n}'

#@save
def _regenerating():
    """The chapter's regeneration names in D2L_RERUN; D2L_RERUN=1 (every
    campaign of the chapter) is refused."""
    if os.environ.get('D2L_RERUN', '').strip() == '1':
        raise ValueError('D2L_RERUN=1 would launch every campaign of this '
                         'chapter; name one or more of: '
                         + ', '.join(d2l.PRETRAIN_RERUN_NAMES))
    return d2l.lm_rerun(d2l.PRETRAIN_RERUN_NAMES)

#@save
def _jsonable(x):
    """x as plain JSON values (tuples become lists, namedtuples dicts)."""
    if hasattr(x, '_asdict'):
        x = x._asdict()
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        x = {f.name: getattr(x, f.name) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x

#@save
def _label_of(result):
    if hasattr(result, 'attrs'):
        return result.attrs.get('label', '')
    return getattr(result, 'label', '') or ''

#@save
def _with_label(result, label, meta=None):
    """result with its label (and stored meta) set."""
    if hasattr(result, 'attrs'):
        result.attrs['label'] = label
        if meta is not None:
            result.attrs['config'] = meta.get('config')
        return result
    if hasattr(result, '_replace') and 'label' in result._fields:
        return result._replace(label=label)
    return result

#@save
def _write_result(path, kind, result, meta):
    """Write result (a DataFrame; for json-table also a mapping or
    namedtuple of DataFrames and plain values) with meta = {config, label},
    through a temporary file."""
    import gzip
    tmp = f'{path}.{os.getpid()}.tmp'
    if kind == 'parquet':
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(result, preserve_index=False)
        table = table.replace_schema_metadata(dict(
            table.schema.metadata or {}, d2l=json.dumps(meta)))
        pq.write_table(table, tmp)
    elif kind == 'jsonl':
        with gzip.open(tmp, 'wt', encoding='utf-8') as f:
            f.write(json.dumps({'_meta': meta}) + '\n')
            for rec in result.to_dict('records'):
                f.write(json.dumps(_jsonable(rec)) + '\n')
    elif kind == 'json-table':
        parts = (result._asdict() if hasattr(result, '_asdict') else
                 dict(result) if isinstance(result, dict) else
                 {'table': result})
        doc = dict(meta=meta, tables={}, values={})
        for k, v in parts.items():
            if isinstance(v, pd.DataFrame):
                doc['tables'][k] = json.loads(v.to_json(orient='split',
                                                        index=False))
            else:
                doc['values'][k] = _jsonable(v)
        with gzip.open(tmp, 'wt', encoding='utf-8') as f:
            json.dump(doc, f)
    else:
        raise ValueError(f'cannot write kind {kind!r}')
    os.replace(tmp, path)

#@save
def _read_result(path, kind, key=None):
    """A stored result: a DataFrame (with attrs label and config), a
    namedtuple of tables and values, a state dict, a folder, or shards."""
    import gzip
    if kind == 'parquet':
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        meta = json.loads((table.schema.metadata or {}).get(b'd2l', b'{}'))
        return _with_label(table.to_pandas(), meta.get('label', ''), meta)
    if kind == 'jsonl':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            recs = [json.loads(line) for line in f]
        meta = recs.pop(0)['_meta'] if recs and '_meta' in recs[0] else {}
        return _with_label(pd.DataFrame(recs), meta.get('label', ''), meta)
    if kind == 'json-table':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            doc = json.load(f)
        meta, tables = doc['meta'], {}
        for k, t in doc['tables'].items():
            tables[k] = _with_label(pd.DataFrame(t['data'],
                                                 columns=t['columns']),
                                    meta.get('label', ''), meta)
        fields = dict(tables, **doc['values'])
        if list(fields) == ['table']:
            return tables['table']
        Tables = collections.namedtuple('Tables', list(fields) + (
            [] if 'label' in fields else ['label']))
        return Tables(**fields, **({} if 'label' in fields else
                                   {'label': meta.get('label', '')}))
    if kind == 'safetensors':
        from safetensors.torch import load_file
        return load_file(path)
    if kind in ('tar', 'shards'):
        folder = os.path.join(os.path.dirname(path), 'extracted')
        if not os.path.exists(folder):
            d2l.extract(path, folder)
        if kind == 'tar':
            return folder
        with open(os.path.join(folder, 'manifest.json')) as f:
            split = next(iter(json.load(f)['splits']))
        return d2l.TokenShards(folder, split)
    raise ValueError(f'unknown kind {kind!r}')

#@save
def _stored_config(result):
    if hasattr(result, 'attrs'):
        return result.attrs.get('config')
    tables = [v for v in result if isinstance(v, pd.DataFrame)]
    return tables[0].attrs.get('config') if tables else None

#@save
def _config_diff(page, stored, field=''):
    """The first field where two JSON values differ, or None."""
    if isinstance(page, dict) and isinstance(stored, dict):
        for k in sorted(set(page) | set(stored)):
            d = _config_diff(page.get(k), stored.get(k), f'{field}.{k}'
                             if field else k)
            if d:
                return d
        return None
    if isinstance(page, list) and isinstance(stored, list) and len(
            page) == len(stored):
        for i, (a, b) in enumerate(zip(page, stored)):
            d = _config_diff(a, b, f'{field}[{i}]')
            if d:
                return d
        return None
    return None if page == stored else (field, page, stored)

#@save
def _check_config(result, config, key):
    """Raise if the configuration stored with result differs from config."""
    d = _config_diff(_jsonable(config), _stored_config(result))
    if d:
        raise ValueError(f'page configuration differs from the committed '
                         f'run {key}: {d[0]} {d[1]!r} != {d[2]!r}')

#@save
def load_artifact(name):
    """A committed result of this chapter by its name (PRETRAIN_ARTIFACTS):
    a table or log as a DataFrame, weights as a state dict, an archive as a
    folder, a shard set as TokenShards. A regeneration run reads the file it
    has just written (the next version) when there is one."""
    if name not in d2l.PRETRAIN_ARTIFACTS:
        raise KeyError(f'unknown result {name!r}; one of '
                       f'{sorted(d2l.PRETRAIN_ARTIFACTS)}')
    key, kind = d2l.PRETRAIN_ARTIFACTS[name]
    local = os.path.join(_rerun_dir(), _next_key(key) + _ARTIFACT_EXT[kind])
    if _regenerating() and os.path.exists(local):
        return _read_result(local, kind, key)
    if key not in d2l.DATA_HUB:
        raise KeyError(f'{name} is not committed yet')
    with _quiet():
        path = d2l.download_keyed(key)
    return _read_result(path, kind, key)

#@save
def _pending_result(name, fallback=None):
    """What a pending result shows: the fallback result labelled as such, or
    an empty table labelled '<name> pending'."""
    if fallback is not None:
        try:
            result = load_artifact(fallback)
        except KeyError:
            result = None
        if result is not None:
            return _with_label(result, 'T2 run pending; showing the T1 '
                                       'configuration')
    return _with_label(pd.DataFrame(), f'{name} pending')

#@save
def _pretrain_result(name, compute, config, *, fallback=None):
    """The result of regeneration name for the call's configuration:
    computed (and written as the next version) when D2L_RERUN names it;
    else the committed result, whose stored configuration must equal
    config; else, in a regeneration run or for a campaign declared pending,
    the pending result; else KeyError. compute() may return None (parts of
    the result are still missing): the pending result, nothing written."""
    assert name in d2l.PRETRAIN_RERUN_NAMES, name
    artifact = d2l._RERUN_ARTIFACT[name]
    key, kind = d2l.PRETRAIN_ARTIFACTS[artifact]
    config, rerun = _jsonable(config), _regenerating()
    path = os.path.join(_rerun_dir(), _next_key(key) + _ARTIFACT_EXT[kind])
    if name in rerun:
        result = compute()
        if result is None:
            return _pending_result(name, fallback)
        meta = dict(config=config, label=_label_of(result))
        d2l.lm_save_rerun(os.path.basename(path), lambda p: _write_result(
            p, kind, result, meta), rerun_dir=_rerun_dir())
        return _with_label(result, meta['label'], meta)
    if key in d2l.DATA_HUB or (rerun and os.path.exists(path)):
        result = load_artifact(artifact)
        _check_config(result, config, key)
        return result
    if rerun or name in d2l.PRETRAIN_PENDING:
        return _pending_result(name, fallback)
    raise KeyError(f'{name} is not committed yet')

#@save
def _pretrain_merged(name, parts, merge, config, *, fallback=None):
    """A result merged from parts {regeneration name: (compute, config)},
    each written to its own file when D2L_RERUN names it. The merge
    (merge({part: result})) runs in a regeneration run once every part's
    file exists, and only when D2L_RERUN names the merge step name (if not
    None). Loading, configuration check and pending as _pretrain_result."""
    artifact = d2l._RERUN_ARTIFACT[next(iter(parts))]
    key, kind = d2l.PRETRAIN_ARTIFACTS[artifact]
    names = set(parts) | ({name} if name else set())
    assert names <= set(d2l.PRETRAIN_RERUN_NAMES), names
    config, rerun = _jsonable(config), _regenerating()
    nxt, ext = _next_key(key), _ARTIFACT_EXT[kind]
    if key in d2l.DATA_HUB and not rerun & names:
        result = load_artifact(artifact)
        _check_config(result, config, key)
        return result
    if not rerun:
        if name in d2l.PRETRAIN_PENDING or names & d2l.PRETRAIN_PENDING:
            return _pending_result(name or artifact, fallback)
        raise KeyError(f'{artifact} is not committed yet')
    results = {}
    for part, (compute, part_config) in parts.items():
        path = os.path.join(_rerun_dir(), f'{nxt}.{part}{ext}')
        meta = dict(config=_jsonable(part_config), label='')
        if part in rerun:
            result = compute()
            meta['label'] = _label_of(result)
            d2l.lm_save_rerun(os.path.basename(path), lambda p: _write_result(
                p, kind, result, meta), rerun_dir=_rerun_dir())
        elif os.path.exists(path):
            result = _read_result(path, kind)
            _check_config(result, part_config, f'{nxt}.{part}')
        else:
            result = None
        results[part] = result
    if any(r is None for r in results.values()) or (name and
                                                    name not in rerun):
        return _pending_result(name or artifact, fallback)
    merged = merge(results)
    meta = dict(config=config, label=_label_of(merged))
    d2l.lm_save_rerun(nxt + ext, lambda p: _write_result(p, kind, merged,
                                                         meta),
                      rerun_dir=_rerun_dir())
    return _with_label(merged, meta['label'], meta)
```

```{.python .input #data-path-lib-quiet}
%%tab pytorch
import contextlib  #@save

@contextlib.contextmanager
def _quiet():
    """Swallow what a fetch prints (download lines, progress bars), its
    benign warnings and the Transformers/Hub logs below error; an exception
    carries the swallowed text as a note."""
    import io
    import logging
    import warnings
    buf = io.StringIO()
    loggers = [logging.getLogger(n) for n in ('transformers',
                                              'huggingface_hub', 'datasets')]
    levels = [lg.level for lg in loggers]
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(
                buf), warnings.catch_warnings():
            for category in (FutureWarning, DeprecationWarning, UserWarning):
                warnings.simplefilter('ignore', category)
            for lg in loggers:
                lg.setLevel(logging.ERROR)
            yield buf
    except Exception as e:
        e.add_note('output of the failed call:\n' + buf.getvalue()[-4000:])
        raise
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)
```

## Attention Internals

```{.python .input #data-path-lib-attention}
%%tab pytorch
#@save
def _rope_tables(pos, head_dim, theta, inv_freq=None, mscale=1.0):
    """Cosines and sines of the RoPE angles of positions pos, (T,) or (B, T),
    in the layout d2l.rope broadcasts ((T, d/2) or (B, 1, T, d/2)); the
    frequencies as qwen3_rope_tables computes them unless inv_freq is
    given; both tables scaled by mscale (YaRN's attention scale)."""
    if inv_freq is None:
        inv_freq = 1.0 / float(theta) ** (torch.arange(
            0, head_dim, 2, dtype=torch.float32, device='cpu') / head_dim)
    angles = pos.float()[..., None] * inv_freq.to(pos.device)
    cos, sin = torch.cos(angles), torch.sin(angles)
    if mscale != 1.0:
        cos, sin = cos * mscale, sin * mscale
    if pos.dim() == 2:
        cos, sin = cos[:, None], sin[:, None]
    return cos, sin

#@save
def _flex_available():
    if not hasattr(_flex_available, 'ok'):
        try:
            from torch.nn.attention import flex_attention  # noqa: F401
            _flex_available.ok = True
        except ImportError:
            _flex_available.ok = False
    return _flex_available.ok

#@save
def _attn_backend(attn, X, ctx):
    """'causal' (one document per row, Qwen3Attention's path), or the
    backend that applies the document mask: 'math' for a softcap."""
    if ctx is None and attn.softcap is None and attn.backend != 'math':
        return 'causal'
    if attn.softcap is not None or attn.backend == 'math':
        return 'math'
    if attn.backend == 'auto':
        return 'flex' if X.is_cuda and _flex_available() else 'sdpa'
    return attn.backend

#@save
def _flex_attention():
    """flex_attention, compiled once when called outside a compiled region
    (inside one, the region's compiler handles it)."""
    from torch.nn.attention.flex_attention import flex_attention
    if torch.compiler.is_compiling():
        return flex_attention
    if not hasattr(_flex_attention, 'fn'):
        _flex_attention.fn = torch.compile(flex_attention, dynamic=False)
    return _flex_attention.fn

#@save
_FLEX_MASKS = {}  # (B, T, device): (doc_ids buffer, mask_mod)

#@save
@torch.compiler.disable
def _flex_doc_mask(doc_ids):
    """FlexAttention block mask of packed windows. The doc_ids are copied
    into a buffer kept per (B, T, device), so the mask predicate keeps its
    identity across steps and nothing retraces; the mask is valid until
    the next call with the same shape (one forward and backward)."""
    from torch.nn.attention.flex_attention import create_block_mask
    B, T = doc_ids.shape
    key = (B, T, str(doc_ids.device))
    if key not in _FLEX_MASKS:
        buf = torch.empty_like(doc_ids)
        _FLEX_MASKS[key] = (buf, d2l.doc_mask_mod(buf))
    buf, mask_mod = _FLEX_MASKS[key]
    buf.copy_(doc_ids)
    return create_block_mask(mask_mod, B, None, T, T, device=doc_ids.device)

#@save
def _doc_mask(ctx, backend, T, device):
    """The mask of backend: a FlexAttention block mask, or a boolean mask
    (B, 1, T, T) of same-document and causal (causal only without ctx)."""
    if backend == 'flex':
        return _flex_doc_mask(ctx.doc_ids)
    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    if ctx is None:
        return causal
    d = ctx.doc_ids
    return ((d[:, :, None] == d[:, None, :]) & causal)[:, None]

#@save
def _packed_mask(model, ctx, X):
    """The mask every block of a PackedGPT uses, built once per forward."""
    backend = _attn_backend(model.blks[0].attention, X, ctx)
    return None if backend == 'causal' else _doc_mask(ctx, backend,
                                                      X.shape[1], X.device)

#@save
def _math_attention(q, k, v, mask, softcap=None):
    """Attention from explicit fp32 scores (optionally soft-capped)."""
    g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    s = q.float() @ k.float().transpose(-2, -1) / math.sqrt(q.shape[-1])
    if softcap is not None:
        s = d2l.softcap(s, softcap)
    s = s.masked_fill(~mask, float('-inf'))
    return (torch.softmax(s, -1) @ v.float()).to(q.dtype)

#@save
_PACKED_ARGS = ('vocab_size', 'num_hiddens', 'num_heads', 'num_kv_heads',
                'num_blks', 'head_dim', 'qk_norm', 'rope_theta', 'ffn_hidden',
                'norm_eps', 'tie_embeddings', 'max_len', 'bias', 'pos',
                'rope_inv_freq', 'rope_mscale', 'softcap', 'backend')

#@save
def _packed_config(model):
    """PackedGPT.config(): constructor arguments (max_len as now accepted)
    plus rope_scheme and rope_scale."""
    a = model.blks[0].attention
    inv = model.rope_inv_freq
    return dict(
        vocab_size=model.token_emb.num_embeddings,
        num_hiddens=model.token_emb.embedding_dim, num_heads=a.num_heads,
        num_kv_heads=a.num_kv_heads, num_blks=len(model.blks),
        head_dim=a.head_dim, qk_norm=a.qk_norm,
        rope_theta=float(model.rope_theta),
        ffn_hidden=model.blks[0].ffn.W_1.out_features,
        norm_eps=model.norm.eps, tie_embeddings=not hasattr(model, 'lm_head'),
        max_len=model.max_len, bias=a.W_q.bias is not None, pos=model.pos,
        rope_inv_freq=None if inv is None else inv.tolist(),
        rope_mscale=model.rope_mscale, softcap=a.softcap, backend=a.backend,
        rope_scheme=model.rope_scheme, rope_scale=model.rope_scale)

#@save
def _packed_from_config(config, meta=False, **kw):
    """The PackedGPT a config() describes (on the meta device if meta)."""
    args = {k: config[k] for k in _PACKED_ARGS if k in config}
    scale = config.get('rope_scale', 1.0)
    args['max_len'] = round(args['max_len'] / scale)  # the trained length
    inv, m = args.pop('rope_inv_freq', None), args.pop('rope_mscale', 1.0)
    args.update(kw)
    with torch.device('meta') if meta else contextlib.nullcontext():
        model = d2l.PackedGPT(**args)
    model.set_rope(inv, m, config.get('rope_scheme', 'none'), scale)
    return model

#@save
def _qwen3_shape(model):
    """PackedGPT's constructor arguments for the shape of a d2l.Qwen3GPT."""
    a, blk = model.blks[0].attention, model.blks[0]
    return dict(vocab_size=model.token_emb.num_embeddings,
                num_hiddens=model.token_emb.embedding_dim,
                num_heads=a.num_heads, num_kv_heads=a.num_kv_heads,
                num_blks=len(model.blks), head_dim=a.head_dim,
                qk_norm=a.qk_norm, rope_theta=model.rope_theta,
                ffn_hidden=blk.ffn.W_1.out_features, norm_eps=model.norm.eps,
                tie_embeddings=not hasattr(model, 'lm_head'),
                max_len=model.max_len, bias=a.W_q.bias is not None)

#@save
def _check_same_tensors(mine, theirs):
    """Raise on the first name whose tensor is missing or differs in shape."""
    for name in sorted(set(mine) | set(theirs)):
        if name not in theirs or name not in mine:
            side = 'source' if name not in theirs else 'PackedGPT'
            raise ValueError(f'adopt: {name} is missing in the {side}')
        if mine[name].shape != theirs[name].shape:
            raise ValueError(f'adopt: {name} has shape '
                             f'{tuple(theirs[name].shape)}, PackedGPT '
                             f'expects {tuple(mine[name].shape)}')
```

## Stream Internals

```{.python .input #data-path-lib-stream}
%%tab pytorch
#@save
def _layout_hash(shards):
    """SHA-256 over the manifest's per-file SHA-256 values, in order."""
    h = hashlib.sha256()
    for s in shards.shards:
        for ext in d2l._SHARD_EXT:
            h.update(s['sha256'][ext].encode())
    return h.hexdigest()

#@save
def _window_indices(stream, p):
    """The global batch of physical step p: a slice of the permutation of
    the window indices seeded by (seed, epoch)."""
    e = stream.epoch + p // stream.steps_per_epoch
    j = p % stream.steps_per_epoch
    cached = getattr(stream, '_perm', None)
    if cached is None or cached[0] != (stream.seed, e):
        rng = np.random.default_rng([stream.seed, e])
        stream._perm = ((stream.seed, e), rng.permutation(stream.n_seq))
    G = stream.global_batch
    return stream._perm[1][j * G:(j + 1) * G]

#@save
def _windows(stream, idx):
    """Windows idx of seq_len + 1 tokens, as an int64 array."""
    T = stream.seq_len
    return np.stack([stream.shards.read(int(i) * T, T + 1)
                     for i in idx]).astype(np.int64)

#@save
RestartOverlap = collections.namedtuple(
    'RestartOverlap', ['naive_match', 'naive_repeats', 'stateless_match',
                       'stateless_repeats', 'steps'])

#@save
def reseeded_restart_overlap(n_seq=10_000, batch=64, crash_step=40,
                             steps=50, seed=0):
    """A run that crashes at crash_step and restarts with a re-seeded
    shuffle, against a stateless cursor: batches equal to the uninterrupted
    run's, and windows seen twice."""
    perm = np.random.default_rng([seed, 0]).permutation(n_seq)
    after = np.random.default_rng([seed + 1, 0]).permutation(n_seq)
    run = [perm[s * batch:(s + 1) * batch] for s in range(steps)]
    naive = run[:crash_step] + [after[s * batch:(s + 1) * batch]
                                for s in range(crash_step, steps)]
    stateless = run[:crash_step] + [perm[s * batch:(s + 1) * batch]
                                    for s in range(crash_step, steps)]

    def overlap(batches):  # (batches equal to the run's, windows repeated)
        seen = np.concatenate(batches)
        return (int(sum(np.array_equal(a, b) for a, b in zip(run, batches))),
                int(len(seen) - len(np.unique(seen))))
    return RestartOverlap(*overlap(naive), *overlap(stateless), steps)
```

## DDP Helpers and Small-Sample Intervals

```{.python .input #data-path-lib-ddp}
%%tab pytorch
#@save
def _ddp_setup():
    """Join the process group torchrun describes: (rank, world, device)."""
    import torch.distributed as dist
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    cuda = torch.cuda.is_available()
    device = torch.device('cuda', local) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group('nccl' if cuda else 'gloo',
                                device_id=device if cuda else None)
    return rank, world, device

#@save
def _rank_world():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1

#@save
def _all_reduce(x, op):
    import torch.distributed as dist
    if _rank_world()[1] == 1 or not torch.is_tensor(x):
        return x
    x = x.clone()
    dist.all_reduce(x, op=getattr(dist.ReduceOp, op))
    return x

#@save
def _all_reduce_sum(x):
    """x summed over the ranks (x itself without a process group)."""
    return _all_reduce(x, 'SUM')

#@save
def _all_reduce_max(x):
    return _all_reduce(x, 'MAX')

#@save
def _ddp_module(model):
    m = getattr(model, '_orig_mod', model)  # under torch.compile
    return m if isinstance(m, torch.nn.parallel.DistributedDataParallel) \
        else None

#@save
_FORCE_NO_SYNC = [False]  # the throughput probe times steps without sync

#@save
def _no_sync(model, skip):
    """DDP's no_sync() while skip (micro-batches before the last), else a
    null context."""
    ddp = _ddp_module(model)
    if ddp is not None and (skip or _FORCE_NO_SYNC[0]):
        return ddp.no_sync()
    return contextlib.nullcontext()

#@save
def _unwrap(model):
    """The module inside torch.compile and DistributedDataParallel."""
    model = getattr(model, '_orig_mod', model)
    return getattr(model, 'module', model) if isinstance(
        model, torch.nn.parallel.DistributedDataParallel) else model

#@save
_T975 = (12.7062, 4.30265, 3.18245, 2.77645, 2.57058, 2.44691, 2.36462,
         2.30600, 2.26216, 2.22814)  # Student t, 97.5% point, dof 1-10

#@save
def t_interval(x, alpha=0.05):
    """(mean, half-width) of the two-sided 1 - alpha t interval of x."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if alpha == 0.05 and 2 <= n <= 11:
        t = _T975[n - 2]
    else:
        from scipy import stats
        t = float(stats.t.ppf(1 - alpha / 2, n - 1))
    return float(x.mean()), float(t * x.std(ddof=1) / math.sqrt(n))
```
