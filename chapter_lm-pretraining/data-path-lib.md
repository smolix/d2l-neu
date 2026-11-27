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
- `reference_shards` (the FineWeb-Edu shards of 20.6, fetched quietly) and
  `chunked_lm_loss_and_nll` (the loss with the NLL that `pretrain_step`
  logs);
- the attention internals of `PackedAttention`/`PackedGPT`: RoPE tables from
  per-document positions, the backend choice (and the argument checks), the
  boolean and FlexAttention document masks, explicit-score attention, the
  model's configuration and the Qwen3 shape it adopts; and the methods later
  sections use (`set_rope`, `forward_cached`, `config`, `from_arch`,
  `adopt`), merged into the classes by `make lib`;
- `PackedStream` internals (layout hash, skip map, permutation slice, window
  reads, `physical_index`, `skip`);
- the naive-restart contrast of the opening (`reseeded_restart_overlap`);
- the DDP helpers shared with 21.2 and the small-sample t interval;
- the drivers behind the section's two measurements, `packing_throughput`
  (four input pipelines timed on one GPU) and `packing_comparison`
  (masked against unmasked training, three paired seeds, and the attention
  leak of the unmasked model), each computing under its regeneration name
  and otherwise loading its committed result.

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble inside its functions (`torch.distributed`, FlexAttention,
`pyarrow`, `safetensors`, `scipy`).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_data_path`

## Registries

The hosted results (d2l-data R2), one entry per upload.

```{.python .input #data-path-lib-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-pdp-throughput-v1'] = (
    d2l.LM_DATA_URL + 'lm-pdp-throughput-v1.parquet',
    '5fbb3d679b29faee9cb5c299ecf1131c399b8ff4')
d2l.DATA_HUB['lm-pdp-mask-runs-v1'] = (
    d2l.LM_DATA_URL + 'lm-pdp-mask-runs-v1.json.gz',
    '17ed3675834450ad7c0bfaf6b6183095880ec525')
```

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
    'inst-t1-softcap', 'inst-t1-clip', 'inst-t1-warmup', 'size-xfer-pilot',
    'size-xfer', 'spike-replay', 'eps-headroom', 'reference-run', 'ladder',
    'inst-t2', 'reference-evals', 't1-evals', 'ladder-evals',
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
    'inst-t1-softcap': 'instability-t1', 'inst-t1-clip': 'instability-t1',
    'inst-t1-warmup': 'instability-t1', 'size-xfer-pilot': 'size-transfer',
    'size-xfer': 'size-transfer',
    'spike-replay': 'spike-replay', 'eps-headroom': 'eps-headroom',
    'reference-run': 'reference-run-log', 'ladder': 'ladder-logs',
    'inst-t2': 'instability-t2', 'reference-evals': 'reference-evals',
    't1-evals': 't1-evals', 'ladder-evals': 'ladder-evals', 'gemm-sweep': 'gemm-sweep',
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
    'lm-pdp-mask-runs-v1', 'json-table')
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
def _regenerates(artifact):
    """Whether D2L_RERUN names a campaign that writes artifact (a load
    name); only such a run reads the local next-version file of it."""
    return any(d2l._RERUN_ARTIFACT.get(name) == artifact
               for name in _regenerating())

#@save
def _label_of(result):
    if hasattr(result, 'attrs'):
        return result.attrs.get('label', '')
    return getattr(result, 'label', '') or ''

#@save
def _show_label(result):
    """Print a result's label (a smoke, T1 or pending run) on its own line,
    so that it never passes for the committed run; return the result."""
    if _label_of(result):
        print(f'[{_label_of(result)}]')
    return result

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
        if not os.path.exists(folder):  # whole or not at all
            tmp = f'{folder}.{os.getpid()}.tmp'
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            d2l.extract(path, tmp)
            os.replace(tmp, folder)
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
    folder, a shard set as TokenShards. A run that regenerates it reads the
    file it has just written (the next version) when there is one."""
    if name not in d2l.PRETRAIN_ARTIFACTS:
        raise KeyError(f'unknown result {name!r}; one of '
                       f'{sorted(d2l.PRETRAIN_ARTIFACTS)}')
    key, kind = d2l.PRETRAIN_ARTIFACTS[name]
    local = os.path.join(_rerun_dir(), _next_key(key) + _ARTIFACT_EXT[kind])
    if _regenerates(name) and os.path.exists(local):
        return _read_result(local, kind, key)
    if key not in d2l.DATA_HUB:
        raise KeyError(f'{name} is not committed yet')
    with _quiet():
        path = d2l.download_keyed(key)
    return _read_result(path, kind, key)

#@save
def _pending_result(name, fallback=None):
    """What a pending result shows: the fallback result labelled as such
    (keeping its own label, such as 't1 preset'), or an empty table
    labelled '<name> pending'."""
    if fallback is not None:
        try:
            result = load_artifact(fallback)
        except KeyError:
            result = None
        if result is not None:
            own = _label_of(result)
            label = 'T2 run pending; showing the T1 configuration'
            return _with_label(result, f'{label} ({own})' if own else label)
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
    if key in d2l.DATA_HUB or (_regenerates(artifact) and
                               os.path.exists(path)):
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
    None). A run computes its own parts before it looks for the others'
    files, so of parts run side by side the last to finish merges; with
    D2L_PRETRAIN_MERGE=1 (tools/lm_regen.py --merge) a named part whose
    file exists is read, not computed. Loading, configuration check and
    pending as _pretrain_result."""
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
    if not rerun & names:  # part files on disk are another run's
        if rerun or name in d2l.PRETRAIN_PENDING or (
                names & d2l.PRETRAIN_PENDING):
            return _pending_result(name or artifact, fallback)
        raise KeyError(f'{artifact} is not committed yet')
    results, merge_only = {}, os.environ.get('D2L_PRETRAIN_MERGE') == '1'
    path = {part: os.path.join(_rerun_dir(), f'{nxt}.{part}{ext}')
            for part in parts}
    for part, (compute, part_config) in parts.items():
        if part in rerun and not (merge_only and os.path.exists(path[part])):
            result = compute()
            meta = dict(config=_jsonable(part_config),
                        label=_label_of(result))
            d2l.lm_save_rerun(os.path.basename(path[part]), lambda p:
                              _write_result(p, kind, result, meta),
                              rerun_dir=_rerun_dir())
            results[part] = result
    for part, (_, part_config) in parts.items():  # the others, after
        if part not in results and os.path.exists(path[part]):
            results[part] = _read_result(path[part], kind)
            _check_config(results[part], part_config, f'{nxt}.{part}')
    results = {part: results.get(part) for part in parts}
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

#@save
def reference_shards(which='100m', **kwargs):
    """d2l.fwedu_shards(which) (the reference tokenizer's FineWeb-Edu
    shards, :numref:`sec_lm_tokenizers`) without its download messages."""
    with _quiet():
        return d2l.fwedu_shards(which, **kwargs)
```

The loss of a pretraining step: `chunked_lm_loss` with the summed NLL
alongside, which `pretrain_step` logs when the z-loss term is on, and, on
request, the summed log-partition (`pretrain_step` logs its mean, `log_z`).

```{.python .input #data-path-lib-losses}
%%tab pytorch
#@save
def _lm_loss_terms_lse(h, W, y, ignore_index=-100):
    """lm_loss_terms (summed NLL and logsumexp^2) and the summed logsumexp
    over the valid targets y, from the same fp32 logits."""
    logits = F.linear(h, W)
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    lse, valid = torch.logsumexp(logits, -1), y != ignore_index
    nll = lse - logits.gather(-1, y.clamp(min=0)[:, None])[:, 0]
    return torch.stack([torch.where(valid, nll, 0.0).sum(),
                        torch.where(valid, lse ** 2, 0.0).sum(),
                        torch.where(valid, lse, 0.0).sum()])

#@save
def chunked_lm_loss_and_nll(hidden, weight, labels, chunk=8192, z_loss=0.0,
                            ignore_index=-100, return_lse=False):
    """chunked_lm_loss's (sum_loss, n_valid) and the summed NLL without the
    z-loss term, from the same chunks; with return_lse also the summed
    log-partition logsumexp over the valid targets (no gradient needed)."""
    H, y = hidden.flatten(0, -2), labels.flatten()
    terms = _lm_loss_terms_lse if return_lse else d2l.lm_loss_terms
    parts = sum(torch.utils.checkpoint.checkpoint(
        terms, H[i:i + chunk], weight, y[i:i + chunk], ignore_index,
        use_reentrant=False) for i in range(0, len(y), chunk))
    nll, z = parts[0], parts[1]
    out = (nll + z_loss * z, (y != ignore_index).sum(), nll)
    return out + (parts[2],) if return_lse else out
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
    if attn.backend not in ('auto', 'sdpa', 'flex', 'math'):
        raise ValueError(f"backend is 'auto', 'sdpa', 'flex' or 'math', "
                         f'not {attn.backend!r}')
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
    if not len(model.blks):  # the zero-layer reference model
        return None
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

## Scaled RoPE, Cached Decoding and Construction

What later sections add to `PackedAttention` and `PackedGPT`: scaled RoPE
frequencies and YaRN's attention scale (`set_rope`, used for long context),
the cached decoding step that uses them (`forward_cached`), the model's
configuration (`config`), and the two other constructors (`from_arch` for
the presets of :numref:`sec_lm_data_path`'s next section, `adopt` for a
loaded `d2l.Qwen3GPT`). Until `set_rope` is called a model uses base RoPE.

```{.python .input #data-path-lib-packed-rope}
%%tab pytorch
#@save
_BASE_ROPE = None, 1.0  # (rope_inv_freq, rope_mscale) until set_rope
d2l.PackedAttention.rope_inv_freq, d2l.PackedAttention.rope_mscale = _BASE_ROPE
d2l.PackedGPT.rope_inv_freq, d2l.PackedGPT.rope_mscale = _BASE_ROPE
d2l.PackedGPT.rope_scheme, d2l.PackedGPT.rope_scale = 'none', 1.0
d2l.PackedGPT.trained_len = None  # max_len before the first set_rope

#@save
def _packed_rope_tables(model, pos):
    """RoPE tables of positions pos with the model's frequencies and scale
    (base RoPE until set_rope); pos is 'rope' here."""
    if model.pos != 'rope':
        raise ValueError(f"pos is 'rope' or 'learned', not {model.pos!r}")
    return _rope_tables(pos, model.head_width, model.rope_theta,
                        getattr(model, 'rope_inv_freq', None),
                        getattr(model, 'rope_mscale', 1.0))
```

`make lib` merges the methods below into the class bodies.

```{.python .input #data-path-lib-packed-methods}
%%tab pytorch
@d2l.add_to_class(d2l.PackedAttention)  #@save
def set_rope(self, inv_freq=None, mscale=1.0):
    """RoPE frequencies (None: base rope_theta) and YaRN's scale m."""
    self.rope_inv_freq = (None if inv_freq is None else
                          torch.as_tensor(inv_freq, dtype=torch.float32))
    self.rope_mscale = mscale

@d2l.add_to_class(d2l.PackedAttention)  #@save
def qkv(self, X, offset=0, tables=None):
    """Qwen3Attention.qkv; without tables, the RoPE angles of positions
    offset, offset + 1, ... use this block's (possibly scaled) RoPE."""
    if self.rope and tables is None:
        tables = _rope_tables(
            torch.arange(X.shape[1], device=X.device) + offset,
            self.head_dim, self.rope_theta, self.rope_inv_freq,
            self.rope_mscale)
    return d2l.Qwen3Attention.qkv(self, X, offset, tables)

@d2l.add_to_class(d2l.PackedGPT)  #@save
def set_rope(self, inv_freq=None, mscale=1.0, scheme='none', scale=1.0):
    """Use RoPE frequencies inv_freq (None: base) and YaRN's scale m in
    every block; the model then accepts scale times its trained length."""
    self.trained_len = self.trained_len or self.max_len
    self.rope_inv_freq = (None if inv_freq is None else
                          torch.as_tensor(inv_freq, dtype=torch.float32))
    self.rope_mscale, self.rope_scheme, self.rope_scale = (mscale, scheme,
                                                           scale)
    self.max_len = round(scale * self.trained_len)
    for blk in self.blks:
        blk.attention.set_rope(inv_freq, mscale)

@d2l.add_to_class(d2l.PackedGPT)  #@save
@torch.inference_mode()
def forward_cached(self, X, ks, vs, t, all_logits=False):
    """Qwen3GPT.forward_cached with this model's (possibly scaled) RoPE;
    all_logits=True returns the logits of every new position."""
    if self.pos != 'rope':
        raise NotImplementedError('forward_cached needs RoPE positions; '
                                  'this model has a learned table')
    tables = _packed_rope_tables(
        self, torch.arange(t, t + X.shape[1], device=X.device))
    H = self.token_emb(X)
    for i, blk in enumerate(self.blks):
        H = H + blk.attention.forward_step(blk.norm1(H), (ks[i], vs[i], t),
                                           tables)
        H = H + blk.ffn(blk.norm2(H))
    H = self.norm(H if all_logits else H[:, -1])
    W = self.lm_head.weight if hasattr(self, 'lm_head') else \
        self.token_emb.weight
    return F.linear(H, W), ks, vs

@d2l.add_to_class(d2l.PackedGPT)  #@save
def config(self):
    """Every constructor argument and the RoPE scaling, as JSON values."""
    return _packed_config(self)

@d2l.add_to_class(d2l.PackedGPT)  #@save
@classmethod
def from_arch(cls, arch, **kw):
    """A model of the shape d2l.r2_config(...).arch gives."""
    kw = dict(dict(head_dim=arch.d_h, ffn_hidden=arch.f,
                   qk_norm=arch.qk_norm, rope_theta=arch.rope_base,
                   tie_embeddings=arch.tie, max_len=arch.T), **kw)
    return cls(arch.vocab, arch.d, arch.H, arch.H_kv, arch.L, **kw)

@d2l.add_to_class(d2l.PackedGPT)  #@save
@classmethod
def adopt(cls, model, backend='auto'):
    """A PackedGPT holding the weights of a d2l.Qwen3GPT (no copy)."""
    with torch.device('meta'):
        new = cls(**_qwen3_shape(model), backend=backend)
    _check_same_tensors(new.state_dict(), model.state_dict())
    new.load_state_dict(model.state_dict(), assign=True)
    return new
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
def _physical_step(skipped, step):
    """The step of the unskipped schedule that logical step uses."""
    for k in sorted(skipped):
        step += k <= step
    return step

#@save
def _window_indices(stream, step):
    """The global batch of logical step `step`: a slice of the permutation
    of the window indices seeded by (seed, epoch), at the physical step the
    skip set leaves it (every rank of a stream calls this first)."""
    if stream.global_batch % stream.world:
        raise ValueError(f'global_batch {stream.global_batch} is not a '
                         f'multiple of world {stream.world}')
    p = _physical_step(stream.skipped, step)
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
```

```{.python .input #data-path-lib-stream-methods}
%%tab pytorch
@d2l.add_to_class(d2l.PackedStream)  #@save
def physical_index(self, step):
    """The step of the unskipped schedule that logical step uses."""
    return _physical_step(self.skipped, step)

@d2l.add_to_class(d2l.PackedStream)  #@save
def skip(self, step):
    """Drop physical step `step` (a bad batch); later steps move up."""
    self.skipped = tuple(sorted(set(self.skipped) | {step}))
```

```{.python .input #data-path-lib-restart}
%%tab pytorch
#@save
RestartOverlap = collections.namedtuple(
    'RestartOverlap', ['steps', 'windows', 'naive_match', 'naive_seen',
                       'stateless_match', 'stateless_seen'])

#@save
def reseeded_restart_overlap(n_seq=10_000, batch=64, crash_step=40,
                             steps=50, seed=0):
    """A run that crashes at crash_step and restarts with a re-seeded
    shuffle, against a stateless cursor, over the steps after the crash:
    their number and windows, how many batches equal the uninterrupted
    run's, and how many windows the run had already seen before the crash."""
    perm = np.random.default_rng([seed, 0]).permutation(n_seq)
    after = np.random.default_rng([seed + 1, 0]).permutation(n_seq)
    later = range(crash_step, steps)
    planned = [perm[s * batch:(s + 1) * batch] for s in later]
    reseeded = [after[s * batch:(s + 1) * batch] for s in later]
    before = perm[:crash_step * batch]

    def overlap(batches):  # (batches as planned, windows seen before)
        return (int(sum(np.array_equal(a, b)
                        for a, b in zip(planned, batches))),
                int(np.isin(np.concatenate(batches), before).sum()))
    return RestartOverlap(len(later), len(later) * batch, *overlap(reseeded),
                          *overlap(planned))
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

## The Measurements: Pipeline Throughput and the Mask Comparison

`packing_throughput` times training steps of one model under four input
pipelines; `packing_comparison` trains masked and unmasked models with
paired seeds and probes the unmasked model's attention. Each builds its
configuration from its arguments and leaves the choice between measuring
and loading to `_pretrain_result`.

```{.python .input #data-path-lib-throughput}
%%tab pytorch
#@save
PACKING_PIPELINES = ('padded', 'packed', 'sdpa-mask', 'flex-mask')

#@save
def _padded_batch(shards, T, G, step):
    """Documents step * G ... step * G + G - 1, one per row, cut after T
    targets and padded with EOS; padded targets are -100."""
    X = torch.full((G, T), shards.eos_id, dtype=torch.int64)
    Y = torch.full((G, T), -100, dtype=torch.int64)
    for r in range(G):
        d = torch.from_numpy(shards.doc((step * G + r) % shards.n_docs)).long()
        n = min(len(d) - 1, T)
        X[r, :n], Y[r, :n] = d[:n], d[1:n + 1]
    pos = torch.arange(T, dtype=torch.int32).expand(G, T).contiguous()
    return dict(input_ids=X, labels=Y, doc_ids=torch.zeros_like(pos),
                pos_ids=pos)

#@save
def _pipeline_batches(pipeline, shards, T, G, steps, seed):
    """The first steps global batches of a pipeline: padded documents, or
    the windows of a PackedStream."""
    if pipeline == 'padded':
        return [_padded_batch(shards, T, G, s) for s in range(steps)]
    stream = d2l.PackedStream(shards, T, G, seed=seed)
    return [stream.batch(s) for s in range(steps)]

#@save
def _timed_step(model, optimizer, micro, n_valid, masked, cfg):
    """One training step as pretrain_step takes it (bf16 autocast, chunked
    token-weighted loss, clipping, AdamW), with the document mask only if
    masked (else causal attention over the whole row)."""
    for mb in micro:
        ctx = d2l.PackCtx(mb['doc_ids'], mb['pos_ids']) if masked else None
        with torch.autocast(mb['input_ids'].device.type, torch.bfloat16,
                            enabled=cfg.dtype == 'bf16'):
            H, W = model(mb['input_ids'], ctx, return_hidden=True)
            loss, _ = d2l.chunked_lm_loss(H, W, mb['labels'],
                                          z_loss=cfg.z_loss)
        (loss / n_valid).backward()
    params = [p for g in optimizer.param_groups for p in g['params']]
    torch.nn.utils.clip_grad_norm_(params, cfg.clip or math.inf)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

#@save
def _throughput_row(cfg, pipeline, T, batches, warmup, device):
    """Useful (valid-target) tokens per second, median and quartiles over
    the steps after warmup, the median step time, the padding share and the
    peak memory of one pipeline at window length T."""
    torch._dynamo.reset()  # a fresh compile for every model
    torch.manual_seed(cfg.seed)
    backend = {'sdpa-mask': 'sdpa', 'flex-mask': 'flex'}.get(pipeline,
                                                             'auto')
    raw = d2l.PackedGPT.from_arch(cfg.arch._replace(T=T),
                                  backend=backend).to(device)
    cuda = device.type == 'cuda'
    optimizer = torch.optim.AdamW(d2l.param_groups(raw, cfg.wd),
                                  lr=cfg.peak_lr, betas=cfg.betas,
                                  eps=cfg.eps, fused=cuda)
    model = torch.compile(raw) if cfg.compile else raw
    m = max(1, cfg.micro_batch * cfg.arch.T // T)  # same tokens per micro
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (
        lambda: None)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    rates, times = [], []
    for k, b in enumerate(batches):
        b = {key: v.to(device) for key, v in b.items()}
        n_valid = (b['labels'] != -100).sum()
        micro = [{key: v[i:i + m] for key, v in b.items()}
                 for i in range(0, len(b['labels']), m)]
        sync()
        t = time.perf_counter()
        _timed_step(model, optimizer, micro, n_valid,
                    pipeline.endswith('-mask'), cfg)
        sync()
        if k >= warmup:
            times.append(time.perf_counter() - t)
            rates.append(n_valid.item() / times[-1])
    padding = (float(np.mean([(b['labels'] == -100).float().mean()
                              for b in batches]))
               if pipeline == 'padded' else 0.0)  # every -100 is a pad
    q1, med, q3 = np.percentile(rates, [25, 50, 75])
    row = dict(pipeline=pipeline, T=T, tok_s=med, tok_s_q1=q1, tok_s_q3=q3,
               step_ms=1e3 * float(np.median(times)), padding=padding,
               peak_gib=(torch.cuda.max_memory_allocated(device) / 2 ** 30
                         if cuda else float('nan')), status='ok')
    del model, raw, optimizer
    if cuda:
        torch.cuda.empty_cache()
    return row

#@save
def _throughput_table(cfg, pipelines, seq_lens, steps, warmup, shards,
                      device):
    """_throughput_row for every pipeline and window length, at cfg's
    tokens per step; a FlexAttention row that fails reads 'flex
    unavailable' (any other failure raises)."""
    rows = []
    for T in seq_lens:
        G = cfg.global_batch * cfg.arch.T // T  # sequences per step
        for p in pipelines:
            batches = _pipeline_batches(p, shards, T, G, warmup + steps,
                                        cfg.data_seed)
            try:
                rows.append(_throughput_row(cfg, p, T, batches, warmup,
                                            device))
            except Exception as e:
                if p != 'flex-mask':
                    raise
                print(f'flex-mask at T={T}: {type(e).__name__}: {e}',
                      file=sys.stderr)
                rows.append(dict(pipeline=p, T=T, status='flex unavailable'))
    df = pd.DataFrame(rows)
    df['gpu'] = (torch.cuda.get_device_name(device)
                 if device.type == 'cuda' else 'cpu')
    return df

#@save
def packing_throughput(cfg, pipelines=PACKING_PIPELINES, seq_lens=(1024,
                       4096), steps=50, warmup=5):
    """Training throughput of cfg's model under each input pipeline at each
    window length T, at cfg's tokens per step: 'padded' (one document per
    row), 'packed' (windows, causal attention across documents),
    'sdpa-mask' and 'flex-mask' (windows with the document mask). Columns:
    useful tokens/s (median, quartiles over steps), step time, padding
    share, peak memory. The committed table, or measured on one GPU
    (compiled) under D2L_RERUN=pdp-throughput."""
    config = dict(cfg=d2l._cfg_json(cfg), pipelines=list(pipelines),
                  seq_lens=list(seq_lens), steps=steps, warmup=warmup)

    def compute():
        if not torch.cuda.is_available():
            raise RuntimeError('pdp-throughput measures on a GPU')
        return _throughput_table(cfg, pipelines, seq_lens, steps, warmup,
                                 d2l._train_shards(cfg), torch.device('cuda'))
    return _show_label(d2l._pretrain_result('pdp-throughput', compute,
                                            config))
```

```{.python .input #data-path-lib-mask-runs}
%%tab pytorch
#@save
PackingRuns = collections.namedtuple('PackingRuns',
                                     ['table', 'losses', 'leak', 'label'])

#@save
class _UnmaskedStream(d2l.PackedStream):
    """A PackedStream read the way a plain causal model reads it: every
    window is one document for attention (doc_ids 0, positions 0 ... T-1);
    windows and targets are PackedStream's, so a pair of runs differs only
    in the mask."""
    def batch(self, step):
        b = super().batch(step)
        b['doc_ids'] = torch.zeros_like(b['doc_ids'])
        b['pos_ids'] = torch.arange(self.seq_len, dtype=torch.int32).expand_as(
            b['pos_ids']).contiguous()
        return b

#@save
def _mask_run_config(cfg, tokens):
    """cfg shortened to tokens (rounded to whole steps): WSD with 5% warmup
    and 10% decay, validation at the end only, nothing saved."""
    steps = round(tokens / (cfg.global_batch * cfg.arch.T))
    return dataclasses.replace(
        cfg, steps=steps, warmup=round(0.05 * steps),
        decay=round(0.1 * steps), val_every=0, probe_every=0, stats_every=0,
        ckpt_steps=(), state_steps=(), state_every=0, branches=(),
        out_dir=None)

#@save
def _foreign_mass(q, k, doc_ids, chunk=256):
    """Per window: the mean, over queries that have an earlier key of
    another document, of the causal attention mass on such keys (averaged
    over heads; fp32), and the share uniform attention would put there;
    nan for a window that holds one document."""
    B, H, T, d_h = q.shape
    q, k = q.float(), k.float()
    j = torch.arange(T, device=q.device)
    mass, base, n = (torch.zeros(B, device=q.device) for _ in range(3))
    for a in range(0, T, chunk):
        i = torch.arange(a, min(a + chunk, T), device=q.device)
        causal = j[None, :] <= i[:, None]                   # (c, T)
        s = q[:, :, i] @ k.transpose(-2, -1) / math.sqrt(d_h)
        p = torch.softmax(s.masked_fill(~causal, -math.inf), -1)
        foreign = (doc_ids[:, i, None] != doc_ids[:, None, :]) & causal
        has = foreign.any(-1)                               # (B, c)
        m = (p * foreign[:, None]).sum(-1).mean(1)          # over heads
        mass += torch.where(has, m, 0.0).sum(-1)
        base += torch.where(has, foreign.sum(-1) / (i + 1), 0.0).sum(-1)
        n += has.sum(-1)
    return mass / n, base / n

#@save
def _leak_table(model, X, eos_id=0, batch=8, n_boot=2000, seed=0):
    """Per block of a model trained without the document mask: the mean
    over windows X of the attention mass on other documents' keys, a 95%
    bootstrap interval over windows, and the uniform-attention share."""
    dev = next(model.parameters()).device
    doc_ids = d2l.doc_ids_from_eos(X.int(), eos_id)[0]
    leak, uniform = [], []
    with torch.no_grad():
        for a in range(0, len(X), batch):
            Xb, db = X[a:a + batch].to(dev), doc_ids[a:a + batch].to(dev)
            per_layer = []
            for attn, Y, tables in d2l._attention_inputs(model, Xb, None):
                q, k, _ = attn.qkv(Y.float(), tables=tables)
                k = k.repeat_interleave(attn.num_heads // attn.num_kv_heads,
                                        1)
                m, u = _foreign_mass(q, k, db)
                per_layer.append(m)
            leak.append(torch.stack(per_layer, 1).cpu())
            uniform.append(u.cpu())
    M, U = torch.cat(leak).double().numpy(), torch.cat(uniform).numpy()
    keep = ~np.isnan(M[:, 0])  # windows with more than one document
    M, U = M[keep], U[keep]
    idx = np.random.default_rng(seed).integers(0, len(M), (n_boot, len(M)))
    lo, hi = np.percentile(M[idx].mean(1), [2.5, 97.5], axis=0)
    return pd.DataFrame(dict(layer=np.arange(M.shape[1]), leak=M.mean(0),
                             lo=lo, hi=hi, uniform=float(U.mean()),
                             windows=len(M)))

#@save
def _mask_runs(cfg, seeds, train, val_fn, leak_X, device, every=10):
    """For each seed, train cfg's model on train (on device) with and
    without the document mask (the stream seed is the model seed, so a pair
    sees the same batches): final validation loss (val_fn), the training
    loss every `every` steps, and the leak of the unmasked seeds[0] model on
    leak_X."""
    finals, traces, leak = [], [], None
    for seed in seeds:
        row = dict(seed=seed)
        for arm, Stream in (('masked', d2l.PackedStream),
                            ('unmasked', _UnmaskedStream)):
            torch._dynamo.reset()
            torch.manual_seed(seed)
            model = d2l.PackedGPT.from_arch(cfg.arch).to(device)
            stream = Stream(train, cfg.arch.T, cfg.global_batch, seed=seed)
            log = d2l.pretrain(model, stream, dataclasses.replace(
                cfg, seed=seed, data_seed=seed), val_fn=val_fn)
            row[arm] = float(log['val_loss'].dropna().iloc[-1])
            keep = log[(log['step'] % every == 0) | (log['step'] ==
                                                      cfg.steps)]
            traces.append(pd.DataFrame(dict(arm=arm, seed=seed,
                                            step=keep['step'],
                                            loss=keep['loss'])))
            if arm == 'unmasked' and seed == seeds[0]:
                leak = _leak_table(model, leak_X, train.eos_id)
            del model, stream
        finals.append(row)
    return PackingRuns(pd.DataFrame(finals), pd.concat(traces,
                                                       ignore_index=True),
                       leak, '')

#@save
def packing_comparison(cfg, tokens=25_000_000, seeds=(0, 1, 2),
                       leak_windows=64):
    """Masked against unmasked training of cfg's model for tokens tokens
    with paired seeds, both validated with the document mask (each
    document on its own, as at inference): PackingRuns(table: seed, masked,
    unmasked final validation loss; losses: training loss every 10 steps;
    leak: per block, the unmasked seeds[0] model's attention mass on other
    documents over leak_windows validation windows, with a 95% bootstrap
    interval). The committed result, or trained on one GPU under
    D2L_RERUN=pdp-mask-runs."""
    run_cfg = _mask_run_config(cfg, tokens)
    config = dict(cfg=d2l._cfg_json(run_cfg), tokens=tokens,
                  seeds=list(seeds), leak_windows=leak_windows)

    def compute():
        if not torch.cuda.is_available():
            raise RuntimeError('pdp-mask-runs trains on a GPU')
        val = d2l.reference_shards('val')
        T = cfg.arch.T
        leak_X = torch.from_numpy(np.stack([
            val.read(i * T, T) for i in range(leak_windows)]).astype(np.int64))
        return _mask_runs(run_cfg, seeds, d2l._train_shards(run_cfg),
                          d2l._val_fn(run_cfg), leak_X, torch.device('cuda'))
    result = _show_label(d2l._pretrain_result('pdp-mask-runs', compute,
                                              config))
    if isinstance(result, pd.DataFrame):  # the labelled pending result
        return PackingRuns(result, result, result, d2l._label_of(result))
    return PackingRuns(result.table, result.losses, result.leak,
                       result.label)
```
