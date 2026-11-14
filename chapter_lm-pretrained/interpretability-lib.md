# Interpretability Plumbing (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_interpretability` explains in a sentence and does not show:
the argument checks of `d2l.intervene` (`_intervene_site`,
`_intervene_args`); the range reads of parquet files on the Hub and the
document walk that resolves each data subset once (`lm_hub_rows`,
`lm_walk_docs`, `lm_round_robin`); the Universal Dependencies reader and the
word-level features of the probe (`read_conllu`, `lm_word_features`,
`type_tag_baseline`); the cluster bootstrap and its formatting
(`cluster_resample`, `percentile_ci`, `fmt_ci`), the detection statistics of
the label test (`detection_stats`) and the CSV and text forms of the
patching tables (`long_csv`, `block_runs`); the regeneration bookkeeping
(`lm_write_zip`, `cache_stream`, `lm_save_sae`, `lm_load_sae`); the stream
and feature reads (`lm_stream`, `sae_fire_rate`, `sae_acts`); and the
steering measurements (`char_fraction`, `lm_doc_ce`, `frontier_envelope`,
`frontier_area`, `cost_of_half`). Every block is tagged `%%tab pytorch` and
imports modules outside the library preamble inside its functions: a
module-level import would join the `_preamble` shard and move the freshness
fingerprint of every notebook in the book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_interpretability`

```{.python .input #interpretability-lib-intervene}
%%tab pytorch
#@save
def _intervene_site(model, site):
    """site as (kind, layer) if it names a site of model, else ValueError."""
    if not (isinstance(site, tuple) and len(site) == 2
            and site[0] in d2l._Intervention.KINDS
            and isinstance(site[1], (int, np.integer))
            and 0 <= site[1] < len(model.blks)):
        raise ValueError(f'not a site of this model: {site!r}')
    return site[0], int(site[1])

#@save
def _intervene_args(model, sites, record, stop_after):
    """The checked arguments of d2l.intervene: the sites to edit (site ->
    fn), the sites to record (a set, with stop_after) and stop_after."""
    assert isinstance(model, d2l.GPT), 'intervene instruments a d2l.GPT'
    for blk in model.blks:
        assert blk.pre_norm, 'intervene needs pre-norm blocks'
        assert not (blk.training and blk.dropout.p > 0), (
            'dropout must be inactive: call model.eval()')
    sites = {_intervene_site(model, s): fn for s, fn in (sites or {}).items()}
    for site, fn in sites.items():
        if fn is None:  # recording is record='s job
            raise ValueError(f'site {site} maps to None; use record=')
    stop = None if stop_after is None else _intervene_site(model, stop_after)
    record = {_intervene_site(model, s) for s in record}
    if stop is not None:  # a stopped forward is for reading
        record.add(stop)
    return sites, record, stop
```

```{.python .input #interpretability-lib-data}
%%tab pytorch
#@save
def lm_hub_rows(repo, path, revision, start, stop, column='text'):
    """Rows start, ..., stop - 1 of one column of a parquet file of a Hub
    dataset, reading only the row groups that hold them (range requests);
    returns the values and the compressed bytes read."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f'datasets/{repo}@{revision}/{path}',
                             'rb') as f:
        pf = pq.ParquetFile(f)
        md, col = pf.metadata, pf.schema_arrow.names.index(column)
        ends = np.cumsum([md.row_group(g).num_rows
                          for g in range(md.num_row_groups)])
        groups = [g for g, e in enumerate(ends)
                  if e > start and e - md.row_group(g).num_rows < stop]
        values = pf.read_row_groups(groups, columns=[column]).column(0)
        first = ends[groups[0]] - md.row_group(groups[0]).num_rows
        nbytes = sum(md.row_group(g).column(col).total_compressed_size
                     for g in groups)
    return values.to_pylist()[start - first:stop - first], nbytes

#@save
def lm_walk_docs(encode, source, rows, subsets, block=1000):
    """Visit the rows of source = (repo, path, revision) in the order rows
    gives and keep the first n qualifying documents of each subset in turn;
    subsets maps a name to (n, take), take(ids) -> ids or None, and
    encode(text) -> ids. Rows are read a block of block rows at a time.
    Returns a tensor of documents per subset, and the rows visited (first,
    last), the blocks read and their compressed bytes."""
    docs, cache, used, nbytes, rows = {}, {}, [], 0, iter(rows)
    for name, (n, take) in subsets.items():
        kept = []
        while len(kept) < n:
            row = next(rows)
            start = row - row % block
            if start not in cache:
                cache[start], b = lm_hub_rows(*source, start, start + block)
                nbytes += b
            if (ids := take(encode(cache[start][row - start]))) is not None:
                kept.append(ids)
            used.append(row)
        docs[name] = torch.tensor(kept)
    return docs, dict(rows=(min(used), max(used)), blocks=len(cache),
                      nbytes=nbytes)

#@save
def lm_round_robin(stride, first=0, blocks=20, block=1000):
    """Row i of each of the blocks at first + b * stride, then row i + 1."""
    return (first + b * stride + i for i in range(block)
            for b in range(blocks))
```

```{.python .input #interpretability-lib-probe}
%%tab pytorch
#@save
def read_conllu(path, stride, max_words):
    """Every stride-th sentence of a CoNLL-U file, up to max_words words:
    the text rebuilt with SpaceAfter=No, and per word (start, end, form,
    UPOS)."""
    with open(path, encoding='utf-8') as f:
        sentences, out, n = f.read().strip().split('\n\n'), [], 0
    for sentence in sentences[::stride]:
        text, words, end = '', [], 0
        for c in (line.split('\t') for line in sentence.split('\n')
                  if not line.startswith('#')):
            spaced = 'SpaceAfter=No' not in c[9]
            if '-' in c[0]:  # a multiword token; its words follow, unspaced
                end, space = int(c[0].split('-')[1]), spaced
            elif '.' not in c[0]:  # skip empty nodes
                words.append((len(text), len(text) + len(c[1]), c[1], c[3]))
                text += c[1]
                i = int(c[0])
                if i == end and space or i > end and spaced:
                    text += ' '
        if n + len(words) > max_words:
            break
        out.append((text.rstrip(), words))
        n += len(words)
    return out

#@save
def lm_word_features(model, tokenizer, sentences, layer, batch=32):
    """Per word of sentences (read_conllu's), at its last token: the token
    embedding and the stream after block layer, on the CPU. Right padding
    is safe: a causal model never reads a later position."""
    dev, site, emb, hidden = model.token_emb.weight.device, (
        'resid_post', layer), [], []
    for i in range(0, len(sentences), batch):
        chunk = [tokenizer(t, return_offsets_mapping=True,
                           add_special_tokens=False)
                 for t, _ in sentences[i:i + batch]]
        T = max(len(e.input_ids) for e in chunk)
        X = torch.tensor([e.input_ids + [0] * (T - len(e.input_ids))
                          for e in chunk], device=dev)
        with torch.no_grad(), d2l.intervene(model, stop_after=site) as ctx:
            model(X)
        H = ctx.cache[site][0]
        with torch.no_grad():
            E0 = model.token_emb(X)
        for j, (e, (_, ws)) in enumerate(zip(chunk, sentences[i:i + batch])):
            starts = [s for s, _ in e.offset_mapping]
            pos = [int(np.searchsorted(starts, w[1])) - 1 for w in ws]
            emb.append(E0[j, pos].cpu())
            hidden.append(H[j, pos].cpu())
    return {'embedding': torch.cat(emb), f'block {layer}': torch.cat(hidden)}

#@save
def type_tag_baseline(train_types, train_tags, test_types):
    """Per test word, the most frequent training tag of its word type, or
    the most frequent training tag for a type unseen in training."""
    best = pd.DataFrame({'type': train_types, 'tag': train_tags}).groupby(
        'type')['tag'].agg(lambda t: t.value_counts().idxmax())
    default = np.bincount(train_tags).argmax()
    return np.array([best.get(w, default) for w in test_types])
```

```{.python .input #interpretability-lib-stats}
%%tab pytorch
#@save
def cluster_resample(clusters, n_boot=2000, seed=0):
    """Per replicate, the item indices of a resample of whole clusters
    (clusters holds one cluster id per item)."""
    ids, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(ids))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([members[g] for g in rng.integers(
            len(ids), size=len(ids))])

#@save
def percentile_ci(values, alpha=0.05, names=None):
    """Percentile endpoints of bootstrap replicates (along axis 0), over
    the replicates where the statistic is defined; prints how many are not
    for each statistic (column) named in names."""
    values = np.asarray(values, dtype=float)
    undefined = np.isnan(values).sum(0)
    for name, k in zip(names or (), np.atleast_1d(undefined)):
        if k:
            print(f'  {name}: {k} of {len(values)} replicates undefined, '
                  'left out')
    return np.nanquantile(values, [alpha / 2, 1 - alpha / 2], axis=0)

#@save
def fmt_ci(mean, ci, digits=3):
    """A value and its interval as text: 'mean [lo, hi]'."""
    return f'{mean:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]'

#@save
def detection_stats(y, score, threshold):
    """Precision and recall of score > threshold as a detector of the
    boolean labels y, and the AUC of score (Mann-Whitney, mid-ranks)."""
    from scipy.stats import rankdata
    active, n = score > threshold, y.sum()
    auc = (rankdata(score)[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n))
    return [(y & active).sum() / active.sum(), (y & active).sum() / n, auc]

#@save
def long_csv(T, axes, column):
    """The array T as CSV bytes, one row per entry: per axis a column that
    names the entry's index, axes = [(name, labels), ...], then its value
    in column."""
    index = np.indices(T.shape).reshape(T.ndim, -1)
    named = {name: np.asarray(labels)[i] for (name, labels), i in zip(
        axes, index)}
    return pd.DataFrame({**named, column: T.ravel()}).to_csv(
        index=False).encode()

#@save
def block_runs(cells, classes):
    """(block, class) cells as text, runs of blocks per class in the order
    of classes: "'subject' 0-3, 7; ' of' 20-27" ('none' if empty)."""
    parts = []
    for c in classes:
        ls = np.array([l for l, c_ in cells if c_ == c])
        if len(ls):
            runs = np.split(ls, np.flatnonzero(np.diff(ls) > 1) + 1)
            parts.append(f'{c!r} ' + ', '.join(
                f'{r[0]}-{r[-1]}' if len(r) > 1 else f'{r[0]}' for r in runs))
    return '; '.join(parts) or 'none'
```

```{.python .input #interpretability-lib-sae}
%%tab pytorch
#@save
def lm_write_zip(path, members):
    """A zip archive of members (name -> bytes), byte-identical for
    identical members."""
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)

#@save
def cache_stream(model, windows, layer, batch=32):
    """The stream after block layer at the positions >= 1 of windows
    (N, T), without norm outliers (above 10 times the batch median), as
    one fp16 tensor in host memory, filled in place."""
    site, n, dev = ('resid_post', layer), 0, model.token_emb.weight.device
    acts = torch.empty(windows[:, 1:].numel(), model.num_hiddens,
                       dtype=torch.float16)
    for X in windows.split(batch):
        with torch.no_grad(), d2l.intervene(model, stop_after=site) as ctx:
            model(X.to(dev))
        h = ctx.cache[site][0][:, 1:].flatten(0, 1)
        norms = h.float().norm(dim=-1)
        h = h[norms <= 10 * norms.median()]
        acts[n:n + len(h)] = h.half().cpu()
        n += len(h)
    assert torch.isfinite(acts[:n]).all()
    return acts[:n]

#@save
def lm_save_sae(sae, tokens, name):
    """Write sae in fp16, with its config and its training-token count, as
    the regenerated artifact name (d2l.lm_save_rerun); sae then holds the
    fp16 values the file publishes."""
    state = {'state_dict': {k: v.half() for k, v in sae.state_dict().items()},
             'config': dict(d_model=sae.d_model, d_sae=sae.d_sae, k=sae.k,
                            center=sae.center), 'tokens': tokens}
    d2l.lm_save_rerun(name, lambda p: torch.save(state, p))
    sae.load_state_dict(state['state_dict'])

#@save
def lm_load_sae(key, device=None):
    """The TopKSAE that lm_save_sae wrote, registered as DATA_HUB[key],
    and the number of tokens it was trained on."""
    saved = torch.load(d2l.download_keyed(key), map_location='cpu',
                       weights_only=True)
    sae = d2l.TopKSAE(**saved['config'])
    sae.load_state_dict(saved['state_dict'])
    return sae.to(device), saved['tokens']

#@save
def lm_stream(model, docs, site, batch=16):
    """The activation at site (recorded by d2l.intervene, which stops the
    forward there) at the positions >= 1 of docs: (docs, positions, d)."""
    out, dev = [], model.token_emb.weight.device
    for X in docs.split(batch):
        with torch.no_grad(), d2l.intervene(model, stop_after=site) as ctx:
            model(X.to(dev))
        out.append(ctx.cache[site][0][:, 1:])
    return torch.cat(out)

#@save
@torch.no_grad()
def sae_fire_rate(sae, H, chunk=2048):
    """The fraction of the positions of H (..., d) where each feature of
    sae fires."""
    flat = H.reshape(-1, H.shape[-1])
    fired = sum((sae.encode(h) > 0).sum(0) for h in flat.split(chunk))
    return fired / len(flat)

#@save
@torch.no_grad()
def sae_acts(sae, H, feats, chunk=2048):
    """The activations of the features feats of sae on H (docs, positions,
    d): (docs, positions, len(feats)), in fp32."""
    return torch.cat([sae.encode(h)[:, feats] for h in H.flatten(0, 1).split(
        chunk)]).view(*H.shape[:2], len(feats)).float()
```

```{.python .input #interpretability-lib-steering}
%%tab pytorch
#@save
def char_fraction(text, pattern):
    """The fraction of the non-space characters of text that match the
    compiled regular expression pattern (0 for an empty text)."""
    chars = [c for c in text if not c.isspace()]
    return sum(pattern.match(c) is not None for c in chars) / max(
        len(chars), 1)

#@save
@torch.no_grad()
def lm_doc_ce(model, docs, prefix_ids=(), mask=None, batch=16):
    """Per document of docs (N, T), after the tokens prefix_ids: the mean
    cross-entropy of its tokens 1 to T - 1, and the same renormalized over
    the vocabulary outside the boolean mask (V,) and averaged over the
    targets outside it. Returns an (N, 2) array."""
    dev, P, out = model.token_emb.weight.device, len(prefix_ids), []
    mask = mask.to(dev)
    X = torch.cat([torch.tensor(list(prefix_ids), dtype=torch.long).expand(
        len(docs), P), docs], 1)
    for Xb in X.split(batch):
        Xb = Xb.to(dev)
        z, y = model(Xb)[:, P:-1].float(), Xb[:, P + 1:]
        z_y = z.gather(-1, y[..., None])[..., 0]
        ce = z.logsumexp(-1) - z_y
        ce_rest = z.masked_fill(mask, -torch.inf).logsumexp(-1) - z_y
        rest = ~mask[y]
        out.append(torch.stack([ce.mean(1), (ce_rest * rest).sum(1)
                                / rest.sum(1)], 1).cpu())
    return torch.cat(out).numpy()

#@save
def frontier_envelope(cost, sw):
    """The monotone envelope of the points (cost, sw): the most switched at
    each cost or less, costs clipped at 0; one vertex per distinct cost."""
    order = np.argsort(cost, kind='stable')
    x, y = np.clip(cost[order], 0, None), np.maximum.accumulate(sw[order])
    last = np.append(x[1:] > x[:-1], True)  # ties: keep the last, the most
    return x[last], y[last]

#@save
def frontier_area(cost, sw, cap):
    """The area under the frontier from cost 0 to cap (linear between the
    envelope's vertices, flat past the last)."""
    grid = np.linspace(0, cap, 101)
    return np.trapezoid(np.interp(grid, *frontier_envelope(cost, sw)), grid)

#@save
def cost_of_half(cost, sw):
    """The cost at which a strength grid (cost and switched fraction per
    strength, in increasing strength) first switches half: linear between
    the first strength that does and the strength before it; NaN if none
    does. Read along the strengths, not the envelope: past the cliff the
    cost falls as the output breaks down."""
    j = np.flatnonzero(np.asarray(sw) >= 0.5)
    if not len(j):
        return np.nan
    if j[0] == 0:
        return float(cost[0])
    (c0, c1), (s0, s1) = cost[j[0] - 1:j[0] + 1], sw[j[0] - 1:j[0] + 1]
    return float(c0 + (0.5 - s0) * (c1 - c0) / (s1 - s0))
```
