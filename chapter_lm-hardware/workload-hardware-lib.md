# Training, Prefill and Decode Workloads: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_workload_hardware` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1): the
section's registry entries, one statement each (the box's workload grid and
MoE coverage, the InferenceX snapshot and the campaign logs C-T2 and
C-T2pair; the `DATA_HUB` entries follow the uploads) and its regeneration
jobs; `HW_WORKLOAD_CONFIGS`; the benchmark models (random weights from a
pinned configuration, the real-weights control, the static cache and the
attention function the harness binds); the calibration helpers; the grid
runner with its capacity cut, live checks and displays; the model fits
(M0-M3) and the frontier plot; the MoE router recording, its inputs and
the coverage plot; the InferenceX reader and the ranking; the two
regeneration drivers. Blocks are tagged `%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_workload_hardware`

## Registries

Both box results are multi-part (`tar`): the workload grid of each
configuration and the weight-value controls; the coverage table with the
recorded router choices.

```{.python .input #workload-hardware-lib-entries}
%%tab pytorch
#@save
d2l.HW_ARTIFACTS['box-workloads'] = ('lm-hw-buildbox-workloads-v1', 'tar')
d2l.HW_ARTIFACTS['box-moe'] = ('lm-hw-buildbox-moe-v1', 'tar')
d2l.HW_STATIC['inferencex-snapshot'] = ('lm-hw-inferencex-snapshot-v1', 'tar')
d2l.HW_CAMPAIGNS['a100-workloads'] = (
    'lm-hw-a100-workloads-v1', 'tar', 'C-T2', 'A100 PCIe 40 GB')
d2l.HW_CAMPAIGNS['t2pair-moe'] = (
    'lm-hw-t2pair-moe-v1', 'tar', 'C-T2pair',
    'two A100 40 GB with an NVLink bridge')
```

```{.python .input #workload-hardware-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-workloads'] = (
    'regen_workloads', lambda: dict(config='box'))
d2l.HW_JOBS['box-moe'] = (
    'regen_moe', lambda: dict(config='box'))
```

The same drivers on the campaigns' hardware write the campaign logs (C-T2pair
is a new tier: two A100s with an NVLink bridge).

```{.python .input #workload-hardware-lib-config-logs}
%%tab pytorch
#@save
d2l.HW_CONFIG_LOGS[('box-workloads', 't2')] = 'a100-workloads'
d2l.HW_CONFIG_LOGS[('box-moe', 't2pair')] = 't2pair-moe'
d2l._HW_TIERS['t2pair'] = 'T2pair'
```

## Configurations

`box` runs on one RTX 4090 (Qwen3-8B itself), `t1` is a reader's 16 GB GPU
(its capacity cut at 0.9 x 16 GB), `t2` the A100 PCIe 40 GB of campaign C-T2.
Decode series are (model, batches, contexts, series); `cross` marks the four
Qwen3-0.6B points that chapter 21's decode table times with the book's own
model. Training rows are (model, layers kept or None for all, step shapes
(b, s): b sequences of s tokens). Qwen3-0.6B trains on one sequence of 512 to
2,048 tokens: pure bf16 AdamW holds 4.8 GB of state and 2,048 tokens of its 28
layers' activations about 6 GB more (13 GiB at the peak, with the loss
chunk), so 4,096 tokens would pass 20 GiB. The MoE series: Qwen3-30B-A3B's
router on `n_prose` FineWeb-Edu validation documents and `n_code` windows of
the held-out code text, `n_tokens` each, coverage by `draws` resamples per
batch. The eager Qwen3-8B series stops at b = 4: at b = 8 its one-pass prefill
of 16,384 tokens and the copy of the cache each repeat starts from peak at
22 GiB.

```{.python .input #workload-hardware-lib-configs}
%%tab pytorch
#@save
_HW_PROMPTS = (256, 512, 1024, 2048, 4096, 8192, 16384)

#@save
_HW_SMALL_STEPS = ((1, 512), (1, 1024), (1, 2048))

#@save
_HW_MOE = dict(model='Qwen/Qwen3-30B-A3B', n_prose=1536, n_code=64,
               n_tokens=256, micro_batch=16, draws=200, seed=0,
               batches=(1, 2, 4, 8, 16, 32, 64, 128, 256),
               step_batches=(1, 8, 64), step_context=256)

#@save
HW_WORKLOAD_CONFIGS = {
    'box': dict(
        chip='RTX 4090', mem_gb=None,
        decode=(('Qwen/Qwen3-8B', (1, 2, 4, 8, 16, 32, 64),
                 (512, 2048, 8192), 'grid'),
                ('Qwen/Qwen3-0.6B', (1, 4, 16, 64, 128), (512, 2048), 'grid'),
                ('Qwen/Qwen3-0.6B', (1, 8, 32, 128), (1024,), 'cross')),
        prefill=(('Qwen/Qwen3-8B', _HW_PROMPTS),
                 ('Qwen/Qwen3-0.6B', _HW_PROMPTS)),
        train=(('Qwen/Qwen3-8B', (1, 2, 3), ((1, 2048), (2, 2048))),
               ('Qwen/Qwen3-0.6B', (None,), _HW_SMALL_STEPS)),
        eager=(('Qwen/Qwen3-8B', (1, 4), 2048),
               ('Qwen/Qwen3-0.6B', (1, 8), 2048)),
        control=('Qwen/Qwen3-8B', 1, 2048), moe=_HW_MOE),
    't1': dict(
        chip='RTX 4090', mem_gb=16,
        decode=(('Qwen/Qwen3-0.6B', (1, 4, 16, 64, 128), (512, 2048), 'grid'),
                ('Qwen/Qwen3-1.7B', (1, 8, 32), (2048,), 'grid')),
        prefill=(('Qwen/Qwen3-0.6B', _HW_PROMPTS),),
        train=(('Qwen/Qwen3-0.6B', (None,), _HW_SMALL_STEPS),),
        eager=(('Qwen/Qwen3-0.6B', (1, 8), 2048),),
        control=('Qwen/Qwen3-0.6B', 1, 2048), moe=None),
    't2': dict(
        chip='A100 PCIe 40 GB', mem_gb=None,
        decode=(('Qwen/Qwen3-8B', (1, 2, 4, 8, 16, 32, 64),
                 (512, 2048, 8192), 'grid'),),
        prefill=(('Qwen/Qwen3-8B', _HW_PROMPTS),),
        train=(('Qwen/Qwen3-8B', (2, 4, 6), ((1, 2048), (2, 2048))),),
        eager=(('Qwen/Qwen3-8B', (1, 8), 2048),),
        control=('Qwen/Qwen3-8B', 1, 2048), moe=None),
    't2pair': dict(chip='A100 PCIe 40 GB', mem_gb=None, moe=_HW_MOE)}
```

## Rates, Models and the Static Cache

```{.python .input #workload-hardware-lib-rates}
%%tab pytorch
#@save
def hw_rates(chip, calib=None, dtype_bytes=2):
    """(FLOP/s, byte/s) of a chip_specs row: the dense peak of the weight
    format (bf16, or FP8 for dtype_bytes=1 where the row has one) and the
    memory bandwidth, or the calibration's P_eff_bf16 (scaled like the
    datasheet for FP8, times a fitted eta_compute) and beta_eff. A chip
    without a published FLOP figure gets an infinite one: its compute term
    drops out and the prediction is a bandwidth bound."""
    p16 = chip.p16_tflops * 1e12
    p8 = chip.p8_tflops * 1e12 if dtype_bytes == 1 else math.nan
    P = p8 if np.isfinite(p8) else p16
    P = P if np.isfinite(P) else math.inf
    beta = chip.mem_bw_tbs * 1e12
    if calib:
        scale = P / p16 if np.isfinite(P) and np.isfinite(p16) else 1.0
        P = calib['P_eff_bf16'] * scale * calib.get('eta_compute', 1.0)
        beta = calib['beta_eff']
    return P, beta

#@save
def _hw_peak():
    """Peak GPU memory allocated since the last reset, in GiB."""
    return torch.cuda.max_memory_allocated() / 2**30

#@save
_HW_CONFIGS = {}  # (model id, layers kept): configuration, never modified

#@save
def _hw_model_config(model_id, n_layers=None):
    """The pinned configuration of model_id keeping its first n_layers (a
    shared copy: read it, do not modify it)."""
    import copy
    key = (model_id, None if n_layers is None else int(n_layers))
    if key not in _HW_CONFIGS:
        cfg = copy.deepcopy(d2l.hw_config(model_id))
        if n_layers is not None:
            cfg.num_hidden_layers = int(n_layers)
            if getattr(cfg, 'layer_types', None):
                cfg.layer_types = cfg.layer_types[:int(n_layers)]
        _HW_CONFIGS[key] = cfg
    return _HW_CONFIGS[key]
```

The attention function the benchmarks bind. A decode step of the harness has
one query token per sequence and a cache whose every slot is in use, so the
mask Transformers builds is all true; with a mask, Transformers' SDPA path
copies each KV head once per query head of its group before attending
(four copies for Qwen3-8B), which a serving kernel never does. Without the
mask SDPA attends through its grouped-query path and reads each cached byte
once. A prompt (more than one query token) goes to Transformers' SDPA
unchanged.

```{.python .input #workload-hardware-lib-attention}
%%tab pytorch
#@save
_HW_ATTENTION = 'd2l_full_cache'

#@save
def _full_cache_attention(module, query, key, value, attention_mask,
                          scaling=None, **kwargs):
    """Transformers attention for the benchmarks: one query token over a
    fully used cache through SDPA's grouped-query path (the all-true mask
    dropped); more tokens through Transformers' SDPA."""
    from transformers.integrations.sdpa_attention import (
        sdpa_attention_forward)
    if query.shape[2] == 1:
        out = F.scaled_dot_product_attention(query, key, value, scale=scaling,
                                             enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None
    return sdpa_attention_forward(module, query, key, value, attention_mask,
                                  scaling=scaling, **kwargs)

#@save
def _hw_bind_attention():
    """Register _full_cache_attention (and SDPA's mask builder) under
    _HW_ATTENTION with Transformers; repeated calls are harmless."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
    AttentionInterface.register(_HW_ATTENTION, _full_cache_attention)
    AttentionMaskInterface.register(_HW_ATTENTION, sdpa_mask)

#@save
def _hw_revision(model_id):
    return d2l.HF_REVISIONS.get(model_id) or d2l.HW_REVISIONS[model_id]

#@save
def hw_random_model(model_id, n_layers=None, seed=0):
    """model_id with random bf16 weights on the GPU, built from its pinned
    configuration (no download), its first n_layers only if given, with
    the benchmark attention, in eval mode."""
    import transformers
    cfg = _hw_model_config(model_id, n_layers)
    _hw_bind_attention()
    torch.manual_seed(seed)
    with d2l._quiet(), torch.device('cuda'):
        model = transformers.AutoModelForCausalLM.from_config(
            cfg, dtype=torch.bfloat16, attn_implementation=_HW_ATTENTION)
    return model.eval()

#@save
def hw_pretrained_model(model_id):
    """model_id with its real weights at the pinned revision (from the Hub
    cache), built exactly as hw_random_model builds it (same class, dtype,
    attention and GPU), in eval mode: only the weight values differ."""
    import transformers
    from transformers.utils import logging as hf_logging
    _hw_bind_attention()
    bar = hf_logging.is_progress_bar_enabled()
    hf_logging.disable_progress_bar()  # a notebook shows it despite _quiet
    try:
        with d2l._quiet():
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id, revision=_hw_revision(model_id),
                dtype=torch.bfloat16, attn_implementation=_HW_ATTENTION,
                device_map='cuda', use_safetensors=True)
    finally:
        if bar:
            hf_logging.enable_progress_bar()
    return model.eval()

#@save
def _hw_twin(model_id, n_layers=None, width=256, vocab=8192, head_dim=32):
    """A narrow twin of the model: the same layers and the same grouping of
    query heads (one KV head with its group), head size head_dim, hidden and
    MLP width `width`, a vocabulary of `vocab`; it launches the kernels of
    the model while its GPU work is small (attention included)."""
    import copy
    import transformers
    cfg = copy.deepcopy(_hw_model_config(model_id, n_layers))
    group = cfg.num_attention_heads // cfg.num_key_value_heads
    cfg.num_attention_heads, cfg.num_key_value_heads = group, 1
    cfg.head_dim = head_dim
    cfg.hidden_size = cfg.intermediate_size = width
    cfg.vocab_size = vocab
    if getattr(cfg, 'moe_intermediate_size', None):
        cfg.moe_intermediate_size = width
    _hw_bind_attention()
    with d2l._quiet(), torch.device('cuda'):
        twin = transformers.AutoModelForCausalLM.from_config(
            cfg, dtype=torch.bfloat16, attn_implementation=_HW_ATTENTION)
    return twin.eval()

#@save
def hw_static_cache(model, b, L):
    """A static cache of L + 1 positions for b sequences, holding L tokens
    (zeros: values do not change the time), and refill(), which sets every
    layer's fill level back to L in one kernel, so a captured decode step
    writes position L and attends over all L + 1 at every replay."""
    from transformers import StaticCache
    c = model.config
    cache = StaticCache(config=c, max_cache_len=L + 1)
    cache.early_initialization(b, c.num_key_value_heads, c.head_dim,
                               model.dtype, model.device)
    level = torch.full((len(cache.layers),), L, dtype=torch.long,
                       device=model.device)
    for i, layer in enumerate(cache.layers):
        layer.cumulative_length = level[i]  # views of one tensor
    return cache, lambda: level.fill_(L)

#@save
def _graph_kernels(fn):
    """Kernels in a CUDA graph of one call of fn (counted from the graph's
    nodes through the CUDA runtime), or None where that is unavailable."""
    import ctypes
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(g):
        fn()
    try:
        rt = ctypes.CDLL(f'libcudart.so.{torch.version.cuda.split(".")[0]}')
        n = ctypes.c_size_t(0)
        handle = ctypes.c_void_p(g.raw_cuda_graph())
        if rt.cudaGraphGetNodes(handle, None, ctypes.byref(n)) != 0:
            return None
        nodes = (ctypes.c_void_p * n.value)()
        rt.cudaGraphGetNodes(handle, nodes, ctypes.byref(n))
        kind, kernels = ctypes.c_int(0), 0
        for node in nodes:  # cudaGraphNodeTypeKernel is 0
            rt.cudaGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind))
            kernels += kind.value == 0
        return int(kernels)
    except (OSError, AttributeError, RuntimeError):
        return None

#@save
def _decode_kernels(model, b=1, L=2048):
    """Kernels one CUDA-graph decode step of hw_decode_bench runs."""
    token = torch.zeros((b, 1), dtype=torch.long, device='cuda')
    position = torch.tensor([L], device='cuda')
    with torch.inference_mode():
        cache, refill = hw_static_cache(model, b, L)

        def step():
            refill()
            model(token, past_key_values=cache, cache_position=position,
                  logits_to_keep=1)
        return _graph_kernels(step)
```

## Calibration

`hw_calibrate` (on the page) reads three tables of `box-microbench`: the copy
ladder, the bf16 products of the shape sweep and the format ladder. Measured
now, the same tables come from a short sweep.

```{.python .input #workload-hardware-lib-calibrate}
%%tab pytorch
#@save
def _calib_measure(copies=(2**18, 2**20, 2**22, 2**24, 2**26, 2**28, 2**30),
                   squares=(4096, 8192, 16384)):
    """hw_calibrate's tables measured now: copies of 256 KiB to 1 GiB and
    square bf16 products (as memory allows), each after a short warm-up."""
    ladder = _hw_frame([dict(bytes=s, kind='copy', **_hw_steady(
        lambda: d2l.hw_copy_bench(s, 'copy'), warm_s=0.25))
        for s in copies if _hw_fits(2 * s)])
    sweep = pd.DataFrame([dict(M=n, N=n, K=n, tflops=_hw_steady(
        lambda: d2l.hw_matmul_bench(n, n, n, 'bf16'), warm_s=0.25)['tflops'])
        for n in squares if _hw_fits(6 * n * n)])
    dtypes = pd.DataFrame(dict(format=['bf16'], tflops=[sweep.tflops.max()]))
    return {'ladder': ladder, 'shape-sweep': sweep, 'dtypes': dtypes}

#@save
def _calib_allreduce():
    """alpha_ar (s) and beta_ar (byte/s) of the box's all-reduce from the
    committed fit of sec_lm_server_paths, or None while it is absent (A5
    allows null)."""
    out = dict(alpha_ar=None, beta_ar=None)
    if not (hasattr(d2l, 'box_alpha_beta') and hw_registered('box-p2p')):
        return out
    try:
        fit = d2l.box_alpha_beta()
        out.update(alpha_ar=fit.get('alpha'), beta_ar=fit.get('beta'))
    except (KeyError, AttributeError, TypeError, OSError):
        pass
    return out

#@save
def show_calib(calib, chip, live=None):
    """The calibration against the chip's datasheet, then the live line:
    live (hw_calibrate measured now) compared on the committed GPU, its
    values otherwise."""
    if calib is None:
        print('box-microbench pending: no calibration, the datasheet only')
        return
    p16, bw = chip.p16_tflops, chip.mem_bw_tbs * 1e3
    P, beta = calib['P_eff_bf16'] / 1e12, calib['beta_eff'] / 1e9
    print(f'P_eff, the best bf16 product: {P:.0f} TFLOP/s '
          f'({P / p16:.0%} of the datasheet\'s {p16:.0f})')
    print(f'beta_eff, a DRAM-sized copy: {beta:,.0f} GB/s '
          f'({beta / bw:.0%} of the datasheet\'s {bw:,.0f})')
    print(f'alpha_k, one kernel in a CUDA graph: '
          f'{calib["alpha_k"] * 1e6:.1f} us')
    if calib.get('alpha_ar') is None:
        print('all-reduce alpha and beta (sec_lm_server_paths): not '
              'committed yet')
    else:
        print(f'all-reduce: alpha {calib["alpha_ar"] * 1e6:.0f} us, '
              f'beta {calib["beta_ar"] / 1e9:.1f} GB/s')
    if live is None:
        return
    # (label, committed, live, relative noise): a launch cost under a
    # microsecond moves by several percent between runs
    pts = [('P_eff', P, live['P_eff_bf16'] / 1e12, 0.0),
           ('beta_eff', beta, live['beta_eff'] / 1e9, 0.0),
           ('alpha_k', calib['alpha_k'] * 1e6, live['alpha_k'] * 1e6, 0.05)]
    if live.get('gpu') == calib.get('gpu'):
        print(f'live check on {live["gpu"]} (calibration): '
              f'{_hw_agreement(pts, True, "")}')
    else:
        units = ('TFLOP/s', 'GB/s', 'us')
        print(f'live values on {live.get("gpu")} (calibration): ' + ', '.join(
            f'{p[0]} {_hw_sig(p[2])} {u}' for p, u in zip(pts, units)))
```

## The Workload Grid

```{.python .input #workload-hardware-lib-grid}
%%tab pytorch
#@save
def _cut_chip(spec):
    """The chip_specs row the capacity cut of a configuration uses (the
    configuration's mem_gb, if given, replaces the row's capacity)."""
    chip = d2l.chip_specs().loc[spec['chip']].copy()
    if spec.get('mem_gb'):
        chip['mem_gb'] = spec['mem_gb']
    return chip

#@save
def _short(model_id):
    return model_id.split('/')[-1]

#@save
def _grid_models(spec):
    """The configuration's models in order of first appearance."""
    ids = [d[0] for d in spec['decode']] + [p[0] for p in spec['prefill']]
    ids += [t[0] for t in spec['train']] + [e[0] for e in spec['eager']]
    return list(dict.fromkeys(ids))

#@save
def _serving_points(spec, model_id, chip):
    """(decode points kept, points cut by capacity) of one model."""
    sm = d2l.ServingModel(_hw_model_config(model_id), chip)
    kept, cut = [], []
    for mid, batches, contexts, series in spec['decode']:
        if mid != model_id:
            continue
        for L in contexts:
            for b in batches:
                (kept if b <= sm.max_batch(L) else cut).append((b, L, series))
    return kept, cut

#@save
def _serving_rows(spec, model_id, chip):
    """Decode, prefill and eager points of one model on one card."""
    kept, cut = _serving_points(spec, model_id, chip)
    model, rows = hw_random_model(model_id), []
    base = dict(model=model_id, n_layers=model.config.num_hidden_layers)
    for b, L, series in kept:
        rows.append(dict(base, workload='decode', series=series, b=b, L=L,
                         **d2l.hw_decode_bench(model, b, L)))
        d2l.lm_free()
    prompts = next((p for mid, p in spec['prefill'] if mid == model_id), ())
    for n in prompts:
        rows.append(dict(base, workload='prefill', series='grid', b=1, L=n,
                         **d2l.hw_prefill_bench(model, n)))
        d2l.lm_free()
    for mid, batches, L in spec['eager']:
        for b in batches if mid == model_id else ():
            rows.append(dict(base, workload='eager', series='eager', b=b,
                             L=L, **d2l.hw_decode_bench(model, b, L,
                                                        graph=False)))
            d2l.lm_free()
    kernels = _decode_kernels(model)
    del model
    d2l.lm_free()
    if prompts:  # the CPU's time to launch an eager prefill
        twin = _hw_twin(model_id)
        floors = {n: d2l.hw_prefill_bench(twin, n)['ms'] for n in prompts}
        for r in rows:
            if r['workload'] == 'prefill':
                r['floor_ms'] = floors[r['L']]
        del twin
        d2l.lm_free()
    return rows, [[model_id, b, L] for b, L, _ in cut], kernels

#@save
def _train_rows(spec, model_id):
    """Training points of one model, one depth on the card at a time."""
    rows = []
    for mid, layers, shapes in spec['train']:
        for n_layers in layers if mid == model_id else ():
            model = hw_random_model(model_id, n_layers)
            for b, s in shapes:
                rows.append(dict(model=model_id, workload='train',
                                 series='grid', b=b, L=s,
                                 n_layers=model.config.num_hidden_layers,
                                 **d2l.hw_train_bench(model, b, s)))
            del model
            d2l.lm_free()
            twin = _hw_twin(model_id, n_layers)  # the launch floor
            for r, (b, s) in zip(rows[-len(shapes):], shapes):
                r['floor_ms'] = d2l.hw_train_bench(twin, b, s)['ms']
            del twin
            d2l.lm_free()
    return rows

#@save
def _workload_compute(name):
    """Measure every point of HW_WORKLOAD_CONFIGS[name] on this GPU, one
    model on the card at a time."""
    spec, rows, cut, kernels = d2l.HW_WORKLOAD_CONFIGS[name], [], [], {}
    chip = _cut_chip(spec)
    calib = d2l.hw_calibrate()  # the card as it was during this run
    for model_id in _grid_models(spec):
        r, c, kernels[model_id] = _serving_rows(spec, model_id, chip)
        rows += r + _train_rows(spec, model_id)
        cut += c
    df = _hw_frame(rows)
    df.attrs.update(config=name, chip=spec['chip'], cut=cut,
                    kernels_per_step=kernels, calib=calib)
    return df

#@save
def _show_grid(grid):
    """Counts of the measured points per model and workload, the points the
    capacity cut dropped, and eager against graph decode at b = 1."""
    if grid.empty:
        print(grid.attrs.get('label', 'no grid'))
        return
    names = {'decode': 'decode', 'prefill': 'prefill', 'train': 'training'}
    parts = []
    for model_id, g in grid.groupby('model', sort=False):
        counts = g.workload.value_counts()
        parts.append(_short(model_id) + ' ' + ', '.join(
            f'{counts.get(w, 0)} {n}' for w, n in names.items()))
    print(textwrap.fill('points measured: ' + '; '.join(parts), 79,
                        subsequent_indent='  '))
    cut = {}
    for model_id, b, L in grid.attrs.get('cut', []):
        cut.setdefault(_short(model_id), []).append(f'({b}, {L:,})')
    text = '; '.join(f'{m} ' + ' '.join(p) for m, p in cut.items())
    print(textwrap.fill('cut by capacity, (b, L): ' + (text or 'none'), 79,
                        subsequent_indent='  '))
    ratios = []
    for _, e in grid[(grid.workload == 'eager') & (grid.b == 1)].iterrows():
        g = grid[(grid.workload == 'decode') & (grid.model == e.model)
                 & (grid.b == 1) & (grid.L == e.L)]
        if len(g):
            ratios.append(f'{_short(e.model)} {e.ms / g.ms.iloc[0]:.0f}x')
    if ratios:
        L = int(grid[grid.workload == 'eager'].L.iloc[0])
        print(textwrap.fill(f'eager decode at b = 1, L = {L:,} (no CUDA '
                            'graph): ' + ', '.join(ratios) + ' the graph step',
                            79, subsequent_indent='  '))
    if 'floor_ms' in grid:
        floors = grid.dropna(subset=['floor_ms']).groupby(
            ['model', 'workload'], sort=False).floor_ms
        parts = [f'{_short(m)} {w} {lo:.0f}-{hi:.0f} ms' for (m, w), lo, hi
                 in zip(floors.min().index, floors.min(), floors.max())]
        print(textwrap.fill('launch floor of the eager steps (narrow twin): '
                            + ', '.join(parts), 79, subsequent_indent='  '))
```

The live check repeats four committed points on the reader's GPU: a decode,
a prefill and a training point of the first model, and a decode point of the
second.

```{.python .input #workload-hardware-lib-grid-live}
%%tab pytorch
#@save
def _live_rows(grid):
    """Row indices of the four live-check points."""
    models = list(dict.fromkeys(grid.model))
    picks = []
    for workload, model_id, prefer in (
            ('decode', models[0], dict(b=1, L=2048)),
            ('prefill', models[0], dict(L=2048)),
            ('train', models[0], {}),
            ('decode', models[-1], dict(b=16, L=2048))):
        rows = grid[(grid.workload == workload) & (grid.model == model_id)]
        best = rows
        for k, v in prefer.items():
            best = best[best[k] == v]
        if len(rows):
            picks.append((best if len(best) else rows).index[0])
    return list(dict.fromkeys(picks))

#@save
def _grid_live(r, same):
    pts = []
    for i in _live_rows(r):
        row = r.loc[i]
        layers = None if row.workload != 'train' else int(row.n_layers)
        model = hw_random_model(row.model, layers)
        b, L, name = int(row.b), int(row.L), _short(row.model)
        if row.workload == 'decode':
            live = d2l.hw_decode_bench(model, b, L, steps=16)
            label = f'{name} decode b={b}, L={L:,}'
        elif row.workload == 'prefill':
            live = d2l.hw_prefill_bench(model, L, repeats=2)
            label = f'{name} prefill {L:,}'
        else:
            live = d2l.hw_train_bench(model, b, L, repeats=3, warmup=2)
            label = f'{name} training'
        del model
        d2l.lm_free()
        pts.append((label, row.ms, live['ms'],
                    _hw_noise(r, r.index.get_loc(i), 'ms')))
    return _hw_agreement(pts, same, 'ms')

#@save
def hw_workload_grid(config='box'):
    """The measured grid of HW_WORKLOAD_CONFIGS[config]: one row per point
    (workload decode, prefill, train or eager; model; b; L the context,
    prompt or training sequence length; n_layers; ms and MAD; peak GiB), the
    points cut by capacity, the kernels per decode step and the run's own
    calibration in attrs."""
    spec = d2l.HW_WORKLOAD_CONFIGS[config]
    return _hw_result(
        'box-workloads', lambda: _workload_compute(config),
        dict(config=config, grid={k: v for k, v in spec.items()
                                  if k != 'moe'}),
        entry=f'grid-{config}', live_check=_grid_live, show=_show_grid,
        live_label='workloads')
```

## The Weight-Value Control

```{.python .input #workload-hardware-lib-control}
%%tab pytorch
#@save
_HW_ARMS = ('random', 'real', 'random again')

#@save
def _control_compute(model_id, b, L):
    """Random, real, then random weights again (the run-to-run noise), one
    model on the card at a time, with the kernels of each decode step."""
    rows = []
    for arm in _HW_ARMS:
        model = (hw_pretrained_model(model_id) if arm == 'real'
                 else hw_random_model(model_id))
        rows.append(dict(arm=arm, kernels=_decode_kernels(model, b, L),
                         **d2l.hw_decode_bench(model, b, L)))
        del model
        d2l.lm_free()
    return pd.DataFrame(rows)

#@save
def _control_noise(t):
    """The control's noise (ms): 3 MADs or the random arms' difference."""
    return max(3 * t.mad.max(), math.fabs(t.ms['random'] -
                                          t.ms['random again']))

#@save
def _show_control(r, model_id, b, L):
    if r.empty:
        print(r.attrs.get('label', 'no control'))
        return
    t = r.set_index('arm')
    random = (t.ms['random'] + t.ms['random again']) / 2
    diff, noise = t.ms['real'] - random, _control_noise(t)
    same = 'the same' if t.kernels.nunique() == 1 else 'different'
    print(f'{_short(model_id)}, b = {b}, L = {L:,}: random '
          f'{t.ms["random"]:.2f} ms, real {t.ms["real"]:.2f}, random again '
          f'{t.ms["random again"]:.2f}')
    print(f'real minus random {diff:+.3f} ms ({diff / random:+.1%}), noise '
          f'{noise:.3f} ms, {same} kernels ({int(t.kernels.iloc[0]):,})')

#@save
def hw_weight_control(model_id, b, L):
    """Decode step time at (b, L) with random weights, model_id's real
    weights and random weights again, back to back, one model on the card
    at a time (the two random arms measure the run-to-run noise)."""
    def live(r, same):
        new = _control_compute(model_id, b, L).set_index('arm')
        t = r.set_index('arm')
        return _hw_agreement([(w, t.ms[w], new.ms[w], t.mad[w] / t.ms[w])
                              for w in _HW_ARMS], same, 'ms')
    return _hw_result(
        'box-workloads', lambda: _control_compute(model_id, b, L),
        dict(model=model_id, b=b, L=L), entry=f'control-{_short(model_id)}',
        live_check=live, show=lambda r: _show_control(r, model_id, b, L),
        live_label='weights')
```

## A Ladder of Models

M0 is the datasheet; M1 the calibration (`P_eff`, `beta_eff`); M2 adds the
launch terms, both measured, none fitted: in a CUDA graph (decode) the kernels
of one step (counted from its graph, scaled to the layers a point keeps) times
`alpha_k`; an eager step (prefill, training) takes at least its launch floor,
the time of the same step on a narrow twin of the model (same layers and
kernels; width 256, one KV head with its query group, head size 32), which
is the CPU's time to launch the kernels. M3
fits `eta_compute`, `eta_kv` and `kappa` on the calibration subset (every
third measured point, in grid order) by least squares on log ratios, so
relative errors count alike.

```{.python .input #workload-hardware-lib-fit}
%%tab pytorch
#@save
_HW_FIT_WORKLOADS = ('train', 'prefill', 'decode')

#@save
def _ladder_calibs(calib, grid, model_id, n_layers, extra=None):
    """The calibrations of M0-M2 (and M3 with extra) for one point."""
    full = _hw_model_config(model_id).num_hidden_layers
    k = (grid.attrs.get('kernels_per_step') or {}).get(model_id)
    m1 = {key: calib[key] for key in ('P_eff_bf16', 'beta_eff', 'alpha_k')}
    m2 = dict(m1, kernels_per_step=(k or 0) * n_layers / full)
    return {'M0': None, 'M1': m1, 'M2': m2, 'M3': dict(m2, **(extra or {}))}

#@save
def _predict(row, chip, calib):
    """Predicted seconds of a grid row under one calibration; with launch
    terms (M2, M3) an eager step takes at least its launch floor."""
    cfg = _hw_model_config(row.model, row.n_layers)
    b, L = int(row.b), int(row.L)
    if row.workload == 'decode':
        return d2l.ServingModel(cfg, chip, calib).decode_step(b, L)
    if row.workload == 'train':
        t = d2l.TrainStepModel(cfg, chip, calib).step_time(b, L)
    else:
        t = d2l.ServingModel(cfg, chip, calib).prefill_time(L, b)
    floor = getattr(row, 'floor_ms', math.nan) / 1e3
    launch = calib is not None and 'kernels_per_step' in calib
    return max(t, floor) if launch and np.isfinite(floor) else t

#@save
def _ladder_errors(points, chip, calib, grid, extra=None, models=None):
    """Relative error (predicted / measured - 1) of each model M0-M3."""
    models = models or ('M0', 'M1', 'M2', 'M3')
    out = {m: [] for m in models}
    for row in points.itertuples():
        calibs = _ladder_calibs(calib, grid, row.model, row.n_layers, extra)
        for m in models:
            out[m].append(_predict(row, chip, calibs[m]) / (row.ms / 1e3) - 1)
    return pd.DataFrame(out, index=points.index)

#@save
def _fit_m3(points, chip, calib, grid):
    """eta_compute, eta_kv and kappa minimizing the log ratios on points."""
    from scipy.optimize import least_squares
    names = ('eta_compute', 'eta_kv', 'kappa')

    def residuals(x):
        extra = dict(zip(names, np.exp(x)))
        e = _ladder_errors(points, chip, calib, grid, extra, ('M3',))['M3']
        return np.log1p(e.to_numpy())
    fit = least_squares(residuals, np.zeros(3), bounds=(
        np.log([0.05, 0.05, 0.1]), np.log([2.0, 2.0, 20.0])))
    return dict(zip(names, np.exp(fit.x)))

#@save
def _grid_entry(log):
    """The workload grid of a campaign log (its grid-* entry), or log."""
    if isinstance(log, dict):
        keys = [k for k in log if k.startswith('grid-')]
        return log[keys[0]] if keys else pd.DataFrame()
    return log

#@save
def fit_workload_models(grid, calib=None, chip=None, within=0.15):
    """M0-M3 on the grid's measured points (calib: hw_calibrate's dict,
    default the run's own; chip: its chip_specs row, default the grid's):
    median and max relative error per workload on the held-out points, the
    fitted M3 parameters, and how many held-out points M0 and M3 predict
    within `within`; returns the errors, the parameters and each model's M3
    calibration (for ServingModel). A campaign log that is not committed
    prints its status."""
    if isinstance(grid, CampaignPending):
        print(hw_status_line(grid))
        return None
    grid = _grid_entry(grid)
    if grid.empty:
        print(grid.attrs.get('label', 'no grid'))
        return None
    calib = calib or grid.attrs['calib']
    chip = chip if chip is not None else d2l.chip_specs().loc[
        grid.attrs['chip']]
    points = grid[grid.workload.isin(_HW_FIT_WORKLOADS)]
    is_cal = np.arange(len(points)) % 3 == 0  # the calibration subset C
    extra = _fit_m3(points[is_cal], chip, calib, grid)
    errors = _ladder_errors(points, chip, calib, grid, extra)
    errors['model'], errors['workload'] = points.model, points.workload
    errors['held_out'] = ~is_cal
    _show_ladder_fit(errors, extra, int(is_cal.sum()), within)
    calibs = {}  # each model's M3 calibration at full depth
    for m in dict.fromkeys(points.model):
        depth = _hw_model_config(m).num_hidden_layers
        calibs[m] = _ladder_calibs(calib, grid, m, depth, extra)['M3']
    return dict(errors=errors, params=extra, calib=calibs)

#@save
def _show_ladder_fit(errors, extra, n_cal, within):
    held = errors[errors.held_out]
    table = {}
    for m in ('M0', 'M1', 'M2', 'M3'):
        table[m] = {}
        for w in _HW_FIT_WORKLOADS:
            e = held[held.workload == w][m].abs() * 100
            table[m][w] = f'{e.median():.0f} / {e.max():.0f}' if len(e) else ''
    show_table(pd.DataFrame(table).T.rename_axis('model'))
    print('(median / max relative error in %, held-out points)')
    print(f'M3 fitted on {n_cal} of {len(errors)} points: ' + ', '.join(
        f'{k} {v:.2f}' for k, v in extra.items()))
    parts = []
    for model_id, h in held.groupby('model', sort=False):
        ok = [int((h[m].abs() <= within).sum()) for m in ('M0', 'M3')]
        parts.append(f'{_short(model_id)} {ok[0]} / {ok[1]} of {len(h)}')
    print(textwrap.fill(f'held-out points within {within:.0%}, M0 / M3: '
                        + ', '.join(parts), 79, subsequent_indent='  '))
```

## The Frontier Plot

```{.python .input #workload-hardware-lib-frontier}
%%tab pytorch
#@save
def _measured_decode(grid, model_id, contexts):
    """The measured CUDA-graph decode points of model_id at the contexts:
    b, L, ms, user_tps, gpu_tps, sorted by L and b."""
    if grid is None or grid.empty:
        return pd.DataFrame(columns=['b', 'L', 'ms', 'user_tps', 'gpu_tps'])
    d = grid[(grid.workload == 'decode') & (grid.model == model_id)
             & (grid.series == 'grid') & grid.L.isin(contexts)]
    d = d[['b', 'L', 'ms']].sort_values(['L', 'b']).reset_index(drop=True)
    return d.assign(user_tps=1e3 / d.ms, gpu_tps=d.b * 1e3 / d.ms)

#@save
def plot_frontier(frontier, model, measured=None):
    """Per-GPU against per-user tokens/s along a ServingModel's frontier at
    each context (lines up to the capacity limit) with the measured CUDA-
    graph decode points of the same model (markers); prints the compute
    roof and the largest distance of a measured point from its line (the
    step-time error at the same b). Returns the measured points."""
    contexts = list(dict.fromkeys(frontier.L))
    pts = _measured_decode(measured, model.cfg._name_or_path, contexts)
    X, Y, fmts, legend = [], [], [], []
    for i, L in enumerate(contexts):
        f = frontier[(frontier.L == L) & (frontier.regime != 'infeasible')]
        X.append(f.user_tps.tolist()), Y.append(f.gpu_tps.tolist())
        fmts.append(f'C{i}-'), legend.append(f'L = {L:,}')
    for i, L in enumerate(contexts):
        p = pts[pts.L == L]
        if len(p):
            X.append(p.user_tps.tolist()), Y.append(p.gpu_tps.tolist())
            fmts.append(f'C{i}o')
    d2l.plot(X, Y, xlabel='tokens/s per user', ylabel='tokens/s per GPU',
             legend=legend, fmts=fmts, xscale='log', yscale='log',
             figsize=(5, 3.5))
    roof = model.P / (2 * model.n_matmul()) / model.n_chips
    print(f'compute roof P / (2 N_act): {roof:,.0f} tokens/s per GPU')
    if len(pts):
        err = [math.fabs(model.decode_step(int(r.b), int(r.L))
                         / (r.ms / 1e3) - 1) for r in pts.itertuples()]
        print(f'measured decode points: {len(pts)}, at most {max(err):.0%} '
              'off the line (step time at the same b)')
    return pts
```

## Expert Coverage of a Mixture of Experts

The regeneration records Qwen3-30B-A3B's routing on real text: the model is
split layer by layer over the box's four cards (`device_map='auto'`),
sequences of exactly `n_tokens` tokens go through the model body (no LM
head) in micro-batches, and a forward hook on each layer's router keeps the
top-K experts of every sequence's last position. Inputs: the first
FineWeb-Edu validation documents (ODC-By) with at least `n_tokens` tokens
under both the reference tokenizer and Qwen3's (decoded exactly by the
byte-level reference tokenizer, re-tokenized by Qwen3's), and disjoint
windows of the held-out code text of :numref:`sec_lm_scoring` (whose files
carry their own licences in that artifact). Coverage at batch b is computed
afterwards by resampling b sequences without replacement.

```{.python .input #workload-hardware-lib-moe}
%%tab pytorch
#@save
class MoECoverage(collections.namedtuple(
        'MoECoverage', ['table', 'E', 'K', 'n_layers', 'routes', 'steps'])):
    """Measured expert coverage: table (source, b, mean, lo, hi, uniform:
    distinct experts per layer at batch b, mean and 5-95% over draws, and
    the uniform formula), the router's top-K sets per source (sequences x
    layers x K), eager decode steps of the layer-split model."""
    def coverage(self, b, source='prose'):
        """Fraction of the E experts that b sequences touch (interpolated
        in log b; NaN while the result is pending)."""
        t = self.table[self.table.source == source]
        if len(t) == 0:
            return math.nan
        return float(np.interp(np.log2(b), np.log2(t.b.to_numpy(float)),
                               t['mean'].to_numpy(float))) / self.E

#@save
def _moe_pending():
    table = pd.DataFrame(columns=['source', 'b', 'mean', 'lo', 'hi',
                                  'uniform'])
    return MoECoverage(table, 128, 8, 48, {}, pd.DataFrame())

#@save
def _moe_inputs(spec):
    """{'prose': ids, 'code': ids}, each (sequences x n_tokens) of Qwen3
    tokens, and the FineWeb-Edu document indices used."""
    import transformers
    n = spec['n_tokens']
    with d2l._quiet():
        tok = transformers.AutoTokenizer.from_pretrained(
            spec['model'], revision=_hw_revision(spec['model']))
        shards, ref = d2l.fwedu_shards('val'), d2l.r2_tokenizer()
        with open(d2l.download_keyed('lm-heldout-texts-v1')) as f:
            texts = json.load(f)
    prose, docs = [], []
    for i, doc in shards.iter_docs():
        if len(doc) - 1 < n:  # the document's tokens without its EOS
            continue
        ids = tok(ref.decode(doc[:-1].tolist()))['input_ids']
        if len(ids) >= n:
            prose.append(ids[:n]), docs.append(i)
        if len(prose) == spec['n_prose']:
            break
    code = next(t['text'] for t in texts['texts'] if t['name'] == 'code')
    ids = tok(code)['input_ids']
    windows = [ids[j * n:(j + 1) * n] for j in range(spec['n_code'])]
    if len(windows[-1]) != n:
        raise ValueError('the held-out code text is too short')
    return {'prose': np.array(prose), 'code': np.array(windows)}, docs

#@save
def _moe_routes(model, ids, micro_batch):
    """Top-K experts (sequences x layers x K, sorted) chosen at each
    sequence's last position, from forward hooks on the layers' routers."""
    layers, chosen, shape = model.model.layers, {}, {}

    def hook(i):
        def record(module, inputs, output):  # (logits, scores, indices)
            idx = output[2].view(shape['b'], -1, output[2].shape[-1])
            chosen[i] = idx[:, -1].cpu()
        return record
    handles = [layer.mlp.gate.register_forward_hook(hook(i))
               for i, layer in enumerate(layers)]
    out = []
    try:
        with torch.inference_mode():
            for j in range(0, len(ids), micro_batch):
                x = torch.as_tensor(ids[j:j + micro_batch])
                shape['b'] = len(x)
                model.model(input_ids=x.to(model.device))
                out.append(torch.stack([chosen[i]
                                        for i in range(len(layers))], 1))
    finally:
        for h in handles:
            h.remove()
    return torch.cat(out).sort(-1).values.numpy().astype(np.uint8)

#@save
def _coverage_table(routes, E, K, batches, draws, seed):
    """Distinct experts per layer in the union of b random sequences
    (without replacement), averaged over layers: mean and 5-95% over
    draws, per source, with the uniform formula beside it."""
    rng, rows = np.random.default_rng(seed), []
    for source, r in routes.items():
        hit = np.zeros((len(r), r.shape[1], E), dtype=bool)
        np.put_along_axis(hit, r.astype(np.int64), True, axis=2)
        for b in batches:
            if b > len(r):
                continue
            touched = [hit[rng.choice(len(r), b, replace=False)].any(0)
                       .sum(-1).mean() for _ in range(draws)]
            rows.append(dict(source=source, b=b, mean=np.mean(touched),
                             lo=np.percentile(touched, 5),
                             hi=np.percentile(touched, 95),
                             uniform=d2l.moe_expert_coverage(E, K, b)))
    return pd.DataFrame(rows)

#@save
def _moe_steps(model, batches, context):
    """Eager decode milliseconds of the layer-split model at each batch
    (Transformers' default grouped_mm experts; median of hf_decode_ms)."""
    rows = []
    for b in batches:
        prompt = torch.randint(model.config.vocab_size, (b, context),
                               device=model.device)
        cache = d2l.hf_prefill(model, prompt, context)[0].past_key_values
        ms = d2l.hf_decode_ms(model, cache, prompt[:, -1:], steps=16)[0]
        rows.append(dict(b=b, L=context, ms=ms))
        del cache
        d2l.lm_free()
    return pd.DataFrame(rows)

#@save
def _moe_compute(spec):
    ids, docs = _moe_inputs(spec)
    with d2l._quiet():
        model = d2l.load_hf(spec['model'], _hw_revision(spec['model']),
                            device_map='auto')[0].eval()
    E, K = model.config.num_experts, model.config.num_experts_per_tok
    n_layers = model.config.num_hidden_layers
    routes = {s: _moe_routes(model, x, spec['micro_batch'])
              for s, x in ids.items()}
    steps = (_moe_steps(model, spec['step_batches'], spec['step_context'])
             if spec['step_batches'] else pd.DataFrame())
    del model
    d2l.lm_free()
    table = _coverage_table(routes, E, K, spec['batches'], spec['draws'],
                            spec['seed'])
    table.attrs.update(docs=docs, sources=dict(
        prose='FineWeb-Edu validation split (ODC-By)',
        code='held-out code text of sec_lm_scoring'))
    return MoECoverage(table, E, K, n_layers, routes, steps)

#@save
def hw_moe_coverage(config='box'):
    """The measured expert coverage of HW_WORKLOAD_CONFIGS[config]['moe']
    (a MoECoverage): the committed four-card regeneration; no live check
    (the model needs about 61 GB over several cards)."""
    spec = d2l.HW_WORKLOAD_CONFIGS[config]['moe']
    return _hw_result('box-moe', lambda: _moe_compute(spec),
                      dict(config=config, moe=spec),
                      entry=f'coverage-{config}', empty=_moe_pending)

#@save
def plot_coverage(cov, formula):
    """Distinct experts per layer against batch: the measured mean with its
    5-95% band per source, the uniform formula(E, K, b) as a dashed line;
    prints measured over uniform at b = 8 and b = 64."""
    t = cov.table
    if len(t) == 0:
        print('box-moe pending: the uniform formula only')
        t = pd.DataFrame(dict(b=[1, 2, 4, 8, 16, 32, 64, 128, 256]))
    b = np.array(sorted(set(t.b)), dtype=float)
    X, Y = [b.tolist()], [[formula(cov.E, cov.K, x) for x in b]]
    fmts, legend = ['k--'], ['uniform routing']
    groups = list(t.groupby('source', sort=False)) if 'source' in t else []
    for i, (source, g) in enumerate(groups):
        X.append(g.b.tolist()), Y.append(g['mean'].tolist())
        fmts.append(f'C{i}-o'), legend.append(f'measured, {source}')
    d2l.plot(X, Y, xlabel='batch b (sequences)',
             ylabel='experts touched per layer', legend=legend, fmts=fmts,
             xscale='log', figsize=(5, 3.5))
    ax = d2l.plt.gca()
    for i, (source, g) in enumerate(groups):
        ax.fill_between(g.b, g.lo, g.hi, color=f'C{i}', alpha=0.2, lw=0)
    for b in (8, 64):
        rows = [(s, g[g.b == b].iloc[0]) for s, g in groups
                if (g.b == b).any()]
        parts = [f'{s} {r["mean"] / r.uniform:.2f}' for s, r in rows]
        if parts:
            print(f'measured / uniform at b = {b}: ' + ', '.join(parts))
```

## Other Chips and the InferenceX Snapshot

The snapshot (static table `inferencex-snapshot`) is a tar of `snapshot.csv`
(columns `model, hw, framework, precision, isl, osl, conc, tp, tps_per_gpu,
tps_per_user, commit, retrieved`), the upstream `LICENSE` and `NOTICE.md`.
Its rows are one dense 70B-class model (`HW_SHAPES['dense-70b']`) at one
interactivity point. While it is not committed, the reader returns a pending
`Snapshot` (false) that still carries the shape, precision and interactivity
point the predictions use.

```{.python .input #workload-hardware-lib-inferencex}
%%tab pytorch
#@save
_HW_IX_CHIPS = {'H100': 'H100 SXM', 'H200': 'H200 SXM', 'B200': 'B200',
                'MI355X': 'MI355X'}

#@save
class Snapshot(collections.namedtuple(
        'Snapshot', ['table', 'model_cfg', 'dtype_bytes', 'kv_bytes',
                     'user_tps', 'context', 'pending'])):
    """An InferenceX snapshot: per chip (chip_specs names), tensor-parallel
    degree and measured tokens/s per GPU at user_tps tokens/s per user."""
    def __bool__(self):
        return not self.pending

#@save
def _read_snapshot(path, shape):
    """A Snapshot from an InferenceX snapshot.csv."""
    df = pd.read_csv(path)
    df['chip'] = df.hw.map(_HW_IX_CHIPS)
    dtype = 1 if str(df.precision.iloc[0]).lower().startswith('fp8') else 2
    return Snapshot(df[df.chip.notna()].reset_index(drop=True), shape, dtype,
                    dtype, float(df.tps_per_user.median()),
                    int(df.isl.iloc[0] + df.osl.iloc[0] // 2), False)

#@save
def inferencex_snapshot():
    """The committed InferenceX snapshot (Apache-2.0, with LICENSE and
    NOTICE), or a pending Snapshot with the default prediction setting
    (dense 70B shape, FP8 weights and KV, 50 tokens/s per user, 2,048-token
    context) while it is not committed."""
    shape = d2l.HW_SHAPES['dense-70b']
    if not hw_registered('inferencex-snapshot'):
        return Snapshot(pd.DataFrame(columns=['chip', 'tp', 'tps_per_gpu']),
                        shape, 1, 1, 50.0, 2048, True)
    return _read_snapshot(os.path.join(_hw_static('inferencex-snapshot'),
                                       'snapshot.csv'), shape)

#@save
def _tps_at(sm, u, L):
    """Tokens/s per GPU of a ServingModel when each user gets at least u
    tokens/s at context L: at the largest batch whose step takes at most
    1/u and fits (NaN when even b = 1 is slower or nothing fits)."""
    b_max = sm.max_batch(L)
    if b_max < 1 or sm.decode_step(1, L) > 1 / u:
        return math.nan
    lo, hi = 1, b_max
    while lo < hi:  # the step time grows with b: bisect
        mid = (lo + hi + 1) // 2
        if sm.decode_step(mid, L) <= 1 / u:
            lo = mid
        else:
            hi = mid - 1
    return lo / sm.decode_step(lo, L) / sm.n_chips

#@save
def _kendall_tau(x, y):
    """Kendall's tau of two equally long sequences (ties count zero)."""
    n, s = len(x), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return s / (n * (n - 1) / 2) if n > 1 else math.nan

#@save
def rank_against(snap, specs, chips=('H100 SXM', 'H200 SXM', 'B200', 'MI355X',
                                     'RTX 4090', 'Groq LPU'), default_tp=8):
    """Datasheet predictions of tokens/s per GPU at the snapshot's
    interactivity point for each chip (its tensor-parallel degree from the
    snapshot, else default_tp; the Groq row on the chips that hold the
    weights) beside the InferenceX value and both ranks; Kendall's tau over
    the chips in the snapshot."""
    W = d2l.ServingModel(snap.model_cfg, specs.loc[chips[0]],
                         dtype_bytes=snap.dtype_bytes).weight_bytes()
    measured = snap.table.set_index('chip') if len(snap.table) else None
    rows = {}
    for chip in chips:
        row = specs.loc[chip]
        if measured is not None and chip in measured.index:
            tp, ix = int(measured.tp[chip]), float(measured.tps_per_gpu[chip])
        elif row.mem_type == 'SRAM':
            tp, ix = d2l.sram_chips_needed(W, row.mem_gb * 1e9), math.nan
        else:
            tp, ix = default_tp, math.nan
        sm = d2l.ServingModel(snap.model_cfg, row, n_chips=tp,
                              dtype_bytes=snap.dtype_bytes,
                              kv_bytes=snap.kv_bytes)
        predicted = _tps_at(sm, snap.user_tps, snap.context)
        rows[chip] = dict(tp=tp, predicted=predicted, inferencex=ix)
    t = pd.DataFrame(rows).T.astype(float)
    ranked = t.index.isin(list(_HW_IX_CHIPS.values()))  # never the others
    t['rank'] = t.predicted[ranked].rank(ascending=False)
    t['ix_rank'] = t.inferencex.rank(ascending=False)
    show_table(t, digits=0, headers={
        'tp': 'chips', 'predicted': 'predicted tok/s/GPU',
        'inferencex': 'InferenceX', 'rank': 'rank',
        'ix_rank': 'InferenceX rank'})
    both = t.dropna(subset=['inferencex'])
    if not snap:
        fmt = 'FP8' if snap.dtype_bytes == 1 else 'bf16'
        print('InferenceX snapshot not committed yet: predictions only')
        print(f'(dense 70B, {fmt}, {snap.user_tps:.0f} tokens/s per user, '
              f'context {snap.context:,})')
    elif len(both) > 1:
        tau = _kendall_tau(both.predicted.tolist(), both.inferencex.tolist())
        print(f'Kendall tau over {len(both)} chips: {tau:.2f}')
    absent = [c for c in t.index if c not in both.index]
    if snap and absent:
        print('predicted, not ranked (absent from the snapshot): '
              + ', '.join(absent))
    for chip in t.index[specs.loc[t.index].mem_type == 'SRAM']:
        print(textwrap.fill(
            f'{chip}: a bound from the SRAM bandwidth of {int(t.tp[chip])} '
            'chips alone; the hops between chips, not modeled here, set '
            'its speed', 79))
    return t
```

## The Regeneration Drivers

`tools/hw_regen.py box-workloads` (one whole card) runs the page's grid on
the full configuration, then the reader's `t1` configuration on the same card
and the two weight-value controls, and logs every point in a BenchLog.
`tools/hw_regen.py box-moe --processes 1` (all four cards; routing is a
count, not a timing, so one process) records the router.

```{.python .input #workload-hardware-lib-drivers}
%%tab pytorch
#@save
def _hw_regen_check(name, config):
    if not torch.cuda.is_available():
        raise RuntimeError(f'{name} needs a CUDA GPU')
    if (os.environ.get('D2L_HW_CONFIG') or 'box') != config:
        raise ValueError(f'config {config!r} needs D2L_HW_CONFIG={config} '
                         '(tools/hw_regen.py --config)')
    _hw_check_hardware(name, config)

#@save
def regen_workloads(config='box'):
    """Driver of box-workloads: the grid of config (and on the box the
    t1 grid too), the weight-value controls, a BenchLog record per point."""
    _hw_regen_check('box-workloads', config)
    folder = _hw_regen_dir('box-workloads')
    configs = ('box', 't1') if config == 'box' else (config,)
    with d2l.BenchLog(os.path.join(folder, 'benchlog.jsonl'), 'torch',
                      _HW_TIERS[config]) as log:
        for name in configs:
            grid = hw_workload_grid(name)
            for rec in grid.to_dict('records'):
                log.record(f'hw_{rec["workload"]}_bench', {k: rec[k] for k in (
                    'model', 'b', 'L', 'n_layers', 'series')}, rec)
            model_id, b, L = d2l.HW_WORKLOAD_CONFIGS[name]['control']
            for rec in hw_weight_control(model_id, b, L).to_dict('records'):
                log.record('hw_decode_bench', dict(model=model_id, b=b, L=L),
                           rec)

#@save
def regen_moe(config='box'):
    """Driver of box-moe: the router recording of HW_WORKLOAD_CONFIGS
    [config]['moe'] on every visible GPU (the box's four)."""
    _hw_regen_check('box-moe', config)
    cov = hw_moe_coverage(config)
    folder = _hw_regen_dir('box-moe')
    with d2l.BenchLog(os.path.join(folder, 'benchlog.jsonl'), 'torch',
                      _HW_TIERS[config]) as log:
        for rec in cov.table.to_dict('records'):
            log.record('moe router', {k: rec[k] for k in ('source', 'b')}, rec)
        for rec in cov.steps.to_dict('records'):
            log.record('hf_decode_ms', {k: rec[k] for k in ('b', 'L')}, rec)
```
