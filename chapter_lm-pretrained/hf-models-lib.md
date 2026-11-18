# Pretrained Models: Kernels and Measurement Harness (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_hf_models` explains in a sentence and does not show:

- the kernel routing that `d2l.load_hf` applies: `bind_hf_kernels` (route
  Transformers' Gated DeltaNet functions by device and dtype, register the
  `flex_sdpa` attention), `hf_attention` (which attention a configuration
  gets on a device), `hf_kernels` (what a loaded model runs) and
  `reference_kernels` (run a model on Transformers' reference code), with
  their private helpers and `_TEXT_KEY_MAPPING`;
- the timing, profiling and capacity harness of the section's measurements:
  `hf_prefill`, `hf_decode_ms`, `hf_step_kernels`, `hf_capacity`,
  `hf_measure`, `hf_profile_steps`, `hf_compare_kernels`, the two table
  views `hf_memory_table` and `hf_time_table`, and `show_kernels`.

Every block is tagged `%%tab pytorch` and imports third-party modules
(`transformers`, `pandas`) inside its functions. The harness calls the
section's shown functions through `d2l` (`cache_bytes`, `bytes_read`,
`config_memory`).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_hf_models`

```{.python .input #hf-models-lib-routing}
%%tab pytorch
#@save
_HF_LINEAR_KERNELS = ('torch_chunk_gated_delta_rule',  # a prompt, by chunks
                      'torch_recurrent_gated_delta_rule',  # one token
                      'causal_conv1d_fn', 'causal_conv1d_update')

#@save
_HF_LINEAR_MODELS = ('qwen3_5', 'qwen3_5_moe', 'qwen3_next')  # Gated DeltaNet

#@save
def _hf_route(bound):
    """Transformers' kernel function `bound`, run by the fused kernel it
    bound at import for CUDA tensors in bf16 or fp16 and by its PyTorch
    reference code for the rest: the fused Triton kernels need a GPU, and
    they multiply fp32 inputs in TF32, which an fp32 check does not expect."""
    import functools
    import inspect
    reference = bound.__wrapped__
    fused = inspect.getclosurevars(bound).nonlocals.get('implementation')

    @functools.wraps(reference)
    def routed(*args, **kwargs):
        x = args[0] if args else next(iter(kwargs.values()))
        if (routed.fused is None or not x.is_cuda
                or x.dtype not in (torch.bfloat16, torch.float16)):
            return reference(*args, **kwargs)
        return bound(*args, **kwargs)
    routed.fused = None if fused is reference else fused  # None: reference
    return routed

#@save
def _hf_flex_sdpa_mask(*args, q_length, **kwargs):
    """The flex_sdpa mask: a block mask for a prompt, SDPA's mask (one row
    at most) for a single token."""
    from transformers import masking_utils
    mask = (masking_utils.flex_attention_mask if q_length > 1
            else masking_utils.sdpa_mask)
    return mask(*args, q_length=q_length, **kwargs)

#@save
def _hf_flex_sdpa(module, query, key, value, attention_mask, **kwargs):
    """Flex attention for a prompt, which materializes no n x n mask; SDPA
    for a single token, where flex's per-step block mask costs time."""
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations import flex_attention, sdpa_attention
    if not isinstance(attention_mask, BlockMask):
        return sdpa_attention.sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs)
    if query.shape[-1] > 128:  # flex's default blocks for wide heads need
        kwargs.setdefault('kernel_options', dict(  # 148 KiB of shared
            BLOCK_M=32, BLOCK_N=32, num_stages=1))  # memory; sm_89 has 99
    return flex_attention.flex_attention_forward(
        module, query, key, value, attention_mask, **kwargs)

#@save
def bind_hf_kernels():
    """Route the linear-attention functions of Transformers' Gated DeltaNet
    models by device and dtype (`_hf_route`), importing their modules first
    so that a later `from_pretrained` finds them routed; register the
    flex_sdpa attention."""
    import importlib
    import sys
    import warnings
    from transformers import AttentionInterface, AttentionMaskInterface
    AttentionInterface.register('flex_sdpa', _hf_flex_sdpa)
    AttentionMaskInterface.register('flex_sdpa', _hf_flex_sdpa_mask)
    # Flex attention compiles once per new shape; past torch's default of 8
    # recompilations it would fall back to code that materializes the n x n
    # scores, which is an error here rather than a silent slowdown
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, 64)
    warnings.filterwarnings('error',
                            'flex_attention called without torch.compile')
    for name in _HF_LINEAR_MODELS:
        try:
            with warnings.catch_warnings():  # fla's notices on a CPU host
                warnings.simplefilter('ignore')
                importlib.import_module(f'transformers.models.{name}.'
                                        f'modeling_{name}')
        except ImportError:  # a Transformers version without this model
            pass
    for name, module in list(sys.modules.items()):
        if name.startswith('transformers.models.') and module is not None:
            for fn in _HF_LINEAR_KERNELS:
                bound = getattr(module, fn, None)
                if bound is not None and not hasattr(bound, 'fused'):
                    setattr(module, fn, _hf_route(bound))

#@save
def hf_attention(config, device):
    """The attention that runs `config` on `device` without an n x n mask:
    flex_sdpa if it has sliding-window layers and the device is a GPU of
    compute capability 8.0 or more, SDPA otherwise."""
    layers = getattr(config.get_text_config(), 'layer_types', None) or []
    device = torch.device(device)
    return ('flex_sdpa' if 'sliding_attention' in layers
            and device.type == 'cuda'
            and torch.cuda.get_device_capability(device) >= (8, 0)
            else 'sdpa')

#@save
_TEXT_KEY_MAPPING = {  # checkpoint key renames for a multimodal file's text
    'gemma4_text': {r'^model\.language_model\.': 'model.'}}
```

```{.python .input #hf-models-lib-kernels}
%%tab pytorch
#@save
def hf_kernels(model):
    """The code a model's attention and Gated DeltaNet layers run on a GPU
    in the model's dtype (binding the latter by device and dtype first, as
    load_hf does)."""
    import sys
    bind_hf_kernels()
    module = sys.modules[type(model).__module__]
    kernels = {'attention': model.config._attn_implementation}
    if 'linear_attention' in (getattr(model.config, 'layer_types', None)
                              or []):
        for key, fn in (('delta rule', 'torch_chunk_gated_delta_rule'),
                        ('short convolution', 'causal_conv1d_fn')):
            fused = getattr(module, fn).fused
            kernels[key] = ('PyTorch' if fused is None or model.dtype
                            not in (torch.bfloat16, torch.float16)
                            else fused.__module__.split('.')[0])
    return kernels

#@save
class reference_kernels:
    """Run `model` on Transformers' reference code inside a `with` block:
    SDPA with explicit masks and the PyTorch Gated DeltaNet."""
    def __init__(self, model):
        import sys
        bind_hf_kernels()
        module = sys.modules[type(model).__module__]
        self.model, self.routed = model, [
            getattr(module, fn) for fn in _HF_LINEAR_KERNELS
            if hasattr(module, fn)]

    def __enter__(self):
        self.saved = (self.model.config._attn_implementation,
                      [f.fused for f in self.routed])
        self.model.set_attn_implementation('sdpa')
        for f in self.routed:
            f.fused = None
        return self.model

    def __exit__(self, *exc):
        attention, fused = self.saved
        for f, kernel in zip(self.routed, fused):
            f.fused = kernel
        self.model.set_attn_implementation(attention)
```

```{.python .input #hf-models-lib-measure}
%%tab pytorch
#@save
def _hf_allocated(stat='current'):
    """Bytes the live tensors requested (or their peak), without the
    allocator's rounding of each block."""
    return torch.cuda.memory_stats()[f'requested_bytes.all.{stat}']

#@save
def hf_prefill(model, ids, n, keep=1):
    """Prefill the first n tokens: the output, the allocated bytes before,
    and the growth and the peak above them."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = _hf_allocated()
    with torch.inference_mode():
        out = model(ids[:, :n], logits_to_keep=keep, use_cache=True)
    torch.cuda.synchronize()
    return (out, before, _hf_allocated() - before,
            _hf_allocated('peak') - before)

#@save
def hf_decode_ms(model, cache, token, steps=64, warmup=3, repeats=10):
    """Median, first and third quartile of milliseconds per greedy decode
    step; every repeat starts from a fresh copy of the prefilled cache.
    A profiler session leaves its launch hooks on for the rest of the
    process (about 3 us more per kernel launch), so time before profiling."""
    import copy

    def loop(c, token):
        for _ in range(steps):
            logits = model(token, past_key_values=c, logits_to_keep=1).logits
            token = logits[:, -1].argmax(-1, keepdim=True)  # on the device

    times = []
    with torch.inference_mode():
        for _ in range(warmup + repeats):
            c = copy.deepcopy(cache)  # outside the timer
            seconds = d2l.Benchmark(lambda: loop(c, token), 0, 1).time
            times.append(1e3 * seconds / steps)
    return np.percentile(times[warmup:], [50, 25, 75])

#@save
def hf_step_kernels(model, cache, token, steps=10):
    """Device milliseconds per kernel of one decode step, and the number of
    kernels the step launches."""
    import copy
    c = copy.deepcopy(cache)

    def step():
        model(token, past_key_values=c, logits_to_keep=1)

    with torch.inference_mode():
        step()  # warm-up
        torch.cuda.synchronize()  # the copy and warm-up finish unprofiled
        kernels = d2l.kernel_times(step, steps)
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                acc_events=True) as prof:
            step()
            torch.cuda.synchronize()
    launches = sum(e.device_type == torch.autograd.DeviceType.CUDA
                   for e in prof.events())
    assert launches > 0
    return kernels, launches

#@save
def _hf_fits(model, ids, batch, n, budget):
    """The allocator's overhead (reserved minus requested bytes, each at
    its peak) of a prefill of `batch` copies of the first n tokens with the
    allocator capped at `budget` bytes, weights included; None if it runs
    out of memory."""
    import gc
    x = ids.repeat(1, -(-n // ids.shape[1]))[:, :n].repeat(batch, 1)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_per_process_memory_fraction(
        budget / torch.cuda.get_device_properties(x.device).total_memory)
    try:
        with torch.inference_mode():
            model(x, logits_to_keep=1, use_cache=True)
        stats = torch.cuda.memory_stats()
        overhead = (stats['reserved_bytes.all.peak']
                    - stats['requested_bytes.all.peak'])
    except torch.OutOfMemoryError:
        overhead = None
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    del x
    gc.collect()
    torch.cuda.empty_cache()
    return overhead

#@save
def hf_capacity(model, config, ids, budget, n=4096):
    """The largest batch at n tokens (doubling, then bisection) and the
    largest power-of-two context at batch 1 that fit `budget` bytes,
    weights included: measured, and by `config_memory` for a prefill.
    Returns (batch, context) twice and the allocator's overhead in bytes
    at the measured batch."""
    t = config.get_text_config()
    overhead = {1: _hf_fits(model, ids, 1, n, budget)}
    assert overhead[1] is not None
    lo = 1
    while (fit := _hf_fits(model, ids, 2 * lo, n, budget)) is not None:
        lo, overhead[2 * lo] = 2 * lo, fit
    hi = 2 * lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (fit := _hf_fits(model, ids, mid, n, budget)) is not None:
            lo, overhead[mid] = mid, fit
        else:
            hi = mid
    context = n
    while (2 * context <= t.max_position_embeddings
           and _hf_fits(model, ids, 1, 2 * context, budget) is not None):
        context *= 2
    mem = d2l.config_memory(config, n, 1, window_cached=n)
    batch = (budget - mem['weights']) // (mem['total'] - mem['weights'])
    longest = max(2**k for k in range(8, 19)
                  if 2**k <= t.max_position_embeddings and d2l.config_memory(
                      config, 2**k, 1, window_cached=2**k)['total'] <= budget)
    return (lo, context), (int(batch), longest), overhead[lo]
```

```{.python .input #hf-models-lib-measure-run}
%%tab pytorch
#@save
def hf_measure(model, config, ids, beta, contexts=(256, 1024, 4096, 16384)):
    """Memory and decode time at each context, one row (dict) per context:
    the cache walker's bytes, the allocator's growth after the prefill and
    after one decode step, the peak, each with its value from
    `config_memory`, the decode milliseconds and the bytes a step reads
    (`bytes_read`) over the bandwidth `beta`. Prints the live cache's
    bytes by dtype at 1,024 tokens."""
    GiB = 2**30
    b, t = model.dtype.itemsize, config.get_text_config()
    rows = []
    with torch.inference_mode():  # cuBLAS allocates its workspace on the
        out = model(ids[:, :16], logits_to_keep=1, use_cache=True)
        model(out.logits[:, -1:].argmax(-1),  # first decode step
              past_key_values=out.past_key_values)
    del out
    for n in contexts:  # a prefill holds every position: window_cached=n
        pinned = d2l.config_memory(config, n, 1, b, window_cached=n)
        need = 1.1 * (pinned['total'] - pinned['weights'])
        torch.cuda.empty_cache()  # cached blocks count as free below
        if need > torch.cuda.mem_get_info()[0]:
            print(f'{n} tokens skipped: {need / GiB:.1f} GiB not free')
            continue
        try:  # a new length compiles flex attention or tunes fla's
            hf_prefill(model, ids, n)  # kernels, with their own buffers
            out, before, growth, peak = hf_prefill(model, ids, n)
        except torch.OutOfMemoryError:  # the activations estimate fell short
            print(f'{n} tokens skipped: out of memory in the prefill')
            continue  # the next context empties the allocator's cache
        cache, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
        logical, resident = d2l.cache_bytes(cache)
        mem = d2l.config_memory(config, n, 1, b)
        assert sum(logical.values()) == mem['kv_cache'] + mem[
            'recurrent_state']  # the walker counts what the formula counts
        if n == 1024:
            print('live cache at 1,024 tokens:', ', '.join(
                f'{v:,} B {str(k).removeprefix("torch.")}'
                for k, v in logical.items()))
            linear = [layer for layer in cache.layers
                      if hasattr(layer, 'recurrent_states')]
            if linear:  # a hybrid's state and convolution inputs
                print('recurrent state', str(
                    linear[0].recurrent_states[0].dtype).removeprefix(
                        'torch.'), '| convolution inputs', str(
                    linear[0].conv_states[0].dtype).removeprefix('torch.'))
        wall = hf_decode_ms(model, cache, token)
        logits_bytes = out.logits.nbytes
        del out
        with torch.inference_mode():  # sliding layers drop prompt keys here
            step = model(token, past_key_values=cache, logits_to_keep=1)
        after = d2l.config_memory(
            config, n + 1, 1, b,
            window_cached=getattr(t, 'sliding_window', None))
        R = d2l.bytes_read(config, n, b)
        rows.append(dict(
            model=model.name_or_path, n=n,
            walker_logical=sum(logical.values()),
            walker_resident=sum(resident.values()),
            formula=mem['kv_cache'] + mem['recurrent_state'], growth=growth,
            growth_derived=(pinned['kv_cache'] + pinned['recurrent_state']
                            + logits_bytes),
            growth_after_step=_hf_allocated() - before,
            growth_after_step_derived=(after['kv_cache'] + after[
                'recurrent_state'] + step.logits.nbytes),
            peak=peak, peak_formula=pinned['total'] - pinned['weights'],
            wall_ms_median=wall[0], wall_ms_q1=wall[1], wall_ms_q3=wall[2],
            bytes_read=R, r_over_beta_measured_ms=1e3 * R / beta))
        del cache, step
    return rows

#@save
def hf_profile_steps(model, ids, rows):
    """Add one decode step's device time (busy and matrix products) and
    launch count to each row, on a fresh prefill; return the step's
    kernels at each context, {n: {kernel: ms}}."""
    kernels = {}
    for row in rows:
        out = hf_prefill(model, ids, row['n'])[0]
        kernels[row['n']], row['launches'] = hf_step_kernels(
            model, out.past_key_values, out.logits[:, -1:].argmax(-1))
        row['busy_ms'] = sum(kernels[row['n']].values())
        row['matmul_ms'] = sum(ms for name, ms in kernels[row['n']].items()
                               if _MATMUL_KERNELS.search(name))
        del out
    return kernels

#@save
def hf_compare_kernels(model, config, ids, contexts, decode_at=4096,
                       positions=64):
    """A prefill on the fused kernels and inside reference_kernels, per
    context: peak memory above the weights against the formula's peak (GiB),
    seconds, milliseconds per decode step at `decode_at` tokens, and, on the
    reference row, how the last `positions` positions' logits compare with
    the fused run's: the share whose most likely next token agrees, and the
    largest difference. For scale, the reference at batch 2 against batch 1
    at the first context: rounding alone. A run that does not fit says so."""
    import contextlib
    import pandas as pd

    def compare(a, b):
        a, b = a.float(), b.float()
        return {'top-1 agree': round(float(
                    (a.argmax(-1) == b.argmax(-1)).float().mean()), 2),
                'max |diff|': round(float((a - b).abs().max()), 3)}

    rows = []
    for n in contexts:
        x = ids.repeat(1, -(-n // ids.shape[1]))[:, :n]
        formula = d2l.config_memory(config, n, 1, window_cached=n)
        logits = {}
        for name, kernels in (('fused', contextlib.nullcontext),
                              ('reference', reference_kernels)):
            row = dict(n=n, kernels=name, formula=round(
                (formula['total'] - formula['weights']) / 2**30, 3))
            out = cache = None
            with kernels(model):
                try:  # also compiles and tunes the kernels for n tokens
                    out, _, _, peak = hf_prefill(model, x, n)
                    cache = out.past_key_values
                    token = out.logits[:, -1:].argmax(-1)
                    out = None
                    with torch.inference_mode():
                        seconds = d2l.Benchmark(lambda: model(
                            x, logits_to_keep=1, use_cache=True), 1, 3).time
                        logits[name] = model(x, logits_to_keep=positions,
                                             use_cache=False).logits[0]
                    row |= dict(peak=round(peak / 2**30, 3),
                                prefill_s=round(seconds, 3))
                    if n == decode_at:
                        row['decode_ms'] = round(float(hf_decode_ms(
                            model, cache, token, repeats=5)[0]), 2)
                except torch.OutOfMemoryError:
                    row['peak'] = 'out of memory'
                out = cache = None
            rows.append(row)
        if len(logits) == 2:
            rows[-1] |= compare(logits['fused'], logits['reference'])
        if n == contexts[0] and 'reference' in logits:
            with reference_kernels(model), torch.inference_mode():
                twice = model(x.repeat(2, 1), logits_to_keep=positions,
                              use_cache=False).logits[0]
            rows.append(dict(n=n, kernels='ref., batch 2')
                        | compare(twice, logits['reference']))
            del twice
        del logits
    return pd.DataFrame(rows)
```

```{.python .input #hf-models-lib-show}
%%tab pytorch
#@save
_MATMUL_KERNELS = re.compile(  # matrix products; attention kernels are not
    '^(?!.*(fmha|flash)).*(gemm|gemv|xmma|cutlass|matmul|splitK)', re.I)

#@save
def hf_memory_table(rows):
    """The memory columns of hf_measure's rows: the walker's bytes and the
    formula's (MiB), the allocator's growth after the prefill and after one
    step, and the prefill peak, each over its derived value; storages held
    over logical bytes (a sliding layer's view pins the prompt's keys)."""
    import pandas as pd
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        'n': df.n, 'walker MiB': df.walker_logical / 2**20,
        'formula MiB': df.formula / 2**20,
        'growth': df.growth / df.growth_derived,
        'after step': df.growth_after_step / df.growth_after_step_derived,
        'peak': df.peak / df.peak_formula,
        'held': df.walker_resident / df.walker_logical}).round(2)

#@save
def hf_time_table(rows, beta_sheet):
    """The time columns of the profiled rows, in ms per decode step: wall
    time (median and quartiles), the device's busy and matrix-product time,
    kernel launches, and the bytes read over the datasheet bandwidth
    `beta_sheet` and over the measured one."""
    import pandas as pd
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        'n': df.n, 'wall': df.wall_ms_median, 'q1': df.wall_ms_q1,
        'q3': df.wall_ms_q3, 'busy': df.busy_ms, 'matmul': df.matmul_ms,
        'launches': df.launches,
        'R/beta sheet': 1e3 * df.bytes_read / beta_sheet,
        'R/beta meas.': df.r_over_beta_measured_ms}).round(2)

#@save
def _kernel_name(name):
    """A kernel's C++ name without return type, namespaces and arguments."""
    name = re.sub(r'^(void |std::enable_if<.*?>::type )|\(anonymous '
                  r'namespace\)::', '', name)
    return re.sub(r'\b(at|native|internal|cutlass)::|\(.*', '', name)[:52]

#@save
def show_kernels(kernels, rows=4):
    """The kernels of a decode step with the most device time: ms and share
    of the busy time."""
    import collections
    busy, named = sum(kernels.values()), collections.Counter()
    for name, ms in kernels.items():
        named[_kernel_name(name)] += ms
    for name, ms in named.most_common(rows):
        print(f'{ms:6.2f} ms {ms / busy:4.0%}  {name}')
```
