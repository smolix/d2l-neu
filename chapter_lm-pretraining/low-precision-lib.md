# Low-Precision Training: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is
never rendered or executed as a notebook: `make lib` only scans it for
`#@save` blocks, and `gen_notebooks.py` excludes it via `LIB_ONLY_FILES`.
It holds the library code of :numref:`sec_lm_low_precision`
that the section explains in a sentence but does not show
(docs/code-guide.md 7.1):

- 21.3's registry entries (`PRETRAIN_ARTIFACTS`, `PRETRAIN_JOBS`; the
  `DATA_HUB` entries follow the uploads);
- the tile layouts behind `fp8_quant`'s granularities, the operand
  statistics behind `Fp8Linear.stats()`, the GEMM and layout helpers of
  `Fp8Linear` and `Fp4Linear`, the Hadamard matrix;
- the opening's activation capture (`mlp_input`);
- the Inductor workaround for fp8 copies (`_lowp_compile_settings`);
- the GEMM sweep (`fp8_gemm_sweep`) and the six-arm driver (`lowp_arms`:
  model conversion, torchao's float8 recipes behind a guarded import,
  profiling, operand statistics, one file per run so that a regeneration
  resumes) with their plots.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_low_precision`

## Registries

The hosted results (d2l-data R2), one entry per upload.

```{.python .input #low-precision-lib-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-lowp-runs-v1'] = (
    d2l.LM_DATA_URL + 'lm-lowp-runs-v1.json.gz',
    '4954512fee110cd470d3888a497aa68732336892')
d2l.DATA_HUB['lm-fp8-gemm-sweep-v1'] = (
    d2l.LM_DATA_URL + 'lm-fp8-gemm-sweep-v1.json.gz',
    '2e30da36f8761317419d1608dc15e1a1b6ba8c5f')
```

```{.python .input #low-precision-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['gemm-sweep'] = (
    'lm-fp8-gemm-sweep-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lowp-runs'] = (
    'lm-lowp-runs-v1', 'json-table')
```

The jobs of `tools/lm_regen.py`: the page's two computing calls written a
second time (the load check compares them with the committed runs).

```{.python .input #low-precision-lib-jobs}
%%tab pytorch
#@save
d2l.PRETRAIN_JOBS['gemm-sweep'] = ('fp8_gemm_sweep', lambda: dict(
    m=16384, ns=(512, 768, 1024, 1536, 2048, 3072, 4096),
    shapes=[(16384, 768, 768), (16384, 768, 256), (16384, 768, 2048),
            (16384, 2048, 768)]))
d2l.PRETRAIN_JOBS['lowp-runs'] = ('lowp_arms', lambda: dict(
    cfg=d2l.r2_config('fp8'),
    arms={'bf16': ('bf16', (0, 1, 2, 3)),
          'tensorwise': ('torchao-tensorwise', (0, 1)),
          'rowwise': ('torchao-rowwise', (0,)),
          'delayed': ('fp8-delayed', (0,)),
          'nvfp4-mlp': ('nvfp4-mlp', (0,))}))
```

## Formats, Tiles and Operand Statistics

```{.python .input #low-precision-lib-tiles}
%%tab pytorch
#@save
_FP8_MANTISSA = {'e4m3': 3, 'e5m2': 2}

#@save
def _fp8_tiles(x, granularity):
    """x as rows of its scaling tiles (one scale per row) and the function
    that puts a tile-shaped tensor back into x's shape; a (rows, 1) tensor
    of scales comes back in a shape that broadcasts against x."""
    shape = x.shape
    if granularity == 'tensor':
        return x.reshape(1, -1), lambda t: (t.reshape(()) if t.shape[-1] == 1
                                            else t.reshape(shape))
    if granularity == 'row':
        return x.reshape(-1, shape[-1]), lambda t: t.reshape(
            *shape[:-1], t.shape[-1])
    if granularity not in ('block1x128', 'block128x128'):
        raise ValueError(f'unknown granularity {granularity!r}')
    X = x.reshape(-1, shape[-1])
    M, K = X.shape
    b = 128 if granularity == 'block128x128' else 1
    if K % 128 or M % b:
        raise ValueError(f'{granularity} needs shapes divisible by the tile;'
                         f' got {M} x {K}')
    tiles = X.reshape(M // b, b, K // 128, 128).transpose(1, 2).reshape(
        -1, b * 128)

    def untile(t):
        t = t.expand(-1, b * 128) if t.shape[-1] == 1 else t
        return t.reshape(M // b, K // 128, b, 128).transpose(1, 2).reshape(
            shape)
    return tiles, untile

#@save
def _fp8_granularity(granularity, role):
    """Fp8Linear's granularity per operand: 'block' means 1 x 128 tiles for
    activations and gradients and 128 x 128 blocks for weights (the
    DeepSeek-V3 recipe)."""
    if granularity == 'block':
        return 'block128x128' if role == 'w' else 'block1x128'
    return granularity

#@save
def _fp8_fractions(x, scale, fmt):
    """Saturated (|x / scale| above the format's max) and underflowing
    (non-zero, below half the smallest subnormal) shares of x."""
    dtype = {'e4m3': torch.float8_e4m3fn, 'e5m2': torch.float8_e5m2}[fmt]
    info = torch.finfo(dtype)
    sub = info.tiny * 2.0 ** -_FP8_MANTISSA[fmt]  # smallest subnormal
    z, nonzero = (x.float() / scale).abs(), x != 0
    clipped = z > info.max * (1 + 2 ** -20)  # not the amax's own rounding
    return (clipped.float().mean(),
            ((z < sub / 2) & nonzero).sum() / nonzero.sum().clamp(min=1))

#@save
def _fp8_record(x, q, scale, fmt, granularity, role):
    """One operand cast as Fp8Linear.stats() reads it: saturation,
    underflow and relative error under the scale used, and saturation and
    underflow under a current scale of the same tensor (the replay delayed
    scaling is read against). Device tensors, no host synchronization."""
    x = x.detach()
    sat, under = _fp8_fractions(x, scale, fmt)
    err = torch.linalg.vector_norm(q.float() * scale - x.float()) / (
        torch.linalg.vector_norm(x.float()).clamp(min=1e-30))
    _, current = d2l.fp8_quant(x, fmt, granularity)
    sat_c, under_c = _fp8_fractions(x, current, fmt)
    return role, torch.stack([sat, under, err, sat_c, under_c])

#@save
_FP8_STAT_COLUMNS = ('saturation', 'underflow', 'error',
                     'saturation_current', 'underflow_current')

#@save
def _fp8_stats_table(records):
    """The mean over records of each statistic per operand role (a
    DataFrame indexed by role: x, w, dy)."""
    rows = {}
    for role, v in records:
        rows.setdefault(role, []).append(v)
    roles = [r for r in ('x', 'w', 'dy') if r in rows]
    if not roles:
        return pd.DataFrame(columns=_FP8_STAT_COLUMNS)
    means = torch.stack([torch.stack(rows[r]).mean(0) for r in roles])
    return pd.DataFrame(means.cpu().numpy(), index=roles,
                        columns=_FP8_STAT_COLUMNS)
```

`Fp8Linear` and `Fp4Linear` multiply through these helpers: a simulated
product dequantizes and multiplies in fp32 (autocast off); a native one
calls `torch._scaled_mm`, which wants the first operand row-major and the
second column-major, tensor scales, and dimensions divisible by 16.

```{.python .input #low-precision-lib-gemm}
%%tab pytorch
#@save
def has_fp8_cores(device):
    """Whether device has fp8 tensor cores (CUDA, compute capability 8.9:
    Ada, or newer)."""
    device = torch.device(device)
    return device.type == 'cuda' and (
        torch.cuda.get_device_capability(device) >= (8, 9))

#@save
def device_name(device):
    """A short name of device: 'RTX 4090', 'A100-SXM4-80GB', or 'CPU'."""
    device = torch.device(device)
    if device.type != 'cuda':
        return device.type.upper()
    name = torch.cuda.get_device_name(device)
    return re.sub(r'^NVIDIA (GeForce )?', '', name)

#@save
def _out_dtype(X):
    """The dtype nn.Linear would return for X: bf16 under autocast."""
    if torch.is_autocast_enabled(X.device.type):
        return torch.get_autocast_dtype(X.device.type)
    return X.dtype

#@save
def _f32_mm(A, B, out_dtype):
    """A @ B in fp32 (autocast off), returned as out_dtype."""
    with torch.autocast(A.device.type, enabled=False):
        return (A.float() @ B.float()).to(out_dtype)

#@save
def _fp8_mm(a, b, native, out_dtype, ta=False, tb=False):
    """op(A) @ op(B) for fp8 operands a = (q, scale) and b (op transposes
    when ta, tb): simulated, or native with tensor scales."""
    if not native:
        A, B = a[0].float() * a[1], b[0].float() * b[1]
        return _f32_mm(A.t() if ta else A, B.t() if tb else B, out_dtype)
    if a[1].numel() != 1 or b[1].numel() != 1:
        raise ValueError('native fp8 GEMMs take tensor scales only')
    A = (a[0].t() if ta else a[0]).contiguous()  # row-major
    B = b[0].t() if tb else b[0]
    if B.stride(0) != 1:
        B = B.t().contiguous().t()  # column-major
    out = torch.bfloat16 if out_dtype == torch.float32 else out_dtype
    return torch._scaled_mm(A, B, scale_a=a[1].reshape(()).float(),
                            scale_b=b[1].reshape(()).float(),
                            out_dtype=out).to(out_dtype)

#@save
def _hadamard(n, generator=None):
    """The n x n Hadamard matrix (Sylvester's construction, n a power of
    two) scaled to be orthogonal; with a generator, its columns flip sign
    at random (the random Hadamard transform)."""
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)])
    if H.shape[0] != n:
        raise ValueError(f'n = {n} is not a power of two')
    H = H / math.sqrt(n)
    if generator is not None:
        H = H * (torch.randint(0, 2, (n,), generator=generator) * 2 - 1)
    return H

#@save
def _rotate_rows(X, H):
    """H applied to each group of len(H) consecutive rows of X (in fp32):
    the rotation over tokens of Fp4Linear's weight gradient."""
    n = H.shape[0]
    if X.shape[0] % n:
        raise ValueError(f'{X.shape[0]} rows do not split into groups of {n}')
    Xg = X.float().reshape(-1, n, X.shape[-1])
    return torch.einsum('ij,gjc->gic', H.to(X.device), Xg).reshape(X.shape)

#@save
def _fp4_weight(W, block, block2d, dim=-1):
    """W fake-quantized to NVFP4: in block x block tiles (one scale each,
    so the forward and the input-gradient GEMM read the same quantized W)
    or in 1 x block blocks along dim."""
    if not block2d:
        return d2l.fake_quant_fp4(W, block, dim=dim)
    N, K = W.shape
    if N % block or K % block:
        raise ValueError(f'{N} x {K} does not split into {block}-blocks')
    tiles = W.reshape(N // block, block, K // block, block).transpose(1, 2)
    q = d2l.fake_quant_fp4(tiles.reshape(-1, block * block), block * block)
    return q.reshape(N // block, K // block, block, block).transpose(
        1, 2).reshape(N, K)
```

## The Opening's Activations

```{.python .input #low-precision-lib-capture}
%%tab pytorch
#@save
class _Captured(Exception):
    pass

#@save
def mlp_input(model, block, n_seqs=4, T=1024):
    """The input of block `block`'s MLP (after its RMSNorm) on the first
    n_seqs validation windows of T tokens, as an fp32 (n_seqs * T, d)
    matrix; the forward pass stops there."""
    with d2l._quiet():
        shards = d2l.fwedu_shards('val')
    X = torch.from_numpy(np.stack([shards.read(i * T, T)
                                   for i in range(n_seqs)]).astype(np.int64))
    dev = next(model.parameters()).device
    ctx = d2l.PackCtx(*(t.to(dev) for t in d2l.doc_ids_from_eos(
        X.int(), shards.eos_id)))
    seen = {}

    def grab(module, args):
        seen['x'] = args[0].detach()
        raise _Captured
    hook = model.blks[block].ffn.register_forward_pre_hook(grab)
    try:
        with torch.inference_mode():
            model(X.to(dev), ctx, logits_to_keep=1)
    except _Captured:
        pass
    finally:
        hook.remove()
    return seen['x'].float().flatten(0, -2).cpu()
```

## A Compiler Workaround for fp8 Copies

With torch 2.11.0 and Triton 3.6.0 on an RTX 4090, a Triton kernel that
casts a tile to fp8 and stores it in two layouts (the row-major copy and
the transposed copy a backward GEMM reads) writes about a quarter of the
transposed copy wrong: torch.compile fuses torchao's casts into exactly
such kernels, and the first compiled step of the tensorwise recipe has
millions of non-finite weight gradients (eager torchao is fine; a plain
Triton kernel with the two stores reproduces it, int8 and fp16 do not).
Inside the arms and the sweep, an Inductor pass therefore moves every fp8
copy (`aten.clone` of an fp8 tensor) into an opaque operator that runs
PyTorch's own copy kernel. Before any run, `_fp8_compile_check` compares
a compiled torchao layer pair with the eager one for every torchao recipe
of the arms, and `_train_gemms_check` the sweep's compiled casts and GEMMs
with the eager ones. The copies cost time of their own; the arms' profile
reports it as a row.

```{.python .input #low-precision-lib-workaround}
%%tab pytorch
#@save
def _fp8_copy_op():
    """torch.ops.d2l.fp8_copy, registered on first use: x.contiguous() as
    an operator Inductor cannot fuse."""
    if not hasattr(torch.ops.d2l, 'fp8_copy'):
        @torch.library.custom_op('d2l::fp8_copy', mutates_args=())
        def fp8_copy(x: torch.Tensor) -> torch.Tensor:
            return x.contiguous()

        @fp8_copy.register_fake
        def _(x):
            return torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return torch.ops.d2l.fp8_copy.default

#@save
def _fp8_clone_pass():
    """An Inductor post-grad pass: every aten.clone of an fp8 tensor becomes
    torch.ops.d2l.fp8_copy (no two-layout fp8 stores in one kernel)."""
    from torch._inductor.custom_graph_pass import CustomGraphPass
    op, fp8 = _fp8_copy_op(), (torch.float8_e4m3fn, torch.float8_e5m2)

    class Fp8ClonePass(CustomGraphPass):
        def __call__(self, graph):
            for node in list(graph.nodes):
                v = node.meta.get('val')
                if (node.op == 'call_function'
                        and str(node.target) == 'aten.clone.default'
                        and isinstance(v, torch.Tensor) and v.dtype in fp8):
                    with graph.inserting_before(node):
                        new = graph.call_function(op, (node.args[0],))
                    new.meta = dict(node.meta)
                    node.replace_all_uses_with(new)
                    graph.erase_node(node)

        def uuid(self):
            return 'd2l-fp8-clone-pass-v1'
    return Fp8ClonePass()

#@save
def _lowp_compile_settings():
    """The Inductor settings of the arms and the sweep (the fp8 copy pass)."""
    import torch._inductor.config as inductor_config
    return inductor_config.patch(
        {'post_grad_custom_post_pass': _fp8_clone_pass()})

#@save
def _fp8_compile_check(recipes=('tensorwise',), tol=0.1):
    """Gradients of a compiled torchao MLP (1,024 -> 4,096 -> 1,024 on 8,192
    tokens) against the eager ones, per float8 recipe: raises unless every
    gradient is finite and within tol (relative Frobenius error); returns
    {recipe: errors}."""
    from torchao.float8 import (Float8LinearConfig,
                                convert_to_float8_training)
    dev = torch.device('cuda')
    torch.manual_seed(0)
    X = torch.randn(8192, 1024, device=dev)

    def grads(recipe, compiled):
        torch.manual_seed(0)
        mlp = nn.Sequential(nn.Linear(1024, 4096, bias=False), nn.GELU(),
                            nn.Linear(4096, 1024, bias=False)).to(dev)
        convert_to_float8_training(
            mlp, config=Float8LinearConfig.from_recipe_name(recipe))
        net = torch.compile(mlp) if compiled else mlp
        with torch.autocast('cuda', torch.bfloat16):
            net(X).float().pow(2).mean().backward()
        return [p.grad.float() for p in mlp.parameters()]
    out = {}
    for recipe in recipes:
        torch._dynamo.reset()
        with _lowp_compile_settings(), d2l._quiet():
            got = grads(recipe, True)
        ref = grads(recipe, False)
        torch._dynamo.reset()
        out[recipe] = _check_close(got, ref, tol, f'compiled {recipe}')
    return out

#@save
def _check_close(got, ref, tol, what):
    """Relative Frobenius errors of got against ref; raises unless each is
    finite and below tol."""
    errs = [float(torch.linalg.norm(g.float() - r.float()) /
                  torch.linalg.norm(r.float())) for g, r in zip(got, ref)]
    if not all(math.isfinite(e) and e < tol for e in errs):
        raise RuntimeError(f'{what} differs from eager: {errs}')
    return errs
```

## The GEMM Sweep

Each shape is timed as one linear layer's training step: the forward
product and its two backward products, in bf16, in fp8 with the casts a
training step pays (current tensor scales, the operands cast once plus the
transposed copies the backward products read; compiled, so the casts fuse
as torchao's do), and in fp8 with operands cast beforehand (the GEMMs
alone). The median of three repeats of `d2l.Benchmark` (10 warm-up calls,
50 timed) on a card no other job shares.

```{.python .input #low-precision-lib-sweep}
%%tab pytorch
#@save
GemmSweep = collections.namedtuple('GemmSweep', 'table shapes label')

#@save
_SWEEP_COLUMNS = ('bf16', 'fp8 with casts', 'fp8 GEMM only')

#@save
def _train_gemms_fp8(X, W, dY):
    """One linear layer's three products in fp8, casts included."""
    x, w = d2l.fp8_quant(X), d2l.fp8_quant(W)
    dy = d2l.fp8_quant(dY, 'e5m2')
    return (_fp8_mm(x, w, True, torch.bfloat16, tb=True),
            _fp8_mm(dy, w, True, torch.bfloat16),
            _fp8_mm(dy, x, True, torch.bfloat16, ta=True))

#@save
def _train_gemms_check(compiled, m=4096, n=1024, tol=1e-2):
    """The compiled _train_gemms_fp8 against the eager one on an m x n x n
    layer: raises unless all three products agree within tol."""
    dev = torch.device('cuda')
    g = torch.Generator(dev).manual_seed(0)
    X, W, dY = (torch.randn(r, n, device=dev, generator=g,
                            dtype=torch.bfloat16) for r in (m, n, m))
    return _check_close(compiled(X, W, dY), _train_gemms_fp8(X, W, dY), tol,
                        'compiled _train_gemms_fp8')

#@save
def _gemm_times(m, k, n, repeats, compiled):
    """Seconds per training step of one m x k -> n linear layer: bf16, fp8
    with casts, fp8 GEMMs only (medians over repeats)."""
    dev = torch.device('cuda')
    X = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    W = torch.randn(n, k, device=dev, dtype=torch.bfloat16) / math.sqrt(k)
    dY = torch.randn(m, n, device=dev, dtype=torch.bfloat16)
    x, w = d2l.fp8_quant(X), d2l.fp8_quant(W)
    dy = d2l.fp8_quant(dY, 'e5m2')
    # the layouts torch._scaled_mm reads, made once for the GEMM-only time
    xr, xc = x[0].contiguous(), x[0].t().contiguous().t()
    wt, wc = w[0].t(), w[0].t().contiguous().t()
    dyr, dyt = dy[0].contiguous(), dy[0].t().contiguous()
    mm, bf = torch._scaled_mm, torch.bfloat16

    def gemm_only():
        mm(xr, wt, x[1], w[1], out_dtype=bf)
        mm(dyr, wc, dy[1], w[1], out_dtype=bf)
        mm(dyt, xc, dy[1], x[1], out_dtype=bf)
    fns = (lambda: (X @ W.t(), dY @ W, dY.t() @ X),
           lambda: compiled(X, W, dY), gemm_only)
    times = [[d2l.Benchmark(f, warmup=10, repeats=50).time for f in fns]
             for _ in range(repeats)]
    return np.median(np.array(times), axis=0)

#@save
def fp8_gemm_sweep(m, ns, shapes, repeats=3):
    """TFLOP/s of one linear layer's training step (Y = X W^T, dX = dY W,
    dW = dY^T X) in bf16, fp8 with casts and fp8 GEMMs only, at m x n x n
    for each n in ns (table), and milliseconds at each (m, k, n) of shapes
    (shapes). Needs fp8 tensor cores (Ada or newer): measured only in a
    regeneration, on a whole card in a process of its own; else the
    committed result."""
    config = dict(m=m, ns=list(ns), shapes=[list(s) for s in shapes],
                  repeats=repeats)

    def compute():
        if not has_fp8_cores('cuda' if torch.cuda.is_available() else 'cpu'):
            raise RuntimeError('the GEMM sweep needs an Ada or newer GPU')
        tflop = lambda m, k, n: 3 * 2 * m * k * n / 1e12
        with _lowp_compile_settings(), d2l._quiet():
            compiled = torch.compile(_train_gemms_fp8)
            _train_gemms_check(compiled)
            rows = [[n, *(tflop(m, n, n) / _gemm_times(m, n, n, repeats,
                                                         compiled))]
                    for n in ns]
            srows = [[*s, *(1e3 * _gemm_times(*s, repeats, compiled))]
                     for s in shapes]
        gpu = torch.cuda.get_device_name()
        return GemmSweep(
            pd.DataFrame(rows, columns=['n', *_SWEEP_COLUMNS]),
            pd.DataFrame(srows, columns=['m', 'k', 'n', *_SWEEP_COLUMNS]),
            label='' if gpu == 'NVIDIA GeForce RTX 4090' else gpu)
    result = d2l._pretrain_result('gemm-sweep', compute, config)
    return d2l._show_label(d2l._as_tuple(GemmSweep, result, {
        'table': ['n', *_SWEEP_COLUMNS],
        'shapes': ['m', 'k', 'n', *_SWEEP_COLUMNS]}))

#@save
def plot_gemm_sweep(sweep, peaks=(165, 330)):
    """TFLOP/s against n of the sweep's three series, with the datasheet
    peaks (bf16, fp8) as dotted lines."""
    t = sweep.table
    n = t['n'].tolist()
    d2l.plot(n, [t[c].tolist() for c in _SWEEP_COLUMNS] +
             [[p] * len(n) for p in peaks], 'n (16,384 x n x n, training)',
             'TFLOP/s', xscale='log', fmts=('-o', '-s', '-^', 'k:', 'k:'),
             ylim=(0, 1.1 * max(peaks)), figsize=(4.5, 3.3))
    ax = d2l.plt.gca()
    ax.legend([*_SWEEP_COLUMNS, 'datasheet peaks'], loc='lower center',
              bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    d2l.set_log_ticks(ax.xaxis)
```

## The Six Arms

`lowp_arms` trains the `fp8` preset once per arm and seed. Each run's
result goes to a file of its own under the campaign folder, checked against
its configuration on reuse, so an interrupted regeneration resumes with the
next run. The torchao arms convert every block linear whose dimensions are
multiples of 16 (the tied head is not an `nn.Linear`); the from-scratch arms
swap the block linears for `Fp8Linear`, or the MLP's for `Fp4Linear`, which
run eagerly inside the compiled model (they keep amax histories and record
statistics, which a compiled graph would not), so their speed is not
claimed.

```{.python .input #low-precision-lib-arms}
%%tab pytorch
#@save
LowpRuns = collections.namedtuple('LowpRuns',
                                  'table losses stats profile label')

#@save
_LOWP_COLUMNS = {
    'table': ['arm', 'n', 'native', 'val loss', 'tok/s', 'IQR', '÷bf16',
              'GiB'],
    'losses': ['arm', 'seed', 'step', 'tokens', 'loss', 'val_loss'],
    'stats': ['step', 'role', *_FP8_STAT_COLUMNS],
    'profile': ['arm', 'kernels', 'ms', 'share']}

#@save
_PROFILE_COLUMNS = ('kernels', 'ms', 'share')

#@save
_BLOCK_LINEARS = ('attention.W_q', 'attention.W_k', 'attention.W_v',
                  'attention.W_o', 'ffn.W_g', 'ffn.W_1', 'ffn.W_2')

#@save
def _swap_linears(model, make, names=_BLOCK_LINEARS):
    """Replace the named nn.Linear of every block by make(in, out), keeping
    its weights; the new modules run eagerly inside a compiled model."""
    for blk in model.blks:
        for name in names:
            parent, attr = name.split('.')
            old = getattr(getattr(blk, parent), attr)
            new = make(old.in_features, old.out_features).to(
                old.weight.device, old.weight.dtype)
            new.weight.data.copy_(old.weight.data)
            new.forward = torch.compiler.disable(new.forward)
            setattr(getattr(blk, parent), attr, new)

#@save
def _rowwise_ok(dev):
    """Whether torch._scaled_mm runs one rowwise-scaled product on dev."""
    try:
        a = torch.ones(32, 32, device=dev).to(torch.float8_e4m3fn)
        torch._scaled_mm(a, a.t(), torch.ones(32, 1, device=dev),
                         torch.ones(1, 32, device=dev),
                         out_dtype=torch.bfloat16)
        return True
    except RuntimeError:
        return False

#@save
def _torchao_float8(model, recipe):
    """torchao's float8 training (recipe 'tensorwise' or 'rowwise') on the
    block linears whose dimensions are multiples of 16; False (and nothing
    converted) when this card cannot run the recipe natively."""
    try:
        from torchao.float8 import (Float8LinearConfig,
                                    convert_to_float8_training)
    except ImportError:
        return False
    dev = next(model.parameters()).device
    if not has_fp8_cores(dev):
        return False
    if recipe == 'rowwise' and not _rowwise_ok(dev):
        return False
    keep = lambda mod, fqn: ('blks.' in fqn and mod.in_features % 16 == 0
                             and mod.out_features % 16 == 0)
    with d2l._quiet():
        convert_to_float8_training(
            model, config=Float8LinearConfig.from_recipe_name(recipe),
            module_filter_fn=keep)
    return True

#@save
def _lowp_convert(model, recipe):
    """Convert model for a recipe; returns 'yes' (native fp8, compiled),
    'eager' (native fp8 GEMMs outside the compiled graph), 'simulated', or
    '—' (bf16)."""
    dev = next(model.parameters()).device
    if recipe == 'bf16':
        return '—'
    if recipe in ('torchao-tensorwise', 'torchao-rowwise'):
        gran = recipe.split('-')[1]
        if _torchao_float8(model, gran):
            return 'yes'
        if gran == 'tensorwise':
            raise RuntimeError('torchao tensorwise needs an Ada or newer GPU')
        _swap_linears(model, lambda i, o: d2l.Fp8Linear(
            i, o, granularity='row'))
        return 'simulated'
    if recipe == 'fp8-delayed':  # native GEMMs if the card has them, run
        native = has_fp8_cores(dev)  # eagerly: its speed is not claimed
        _swap_linears(model, lambda i, o: d2l.Fp8Linear(
            i, o, scale_mode='delayed', native=native))
        return 'eager' if native else 'simulated'
    if recipe == 'nvfp4-mlp':
        _swap_linears(model, d2l.Fp4Linear, ('ffn.W_g', 'ffn.W_1',
                                             'ffn.W_2'))
        return 'simulated'
    raise ValueError(f'unknown recipe {recipe!r}')

#@save
def _stats_hook(model, rows, every=10):
    """A pretrain hook: Fp8Linear operand statistics on every `every`-th
    step (the mean over the layers per operand role)."""
    mods = [m for m in model.modules() if isinstance(m, d2l.Fp8Linear)]

    def hook(step, net, opt, batch, row):
        if mods and mods[0].track:
            table = _fp8_stats_table([r for m in mods for r in m.records])
            for role, v in table.iterrows():
                rows.append(dict(step=row['step'], role=role, **v))
        for m in mods:
            m.records.clear()
            m.track = (step + 2) % every == 0  # the next step is tracked
    return hook

#@save
def _profile_hook(cfg, out, start=200, stop=210):
    """A pretrain hook that profiles the steps after `start` up to `stop`
    (1-based) and writes out['table'] (_profile_table)."""
    from torch.profiler import ProfilerActivity, profile
    state = {}

    def hook(step, net, opt, batch, row):
        if row['step'] == start:
            torch.cuda.synchronize()
            state['prof'] = profile(activities=[ProfilerActivity.CPU,
                                                ProfilerActivity.CUDA],
                                    record_shapes=True)
            state['prof'].__enter__()
            state['t0'] = time.perf_counter()
        elif row['step'] == stop and 'prof' in state:
            torch.cuda.synchronize()
            wall = (time.perf_counter() - state['t0']) / (stop - start)
            state['prof'].__exit__(None, None, None)
            out['table'] = _profile_table(state.pop('prof'), cfg, wall,
                                          stop - start)
    return hook

#@save
def _profile_table(prof, cfg, wall, steps):
    """Milliseconds per step and share of the step: the GEMMs of the block
    linears, the head's GEMMs (a dimension equal to the vocabulary), the
    fp8 copies of the compiler workaround (torch.ops.d2l.fp8_copy), the
    other kernels, and the rest of the wall-clock step (host, launches)."""
    parts = {'block GEMMs': 0.0, 'head GEMMs': 0.0, 'fp8 copies': 0.0}
    total = 0.0  # the kernels' time: device events, not annotations
    for e in prof.key_averages(group_by_input_shape=True):
        t = e.self_device_time_total / 1e3 / steps  # ms per step
        if e.device_type == torch.autograd.DeviceType.CUDA:
            if not (getattr(e, 'is_user_annotation', False)
                    or e.key.startswith('## ')):
                total += t
        elif e.key in ('aten::mm', 'aten::addmm', 'aten::bmm',
                       'aten::_scaled_mm'):  # the kernels it launched
            shapes = [s for s in e.input_shapes if s]
            head = any(cfg.arch.vocab in s for s in shapes)
            parts['head GEMMs' if head else 'block GEMMs'] += t
        elif e.key == 'd2l::fp8_copy':  # its kernels run in child ops
            parts['fp8 copies'] += e.device_time_total / 1e3 / steps
    ms = 1e3 * wall
    rows = [*parts.items(), ('other kernels', total - sum(parts.values())),
            ('host and launches', max(ms - total, 0.0))]
    return pd.DataFrame([(k, v, v / ms) for k, v in rows],
                        columns=_PROFILE_COLUMNS)

#@save
def _lowp_run(cfg, recipe, seed, profile=False, stats=False):
    """One run of lowp_arms on the GPU: the log, the peak memory, whether
    it ran natively, and the statistics or profile it was asked for. The
    validation runs under torch.no_grad, not inference mode: torchao's
    float8 layers refuse inference tensors (torchao 0.17)."""
    dev = torch.device('cuda')
    c = dataclasses.replace(cfg, seed=seed, data_seed=seed, probe_every=0,
                            stats_every=0)  # nothing here reads the probes
    torch._dynamo.reset()
    torch.manual_seed(seed)
    model = d2l.PackedGPT.from_arch(c.arch).to(dev)
    native = _lowp_convert(model, recipe)
    stream = d2l.PackedStream(d2l._train_shards(c), c.arch.T,
                              c.global_batch, seed=c.data_seed)
    rows, prof, hooks = [], {}, []
    if stats:
        hooks.append(_stats_hook(model, rows))
    if profile:
        hooks.append(_profile_hook(c, prof))
    torch.cuda.reset_peak_memory_stats(dev)
    with _lowp_compile_settings(), d2l._quiet():
        log = d2l.pretrain(model, stream, c, hooks=hooks,
                           val_fn=d2l._val_fn(c, inference=False))
    if not np.isfinite(log['loss']).all():
        raise RuntimeError(f'{recipe} seed {seed}: non-finite loss')
    return dict(log=log, native=native, stats=pd.DataFrame(rows),
                gib=torch.cuda.max_memory_allocated(dev) / 2 ** 30,
                profile=prof.get('table'))

#@save
def _lowp_run_cached(cfg, name, recipe, seed, profile, stats):
    """_lowp_run, or the file an earlier attempt of the same run wrote."""
    folder = os.path.join(d2l._job_root(), 'lowp')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'{name}-s{seed}.json.gz')
    config = dict(cfg=d2l._cfg_json(cfg), recipe=recipe, seed=seed,
                  profile=profile, stats=stats)
    if os.path.exists(path):
        r = d2l._read_result(path, 'json-table')
        d2l._check_config(r, config, path)
        return r._asdict()
    r = _lowp_run(cfg, recipe, seed, profile, stats)
    parts = dict(log=r['log'], stats=r['stats'], profile=(
        r['profile'] if r['profile'] is not None else pd.DataFrame(
            columns=_PROFILE_COLUMNS)),
                 native=r['native'], gib=r['gib'])
    d2l._write_result(path, 'json-table', parts,
                      dict(config=d2l._jsonable(config), label=''))
    return parts

#@save
def _lowp_merge(runs, arms):
    """LowpRuns from {(arm, seed): run}; the first arm is the baseline.
    Speeds are left out (NaN) for arms whose fp8 GEMMs do not run natively
    inside the compiled step (native 'eager' or 'simulated')."""
    losses, table, stats, profiles = [], [], None, []
    for name, (recipe, seeds) in arms.items():
        vals, rates, iqrs, gib = [], [], [], 0.0
        for seed in seeds:
            r = runs[(name, seed)]
            log = r['log']
            window = log.loc[log['step'].between(200, 400), 'tok_s']
            rates.append(window.median())
            iqrs.append(window.quantile(0.75) - window.quantile(0.25))
            gib = max(gib, r['gib'])
            part = log.reindex(columns=['step', 'tokens', 'loss',
                                        'val_loss'])
            part.insert(0, 'seed', seed)
            part.insert(0, 'arm', name)
            losses.append(part)
            vals.append(log['val_loss'].dropna().iloc[-1])
            if len(r['stats']):
                stats = r['stats']
            if len(r['profile']):
                profiles.append(r['profile'].assign(arm=name))
        table.append([name, len(seeds), r['native'], float(np.mean(vals)),
                      float(np.mean(rates)), float(np.mean(iqrs)), 0.0,
                      gib])
    table = pd.DataFrame(table, columns=_LOWP_COLUMNS['table'])
    table['÷bf16'] = table['tok/s'] / table['tok/s'].iloc[0]
    claimed = table['native'].isin(['yes', '—'])
    table.loc[~claimed, ['tok/s', 'IQR', '÷bf16']] = np.nan
    empty = lambda k: pd.DataFrame(columns=_LOWP_COLUMNS[k])
    profile = (pd.concat(profiles, ignore_index=True)[_LOWP_COLUMNS[
        'profile']] if profiles else empty('profile'))
    return LowpRuns(table, pd.concat(losses, ignore_index=True),
                    stats if stats is not None else empty('stats'),
                    profile, label='')

#@save
def lowp_arms(cfg, arms):
    """Train cfg once per arm and seed; arms = {name: (recipe, seeds)} with
    recipes 'bf16', 'torchao-tensorwise', 'torchao-rowwise' (native if the
    card can, else the simulated rowwise Fp8Linear), 'fp8-delayed' (the
    from-scratch Fp8Linear with delayed scaling, operand statistics every
    10 steps) and 'nvfp4-mlp' (Fp4Linear on the MLP). A run's model and
    data seed are its seed. The first arm is the baseline; every arm's
    first seed is profiled over steps 200-210. Returns LowpRuns: table (per
    arm: runs, native, final validation loss (seed mean), tokens/s (median
    over steps 200-400) and its IQR, ratio to the baseline, peak GiB;
    speeds only where the fp8 GEMMs run natively in the compiled step),
    losses, stats, profile (per arm). Computed only in a regeneration (one
    whole card, about an hour); else the committed result."""
    config = dict(cfg=d2l._cfg_json(cfg),
                  arms={k: [r, list(s)] for k, (r, s) in arms.items()})

    def compute():
        if not has_fp8_cores('cuda' if torch.cuda.is_available() else 'cpu'):
            raise RuntimeError('the arms need an Ada or newer GPU')
        _fp8_compile_check(_torchao_recipes(arms))
        runs = {(name, s): _lowp_run_cached(
            cfg, name, recipe, s, profile=(s == seeds[0]),
            stats=(recipe == 'fp8-delayed'))
            for name, (recipe, seeds) in arms.items() for s in seeds}
        return _lowp_merge(runs, arms)
    result = d2l._pretrain_result('lowp-runs', compute, config)
    return d2l._show_label(d2l._as_tuple(LowpRuns, result, _LOWP_COLUMNS))

#@save
def _torchao_recipes(arms):
    """The torchao float8 recipes among arms that this card runs natively
    (the ones the compile check covers)."""
    recipes = {r.split('-')[1] for r, _ in arms.values()
               if r.startswith('torchao-')}
    if 'rowwise' in recipes and not _rowwise_ok(torch.device('cuda')):
        recipes.remove('rowwise')
    return tuple(sorted(recipes))

#@save
def plot_lowp_runs(runs, bound):
    """Left: each arm's validation loss minus the baseline's seed mean
    against tokens (seed means), with the band of the baseline's seeds.
    Right: tokens/s relative to the baseline for the arms whose speed is
    measured, with the Amdahl bound."""
    fig, (ax1, ax2) = d2l.plt.subplots(1, 2, figsize=(7.5, 3))
    v = runs.losses.dropna(subset=['val_loss'])
    base = runs.table['arm'].iloc[0]
    seeds = v[v['arm'] == base].pivot_table('val_loss', 'tokens', 'seed')
    mean = seeds.mean(axis=1)
    x = mean.index / 1e6
    ax1.fill_between(x, seeds.min(axis=1) - mean, seeds.max(axis=1) - mean,
                     color='C0', alpha=0.2, label=f'{base} seeds')
    for i, arm in enumerate(runs.table['arm']):
        a = v[v['arm'] == arm].groupby('tokens')['val_loss'].mean()
        ax1.plot(a.index / 1e6, a.values - mean.reindex(a.index).values,
                 '-o', ms=3, color=f'C{i}', label=arm)
    ax1.set_xlabel('tokens (millions)')
    ax1.set_ylabel(f'validation loss − {base} mean')
    ax1.legend(fontsize=8)
    t = runs.table.dropna(subset=['÷bf16'])
    ax2.bar(t['arm'], t['÷bf16'], color='C0')
    ax2.set_ylim(0, 1.15 * max(bound, t['÷bf16'].max()))
    ax2.axhline(bound, color='black', ls='--', lw=1)
    ax2.text(len(t) - 0.5, bound, 'Amdahl bound', ha='right', va='bottom',
             fontsize=8)
    ax2.set_ylabel('tokens/s ÷ bf16')
    ax2.tick_params(axis='x', labelsize=8)
    fig.tight_layout()

#@save
def plot_lowp_stats(stats, marks=()):
    """Per operand role, against step: the saturated share under delayed
    scales (left; current scales never saturate) and the underflowing share
    under delayed (solid) and current (dashed) scales of the same tensors
    (right, log scale), both in parts per million, with vertical lines at
    marks (warmup end, decay start)."""
    fig, (ax1, ax2) = d2l.plt.subplots(1, 2, figsize=(7.5, 2.8))
    for i, (role, s) in enumerate(stats.groupby('role', sort=False)):
        s = s.groupby('step')[list(_FP8_STAT_COLUMNS)].mean()
        c = f'C{i}'
        ax1.plot(s.index, 1e6 * s['saturation'], color=c, label=role)
        ax2.plot(s.index, 1e6 * s['underflow'], color=c, label=role)
        ax2.plot(s.index, 1e6 * s['underflow_current'], color=c, ls='--')
    ax2.set_yscale('log', nonpositive='mask')
    ax2.legend([d2l.plt.Line2D([], [], color='black'),
                d2l.plt.Line2D([], [], color='black', ls='--')],
               ['delayed', 'current'], fontsize=8)
    for ax, what in ((ax1, 'saturation'), (ax2, 'underflow')):
        for m in marks:
            ax.axvline(m, color='black', ls=':', lw=1)
        ax.set_xlabel('step')
        ax.set_ylabel(f'{what} (ppm)')
    ax1.legend(fontsize=8, title='operand', title_fontsize=8)
    fig.tight_layout()
```
