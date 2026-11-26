# Case Study: Optimizing a Transformer
:label:`sec_fast_transformer`

Each section of this chapter has isolated one cause of a slow training step and the change that removes it: host dispatch overhead, memory traffic from small kernels, wide number formats, and single-GPU boundaries. A real training step suffers from several at once, and removing one shifts the bottleneck to the others. Here we take a single model, as a practitioner would, and ask how much faster its training step can become—and in what order the changes should be applied.

The model is the GPT of :numref:`sec_gpt` with its width doubled to 512 (19 million parameters). One training step on 64 windows of 128 bytes ($8{,}192$ tokens) performs about $6 \times 1.9 \times 10^{7} \times 8{,}192 \approx 9 \times 10^{11}$ FLOPs: two per parameter and token in the forward pass and four in the backward pass. On an RTX 4090 ($165\text{ TFLOP/s}$ in bf16), the arithmetic needs less than $6\text{ ms}$. A plain fp32 step with AdamW takes about $25\text{ ms}$. The number format alone accounts for about $6\text{ ms}$ of that $19\text{ ms}$ gap (tf32 tensor cores run at half the bf16 rate, requiring $11\text{ ms}$); the remaining $13\text{ ms}$ or so is time in which the tensor cores run below peak or sit idle.

We follow the method of :numref:`sec_perf_model`: count, predict, measure, and explain the difference. Before each change, the profile of the previous step bounds what the change can gain; after it, we compare. Each change is measured cumulatively on top of the previous ones. The section concludes by checking that the fastest step still trains the model and accounting for where its remaining time goes.

```{.python .input #fast-transformer-setup}
%%tab jax
# Let XLA overlap collectives with computation. JAX reads XLA_FLAGS once,
# when it starts, so this cell runs before the imports. Our outputs come
# from a run with three more settings in the environment, which you set
# before starting: --xla_gpu_enable_triton_gemm=false in XLA_FLAGS, so that
# XLA calls cuBLAS for the matrix products (the Triton kernels it would
# generate run the products of the large minibatches more slowly on the
# RTX 4090); NCCL_LOCAL_REGISTER=0, since NCCL's buffer registration fails
# on GPUs without peer-to-peer access and would print a warning at every
# collective; and XLA_PYTHON_CLIENT_ALLOCATOR=vmm, which maps each array
# into one contiguous range: the fp32 step of the training check needs
# about 13 GiB in one piece after many other arrays have come and gone.
import os
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                           ' --xla_gpu_enable_latency_hiding_scheduler=true')
```

```{.python .input #fast-transformer-case-study-making-a-transformer-fast}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #fast-transformer-case-study-making-a-transformer-fast}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #fast-transformer-case-study-optimizing-a-transformer-1}
%%tab jax
import re
from jax.sharding import NamedSharding, PartitionSpec as P
```

```{.python .input #fast-transformer-case-study-optimizing-a-transformer-1}
%%tab pytorch
import os
import re
from torch.utils.checkpoint import checkpoint
```

```{.python .input #fast-transformer-case-study-optimizing-a-transformer-2}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 products on tf32 cores
# Detach PyTorch's profiler from the GPU when a profile ends; without this,
# every kernel launch after the first profile stays about 30% slower
os.environ['TEARDOWN_CUPTI'] = '1'
```

## Experimental Setup
:label:`subsec_ft-subject`

Every change in this section is judged by the same measurement, so the model, data, and throughput benchmark stay fixed throughout.

### Model and Data

We train `d2l.GPT` (:numref:`sec_gpt`): six pre-norm blocks with eight attention heads each, rotary position embeddings, RMSNorm, SwiGLU, and an input embedding tied to the output layer. The data are 128-byte windows of WikiText-103 drawn at random offsets over the 256-byte vocabulary (`d2l.WikiText103`, :numref:`sec_test_models`).

We double the width from 256 to 512 so the matrix products sit comfortably in the compute-bound regime (:numref:`sec_perf_model`). Multiplying an $n \times d$ activation matrix by a $d \times d'$ weight matrix performs $2ndd'$ FLOPs and moves $b(nd + dd' + nd')$ bytes ($b$ bytes per element). For $n = 8{,}192$ tokens at width $d = 512$, the block projections reach $125\text{--}185\text{ FLOP/byte}$ in tf32 ($b = 4$) and $250\text{--}370\text{ FLOP/byte}$ in bf16 ($b = 2$), well above the RTX 4090 ridge points ($82\text{ FLOP/byte}$ for tf32 and $164\text{ FLOP/byte}$ for bf16). A faster number format therefore shortens these products directly. At width 256 they straddle the ridge point (Exercise :exref:`ex_ft-width`).

The 270 MB training corpus fits in GPU memory as a flat byte array. The next cell places it on the GPU, defines `windows` to sample minibatches directly on the device without host transfers, and builds the model.

```{.python .input #fast-transformer-the-subject-1}
%%tab pytorch
device = d2l.try_gpu()
data = d2l.WikiText103(batch_size=64, num_steps=128)
corpus = torch.tensor(data.train, device=device)  # One byte per token

def windows(corpus, batch_size, num_steps=128):  #@save
    """Endless minibatches of windows at random offsets, each position with
    the next byte as its target."""
    offsets = torch.arange(num_steps + 1, device=corpus.device)
    while True:
        starts = torch.randint(len(corpus) - num_steps, (batch_size, 1),
                               device=corpus.device)
        W = corpus[starts + offsets].long()
        yield W[:, :-1], W[:, 1:]

model = d2l.GPT(len(data.vocab), num_hiddens=512).to(device)
num_params = d2l.num_params(model)
print(f'{num_params / 1e6:.1f}M parameters, '
      f'{len(corpus) / 1e6:.0f}M training bytes on the GPU')
```

```{.python .input #fast-transformer-the-subject-1}
%%tab jax
data = d2l.WikiText103(batch_size=64, num_steps=128)
corpus = jnp.asarray(data.train)  # One byte per token

def windows(corpus, batch_size, num_steps=128, *, sharding=None):  #@save
    """Endless minibatches of windows at random offsets, each position with
    the next byte as its target; `sharding` spreads them over devices."""
    offsets = jnp.arange(num_steps + 1)
    while True:
        starts = jax.random.randint(d2l.get_key(), (batch_size, 1), 0,
                                    len(corpus) - num_steps)
        if sharding is not None:
            starts = jax.device_put(starts, sharding)
        W = corpus[starts + offsets].astype(jnp.int32)
        yield W[:, :-1], W[:, 1:]

model = d2l.GPT(len(data.vocab), num_hiddens=512,
                rngs=nnx.Rngs(d2l.get_key()))
num_params = d2l.num_params(model)
print(f'{num_params / 1e6:.1f}M parameters, '
      f'{len(corpus) / 1e6:.0f}M training bytes on the GPU')
```

At 16 bytes per parameter (weight, gradient, and two Adam moments), the 19M-parameter training state occupies only $0.3\text{ GiB}$, requiring neither gradient accumulation nor sharding (:numref:`sec_memory_precision`, :numref:`sec_sharding`). Keeping the corpus on the GPU also isolates the training step from the input pipeline (:numref:`sec_hardware`).

### Throughput Measurement

We measure end-to-end training throughput in tokens per second, including minibatch sampling, forward and backward passes, and the optimizer update:

| setting | value |
|:--|:--|
| hardware | 1 of 4 RTX 4090 GPUs (2 and 4 for data parallelism) |
| model | `d2l.GPT`: 6 blocks, width 512, 8 heads, byte vocabulary, context 128 |
| data | random 128-byte windows of WikiText-103 drawn on the GPU every step |
| optimizer | AdamW ($\text{lr} = 10^{-3}$, weight decay $0.01$) |
| precision | fp32 with tf32 matrix products until bf16 is introduced |
| timing | `d2l.Benchmark`: 20 warm-up steps, then 3 windows of 50 steps between device syncs |
| result | median tokens/s across the 3 windows, ratio to previous kept config, and relative window spread |

`tokens_per_second` applies these rules via `d2l.Benchmark` (:numref:`sec_perf_model`). `record` stores each result and prints its speedup over the previous kept configuration along with the relative spread across the three measurement windows ($\text{range} / \text{median}$). Configurations passed with `kept=False` are stored as branches that subsequent configurations do not build upon.

```{.python .input #fast-transformer-the-subject-2}
%%tab pytorch, jax
def tokens_per_second(step, batches, warmup=20, repeats=50, num_windows=3):
    """Median training throughput of `step` on minibatches drawn from
    `batches`, over `num_windows` timed windows of `repeats` steps each, and
    the spread of the windows (largest minus smallest) relative to it."""
    X, _ = next(batches)
    speeds = [math.prod(X.shape) / d2l.Benchmark(
                  lambda: step(*next(batches)), warmup if i == 0 else 0,
                  repeats).time for i in range(num_windows)]
    return np.median(speeds), np.ptp(speeds) / np.median(speeds)

results, branches = {}, {}
def record(name, measured, kept=True):
    """Store a measured throughput and print it with its ratio to the last
    kept configuration and the spread of its windows; a configuration that
    is not kept is stored in `branches`."""
    speed, spread = measured
    previous = list(results.values())[-1] if results else speed
    (results if kept else branches)[name] = speed
    print(f'{name}: {speed:,.0f} tokens/s ({speed / previous:.2f}x), '
          f'windows within {spread:.1%}')
```

Steps that perform substantial host work are sensitive to background host activity and show a wider window spread than compiled or graph-captured steps. Because day-to-day host load varies more than consecutive windows in a single run, we read speedup ratios to one decimal place and trust small gains of a few percent only when the profile predicts them.

## A Single GPU
:label:`subsec_ft-rungs`

We first optimize the training step on a single GPU, one change at a time, starting from the baseline implementation of :numref:`sec_gpt`.

### Baseline
:label:`subsec_ft-baseline`

The baseline is the standard training step of :numref:`sec_gpt` with AdamW: forward pass, cross-entropy loss, backward pass, and parameter update.

:begin_tab:`pytorch`
The step runs eagerly: Python and autograd dispatch each forward and backward operation as individual kernel launches, while AdamW (`fused=True`) updates all parameters and moments in a few fused kernels. `make_step` packages the training step; its `forward` and `bf16` arguments support the modifications in the subsections ahead.
:end_tab:

:begin_tab:`jax`
`train_step` is compiled with `nnx.jit` as in `d2l.train_steps` (:numref:`sec_test_models`). It takes the model and optimizer as arguments and computes the cross-entropy loss from fp32 logits.
:end_tab:

```{.python .input #fast-transformer-rung-0-baseline-profiled}
%%tab pytorch
def make_step(model, forward=None, bf16=False, warmup=0):  #@save
    """An AdamW training step of `model` on a minibatch (X, Y); `forward`
    replaces the model's forward pass, `bf16` turns on autocast, and a
    `warmup` > 0 raises the learning rate linearly to 1e-3 over as many
    steps."""
    forward = model if forward is None else forward
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=0.01, fused=True)
    if warmup:
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda i: min(1, (i + 1) / warmup))
    def step(X, Y):
        with torch.autocast('cuda', torch.bfloat16, enabled=bf16):
            loss = F.cross_entropy(forward(X).flatten(0, 1), Y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if warmup:
            schedule.step()
        return loss
    return step

step = make_step(model)
record('baseline', tokens_per_second(step, windows(corpus, 64)))
```

```{.python .input #fast-transformer-rung-0-baseline-profiled}
%%tab jax
def train_step(model, optimizer, X, Y):
    """Loss, gradients, and AdamW update for one minibatch."""
    def loss_fn(model):
        logits = model(X).astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, Y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

adamw = optax.adamw(1e-3, weight_decay=0.01)
optimizer = nnx.Optimizer(model, adamw, wrt=nnx.Param)
jit_step = nnx.jit(train_step)
step = lambda X, Y: jit_step(model, optimizer, X, Y)
record('baseline', tokens_per_second(step, windows(corpus, 64)))
```

As expected, the baseline windows spread roughly ten times more widely than later configurations because of per-step host work. To see where the step spends its time, `profile` records five steps with `d2l.kernel_times`, displays the breakdown with `d2l.show_profile` (:numref:`subsec_perf-profiler`) (total device kernel time versus wall-clock step time, the share taken by matrix products including fused attention kernels, and the top kernels), and saves the profile for our final accounting.

```{.python .input #fast-transformer-the-baseline-2}
%%tab pytorch, jax
profiles = {}
def profile(step, batches):
    """Kernel profile of `step`, the configuration recorded last, on
    minibatches drawn from `batches`: printed against the throughput that
    `record` measured, and stored in `profiles` under the same name."""
    name, speed = list(results.items())[-1]
    X, _ = next(batches)
    kernels = d2l.kernel_times(lambda: step(*next(batches)), steps=5)
    d2l.show_profile(kernels, 1e3 * math.prod(X.shape) / speed, rows=4)
    profiles[name] = kernels

profile(step, windows(corpus, 64))
```

The device runs kernels for nearly the entire step, with matrix products taking about two-thirds of the time. Because the host mostly keeps up, the step is bottlenecked by device work.

:begin_tab:`pytorch`
The remaining third outside the matrix products is fragmented across many short kernels (SwiGLU, rotary embeddings, residual additions, and RMSNorm), each reading and writing device memory and thus limited by memory bandwidth (:numref:`subsec_comp-fusion`).
:end_tab:

:begin_tab:`jax`
Because `nnx.jit` compiles the entire step, XLA has already fused the elementwise operations into surrounding kernels. The third of device time outside the matrix products goes to these fused kernels, which remain limited by the bytes they move.
:end_tab:

### Host Overhead and Small Kernels

Two remedies target the time spent outside matrix products, and the profile bounds what each can gain. The bound is *Amdahl's law* :cite:`Amdahl.1967`: if a change speeds up a fraction $f$ of the step by a factor $s$, the step as a whole speeds up by $1/(1 - f + f/s)$, which stays below $1/(1 - f)$ however large $s$ becomes. Applied to the two remedies:

1. *Reducing host overhead* targets idle gaps where the device waits for the host (:numref:`sec_perf_model`); its speedup is bounded by the idle fraction of the step.
2. *Kernel fusion* merges chains of bandwidth-bound elementwise kernels so intermediate arrays stay on chip; its speedup is bounded by the non-matmul device time.

:begin_tab:`pytorch`
Our baseline is in the second regime: the GPU is idle for only a few percent of the step, while non-matmul kernels consume $8.8\text{ ms}$ out of $24.5\text{ ms}$ of device time. Even if compilation eliminated all non-matmul kernels and idle gaps, the step would shrink to the $15.6\text{ ms}$ of matrix products—at most a $1.6\times$ speedup. `torch.compile` (:numref:`sec_compilation`) fuses elementwise operations across the forward and backward passes while leaving matrix products and the fused optimizer update intact. Passing `mode='reduce-overhead'` and `dynamic=False` also captures static-shape CUDA graphs (:numref:`subsec_comp-overhead`), eliminating the few percent of host idle time now and preventing host dispatch from becoming a bottleneck as device execution shrinks later. First, we verify that the compiled model matches the eager logits.
:end_tab:

:begin_tab:`jax`
Our step is already fused by XLA and the GPU waits for only 2% of the step, so reducing host work can gain at most 2% right now. We remove it anyway because it is a fixed per-step cost that would surface once later optimizations shorten device execution. On every call, `nnx.jit` traverses the Python model and optimizer objects to extract and update their arrays. Below, we benchmark this traversal overhead on a no-op function using plain `nnx.jit`, `nnx.jit(..., graph=False)` (tree mode), and `nnx.jit_partial(..., graph=False)` (which binds the model and optimizer state once so each step passes only the minibatch).
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-1}
%%tab pytorch
compiled = torch.compile(model, dynamic=False, mode='reduce-overhead')
X, _ = next(windows(corpus, 64))
high = model(X)
error = ((compiled(X) - high).norm() / high.norm()).item()
print(f'relative difference of the compiled logits: {error:.1e}')
assert error < 1e-4
step = make_step(model, compiled)
record('+ compile', tokens_per_second(step, windows(corpus, 64)))
```

```{.python .input #fast-transformer-rungs-each-one-measured-1}
%%tab jax
print(f'{len(jax.tree.leaves(nnx.state((model, optimizer))))} arrays in '
      'the model and the optimizer')
noop = lambda model, optimizer: None
walk, tree = nnx.jit(noop), nnx.jit(noop, graph=False)
bound = nnx.jit_partial(noop, model, optimizer, graph=False)
for name, call in (('nnx.jit', lambda: walk(model, optimizer)),
                   ('nnx.jit, graph=False', lambda: tree(model, optimizer)),
                   ('nnx.jit_partial', bound)):
    seconds = d2l.Benchmark(call, repeats=100).time
    print(f'{name}: {1e3 * seconds:.2f} ms per call that does nothing')
```

:begin_tab:`pytorch`
The compiled logits match to four significant digits (minor differences arise from reordered fp32 rounding inside fused kernels amplified by tf32 matrix products). Throughput rises by over 10%, well within the $1.6\times$ upper bound.
:end_tab:

:begin_tab:`jax`
A no-op call through `nnx.jit` takes nearly $9\text{ ms}$ of host time, over a third of the step's device time. That cost is hidden only as long as the host stays ahead of the device; switching to tree traversal removes most of it, and binding with `nnx.jit_partial` eliminates nearly all the rest. Next, we verify that `nnx.jit_partial` produces identical loss and weight updates to `nnx.jit`.
:end_tab:

```{.python .input #fast-transformer-host-overhead-1}
%%tab jax
twin = nnx.clone((model, optimizer))  # Equal states for the check
step = nnx.jit_partial(train_step, model, optimizer, graph=False)
X, Y = next(windows(corpus, 64))
loss, twin_loss = step(X, Y), jit_step(*twin, X, Y)
gaps = jax.tree.map(lambda a, b: jnp.abs(a - b).max(),
                    nnx.state(model, nnx.Param), nnx.state(twin[0], nnx.Param))
gap = max(float(g) for g in jax.tree.leaves(gaps))
print(f'losses {loss:.4f} and {twin_loss:.4f}, largest difference of the '
      f'updated weights {gap:.1e}')
assert gap < 1e-6
record('+ jit_partial', tokens_per_second(step, windows(corpus, 64)))
```

:begin_tab:`jax`
The updated weights agree exactly, and throughput rises by about 1%, within the 2% bound and the baseline noise spread.
:end_tab:

A second profile shows how the step changed:

```{.python .input #fast-transformer-rungs-each-one-measured-5}
%%tab pytorch, jax
profile(step, windows(corpus, 64))
```

The device remains busy for nearly the entire step, and matrix products now account for roughly two-thirds or more of device time.

:begin_tab:`pytorch`
Device time per step dropped from $24.5\text{ ms}$ to $22.3\text{ ms}$ and inter-kernel gaps vanished: Inductor fusion shaved $2.2\text{ ms}$ (a quarter of non-matmul time), while CUDA graph replay closed about $1\text{ ms}$ of host launch gaps.
:end_tab:

:begin_tab:`jax`
Device kernel times are unchanged because `nnx.jit_partial` only reduced host-side dispatch overhead.
:end_tab:

### Reduced Precision
:label:`subsec_ft-precision`

With the GPU busy and matrix products dominating device time, the next step is to speed up the matrix products. Switching from tf32 to bf16 doubles RTX 4090 tensor-core peak throughput to $165\text{ TFLOP/s}$ and halves the memory footprint of bf16 activations (:numref:`sec_memory_precision`). If matrix products take a fraction $f \in [2/3, 0.7]$ of the step and only they double in speed, the step shrinks to $1 - f/2$ of its time—a speedup of about $1.5\times$ (rising toward $2\times$ if non-matmul kernels also move half as many bytes).

We apply the mixed-precision recipe of :numref:`subsec_mp-mixed`: matrix products (including attention) and intermediate elementwise operations run in bf16, while master weights, gradients, Adam moments, normalizations, the loss, and the residual stream remain in fp32 (keeping the residual stream in fp32 prevents small block updates from rounding away in bf16's 8-bit significand). Because bf16 matches fp32's exponent range, no loss scaling is needed. Before timing, we verify that the bf16 forward pass matches fp32 logits to within bf16 precision (~1%).

:begin_tab:`pytorch`
`torch.autocast` applies mixed precision per operation (:numref:`sec_numerics`), casting matrix products and attention inputs to bf16 while keeping embeddings, normalizations, the residual stream, and the loss softmax in fp32.
:end_tab:

:begin_tab:`jax`
`d2l.to_bf16` (:numref:`subsec_mp-mixed`) sets `dtype=jnp.bfloat16` on the linear layers while leaving `param_dtype` and all other layers (embeddings, normalizations, and the tied output projection) in fp32.
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-2}
%%tab pytorch
X, _ = next(windows(corpus, 64))
with torch.autocast('cuda', torch.bfloat16):
    low = compiled(X).float()
high = compiled(X)
error = ((low - high).norm() / high.norm()).item()
print(f'relative difference of the bf16 logits: {error:.1e}')
assert error < 0.01
step = make_step(model, compiled, bf16=True)
record('+ bf16', tokens_per_second(step, windows(corpus, 64)))
```

```{.python .input #fast-transformer-rungs-each-one-measured-2}
%%tab jax
xla16 = d2l.GPT(len(data.vocab), num_hiddens=512,
                rngs=nnx.Rngs(d2l.get_key()))
nnx.update(xla16, nnx.state(model, nnx.Param))  # The same weights
d2l.to_bf16(xla16)
X, _ = next(windows(corpus, 64))
high = model(X)
error = float(jnp.linalg.norm(xla16(X) - high) / jnp.linalg.norm(high))
print(f'relative difference of the bf16 logits: {error:.1e}')
assert error < 0.01
step = nnx.jit_partial(train_step, xla16,
                       nnx.Optimizer(xla16, adamw, wrt=nnx.Param), graph=False)
record('+ bf16', tokens_per_second(step, windows(corpus, 64)))
```

The bf16 logits agree with fp32 to within 1%.

:begin_tab:`pytorch`
Throughput rises by about $1.75\times$—exceeding the $1.55\times$ predicted from halving the 71% matrix-product share alone because bf16 GEMMs run at higher utilization than tf32 and intermediate fused kernels move 2-byte bf16 activations instead of 4-byte floats. Note that `F.scaled_dot_product_attention` has already been using a fused FlashAttention kernel since the baseline.
:end_tab:

:begin_tab:`jax`
Throughput rises by $1.45\times$, just under the $1.48\times$ predicted by halving the 65% matrix-product share alone, indicating that non-matmul kernels did not speed up. One culprit is attention: by default, `jax.nn.dot_product_attention` uses XLA's implementation, which materializes the full attention matrix in device memory (:numref:`sec_attention-at-scale`). Now that activations are in 16-bit precision, we can switch to cuDNN's fused FlashAttention kernel (`implementation='cudnn'`), which keeps the attention scores on chip.
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-9}
%%tab jax
class FusedGPT(d2l.GPT):
    """d2l.GPT with cuDNN's fused attention."""
    class CausalAttention(d2l.GPT.CausalAttention):
        @staticmethod
        def attend(q, k, v):  # cuDNN takes 16-bit inputs only
            return jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                                implementation='cudnn')
```

:begin_tab:`jax`
We instantiate `model16` from `FusedGPT` with the same parameters converted to
bf16, verify that its logits still match the fp32 reference `high` within 1%,
and benchmark the training step:
:end_tab:

```{.python .input #fast-transformer-fused-attention}
%%tab jax
model16 = FusedGPT(len(data.vocab), num_hiddens=512,
                   rngs=nnx.Rngs(d2l.get_key()))
nnx.update(model16, nnx.state(model, nnx.Param))  # The same weights
d2l.to_bf16(model16)
error = float(jnp.linalg.norm(model16(X) - high) / jnp.linalg.norm(high))
print(f'relative difference of the bf16 logits: {error:.1e}')
assert error < 0.01
step = nnx.jit_partial(train_step, model16,
                       nnx.Optimizer(model16, adamw, wrt=nnx.Param),
                       graph=False)
record('+ fused attention', tokens_per_second(step, windows(corpus, 64)))
```

:begin_tab:`jax`
Switching to cuDNN fused attention raises throughput by another third while preserving 1% logit agreement, bringing the combined bf16 + fused-attention speedup close to $2\times$.
:end_tab:

### Larger Minibatches

Increasing the minibatch size amortizes fixed per-step costs over more tokens. Because the device is already busy throughout the step, only device work that scales with parameter count rather than token count can be amortized:

- AdamW reads weights, gradients, and two moments, and writes weights and two moments ($28\text{ bytes/param}$ in fp32).
- Mixed precision casts fp32 master weights to bf16 in the forward pass and bf16 weight gradients back to fp32 in the backward pass ($12\text{ bytes/param}$).

Together, these move $40\text{ bytes/param}$, or $0.76\text{ GB/step}$. At $1.0\text{ TB/s}$ of memory bandwidth, this fixed work takes about $0.75\text{ ms}$—roughly $1/16$ of a $12\text{--}13\text{ ms}$ bf16 step at batch size 64. Growing the minibatch $8\times$ from 64 to 512 windows ($65{,}536$ tokens) amortizes $7/8$ of this fixed cost, predicting around a 5% throughput gain.

Even a modest single-GPU gain is worthwhile because data parallelism scales well only when the per-GPU step time is much longer than the gradient allreduce (:numref:`subsec_ft-dp`). Keep in mind, however, that larger batches reduce sample efficiency once the batch size exceeds the critical batch size (:numref:`sec_batch_size`, Exercise :exref:`ex_ft-target`). Because activation memory grows linearly with batch size, we also record peak memory beyond weights and optimizer state (:numref:`subsec_mp-measuring`).

```{.python .input #fast-transformer-rungs-each-one-measured-3}
%%tab pytorch
torch.cuda.reset_peak_memory_stats()
start = torch.cuda.memory_allocated()
record('+ batch 512', tokens_per_second(step, windows(corpus, 512)))
memory = (torch.cuda.max_memory_allocated() - start) / 2**30
print(f'memory beyond weights and optimizer state: {memory:.1f} GiB')
```

```{.python .input #fast-transformer-rungs-each-one-measured-3}
%%tab jax
batches = windows(corpus, 512)
plan = step.lower(*next(batches)).compile().memory_analysis()
record('+ batch 512', tokens_per_second(step, batches))
memory = plan.temp_size_in_bytes / 2**30
print(f'memory beyond weights and optimizer state: {memory:.1f} GiB')
```

At batch size 512, activation memory uses about a third of the GPU's $24\text{ GiB}$, and throughput increases by a few percent (Exercise :exref:`ex_ft-fixed` separates the fixed per-step cost from the per-token cost across batch sizes). A third profile shows the breakdown at batch size 512:

```{.python .input #fast-transformer-larger-minibatches-1}
%%tab pytorch, jax
profile(step, windows(corpus, 512))
```

The device is busy for the entire step: matrix products (including fused attention) take just under 60% of the time, while fused bandwidth-bound kernels for RMSNorm, SwiGLU, rotary embeddings, and fp32/bf16 casts take the rest.

### Activation Checkpointing

Activation checkpointing (:numref:`subsec_mp-checkpointing`) saves memory by storing only each block's input during the forward pass and recomputing its internal activations during the backward pass. Because batch size 512 uses only a third of GPU memory, checkpointing is unnecessary here and can only slow the step down. We can predict the slowdown in two ways:

1. *By FLOP count*: recomputing the forward pass raises per-token block arithmetic from $6$ to $8\text{ FLOPs/param}$, predicting $6/8 = 0.75$ of the uncheckpointed throughput.
2. *By timing the forward pass*: below, we time a standalone forward pass $t_{\text{fwd}}$ at batch size 512 and compute $1 / (1 + t_{\text{fwd}} / t_{\text{step}})$.

```{.python .input #fast-transformer-activation-checkpointing-1}
%%tab pytorch
X512, _ = next(windows(corpus, 512))  # One minibatch of the last step
with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
    t_forward = d2l.Benchmark(lambda: compiled(X512)).time
```

```{.python .input #fast-transformer-activation-checkpointing-1}
%%tab jax
X512, _ = next(windows(corpus, 512))  # One minibatch of the last step
forward = nnx.jit_partial(lambda model, X: model(X), model16, graph=False)
t_forward = d2l.Benchmark(lambda: forward(X512)).time
```

Comparing `t_forward` against the full step duration `t_step` at batch size 512
predicts the fraction of throughput `1 / (1 + t_forward / t_step)` retained
when every block's forward pass is repeated:

```{.python .input #fast-transformer-activation-checkpointing-2}
%%tab pytorch, jax
t_step = 512 * 128 / results['+ batch 512']  # Seconds per step
print(f'forward pass {1e3 * t_forward:.1f} ms of a {1e3 * t_step:.1f} ms '
      f'step: repeating it keeps at least {1 / (1 + t_forward / t_step):.2f} '
      f'of the throughput')
```

Both estimates agree: the forward pass takes about a third of the step, so repeating it should retain at least three-quarters of the throughput (slightly more because embeddings and the output projection are not checkpointed). We now wrap each block in `Checkpointed`, verify logit equality, and measure both throughput (`kept=False`) and activation memory.

:begin_tab:`pytorch`
`Checkpointed` calls `torch.utils.checkpoint.checkpoint`.
:end_tab:

:begin_tab:`jax`
`Checkpointed` calls `nnx.remat`.
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-4}
%%tab pytorch
class Checkpointed(nn.Module):
    """A module that keeps only its input for the backward pass."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, X, **kwargs):
        return checkpoint(self.module, X, use_reentrant=False, **kwargs)

remat = d2l.GPT(len(data.vocab), num_hiddens=512).to(device)
remat.load_state_dict(model.state_dict())
remat.blks = nn.ModuleList(Checkpointed(blk) for blk in remat.blks)
assert torch.allclose(remat(X), model(X), atol=1e-6)
step = make_step(remat, torch.compile(remat, dynamic=False,
                                      mode='reduce-overhead'), bf16=True)
with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
    forward = d2l.kernel_times(lambda: compiled(X512), steps=5)  # Alone
torch.cuda.reset_peak_memory_stats()
start = torch.cuda.memory_allocated()
batches = windows(corpus, 512)
record('+ checkpointing', tokens_per_second(step, batches), kept=False)
memory = (torch.cuda.max_memory_allocated() - start) / 2**30
print(f'memory beyond weights and optimizer state: {memory:.1f} GiB')

def split(kernels):
    """Device ms per step of the matrix products and of all other kernels."""
    products = sum(ms for name, ms in kernels.items()
                   if re.search(d2l.PRODUCT_KERNELS, name, re.I))
    return np.array([products, sum(kernels.values()) - products])

alone = split(forward)
added = (split(d2l.kernel_times(lambda: step(*next(batches)), steps=5))
         - split(profiles['+ batch 512']))
print(f'forward pass alone:  products {alone[0]:4.1f} ms, other kernels '
      f'{alone[1]:4.1f} ms\nadded to the step:   products {added[0]:4.1f} ms, '
      f'other kernels {added[1]:4.1f} ms')
```

```{.python .input #fast-transformer-rungs-each-one-measured-4}
%%tab jax
class Checkpointed(nnx.Module):
    """A module that keeps only its input for the backward pass."""
    def __init__(self, module):
        self.module = module

    def __call__(self, X, **kwargs):
        return nnx.remat(lambda module, X, kwargs: module(X, **kwargs))(
            self.module, X, kwargs)

remat = FusedGPT(len(data.vocab), num_hiddens=512,
                 rngs=nnx.Rngs(d2l.get_key()))
nnx.update(remat, nnx.state(model16, nnx.Param))
remat.blks = nnx.List(Checkpointed(blk) for blk in remat.blks)
d2l.to_bf16(remat)
assert jnp.allclose(remat(X), model16(X), atol=1e-6)
optimizer = nnx.Optimizer(remat, adamw, wrt=nnx.Param)
step = nnx.jit_partial(train_step, remat, optimizer, graph=False)
batches = windows(corpus, 512)
plan = step.lower(*next(batches)).compile().memory_analysis()
record('+ checkpointing', tokens_per_second(step, batches), kept=False)
memory = plan.temp_size_in_bytes / 2**30
print(f'memory beyond weights and optimizer state: {memory:.1f} GiB')
```

Checkpointing cuts activation memory by roughly two-thirds while retaining more throughput than the standalone forward-pass time predicted.

:begin_tab:`pytorch`
Throughput drops to $0.85\times$ (above the $0.77\times$ bound). As the kernel split reveals, recomputing the block matrix products adds $14\text{ ms}$ (close to the $17\text{ ms}$ of all forward products including the output projection), whereas recomputing the non-matmul kernels adds only $3\text{ ms}$ instead of $13\text{ ms}$: Inductor fuses the recomputed elementwise operations directly into the backward kernels that consume them, making their recomputation nearly free in memory traffic.
:end_tab:

:begin_tab:`jax`
Throughput drops to $0.79\times$ (above the $0.76\times$ bound) because XLA similarly fuses recomputed elementwise work into the backward pass (Exercise :exref:`ex_ft-remat`).
:end_tab:

Checkpointing is essential when a model or batch size would otherwise exceed device memory, but here memory is plentiful, so we leave it out of the configurations that follow.

## Data Parallelism
:label:`subsec_ft-dp`

Our fastest single-GPU step leaves three of the machine's four RTX 4090s idle. Before running on multiple GPUs, we use the cost model of :numref:`subsec_mg-accounting` to predict the data-parallel speedup.

### Predicted Speedup

In data parallelism (:numref:`sec_multi_gpu`), each of $k$ GPUs holds a replica of the model, computes gradients on its own minibatch, and sums gradients via an allreduce of size $S$ bytes in $c$ buckets:

$$
t_{\textrm{compute}}(B/k) + c\,\alpha +
\frac{2(k-1)}{k}\,S\,\beta_{\textrm{link}}.
$$

On our PCIe machine without peer-to-peer access, GPU transfers stage through host memory at a bus bandwidth of roughly $2.4\text{ GB/s}$ (:numref:`sec_multi_gpu_concise`), or $\beta_{\textrm{link}} \approx 0.42\text{ ns/byte}$. Under weak scaling (fixed per-GPU batch size), per-bucket latency $c\,\alpha$ is under a millisecond and negligible next to tens of milliseconds of step time, simplifying the step time without overlap to:

$$t_{\textrm{step}}(k) \approx t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k),$$
:eqlabel:`eq_ft-dp`

for a throughput multiplier of $k\, t_{\textrm{step}}(1) / (t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k))$. Adding a second GPU increases throughput only if $t_{\textrm{allreduce}}(2) < t_{\textrm{step}}(1)$.

For our 19M-parameter GPT, fp32 gradients occupy $S = 76\text{ MB}$, giving $t_{\textrm{allreduce}}(2) \approx S\beta_{\textrm{link}} \approx 31\text{ ms}$ on two GPUs and $t_{\textrm{allreduce}}(4) \approx 1.5\,S\beta_{\textrm{link}} \approx 47\text{ ms}$ on four:

- *Compiled fp32 at batch 64 per GPU* ($t_{\textrm{step}}(1) \approx 22\text{ ms}$): the allreduce takes longer than the step itself. Equation :eqref:`eq_ft-dp` predicts a *slowdown* on 2 GPUs ($\sim 0.85\times$) and only $\sim 1.3\times$ on 4 GPUs.
- *Compiled bf16 at batch 512 per GPU* ($t_{\textrm{step}}(1) \approx 90\text{--}100\text{ ms}$): the step is much longer than the allreduce, predicting $\sim 1.5\times$ on 2 GPUs and $2.6\text{--}2.7\times$ on 4 GPUs.

Because frameworks overlap gradient allreduce buckets with the backward pass (:numref:`fig_ddp_overlap`), :eqref:`eq_ft-dp` is usually a lower bound on speedup, and the margin above it measures how much communication was hidden behind backward computation. It is not a guaranteed one: NCCL's copy kernels compete with the backward pass for the same GPU (:numref:`subsec_mgp-scaling`), and this contention can eat the margin.

### Measured Speedup

`data_parallel(k, bf16, batch_size)` measures training throughput on $k$ GPUs alongside the standalone allreduce time for a $76\text{ MB}$ gradient buffer on the same GPUs.

:begin_tab:`pytorch`
`DistributedDataParallel` runs one process per GPU via `d2l.torchrun` (:numref:`sec_multi_gpu_concise`), overlapping bucketed allreduces with the backward pass. It compiles without CUDA graphs (device execution is long enough to hide host dispatch) and benchmarks $k = 1$ under identical conditions.
:end_tab:

:begin_tab:`jax`
JAX drives all $k$ GPUs from a single process (:numref:`sec_multi_gpu_concise`), replicating the model, optimizer, and corpus across an `Auto` mesh while sharding each $k \times \text{batch\_size}$ minibatch across `'data'`. XLA automatically inserts the gradient allreduce and overlaps it with the backward pass via the latency-hiding scheduler enabled in the first cell.
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-7}
%%tab pytorch
DDP_GPT = r'''
import json, sys, torch
from torch.nn.parallel import DistributedDataParallel as DDP
from d2l import torch as d2l

rank, k, device = d2l.init_distributed()
bf16, batch_size = sys.argv[1] == 'bf16', int(sys.argv[2])
torch.set_float32_matmul_precision('high')
torch._inductor.config.compile_threads = 1  # k processes compile at once
data = d2l.WikiText103(batch_size, 128)
corpus = torch.tensor(data.train, device=device)
model = DDP(d2l.GPT(len(data.vocab), num_hiddens=512).to(device))
step = d2l.make_step(model, torch.compile(model, dynamic=False), bf16)
batches = d2l.windows(corpus, batch_size)
seconds = d2l.Benchmark(lambda: step(*next(batches)), 20, 50).time
grads = torch.zeros(d2l.num_params(model), device=device)
allreduce = d2l.Benchmark(lambda: torch.distributed.all_reduce(grads),
                          repeats=10).time
if rank == 0:
    print(json.dumps({'speed': k * batch_size * 128 / seconds,
                      'allreduce': allreduce}))
torch.distributed.destroy_process_group()
'''

# The CUDA graphs of the compiled steps above hold memory on GPU 0, which
# the process of rank 0 needs: reset the compiler and return the memory
torch.compiler.reset()
torch.cuda.empty_cache()

def data_parallel(k, bf16, batch_size):
    """Tokens per second on k GPUs, and seconds per allreduce of all
    gradients."""
    r = d2l.torchrun(DDP_GPT, k, 'bf16' if bf16 else 'fp32', batch_size)[0]
    return r['speed'], r['allreduce']
```

```{.python .input #fast-transformer-rungs-each-one-measured-7}
%%tab jax
def data_parallel(k, bf16, batch_size):
    """Tokens per second on k GPUs, and seconds per allreduce of all
    gradients."""
    mesh = jax.make_mesh((k,), ('data',), devices=jax.devices()[:k],
                         # Auto: the compiler places intermediate results
                         axis_types=(jax.sharding.AxisType.Auto,))
    copy, split = NamedSharding(mesh, P()), NamedSharding(mesh, P('data'))
    net = (FusedGPT if bf16 else d2l.GPT)(len(data.vocab), num_hiddens=512,
                                          rngs=nnx.Rngs(d2l.get_key()))
    net = d2l.to_bf16(net) if bf16 else net
    optimizer = nnx.Optimizer(net, adamw, wrt=nnx.Param)
    nnx.update((net, optimizer),
               jax.device_put(nnx.state((net, optimizer)), copy))
    step = nnx.jit_partial(train_step, net, optimizer, graph=False)
    batches = windows(jax.device_put(corpus, copy), k * batch_size,
                      sharding=split)
    speed, _ = tokens_per_second(step, batches)
    grads = jnp.zeros((k, num_params), device=split)  # All gradients per GPU
    psum = jax.jit(jax.shard_map(lambda g: jax.lax.psum(g, 'data'),
                                 mesh=mesh, in_specs=P('data'),
                                 out_specs=P('data')))
    return speed, d2l.Benchmark(lambda: psum(grads), repeats=10).time
```

We now run `data_parallel` for both the short fp32 step (`batch_size=64`) and
the long bf16 step (`batch_size=512`) across $k \in \{1, 2, 4\}$ GPUs,
comparing the no-overlap prediction :eqref:`eq_ft-dp` against the measured
speedup:

```{.python .input #fast-transformer-rungs-each-one-measured-8}
%%tab pytorch, jax
for bf16, batch_size in ((False, 64), (True, 512)):
    tokens = batch_size * 128  # Per GPU and step
    one, _ = data_parallel(1, bf16, batch_size)
    print(f'{"bf16" if bf16 else "fp32"}, batch {batch_size} per GPU: '
          f'{one:,.0f} tokens/s on one GPU')
    for k in (2, 4):
        if k <= d2l.num_gpus():
            speed, allreduce = data_parallel(k, bf16, batch_size)
            predicted = k * tokens / (tokens / one + allreduce)
            print(f'  {k} GPUs: allreduce {1e3 * allreduce:3.0f} ms, '
                  f'speedup predicted {predicted / one:.2f}x, '
                  f'measured {speed / one:.2f}x')
```

The standalone allreduce takes $31\text{ ms}$ on 2 GPUs and $48\text{ ms}$ on 4 GPUs, matching our link-rate estimate within a millisecond. Both predictions of opposite sign hold: at batch 64, 2 GPUs are slower than 1 GPU and 4 GPUs gain less than $1.5\times$; at batch 512 in bf16, 4 GPUs reach roughly $3\times$ single-GPU throughput (the largest speedup of any single change in this section), with communication overlap pushing the measured speedup above the no-overlap floor :eqref:`eq_ft-dp`. (On an NVLink machine with $900\text{ GB/s}$ per direction, a $76\text{ MB}$ allreduce takes a fraction of a millisecond and scales nearly linearly even at batch 64.)

## Results
:label:`subsec_ft-waterfall`

We now summarize all single-GPU optimizations side by side, verify that the final step still trains the model properly, and account for where its remaining time goes.

### Cumulative Comparison

The table below lists the single-GPU configurations in order. Each `step` ratio is measured relative to the row above it, `total` is the cumulative speedup over the baseline, and checkpointing appears below the line as an unkept branch. `TFLOP/s` converts throughput to arithmetic rate at $6\text{ FLOPs}$ per parameter and token, and `MFU` divides by the RTX 4090's $165.2\text{ TFLOP/s}$ bf16 peak (*model FLOPs utilization*, :citet:`chowdhery2022palm`; Exercise :exref:`ex_ft-mfu` adds the attention-score FLOPs).

```{.python .input #fast-transformer-the-waterfall}
%%tab pytorch, jax
flops = 6 * num_params  # Per token, without the attention scores
peak = {'tf32': 82.6e12, 'bf16': 165.2e12}  # FLOP/s of the RTX 4090
print(f'{"configuration":18}{"tokens/s":>10}{"step":>7}{"total":>7}'
      f'{"TFLOP/s":>9}{"MFU":>6}')
first = previous = next(iter(results.values()))
for name, speed in results.items():
    print(f'{name:18}{speed:10,.0f}{speed / previous:6.2f}x'
          f'{speed / first:6.2f}x{speed * flops / 1e12:9.1f}'
          f'{speed * flops / peak["bf16"]:6.0%}')
    previous = speed
print('-' * 57)
for name, speed in branches.items():  # Measured on top of the last row
    print(f'{name:18}{speed:10,.0f}{speed / previous:6.2f}x{"":7}'
          f'{speed * flops / 1e12:9.1f}{speed * flops / peak["bf16"]:6.0%}'
          '  (not kept)')
```

Cumulatively, the single-GPU optimizations double training throughput, reaching roughly 50% of the RTX 4090's bf16 peak. Nearly all of the gain comes from speeding up the matrix products (bf16) and reducing memory traffic around them (kernel fusion), whereas removing host overhead and enlarging the minibatch contribute a few percent each, just as the profiles predicted.

Note that each speedup ratio depends on its position in the sequence (Exercise :exref:`ex_ft-order`) and on the hardware balance: on a faster GPU (or with a smaller model), device execution is shorter, making host dispatch removal much more impactful.

### Training Check

Higher throughput is useless if an optimization silently breaks training. We verify correctness by training two identically initialized models (one with the baseline step and one with the final compiled bf16 step) on the same 200 minibatches of 512 windows ($\sim 13\text{M}$ tokens), using a 50-step linear learning-rate warm-up to prevent late AdamW loss spikes (:numref:`sec_gpt`). We check that both models agree within $0.15\text{ nats/byte}$ on the mean loss of the last 20 steps and on the validation loss (`d2l.lm_perplexity`, :numref:`sec_test_models`), and that both beat the $2.45\text{ nats/byte}$ bigram baseline of :numref:`sec_gpt`.

```{.python .input #fast-transformer-the-waterfall-1}
%%tab pytorch
learner = d2l.GPT(len(data.vocab), num_hiddens=512).to(device)
reference = d2l.GPT(len(data.vocab), num_hiddens=512).to(device)
reference.load_state_dict(learner.state_dict())  # One initialization
steps = {'baseline': make_step(reference, warmup=50),
         'final': make_step(learner, torch.compile(
             learner, dynamic=False, mode='reduce-overhead'), bf16=True,
             warmup=50)}
batches = windows(corpus, 512)
minibatches = [next(batches) for _ in range(200)]  # The same for both
losses = {name: torch.stack([step(X, Y).detach()
                             for X, Y in minibatches]).tolist()
          for name, step in steps.items()}
validation = {name: math.log(d2l.lm_perplexity(net, data.val_dataloader()))
              for name, net in (('baseline', reference), ('final', learner))}
```

```{.python .input #fast-transformer-the-waterfall-1}
%%tab jax
learner = d2l.to_bf16(FusedGPT(len(data.vocab), num_hiddens=512,
                               rngs=nnx.Rngs(d2l.get_key())))
reference = d2l.GPT(len(data.vocab), num_hiddens=512,
                    rngs=nnx.Rngs(d2l.get_key()))
nnx.update(reference, nnx.state(learner, nnx.Param))  # One initialization
# The learning rate rises linearly to 1e-3 over the first 50 steps
warm = optax.adamw(optax.linear_schedule(1e-3 / 50, 1e-3, 49),
                   weight_decay=0.01)
reference_optimizer = nnx.Optimizer(reference, warm, wrt=nnx.Param)
steps = {'baseline': lambda X, Y: jit_step(reference, reference_optimizer,
                                           X, Y),
         'final': nnx.jit_partial(train_step, learner, nnx.Optimizer(
             learner, warm, wrt=nnx.Param), graph=False)}
batches = windows(corpus, 512)
minibatches = [next(batches) for _ in range(200)]  # The same for both
losses = {name: jnp.stack([step(X, Y) for X, Y in minibatches]).tolist()
          for name, step in steps.items()}
validation = {name: math.log(d2l.lm_perplexity(net, data.val_dataloader()))
              for name, net in (('baseline', reference), ('final', learner))}
```

We now compare the training loss trajectories at steps 1, 50, 100, and 200
alongside the final 20-step mean (`late`) and validation log-perplexity
(`validation`), asserting that both runs beat the $2.45\text{ nats/byte}$
bigram threshold and agree within $0.15\text{ nats/byte}$:

```{.python .input #fast-transformer-training-check}
%%tab pytorch, jax
checkpoints = (1, 50, 100, 200)
late = {name: np.mean(curve[-20:]) for name, curve in losses.items()}
print(f'{"loss at step":14}' + ''.join(f'{i:>7}' for i in checkpoints)
      + f'{"mean of last 20":>17}{"validation":>12}')
for name, curve in losses.items():
    print(f'{name:14}' + ''.join(f'{curve[i - 1]:7.2f}' for i in checkpoints)
          + f'{late[name]:17.2f}{validation[name]:12.2f}')
    assert validation[name] < 2.45  # Better than the bigram model
for values in (late, validation):  # The final step trains like the baseline
    assert abs(values['baseline'] - values['final']) < 0.15
```

Both curves start at $\log 256 \approx 5.55$, track each other closely across all 200 steps, and reach nearly identical validation losses well below the bigram threshold. Notice, however, that 200 steps at batch size 512 ($13\text{M}$ tokens) end at a higher loss than 1,000 cosine-decayed steps at batch size 64 ($8\text{M}$ tokens) in :numref:`sec_gpt`: raw token throughput is not identical to time-to-target-loss (Exercise :exref:`ex_ft-target`).

### Account of the Final Step
:label:`subsec_ft-account`

Finally, `account` normalizes the baseline and final profiles to milliseconds per $8{,}192$ tokens, comparing total wall-clock time against ideal peak arithmetic time, actual matrix-product kernel time, and non-matmul kernel time.

```{.python .input #fast-transformer-account}
%%tab pytorch, jax
def account(kernels, tokens, speed):
    """Milliseconds per 8,192 tokens of a step of `tokens` tokens that runs
    at `speed` tokens/s: the step, its matrix products, and all its other
    kernels."""
    per_step = 8192 / tokens  # Steps per 8,192 tokens
    products = per_step * sum(  # Kernel names as in d2l.show_profile
        ms for name, ms in kernels.items()
        if re.search(d2l.PRODUCT_KERNELS, name, re.I))
    other = per_step * sum(kernels.values()) - products
    return 1e3 * 8192 / speed, products, other

print(f'{"ms per 8,192 tokens":24}{"step":>6}{"at peak":>9}'
      f'{"products":>10}{"other":>7}')
for name, tokens, fmt in (('baseline', 64 * 128, 'tf32'),
                          ('+ batch 512', 512 * 128, 'bf16')):
    wall, products, other = account(profiles[name], tokens, results[name])
    arithmetic = 1e3 * 8192 * flops / peak[fmt]  # At the peak of the format
    print(f'{name + " (" + fmt + ")":24}{wall:6.1f}{arithmetic:9.1f}'
          f'{products:10.1f}{other:7.1f}')
```

Per $8{,}192$ tokens, the final step spends about $7\text{ ms}$ in matrix products (running at over 80% of the $5.7\text{ ms}$ bf16 ideal) and $5\text{ ms}$ in bandwidth-bound kernels between them—a classic roofline split where contractions perform over 99% of the FLOPs but take roughly 60% of the time :cite:`Ivanov.Dryden.Ben-Nun.ea.2021`. Because neither faster arithmetic alone (e.g., fp8) nor eliminating non-matmul memory traffic alone can halve a step split $7\text{ ms}$ to $5\text{ ms}$, another $2\times$ speedup would require both.

## Summary

A kernel profile classifies a training step's bottleneck and bounds the gain of each candidate optimization before it is applied: reducing host dispatch saves at most the device idle time, kernel fusion saves at most the non-matmul device time, and doubling tensor-core precision throughput gives a speedup between $1/(1 - f/2)$ and $2$ when matrix products occupy a fraction $f$ of the step. Because removing one bottleneck shifts the remaining shares of the step, optimizations must be verified for numerical equivalence and measured cumulatively. On a single RTX 4090, our 19M-parameter GPT step doubled in speed to reach 45–50% MFU, driven almost entirely by bf16 matrix products and kernel fusion, leaving a final step split roughly 60/40 between near-peak matrix contractions and bandwidth-bound kernels. Finally, the data-parallel cost model $t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k)$ predicted both the slowdown on two GPUs for the short fp32 step ($22\text{ ms}$ step versus a $31\text{ ms}$ PCIe allreduce) and the roughly $3\times$ speedup on four GPUs for the batch-512 bf16 step ($90\text{--}100\text{ ms}$ step versus a $48\text{ ms}$ allreduce).

## Exercises

1. [code] **Model FLOPs utilization.** The GPT of :numref:`subsec_ft-subject` has $L = 6$ blocks of width $d = 512$ and context length $T = 128$. Up to SwiGLU rounding, each block holds $12d^2$ weights in its matrix products, and training costs about $6\text{ FLOPs}$ per weight and token.
    1. The attention score product $\mathbf{Q}\mathbf{K}^\top$ and value aggregation are weightless matrix products. Count their training FLOPs per token and block (taking the backward pass as twice the forward pass and including masked entries). Show that their ratio to the weight FLOPs is $T/(6d)$, and evaluate this ratio for $T = 128$ and $T = 1{,}024$.
    1. Recompute the `TFLOP/s` and `MFU` columns of the cumulative table with attention FLOPs included, along with the fraction of each configuration's native format peak ($165\text{ TFLOP/s}$ in bf16, $83\text{ TFLOP/s}$ in tf32).
    1. Name two configurations or kernels in :numref:`subsec_ft-rungs` where theoretical model FLOPs differ from hardware-executed FLOPs, and state the direction of each difference.
   :exlabel:`ex_ft-mfu`

1. [code] **Fixed cost of a step.** Model the bf16 step time at batch size $B$ windows as $t(B) = a + bB$, with fixed per-step cost $a$ and per-window cost $b$.
    1. Measure the step time at $B = 32, 64, 128, 256, 512$, fit $a$ and $b$ by least squares, and report the fraction of the $B = 64$ step accounted for by $a$.
    1. The text estimated AdamW updates and fp32/bf16 weight casts at $\sim 0.75\text{ ms/step}$. What fraction of $a$ does this explain? Identify other $B$-independent work and check against a profile at $B = 32$.
    1. Use your linear fit to predict throughput at $B = 1{,}024$ and $B = 2{,}048$, check whether they fit in $24\text{ GiB}$ of GPU memory, and measure those that do.
   :exlabel:`ex_ft-fixed`

1. [code] **Cost of recomputation.** Activation checkpointing in :numref:`subsec_ft-rungs` beat both the FLOP-count estimate and the standalone forward-pass timing estimate.
    1. Profile the standalone forward pass and the checkpointed step with `d2l.kernel_times`. Match the kernels added by checkpointing against the standalone forward kernels: which reappear unchanged, and which are fused into backward kernels?
    1. Time the forward pass of the six blocks alone (excluding embeddings and the output layer) and recompute the throughput lower bound.
    1. Checkpoint only every second block: predict the throughput and memory from your measurements before running, then measure and compare.
   :exlabel:`ex_ft-remat`

1. [code] **Causality gate.** Suppose a defective attention kernel silently ignores the causal mask.
    1. Explain why the training loss, validation window loss, and the $2.45\text{ nats/byte}$ bigram threshold in :numref:`subsec_ft-waterfall` would all fail to flag this defect if both baseline and learner used the defective kernel.
    1. Write a test that perturbs the final input token of every sequence in a minibatch and verifies that logits at all earlier positions remain unchanged within numerical tolerance. Run it on `learner` and on a copy whose attention omits the causal mask.

1. **Throughput versus time to target.** The model of :numref:`sec_batch_size` estimates that training at batch size $b$ requires $E(b) = E_{\min}(1 + b / b_{\textrm{noise}})$ training examples to reach a target loss, where $b_{\textrm{noise}}$ is the gradient noise scale.
    1. Let $\rho > 1$ be the throughput ratio of batch 512 to batch 64. Show that batch 512 reaches the target loss in less wall-clock time if and only if
       $$b_{\textrm{noise}} > \frac{512 - 64\rho}{\rho - 1},$$
       and evaluate this threshold for $\rho = 1.2$, $\rho = 1.7$, and the measured $\rho$ from :numref:`subsec_ft-waterfall`.
    1. Classify each change in :numref:`subsec_ft-rungs`, weak-scaling data parallelism, and switching to Muon (:numref:`sec_muon`) by whether it alters $E(b)$, token throughput, or both.
   :exlabel:`ex_ft-target`

1. [code] **Width 256.** At `num_hiddens=256`, `d2l.GPT` matches the architecture trained in :numref:`sec_gpt`.
    1. For the $8{,}192 \times 256$ by $256 \times 256$ output projection $\mathbf{W}_o$, compute the arithmetic intensity in 4-byte tf32 and 2-byte bf16, place both on the RTX 4090 roofline (:numref:`fig_roofline`), and predict the bf16 speedup for this product.
    1. Run the baseline and the first two optimizations of :numref:`subsec_ft-rungs` at `num_hiddens=256`, profile them with `profile` and `account`, and compare the measured matrix-product speedup with your roofline prediction.
   :exlabel:`ex_ft-width`

1. [code] **Order of the changes.** Each speedup ratio in the cumulative table depends on the optimizations applied before it.
    1. Using the baseline profile and the bound in :numref:`subsec_ft-precision`, predict the speedup of applying bf16 directly to the uncompiled baseline.
    1. Measure that speedup and compare it with your prediction and with the `+ bf16` row in the cumulative table. Then apply compilation (the table's second row) on top of bf16 and explain how both ratios shifted using the kernel profiles.
   :exlabel:`ex_ft-order`

[Discussions](https://d2l.discourse.group/t/36793)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.8]{.kicker}

Case Study: Making a Transformer Fast Step by Step<br>**profile-guided Amdahl bounds · fusion & CUDA graphs · bf16 & FlashAttention · batch sizing · multi-GPU scaling · MFU**
:::
:::

::: {.slide title="A GPT Step Needs 6 ms of Math but Takes 25 ms"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Hardware speed limit ($5.7\text{ ms}$):** One step of our 6-block, width-512 GPT ($N = 19\text{M}$, $n = 8{,}192$ tokens) performs $6Nn \approx 0.93\text{ TFLOP}$, requiring only $5.7\text{ ms}$ at $165.2\text{ TFLOP/s}$ (`bf16`).
- **Measured `fp32` baseline ($\sim 25\text{ ms}$):** Runs $> 4\times$ slower than the `bf16` compute floor ($\sim 23\%$ MFU).
- **Decomposing the $19\text{ ms}$ gap:** About $6\text{ ms}$ comes from the `tf32` vs. `bf16` Tensor Core rate; the remaining $\sim 13\text{ ms}$ comes from unfused bandwidth-bound kernels and dispatch gaps.
:::

::: {.col .fig .big}
@fig:ch13-perf-gpt-gap-overview
:::
:::
:::

::: {.slide title="Choosing Model Width 512: Placing Matmuls Above I*"}
[Roofline Positioning]{.kicker}

::: {.cols .vc}
::: {.col}
$$I \;=\; \frac{2 n d d'}{b\,(n d + d d' + n d')}$$

- **Clearing the ridge point:** At $d = 512$ and $n = 8{,}192$ tokens, block projections achieve $I = 125\text{--}185\text{ FLOP/B}$ in `tf32` ($> I^* = 82$) and $250\text{--}370\text{ FLOP/B}$ in `bf16` ($> I^* = 164$).
- **Why width 512 matters:** Because every GEMM sits firmly on the compute roof, doubling Tensor Core $P$ from `tf32` to `bf16` cuts GEMM time directly in half.
:::

::: {.col .fig}
@fig:ch13-perf-width512-roofline
:::
:::
:::

::: {.slide title="End-to-End Experimental Setup and Timing Protocol"}
[Benchmark Discipline]{.kicker}

To ensure every optimization is judged on equal footing, our benchmark holds the model, data pipeline, and timing protocol fixed across all rungs:

- **Zero host-transfer noise:** The entire $270\text{ MB}$ WikiText-103 byte corpus resides in GPU DRAM; each step samples random $128$-byte windows directly on the device.
- **Compact parameter state:** At $16\text{ B/param}$, the $19\text{M}$-parameter AdamW state occupies only $0.3\text{ GiB}$, requiring neither gradient accumulation nor FSDP sharding.
- **Synchronized multi-window measurement (`tokens_per_second`):** Runs $20$ warm-up steps followed by $3$ timed windows of $50$ steps between device barriers, reporting median $\text{tokens/s}$, incremental speedup, and window spread ($\textrm{range}/\textrm{median}$).
- **Correctness gate before timing:** Every rung asserts numerical agreement (`allclose` on logits or weight updates) before entering the benchmark.
:::

::: {.slide title="Using the Previous Profile to Bound the Next Speedup"}
[Predictive Method]{.kicker}

::: {.cols .vc}
::: {.col}
- **Host dispatch & fusion bounds:** Removing idle launch gaps bounds speedup by $t_{\textrm{step}}/t_{\textrm{busy}}$, while fusing non-matmul kernels ($t_{\textrm{other}}$) bounds speedup by $t_{\textrm{step}}/t_{\textrm{products}}$.
- **Precision (`tf32` $\to$ `bf16`):** When GEMMs take share $f$ of the step, doubling GEMM rate gives $\ge \frac{1}{1 - f/2}$ (up to $2.0\times$ as activation traffic also halves).
- **Batch sizing & remat:** Growing $B \to 8B$ amortizes only fixed $40\text{ B/param}$ work ($\le 1.05\times$), while remat costs $+2N\text{ FLOPs/tok}$ ($\ge 0.75\times$).
:::

::: {.col .fig .big}
@fig:ch13-perf-amdahl-bounds
:::
:::
:::

::: {.slide title="Rung 0 (Baseline): Profiling the Eager fp32 Step" only="pytorch"}
[Baseline Profile]{.kicker}

Profiling the eager `float32` baseline (`tf32` matmuls, fused AdamW) at batch size $64$ ($8{,}192$ tokens/step) establishes where the $\sim 25\text{ ms}$ goes:

@!fast-transformer-the-baseline-2

- **Diagnosis:** The GPU is busy for nearly $100\%$ of the $25\text{ ms}$ step (not overhead-bound), with matrix products taking $\sim 64\%$ ($15.6\text{ ms}$) and short bandwidth-bound elementwise/norm kernels taking the remaining $\sim 36\%$ ($8.8\text{ ms}$).
- **Amdahl bound for compilation:** Even if fusion and CUDA graphs eliminated *all* non-matmul kernels and launch gaps, the step could shrink at most from $25\text{ ms} \to 15.6\text{ ms}$ (**$\le 1.6\times$ speedup**).
:::

::: {.slide title="Rung 0 (Baseline): Profiling the Jitted fp32 Step" only="jax"}
[Baseline Profile]{.kicker}

In JAX, our baseline `train_step` is already compiled with `nnx.jit`, so XLA has already fused the elementwise chains:

@!fast-transformer-the-baseline-2

- **Diagnosis:** The GPU is busy for $98\%$ of the $\sim 24\text{ ms}$ step, with matrix products taking $\sim 65\%$ and XLA's fused kernels (plus non-fused XLA attention) taking $\sim 35\%$.
- **Amdahl bound for host-overhead removal:** Because the device waits for the host for only $2\%$ of the step right now, reducing Python dispatch work can gain **at most $2\%$** on this rung, though removing it prevents host dispatch from bottlenecking later rungs once device time shrinks.
:::

::: {.slide title="Rung 1: Inductor Fusion and CUDA Graph Replay" only="pytorch"}
[Compiler & Replay]{.kicker}

We compile the model with `torch.compile(model, dynamic=False, mode='reduce-overhead')`, combining Inductor operator fusion with static-shape CUDA graph replay:

```{.python #fast-transformer-exercises}
compiled = torch.compile(model, dynamic=False, mode='reduce-overhead')
high = model(X)
error = ((compiled(X) - high).norm() / high.norm()).item()
assert error < 1e-4      # verify numerical equivalence before timing
step = make_step(model, compiled)
```

@!fast-transformer-rungs-each-one-measured-1

- **Result ($1.16\times$ speedup):** Logits match to $1.5 \times 10^{-5}$; fusion shaves $2.2\text{ ms}$ off non-matmul kernels ($24.5 \to 22.3\text{ ms}$) and CUDA graphs close $\sim 1\text{ ms}$ of inter-kernel launch gaps.
:::

::: {.slide title="Rung 1: Eliminating PyTree Traversal with jit_partial" only="jax"}
[Host Dispatch Removal]{.kicker}

Timing a no-op call through `nnx.jit` reveals that traversing the Python `nnx.Module` and `nnx.Optimizer` reference graph takes nearly **$9\text{ ms}$ of CPU time per step**—hidden only because the GPU step takes $24\text{ ms}$:

@!fast-transformer-rungs-each-one-measured-1

- **`nnx.jit_partial(..., graph=False)`:** Binds the model and optimizer state once in tree mode so each step passes only the minibatch `(X, Y)`, cutting host dispatch from $\sim 9\text{ ms}$ to $0.23\text{ ms}$ (gaining $\sim 1\%$ throughput now and removing the $9\text{ ms}$ host floor for all later rungs).
:::

::: {.slide title="Profile After Rung 1: Matrix Products Take Two-Thirds"}
[Second Profile]{.kicker}

Profiling the step after Rung 1 shows the GPU busy for $99\text{--}100\%$ of the step, with matrix products consuming $f \approx 65\text{--}71\%$ of device time:

@!fast-transformer-rungs-each-one-measured-5

- **Predicting Rung 2 (`bfloat16` mixed precision):** Switching Tensor Core GEMMs from `tf32` ($82.6\text{ TFLOP/s}$) to `bfloat16` ($165.2\text{ TFLOP/s}$) halves matrix-product time.
- **Theoretical bounds:** If *only* matrix products ($f \approx 0.68$) double in speed, the speedup is $\frac{1}{1 - f/2} \approx \mathbf{1.52\times}$; if non-matmul kernels also halve their memory traffic by moving 2-byte activations, the speedup approaches **$2.0\times$**.
:::

::: {.slide title="Rung 2: Mixed-Precision bf16 Speeds Up the Step 1.75×" only="pytorch"}
[Reduced Precision]{.kicker}

Enabling `torch.autocast('cuda', torch.bfloat16)` runs linear layers and fused FlashAttention (`F.scaled_dot_product_attention`) in `bfloat16` while keeping master weights, norms, the residual stream, and the loss in `float32`:

@!fast-transformer-rungs-each-one-measured-2

- **Why $1.74\times$ sits cleanly inside $[1.55\times, 2.0\times]$:** Logits agree within $0.2\%$ ($< 1\%$ gate). The speedup exceeds the $1.55\times$ matmul-only bound because `bfloat16` GEMMs achieve higher Tensor Core efficiency than `tf32` and intermediate fused kernels move 2-byte `bfloat16` tensors instead of 4-byte `float32` tensors.
:::

::: {.slide title="Why bf16 Beat the Matmul-Only Amdahl Bound" only="pytorch"}
[Precision & Bandwidth]{.kicker}

Decomposing the $1.75\times$ speedup from `torch.autocast('cuda', torch.bfloat16)` highlights how reduced precision accelerates both roofline regimes simultaneously:

- **Compute-bound kernels (GEMMs & FlashAttention):** Tensor Core peak throughput doubles from $82.6 \to 165.2\text{ TFLOP/s}$, and smaller 16-bit operands reduce register-file pressure and shared-memory bank conflicts.
- **Bandwidth-bound kernels (SwiGLU, rotary embeddings, views):** Halving activation element size from $4\text{ B} \to 2\text{ B}$ cuts DRAM read/write traffic in half for every intermediate tensor between linear projections.
- **What stays in `float32` (preventing a full $2.0\times$):** Master weights, gradients, AdamW moments ($28\text{ B/param}$), RMSNorm reductions, residual-stream additions, and cross-entropy log-probabilities remain 32-bit for numerical stability.
:::

::: {.slide title="Rung 2a: Mixed-Precision bf16 Speeds Up Matmuls 1.45×" only="jax"}
[Reduced Precision]{.kicker}

Applying `d2l.to_bf16` sets `dtype=jnp.bfloat16` on all linear projections while keeping master parameters, norms, embeddings, and the residual stream in `float32`:

@!fast-transformer-rungs-each-one-measured-2

- **Why `bf16` alone gave $1.45\times$ (matching the $1.48\times$ matmul-only bound):** Default `jax.nn.dot_product_attention` uses an XLA implementation that materializes the full $T \times T$ attention matrix in DRAM.
:::

::: {.slide title="Rung 2b: cuDNN Fused FlashAttention Adds Another 1.35×" only="jax"}
[On-Chip Attention]{.kicker}

Now that Q, K, and V are in 16-bit `bfloat16`, we can switch `jax.nn.dot_product_attention` to cuDNN's fused FlashAttention kernel (`implementation='cudnn'`), which tiles attention in on-chip SRAM without writing the $T \times T$ score matrix to DRAM:

@!fast-transformer-fused-attention

- **Combined `bf16` + FlashAttention gain ($\approx 1.96\times$):** Eliminating the off-chip attention matrix traffic speeds up the step by another **$1.35\times$**, bringing the cumulative precision + attention speedup to nearly $2\times$.
:::

::: {.slide title="Rung 3: Growing the Minibatch to 512 Windows"}
[Amortizing Fixed Work]{.kicker}

Because the GPU is already $100\%$ busy, growing the batch $8\times$ from $64 \to 512$ windows ($65{,}536$ tokens/step) can only amortize device work that scales with parameter count $N = 19\text{M}$ rather than token count $n$:

- **Fixed per-step memory traffic ($40\text{ B/param} = 0.76\text{ GB}$):** Fused AdamW reads/writes $28\text{ B/param}$ (`fp32` weight, grad, two moments) and mixed precision casts weights/gradients between `fp32` and `bf16` ($12\text{ B/param}$), taking $\sim 0.75\text{ ms}$ at $1\text{ TB/s}$ ($\sim 6\%$ of a $12.5\text{ ms}$ batch-64 step).
- **Prediction ($\le 5\%$ gain):** Amortizing $7/8$ of $0.75\text{ ms}$ predicts a few percent throughput gain, while lengthening the step to $\sim 95\text{ ms}$ so multi-GPU allreduce can scale.

@!fast-transformer-rungs-each-one-measured-3
:::

::: {.slide title="Profile at Batch 512: 60% Matmuls, 40% Memory Traffic"}
[Third Profile]{.kicker}

At batch size $512$, activation memory consumes $\sim 8\text{ GiB}$ (leaving two-thirds of the $24\text{ GiB}$ GPU free), and the profile shows the step's final single-GPU balance:

@!fast-transformer-larger-minibatches-1

- **Where the time goes:** The GPU is busy for $100\%$ of the $\sim 91\text{--}100\text{ ms}$ step; matrix products and fused attention take **$\sim 57\text{--}60\%$** of the step, while fused bandwidth-bound kernels (RMSNorm, SwiGLU, RoPE, and `fp32`/`bf16` casts) take the remaining **$\sim 40\%$**.
:::

::: {.slide title="Branch Test: Predicting Activation Checkpointing Cost"}
[Checkpointing Prediction]{.kicker}

Even though batch size $512$ easily fits in $24\text{ GiB}$, what happens if we turn on per-block activation checkpointing (`kept=False` branch)? We can bound the throughput retention in two ways:

1. **By FLOP count:** Recomputing each block's forward pass during backpropagation increases block arithmetic from $6N \to 8N\text{ FLOPs/token}$, predicting $6/8 = \mathbf{0.75\times}$ throughput.
2. **By standalone forward-pass time ($t_{\textrm{fwd}}$):** Timing the forward pass alone predicts $\frac{1}{1 + t_{\textrm{fwd}} / t_{\textrm{step}}} \approx \mathbf{0.76\text{--}0.77\times}$ throughput.

@!fast-transformer-activation-checkpointing-2
:::

::: {.slide title="Measuring Checkpointing: Fused Remat Beats the Floor"}
[Checkpointing Measured]{.kicker}

Wrapping all 6 Transformer blocks in `Checkpointed` cuts activation memory by two-thirds ($7\text{--}9\text{ GiB} \to 2.6\text{--}3.2\text{ GiB}$) while retaining **$0.79\text{--}0.85\times$** of throughput—beating both the $0.75\times$ FLOP bound and the $0.77\times$ forward-timing bound:

@!fast-transformer-rungs-each-one-measured-4

- **Why recomputation costs less than a standalone forward pass:** While the block matrix products must be re-executed in full ($+14\text{ ms}$), the compiler fuses the recomputed elementwise kernels (SwiGLU, RoPE, norms) directly into the backward kernels that consume them, saving most of their $13\text{ ms}$ of standalone DRAM traffic.
:::

::: {.slide title="Cumulative Single-GPU Waterfall: ~2× Speedup, 45–50% MFU"}
[Single-GPU Results]{.kicker}

Summarizing all single-GPU rungs side by side converts token throughput into achieved `TFLOP/s` ($6N\text{ FLOPs/token}$) and **Model FLOPs Utilization** ($\textrm{MFU} = \textrm{achieved FLOP/s} / 165.2\text{ TFLOP/s}$):

@!fast-transformer-the-waterfall

- **Takeaway:** Cumulative single-GPU optimizations **double throughput ($\sim 2.0\text{--}2.1\times$)**, reaching **$\sim 45\text{--}50\%$ MFU** on the RTX 4090—driven overwhelmingly by `bfloat16` Tensor Cores and operator/attention fusion.
:::

::: {.slide title="Predicting Multi-GPU Scaling From Step and Link Times"}
[Data-Parallel Cost Model]{.kicker}

::: {.cols .vc}
::: {.col}
$$\textrm{Speedup}(k) \;\gtrsim\; \frac{k\, t_{\textrm{step}}(1)}{t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k)}$$

- **Fixed $76\text{ MB}$ allreduce cost:** Over host-staged PCIe ($\beta_{\textrm{link}} \approx 0.42\text{ ns/B}$), syncing $19\text{M}$ `fp32` grads takes $31\text{ ms}$ ($k=2$) and $47\text{ ms}$ ($k=4$).
- **Two opposite predictions:** At batch $64$ ($t_{\textrm{step}}(1) \approx 22\text{ ms} < 31\text{ ms}$), $k=2$ **slows down** ($\sim 0.85\times$); at batch $512$ ($t_{\textrm{step}}(1) \approx 95\text{ ms}$), $k=4$ achieves **$\approx 2.6\text{--}2.7\times$**.
:::

::: {.col .fig}
@fig:ch13-perf-multigpu-batch-crossover
:::
:::
:::

::: {.slide title="Measured Scaling: Batch 64 Loses, Batch 512 Hits 3×"}
[Multi-GPU Measured]{.kicker}

Running production data parallelism on $k = 1, 2, 4$ GPUs tests both predictions of opposite sign:

@!fast-transformer-rungs-each-one-measured-8

::: {.d2l-note .rule}
**Both predictions hold:** The standalone $76\text{ MB}$ allreduce measures $31\text{ ms}$ ($k=2$) and $48\text{ ms}$ ($k=4$). At batch $64$, adding a second GPU slows training down ($\sim 0.85\text{--}0.95\times$); at batch $512$ in `bfloat16`, $4$ GPUs achieve **$\sim 2.9\text{--}3.2\times$ single-GPU throughput** ($\sim 1.9\text{--}2.3\text{M tokens/s}$, or **$6\text{--}7\times$ our single-GPU baseline**), with backward-pass bucket overlap providing the margin above the no-overlap floor.
:::
:::

::: {.slide title="End-to-End Training Gate: Matching Baseline Convergence"}
[Numerical Verification]{.kicker}

A $2\times$ faster single-GPU step is only valid if it learns the same representation as the `float32` baseline. We train two identically initialized models (one using `baseline` and one using `final`, `+ batch 512` in `bf16`) on the exact same $200$ minibatches of $512$ windows ($\sim 13\text{M}$ tokens, with a 50-step linear LR warm-up):

@!fast-transformer-training-check

- **Passed gate:** Both curves start at $\ln 256 \approx 5.55$, track within $0.05\text{ nats/byte}$ across all $200$ steps, and reach the same validation loss ($1.5\text{--}1.6$, well below the $2.45$ bigram threshold).
:::

::: {.slide title="Throughput vs. Time-to-Target-Loss"}
[Optimization Caveat]{.kicker}

::: {.cols .vc}
::: {.col}
- **The critical batch size ($b_{\textrm{noise}}$):** Training at batch size $b$ requires $E(b) = E_{\min}(1 + b/b_{\textrm{noise}})$ total tokens to reach a target validation loss.
- **Diminishing returns past $b_{\textrm{noise}}$:** When $b \gg b_{\textrm{noise}}$, cleaner gradient estimates make nearly the same parameter progress per step as smaller batches, needing more total tokens.
- **Systems rule:** Batch-preserving speedups (`compile`, `bf16`, FlashAttention, strong-scaling DP) always cut time-to-loss; batch-growing changes help only while $b \lesssim b_{\textrm{noise}}$.
:::

::: {.col .fig .big}
@fig:ch13-perf-critical-batch-noise
:::
:::
:::

::: {.slide title="Millisecond Accounting of the Final Step"}
[Final Accounting]{.kicker}

Normalizing our kernel profiles to milliseconds per $8{,}192$ tokens accounts for every millisecond of the baseline and final steps against ideal Tensor Core peak time ($5.7\text{ ms}$ in `bf16`, $11.3\text{ ms}$ in `tf32`):

@!fast-transformer-account

- **Why the final step runs at $\sim 45\text{--}50\%$ MFU ($11.4\text{--}12.5\text{ ms}$ vs. $5.7\text{ ms}$ ideal):**
  - **Matrix products ($\sim 7\text{ ms}$):** Run at **$> 80\%$ of the $5.7\text{ ms}$ hardware peak** while performing $> 99\%$ of all arithmetic.
  - **Bandwidth-bound kernels ($\sim 5\text{ ms}$):** Norms, SwiGLU, RoPE, casts, and AdamW perform $< 1\%$ of the FLOPs but consume **$\sim 40\%$ of the step time** moving bytes through DRAM.
:::

::: {.slide title="Concept Check: Amdahl Bounds, Scaling, and Next Steps"}
[Check Your Understanding]{.kicker}

Use the final millisecond accounting ($7\text{ ms}$ in matrix products near `bf16` peak + $5\text{ ms}$ in bandwidth-bound non-matmul kernels $= 12\text{ ms}$ per $8{,}192$ tokens) to answer three questions:

1. Suppose we switch the matrix products from `bfloat16` ($165\text{ TFLOP/s}$) to `fp8` ($330\text{ TFLOP/s}$), halving matrix-product time from $7\text{ ms} \to 3.5\text{ ms}$ while leaving the $5\text{ ms}$ of non-matmul kernels unchanged. What is the step speedup?
2. Conversely, suppose custom hand-written mega-kernels fuse away *all* $5\text{ ms}$ of non-matmul memory traffic while `bfloat16` matrix products stay at $7\text{ ms}$. What is the step speedup?
3. What would it take to achieve another **$2\times$ single-GPU speedup** ($12\text{ ms} \to 6\text{ ms}$)?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **`fp8` matmuls alone:** Step time drops from $12\text{ ms} \to 3.5 + 5.0 = 8.5\text{ ms}$, a speedup of **$12 / 8.5 \approx 1.41\times$** (far below $2\times$).
2. **Eliminating all non-matmul kernels alone:** Step time drops from $12\text{ ms} \to 7.0\text{ ms}$, a speedup of **$12 / 7.0 \approx 1.71\times$** (also below $2\times$).
3. **Both are required simultaneously:** Because the $12\text{ ms}$ step is split $58\%$ compute-bound ($7\text{ ms}$) and $42\%$ bandwidth-bound ($5\text{ ms}$), halving it to $6\text{ ms}$ requires **both** faster `fp8` matrix products ($7 \to 3.5\text{ ms}$) **and** narrower/fused activation memory traffic ($5 \to 2.5\text{ ms}$).
:::
:::

::: {.slide title="Summary: Step-by-Step Transformer Optimization" only="pytorch"}
[Key Takeaways]{.kicker}

| Optimization Rung | Bottleneck Targeted | Amdahl Prediction | Measured Gain | Cumulative Speedup / MFU |
| :--- | :--- | :---: | :---: | :---: |
| **0. Baseline (`fp32`/`tf32`)** | Reference ($26\text{ ms}$ / $8192$ tok) | — | $1.00\times$ | $1.00\times$ ($\sim 22\%$ MFU) |
| **1. `torch.compile`** | Non-matmul traffic & host dispatch | $\le 1.6\times$ | $1.16\times$ | $1.16\times$ ($\sim 25\%$ MFU) |
| **2. `bf16` (FlashAttention already on)** | Tensor Core GEMMs | $1.5\text{--}2.0\times$ | $1.74\times$ | $\approx 2.0\times$ ($\sim 44\%$ MFU) |
| **3. Batch $64 \to 512$** | Fixed $40\text{ B/param}$ step work ($0.75\text{ ms}$) | $\approx 1.05\times$ | $1.02\times$ | **$\approx 2.1\times$ ($\sim 45\%$ MFU)** |
| **Branch: Checkpointing** | Activation memory ($8.6 \to 3.2\text{ GiB}$) | $\ge 0.77\times$ | $0.85\times$ | Memory saver (not kept) |
| **4. 4-GPU Data Parallel** | Single-GPU ceiling ($48\text{ ms}$ allreduce) | $\approx 2.7\times$ (batch 512) | **$\approx 2.9\times$** over 1 GPU | **$\approx 6\times$ over 1-GPU baseline** |
:::

::: {.slide title="Summary: Step-by-Step Transformer Optimization" only="jax"}
[Key Takeaways]{.kicker}

| Optimization Rung | Bottleneck Targeted | Amdahl Prediction | Measured Gain | Cumulative Speedup / MFU |
| :--- | :--- | :---: | :---: | :---: |
| **0. Baseline (`fp32`/`tf32`)** | Reference ($24\text{ ms}$ / $8192$ tok) | — | $1.00\times$ | $1.00\times$ ($\sim 23\%$ MFU) |
| **1. `jit_partial`** | Host dispatch | $1.02\times$ | $1.01\times$ | $1.01\times$ ($\sim 24\%$ MFU) |
| **2. `bf16` + fused attention** | Tensor Core GEMMs & $T \times T$ attn | $1.5\text{--}2.0\times$ | $1.45\times$, then $1.35\times$ | $\approx 2.0\times$ ($\sim 47\%$ MFU) |
| **3. Batch $64 \to 512$** | Fixed $40\text{ B/param}$ step work ($0.75\text{ ms}$) | $\approx 1.05\times$ | $1.06\times$ | **$\approx 2.1\times$ ($\sim 50\%$ MFU)** |
| **Branch: Checkpointing** | Activation memory ($7.4 \to 2.6\text{ GiB}$) | $\ge 0.76\times$ | $0.79\times$ | Memory saver (not kept) |
| **4. 4-GPU Data Parallel** | Single-GPU ceiling ($48\text{ ms}$ allreduce) | $\approx 2.6\times$ (batch 512) | **$\approx 3.2\times$** over 1 GPU | **$\approx 7\times$ over 1-GPU baseline** |
:::
