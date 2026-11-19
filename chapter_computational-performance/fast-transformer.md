# Case Study: Optimizing a Transformer
:label:`sec_fast_transformer`

The earlier sections of this chapter took the causes of a slow training
step one at a time and found the change that removes each: host overhead,
the memory traffic of small operations, wide number formats, and the limits
of a single GPU. They also showed that recomputing activations saves
memory, at the cost of time, when a step does not fit. A real model
suffers from several of these causes at once, and removing one changes how
much the others matter. How much faster does its training step become when
we apply all the changes in turn?

We answer the question for the GPT of :numref:`sec_gpt`, which has
19 million parameters when we double its width to 512. One training step
on 64 windows of 128 bytes performs about
$6 \times 1.9 \times 10^{7} \times 8{,}192 \approx 9 \times 10^{11}$
floating-point operations: two per parameter and token in the forward pass
and four in the backward pass. The tensor cores of an RTX 4090 execute
$1.65 \times 10^{14}$ of them per second in bf16, so the arithmetic of a
step needs less than 6 ms. A plain training step, in fp32 with an AdamW
update, takes 25 to 30 ms. Three quarters or more of the card's capacity
go unused, and every training run takes several times longer than its
arithmetic requires.

We follow the method of :numref:`sec_perf_model`. We measure the step,
find the resource that limits it, change one thing, check that the model
still computes the same function, and measure again. Each change is
measured on top of the previous ones, because each moves the bottleneck and
so changes what the next one can achieve. For data parallelism we predict
the gain from a cost model before we measure it, and at the end a short
training run checks that the fastest step still trains the model.

```{.python .input #fast-transformer-setup}
%%tab jax
# Let XLA overlap collectives with computation. JAX reads XLA_FLAGS once,
# when it starts, so this cell runs before the imports. The notebook also
# runs with --xla_gpu_enable_triton_gemm=false in XLA_FLAGS, so that XLA
# calls cuBLAS for the matrix products (the Triton kernels it would generate
# run the products of the large minibatches more slowly on the RTX 4090),
# and with NCCL_LOCAL_REGISTER=0: NCCL's buffer registration fails on GPUs
# without peer-to-peer access and would print a warning at every collective.
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
import re
from torch.utils.checkpoint import checkpoint
```

```{.python .input #fast-transformer-case-study-optimizing-a-transformer-2}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 products on tf32 cores
# The notebook runs with TEARDOWN_CUPTI=1 in the environment, so that the
# profiler of the cells below detaches from the GPU when a profile ends;
# without it, every later kernel launch of the process stays slower
```

## Experimental Setup
:label:`subsec_ft-subject`

Every change in this section is judged by the same measurement, so the
model, the data, and the way we measure throughput stay fixed throughout.

### Model and Data

We train `d2l.GPT` as :numref:`sec_gpt` defines it: six pre-norm blocks
with eight attention heads each, rotary position embeddings, RMSNorm and
SwiGLU, and an input embedding shared with the output layer. The data are
also those of :numref:`sec_gpt`: windows of 128 bytes of WikiText-103 at
random offsets, with the 256 byte values as the vocabulary
(`d2l.WikiText103`, :numref:`sec_test_models`).

We change one setting: the width, from 256 to 512. The reason is
arithmetic intensity, the number of FLOPs that an operation performs per byte it
moves (:numref:`sec_perf_model`). A product of an $n \times d$ matrix of
activations with a $d \times d'$ weight matrix performs $2ndd'$ FLOPs and
moves $b(nd + dd' + nd')$ bytes, with $b$ bytes per number. For the
$n = 8{,}192$ tokens of a step at width 512, the products of a block reach
between 250 and 370 FLOPs per byte in bf16. The ridge point of the
RTX 4090, the intensity above which an operation is limited by arithmetic
rather than by memory bandwidth, lies at 164 FLOPs per byte. The products
of the wider GPT are therefore limited by arithmetic, so a faster
arithmetic format shortens them directly. At width 256 the same products
lie around the ridge point; an exercise studies that model.

The training text has 270 million bytes, one per token, so it fits on the
GPU as an array of 270 MB. The next cell places it there and defines
`windows`, which draws every minibatch on the GPU from random offsets; no
step then waits for data from the host. The cell also builds the model.

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

The model has 19 million parameters, nearly all of them in the weight
matrices of its six blocks.

### Throughput Measurement

We measure speed as training throughput, in tokens per second, end to
end: the time of a step includes drawing the minibatch, the forward and
backward passes, and the optimizer update. Every measurement in this
section follows the same rules.

| setting | value |
|:--|:--|
| hardware | one RTX 4090 of the four in our machine; data parallelism uses two and four |
| model | `d2l.GPT`: 6 blocks, width 512, 8 heads, byte vocabulary, context 128 |
| data | windows of WikiText-103 at random offsets, drawn on the GPU, new for every step |
| optimizer | AdamW with learning rate $10^{-3}$ and weight decay 0.01 |
| precision | fp32 with matrix products on the tf32 tensor cores, until bf16 is introduced |
| timing | `d2l.Benchmark`: 20 warm-up steps, then 50 steps between two device synchronizations |
| result | tokens per second, and its ratio to the previous configuration |

`tokens_per_second` applies these rules with `d2l.Benchmark` of
:numref:`sec_perf_model`, whose warm-up calls absorb one-time costs such
as compilation and memory allocation. `record` stores each result and
prints its ratio to the previous one.

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

<!-- WRITER: `tokens_per_second` now times three windows of 50 steps after
the warm-up and returns their median and their spread (largest minus
smallest, relative to the median); `record` prints the spread beside each
ratio, and `record(..., kept=False)` stores a configuration that the later
ones do not build on (checkpointing) apart from the cumulative sequence.
The sentence "One timed window of 50 steps varies by a few percent" must
read the printed spreads instead (two dev runs on an idle GPU: 0.0 to 0.4%
for every configuration after the first; the baselines 2.6 to 4.5% in
PyTorch and 9.5 to 11.5% in JAX, whose host work sits near the margin),
and the measurement-rules table's timing row should say three windows of 50
steps, median. -->

One timed window of 50 steps varies by a few percent from run to run, so
we read ratios to one decimal place.

## A Single GPU
:label:`subsec_ft-rungs`

We first make the step fast on one GPU, one change at a time, starting
from the step as :numref:`sec_gpt` wrote it.

### Baseline
:label:`subsec_ft-baseline`

The baseline is the training step of :numref:`sec_gpt` with plain AdamW:
a forward pass, the cross-entropy loss, a backward pass, and the update.

:begin_tab:`pytorch`
The step runs eagerly. Python issues every operation of the forward pass
and autograd every operation of the backward pass, each as one or more
kernel launches. The optimizer, created with `fused=True`, updates all
weights and moment estimates in a few kernels instead of several per
weight tensor. `make_step` packages the step as a function of a minibatch;
its arguments `forward` and `bf16` bring in the changes of the next
subsections. The notebook runs with the environment variable
`TEARDOWN_CUPTI=1`, which detaches the profiler of the next cells from the
GPU when a profile ends; without it, every later kernel launch of the
process stays slower.
<!-- WRITER: the TEARDOWN_CUPTI sentences are now a code comment in the
cell after the extras (`fast-transformer-case-study-optimizing-a-transformer-2`);
cut them here (FT-3) or move them to the one "Running this notebook" note. -->
:end_tab:

:begin_tab:`jax`
The step is one function, `train_step`, compiled with `nnx.jit` as in the
training loop `d2l.train_steps` of :numref:`sec_test_models`. It receives the
model and the optimizer as arguments and returns the loss, which it computes
from fp32 logits. The notebook runs with `--xla_gpu_enable_triton_gemm=false`
in the environment variable `XLA_FLAGS`, so that XLA calls cuBLAS for the
matrix products: the Triton kernels it would otherwise generate run the
products of the large minibatches more slowly on our GPU.
<!-- WRITER: the Triton GEMM flag (and NCCL_LOCAL_REGISTER=0) are now a
comment in the setup cell `fast-transformer-setup`; cut them here (S5) or
move them to the one "Running this notebook" note. -->
:end_tab:

```{.python .input #fast-transformer-rung-0-baseline-profiled}
%%tab pytorch
def make_step(model, forward=None, bf16=False):  #@save
    """An AdamW training step of `model` on a minibatch (X, Y); `forward`
    replaces the model's forward pass, `bf16` turns on autocast."""
    forward = model if forward is None else forward
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=0.01, fused=True)
    def step(X, Y):
        with torch.autocast('cuda', torch.bfloat16, enabled=bf16):
            loss = F.cross_entropy(forward(X).flatten(0, 1), Y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
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

A profile shows where the time of the step goes. `profile` records five
steps with `d2l.kernel_times` and prints them with `d2l.show_profile`,
both from :numref:`subsec_perf-profiler`: the time during which the device
runs kernels against the wall-clock time of a step, which follows from the
measured throughput, the share of the matrix products, counting fused
attention kernels among them, and the kernels that take the most time.

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

<!-- WRITER: `profile` now takes no throughput argument: it profiles the
configuration that `record` stored last, prints against its measured
(median) throughput, and keeps the kernel times in `profiles`, which the
account at the end of the section reads (`subsec_ft-account`). -->

:begin_tab:`pytorch`
The device runs kernels for nearly the whole step, so the host keeps up
with it. Matrix products take a little under two thirds of
the device time. The rest is spread over many short kernels: the
elementwise operations of SwiGLU, of the rotary embeddings, and of the
residual connections, and the normalizations. Each of them reads its
operands from device memory and writes its result back, so together they
are limited by memory bandwidth (:numref:`subsec_comp-fusion`).
:end_tab:

:begin_tab:`jax`
The device runs kernels for nine tenths of the step or more and waits for
the host in the rest. Matrix products take a little under two thirds of the
device time. The step is a single compiled program in which XLA has already
fused the elementwise operations into the kernels around them, so few short
kernels remain.
:end_tab:

### Host Overhead and Small Kernels

A profile separates two regimes of :numref:`sec_perf_model`. Gaps in
which the device waits for the host mark the overhead-bound regime: the
step runs at the pace of the Python interpreter, and a faster GPU would
not shorten it. Device time spread over many short elementwise kernels
marks the bandwidth-bound regime: each kernel moves its operands through
device memory for a few FLOPs. The remedies differ. Overhead calls for
less work on the host per step, and short kernels call for fusion, which
merges chains of them into one kernel that reads its inputs once and
writes its result once.

:begin_tab:`pytorch`
Our step is in the second regime: the device hardly waits, but much of its
time goes to short kernels.
`torch.compile`
(:numref:`sec_compilation`) captures the forward pass of the model as a
graph and generates kernels for it and for its backward pass, each of
which fuses a chain of elementwise operations. It thus replaces several
hundred operations that Python and autograd dispatch one at a time by far
fewer kernels. The matrix products remain calls to the library, and the
optimizer update stays as it is. With `mode='reduce-overhead'`, the
compiled kernels of the forward and of the backward pass are recorded as
two CUDA graphs and replayed with one launch each
(:numref:`subsec_comp-overhead`), so the host no longer issues them one by
one. With `dynamic=False`, every minibatch shape gets kernels specialized
to it. The first call with a new shape compiles, which takes seconds, and
the warm-up of `tokens_per_second` absorbs it. Before timing anything, we
check that the compiled model computes the same logits as the original
one, measured by their relative difference.
:end_tab:

:begin_tab:`jax`
Our step is close to the first regime: XLA has fused the short kernels, but
the host does much work for every step, and the device waits whenever the
host falls behind. `nnx.jit` lets us pass the model and the optimizer,
Python objects that hold arrays, to a compiled function. On every call it
walks through both objects to collect their arrays, hands the arrays to the
compiled program, and writes the results back into the objects. We time this
work alone, with compiled functions that do nothing, called in three ways:
through `nnx.jit`; through `nnx.jit` with `graph=False`, which treats the
model and the optimizer as trees of arrays instead of walking them as Python
objects; and through `nnx.jit_partial` with `graph=False`, which in addition
binds them to the function once, so that later calls pass only the
minibatch.
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
<!-- WRITER: the check now asserts a relative difference below 1e-4 (dev
and store: about 5e-06): "five significant digits or more", not "three or
more" (FT-7). -->
The logits agree to three or more significant digits. They are not
identical, because the fused fp32 kernels round in a different order than
the separate ones, and the tf32 products carry the difference forward.
Throughput rises by about a tenth.
<!-- WRITER: two dev runs: baseline busy 93 to 94% (store 97%), device
time 23.8 → 21.6 ms per step (fusion), busy 100% after (replay); "+ compile"
1.17 and 1.18 (store 1.13). -->
:end_tab:

:begin_tab:`jax`
A call through `nnx.jit` takes about 8 ms, a third of the device time of a
step. The host prepares the next step while the device runs the current one,
so this work stays hidden only as long as the host finishes it in time. A
CPU that also serves other work falls behind, and the idle part of the
profile measures by how much. Treating the arguments as trees removes most
of the 8 ms, and binding them once removes nearly all of the rest.
Binding changes how the step is called, not what it computes. The next cell
checks this before the timing: one step through `nnx.jit` and one through
the bound step, from equal copies of the model and the optimizer, must give
equal losses and leave equal weights.
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
<!-- WRITER: the gain of binding depends on how busy the host is. Two dev
runs on an idle GPU: baseline busy 95% and 99% of the step, "+ jit_partial"
1.04 and 1.01 (store, under load: busy 85%, 1.17). The prediction (the
gain is at most the idle share of the first profile) held in all three; the
prose must state it as that bound, read from the captured profile, not as
"a little more than a tenth" (FT-4, FT-6). The no-op call through `nnx.jit`
measured 8.6 to 8.7 ms (store 9.55). -->
Throughput rises by as much as the idle time of the first profile
allowed, a little more than a tenth when the device waited a tenth of the
step.
:end_tab:

A second profile shows where the bottleneck has moved:

```{.python .input #fast-transformer-rungs-each-one-measured-5}
%%tab pytorch, jax
profile(step, windows(corpus, 64))
```

The device is now busy for nearly the whole step, and the matrix products
take the largest share of its time. The step is limited by the device,
mostly by the arithmetic of the matrix products.

:begin_tab:`pytorch`
Compared with the first profile, the device time per step has fallen and
the gaps between kernels have closed. The matrix products take as long as
before, while the elementwise kernels have merged into fewer generated
ones, whose names begin with `triton_`.
:end_tab:

:begin_tab:`jax`
The device time is the same as in the first profile, since binding
changed only the work of the host, but the gaps in which the device waited
are gone.
:end_tab:

### Reduced Precision
:label:`subsec_ft-precision`

When the device is busy and the matrix products take most of its time,
the next change should make the products faster. In bf16 the tensor cores
of the RTX 4090 run at 165 TFLOP/s, twice their tf32 rate, and every
activation that is stored or read takes half the bytes
(:numref:`sec_memory_precision`). We apply the mixed-precision recipe of
:numref:`subsec_mp-mixed`: the matrix products of the forward and the
backward pass, those of the attention included, run in bf16, and so do
the elementwise operations between them, such as SwiGLU. The weights,
their gradients, and Adam's moment estimates stay in fp32, and so do the
normalizations, the loss, and the residual stream that connects the
blocks. The stream is kept in fp32 because each block adds to it an update
that can be much smaller than the stream itself, and bf16, with 8
significant bits, would round away an update a few hundred times smaller
than the entry it is added to. Since bf16 has the exponent range of fp32,
the loss needs no scaling. The check before the timing compares the bf16
logits with the fp32 logits of the same weights; they should agree to the
precision of bf16, two to three significant digits.

:begin_tab:`pytorch`
`torch.autocast` applies this recipe operation by operation
(:numref:`sec_numerics`): it runs the matrix products and the attention in
bf16 and computes reductions such as the softmax of the loss in fp32. The
embedding and the normalizations receive no cast, so the stream stays in
fp32. The backward pass follows the dtypes of the forward pass.
:end_tab:

:begin_tab:`jax`
A Flax layer computes in the dtype stored in its attribute `dtype` and
keeps its parameters in `param_dtype` (:numref:`sec_numerics`).
`d2l.to_bf16` of :numref:`subsec_mp-mixed` sets `dtype` to bf16 in the
linear layers only. The embedding and the normalizations
compute in fp32, so the stream stays in fp32, and the output layer, which
reuses the embedding table and takes under 1% of the FLOPs, computes in
fp32 as well. The gradients arrive in fp32, because the derivative of a
cast to bf16 is a cast back.
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

<!-- WRITER: the bf16 check now asserts a relative difference below 0.01
(it was 0.05; FT-7). In the JAX tab this row is now `d2l.GPT` under
`d2l.to_bf16` with XLA's attention (FT-8, option a); the fused attention
is the next row, "+ fused attention", after the moved FusedGPT paragraph.
Two dev runs: "+ bf16" 1.45 and 1.44, "+ fused attention" 1.36 and 1.36
(product 1.97, the store's combined row 1.96); logits 1.7e-03 to 1.9e-03
in both rows; PyTorch "+ bf16" 1.74 twice, logits 1.8e-03 to 2.0e-03. -->

The logits agree to within one percent.

:begin_tab:`pytorch`
Throughput rises by about 1.7 times. That is less than the factor of two
of the tensor cores, because not every part of the step halves its work:
the normalizations and the residual stream still move fp32 values, the
casts between the two formats add traffic of their own, and the optimizer
update still reads and writes fp32 weights and moment estimates.
:end_tab:

:begin_tab:`jax`
<!-- WRITER: the paragraph below described the old combined row; "+ bf16"
is now bf16 alone with XLA's attention, and the fused attention follows
as its own row. Rewrite from the measured split (report). The FusedGPT
paragraph after it was moved here from before the bf16 cell; its "Second,
`FusedGPT` returns its logits in fp32" is gone: `d2l.GPT` under
`d2l.to_bf16` already returns fp32 logits (the output layer reuses the fp32
embedding table), so the override was a no-op and was removed. -->
Throughput about doubles. Faster matrix products alone could not do this,
since they took a little under two thirds of the step. The fused attention
no longer stores the weights of all query–key pairs, and the activations
between the linear layers, those of SwiGLU among them, take half the bytes.
The normalizations, the residual stream, and the optimizer update still
move fp32 values.

`FusedGPT` changes one detail of `d2l.GPT`. `d2l.GPT` leaves
the choice of attention kernel to `jax.nn.dot_product_attention`, whose
default is XLA's own implementation, and that implementation builds the
attention weights of all query–key pairs in device memory
(:numref:`sec_attention-at-scale`); `d2l.GPT` only keeps them out of the
backward pass, which computes them again. `FusedGPT` overrides `attend`
to select cuDNN's fused kernel instead, which accepts only 16-bit inputs,
so the fused attention can follow only once the products run in bf16.
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
<!-- WRITER: one sentence on the measured ratio of "+ fused attention"
(report); the two JAX ratios multiply to about the old combined 1.96. -->
:end_tab:

### Larger Minibatches

Some costs of a step do not grow with the minibatch. The optimizer update
reads and writes every weight and both moment estimates once per step,
however many tokens the step processes, and a kernel launch costs the same
for 64 windows as for 512. A larger minibatch divides these fixed costs
among more tokens, and its larger matrix products keep the tensor cores
busier. The activations grow in proportion to the minibatch, so the step
needs more memory. We raise the minibatch from 64 to 512 windows, or
65,536 tokens per step, and record the memory that the step needs beyond
the weights and the optimizer state.

:begin_tab:`pytorch`
`torch.cuda.max_memory_allocated` reports the most memory allocated since
the last `reset_peak_memory_stats`; we subtract what was allocated before
the steps (:numref:`subsec_mp-measuring`).
:end_tab:

:begin_tab:`jax`
The compiled step reports its memory plan through `memory_analysis()`;
`temp_size_in_bytes` counts the memory that the step needs beyond its
inputs and outputs (:numref:`subsec_mp-measuring`).
:end_tab:

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

The step needs well under half of the card's 24 GiB.

:begin_tab:`pytorch`
Throughput stays the same. With the update fused and the kernels replayed
as CUDA graphs, almost nothing of the step at batch 64 was a fixed cost for
the larger minibatch to divide.
:end_tab:

:begin_tab:`jax`
Throughput rises by less than a tenth. After fusion and binding, little of
the step at batch 64 was a fixed cost for the larger minibatch to divide.
:end_tab:

An exercise separates the fixed cost of a step from its cost per token by
measuring several minibatch sizes. A third profile shows what limits the
step at batch 512:

```{.python .input #fast-transformer-larger-minibatches-1}
%%tab pytorch, jax
profile(step, windows(corpus, 512))
```

The device is busy for the whole step, and the matrix products, those
inside the fused attention included, take half or more of its time.
The largest single kernels are matrix products. Most of the rest goes to
the kernels between them, for the normalizations, SwiGLU, the rotary
embeddings, and the casts between fp32 and bf16, which move bytes through
device memory and leave the tensor cores idle.

:begin_tab:`pytorch`
Matrix products take a little over half of the device time. The kernels
that Inductor generated take most of the rest; the largest of them
computes the backward pass of SwiGLU.
:end_tab:

:begin_tab:`jax`
Matrix products take about three fifths of the device time, and XLA's
fusions take the rest.
:end_tab:

### Activation Checkpointing

Activation checkpointing (:numref:`subsec_mp-checkpointing`) reduces the
memory of a step by recomputing activations: each block keeps only its
input during the forward pass and recomputes everything else during the
backward pass. The step at batch 512 uses less than half of the card's
memory, so memory does not limit it, and our method predicts that
checkpointing can only slow the step down. We measure it to check the prediction, and to see how much
memory it would free for a model that needed it.

:begin_tab:`pytorch`
`Checkpointed` wraps a module so that it runs under `checkpoint` from
`torch.utils.checkpoint`. We give a copy of our model, with its weights,
checkpointed blocks; the rest of its forward pass stays as `d2l.GPT`
defines it. Before the timing we check that it computes the same logits.
:end_tab:

:begin_tab:`jax`
`Checkpointed` wraps a module so that it runs under `nnx.remat`, Flax's
form of `jax.checkpoint`. We give a copy of `model16`, with its weights,
checkpointed blocks; the rest of its forward pass stays as `FusedGPT`
defines it. Before the timing we check that it computes the same logits.
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
torch.cuda.reset_peak_memory_stats()
start = torch.cuda.memory_allocated()
record('+ checkpointing', tokens_per_second(step, windows(corpus, 512)),
       kept=False)
memory = (torch.cuda.max_memory_allocated() - start) / 2**30
print(f'memory beyond weights and optimizer state: {memory:.1f} GiB')
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

<!-- WRITER: checkpointing is now recorded with `kept=False`: its ratio is
still to batch 512, and the cumulative table prints it below a rule as
"(not kept)", outside the running total (FT-12). -->
Checkpointing lowers throughput by a tenth to a quarter and cuts the
memory of the step to less than half. It is the right change for a model
or a minibatch that would not fit otherwise. For this step it is the wrong
one, as the memory of the previous configuration predicted, and the
configurations that follow leave it out.

## Data Parallelism
:label:`subsec_ft-dp`

The fastest single-GPU step still leaves three of the machine's four GPUs
idle. We predict what data parallelism gains on them before we measure it.

### Predicted Speedup

Four GPUs could at best process four times as many tokens per second as
one. In data parallelism (:numref:`sec_multi_gpu`), every GPU holds a copy
of the model and runs the step on a minibatch of its own; before the
update, the GPUs sum their gradients with an allreduce. The cost model of
:numref:`subsec_mg-accounting` puts the time of a step on $k$ GPUs at
$t_{\textrm{compute}}(B/k) + m\alpha + 2N/\beta$: the computation on each
GPU's share of a minibatch of $B$ examples, plus $m$ allreduce calls with
a fixed cost $\alpha$ each, which move the $N$ bytes of the gradients at
a bandwidth $\beta$.

Here the minibatch per GPU stays fixed as GPUs are added, which is called
weak scaling, so the computation term stays the time of a step on one
GPU. For the communication terms we substitute the
measured time of one allreduce of a buffer as large as all gradients, and
obtain

$$t_{\textrm{step}}(k) \approx t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k).$$
:eqlabel:`eq_ft-dp`

A framework that sums the gradients in several buckets makes a few calls
instead of one, which adds a few fixed costs $\alpha$; next to an
allreduce of tens of milliseconds they are small. Since $k$ GPUs process
$k$ minibatches in the time of :eqref:`eq_ft-dp`, throughput grows by the
factor
$k\, t_{\textrm{step}}(1) / (t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k))$.
A second GPU therefore raises throughput only if the allreduce takes less
time than a step.

The gradients of our GPT occupy 76 MB in fp32. Our GPUs have no direct
peer-to-peer path, so the collective library moves every byte through host
memory (:numref:`sec_multi_gpu_concise`), and an allreduce of 76 MB takes
tens of milliseconds. That is as long as a compiled fp32 step at batch 64
or longer, but only a fraction of a bf16 step at batch 512. The cost
model therefore predicts no gain from a second GPU for the first
configuration and a substantial one for the second.

The model leaves out one effect. A framework can start summing the
gradients of the last layers while the backward pass still computes those
of the first (:numref:`fig_ddp_overlap`). The prediction is therefore a
floor, and the distance of a measurement above it shows how much of the
allreduce was hidden behind computation.

### Measured Speedup

We now test the prediction on two and four GPUs.
`data_parallel(k, bf16, batch_size)` trains on $k$ GPUs with the given
precision and minibatch size per GPU. It returns the throughput, together
with the time of one allreduce of a buffer as large as all gradients on
the same GPUs. Both configurations run the step without its host
overhead, one in fp32 at batch 64 and one in bf16 at batch 512.

:begin_tab:`pytorch`
`DistributedDataParallel` runs one process per GPU
(:numref:`sec_multi_gpu_concise`). The step therefore goes into a script,
which `d2l.torchrun` of :numref:`sec_multi_gpu_concise` starts on $k$
GPUs. The script joins the process group with `d2l.init_distributed`,
draws its minibatches with `windows` and builds the compiled step with
`make_step`, both saved from this section, and wraps the model in
`DistributedDataParallel`, which sums the gradients in buckets while the
backward pass runs. It compiles without CUDA graphs: both configurations
keep the device busy long enough to hide the work of the host. Its first
process prints the throughput and the time of one allreduce. Every
process draws its own windows. We run the script on one GPU as well, so
that the single-GPU step is measured under the same conditions. The
script's first process runs on GPU 0, where the CUDA graphs of the
notebook's own compiled steps still hold memory, so the notebook releases
that memory before the first launch.
<!-- WRITER: the release of the CUDA-graph memory and `compile_threads` are
code comments in the cell below; cut the last sentence here (FT-17, S5).
The cell is unchanged. Before it, state the prediction of the allreduce
from beta_link of sec_multi_gpu_concise: S/beta_link for 76 MB at about
2.5 GB/s, about 31 ms at k = 2, and 1.5 S/beta_link, about 46 ms at k = 4
(store: 27 to 31 and 42 to 49 ms; FT-14). -->
:end_tab:

:begin_tab:`jax`
JAX drives all GPUs from one process (:numref:`sec_multi_gpu_concise`).
`data_parallel` builds a mesh of $k$ GPUs, places a copy of the model, the
optimizer state, and the training text on every GPU, and draws each
minibatch with its windows split across the GPUs. The unchanged step then
compiles into a program that computes the gradient of each share on its
own GPU and sums the gradients with an allreduce that the compiler
inserts; the latency-hiding scheduler of the first cell lets these
allreduces run while the backward pass still computes
(:numref:`sec_multi_gpu_concise`). The mesh axis has the type `Auto`, which leaves the placement of
every intermediate result to the compiler; with the default type,
`Explicit`, the embedding lookup would have to state where its output
lives. The time of the allreduce comes from a `psum` over the mesh, as in
:numref:`sec_multi_gpu`. The notebook runs with `NCCL_LOCAL_REGISTER=0`:
NCCL's buffer registration fails on GPUs without peer-to-peer access and
would print a warning at every collective.
<!-- WRITER: NCCL_LOCAL_REGISTER=0 is now a comment in the setup cell
`fast-transformer-setup`; cut the last sentence here (S5). `data_parallel`
takes the median of three windows from `tokens_per_second`; otherwise the
cell is unchanged. -->
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

At batch 64 the allreduce takes about as long as the step or longer, so
two GPUs are predicted to be no faster than one. At batch 512 in bf16 the
step is several times longer, while the allreduce does not depend on the
minibatch. The prediction rises to about three quarters of linear scaling
on two GPUs, and somewhat less on four, where the allreduce moves more
data through each GPU. No measurement lies more than a few percent below
its prediction, and most lie above it, by the part of the allreduce that
ran during the backward pass, which brings them closer to linear
scaling. On this machine, data parallelism therefore
helps only once the step on each GPU is long.

GPUs connected by NVLink
move data more than ten times faster than our path through host memory
(:numref:`tab_gpu_specs`); the allreduce then shrinks by the same factor,
and the cost model predicts most of the linear speedup even at batch 64.

## Results
:label:`subsec_ft-waterfall`

We now put the measurements of all the changes side by side and check that
the fastest step still trains the model.

### Cumulative Comparison

The table collects the single-GPU configurations in the order in which we
applied them. Each ratio in the column `step` belongs to one change,
measured on top of all the changes above it; the column `total` is the
product of the ratios so far. The last column converts throughput into
arithmetic with the count of six FLOPs per parameter and token.

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

<!-- WRITER: the table's `total` now stops at batch 512; checkpointing is
printed below a rule as "(not kept)", with its ratio to batch 512 (FT-12).
The new last column, MFU (model FLOPs utilization), is the TFLOP/s column
over the bf16 peak of 165.2 TFLOP/s, for every row (FR-2; cite
chowdhery2022palm). "The three changes before checkpointing together raise
throughput about twofold or more" becomes "by a factor of two to two and a
half" (dev totals at batch 512: PyTorch 2.04 and 2.07, JAX 2.17 and 2.10;
MFU at batch 512: PyTorch 46%, JAX 49 to 50%). -->

The three changes before checkpointing together raise throughput about
twofold or more. At batch 512 the step performs roughly half of the
165 TFLOP/s that the tensor cores can execute in bf16.
Checkpointing then lowers the throughput again.

Each ratio depends on its place in the sequence. A change that shortens
the work of the device gains nothing while the step waits for the host,
and a change that removes host work gains nothing while the device is
busy. A ranking of techniques that measures each one alone on the same
baseline can therefore mislead.

### Training Check

Throughput says nothing about whether the faster step still trains the
model. We therefore train a fresh model with the final configuration,
bf16 at batch 512, for 300 steps or about 20 million tokens, and then
compute its loss on the validation text with `d2l.lm_perplexity`
(:numref:`sec_test_models`), in nats per byte. Two numbers from
:numref:`sec_gpt` set the scale. A bigram model, which predicts each byte
from the previous one, reaches 2.45 nats per byte; the GPT of width 256
reached about 1.35 after 1,000 steps of 8,192 tokens with a tuned
learning-rate schedule.

```{.python .input #fast-transformer-the-waterfall-1}
%%tab pytorch
learner = d2l.GPT(len(data.vocab), num_hiddens=512).to(device)
reference = d2l.GPT(len(data.vocab), num_hiddens=512).to(device)
reference.load_state_dict(learner.state_dict())  # One initialization
steps = {'baseline': make_step(reference),
         'final': make_step(learner, torch.compile(
             learner, dynamic=False, mode='reduce-overhead'), bf16=True)}
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
reference_optimizer = nnx.Optimizer(reference, adamw, wrt=nnx.Param)
steps = {'baseline': lambda X, Y: jit_step(reference, reference_optimizer,
                                           X, Y),
         'final': nnx.jit_partial(train_step, learner, nnx.Optimizer(
             learner, adamw, wrt=nnx.Param), graph=False)}
batches = windows(corpus, 512)
minibatches = [next(batches) for _ in range(200)]  # The same for both
losses = {name: jnp.stack([step(X, Y) for X, Y in minibatches]).tolist()
          for name, step in steps.items()}
validation = {name: math.log(d2l.lm_perplexity(net, data.val_dataloader()))
              for name, net in (('baseline', reference), ('final', learner))}
```

```{.python .input #fast-transformer-training-check}
%%tab pytorch, jax
checkpoints = (1, 50, 100, 200)
print(f'{"loss at step":14}' + ''.join(f'{i:>7}' for i in checkpoints)
      + f'{"last 20 steps":>18}{"validation":>12}')
for name, curve in losses.items():
    last = curve[-20:]
    print(f'{name:14}' + ''.join(f'{curve[i - 1]:7.2f}' for i in checkpoints)
          + f'{min(last):10.2f} to {max(last):.2f}{validation[name]:12.2f}')
    assert validation[name] < 2.45  # Better than the bigram model
```

<!-- WRITER: the training check is redesigned (FT-19): from one
initialization, the baseline configuration (PyTorch: eager, tf32; JAX:
`nnx.jit`, tf32, XLA attention) and the final one (PyTorch: compiled, bf16;
JAX: bound, bf16, fused attention) train on the same 200 minibatches of
512 windows, drawn once. The shared cell prints the loss of both at steps
1, 50, 100, 200, the range of each over its last 20 steps, and both
validation losses; the bigram threshold stays as an assert only. 300 steps
and "a fresh model with the final configuration" are false; the paragraph
after the cell reads the agreement of the two curves against the printed
ranges. Dev runs (validation baseline/final): PyTorch 2.02/1.97 and
2.19/2.17; JAX 2.05/2.11 and 1.99/2.05; last-20 ranges about 0.07 to 0.13
wide and overlapping in every run. A seed sweep (report) shows that two
runs from one initialization that differ only in numerics (fp32 'highest'
against tf32) already differ by up to 0.05 at step 200, so the two
configurations agree to within what numerics alone produce. The level
after 200 steps varies with the initialization (2.0 to 2.2), so quote no
single value. The deck's training-check slide now
shows `@!fast-transformer-training-check`. -->

The training loss starts near $\log 256 \approx 5.55$, the loss of a
uniform guess over the 256 bytes, and the validation loss ends well below
the bigram level. The faster step trains the model.

### Where the Time Goes
:label:`subsec_ft-account`

<!-- WRITER: new subsection, the chapter's ending (structure.md 3.9,
"Account of the final step"). Predict in words from the third profile
before the cell; the cell prints, per 8,192 tokens, the step, its
arithmetic at the peak of its format (6N per token), the device time of the
matrix products (the same kernel names as `d2l.show_profile`, cuDNN's
fused attention included through "sdpa"), and of all other kernels, for
the baseline (tf32) and for batch 512 (bf16). Dev runs, ms per 8,192
tokens (step, at peak, products, other): PyTorch baseline 25.3 to 25.6,
11.3, 15.1, 8.7; batch 512 12.4, 5.7, 7.0, 5.4. JAX baseline 23.5 to
24.9, 11.3, 14.9 to 15.3, 8.3; batch 512 11.2 to 11.4, 5.7, 6.5 to 6.8,
4.7. The
paragraph that follows ("The fastest configuration still leaves ...") is
the old ending; merge it into this one ending. -->

```{.python .input #fast-transformer-account}
%%tab pytorch, jax
def account(kernels, tokens, speed):
    """Milliseconds per 8,192 tokens of a step of `tokens` tokens that runs
    at `speed` tokens/s: the step, its matrix products, and all its other
    kernels."""
    per_step = 8192 / tokens  # Steps per 8,192 tokens
    # Products are named as in d2l.show_profile: cuBLAS and CUTLASS kernels,
    # PyTorch's fused attention (fmha_cutlass...), cuDNN's (...sdpa...)
    products = per_step * sum(
        ms for name, ms in kernels.items()
        if re.search('gemm|xmma|cutlass|matmul|sdpa', name, re.I))
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

The fastest configuration still leaves about half of the card's bf16
peak unused. What remains lies in the memory-bound kernels between the
matrix products, which the third profile showed, in the efficiency of the
products themselves, and in the width of the numbers. The tensor cores of
the RTX 4090 run fp8 at twice the bf16 rate, but training in fp8 needs a
scaling factor for
each tensor or block of values to keep them within its narrow range. That
recipe, kernels written for a particular model, and training on more GPUs
than one machine holds belong to the Language Models part. The method
stays the same there: measure, find the limit, change one thing, check,
and measure again.

## Summary

A training step can be made about twice as fast or more by a sequence of
changes, each chosen by a measurement. A profile classifies the step: a
device that waits for the host calls for less work on the host per step,
a device busy with many short elementwise kernels calls for fusion, and a
busy device dominated by matrix products calls for faster arithmetic. On
one RTX 4090, fusion or less host work, bf16, and a larger minibatch
together raised the throughput of our 19-million-parameter GPT about
twofold or more. Activation checkpointing lowered it, because
memory did not limit the step. Every change to the computation is checked
for correctness before it is timed, every change is measured on top of
the previous ones, and a short training run confirms that the fastest
step still trains the model. The cost model
$t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k)$ predicts, before any
measurement, that on GPUs connected through host memory data parallelism
gains little for a short step and most of the linear speedup for a long
one.

## Exercises

<!-- WRITER: exercise 1 says each throughput comes from a single window of
50 steps; it is now the median of three windows with the spread printed,
so the exercise extends the body (ten windows, mean and relative standard
deviation, drift design). Exercise 4: the training check now also compares
the final configuration with the baseline on the same minibatches; the
statement that a bigram threshold misses a causal leak still holds, but
"accepts the final configuration when its validation loss lies below that
of a bigram model" must name the new check. -->

1. [code] **Noise floor.** Each throughput in
   :numref:`subsec_ft-waterfall` comes from a single window of 50 steps.
    1. Call `tokens_per_second` ten times in a row for the second
       configuration of the table, and report the mean and the relative
       standard deviation of the ten results.
    1. Estimate the ratio of the third configuration to the second twice:
       from two blocks of ten consecutive windows, and from ten pairs of
       alternating windows. Suppose that the GPU clock drifts slowly
       during the measurement. Explain which of the two designs cancels
       the drift, and compare the two estimates.
    1. Suppose that windows vary independently with the relative standard
       deviation of the first part. Find the number of windows per
       configuration that resolves a change of 3% at two standard errors
       of the ratio. Which ratios in the column `step` does one window per
       configuration resolve?

1. [code] **Model FLOPs utilization.** The GPT of
   :numref:`subsec_ft-subject` has $L = 6$ blocks of width $d = 512$ and
   reads contexts of $T = 128$ tokens. Up to the rounding of the SwiGLU
   width, each block holds $12d^2$ weights in its matrix products, and
   training costs about 6 FLOPs per weight and token, which is the count
   of the table in :numref:`subsec_ft-waterfall`.
    1. The scores $\mathbf{Q}\mathbf{K}^\top$ and the weighted sum of the
       values are matrix products without weights. Count their training
       FLOPs per token and block, taking the backward pass as twice the
       forward pass and counting the masked scores. Show that their ratio
       to the weight FLOPs is $T/(6d)$, and evaluate the ratio for this GPT
       and for a context of 1,024 tokens.
    1. Recompute the last column of the table with these FLOPs included,
       and the fraction of the peak of each configuration's format that it
       reaches: 165 TFLOP/s in bf16 and 83 TFLOP/s in tf32.
    1. A utilization figure can count the FLOPs that the model requires or
       the FLOPs that the hardware executes. Name two configurations or
       kernels of :numref:`subsec_ft-rungs` for which the two counts
       differ, and state the direction of each difference.

1. [code] **Fixed cost of a step.** Model the time of the bf16 step at a
   minibatch of $B$ windows as $t(B) = a + bB$, a fixed cost $a$ plus a
   cost $b$ per window.
    1. Measure the step at $B = 32, 64, 128, 256, 512$, fit $a$ and $b$
       by least squares, and report the fraction of the step at $B = 64$
       that $a$ accounts for.
    1. The AdamW update reads every weight, its gradient, and both moment
       estimates, and writes the weight and both moments, all in fp32.
       Compute the bytes that it moves per step for this GPT and the time
       they take at the 1.0 TB/s of the RTX 4090. What fraction of $a$ is
       this?
    1. Use the fit to predict the throughput at $B = 1{,}024$ and
       $B = 2{,}048$. Check with the memory measured in
       :numref:`subsec_ft-rungs` whether these minibatches fit on the card,
       and measure those that do.

1. [code] **Causality gate.** The training run at the end of
   :numref:`subsec_ft-waterfall` accepts the final configuration when its
   validation loss lies below that of a bigram model. Consider an
   optimized attention kernel that silently ignores the causal mask.
    1. Predict how the training loss of the GPT behaves with this kernel
       and whether the check passes. Explain why the loss on the
       validation windows does not detect the defect either.
    1. Write a gate that changes the last input token of every sequence
       in a minibatch and compares the logits at all earlier positions
       before and after the change. Run it on `learner` and on a copy of
       the model whose attention omits the mask, and justify the tolerance
       that you choose.

1. **Throughput versus time to target.** The cumulative comparison ranks
   configurations by tokens per second, but what a training run needs is
   a short time to reach a target loss. The model of
   :numref:`sec_batch_size` says that training at batch size $b$ needs

    $$E(b) = E_{\min}\left(1 + \frac{b}{b_{\textrm{noise}}}\right)$$

   examples to reach the target, where $E_{\min}$ is the number needed
   with very small batches and $b_{\textrm{noise}}$ is the noise scale of
   the gradient.
    1. Let $\rho > 1$ be the ratio of the throughput at batch 512 to the
       throughput at batch 64. Show that batch 512 reaches the target
       sooner exactly when

        $$b_{\textrm{noise}} > \frac{512 - 64\rho}{\rho - 1},$$

       and evaluate the threshold for $\rho = 1.2$ and for $\rho = 1.7$.
    1. Classify each configuration of :numref:`subsec_ft-rungs`, data
       parallelism under weak scaling, and a switch to the optimizer of
       :numref:`sec_muon` by whether it changes $E(b)$, the throughput, or
       both.

1. [code] **Width 256.** At `num_hiddens=256` the GPT is the model that
   :numref:`sec_gpt` trains.
    1. Consider the product of the $8192 \times 256$ activations of one
       minibatch with a $256 \times 256$ weight matrix, the shape of the
       output projection $\mathbf{W}_o$. Compute its arithmetic intensity
       with 4-byte tf32 operands and with 2-byte bf16 operands, place both
       on the roofline of :numref:`fig_roofline` for the RTX 4090, taking
       the tf32 peak as half the bf16 peak, and predict the speedup of
       bf16 for this product.
    1. Run the baseline and the first two changes of
       :numref:`subsec_ft-rungs` at `num_hiddens=256`. Profile the second
       and the third configuration with `profile`, and split the
       device time of each between the matrix products and all other
       kernels. Account for the measured ratio of the two configurations
       kernel by kernel, and compare it with the prediction.

1. [code] **Data-parallel overlap.** ● Let $\phi$ be the fraction of
   $t_{\textrm{allreduce}}(k)$ that the framework hides under the
   backward pass, so that a step on $k$ GPUs takes

    $$t_{\textrm{step}}(k) = t_{\textrm{step}}(1) + (1 - \phi)\, t_{\textrm{allreduce}}(k).$$

    1. Compute $\phi$ at $k = 2$ and $k = 4$ for both configurations of
       :numref:`subsec_ft-dp` from the printed measurements. What does a
       negative value of $\phi$ indicate?
    1. Repeat the bf16 measurement with 128 and 256 windows per GPU, and
       plot the weak-scaling efficiency and $\phi$ against the minibatch
       per GPU. Explain the trend by the terms of :eqref:`eq_ft-dp`.
    1. Predict from :eqref:`eq_ft-dp`, without overlap, the smallest
       minibatch per GPU at which four GPUs reach 90% of linear scaling in
       bf16, and check the prediction.

1. [extended] **Another model.** ●● Apply the method of this section to
   the Mamba language model of :numref:`subsec_mamba-lm` or to `d2l.ViT`
   of :numref:`sec_vision-transformer`.
    1. Write the rules of the measurement in the form of the table in
       :numref:`subsec_ft-subject`, profile the baseline, and classify its
       limiting regime in the terms of :numref:`subsec_perf-regimes`
       before choosing the first change.
    1. Apply the removal of host overhead, bf16, a larger minibatch, and
       activation checkpointing one at a time, with a correctness check
       before every timing, and tabulate the cumulative comparison.
    1. Identify every change whose effect differs in sign or size from
       the GPT, and explain each difference by the mechanism that the
       change targets.

[Discussions](https://d2l.discourse.group/t/36793)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.7]{.kicker}

Case Study: Optimizing a Transformer<br>
**profile · host overhead · bf16 · larger minibatches · checkpointing · data parallelism**
:::
:::

::: {.slide title="A GPT Step Needs 6 ms of Arithmetic and Takes about 30 ms"}
[The problem]{.kicker}

GPT, width 512, 19M parameters; 64 windows × 128 bytes = 8,192 tokens.

- Arithmetic: $6 \times 1.9 \times 10^7 \times 8192 \approx 0.9$ TFLOP
  per step.
- RTX 4090 in bf16: 165 TFLOP/s, so **under 6 ms** per step.
- Plain fp32 step with AdamW: **about 30 ms**.

::: {.d2l-note .warn}
Four fifths of the card idle; every run takes several times longer than
its arithmetic.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- classify a training step from a kernel profile: waiting for the host,
  busy with short elementwise kernels, or busy with matrix products;
- pick the change that matches the limit, check correctness, and measure
  it on top of the previous changes;
- read a cumulative comparison, including a change that hurts;
- predict a data-parallel speedup from $t_{\textrm{step}}(1)$ and
  $t_{\textrm{allreduce}}(k)$ before measuring it.

**Running example:** the GPT of the Transformers chapter at width 512,
WikiText-103 bytes, one RTX 4090, then two and four.
:::

::: {.slide title="The Case Study Applies the Whole Chapter to One Model"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Performance Model | roofline, three regimes, timing, profiling |
| Hardware | where peak arithmetic and bandwidth come from |
| Compute Graphs and Compilation | fusion and kernel replay |
| Memory and Precision | what fits: bytes per parameter and per token |
| Multi-GPU from First Principles | data parallelism by hand, its cost model |
| Multi-GPU in Practice | production data parallelism, sharding |
| **Case Study: Optimizing a Transformer** | **the whole method on one GPT** |
:::

::: {.col .narrow}
- **Uses:** every section of the chapter.
- **Skips:** fp8 training, custom kernels, more than one machine.
:::
:::
:::

::: {.slide title="At Width 512 the Products Clear the Ridge Point"}
Product of $n \times d$ activations with a $d \times d'$ weight, $b$ bytes
per number:

$$I = \frac{2ndd'}{b\,(nd + dd' + nd')}$$

- $n = 8192$ tokens, width 512, bf16: $I \approx 250$ to $370$ FLOP/byte.
- Ridge point of the RTX 4090: $165\,\textrm{TFLOP/s} / 1.0\,\textrm{TB/s}
  \approx 164$ FLOP/byte.
- At width 256: around the ridge point.

::: {.d2l-note}
The products are limited by arithmetic, so a faster arithmetic format
shortens them directly.
:::
:::

::: {.slide title="Throughput Is Measured End to End, One Change at a Time"}
- Tokens per second, including sampling, forward, backward, and update.
- Training text on the GPU (270 MB); fresh windows every step.
- `d2l.Benchmark`: 20 warm-up steps, then 50 steps between two device
  synchronizations.
- fp32 with tf32 matrix products until bf16 is introduced.
- Every change to the computation checked before it is timed.

::: {.d2l-note}
A window varies by a few percent: read ratios to one decimal.
:::
:::

::: {.slide title="The Eager Step Spends a Third of Its Device Time Outside the Products" only="pytorch"}
[Baseline]{.kicker}

@!fast-transformer-the-baseline-2

- Device busy for nearly the whole step: the host keeps up.
- Matrix products a little under two thirds of the device time; the rest
  are short, bandwidth-bound elementwise and normalization kernels.
:::

::: {.slide title="The Jitted Step Is Fused, but the Device Can Wait for the Host" only="jax"}
[Baseline]{.kicker}

@!fast-transformer-the-baseline-2

- One compiled XLA program per step, elementwise operations fused.
- Device busy for nine tenths of the step or more; matrix products a
  little under two thirds of its time.
:::

::: {.slide title="Compilation Fuses the Elementwise Tail" only="pytorch"}
[Small kernels]{.kicker}

```{.python #fast-transformer-exercises}
compiled = torch.compile(model, dynamic=False, mode='reduce-overhead')
high = model(X)
error = ((compiled(X) - high).norm() / high.norm()).item()
assert error < 1e-4      # same answer first
step = make_step(model, compiled)
```

@!fast-transformer-rungs-each-one-measured-1

- Chains of elementwise operations become one kernel each;
  `reduce-overhead` replays them as CUDA graphs.
- Logits agree to three or more digits: fused kernels round in another
  order. Throughput **about 1.1×**.
:::

::: {.slide title="nnx.jit Walks the Model on Every Call; jit_partial Once" only="jax"}
[Host overhead]{.kicker}

@!fast-transformer-rungs-each-one-measured-1

- `nnx.jit` walks model and optimizer on every call: about 8 ms, hidden
  only while the host keeps ahead of the device.
- `graph=False`: trees, not objects; `jit_partial`: bound once.
- Checked first: equal losses and weights. Throughput rises by the idle
  share of the first profile, **about 1.1×** for an idle tenth.
:::

::: {.slide title="After the Change, Matrix Products Take the Largest Share"}
@!fast-transformer-rungs-each-one-measured-5

- Busy for nearly the whole step.
- Matrix products take the largest share: arithmetic is the main limit.
- Next: make the products faster.
:::

::: {.slide title="Check: Which Change Comes Next?"}
[Check]{.kicker}

The device is busy throughout, and matrix products take the largest
share of its time. Candidates: bf16, a larger minibatch, activation
checkpointing.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** bf16: it doubles the tensor-core rate (82.6 → 165 TFLOP/s)
and halves the bytes of every activation. Checkpointing saves memory and
costs time, and nothing so far says memory is short.
:::
:::
:::

::: {.slide title="bf16 Speeds Up the Step About 1.7 Times, Not 2" only="pytorch"}
[Reduced precision]{.kicker}

@!fast-transformer-rungs-each-one-measured-2

- bf16 products and activations; fp32 weights, gradients, Adam moments,
  norms, residual stream, loss. No loss scaling.
- Check first: logits agree to about 1%.
- Norms, stream, casts, and the update still move fp32 values: less
  than 2×.
:::

::: {.slide title="bf16 About Doubles the Throughput" only="jax"}
[Reduced precision]{.kicker}

@!fast-transformer-rungs-each-one-measured-2

- bf16 products and activations; fp32 weights, gradients, Adam moments,
  norms, residual stream, loss. No loss scaling.
- Check first: logits agree to about 1%.
- More than faster products: the fused attention stores no weights of
  query–key pairs, and the activations of SwiGLU take half the bytes.
:::

::: {.slide title="In JAX, bf16 Is a Compute Dtype on the Linear Layers" only="jax"}
[Reduced precision]{.kicker}

- Flax layers compute in `dtype`, store in `param_dtype`: set `dtype`
  on the linear layers; norms and embedding stay fp32.
- Gradients arrive fp32: the derivative of a cast is a cast back.
- `FusedGPT`: cuDNN's fused attention (16-bit only) instead of XLA's,
  which builds all attention weights.
:::

::: {.slide title="512 Windows per Step Spread What Fixed Cost Remains"}
[Larger minibatches]{.kicker}

@!fast-transformer-rungs-each-one-measured-3

- Fixed per step: optimizer update (every weight and moment), launches.
- After fusion and replay, little of it is left to spread.
- Memory grows with the minibatch; still well under 24 GiB.
:::

::: {.slide title="At Batch 512, the Kernels Between the Products Take the Rest"}
[Third profile]{.kicker}

@!fast-transformer-larger-minibatches-1

- Busy throughout; matrix products (attention included): half or more.
- The rest: norms, SwiGLU, rotary embeddings, casts. They move bytes
  and leave the tensor cores idle.
:::

::: {.slide title="Checkpointing Costs Time When Memory Is Not the Limit"}
[Activation checkpointing]{.kicker}

@!fast-transformer-rungs-each-one-measured-4

- Keeps only each block's input; recomputes the rest in backward.
- Memory falls by more than half; throughput by a tenth to a quarter.

::: {.d2l-note}
The right change only for a model or minibatch that does not fit.
:::
:::

::: {.slide title="Each Change Is Measured on Top of the Last"}
@!fast-transformer-the-waterfall

- Total: about 2× or more from three changes.
- Final step: roughly half of the bf16 peak.
- A change that shortens device work gains nothing while the step waits
  for the host: order matters.
:::

::: {.slide title="The Allreduce Is Added to the Step"}
[Data parallelism]{.kicker}

$$t_{\textrm{step}}(k) \approx t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k)$$

- Weak scaling: every GPU keeps its minibatch; $k$ minibatches per step.
- Speedup $k\,t_{\textrm{step}}(1) / (t_{\textrm{step}}(1) +
  t_{\textrm{allreduce}}(k))$.
- No overlap assumed, so the prediction is a floor.
:::

::: {.slide title="Check: Do Two GPUs Help?"}
[Check]{.kicker}

Suppose an allreduce of the 76 MB of fp32 gradients through host memory
takes 34 ms. Predict the speedup of two GPUs for a 24 ms step and for a
106 ms step.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $2 \cdot 24 / (24 + 34) \approx 0.8$: slower than one GPU.
$2 \cdot 106 / (106 + 34) \approx 1.5$: three quarters of linear. A
second GPU helps only if the allreduce is shorter than a step.
:::
:::
:::

::: {.slide title="Measured Speedups Follow the Cost Model"}
[Data parallelism]{.kicker}

@!fast-transformer-rungs-each-one-measured-8

- Batch 64: allreduce about as long as the step or longer: no gain.
- Batch 512 in bf16: step several times longer, same allreduce: well
  above one GPU.
- At or above the prediction, up to a few percent of noise: the overlap
  with the backward pass.
:::

::: {.slide title="The Optimized Step Still Trains the Model"}
@!fast-transformer-training-check

- 300 steps of bf16 at batch 512 from a fresh model.
- Starts near $\log 256 \approx 5.55$; validation below the bigram level
  of 2.45 nats per byte.
:::

::: {.slide title="Check: Where Would the Next Factor of Two Come From?"}
[Check]{.kicker}

The final step keeps the device busy at about half of the bf16 peak.
One GPU, then four GPUs over PCIe.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** One GPU: fp8 doubles the tensor-core rate (with per-tensor
scaling), or better kernels raise efficiency. Four GPUs: a faster
interconnect (NVLink) shrinks $t_{\textrm{allreduce}}$, or larger steps
hide it.
:::
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Measure, classify, change one thing, check, measure again.
- Short elementwise kernels → fusion; host work → less of it, which
  gains only while the device waits.
- Busy with products → bf16, the largest single gain; batch 512 spreads
  what fixed cost remains.
- Checkpointing: slower, since memory is not the limit.
- Data parallelism: $t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k)$;
  helps only for long steps on this machine.

**Next:** fp8 training and training across machines, in the Language
Models part.
:::
