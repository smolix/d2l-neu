# Case Study: Optimizing a Transformer
:label:`sec_fast_transformer`

Each section of this chapter has isolated one cause of a slow training
step and the change that removes it: the host's cost of issuing work, the
memory traffic of small operations, wide number formats, the limits of a
single GPU. A real training step suffers from several of these causes at
once, and removing one changes how much the others matter. So let us take
one model, as a practitioner would, and ask two questions: how much faster
can its training step become, and in which order should the changes come?

The model is the GPT of :numref:`sec_gpt` with its width doubled to 512,
which gives it 19 million parameters. One training step on 64 windows of
128 bytes performs about $6 \times 1.9 \times 10^{7} \times 8{,}192
\approx 9 \times 10^{11}$ floating-point operations: two per parameter and
token in the forward pass and four in the backward pass. The tensor cores
of an RTX 4090 execute $1.65 \times 10^{14}$ of them per second in bf16,
so the arithmetic of a step needs less than 6 ms. A plain training step,
in fp32 with an AdamW update, takes about 25 ms. A factor of two of this
gap is the number format alone: the plain step multiplies on the tf32
tensor cores, whose peak is half the bf16 one, so even its arithmetic
needs 11 ms. The other factor of two is time in which the tensor cores run
below their peak or not at all: even in its own format, the step takes
more than twice as long as its arithmetic requires.

We follow the method of :numref:`sec_perf_model`: count, predict, measure,
and explain the difference. Before each change, the last profile of the
step bounds what the change can gain; after it, we compare. Each change is
measured on top of the previous ones, because each moves the bottleneck.
The section ends with a check that the fastest step still trains the model
and an account of where its time goes.

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
import re
from torch.utils.checkpoint import checkpoint
```

```{.python .input #fast-transformer-case-study-optimizing-a-transformer-2}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # fp32 products on tf32 cores
# Our outputs come from a run with TEARDOWN_CUPTI=1 in the environment, set
# before starting, so that the profiler of the cells below detaches from
# the GPU when a profile ends; without it, every later kernel launch of the
# process stays slower
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

We change one setting, the width, from 256 to 512, for the sake of
arithmetic intensity, the number of FLOPs that an operation performs per
byte it moves (:numref:`sec_perf_model`). A product of an $n \times d$
matrix of activations with a $d \times d'$ weight matrix performs $2ndd'$
FLOPs and moves $b(nd + dd' + nd')$ bytes, with $b$ bytes per number. For
the $n = 8{,}192$ tokens of a step at width 512, the products of a block,
from the square output projection to the query-key-value projection,
reach 125 to 185 FLOPs per byte with the 4-byte operands of tf32 and 250
to 370 in bf16. Both ranges lie above the ridge points of the RTX 4090, 82
FLOPs per byte for tf32 and 164 for bf16, the intensities above which
arithmetic rather than memory bandwidth limits an operation. A faster
format therefore shortens these products directly. At width 256 they lie
around the ridge point; exercise :exref:`ex_ft-width` studies that model.

The training text has 270 million bytes, one per token, so it fits on the
GPU as an array of 270 MB. The next cell places it there and defines
`windows`, which draws every minibatch on the GPU from random offsets, so
that no step waits for data from the host. The cell also builds the model.

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
matrices of its six blocks. Its training state of 16 bytes per parameter,
four each for the weight, its gradient, and Adam's two moment estimates,
takes 0.3 GiB, so nothing here needs gradient accumulation or sharding
(:numref:`sec_memory_precision`, :numref:`sec_sharding`), and with the
text on the GPU the input pipeline of :numref:`sec_hardware` is not
measured either. Every cell needs a GPU.

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
| timing | `d2l.Benchmark`: 20 warm-up steps, then three windows of 50 steps, each between two device synchronizations |
| result | median tokens per second of the three windows, its ratio to the previous configuration, and the spread of the windows |

`tokens_per_second` applies these rules with `d2l.Benchmark` of
:numref:`sec_perf_model`, whose warm-up calls absorb one-time costs such
as compilation and memory allocation. `record` stores each result and
prints its ratio to the previous configuration with the spread of the
windows: the largest minus the smallest throughput, relative to the
median. A configuration recorded with `kept=False` is a branch on which
later configurations do not build.

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

The spread is our measure of noise, and we can say in advance where it
will be large. A step that does much work on the host is exposed to
everything else the host does, so the baselines should spread more than
the configurations that remove that work. Runs on different days differ by
more than the windows of one run, because the other work on the host
changes. We therefore read ratios to one decimal place and believe a
change of a few percent only where the profile predicts it.

## A Single GPU
:label:`subsec_ft-rungs`

We first make the step fast on one GPU, one change at a time, starting
from the step as :numref:`sec_gpt` wrote it.

### Baseline
:label:`subsec_ft-baseline`

The baseline is the training step of :numref:`sec_gpt` with plain AdamW:
a forward pass, the cross-entropy loss, a backward pass, and the update.

:begin_tab:`pytorch`
The step runs eagerly: Python and autograd issue every operation of the
forward and the backward pass as one or more kernel launches. The
optimizer, created with `fused=True`, updates all weights and moment
estimates in a few kernels. `make_step` packages the step as a function of
a minibatch; its arguments `forward` and `bf16` bring in the changes of
the next subsections.
:end_tab:

:begin_tab:`jax`
The step is one function, `train_step`, compiled with `nnx.jit` as in the
training loop `d2l.train_steps` of :numref:`sec_test_models`. It receives
the model and the optimizer as arguments and returns the loss, which it
computes from fp32 logits.
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

The windows of the baseline spread at least ten times as widely as those
of every later configuration, as expected: this step still leaves work to
the host. A profile shows where its time goes.
`profile` records five steps with `d2l.kernel_times` and prints them
with `d2l.show_profile` (:numref:`subsec_perf-profiler`): the time during which the device runs
kernels against the wall-clock time of a step, the share of the matrix
products, counting fused attention kernels among them, and the largest
kernels. It also keeps the kernel times for the account at the end of the
section.

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

The device runs kernels for nearly the whole step, and the matrix
products take about two thirds of its time. The host keeps up, so the
step is held back by the work of the device itself.

:begin_tab:`pytorch`
The third outside the products is spread over many short kernels: the
elementwise operations of SwiGLU, of the rotary embeddings, and of the
residual connections, and the normalizations. Each of them reads its
operands from device memory and writes its result back, so together they
are limited by memory bandwidth (:numref:`subsec_comp-fusion`).
:end_tab:

:begin_tab:`jax`
The step is a single compiled program in which XLA has already fused the
elementwise operations into the kernels around them. The third outside
the products goes to these fusions, each still limited by the bytes that
it moves.
:end_tab:

### Host Overhead and Small Kernels

Two remedies attack the time that a step spends outside its matrix
products, and the profile tells us which to try first and how much it can
gain. Gaps in which the device waits for the host mark the overhead-bound
regime of :numref:`sec_perf_model`, in which a faster GPU would not
shorten the step; the remedy is less work on the host per step. Device
time spread over many short elementwise kernels marks the bandwidth-bound
regime; the remedy is fusion, which merges a chain of them into one kernel
that reads its inputs once and writes its result once. Less host work can
gain at most the share of the step in which the device waited, and
fusion, which leaves the matrix products alone, at most the time outside
them. A device that waits for the host therefore calls for the first
remedy, and a busy device that spends its time in short kernels for the
second.

:begin_tab:`pytorch`
Our step is in the second regime. The device waited for a few percent of
the step, and the kernels outside the matrix products took 8.8 of its 24.4
ms of device time. Even if compilation removed all of them and the idle time,
the step would shrink to the 15.6 ms of the products: a gain of at most
1.6. `torch.compile` (:numref:`sec_compilation`) generates fused kernels
for the forward and the backward pass and leaves the matrix products and
the optimizer update as they are. With `mode='reduce-overhead'` it also
records the kernels as CUDA graphs and replays each graph with one launch
(:numref:`subsec_comp-overhead`), and `dynamic=False` specializes them to
the minibatch shape. The graphs can gain at most the few percent in which
the device waited; we keep them because they cost nothing here and keep the
host from becoming the limit as later changes shorten the device's step.
Before timing anything, we check that the compiled model computes the same
logits as the original one.
:end_tab:

:begin_tab:`jax`
Our step is already fused, and the device waited for 2% of it, so less
host work can gain at most 2% here. We remove the host work all the same.
It is a fixed cost per step, hidden only while the device's own step is
longer, and every later change shortens that step. `nnx.jit` lets us pass
the model and the optimizer, Python objects that hold arrays, to a
compiled function, and on every call it walks through both objects to
collect their arrays and writes the results back. We time this work alone with compiled functions that do nothing, called through
`nnx.jit`; through `nnx.jit` with `graph=False`, which treats the two
objects as trees of arrays; and through `nnx.jit_partial` with
`graph=False`, which also binds them to the function once, so that later
calls pass only the minibatch.
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
The logits agree to four significant digits or more. They are not
identical, because the fused fp32 kernels round in a different order than
the separate ones, and the tf32 products carry the difference forward.
Throughput rises by more than a tenth, well inside the bound of 1.6.
:end_tab:

:begin_tab:`jax`
A call through `nnx.jit` takes almost 9 ms to do nothing, more than a
third of the device time of a step. It stays hidden as long as the host
prepares the next step while the device runs the current one; a host that
also serves other work falls behind, and the idle share of the profile
then measures by how much. The cost belongs to the convenience of walking
Python objects, not to JAX: trees remove most of it, and binding removes
nearly all of the rest. The next cell checks that binding changes how the
step is called, not what it computes: one step each way, from equal copies
of the model and the optimizer, must give equal losses and leave equal
weights.
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
The updated weights agree exactly, and throughput rises by about 1%,
within the bound and within the spread of the baseline.
:end_tab:

A second profile shows what the change did to the step:

```{.python .input #fast-transformer-rungs-each-one-measured-5}
%%tab pytorch, jax
profile(step, windows(corpus, 64))
```

The device is still busy for nearly the whole step, and the matrix
products now take about two thirds of its time or more. The step is
limited by the device, mostly by the arithmetic of the matrix products.

:begin_tab:`pytorch`
The device time per step fell from 24.4 to 22.2 ms, and the gaps between
kernels closed. The two parts of `torch.compile` divide the gain between
them: fusion shortened the device time by 2.2 ms, a quarter of the time
outside the products, and the replay of CUDA graphs closed the gaps, about
a millisecond per step in this run. The products take as long as before,
and three quarters of the time
outside them remains.
:end_tab:

:begin_tab:`jax`
The device time is the same as in the first profile, since binding
changed only the work of the host.
:end_tab:

### Reduced Precision
:label:`subsec_ft-precision`

When the device is busy and the matrix products take most of its time, the
next change should make the products faster. In bf16 the tensor cores of
the RTX 4090 run at 165 TFLOP/s, twice their tf32 rate, and every
activation kept in bf16 takes half the bytes
(:numref:`sec_memory_precision`). The profile again bounds the gain. If
the products take a fraction $f$ of the step and only they become twice as
fast, the step shrinks to $1 - f/2$ of its time, a gain of about 1.5 for
$f$ between two thirds and seven tenths. If every other kernel also moved
half the bytes, the gain would be 2.

We apply the mixed-precision recipe of :numref:`subsec_mp-mixed`. The
matrix products, those of the attention included, and the elementwise
operations between them run in bf16, while the weights, their gradients,
Adam's moment estimates, the normalizations, the loss, and the residual
stream that connects the blocks stay in fp32. The stream stays in fp32
because each block adds to it an update that can be much smaller than the
stream itself, and bf16, with 8 significant bits, would round away an
update a few hundred times smaller than the entry it is added to. Since
bf16 has the exponent range of fp32, the loss needs no scaling. Before the
timing we compare the bf16 logits with the fp32 logits of the same
weights; they should agree to the precision of bf16, two to three
significant digits.

:begin_tab:`pytorch`
`torch.autocast` applies this recipe operation by operation
(:numref:`sec_numerics`): it runs the matrix products and the attention in
bf16 and computes reductions such as the softmax of the loss in fp32. The
embedding and the normalizations receive no cast, so the stream stays in
fp32, and the backward pass follows the dtypes of the forward pass.
:end_tab:

:begin_tab:`jax`
A Flax layer computes in the dtype of its attribute `dtype` and keeps its
parameters in `param_dtype` (:numref:`sec_numerics`). `d2l.to_bf16` of
:numref:`subsec_mp-mixed` sets `dtype` to bf16 in the linear layers only,
so the embedding, the normalizations, and the output layer, which reuses
the embedding table, compute in fp32. The gradients arrive in fp32,
because the derivative of a cast to bf16 is a cast back.
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

The logits agree to within one percent.

:begin_tab:`pytorch`
Throughput rises by about three quarters: more than the factor of 1.55
that halving the products alone would give, since they took 71% of the
step, and less than 2. The surplus has two sources: the bf16 products run
closer to their
peak than the tf32 ones did, and the kernels between them move fewer
bytes. They do not move half as many, because the normalizations, the residual stream, the
casts, and the optimizer update still carry fp32 values. The attention
needed no new kernel: since the baseline, `F.scaled_dot_product_attention`
has run a fused one, which never writes the attention weights to device
memory.
:end_tab:

:begin_tab:`jax`
Throughput rises by about 1.45 times, a little less than the 1.48 that
halving the products alone predicts for products at 65% of the step. So
the kernels between the products got no faster. Among them is XLA's
attention: `d2l.GPT`
leaves the choice of attention kernel to `jax.nn.dot_product_attention`,
whose default is XLA's own implementation, and that implementation builds
the attention weights of all query–key pairs in device memory
(:numref:`sec_attention-at-scale`). `FusedGPT` selects cuDNN's fused
kernel instead, which keeps them on the chip. The kernel accepts only
16-bit inputs, so it can follow only once the products run in bf16, and we
measure it as a change of its own. The profile counts XLA's attention
among the kernels between the products and does not show it separately,
so it bounds this change only by the time outside the products, which is
loose.
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
The fused attention raises throughput by about another third, and the
logits still agree with the fp32 ones to within one percent. It leaves
the arithmetic of the attention as it was and removes the traffic of its weights: this is the
bandwidth-bound part of the step responding. Together, the two rows
nearly double the throughput.
:end_tab:

### Larger Minibatches

A larger minibatch divides the fixed costs of a step among more tokens,
and the second profile says that there is little to divide. The device is
busy throughout, so only work that it does once per step, however many
tokens the step has, can be spread. Two such costs can be named. AdamW
reads every weight, its gradient, and both moment estimates, and writes
the weight and both moments: 28 bytes per parameter in fp32. In bf16 each
step also casts the fp32 weights to bf16 for the forward pass and the bf16
weight gradients back to fp32, 12 bytes more. Together that is 40 bytes
per parameter, or 0.76 GB per step. At the 1.0 TB/s of the card's memory
it takes about 0.75 ms, a sixteenth of a bf16 step of 12 to 13 ms (8,192
tokens at about two thirds of a million tokens per second). Raising the
minibatch from 64 to 512 windows, or 65,536 tokens per step, spreads 7/8
of this cost and should gain a few percent, about 5% if these are the
only fixed costs.

The change is still worth making, because data parallelism helps only
when the step on each GPU is long (:numref:`subsec_ft-dp`). It is not
free in training terms: beyond the critical batch size of
:numref:`sec_batch_size`, eight times the examples per step do not make a
step eight times as useful (exercise :exref:`ex_ft-target`). Since the
activations grow with the minibatch, we also record the memory that the
step needs beyond the weights and the optimizer state.

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

The step needs about a third of the card's 24 GiB.

:begin_tab:`pytorch`
Throughput rises by about 2%, less than the fixed costs predict. Either some
kernels of the step cost slightly more per token at the larger size, or
the noise between runs hides part of the gain; exercise
:exref:`ex_ft-fixed` measures the fixed cost and the cost per token
separately.
:end_tab:

:begin_tab:`jax`
Throughput rises by about 6%, close to what the update and the casts
predict. Exercise
:exref:`ex_ft-fixed` separates the fixed cost of a step from its cost per
token by measuring several minibatch sizes.
:end_tab:

A third profile shows what limits the step at batch 512:

```{.python .input #fast-transformer-larger-minibatches-1}
%%tab pytorch, jax
profile(step, windows(corpus, 512))
```

The device is busy for the whole step, and the matrix products, those
inside the fused attention included, take a little under three fifths of
its time. The largest single kernels are matrix products. Most of the rest
goes to the kernels between them, for the normalizations, SwiGLU, the
rotary embeddings, and the casts between fp32 and bf16, which move bytes
through device memory and leave the tensor cores idle.

:begin_tab:`pytorch`
The kernels that `torch.compile` generated take most of the time outside
the products; the largest of them computes the backward pass of SwiGLU.
:end_tab:

:begin_tab:`jax`
XLA's fusions take the time outside the products.
:end_tab:

### Activation Checkpointing

Activation checkpointing (:numref:`subsec_mp-checkpointing`) saves memory
by recomputing activations: each block keeps only its input during the
forward pass and recomputes everything else during the backward pass. The
step at batch 512 uses about a third of the card's memory, so memory does
not limit it, and our method predicts that checkpointing can only slow the
step down. It even predicts by how much. The blocks run their forward pass
twice, 8 FLOPs per parameter and token instead of 6, so a step limited by
the device should keep about $6/8 = 0.75$ of its throughput.

A FLOP count is a proxy for time, and the clock can give a second
estimate. We time the forward pass of the kept configuration alone, on one
minibatch of 512 windows and without computing gradients, and ask what
running it once more would leave of the step.

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

```{.python .input #fast-transformer-activation-checkpointing-2}
%%tab pytorch, jax
t_step = 512 * 128 / results['+ batch 512']  # Seconds per step
print(f'forward pass {1e3 * t_forward:.1f} ms of a {1e3 * t_step:.1f} ms '
      f'step: repeating it keeps at least {1 / (1 + t_forward / t_step):.2f} '
      f'of the throughput')
```

The two estimates agree. The forward pass takes about a third of the step,
as the FLOP count said, and repeating it would keep about three quarters
of the throughput. The cell prints a bound, "at least", because the timed
forward pass includes the embedding and the output layer, which
checkpointing does not run twice; at 256 symbols they are a small share of
the work. We now measure the checkpointed step, to check both estimates
and to see how much memory checkpointing would free for a model that
needed it.

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

Checkpointing cuts the memory of the step to about a third, and it keeps
more of the throughput than either estimate left it.

:begin_tab:`pytorch`
Throughput falls to 0.85 of the previous configuration, well above the
0.77 that the clock allowed. Repeating the forward pass of the blocks
added less than a fifth to the step, while the same forward pass run on
its own takes three tenths of it: inside the backward pass the
recomputation costs only about three fifths of what it costs alone. We
have not established why. The embedding and the output layer explain
little of it, and the compiler, which sees the recomputation and the
backward pass in one program, may run parts of them in shared kernels. A
profile of the checkpointed step would show which kernels the
recomputation adds and how long they take (exercise :exref:`ex_ft-remat`).
:end_tab:

:begin_tab:`jax`
Throughput falls to 0.79 of the previous configuration, close to the 0.76
that the clock allowed. Repeating the forward pass of the blocks added
about a quarter to the step, against the third that the same forward pass
takes on its own. A profile of the checkpointed step would show which
kernels the recomputation adds and how long they take (exercise
:exref:`ex_ft-remat`).
:end_tab:

Checkpointing is the right change for a model or a minibatch that would
not fit otherwise. For this step it is the wrong one, as the memory of the
previous configuration predicted, so the configurations that follow leave
it out.

## Data Parallelism
:label:`subsec_ft-dp`

The fastest single-GPU step still leaves three of the machine's four GPUs
idle. We predict what data parallelism gains on them before we measure
it.

### Predicted Speedup

Four GPUs could at best process four times as many tokens per second as
one. In data parallelism (:numref:`sec_multi_gpu`), every GPU holds a copy
of the model and runs the step on a minibatch of its own; before the
update, the GPUs sum their gradients with an allreduce. The cost model of
:numref:`subsec_mg-accounting` puts the time of a step on $k$ GPUs at

$$
t_{\textrm{compute}}(B/k) + c\,\alpha +
\frac{2(k-1)}{k}\,\frac{S}{\beta_{\textrm{link}}}.
$$

Here $B/k$ is each GPU's share of a minibatch of $B$ examples, $c$ is the
number of allreduce calls, each with a fixed cost $\alpha$, $S$ is the
size of the gradients in bytes, and $\beta_{\textrm{link}}$ is the rate at
which each GPU sends during a large allreduce. Our GPUs stage every byte
through host memory, and :numref:`sec_multi_gpu_concise` measured their
$\beta_{\textrm{link}}$ at about 2.4 GB/s.

In our runs the minibatch per GPU stays fixed as GPUs are added, which is
called *weak scaling*, so the computation term stays the time of a step on
one GPU. The allreduce runs in a few buckets, groups of gradients sent as
soon as they are ready (:numref:`sec_multi_gpu_concise`). Their fixed
costs, a fraction of a millisecond each, are small next to steps of tens
of milliseconds. The model therefore becomes

$$t_{\textrm{step}}(k) \approx t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k).$$
:eqlabel:`eq_ft-dp`

Since $k$ GPUs process $k$ minibatches in the time of :eqref:`eq_ft-dp`,
throughput grows by the factor
$k\, t_{\textrm{step}}(1) / (t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k))$.
A second GPU therefore raises throughput only if the allreduce takes less
time than a step.

Both terms are known before we run anything. The gradients of our GPT
occupy $S = 76$ MB in fp32, so the allreduce should take
$S/\beta_{\textrm{link}} \approx 31$ ms on two GPUs and
$1.5\, S/\beta_{\textrm{link}} \approx 47$ ms on four. A compiled fp32 step
at batch 64 takes a little over 20 ms, less than either; a bf16 step at
batch 512 takes 90 to 100 ms. For the first configuration the cost model
predicts a loss from a second GPU, a speedup of about 0.85, and little
gain from four, about 1.3. For the second it predicts about 1.5 on two
GPUs and 2.6 to 2.7 on four.

The model leaves out one effect. A framework can start summing the
gradients of the last layers while the backward pass still computes those
of the first (:numref:`fig_ddp_overlap`). The prediction is therefore a
floor, and the distance of a measurement above it shows how much of the
allreduce was hidden behind computation.

### Measured Speedup

We now test the prediction on two and four GPUs.
`data_parallel(k, bf16, batch_size)` trains on $k$ GPUs with the given
precision and minibatch size per GPU. It returns the throughput, together
with the measured time of one allreduce of a buffer as large as all
gradients on the same GPUs, from which the cell computes the predicted
speedup. Both configurations run the step without its host overhead, one
in fp32 at batch 64 and one in bf16 at batch 512.

:begin_tab:`pytorch`
`DistributedDataParallel` runs one process per GPU
(:numref:`sec_multi_gpu_concise`), so the step goes into a script that
`d2l.torchrun` starts on $k$ GPUs. The script reuses `windows` and
`make_step` from this section and wraps the model in
`DistributedDataParallel`, which sums the gradients in buckets while the
backward pass runs. It compiles without CUDA graphs, since both
configurations keep the device busy long enough to hide the host, and it
also runs on one GPU, so that the single-GPU step is measured under the
same conditions.
:end_tab:

:begin_tab:`jax`
JAX drives all GPUs from one process (:numref:`sec_multi_gpu_concise`).
`data_parallel` builds a mesh of $k$ GPUs, places a copy of the model, the
optimizer state, and the training text on every GPU, and splits the
windows of each minibatch across the GPUs. The unchanged step then
compiles into a program that computes the gradient of each share on its
own GPU and sums the gradients with an allreduce that the compiler
inserts. The XLA option set in the first cell, the latency-hiding
scheduler, lets the compiler overlap that allreduce with the backward
pass. The time of the allreduce comes from a `psum` over the mesh, as in
:numref:`sec_multi_gpu`.
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

The allreduce takes 31 ms on two GPUs and 48 ms on four, within a
millisecond of the prediction. At batch 64 two GPUs are slower than one,
and four gain less than a factor of 1.5. At batch 512 in bf16 the step is
several times longer while the allreduce stays the same, and the
prediction rises to about three quarters of linear scaling on two GPUs and
two thirds on four. In the measurement, four GPUs process about three
times as many tokens per second as one, the largest gain of any single
change in this section. Each measurement lies above its prediction or within a few
percent below it, and the distance above is the part of the allreduce
that ran during the backward pass. The cost model made two predictions of
opposite sign here, a loss from a second GPU for the short step and a gain
for the long one, and both held. On this machine, data parallelism gains
much only once the step on each GPU is several times longer than the
allreduce.

GPUs that share an NVLink domain send to each other at up to 900 GB/s in
each direction, hundreds of times our $\beta_{\textrm{link}}$
(:numref:`subsec_hw-interconnects`). The allreduce of 76 MB then takes a
fraction of a millisecond, and the cost model predicts nearly linear
scaling even at batch 64.

## Results
:label:`subsec_ft-waterfall`

We now put the measurements of all the changes side by side, check that
the fastest step still trains the model, and account for where its time
goes.

### Cumulative Comparison

The table collects the single-GPU configurations in the order in which we
applied them. Each ratio in the column `step` belongs to one change,
measured on top of all the changes above it; `total` is the product of the
ratios so far, and checkpointing, printed below a rule, is a branch
outside it. `TFLOP/s` converts throughput into arithmetic at six FLOPs per
parameter and token, and `MFU` divides it by the bf16 peak. Reports of
large training runs quote this ratio, with the FLOPs of the attention
scores added, as the *model FLOPs utilization*
:cite:`chowdhery2022palm`; exercise :exref:`ex_ft-mfu` adds them here.

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

Together the changes make the step about twice as fast, and the fastest
step performs about half of the 165 TFLOP/s that the tensor cores can
execute in bf16. The changes that shortened the matrix products and the
memory traffic around them did nearly all of this work. Less host work
gained what the idle share of the first profile allowed, a few percent at
most, and the larger minibatch gained a few percent, again as predicted.

:begin_tab:`pytorch`
Reduced precision alone gave about 1.75, inside the bounds that the second
profile set, in which the products held seven tenths of the step. The
compiler, to which this chapter gave a whole section, gains little more
than a tenth here,
because this model already spends most of its time in large matrix
products.
:end_tab:

:begin_tab:`jax`
Reduced precision gave about 1.45, close to what the share of the
products predicted, and the fused attention kernel another 1.35. The
profile could not have shown the second gain in advance, because it counts
XLA's attention among the kernels between the products. A profile read by
categories can hide a whole operation inside one of them.
:end_tab:

Each ratio depends on its place in the sequence. A change that shortens
the work of the device gains nothing while the step waits for the host,
and a change that removes host work gains nothing while the device is
busy. The data-parallel runs show the same for a single technique: two
GPUs make the fp32 step at batch 64 about a tenth slower, and they make
the bf16 step at batch 512 faster by a factor of 1.6 to 1.7. A ranking of
techniques that measures each one alone on the same baseline can
therefore mislead (exercise :exref:`ex_ft-order`).

The order also depends on the machine. These numbers belong to one model
on one RTX 4090 with one host CPU. A faster GPU shortens the device time
of every step but not the host's, so a baseline with as much host work as
ours would wait for the host, and removing host work would gain more. On a
slower GPU removing host work would gain nothing, while fusion would still
shorten the device time. The order of the changes follows from the profile
on the reader's machine, not from this section.

### Training Check

Throughput says nothing about whether the faster step still trains the
model. A cast that changes the logits, gradients that never reach some of
the weights, or a learning rate that the compiled step applies at a
different size would leave it unchanged. We therefore train two copies of
one initialization, one with the step of the baseline and one with the
final step, on the same 200 minibatches of 512 windows, about 13 million
tokens, and compute the loss of both on the validation text with
`d2l.lm_perplexity` (:numref:`sec_test_models`), in nats per byte. If the
final step computes the same training, the two curves should start at the
same loss and stay within a tenth of a nat of each other while training
runs smoothly, and as a floor each validation loss must lie below the 2.45
nats per byte of the bigram model of :numref:`sec_gpt`. Smoothness is not
guaranteed. The timing runs use a constant learning rate without warm-up
or gradient clipping. With such a recipe the loss now and then rises by a
few tenths of a nat or more late in training, and one curve can then lie
well above the other at a single printed step. The check is coarse in the
other direction as well: a defect that changes the weights by a fraction
of a percent over 200 steps, such as a dropped weight decay of 0.01,
passes it.

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

Both curves start at the same loss, near $\log 256 \approx 5.55$, the
loss of a uniform guess over the 256 bytes, since both models start from
the same weights. Through step 100 they stay within a tenth of a nat of
each other, and both validation losses lie below the bigram level.

:begin_tab:`pytorch`
The agreement lasts to the end. At step 200 the two curves differ by a few
hundredths of a nat, over its last 20 steps each moves in a band less than
a tenth of a nat wide, and the two validation losses differ by about a
hundredth, both well below the bigram level.
:end_tab:

:begin_tab:`jax`
In this run the loss of the baseline rose late: over its last 20 steps it
lies between 2.13 and 2.81, and at step 200 it stands 0.22 nats above the
final step, with a validation loss 0.21 higher. The curve of the final
step shows no such rise. A rise in the reference is no evidence against
the step under test, but it removes the comparison at step 200 and on the
validation text, and what remains is the agreement through step 100 and
the bigram floor.
:end_tab:

One pair of runs can show no more than this: the final step starts where
the baseline starts, follows it while training runs smoothly, and ends
below the bigram floor.

The same run shows what a throughput table leaves out. The GPT of
:numref:`sec_gpt` reached 1.35 nats per byte after 8 million tokens. This
run saw 13 million and ends more than half a nat higher. It took 200 steps
where that run took 1,000, and it kept the constant learning rate of the
timing runs instead of a schedule tuned for 200 steps. Throughput is not
progress: a training run needs a short time to a target loss, and exercise
:exref:`ex_ft-target` works out when a larger minibatch shortens it.

### Account of the Final Step
:label:`subsec_ft-account`

The chapter began by asking where the time of a step goes. For the fastest
step we can now answer in milliseconds per 8,192 tokens, and the third
profile says what to expect. It printed a step of about 90 to 100 ms for
65,536 tokens, or 11 to 13 ms per 8,192. The device was busy throughout,
so the step is the sum of its kernels: products that took a little under
three fifths of it, about 7 ms, against the 5.7 ms their arithmetic needs
at the bf16 peak, and kernels between them that take the other 5 ms or so.
`account` computes these numbers from the stored profiles of the baseline
and of the final step.

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

The final step spends about 7 ms per 8,192 tokens in matrix products,
which run at four fifths of the bf16 peak or more, and about 5 ms in the
kernels between them. The two regimes of the roofline thus split the step:
a little more than half of it is limited by arithmetic, and two fifths are
memory traffic. BERT shows a similar split, with contractions that perform
over 99% of the FLOPs and take 61% of the time
:cite:`Ivanov.Dryden.Ben-Nun.ea.2021`. The baseline row closes the account
of the opening: its arithmetic needs 11.3 ms at the tf32 peak, its
products ran at about three quarters of that peak, and the kernels between
them took more than 8 ms. From one row to the other the products became
more than twice as fast, because in bf16 and at the larger minibatch they
also run closer to their peak, and the kernels between them lost about two
fifths of their time.

What would make this step twice as fast again? Not faster arithmetic
alone. The products are within a fifth of their peak, and even fp8, which
runs at twice the bf16 rate on this card (:numref:`sec_hardware`), would
shorten only them: the step would become about 1.4 times faster, and at
most 1.5 even if the fp8 products ran at their peak. Nor fewer bytes
alone. With every kernel between the products gone, the step would still
take the 7 ms of its products, only 1.7 to 1.8 times faster than now. A
factor of two needs both. Normalizations and activations fused into the
kernels of the products, a residual stream in fewer bits, and kernels
written for one model cut the bytes between the products, and fp8
shortens the products themselves. Such kernels, training in fp8 with a
scale for each tensor or block of values, and training on many machines
belong to the Language Models part. After everything, two fifths of the step is still memory
traffic, which is where this chapter began.

## Summary

A profile classifies a training step and bounds what each change can gain
before it is tried: less host work at most the time in which the device
waits, fusion at most the time outside the matrix products, and a format
twice as fast between $1/(1 - f/2)$ and 2 when the products take a
fraction $f$ of the step. Each change is checked before it is timed and
measured on top of the previous ones, since each moves the bottleneck. On
one RTX 4090 the step of our 19-million-parameter GPT became about twice
as fast, nearly all of it from faster matrix products and less memory
traffic around them; a larger minibatch gained a few percent at most, and
activation checkpointing, which memory did not call for, cost less than
the predicted quarter. The cost model
$t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k)$, with the allreduce
predicted from the link rate, forecast a loss from a second GPU for a
short step and a gain for a long one, and the measurements met it. After
all the changes, two fifths of the step is still memory traffic, and a
faster step is not yet a faster training run.

## Exercises

1. [code] **Noise floor.** Each throughput in
   :numref:`subsec_ft-waterfall` is the median of three windows of 50
   steps, printed with the spread of the windows.
    1. Call `tokens_per_second` ten times in a row with `num_windows=1`
       for the second configuration of the table, and report the mean and
       the relative standard deviation of the ten results. Compare them
       with the spread that `record` printed for that configuration.
    1. Estimate the ratio of the third configuration to the second twice:
       from two blocks of ten consecutive windows, and from ten pairs of
       alternating windows. Suppose that the GPU clock drifts slowly
       during the measurement. Explain which of the two designs cancels
       the drift, and compare the two estimates.
    1. Suppose that windows vary independently with the relative standard
       deviation of the first part. Find the number of windows per
       configuration that resolves a change of 3% at two standard errors
       of the ratio. Which ratios in the column `step` does the median of
       three windows resolve? Run the notebook a second time and say
       whether its ratios move by more than this estimate, and why they
       might.

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
    1. Recompute the columns `TFLOP/s` and `MFU` of the table with these
       FLOPs included, and the fraction of the peak of each
       configuration's own format that it reaches: 165 TFLOP/s in bf16
       and 83 TFLOP/s in tf32.
    1. A utilization figure can count the FLOPs that the model requires or
       the FLOPs that the hardware executes. Name two configurations or
       kernels of :numref:`subsec_ft-rungs` for which the two counts
       differ, and state the direction of each difference.
   :exlabel:`ex_ft-mfu`

1. [code] **Fixed cost of a step.** Model the time of the bf16 step at a
   minibatch of $B$ windows as $t(B) = a + bB$, a fixed cost $a$ plus a
   cost $b$ per window.
    1. Measure the step at $B = 32, 64, 128, 256, 512$, fit $a$ and $b$
       by least squares, and report the fraction of the step at $B = 64$
       that $a$ accounts for.
    1. The section put the AdamW update and the casts of the weights and
       their gradients at about 0.75 ms per step. What fraction of $a$ is
       this? Name the other work in a step whose cost
       does not grow with $B$, and check your list against a profile of
       the step at $B = 32$.
    1. Use the fit to predict the throughput at $B = 1{,}024$ and
       $B = 2{,}048$. Check with the memory measured in
       :numref:`subsec_ft-rungs` whether these minibatches fit on the card,
       and measure those that do.
   :exlabel:`ex_ft-fixed`

1. [code] **Cost of recomputation.** Activation checkpointing in
   :numref:`subsec_ft-rungs` was measured against two estimates, one from
   the FLOP count and one from the time of the forward pass alone.
    1. Record the kernel times of the checkpointed step with
       `d2l.kernel_times`, as `profile` does, and split its device time
       with `account`. Compare both parts with those of the step of
       `+ batch 512`, and say which of them carries the extra time.
    1. Time the forward pass of the blocks alone, without the embedding
       and the output layer, and compute the bound of the section again.
       Does it move closer to the measured ratio?
    1. Checkpoint only every second block. Predict the throughput and the
       memory of that step from your measurements before you measure
       them, and compare.
   :exlabel:`ex_ft-remat`

1. [code] **Causality gate.** The training check of
   :numref:`subsec_ft-waterfall` accepts the final configuration when its
   loss curve follows that of the baseline and its validation loss lies
   below that of a bigram model. Consider an optimized attention kernel
   that silently ignores the causal mask.
    1. Predict how the training loss of the GPT behaves with this kernel.
       Explain why the bigram threshold accepts it and why the loss on the
       validation windows does not detect the defect either. Does the
       comparison with the baseline curve catch it, and would it still
       catch it if the baseline used the same kernel?
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

       and evaluate the threshold for $\rho = 1.2$, for $\rho = 1.7$, and
       for the $\rho$ computed from the column `tokens/s` of the cumulative
       table. How sensitive is the threshold to the rounding of $\rho$?
    1. Classify each configuration of :numref:`subsec_ft-rungs`, data
       parallelism under weak scaling, and a switch to the optimizer of
       :numref:`sec_muon` by whether it changes $E(b)$, the throughput, or
       both.
   :exlabel:`ex_ft-target`

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
       and the third configuration with `profile`, and split the device
       time of each between the matrix products and all other kernels with
       `account`. Account for the measured ratio of the two configurations
       kernel by kernel, and compare it with the prediction.
   :exlabel:`ex_ft-width`

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
       minibatch per GPU at which four GPUs reach 80% of linear scaling in
       bf16. Check with the memory measured in :numref:`subsec_ft-rungs`
       that it fits on the card, and test the prediction. Why does 90%
       not fit?

1. [code] **Order of the changes.** Each ratio of the cumulative table is
   measured on top of the changes above it.
    1. From the first profile and the bound of
       :numref:`subsec_ft-precision`, predict the ratio that bf16 would
       give if it were applied to the baseline directly.
    1. Measure that ratio, and compare it with the prediction and with the
       ratio of bf16 in the table. Then apply the first change of the
       table, the one before bf16, on top of bf16 alone, and compare its
       ratio with the one in the table.
    1. Explain both differences with the profiles of the configurations
       involved.
   :exlabel:`ex_ft-order`

1. [extended] **Another model or another GPU.** ●● Apply the method of
   this section to the Mamba language model of :numref:`subsec_mamba-lm`
   or to `d2l.ViT` of :numref:`sec_vision-transformer`, or to the GPT of
   this section on another GPU.
    1. Write the rules of the measurement in the form of the table in
       :numref:`subsec_ft-subject`, profile the baseline, and classify its
       limiting regime in the terms of :numref:`subsec_perf-regimes`
       before choosing the first change. On another GPU, replace the
       constants $P$ and $\beta$ of :numref:`sec_perf_model` by those of
       that GPU, and predict which change gains most before you run it.
    1. Apply the removal of host overhead, bf16, a larger minibatch, and
       activation checkpointing one at a time, with a correctness check
       before every timing, and tabulate the cumulative comparison.
    1. Identify every change whose effect differs in sign or size from
       this section, and explain each difference by the mechanism that
       the change targets.

[Discussions](https://d2l.discourse.group/t/36793)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.8]{.kicker}

Case Study: Optimizing a Transformer<br>
**profile · predict · host work and fusion · bf16 · larger minibatches · checkpointing · data parallelism · account**
:::
:::

::: {.slide title="A GPT Step Needs 6 ms of Arithmetic and Takes about 25 ms"}
[The problem]{.kicker}

GPT, width 512, 19M parameters; 64 windows × 128 bytes = 8,192 tokens.

- Arithmetic: $6 \times 1.9 \times 10^7 \times 8192 \approx 0.9$ TFLOP
  per step.
- RTX 4090: 165 TFLOP/s in bf16, **under 6 ms**; 83 TFLOP/s in tf32,
  11 ms.
- Plain fp32 step with AdamW: **about 25 ms**.

::: {.d2l-note .warn}
One factor of two is the number format; the other is time in which the
tensor cores run below peak or not at all.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- read a kernel profile: waiting for the host, busy in short kernels, or
  busy in matrix products;
- bound the gain of each change from the last profile, before measuring;
- check correctness before timing, and measure on top of earlier changes;
- predict a data-parallel speedup from the step time and the link rate;
- account for the milliseconds of the final step.

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
| Computation Graphs and Compilation | fusion and kernel replay |
| Memory and Precision | bytes per parameter and token, bf16, checkpointing |
| Multi-GPU from First Principles | data parallelism by hand, its cost model |
| Data Parallelism in Practice | production libraries, the link rate |
| Sharding the Training State | a state that no GPU holds |
| **Case Study: Optimizing a Transformer** | **the whole method on one GPT** |
:::

::: {.col .narrow}
- **Uses:** every section but sharding; the training state is 0.3 GiB.
- **Skips:** fp8 training, custom kernels, more than one machine.
:::
:::
:::

::: {.slide title="At Width 512 the Products Clear the Ridge Point in Both Formats"}
Product of $n \times d$ activations with a $d \times d'$ weight, $b$ bytes
per number:

$$I = \frac{2ndd'}{b\,(nd + dd' + nd')}$$

| $n = 8192$, width 512 | intensity | ridge point |
|---|---|---|
| tf32, $b = 4$ | 125 to 185 FLOP/byte | 82 |
| bf16, $b = 2$ | 250 to 370 FLOP/byte | 164 |

Ridge point = peak FLOP/s ÷ memory bandwidth.

::: {.d2l-note}
Compute-bound products: a faster format shortens them directly. At width
256 they sit around the ridge.
:::
:::

::: {.slide title="Throughput Is Measured End to End, One Change at a Time"}
- Tokens per second: sampling, forward, backward, update.
- Training text on the GPU (270 MB); fresh windows every step.
- 20 warm-up steps, then three windows of 50 steps: median and spread.
- tf32 products until bf16; every change checked before it is timed.

::: {.d2l-note}
Without host work, windows agree within 0.2%; runs differ more. Read ratios to
one decimal.
:::
:::


::: {.slide title="The Last Profile Bounds What the Next Change Can Gain"}
[Predict first]{.kicker}

| change | gain at most |
|---|---|
| less host work | step ÷ busy device time |
| fusion | step ÷ time in matrix products |
| bf16, products a share $f$ | between $1/(1 - f/2)$ and 2 |
| larger minibatch | step ÷ (step − fixed cost) |
| checkpointing | a loss: $6/8 = 0.75$ by FLOPs, step ÷ (step + forward) by time |

::: {.d2l-note}
The measurement then tests the diagnosis, not only the change.
:::
:::

::: {.slide title="The Eager Step Is Busy, with a Third of Its Time Outside the Products" only="pytorch"}
[Baseline]{.kicker}

@!fast-transformer-the-baseline-2

- Device busy for nearly the whole step: the host keeps up.
- The third outside the products: short elementwise and normalization
  kernels, bandwidth-bound.
- Fusion can gain at most step ÷ products $= 25.5 / 15.6 \approx 1.6$.
:::

::: {.slide title="The Jitted Step Is Busy, and XLA Has Fused It Already" only="jax"}
[Baseline]{.kicker}

@!fast-transformer-the-baseline-2

- One compiled program per step; elementwise operations fused.
- The device waits 2% of the step: less host work can gain at most 2%.
- The third outside the products: XLA fusions, bandwidth-bound.
:::

::: {.slide title="Compilation Gains Over a Tenth of a Possible 1.6" only="pytorch"}
[Small kernels]{.kicker}

```{.python #fast-transformer-exercises}
compiled = torch.compile(model, dynamic=False, mode='reduce-overhead')
high = model(X)
error = ((compiled(X) - high).norm() / high.norm()).item()
assert error < 1e-4      # same answer first
step = make_step(model, compiled)
```

@!fast-transformer-rungs-each-one-measured-1

- Fusion: device time 24.4 → 22.2 ms. CUDA graphs: idle gaps closed.

::: {.d2l-note}
The products, two thirds of the step, are untouched.
:::
:::

::: {.slide title="nnx.jit Spends Almost 9 ms per Call Walking the Model; Binding Removes It" only="jax"}
[Host overhead]{.kicker}

@!fast-transformer-rungs-each-one-measured-1

- `nnx.jit` walks model and optimizer on every call; `graph=False`
  treats them as trees; `jit_partial` binds them once.
- Checked first: equal losses and weights. Gain about 1%: the walk was
  hidden behind a 24 ms device step.
- Removed anyway: every later change shortens the device's step.

::: {.d2l-note}
The cost belongs to the convenience of walking Python objects, not to
JAX.
:::
:::

::: {.slide title="After the Second Change the Products Take About Two Thirds"}
@!fast-transformer-rungs-each-one-measured-5

- Device busy for nearly the whole step: the host is not the limit.
- The rest is memory traffic between the products.

::: {.d2l-note}
The arithmetic of the matrix products is now the main limit: make the
products faster, with a faster number format.
:::
:::

::: {.slide title="Check: Which Change Comes Next, and How Much Can It Gain?"}
[Check]{.kicker}

The device is busy; matrix products take a share $f$ of two thirds to
seven tenths of the step. Candidates: bf16, a larger minibatch,
activation checkpointing.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** bf16. If only the products halve: $1/(1 - f/2) \approx 1.5$;
if every kernel also halves its bytes: 2. Checkpointing costs time, and
memory is not short.
:::
:::
:::

::: {.slide title="bf16 Speeds Up the Step About 1.75 Times: Not 1.55, Not 2" only="pytorch"}
[Reduced precision]{.kicker}

@!fast-transformer-rungs-each-one-measured-2

- bf16 products and activations; fp32 weights, gradients, Adam moments,
  norms, residual stream, loss. No loss scaling.
- Above 1.55: bf16 products run nearer their peak; fewer bytes between
  them.
- Below 2: norms, stream, casts, and the update still move fp32.

::: {.d2l-note}
Inside the bounds that the second profile set before the run.
:::
:::

::: {.slide title="bf16 Alone Gains About 1.45; Halving the Products Predicts 1.48" only="jax"}
[Reduced precision]{.kicker}

@!fast-transformer-rungs-each-one-measured-2

- `d2l.to_bf16`: bf16 compute in the linear layers; norms, embedding,
  stream in fp32.
- Products at 65%: $1/(1 - 0.325) \approx 1.48$.
- Among the kernels between them: XLA's attention writes all
  $T \times T$ weights to memory ($T$ = 128 positions).

::: {.d2l-note}
The kernels between the products got no faster.
:::
:::

::: {.slide title="cuDNN's Fused Attention Gains About a Third More" only="jax"}
[Reduced precision]{.kicker}

@!fast-transformer-fused-attention

- Same arithmetic; the attention weights never reach device memory.
- Takes 16-bit inputs only, so it comes after bf16.
- The profile counted XLA's attention among the other kernels, so it
  could not foresee this gain.

::: {.d2l-note}
The bandwidth-bound part responds. Together with bf16: nearly 2×.
:::
:::

::: {.slide title="512 Windows per Step Gain a Few Percent at Most"}
[Larger minibatches]{.kicker}

@!fast-transformer-rungs-each-one-measured-3

- Busy device: only fixed work can be spread. AdamW 28 B, weight and
  gradient casts 12 B: $40\,\textrm{B} \times 19\textrm{M} = 0.76$ GB
  per step.
- 0.75 ms of a 12 to 13 ms step: about 5% if nothing else is fixed.

::: {.d2l-note}
Kept because data parallelism needs long steps; not free for training.
:::
:::

::: {.slide title="At Batch 512, Two Fifths of the Step Lie Between the Products"}
[Third profile]{.kicker}

@!fast-transformer-larger-minibatches-1

- Busy throughout; the largest kernels are matrix products.
- The rest: norms, SwiGLU, rotary embeddings, casts. They move bytes and
  leave the tensor cores idle.

::: {.d2l-note}
Two fifths of the step moves bytes, not FLOPs.
:::
:::

::: {.slide title="Checkpointing Costs Less Than the Predicted Quarter"}
[Activation checkpointing]{.kicker}

@!fast-transformer-activation-checkpointing-2

@!fast-transformer-rungs-each-one-measured-4

- FLOPs: the blocks run forward twice, 8 per parameter and token, not 6:
  ratio 0.75.
- Clock: the forward pass takes about a third of the step, so at least
  three quarters remain.
- Measured: above both estimates; memory falls to about a third.

::: {.d2l-note}
The right change only for a model or minibatch that does not fit.
:::
:::

::: {.slide title="Each Change Is Measured on Top of the Last"}
@!fast-transformer-the-waterfall

- About 2× in total, nearly all from faster products and less traffic
  around them.
- MFU = FLOPs per second at 6 per parameter and token ÷ bf16 peak: about
  half.
- Checkpointing: a branch, outside the total.

::: {.d2l-note}
A change gains only where the limit lies: order matters.
:::
:::

::: {.slide title="The Allreduce Should Take 31 ms on Two GPUs and 47 ms on Four"}
[Data parallelism]{.kicker}

$$t_{\textrm{step}}(k) \approx t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k),
\quad t_{\textrm{allreduce}} = \tfrac{2(k-1)}{k}\,\tfrac{S}{\beta_{\textrm{link}}}$$

- $S$ = 76 MB of fp32 gradients; link rate
  $\beta_{\textrm{link}} \approx 2.4$ GB/s through host memory.
- Weak scaling (fixed minibatch per GPU): speedup
  $k\,t_{\textrm{step}}(1) / (t_{\textrm{step}}(1) + t_{\textrm{allreduce}}(k))$.

::: {.d2l-note}
No overlap assumed, so the prediction is a floor.
:::
:::

::: {.slide title="Check: Do Two GPUs Help?"}
[Check]{.kicker}

The allreduce takes 31 ms on two GPUs. Predict the speedup of two GPUs
for a 22 ms step (fp32, batch 64) and for a 95 ms step (bf16, batch 512).

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $2 \cdot 22 / (22 + 31) \approx 0.83$: slower than one GPU.
$2 \cdot 95 / (95 + 31) \approx 1.5$: three quarters of linear. A
second GPU helps only if the allreduce is shorter than a step.
:::
:::
:::

::: {.slide title="Measured Speedups Follow the Cost Model"}
[Data parallelism]{.kicker}

@!fast-transformer-rungs-each-one-measured-8

- Allreduce within a millisecond of the prediction.
- Batch 64: two GPUs lose, four gain little.
- Batch 512 in bf16: four GPUs about 3× one GPU.
- Distance above the floor: overlap with the backward pass.

::: {.d2l-note}
Two predictions of opposite sign, a loss and a gain; both held.
:::
:::

::: {.slide title="The Final Step Trains Like the Baseline"}
[Training check]{.kicker}

@!fast-transformer-training-check

- One initialization, the same 200 minibatches of 512 windows.
- The same loss at step 1, within a tenth of a nat through step 100;
  both validation losses below the bigram model's 2.45.
- Constant rate, no warm-up or clipping: a curve can rise late, so one
  late value settles little.
- A coarse check: a dropped weight decay would pass it.

::: {.d2l-note .warn}
13M tokens, yet more than half a nat above the 1.35 the Transformers
chapter reached after 8M: throughput is not progress.
:::
:::

::: {.slide title="Two Fifths of the Final Step Is Still Memory Traffic"}
[Account]{.kicker}

@!fast-transformer-account

- Per 8,192 tokens: arithmetic at peak 5.7 ms; products about 7 ms, at
  four fifths of peak or more; the kernels between them about 5 ms.

::: {.d2l-note}
Against the baseline: products more than twice as fast, the kernels
between them about two fifths shorter.
:::
:::

::: {.slide title="Check: Where Would the Next Factor of Two Come From?"}
[Check]{.kicker}

Per 8,192 tokens the final step spends about 7 ms in products near their
peak and about 5 ms in other kernels. fp8 doubles the tensor-core rate.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** fp8 shortens only the products: about 1.4×, at most 1.5×.
No kernels between the products at all: 1.7 to 1.8×. Two needs both:
fused norms and activations, narrower activations, kernels for one model,
and faster products. On four GPUs: a faster link.
:::
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Count, predict, measure, explain the difference.
- Bounds from the profile: idle share; time outside the products;
  $1/(1 - f/2)$ to 2 for bf16.
- Faster products and less traffic did nearly all of the 2×; batch 512 a
  few percent; checkpointing a loss, smaller than the predicted quarter.
- Over host memory, two GPUs slow a 22 ms step and speed up a 95 ms one.
- Final step: products near peak, two fifths memory traffic.

**Next:** fp8, kernels for one model, many machines: the Language Models
part.
:::
