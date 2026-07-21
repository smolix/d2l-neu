# Case Study: Making a Transformer Fast
:label:`sec_fast_transformer`

This section teaches nothing new. That is the point. Six sections built a
method — measure, classify the regime, apply the matching fix, re-measure —
and a toolbox: the roofline and the profiler (:numref:`sec_perf_model`),
the hardware that sets the constants (:numref:`sec_hardware`), compilation
(:numref:`sec_compilation`), precision and memory
(:numref:`sec_memory_precision`), and data parallelism
(:numref:`sec_multi_gpu`, :numref:`sec_multi_gpu_concise`). Here we run the
whole loop end to end on a real model — the GPT of :numref:`sec_gpt` — and
take it down a *waterfall*: one rung per technique, each measured, each
attributed to the section that taught it, re-profiling as the bottleneck
moves. The result is a single plot that is the chapter's closing argument:
the method is real, and it compounds.

*Prerequisites: the entire chapter. The subject is* `d2l.GPT` *and*
`d2l.TimeMachine` *from* :numref:`sec_gpt`*, reused verbatim.*

```{.python .input #fast-transformer-case-study-making-a-transformer-fast}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import json
import os
import pathlib
import subprocess
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

torch.set_float32_matmul_precision('high')
# The profiler's CUDA instrumentation must not outlive its cell — without
# this, every timing after the first profile pays a per-launch tax.
os.environ['TEARDOWN_CUPTI'] = '1'
```

```{.python .input #fast-transformer-case-study-making-a-transformer-fast}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import optax
import os
import time

# XLA's NCCL buffer registration fails harmlessly on this P2P-less box;
# opt out rather than let every collective print the warning (see 13.6).
os.environ.setdefault('NCCL_LOCAL_REGISTER', '0')
```

## The Subject
:label:`subsec_ft-subject`

We take ch. 11's GPT at a width where the experiment can be read cleanly.

:begin_tab:`pytorch`
At the `num_hiddens=256` of :numref:`sec_gpt`, the model's matmuls are
small enough that some techniques do not clear the measurement noise —
worse, bf16 actively *hurts*, because width-256 matmuls are too small to
feed the tensor cores and the casting overhead dominates (an instructive
failure, pursued in the exercises). Widening to `num_hiddens=512` (about
19M parameters, six blocks, char-level *Time Machine*, context 128) puts
every rung comfortably above the noise floor. The metric is training
throughput in **tokens per second**, measured with the discipline of
:numref:`sec_perf_model`.
:end_tab:

:begin_tab:`jax`
At the `num_hiddens=256` of :numref:`sec_gpt` the matmuls are small
enough that several rungs would sit inside the measurement noise;
widening to `num_hiddens=512` (about 19M parameters, six blocks,
char-level *Time Machine*, context 128) gives every real effect room to
show. The metric is training throughput in **tokens per second**,
measured with the discipline of :numref:`sec_perf_model`.
:end_tab:

The case study runs in both frameworks, deliberately in parallel: the
same model, the same data, the same metric, the same discipline. But the
ladders diverge exactly where the frameworks do — what a "compile" rung
even means, which fix the diagnosis demands next, how data parallelism is
launched — and that divergence is itself part of the closing lesson: the
*method* transfers; the recipe does not.

```{.python .input #fast-transformer-the-subject-1}
%%tab pytorch
device = d2l.try_gpu()
data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char')
vocab_size = len(data.vocab)

def make_gpt():
    torch.manual_seed(0)
    return d2l.GPT(vocab_size, num_hiddens=512, num_blks=6).to(device)

model = make_gpt()
print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters')

def batches(dm):
    while True:
        for X, Y in dm.train_dataloader():
            if X.shape[0] == dm.batch_size:   # constant shape: no retrace
                yield X.to(device), Y.to(device)
stream = batches(data)
```

```{.python .input #fast-transformer-the-subject-1}
%%tab jax
data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char')
vocab_size = len(data.vocab)

def make_gpt():
    return d2l.GPT(vocab_size, num_hiddens=512, num_blks=6,
                   rngs=nnx.Rngs(0))

model = make_gpt()
n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
print(f'{n_params / 1e6:.1f}M parameters')

def batches(dm, batch_size=None, seed=0):
    """Shuffled minibatches, staged on device once (the corpus is 5 MB)."""
    X, Y = jnp.asarray(dm.X[:dm.num_train]), jnp.asarray(dm.Y[:dm.num_train])
    B = dm.batch_size if batch_size is None else batch_size
    n = (len(X) // B) * B                 # whole batches only: one shape
    key = jax.random.key(seed)
    while True:
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, len(X))[:n]
        for i in range(0, n, B):
            idx = perm[i:i + B]
            yield X[idx], Y[idx]
stream = batches(data)
```

:begin_tab:`pytorch`
Before any rung, the measurement itself must be trustworthy, and two traps
had to be defused here — both worth naming, because each corrupted an
early draft of this section by tens of percent. First, the dataset's
ragged final batch (10,000 windows do not divide by 64) hands
`torch.compile` a new shape once per epoch, and the retrace it triggers
(:numref:`sec_compilation`) lands *inside* a timing window; the stream
above drops the ragged tail so the compiler sees one shape. Second, the
profiler's instrumentation is not free and, by default, does not fully
detach when its `with` block ends — every measurement taken afterwards in
the same process inherits a per-launch tax that hits small-batch
configurations hardest; the `TEARDOWN_CUPTI` line in the imports cell
makes profiling stop costing once it stops running. Both are the
chapter's opening lesson in miniature: *the first thing to profile is
your own experiment.* Beyond that, every measurement warms up long enough
for one-time costs — allocator growth, autotuning, compilation caches,
and whatever clock state the driver is in — to settle, and timed windows
close with a device sync:
:end_tab:

:begin_tab:`jax`
Before any rung, the measurement itself must be trustworthy, and the
traps that corrupted early drafts of the PyTorch tab have JAX
counterparts, only sharper. A changed batch shape here does not cost a
mere retrace — it triggers a full XLA recompilation, seconds long,
*inside* the timing window; the stream above rounds every epoch down to
whole batches so a jitted step compiles once
(:numref:`sec_compilation`'s shape specialization). The stream also
does something the PyTorch tab does not need to: it stages the corpus —
all 5 MB of it — on the device once and shuffles *there*. The book's
stock `tf.data` loader is perfectly adequate for every training loop so
far, but it costs a millisecond-and-change of host work per batch, and
an early draft of this section timed the rungs through it: invisible
next to the un-jitted baseline, it silently taxed the fastest rungs
below by roughly a quarter — the bottleneck had moved *off the GPU and
into the input pipeline*, exactly the failure mode
:numref:`sec_perf_model` warned about. Finally, asynchronous dispatch
means a window that does not close on `block_until_ready` measures how
fast Python *enqueues* work, not how fast the device finishes it, and
the first call of every freshly jitted function pays its compilation —
so every warmup below is long enough to absorb one. Warm up until the
one-time costs settle; close every timed window with a sync:
:end_tab:

```{.python .input #fast-transformer-the-subject-2}
%%tab pytorch
def throughput(step_fn, warmup=60, timed=100):
    """Tokens/s with warmup + device sync (see :numref:`sec_perf_model`)."""
    for _ in range(warmup):
        X, Y = next(stream); step_fn(X, Y)
    torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
    for _ in range(timed):
        X, Y = next(stream); step_fn(X, Y); n += X.numel()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)
```

```{.python .input #fast-transformer-the-subject-2}
%%tab jax
def throughput(step_fn, stream, warmup=60, timed=100):
    """Tokens/s with warmup + sync (see :numref:`sec_perf_model`)."""
    for _ in range(warmup):
        loss = step_fn(*next(stream))
    jax.block_until_ready(loss)
    t0 = time.perf_counter(); n = 0
    for _ in range(timed):
        X, Y = next(stream); loss = step_fn(X, Y); n += X.size
    jax.block_until_ready(loss)
    return n / (time.perf_counter() - t0)
```

:begin_tab:`pytorch`
One note on the metric: `throughput` times the *whole* pipeline —
`next(stream)` includes the `DataLoader` and the host-to-device copy — so
these are end-to-end tokens per second, the number a training run actually
delivers, not a kernels-only figure. (At this scale the input pipeline is
under two percent of a step; we measured that too.)
:end_tab:

:begin_tab:`jax`
One note on the metric: `throughput` times the *whole* pipeline —
`next(stream)` includes the shuffle, the batch gather, and the dispatch
of both — so these are end-to-end tokens per second, the number a
training run actually delivers, not a kernels-only figure. One thing it
deliberately does *not* do is feed the same batch twice: a benchmark
that re-times one cached batch lets the GPU serve every read from cache
and flatters the throughput by double-digit percent (we measured that
too). Honest data is fresh data.
:end_tab:

:begin_tab:`jax`
One measurement has to be taken out of order, and memory is why. Rung 3
will pit a batch-512 step in bf16 against the *same step in fp32* as a
control. That fp32 control's working set is about 15 GiB (its compiler
plan, in Rung 3, confirms it) — nearly this whole card — so on a GPU
already holding the rest of the ladder's compiled programs it simply
will not fit alongside them. The framework preallocates a large pool and
does not hand it back, so once the un-jitted baseline below has grown the
pool, the fp32-512 control can no longer find a contiguous 15 GiB. So we
clock it *now*, on a clean allocator, and keep the number for Rung 3 —
using a throwaway model so nothing downstream is disturbed. That a
measurement's *scheduling* is dictated by memory is itself
:numref:`sec_memory_precision`'s lesson, arriving a little early:
:end_tab:

```{.python .input #fast-transformer-the-subject-3}
%%tab jax
def fp32_step(model, optimizer, X, Y):
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(
            model(X).reshape(-1, vocab_size), Y.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

_m = make_gpt(); _o = nnx.Optimizer(_m, optax.adamw(1e-3), wrt=nnx.Param)
_jit, _big = nnx.jit(fp32_step), batches(data, batch_size=512)
_Xb, _Yb = next(_big)
ctl512 = throughput(lambda X, Y: _jit(_m, _o, X, Y), _big,
                    warmup=20, timed=20)                # fp32 @512, clean pool
plan32_gib = (_jit.lower(_m, _o, _Xb, _Yb).compile()
              .memory_analysis().temp_size_in_bytes / 2**30)
del _m, _o, _jit, _big, _Xb, _Yb; jax.clear_caches()    # release before R0
print(f'fp32 control @512 (clocked up front): {ctl512:.0f} tokens/s, '
      f'plan {plan32_gib:.1f} GiB')
```

## Rung 0: Baseline, Profiled
:label:`subsec_ft-baseline`

:begin_tab:`pytorch`
Eager execution, tf32-fair (matmul precision already `'high'`). Before
optimizing, we *classify*: profile one step and read where the time goes.
:end_tab:

:begin_tab:`jax`
JAX has no serious eager baseline, and it would be dishonest to pretend
otherwise. :numref:`sec_compilation` measured why: an un-jitted step
dispatches every operation — the forward, the hundreds of intermediate
gradients, the optimizer's whole tree of updates — one at a time, and
re-traces the function on every call; eager JAX is a development surface,
not a training mode. (It is also already tf32-fair: on this card JAX's
default matmul precision is tensor-core tf32, the fair baseline
:numref:`sec_memory_precision` established.) But a waterfall wants a
floor, so we measure the un-jitted step once, briefly, to know exactly
what the first rung is worth:
:end_tab:

```{.python .input #fast-transformer-rung-0-baseline-profiled}
%%tab pytorch
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

def step_eager(X, Y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(X).reshape(-1, vocab_size), Y.reshape(-1))
    loss.backward(); opt.step()

for _ in range(3):
    step_eager(*next(stream))
with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], acc_events=True) as prof:
    for _ in range(5):
        step_eager(*next(stream))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=6))
tput0 = throughput(step_eager)
print(f'R0 eager: {tput0:.0f} tokens/s')
```

```{.python .input #fast-transformer-rung-0-baseline-profiled}
%%tab jax
optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

def train_step(model, optimizer, X, Y):
    def loss_fn(model):
        logits = model(X)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, vocab_size), Y.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

tput0 = throughput(lambda X, Y: train_step(model, optimizer, X, Y),
                   stream, warmup=3, timed=10)
print(f'R0 un-jitted: {tput0:.0f} tokens/s')
```

:begin_tab:`pytorch`
Read the profile for the model's character. Matmuls (the attention
projections and the feed-forward blocks) carry about half of the device
time — but only about half: the rest is a long tail of small elementwise
and normalization kernels (note the *hundreds* of `aten::mul` calls per
five steps), each launched separately, none individually visible. And the
CPU-total column is nearly as large as the CUDA total — remembering
:numref:`sec_perf_model`'s caveat that these totals overlap across
threads and streams, that still says the dispatch thread is busy most of
the step. The diagnosis, then: partly compute-bound, but with a fusible
elementwise tail and non-trivial launch traffic — which predicts that
compilation should pay something here, and tensor-core precision after
it. The rungs will check both.
:end_tab:

:begin_tab:`jax`
Read the number for what it is: a price list for a mistake, not a
baseline. Every rung below reports its ratio against the *jitted* step,
because that is the configuration a JAX practitioner actually starts
from — the PyTorch tab's eager baseline is a real place to train; this
one is not. Note also what we did *not* do here: profile the un-jitted
step. There is nothing to classify in it — the diagnosis is "Python
dispatch, wall to wall" by construction — so classification is deferred
one rung, to the first program worth examining: the compiled one.
:end_tab:

## Rungs, Each One Measured
:label:`subsec_ft-rungs`

:begin_tab:`pytorch`
**Rung 1 — compile (:numref:`sec_compilation`).** The profile promised a
fusible tail and busy dispatch; compilation is the matching fix. Before
the timing, the correctness gate — the compiled model must produce the
same logits as the eager one (they agree to about $10^{-2}$; the residue
is tf32 matmuls re-associated by fusion, not a wrong answer). *Same
answer first, then faster:*
:end_tab:

:begin_tab:`jax`
**Rung 1 — jit the whole step (:numref:`sec_compilation`).** The rung is
one transformation: `nnx.jit` swallows the loss, the gradients, *and* the
optimizer update — 13.3's whole-step lesson — and the training step
becomes a single compiled program. This is the framework's table stakes,
not an optimization trick, which is why the waterfall's honest ratios
start here. The correctness gate first: the jitted forward must produce
the same logits as the op-by-op one (they agree to a few times $10^{-4}$;
the residue is tf32 arithmetic re-associated by fusion, not a wrong
answer). *Same answer first, then faster:*
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-1}
%%tab pytorch
compiled = torch.compile(model)
def step_compiled(X, Y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(compiled(X).reshape(-1, vocab_size), Y.reshape(-1))
    loss.backward(); opt.step()

t0 = time.perf_counter(); step_compiled(*next(stream)); torch.cuda.synchronize()
print(f'first (compiling) step: {time.perf_counter() - t0:.1f} s')
X, _ = next(stream)
assert torch.allclose(compiled(X), model(X), atol=1e-2, rtol=1e-3)
tput1 = throughput(step_compiled)
print(f'R1 compiled: {tput1:.0f} tokens/s ({tput1 / tput0:.2f}x)')
```

```{.python .input #fast-transformer-rungs-each-one-measured-1}
%%tab jax
step_jit = nnx.jit(train_step)         # the rung is one transformation

t0 = time.perf_counter()
jax.block_until_ready(step_jit(model, optimizer, *next(stream)))
print(f'first (compiling) step: {time.perf_counter() - t0:.1f} s')
fwd = nnx.jit(lambda m, X: m(X))
X, Y = next(stream)
assert jnp.allclose(fwd(model, X), model(X), atol=1e-2, rtol=1e-3)
tput1 = throughput(lambda X, Y: step_jit(model, optimizer, X, Y), stream)
print(f'R1 jit: {tput1:.0f} tokens/s ({tput1 / tput0:.0f}x)')
```

:begin_tab:`pytorch`
The first call pays about two seconds of compile time; steady state pays
that back at roughly 1.3× — real money, from a model that is *not*
overhead-bound in the :numref:`sec_perf_model` sense. The gain is the
profile's tail, cashed in: the elementwise chains between matmuls fuse
into a handful of generated kernels, and the launch traffic drops with
them. A re-profile confirms the bottleneck moved:
:end_tab:

:begin_tab:`jax`
The first call pays several seconds of XLA compile time; steady state
repays that at roughly twenty-fold — the two-orders-of-magnitude shape
:numref:`sec_compilation` measured on its toy step, compressed here
because this model's kernels are chunky enough to partly hide the
dispatch. Now there is a program worth classifying. The profile the
PyTorch tab reads is one tool; the compiler's own accounting is the more
natural one in JAX (:numref:`sec_memory_precision` used the same trick
for memory): lower the step, compile it, and ask the compiled object
what it plans to execute —
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-5}
%%tab pytorch
with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], acc_events=True) as prof:
    for _ in range(5):
        step_compiled(*next(stream))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
```

```{.python .input #fast-transformer-rungs-each-one-measured-5}
%%tab jax
compiled = step_jit.lower(model, optimizer, X, Y).compile()
flops = compiled.cost_analysis()['flops']
t_step = 64 * 128 / tput1
print(f'XLA counts {flops / 1e9:.0f} GFLOP per step; at {1e3 * t_step:.1f} '
      f'ms per step that is {flops / t_step / 1e12:.0f} TFLOP/s achieved')
```

:begin_tab:`pytorch`
The elementwise tail has folded into `CompiledFunction` regions, and the
matmuls' share of device time has *risen* — the step is now more
tensor-core-bound than before, which is precisely the regime that
precision attacks. So, precision next.
:end_tab:

:begin_tab:`jax`
Put the two numbers against :numref:`tab_gpu_specs`: the compiled step
achieves about a third of the card's ~83 tf32 TFLOP/s. That
classification carries real information, but read it carefully — it is
*ambiguous*. Jit already cured R0's op-by-op dispatch disease, and the
step is not at the compute roof either; the gap could be arithmetic
that is slow (a mixed workload of matmuls, attention softmax,
normalizations, and rope trigonometry idles the tensor cores part of
the time), or it could be a fixed per-step cost that even a fused
program still pays. The cheap experiment that separates the suspects is
the format ladder's next rung: bf16 doubles the tensor-core roof and
halves every activation byte — if arithmetic is the wall, the effect
must show. So, precision next — as a *measurement* as much as an
optimization.
:end_tab:

:begin_tab:`pytorch`
**Rung 2 — bf16 (:numref:`sec_memory_precision`).** Add bf16 autocast on
the forward and loss, with backward outside the context, as
:numref:`sec_memory_precision` prescribed (no `GradScaler` — bf16 shares
fp32's exponent range). At width 512 the matmuls are large enough for the
tensor cores to pay:
:end_tab:

:begin_tab:`jax`
**Rung 2 — bf16, threaded explicitly (:numref:`sec_memory_precision`).**
There is no autocast to lean on: precision in JAX is explicit — you
thread dtypes through the computation yourself, and nothing casts unless
you say so. The apparently obvious move is the functional cast the
builder's guide used on a small stack: split the model, `astype` every
floating array to bf16, merge back into a compute copy (the fp32
originals stay in the model and the optimizer as the master weights, and
no `GradScaler` — bf16 shares fp32's exponent range). Measure it:
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-2}
%%tab pytorch
def step_bf16(X, Y):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = F.cross_entropy(compiled(X).reshape(-1, vocab_size), Y.reshape(-1))
    loss.backward()
    opt.step()
tput2 = throughput(step_bf16)
print(f'R2 +bf16: {tput2:.0f} tokens/s ({tput2 / tput1:.2f}x)')
```

```{.python .input #fast-transformer-rungs-each-one-measured-2}
%%tab jax
def bf16_arrays(model):
    """Cast every floating array of a copy -- surely bf16 now?"""
    graphdef, state = nnx.split(model)
    state = jax.tree.map(lambda x: x.astype(jnp.bfloat16)
                         if jnp.issubdtype(x.dtype, jnp.floating) else x,
                         state)
    return nnx.merge(graphdef, state)

@nnx.jit
def step_naive(model, optimizer, X, Y):
    def loss_fn(model):
        logits = bf16_arrays(model)(X)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, vocab_size).astype(jnp.float32),
            Y.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

naive = throughput(lambda X, Y: step_naive(model, optimizer, X, Y), stream)
print(f'"bf16", naive cast: {naive:.0f} tokens/s ({naive / tput1:.2f}x)')
print('the receipt:', bf16_arrays(model).token_emb(X).dtype)
```

:begin_tab:`jax`
Flat — and the second print is the receipt. The embedding layer still
emits fp32: flax modules record a *compute dtype* at construction
(`Embed` remembers its `param_dtype`), and their `__call__` promotes
inputs and parameters to it — so the bf16 table we so carefully installed
was promoted straight back to fp32 at the first layer, and everything
downstream followed. Our "bf16 rung" measured a wardrobe change, not a
precision change. This is the JAX tab's version of the PyTorch tab's
corrupted-measurement traps, and it is nastier for being *silent*: no
error, no warning, a plausible number. The measurement — plus one dtype
print — caught it. Threading bf16 for real takes two more moves. First,
set the compute dtype the modules remember, not just the arrays. Second,
one leak is in the model's own arithmetic: rope's fp32 trigonometry
promotes the query and key back to fp32 on their way into attention —
and the attention kernel, rightly, refuses mixed-precision inputs
(loudly, this time) — so a two-line subclass casts them back. The fp32
masters stay in the model and the optimizer; the gradients arrive fp32 on
their own, because the transpose of a cast is a cast back:
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-9}
%%tab jax
def to_bf16(model):
    """A bf16 compute copy: cast every floating array AND the compute
    dtype each flax module remembers from construction."""
    copy = bf16_arrays(model)
    for _, m in nnx.iter_graph(copy):
        if isinstance(m, nnx.Module) and hasattr(m, 'dtype'):
            m.dtype = jnp.bfloat16
    return copy

class Bf16GPT(d2l.GPT):
    """ch. 11's GPT with one dtype leak patched: rope's fp32 trigonometry
    promotes q and k back to fp32 (and the attention kernel rightly
    refuses mixed inputs), so cast them back after rope."""
    class CausalAttention(d2l.GPT.CausalAttention):
        def __call__(self, X, *_):
            B, T, D = X.shape
            q, k, v = jnp.split(self.W_qkv(X), 3, axis=-1)
            q, k, v = (u.reshape(B, T, self.num_heads, -1)
                       for u in (q, k, v))
            if self.rope:
                q, k = self._rope(q), self._rope(k)
            q, k = q.astype(v.dtype), k.astype(v.dtype)
            Y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
            return self.W_o(Y.reshape(B, T, D)), None

def make_bf16_gpt():
    return Bf16GPT(vocab_size, num_hiddens=512, num_blks=6,
                   rngs=nnx.Rngs(0))

model16 = make_bf16_gpt()
opt16 = nnx.Optimizer(model16, optax.adamw(1e-3), wrt=nnx.Param)

@nnx.jit
def step_bf16(model, optimizer, X, Y):
    def loss_fn(model):
        logits = to_bf16(model)(X)         # bf16 compute copy of the step;
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, vocab_size).astype(jnp.float32),  # fp32 loss
            Y.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)   # grads arrive fp32
    optimizer.update(model, grads)
    return loss

print('the receipt, again:', to_bf16(model16).token_emb(X).dtype)
plan16 = step_bf16.lower(model16, opt16, X, Y).compile().memory_analysis()
print(f'planned temp: bf16 {plan16.temp_size_in_bytes / 2**20:.0f} MiB '
      f'(fp32: {compiled.memory_analysis().temp_size_in_bytes / 2**20:.0f} '
      f'MiB)')
tput2 = throughput(lambda X, Y: step_bf16(model16, opt16, X, Y), stream)
assert all(p.dtype == jnp.float32                      # fp32 masters intact
           for p in jax.tree.leaves(nnx.state(model16, nnx.Param)))
print(f'R2 +bf16, threaded: {tput2:.0f} tokens/s ({tput2 / tput1:.2f}x)')
```

:begin_tab:`jax`
The second surprise, and the better lesson: bf16 is now provably *real* —
the receipt prints `bfloat16`, the compiler's planned temporaries drop by
about two-fifths — and the throughput *still* barely moves, nowhere near
the factor the format ladder promised. Diagnose before despairing. The
device-side work genuinely shrank — the plan says so; what did not
shrink is everything around it — the per-call Python that walks the
module graph, the
stream's dispatches, the launch path — a fixed per-step toll that the
fp32 step was already brushing against. At batch 64 this step is
*overhead-bound* in exactly :numref:`sec_perf_model`'s sense, and making
the arithmetic faster cannot move a wall made of dispatch. (The tell,
if you re-run this notebook: the ratio above wobbles from run to run —
deterministic device work does not do that; host state does.) The
matching fix is not more precision. It is *amortization*: make every
step carry more tokens, so the fixed toll is divided by a bigger
number. That is the next rung — where the bf16 investment, correct all
along but bought where the wall wasn't, will finally cash.
:end_tab:

:begin_tab:`pytorch`
**Rung 3 — raise the batch, raise the intensity
(:numref:`sec_memory_precision`, :numref:`sec_perf_model`).** A bigger
per-device batch raises the matmuls' arithmetic intensity, climbing the
roofline toward the compute roof — a rung in its own right, whatever the
precision. Memory is its usual price, so we also run a control: a few
fp32 steps at batch 512, to see what the batch *would* cost without bf16.
The control shows fp32 at 512 still fits this 24 GB card — snugly — so
bf16 did not unlock this batch; what it bought is *headroom*, roughly
half the footprint, which becomes the difference between fitting and not
the moment the model, context, or batch grows again:
:end_tab:

:begin_tab:`jax`
**Rung 3 — raise the batch
(:numref:`sec_memory_precision`, :numref:`sec_perf_model`).** A bigger
per-device batch is the roofline's classic rung — and in this tab it is
also the direct answer to R2's diagnosis: eight times the tokens per
step divides the fixed host toll by eight. Two controls come along, so
the win can be attributed honestly. The compiler's plan prices the
memory of the bf16 and fp32 steps before anything runs
(:numref:`sec_memory_precision`'s trick), and the fp32 step also *runs*
at 512, to separate what the batch buys from what bf16 buys. (A new
shape is a new program, :numref:`sec_compilation` — the warmups below
each absorb one more compile.)
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-3}
%%tab pytorch
big = d2l.TimeMachine(batch_size=512, num_steps=128, tokenization='char')
big_stream = batches(big)

def throughput_big(step_fn, warmup=60, timed=50):
    for _ in range(warmup):
        step_fn(*next(big_stream))
    torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
    for _ in range(timed):
        X, Y = next(big_stream); step_fn(X, Y); n += X.numel()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)

torch.cuda.reset_peak_memory_stats()      # control: batch 512 in fp32
for _ in range(3):
    step_eager(*next(big_stream))
mem_fp32 = torch.cuda.max_memory_allocated() / 2**30
torch.cuda.reset_peak_memory_stats()
tput3 = throughput_big(step_bf16)
print(f'R3 +batch-up (512): {tput3:.0f} tokens/s ({tput3 / tput2:.2f}x), '
      f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB '
      f'(fp32 control: {mem_fp32:.1f} GiB)')
```

```{.python .input #fast-transformer-rungs-each-one-measured-3}
%%tab jax
big_stream = batches(data, batch_size=512)   # same corpus, bigger gathers
Xb, Yb = next(big_stream)

# The bf16-512 step (~9 GiB) fits alongside the pool the ladder has already
# grown; the fp32-512 control (~15 GiB) did not, which is why it was clocked
# up front (`ctl512`, `plan32_gib`). Here we run bf16-512 and read its plan.
tput3 = throughput(lambda X, Y: step_bf16(model16, opt16, X, Y),
                   big_stream, warmup=30, timed=30)
c16 = step_bf16.lower(model16, opt16, Xb, Yb).compile()
plan16_gib = c16.memory_analysis().temp_size_in_bytes / 2**30
tf512 = c16.cost_analysis()['flops'] / (512 * 128 / tput3) / 1e12
print(f'R3 +batch-up (512): {tput3:.0f} tokens/s ({tput3 / tput2:.2f}x), '
      f'{tf512:.0f} TFLOP/s achieved')
print(f'fp32 control at 512: {ctl512:.0f} tokens/s '
      f'-> bf16 pays {tput3 / ctl512:.2f}x here')
print(f'planned temp: bf16 {plan16_gib:.1f} GiB '
      f'(fp32 control: {plan32_gib:.1f} GiB)')
```

:begin_tab:`pytorch`
Another rung, another re-profile. At batch 512 the average matmul runs
several times longer per call — bigger tiles, better tensor-core
utilization — and the launch-and-dispatch side has receded to a footnote:
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-6}
%%tab pytorch
with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], acc_events=True) as prof:
    for _ in range(5):
        step_bf16(*next(big_stream))
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
```

:begin_tab:`jax`
Read the three printed lines together, because they close R2's case and
then break a lazy assumption. The bf16 step at batch 512 delivers most
of a factor of two over the stalled batch-64 rung, and the achieved
TFLOP/s roughly doubles — the tensor cores are finally earning their
silicon. But look at the fp32 control: at batch 512 it is *slower than
fp32 at batch 64*. The batch rung, taken alone, is a *negative* rung
for the fp32 program — its compute rate does not rise, holding around
twenty TFLOP/s, no better than the batch-64 step — and the plan says
why: roughly 15 GiB of temporaries,
dominated by materialized fp32 attention buffers, and a plan that large
is not just occupancy, it is *traffic* — the fp32 step crosses into the
bandwidth regime just as the host toll recedes. So neither rung pays by
itself here: bf16 at batch 64 hit the dispatch wall; batch-up at fp32
hits the memory wall; *together* they pay, because bf16 halves the
bytes exactly where the bigger batch needs it, and the bigger batch
amortizes the dispatch exactly where bf16 needs it. Techniques
interact. That is why the waterfall is cumulative, why every rung is
re-measured on top of the last — and why a recipe of independent "tips"
would have called both of these rungs wrong.
:end_tab:

:begin_tab:`pytorch`
**Rung 4 — activation checkpointing, and why it does *not* help here
(:numref:`sec_memory_precision`).** This is a deliberate *negative* rung.
Checkpointing trades compute for memory — but at this scale memory is not
the binding constraint (the bf16 batch-512 step fits in about a third of
the card). Recomputing activations therefore only *costs* time and saves
memory we did not need. Measuring the loss is the lesson: knowing when a
technique does not apply is as much the method as knowing when it does.
The wrapper below mirrors `GPT.forward` exactly, checkpointing each
block — and because a mirror can drift, we assert it is one before timing
anything:
:end_tab:

:begin_tab:`jax`
**Rung 4 — activation checkpointing, and why it does *not* help here
(:numref:`sec_memory_precision`).** This is a deliberate *negative* rung.
Checkpointing trades compute for memory — but the compiler's plan just
told us memory is not the binding constraint: the bf16 batch-512 step
wants well under half the card. Recomputing activations therefore only
*costs* time and saves memory we did not need. Measuring the loss is the
lesson: knowing when a technique does not apply is as much the method as
knowing when it does. The wrapper below mirrors `GPT.__call__` exactly,
running each block under `jax.checkpoint`
(:numref:`sec_memory_precision`) via the same `nnx.split`/`nnx.merge`
functional idiom as the bf16 copy — and because a mirror can drift, we
assert it is one before timing anything:
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-4}
%%tab pytorch
class CheckpointedGPT(nn.Module):
    """ch. 11's GPT with its transformer blocks run under checkpointing.
    Reuses the exact submodules; only the block loop differs."""
    def __init__(self, gpt):
        super().__init__(); self.gpt = gpt
    def forward(self, X):
        m = self.gpt                              # mirror GPT.forward exactly,
        H = m.token_emb(X)                        # but checkpoint each block
        if m.pos == 'learned':                    # (default 'rope' adds no emb)
            H = H + m.pos_emb(torch.arange(X.shape[1], device=X.device))
        for blk in m.blks:
            H = ckpt.checkpoint(blk, H, use_reentrant=False)
        return F.linear(m.norm(H), m.token_emb.weight)

ckpt_gpt = CheckpointedGPT(model)
X, _ = next(big_stream)
assert torch.allclose(ckpt_gpt(X), model(X), atol=1e-6)   # an exact mirror

ckpt_model = torch.compile(ckpt_gpt)
def step_ckpt(X, Y):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = F.cross_entropy(ckpt_model(X).reshape(-1, vocab_size), Y.reshape(-1))
    loss.backward()
    opt.step()
torch.cuda.reset_peak_memory_stats()
tput4 = throughput_big(step_ckpt)
print(f'R4 +checkpoint: {tput4:.0f} tokens/s ({tput4 / tput3:.2f}x), '
      f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')
```

```{.python .input #fast-transformer-rungs-each-one-measured-4}
%%tab jax
def ckpt_forward(model, X):
    """ch. 11's GPT.__call__ with each block under jax.checkpoint.
    Reuses the exact submodules; only the block loop differs."""
    H = model.token_emb(X)
    if model.pos == 'learned':                # (default 'rope' adds no emb)
        H = H + model.pos_emb(jnp.arange(X.shape[1]))
    for blk in model.blks:
        graphdef, state = nnx.split(blk)
        H = jax.checkpoint(
            lambda state, H: nnx.merge(graphdef, state)(H))(state, H)
    return model.token_emb.attend(model.norm(H))

half = to_bf16(model16)
assert jnp.allclose(ckpt_forward(half, Xb), half(Xb), atol=1e-6)  # a mirror

@nnx.jit
def step_ckpt(model, optimizer, X, Y):
    def loss_fn(model):
        logits = ckpt_forward(to_bf16(model), X)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, vocab_size).astype(jnp.float32),
            Y.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

plan_ck = step_ckpt.lower(model16, opt16, Xb, Yb).compile().memory_analysis()
tput4 = throughput(lambda X, Y: step_ckpt(model16, opt16, X, Y),
                   big_stream, warmup=30, timed=30)
print(f'R4 +checkpoint: {tput4:.0f} tokens/s ({tput4 / tput3:.2f}x), '
      f'planned temp {plan_ck.temp_size_in_bytes / 2**30:.1f} GiB')
```

:begin_tab:`pytorch`
**Rung 5 — data parallel across 2–4 GPUs
(:numref:`sec_multi_gpu_concise`), predicted first.** Before measuring, we
*predict*, using the accounting of :numref:`sec_multi_gpu`. This GPT has
about 19M parameters, so roughly 76 MB of gradients must allreduce every
step; the NCCL collective on our host-staged box sustains around five
GB/s per device in :numref:`sec_multi_gpu`'s bytes-per-device convention
(the cell prices with 4.5, toward the conservative end of its run-to-run
range; NCCL's own "busbw" reads ~2 GB/s — the default-fallback figure:
:numref:`sec_multi_gpu_concise` measures a five-fold-faster configured
mode, and shows why these runs nonetheless keep the library's defaults), which
puts the allreduce at
roughly the same tens of milliseconds as the compute itself — a
transformer's parameters are
proportional to its compute, so unlike a convolutional ResNet it offers
little extra compute to hide communication behind. Summing the two terms,
:eqref:`eq_dp_cost` predicts a *weak* two-GPU gain: scarcely more than one
single-GPU throughput at $k=2$, under half of linear at $k=4$. One
refinement before trusting the measurement: the serial sum is a *floor* —
DDP overlaps its bucketed allreduce under backward
(:numref:`fig_ddp_overlap`), so the measurement can land above the
serial prediction, and how far above measures the overlap. We reuse
13.6's sidecar idiom on this very model (eager, per-rank batch 64 — weak
scaling, the convention of :numref:`subsec_mg-accounting`; each rank
streams its own batches, so this is a throughput measurement, not a
convergence claim):
:end_tab:

:begin_tab:`jax`
**Rung 5 — data parallel across 2–4 GPUs
(:numref:`sec_multi_gpu_concise`), predicted first.** Before measuring, we
*predict*, using the accounting of :numref:`sec_multi_gpu`. This GPT has
about 19M parameters, so roughly 76 MB of gradients must be summed across
devices every step — and :numref:`sec_multi_gpu` measured the very
collective JAX will run, `psum` through NCCL on this host-staged box, at
about 4.5 GB/s per device in its bytes-per-device convention. That prices
the allreduce at the same tens of milliseconds as the *entire* compute
step: a transformer's parameters are proportional to its compute, so
unlike a convolutional ResNet it offers little extra compute to hide
communication behind. Summing the two terms, :eqref:`eq_dp_cost` predicts
something blunt — at $k=2$, barely one single-GPU throughput: the second
GPU buys nearly nothing; under half of linear at $k=4$. The mechanism,
though, is 13.6's declarative recipe at its cleanest: build a mesh,
replicate the state, shard the batch, and feed the **unchanged** jitted
step — no launcher, no sidecar script, no rank files; one process, three
annotations. (One version note: the mesh names its axis `Auto` — under
this JAX version's explicit-sharding types the embedding's gather cannot
infer an output layout, and `Auto` hands the whole layout problem back to
GSPMD, which is exactly 13.6's semantics.) Per-device batch 64 — weak
scaling, the convention of :numref:`subsec_mg-accounting`; fresh shuffled
batches each step, so this is a throughput measurement, not a convergence
claim:
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-7}
%%tab pytorch
GPT_DDP = r'''
import json, os, time, torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from d2l import torch as d2l

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank); dist.init_process_group("nccl")
torch.set_float32_matmul_precision("high")

data = d2l.TimeMachine(batch_size=64, num_steps=128, tokenization='char')
torch.manual_seed(0)
model = d2l.GPT(len(data.vocab), num_hiddens=512, num_blks=6).to(rank)
model = DDP(model, device_ids=[rank])
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
torch.manual_seed(rank)                 # each rank streams its own batches

def batches():
    while True:
        for X, Y in data.train_dataloader():
            yield X.to(rank), Y.to(rank)
stream = batches()

def step(X, Y):
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(X).reshape(-1, len(data.vocab)),
                           Y.reshape(-1))
    loss.backward(); opt.step()

for _ in range(30):                     # warmup: clocks, caches, NCCL
    step(*next(stream))
torch.cuda.synchronize(); t0 = time.time(); n = 0
for _ in range(50):
    X, Y = next(stream); step(X, Y); n += X.numel()
torch.cuda.synchronize()
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, f"rank{rank}.json"), "w") as f:
    json.dump({"tokens_per_s": n / (time.time() - t0)}, f)
dist.destroy_process_group()
'''
GPT_DIR = pathlib.Path('ddp_gpt')
GPT_DIR.mkdir(exist_ok=True)
_ = (GPT_DIR / 'train_gpt_ddp.py').write_text(GPT_DDP)

def ddp_gpt_tokens(k):
    for stale in GPT_DIR.glob('rank*.json'):
        stale.unlink()
    torchrun = str(pathlib.Path(sys.executable).parent / 'torchrun')
    subprocess.run([torchrun, '--standalone', f'--nproc-per-node={k}',
                    str(GPT_DIR / 'train_gpt_ddp.py')], check=True)
    total = 0.0
    for r in range(k):
        with open(GPT_DIR / f'rank{r}.json') as f:
            total += json.load(f)['tokens_per_s']
    return total
```

```{.python .input #fast-transformer-rungs-each-one-measured-7}
%%tab jax
def dp_setup(k, make_model):
    """13.6's recipe: replicate the state, shard the batch, jit unchanged."""
    mesh = jax.make_mesh((k,), ('data',),
                         axis_types=(jax.sharding.AxisType.Auto,))
    P = jax.sharding.PartitionSpec
    repl = jax.sharding.NamedSharding(mesh, P())
    shard = jax.sharding.NamedSharding(mesh, P('data'))
    mdl = make_model()
    opt = nnx.Optimizer(mdl, optax.adamw(1e-3), wrt=nnx.Param)
    nnx.update(mdl, jax.device_put(nnx.state(mdl), repl))
    nnx.update(opt, jax.device_put(nnx.state(opt), repl))
    return mdl, opt, shard

def dp_stream(src, k, shard):
    """Global batches: k per-device batches, split along the batch axis."""
    while True:
        XY = [next(src) for _ in range(k)]
        yield (jax.device_put(jnp.concatenate([x for x, _ in XY]), shard),
               jax.device_put(jnp.concatenate([y for _, y in XY]), shard))

def dp_tokens(k, step, make_model, src, warmup=20, timed=40):
    mdl, opt, shard = dp_setup(k, make_model)
    return throughput(lambda X, Y: step(mdl, opt, X, Y),
                      dp_stream(src, k, shard), warmup=warmup, timed=timed)
```

```{.python .input #fast-transformer-rungs-each-one-measured-8}
%%tab pytorch
n_bytes = 4 * sum(p.numel() for p in model.parameters())
beta = 4.5e9   # NCCL allreduce, effective bytes/device/s on this box (13.5)
t_comm, t_cmp = 2 * n_bytes / beta, 64 * 128 / tput0
print(f'per step: t_comm ~ {1e3 * t_comm:.0f} ms vs '
      f't_compute ~ {1e3 * t_cmp:.0f} ms')
torch.cuda.empty_cache()               # hand the ranks the parent's cache
for k in (k for k in (2, 4) if k <= d2l.num_gpus()):
    floor = k * 64 * 128 / (t_cmp + t_comm)
    meas = ddp_gpt_tokens(k)
    print(f'R5 k={k}: no-overlap floor {floor / 1e3:.0f}k tokens/s, '
          f'measured {meas / 1e3:.0f}k ({meas / tput0:.2f}x of one GPU)')
```

```{.python .input #fast-transformer-rungs-each-one-measured-8}
%%tab jax
n_bytes = 4 * n_params
beta = 4.5e9   # psum, effective bytes/device/s on this box (13.5)
t_comm, t_cmp = 2 * n_bytes / beta, 64 * 128 / tput1
print(f'per step: t_comm ~ {1e3 * t_comm:.0f} ms vs '
      f't_compute ~ {1e3 * t_cmp:.0f} ms')
for k in (k for k in (2, 4) if k <= jax.local_device_count()):
    floor = k * 64 * 128 / (t_cmp + t_comm)
    meas = dp_tokens(k, step_jit, make_gpt, stream)
    print(f'R5 k={k}: no-overlap floor {floor / 1e3:.0f}k tokens/s, '
          f'measured {meas / 1e3:.0f}k ({meas / tput1:.2f}x of one GPU)')
```

:begin_tab:`pytorch`
The measurement lands where the model said it would — pinned to the
floor, nowhere near the linear ceiling. At $k=2$ it sits somewhat above
the floor: DDP's overlapped buckets buy back part of the communication
time. At $k=4$ it slips a little *below* it: the floor was priced with
the $\beta$ we measured at two GPUs, and four ranks staging through the
same host path do not quite sustain that constant. Either way, data
parallelism on this fabric turns a second GPU into a modest fraction of
one, and four GPUs into well under half of linear. Communication-hungry
is the diagnosis, and it was written down *before* the launch. On an NVLink
box the same accounting predicts near-linear scaling — the constant
changes, the method does not. Note what this rung is *not*: it uses the
eager baseline, matching the prediction's inputs; stacking DDP on the
compiled-bf16-batch-512 configuration is the exercise, and thinking
through why its efficiency comes out *higher* is the point of doing it.
:end_tab:

:begin_tab:`jax`
The measurement lands where the model said it would — at the floor,
nowhere near the linear ceiling — and in our runs a shade *under* it. At
$k=2$ the second GPU buys nothing at all; the total can even dip below
one GPU's throughput, because the floor is a model and models omit
terms: the global batch stages through one device before scattering,
and the collective must still be scheduled. At $k=4$ the shortfall
widens — the floor was priced with the $\beta$ measured at two GPUs, and
four ranks hammering the same host path do not sustain that constant
(the PyTorch tab's sidecar sees the same sag). Nor does overlap rescue
it: XLA emits overlap-capable collectives, but overlap needs compute to
hide behind, and this rung has none to spare. Communication-hungry is
the diagnosis, and it was written down *before* anything ran. On an
NVLink box the same accounting predicts near-linear scaling — the
constant changes, the method does not. And here the one-process,
annotate-the-layout model pays a dividend the sidecar cannot: stacking
data parallelism on the *full* ladder — bf16, batch 512 — is one more
loop over the same helpers, so we run the measurement the PyTorch tab
leaves as an exercise. Eight-fold more compute per step, the same 76 MB
gradient bill: :eqref:`eq_dp_cost` promises much better efficiency —
:end_tab:

```{.python .input #fast-transformer-rungs-each-one-measured-10}
%%tab jax
for k in (k for k in (2, 4) if k <= jax.local_device_count()):
    meas = dp_tokens(k, step_bf16, make_bf16_gpt, big_stream,
                     warmup=15, timed=25)
    print(f'fast config, k={k}: {meas / 1e3:.0f}k tokens/s '
          f'({meas / (k * tput3):.0%} weak-scaling efficiency)')
```

:begin_tab:`jax`
— and delivers it: the fast configuration scales at markedly higher
efficiency than the batch-64 rung, for the reason the cost model states:
$t_{\text{compute}}$ grew eight-fold while $t_{\text{comm}}$ stayed
fixed. Still well short of linear — the fabric is what it is — but the
*shape* of the improvement was predicted by the same two-term sum that
predicted the failure.
:end_tab:

## The Waterfall
:label:`subsec_ft-waterfall`

:begin_tab:`pytorch`
Collect the single-GPU rungs into one plot — the chapter's closing image.
The waterfall is *cumulative*: each bar inherits every choice to its
left, so a bar's ratio to its neighbor isolates one technique while its
ratio to the first bar prices the whole stack.
:end_tab:

:begin_tab:`jax`
Collect the single-GPU rungs into one plot — the chapter's closing image.
The waterfall is *cumulative*: each bar inherits every choice to its
left, so a bar's ratio to its neighbor isolates one technique. The
labels are ratios to the *jitted* bar — the honest baseline here — with
the un-jitted bar left in the plot as a reminder of what the dishonest
one would look like.
:end_tab:

```{.python .input #fast-transformer-the-waterfall}
%%tab pytorch
rungs = ['R0\neager', 'R1\ncompile', 'R2\n+bf16', 'R3\n+batch', 'R4\n+ckpt']
tputs = [tput0, tput1, tput2, tput3, tput4]
print(f'cumulative, R0 -> R3: {tputs[3] / tputs[0]:.2f}x')
d2l.plt.figure(figsize=(6, 3.5))
bars = d2l.plt.bar(rungs, [t / 1e3 for t in tputs],
                   color=['#7f7f7f', '#1f77b4', '#1f77b4', '#2ca02c', '#d62728'])
d2l.plt.ylabel('throughput (k tokens/s)')
d2l.plt.title('Making a Transformer fast: cumulative rungs')
for b, t in zip(bars, tputs):
    d2l.plt.text(b.get_x() + b.get_width() / 2, t / 1e3,
                 f'{t / tputs[0]:.1f}x', ha='center', va='bottom')
d2l.plt.show()
```

```{.python .input #fast-transformer-the-waterfall}
%%tab jax
rungs = ['R0\nun-jitted', 'R1\njit', 'R2\n+bf16', 'R3\n+batch', 'R4\n+ckpt']
tputs = [tput0, tput1, tput2, tput3, tput4]
print(f'cumulative, R1 -> R3: {tputs[3] / tputs[1]:.2f}x '
      f'(and R0 -> R3: {tputs[3] / tputs[0]:.0f}x)')
d2l.plt.figure(figsize=(6, 3.5))
bars = d2l.plt.bar(rungs, [t / 1e3 for t in tputs],
                   color=['#7f7f7f', '#1f77b4', '#1f77b4', '#2ca02c', '#d62728'])
d2l.plt.ylabel('throughput (k tokens/s)')
d2l.plt.title('Making a Transformer fast: cumulative rungs')
for b, t in zip(bars, tputs):
    d2l.plt.text(b.get_x() + b.get_width() / 2, t / 1e3,
                 f'{t / tputs[1]:.2f}x', ha='center', va='bottom')
d2l.plt.show()
```

:begin_tab:`pytorch`
Read left to right, and let the measurements calibrate the intuition.
Compilation paid about 1.3× — on a model that was *not* overhead-bound —
by fusing the elementwise tail the profile exposed. Bf16 paid about 1.4×
on top, the tensor cores earning their silicon now that the matmuls are
wide enough to feed them. Raising the batch bought roughly another 1.3×
by climbing the roofline. And checkpointing, the red bar, *cost* about
a tenth of the throughput while cutting peak memory to around a third
(about 9 GiB down to 3) — a negative rung for *speed*, because memory was
never the binding constraint at this scale, even though it did exactly
what it promised for memory. The cumulative single-GPU speedup, baseline
to the batch-up rung, is the number the cell prints — about 2.4× in our
runs — and every increment traces to a section of this chapter. That the
pieces compose; that no single rung dominated but three modest ones
multiplied; that a technique can *cost* time when its constraint does not
bind; and that the largest errors we found while building this section
were in the *measurements*, not the model — that is the whole lesson:
**you cannot copy a recipe — you measure, classify, fix, and
re-measure.**
:end_tab:

:begin_tab:`jax`
Read left to right, and notice how different this waterfall's shape is
from the PyTorch tab's — same model, same data, same card. Jit is the
cliff: around twenty-fold, the rung that is not optional. Then the flat
bar that taught the most: bf16, threaded until it was provably real, and
still worth almost nothing at batch 64 — not because the tensor cores
idled by fault of the format, but because the step's wall had moved to
the fixed per-step dispatch work, where precision cannot reach. The
batch rung turns that diagnosis into the ladder's second real win, most
of a factor of two — an *interaction*, the fp32 control showed, since
either rung alone loses on this stack. And
checkpointing, the red bar, *cost* about a tenth of the throughput
while cutting the planned temporaries several-fold — a negative rung for
*speed*, because memory was never the binding constraint here. The
cumulative figure the cell prints runs from the honest baseline: jitted
to batch-up lands at about 1.7× (the print keeps the un-jitted
strawman's thirty-odd-fold for the record; the plot's labels do not).
Two tabs,
two different bottleneck sequences, two different winning moves — and
the same loop found both. That is the whole lesson: **you cannot copy a
recipe — not even from the other tab of the same book — you measure,
classify, fix, and re-measure.**
:end_tab:

One closing check, because a waterfall of throughputs proves nothing
about learning: train the fast configuration — compiled, bf16, batch
512 — for a few hundred steps from a fresh initialization and watch the
loss, smoothed over twenty steps, actually fall. (On this tiny corpus a
few hundred big-batch steps revisit the text over a hundred times, so the
loss falls very far — the assertion, not the final value, is the point.)
Speed that breaks the model is not speed:

```{.python .input #fast-transformer-the-waterfall-1}
%%tab pytorch
learner = make_gpt()
learner_c = torch.compile(learner)
opt_l = torch.optim.AdamW(learner.parameters(), lr=1e-3)
losses = []
for _ in range(300):
    X, Y = next(big_stream)
    opt_l.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = F.cross_entropy(learner_c(X).reshape(-1, vocab_size),
                               Y.reshape(-1))
    loss.backward(); opt_l.step()
    losses.append(loss.detach())       # no host read in the hot loop (13.1)
losses = torch.stack(losses).cpu()
first, last = losses[:20].mean(), losses[-20:].mean()
print(f'smoothed loss: first 20 steps {first:.2f} -> last 20 steps {last:.2f}')
assert last < first, 'speed that breaks the model is not speed'
```

```{.python .input #fast-transformer-the-waterfall-1}
%%tab jax
learner = make_bf16_gpt()
opt_l = nnx.Optimizer(learner, optax.adamw(1e-3), wrt=nnx.Param)
losses = []
for _ in range(300):
    X, Y = next(big_stream)
    losses.append(step_bf16(learner, opt_l, X, Y))  # no host read (13.1)
losses = jnp.stack(losses)
first, last = losses[:20].mean(), losses[-20:].mean()
print(f'smoothed loss: first 20 steps {first:.2f} -> last 20 steps {last:.2f}')
assert last < first, 'speed that breaks the model is not speed'
```

## The Lore, and the Ladder Beyond
:label:`subsec_ft-lore`

The competitive edge of this method is a spectator sport. The
*modded-nanoGPT* speedrun tracks the wall-clock time to train a small GPT
to a target loss, and its record holders read like this chapter's table of
contents stacked to the ceiling: compiled and fused kernels
(:numref:`sec_compilation`), block-sparse and FlashAttention variants
(:numref:`sec_attention-at-scale`), a better optimizer
(:numref:`sec_muon`), aggressive low precision down to fp8 — each record a
new rung on a waterfall like ours. fp8 tensor cores exist on our Ada GPUs,
but the recipe for *training* in fp8 (its own loss-scaling and master-copy
discipline) is deferred to the Language Models part, along with the
multi-node parallelism where these techniques are stacked at frontier
scale.

That is where the chapter ends, seven sections deep into a single idea.
The machine has two budgets, arithmetic and bandwidth; a program lives in
one of three regimes; and the way to make anything fast is always the same
loop — **measure, classify, fix, re-measure** — now demonstrated on a real
model, one rung at a time.

## Summary

:begin_tab:`pytorch`
* The chapter's method applied whole: profile the baseline, classify the
  regime, apply one technique per rung, re-profile. On ch. 11's GPT
  (width 512, ~19M params), compile paid about 1.3×, bf16 about 1.4× on
  top, a bigger batch roughly another 1.3× — no single dominant win, but
  three modest ones multiplying to about 2.4×, with the waterfall cell
  printing the exact cumulative figure.
* Every rung is attributed to a section, and the bottleneck is
  re-profiled as it moves: after compilation the elementwise tail is
  fused away and the matmuls' share rises, which is what makes precision
  the next fix.
* Measurement is a technique too, and it can fail: a ragged final batch
  put a `torch.compile` retrace inside our timing window, and the
  profiler's instrumentation outlived its cell until told otherwise.
  Both had to be fixed before the waterfall meant anything.
* Two rungs teach by *not* helping: bf16 is a negative rung at width 256
  (matmuls too small for the tensor cores), and activation checkpointing
  is a negative rung here (memory is not the binding constraint). Knowing
  when a technique does not apply is the method too.
* Data parallelism is predicted before it is measured: a transformer's
  parameters scale with its compute, so DP on our host-staged fabric is
  communication-hungry — the measured throughput lands at the cost
  model's no-overlap floor (a shade above it at two GPUs, a shade below
  at four, where the host-staged fabric no longer sustains its two-GPU
  bandwidth), nowhere near the linear ceiling. The
  prediction-then-measurement is the demonstration.
:end_tab:

:begin_tab:`jax`
* The chapter's method applied whole: measure the baseline, classify
  with the compiler's own accounting, one technique per rung,
  re-measure. On ch. 11's GPT (width 512, ~19M params), jit is worth
  around twenty-fold over the un-jitted strawman; bf16 at batch 64 then
  measures almost *nothing* (the dispatch wall), the fp32 control shows
  batch-up alone is *negative* (the memory wall) — and together the two
  rungs pay most of a factor of two. Rungs interact; ladders are
  measured cumulatively. The waterfall prints the figure from the
  jitted baseline.
* Precision in JAX is explicit, and *layered*: casting the arrays is not
  enough — flax modules remember a compute dtype from construction, and
  the model's own fp32 arithmetic (rope's trigonometry) promotes values
  straight back, which the attention kernel refuses loudly. The naive
  cast measured exactly like fp32; one dtype print exposed it. The
  masters, the gradients, and the Adam state stay fp32 by construction.
* Measurement is a technique too, and it can fail: the stock `tf.data`
  loader silently taxed the fastest rungs (fixed by staging the 5 MB
  corpus on device), and re-timing one cached batch flatters throughput
  by double-digit percent. Both had to be fixed before the waterfall
  meant anything.
* Rungs that do not help teach the most: bf16 at batch 64 is a *null*
  rung here — the wall was per-step dispatch, which precision cannot
  touch — and checkpointing costs time to save memory the compiler's
  plan said we did not need. Knowing when a technique does not apply is
  the method too.
* Data parallelism is predicted before it is measured: with $\beta$ from
  13.5's measured `psum`, :eqref:`eq_dp_cost` prices the $k=2$ gain at
  roughly nothing — the measurement lands at or below that floor — and
  $k=4$ falls well short of the floor as the host-staged fabric sags.
  Declarative sharding then makes rerunning the *fast* configuration a
  loop rather than a launcher, and its recovered efficiency is the cost
  model's grown compute term, verified.
:end_tab:

## Exercises

1. Reproduce the negative bf16 rung: run the R0→R1→R2 sequence at
   `num_hiddens=256` and confirm bf16 is *slower* than compile-alone. Then
   profile both and explain, in terms of :numref:`fig_roofline`, why the
   width-256 matmuls do not benefit from the tensor cores.
1. Add a rung of your own — `channels_last` memory format, pinned-memory
   dataloading, or `sdpa_kernel` backend pinning — measure it, and slot it
   into the waterfall. Which regime does it attack, and does it clear the
   noise floor?
1. Apply the whole ladder to a *different* model: ch. 12's Mamba capstone
   (:numref:`sec_mamba`) or ch. 11's ViT. Which rungs change sign, and
   why? (Hint: a model's arithmetic intensity decides which fixes pay.)
1. Take a rung that helped here — say, batch-up — and construct a model
   (deeper, or with a much larger vocabulary) where it *hurts*. Explain
   the reversal with the memory anatomy of :numref:`sec_memory_precision`.
1. Rung 5 parallelized the *eager* model. Stack DDP on the full ladder
   instead — compiled, bf16, per-rank batch 512 — and measure tokens/s at
   $k = 2$ and $k = 4$. Efficiency comes out markedly higher than the
   eager rung's. Explain why with :numref:`sec_multi_gpu`'s cost model:
   which term grew, and which stayed fixed?
1. Our R1 measurement was once corrupted by a retrace inside the timing
   window. Remove the ragged-batch filter from `batches`, re-run R1, and
   find the retrace two ways: in the throughput number, and via
   `torch._dynamo` logging. How many steps per epoch does the offending
   batch appear in, and why does a longer timed window dilute but not
   remove the damage?
1. (JAX) The JAX tab never touched the attention kernel: `d2l.GPT`'s
   default lowering materializes the $T \times T$ score matrix, and the
   fused-kernel fix of :numref:`sec_attention-at-scale`
   (`implementation='cudnn'`) is one argument away — but it accepts only
   half-precision inputs, so it needs R2's threading. Subclass
   `CausalAttention` to name the kernel, verify parity against the
   default lowering, and re-measure the ladder at context 128 and again
   at context 1024, reading both tokens/s and the compiler's planned temp
   bytes. Where does the fused kernel start to pay, and why does context
   length matter more than batch size?
1. (JAX) Rung 5 sharded only the batch. Following 13.6's punchline,
   change the `PartitionSpec` so the *parameters* are sharded across the
   mesh as well (the FSDP pattern), leaving the step function untouched.
   Read the compiled HLO for the collectives GSPMD inserts in place of
   the allreduce, and measure tokens/s at $k = 2$ and $k = 4$. At 19M
   parameters this buys no speed — which term of
   :numref:`sec_memory_precision`'s memory anatomy would have to grow
   before it pays?

<!-- slides -->

::: {.slide title="The Method, on a Real Model" only="pytorch"}
Nothing new here — that is the point. Six sections built a
method and a toolbox; now run the whole loop on ch. 11's GPT
and take it down a **waterfall**, one rung per technique, each
measured and attributed.

Subject: `d2l.GPT`, width **512** (~19M params). Why not 256?
At 256, bf16 goes *backwards* — matmuls too small for the
tensor cores. The width choice is itself a measurement.
:::

::: {.slide title="The Method, on a Real Model" only="jax"}
Nothing new here — that is the point. Six sections built a
method and a toolbox; now run the whole loop on ch. 11's GPT
and take it down a **waterfall**, one rung per technique, each
measured and attributed.

Subject: `d2l.GPT`, width **512** (~19M params), char *Time
Machine*, context 128, end-to-end tokens/s. Same model and
metric as the PyTorch tab — the ladder will look nothing like
it. That contrast is the lesson.
:::

::: {.slide title="Profile Your Experiment First" only="pytorch"}
Two traps corrupted an early draft by tens of percent:

- a **ragged final batch** put a `torch.compile` retrace
  *inside* the timing window → keep shapes constant
- the profiler's instrumentation **outlives its cell** →
  `TEARDOWN_CUPTI=1`, or every later timing pays a launch tax

The metric is end-to-end tokens/s — `DataLoader` and H2D
included.
:::

::: {.slide title="Measure Honestly First" only="jax"}
Traps that corrupted early drafts — JAX edition:

- a changed shape is not a retrace, it is a **recompile** →
  whole batches only, one shape
- the stock `tf.data` loader taxed the fast rungs by ~¼ →
  stage the 5 MB corpus **on device**, shuffle there
- no `block_until_ready`, no measurement — you timed the
  *enqueue* (§13.1)
- never re-time one cached batch: the GPU serves it from
  cache. Honest data is fresh data.
:::

::: {.slide title="Rung 0: Baseline, Profiled" only="pytorch"}
Classify before you fix.

@fast-transformer-rung-0-baseline-profiled

Matmuls ≈ half the device time; the rest is a fusible
elementwise tail, with dispatch busy most of the step. That
predicts compile pays, then precision.
:::

::: {.slide title="Rungs 0 → 1: The Only Real Baseline" only="jax"}
Un-jitted JAX is a strawman — measure it once, briefly, to
price the mistake; nobody trains there.

@fast-transformer-rungs-each-one-measured-1

`nnx.jit` on the *whole* step (loss + grads + update) is one
transformation and roughly **20×**. Every honest ratio below
starts from this bar, not from R0.
:::

::: {.slide title="Classify the Compiled Step" only="jax"}
Ask the compiler what it plans to run:

@fast-transformer-rungs-each-one-measured-5

About a third of the tf32 roof (§13.2): not overhead-starved,
not saturated — a mixed workload. Cheapest next rung on §13.4's
ladder: **bf16** — double the roof, half the bytes.
:::

::: {.slide title="The Rungs" only="pytorch"}
- **R1 compile** — fuses the tail: **~1.3×** (and asserts
  compiled ≡ eager first)
- **R2 bf16** — tensor cores, matmuls wide enough: **~1.4×**
- **R3 batch-up** — climb the roofline: **~1.3×** (fp32-512
  control: bf16 bought headroom, not admission)
- **R4 checkpoint** — *negative* for speed (−~10%), but cuts
  peak memory ~3× (unneeded here)

@fast-transformer-rungs-each-one-measured-3
:::

::: {.slide title="Rung 2: bf16 Is a Discipline, Not a Cast" only="jax"}
Cast every array to bf16 → measures **exactly like fp32**.
Silent. The receipt: the embedding still emits fp32 — flax
modules *remember* a compute dtype; rope's fp32 trig promotes
q, k right back.

@fast-transformer-rungs-each-one-measured-9

Threaded for real (receipt: `bfloat16`; the planned
temporaries drop ~40%) — and **still flat**: at batch 64 the
wall is per-step *dispatch*, which precision cannot touch.
Masters and grads stay fp32. No `GradScaler`.
:::

::: {.slide title="Rungs 3–4: Rungs Interact" only="jax"}
@fast-transformer-rungs-each-one-measured-3

**Neither rung pays alone.** fp32 at 512 is *slower* than at
64 — the ~15 GiB plan is traffic, a memory wall. bf16 at 64
was flat — a dispatch wall. Together: most of 2×, ~43 TFLOP/s.
Measure ladders cumulatively. Checkpointing then costs ~a
tenth of the speed for memory the plan says we did not
need — the same negative rung as the PyTorch tab.
:::

::: {.slide title="The Waterfall" only="pytorch"}
@fast-transformer-the-waterfall

Cumulative — each bar inherits every choice to its left. No
dominant rung; three modest wins multiply to **~2.4×**.
Checkpointing is red: a technique that helped a different
model *hurts* this one. A 300-step run confirms the fast
configuration still learns.
:::

::: {.slide title="The Waterfall" only="jax"}
@fast-transformer-the-waterfall

jit is the cliff (~20×). bf16 stalls at batch 64 — the flat
bar that taught the most. Batch-up cashes both rungs at once;
checkpointing is red, as in the other tab. Same model, same
card — a different bottleneck sequence, found by the same
loop. A 300-step run confirms the fast configuration still
learns.
:::

::: {.slide title="Predict, Then Measure: Data Parallel" only="pytorch"}
Transformer params ∝ compute ⇒ ~76 MB of gradients per step
with little compute to hide them. The cost model (§13.5) gives
a **no-overlap floor**; DDP's bucketing buys back some overlap.

@fast-transformer-rungs-each-one-measured-8

Measured lands at the floor — just above it at $k=2$ (overlap),
just below at $k=4$ (the staged fabric sags) — priced *before*
the launch. NVLink changes the constant, not the method.
:::

::: {.slide title="Predict, Then Measure: Data Parallel" only="jax"}
~76 MB of gradients vs §13.5's measured `psum` β ⇒ the
two-term cost model prices $k=2$ at **no gain at all**. Then
measure: mesh + shard the batch + the *unchanged* jitted step —
no launcher, no sidecar.

@fast-transformer-rungs-each-one-measured-8

On the floor at $k=2$, below it at $k=4$ (the staged fabric
sags). And because DP is three annotations, stacking it on the
fast config is a loop — efficiency recovers exactly as the
grown compute term predicts.
:::

::: {.slide title="The Lore, and the Ladder Beyond"}
modded-nanoGPT's speedrun = this chapter's contents stacked to
the ceiling: compiled kernels, FlashAttention, a better
optimizer (Muon), fp8. Each record a new rung.

**measure → classify → fix → re-measure.** Two budgets, three
regimes, one loop — now shown on a real model, seven sections
deep.
:::
