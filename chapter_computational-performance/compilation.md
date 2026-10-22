# Compute Graphs and Compilation
:label:`sec_compilation`

:numref:`sec_perf_model` measured an unfused chain of
elementwise operations — the kind every activation function and
normalization layer contains — ran each operation as its own GPU kernel,
and each kernel made a full round trip to memory: read the input, compute
one cheap thing, write the output, only for the next kernel to read it
again. The program was bandwidth-bound because it transferred every
intermediate tensor. This section shows how compilation fuses the chain and
then examines the resulting computation.

Compilation captures the computation as a graph instead of executing each
operation when Python reaches it (*eager* execution). A compiler can then
rewrite operations across the graph, for example by fusing the elementwise
chain into one kernel that reads once, performs all operations, and writes
once. Both of our
frameworks do this with different capture mechanisms and different failure
modes.

*Prerequisites: the three regimes and the* `d2l.Benchmark` *timer of*
:numref:`sec_perf_model`*; the kernel-launch and memory-round-trip costs
of* :numref:`sec_hardware`*. This section retires the old imperative-
versus-symbolic "hybridize" framing in favor of the modern one: eager by
default, with a tracing compiler you switch on.*

```{.python .input #compilation-compute-graphs-and-compilation}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import time
import torch
from torch import nn

torch.set_float32_matmul_precision('high')
```

```{.python .input #compilation-compute-graphs-and-compilation}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
```

## Computation Graphs
:label:`subsec_comp-graph`

A neural network *is* a computation graph — you have been building them
since :numref:`sec_autograd`. When you write `y = relu(x @ W + b)`,
autograd records a directed graph of operations precisely so it can walk
it backward for gradients (:numref:`fig_compute_graph`). Autograd exposes
this graph to the backward pass, whereas compilation
constructs an intermediate representation for program transformation.
Autograd's graph is a *tape*, not a compiler's whole-program IR. Moreover,
not every node launches a kernel: a view launches nothing, and a library
matmul already combines many operations behind one launch.

![The compute graph of a two-layer network. Autograd already builds this
to run the backward pass; a compiler that sees the whole graph can rewrite
it before running the forward pass.](../img/mdl-perf-compute-graph.svg)
:label:`fig_compute_graph`

Eager execution walks this graph one node at a time, as Python reaches
each line. Every node becomes a kernel launch (5–15 µs of CPU-to-GPU
latency, :numref:`sec_hardware`) and, for a memory-bound op, a round trip
to HBM. Python sees one operation, dispatches it, and moves on; it never
learns that the `add` feeding the `relu` could have been done *inside*
the same kernel. A compiler that captures the whole graph can fuse these operations before
execution.

Earlier frameworks required a choice between *imperative* execution and
*symbolic* graph construction. Current PyTorch and JAX workflows instead use
eager execution for development and enable tracing compilation for
performance. The two frameworks implement this design differently.

## Graph Capture in PyTorch and JAX
:label:`subsec_comp-capture`

The frameworks capture the graph in genuinely different ways, and the
difference determines how each one breaks.

**PyTorch — `torch.compile`, capture from Python bytecode.** `torch.compile`
wraps a module or function and, on first call, its front-end (TorchDynamo)
*inspects the Python bytecode* as it runs, extracting the tensor
operations into a graph while leaving the surrounding Python alone
:cite:`Ansel.Yang.He.ea.2024`. When it meets Python it cannot trace into
— a data-dependent branch, a `print`, an unsupported library call — it
does not fail; it inserts a **graph break**, runs that piece in ordinary
Python, and resumes capturing after it (:numref:`fig_compile_pipelines`,
top). The captured pieces are compiled; the breaks are the price. It also
plants *guards*: cheap runtime checks that the assumptions the graph was
compiled under (tensor shapes, dtypes) still hold, re-compiling if they do
not.

![Two capture pipelines. `torch.compile` extracts a graph from Python
bytecode and falls back to Python on a graph break; `jax.jit` traces the
function into a jaxpr and recompiles when input shapes or dtypes
change.](../img/mdl-perf-compile-pipelines.svg)
:label:`fig_compile_pipelines`

A branch on a tensor's *value* forces a graph break because the trace cannot
know which path it takes until runtime:

```{.python .input #compilation-capture-two-philosophies-1}
%%tab pytorch
def f(x):
    x = x * 2
    if x.sum() > 0:        # Data-dependent: forces a graph break
        return x + 1
    return x - 1

explanation = torch._dynamo.explain(f)(torch.randn(8, device=d2l.try_gpu()))
print(f'graph breaks: {explanation.graph_break_count}')
```

**JAX — `jax.jit`, capture by tracing.** `jax.jit` takes the opposite
stance: it *traces* the function by calling it once with abstract
placeholder values that record every operation they touch, producing a
typed intermediate representation called a **jaxpr**, which XLA lowers to
optimized device code (:numref:`fig_compile_pipelines`, bottom). Tracing
sees tensor operations only — anything that is not a traced array
operation, like a Python `print`, runs *once at trace time* and then
vanishes from the compiled function:

```{.python .input #compilation-capture-two-philosophies-2}
%%tab jax
@jax.jit
def f(x):
    print('this prints once, at trace time')  # Not in the compiled graph
    return jnp.sin(x) + 1

x = jnp.arange(4.0)
print(jax.make_jaxpr(lambda x: jnp.sin(x) + 1)(x))  # The captured graph
_ = f(x); _ = f(x)   # Second call: no print — the trace is cached
```

JAX avoids graph breaks by requiring the entire function to be traceable.
It cannot fall back to Python
mid-graph — but in exchange it never silently splits your function into
compiled fragments. Its main failure mode is recompilation: the trace is
specialized to the *shapes and dtypes* it saw, so calling the compiled
function with a new shape triggers a full **retrace and recompile**. In a
training loop with variable-length batches this can mean recompiling every
step. The escape hatches are to keep shapes static (the
`drop_remainder=True` data-loading discipline of :numref:`sec_fashion_mnist`
was designed for this case), to mark genuinely constant arguments
with `static_argnums`, and to express data-dependent control flow with
`lax.cond`/`lax.scan` so it lives *inside* the graph rather than breaking
it. The frameworks therefore require different checks: `torch.compile`
allows more Python but must be inspected for graph breaks, whereas
`jax.jit` requires pure traced computation and must be inspected for
shape- or dtype-triggered recompilation.

## Operation Fusion
:label:`subsec_comp-fusion`

Recall the unfused elementwise chain from
:numref:`subsec_perf-regimes` — a handful of cheap operations, each its
own kernel, each a full memory round trip. Compilation fuses them into
one kernel that reads the input once, does all the arithmetic in
registers, and writes once:

```{.python .input #compilation-what-the-compiler-does-fusion}
%%tab pytorch
x = torch.randn(4000, 4000, device=d2l.try_gpu())

def gelu_ish(x):
    return 0.5 * x * (1 + torch.tanh(0.8 * (x + 0.04 * x**3)))

compiled = torch.compile(gelu_ish)
# First call compiles; then verify the rewrite: same answer, then faster
assert torch.allclose(gelu_ish(x), compiled(x), atol=1e-6)

print(d2l.Benchmark(lambda: gelu_ish(x), desc='eager'))
print(d2l.Benchmark(lambda: compiled(x), desc='compiled'))
```

```{.python .input #compilation-what-the-compiler-does-fusion}
%%tab jax
x = jax.random.normal(jax.random.PRNGKey(0), (4000, 4000))

def gelu_ish(x):
    return 0.5 * x * (1 + jnp.tanh(0.8 * (x + 0.04 * x**3)))

compiled = jax.jit(gelu_ish)
# First call compiles; then verify the rewrite: same answer, then faster
assert jnp.allclose(gelu_ish(x), compiled(x), atol=1e-6)

print(d2l.Benchmark(lambda: gelu_ish(x), desc='eager'))
print(d2l.Benchmark(lambda: compiled(x), desc='compiled'))
```

The `assert` verifies that compilation preserves the result within
$10^{-6}$, allowing for floating-point reassociation. Performance is compared
only after this correctness check. The fused version is close to an order
of magnitude faster
on this chain — and the reason is entirely on the bytes side of the
roofline: the chain performs the same arithmetic either way, but eager
execution makes one memory round trip *per operation* while the fused
kernel makes a single round trip for the whole chain. Fuse eight
elementwise ops and you cut roughly eight memory traversals to one. More
generally, **fusion exchanges kernel launches and memory traffic for
arithmetic in registers**. Compilation therefore helps most when
:numref:`sec_perf_model` classifies the computation as bandwidth- or
overhead-bound, and barely at all where you are already compute-bound (a
single large matmul is already one well-tuned kernel; there is nothing to
fuse).

When the compiler cannot fuse enough — when a memory access pattern needs
restructuring, not just merging — people write the kernel by hand. The
hand-written FlashAttention kernel of :numref:`sec_attention-at-scale` is
exactly this: the same "keep intermediates on-chip, never round-trip the
big matrix" idea, executed by an expert for a pattern the general
compiler cannot discover. Writing such kernels is its own craft, and it
is deliberately out of scope for this book (:numref:`sec_custom_layer`
drew that fence); Triton :cite:`Tillet.Kung.Cox.2019` (an important
backend of `torch.compile`'s Inductor, which also draws on template and
library kernels) and Pallas let you author them in Python-like
syntax. A compiler applies many common fusion transformations automatically to the
existing program.

## Measured Training-Step Compilation
:label:`subsec_comp-wholestep`

The practical application is compiling a complete training step. The two
frameworks differ in which parts of that step they capture. In PyTorch,
`torch.compile(net)` captures the model's forward *and* the backward that
flows through it, but the optimizer's `opt.step()` below stays eager —
compiling the optimizer too is possible, just not what the one-liner
gives you. In JAX, nothing stops the jitted function from *being* the
whole step: loss, gradients, and the parameter update, one compiled
program. In both cases, the first call pays a fixed compilation cost, and
subsequent calls have lower steady-state execution time. Whether compilation reduces total runtime
depends on the number of subsequent calls:

```{.python .input #compilation-whole-step-compilation-measured}
%%tab pytorch
class GeluIsh(nn.Module):  # Wraps the elementwise chain above as a layer
    def forward(self, x):
        return gelu_ish(x)

net = nn.Sequential(nn.Linear(1024, 1024), GeluIsh(),
                    nn.Linear(1024, 1024), GeluIsh(),
                    nn.Linear(1024, 1024), GeluIsh(),
                    nn.Linear(1024, 1024)).to(d2l.try_gpu())
X = torch.randn(512, 1024, device=d2l.try_gpu())
opt = torch.optim.SGD(net.parameters(), lr=0.01)

def train_step(model):
    opt.zero_grad(set_to_none=True)
    model(X).sum().backward()
    opt.step()

cnet = torch.compile(net)
t0 = time.perf_counter()
train_step(cnet); torch.cuda.synchronize()  # First call: compiles
print(f'first compiled step: {time.perf_counter() - t0:.1f} s')

print(d2l.Benchmark(lambda: train_step(net), desc='eager'))
print(d2l.Benchmark(lambda: train_step(cnet), desc='compiled'))
```

```{.python .input #compilation-whole-step-compilation-measured}
%%tab jax
def loss_fn(params, X):
    h = X
    for W, b in params[:-1]:
        h = jax.nn.gelu(h @ W + b)
    W, b = params[-1]
    return (h @ W + b).sum()

def train_step(params, X, lr=0.01):  # Loss, gradients, AND the update
    loss, grads = jax.value_and_grad(loss_fn)(params, X)
    return loss, jax.tree.map(lambda p, g: p - lr * g, params, grads)

key = jax.random.PRNGKey(0)
shapes = [(1024, 1024), (1024, 1024), (1024, 1024)]
params = [(jax.random.normal(k, s) * 0.03, jnp.zeros(s[1]))
          for k, s in zip(jax.random.split(key, 3), shapes)]
X = jax.random.normal(key, (512, 1024))

t0 = time.perf_counter()
compiled = jax.jit(train_step).lower(params, X).compile()  # AOT: compile now
print(f'ahead-of-time compile: {time.perf_counter() - t0:.1f} s')

print(d2l.Benchmark(lambda: train_step(params, X), desc='eager'))
print(d2l.Benchmark(lambda: compiled(params, X), desc='compiled'))
```

The compiled step is faster in steady state — a fused training step makes
far fewer trips to memory and issues far fewer launches — and the first
call carries a visible one-time cost while the compiler works: a fraction
of a second to a couple of seconds on this toy step, depending on the
framework and the state of its compile cache, and about two seconds for
the real Transformer of :numref:`sec_fast_transformer`. On a short
experiment that price may never be repaid; on a training run of thousands
of steps, the compilation cost becomes negligible per step. The JAX tab also demonstrates
ahead-of-time compilation: `lower(...).compile()` runs the compiler as
an explicit step rather than lazily on first call, returning a compiled
object you can then introspect — `compiled.memory_analysis()` reports the
memory the compiler *planned* before a single byte is allocated, a theme
we develop in :numref:`sec_memory_precision`.

:begin_tab:`jax`
Look at the JAX ratio once more: the un-jitted step is about two orders
of magnitude slower than the compiled one, because every operation in the
loss, the gradient, *and* the update dispatches separately. JAX training
steps should normally use `jax.jit`; un-jitted execution is useful
primarily for debugging because each operation dispatches separately.
:end_tab:

## The Overhead Regime: Capture and Replay
:label:`subsec_comp-overhead`

Fusion reduces bandwidth costs. An *overhead-bound* program instead spends
most of its time waiting for Python to launch small kernels and requires
capture and replay. A model that is a deep stack
of thin layers issues a great many small kernels, and if each takes
longer to *launch* than to *run*, the device remains idle even though
each kernel requires little computation (:numref:`fig_async_timeline`). Compiling for fusion
helps some, but the decisive fix is to stop involving Python at all:
record the entire sequence of launches once, then *replay* it with a
single command. This is what CUDA graphs do, and `torch.compile` exposes
them through one mode flag:

```{.python .input #compilation-the-overhead-regime-capture-and-replay}
%%tab pytorch
# Many small layers: launch overhead dominates the actual arithmetic
deep = nn.Sequential(*[m for _ in range(60)
                       for m in (nn.Linear(256, 256), nn.Tanh())]).to(
    d2l.try_gpu())
x = torch.randn(64, 256, device=d2l.try_gpu())

reduced = torch.compile(deep, mode='reduce-overhead')
with torch.no_grad():  # Forward only: replay wants fixed buffers
    reduced(x)  # Warmup: compiles and captures the CUDA graph
    print(d2l.Benchmark(lambda: deep(x), desc='eager'))
    print(d2l.Benchmark(lambda: reduced(x), desc='reduce-overhead'))
```

```{.python .input #compilation-the-overhead-regime-capture-and-replay}
%%tab jax
# XLA already amortizes launches: the whole jitted graph is one dispatch.
deep_params = [jax.random.normal(k, (256, 256)) * 0.06
               for k in jax.random.split(jax.random.PRNGKey(1), 60)]
x = jax.random.normal(jax.random.PRNGKey(2), (64, 256))

def deep_fwd(params, x):
    for W in params:
        x = jnp.tanh(x @ W)
    return x

compiled = jax.jit(deep_fwd)
compiled(deep_params, x).block_until_ready()

print(d2l.Benchmark(lambda: deep_fwd(deep_params, x), desc='eager'))
print(d2l.Benchmark(lambda: compiled(deep_params, x), desc='jit'))
```

The `reduce-overhead` mode captures the model's kernel launches into a
CUDA graph and replays the whole thing per call, collapsing a
hundred-odd launch latencies into one. Replay is
*rigid*: a replayed graph is a fixed sequence of kernels on fixed memory
addresses, so the input shape must not change between calls (change it
and PyTorch re-captures). That rigidity is also why we time the forward
pass under `torch.no_grad()`: autograd's saved-for-backward activations
are fresh allocations on every call, and their changing addresses would
force a re-capture each time. JAX needs no separate replay mechanism: a
jitted function is *already* a single dispatched
executable, so XLA amortizes launch overhead by construction — the
absence of a "reduce-overhead" knob in JAX is not a missing feature but a
consequence of compile-by-tracing.

## When Compilation Hurts
:label:`subsec_comp-hurts`

Compilation is not free and not always worth it. The checklist:

* **Short runs.** If training lasts only a few dozen steps, compilation
  time may exceed the accumulated steady-state reduction. Report both
  first-call and repeated-call times and compute the break-even call count.
* **Graph breaks in hot loops.** A `torch.compile`d function riddled with
  data-dependent branches compiles many small fragments and falls back to
  Python between them, keeping little of the benefit. `torch._dynamo.explain`
  (used above) counts the breaks; the fix is to remove value-dependent
  control flow from the hot path.
* **Shape churn (JAX).** A `jax.jit`ted function called with a new shape
  every step recompiles every step. Pad to fixed shapes, or accept that
  this particular function should not be jitted.
* **Already compute-bound.** If the profiler shows one big matmul
  dominating, there is nothing to fuse and no overhead to hide.
  Compiling is usually harmless, but measure rather than assume —
  compilation can also regress time or memory — and do not expect a win.

The measured performance regime determines whether compilation is
appropriate. Compilation is the fix for the bandwidth and
overhead regimes; reach for it when :numref:`sec_perf_model`'s method
points there, not reflexively. One last note on portability: the same
capture machinery that compiles a graph can also *export* it — PyTorch's
`torch.export` emits a portable captured graph, and JAX lowers jitted
functions to the StableHLO interchange format; either path lets a model
run outside Python entirely, in a serving runtime or on a phone.
These export paths replace earlier model-serialization mechanisms and
belong to deployment rather than training.

## Summary

* A neural network is already a compute graph (autograd builds it for the
  backward pass). Eager execution walks it one kernel at a time; a
  compiler that captures the whole graph can rewrite it — chiefly by
  *fusing* operations so intermediates stay on-chip.
* `torch.compile` captures from Python bytecode and inserts *graph breaks*
  where it meets untraceable Python; `jax.jit` traces to a shape-
  specialized *jaxpr* with no breaks but *recompiles on new shapes*. Watch
  breaks in one, recompiles in the other.
* Fusion is the fix for the bandwidth regime — close to an order of
  magnitude on our unfused elementwise chain — because it trades memory
  round trips for free register arithmetic. It barely helps
  already-compute-bound kernels.
* `torch.compile(mode="reduce-overhead")` uses CUDA graphs to collapse
  many small launches into one replay, curing the overhead regime; XLA
  amortizes launches by construction and needs no such knob. Both require
  static shapes.
* Compilation costs seconds on the first call and repays over thousands of
  steps. Reach for it when the method diagnoses bandwidth or overhead, not
  reflexively.

## Exercises

:begin_tab:`pytorch`
1. [code] **Value-dependent branches.** The function `f` of
   :numref:`subsec_comp-capture` breaks the graph at its `if`.
    1. Read the break reason in the output of `torch._dynamo.explain(f)`.
       Rewrite `f` with `torch.where` so that `explain` reports no break,
       and check that both versions return the same values.
    1. Time the compiled original and the compiled rewrite with
       `d2l.Benchmark`, once on the 8-element input of
       :numref:`subsec_comp-capture` and once on a $4000 \times 4000$
       input. Explain the time difference at each size.
    1. `torch.where` evaluates both branches. Describe a pair of branches
       for which this costs more than the graph break saves.

2. [code] **Shape changes.** `torch.compile` guards on the shapes of its
   inputs and recompiles when a guard fails
   (:numref:`subsec_comp-capture`).
    1. Compile a function that computes `torch.tanh(x) * 2 + 1`, and call
       it on vectors of lengths 8, 16, 32, and so on up to 8192. Predict
       the number of compilations from the guard rule. Then count them,
       with `torch._logging.set_logs(recompiles=True)` and with a `backend`
       function for `torch.compile` that increments a counter, prints
       `gm.code`, and returns `gm.forward`. Explain the count from the log
       and from the printed graphs.
    1. Repeat with `torch.compile(..., dynamic=False)`. How many
       compilations happen now, and what do the calls after the last
       compilation run?
:end_tab:

:begin_tab:`jax`
1. [code] **Value-dependent branches.** Tracing replaces the values of the
   arguments by placeholders (:numref:`subsec_comp-capture`).
    1. Write a function that doubles `x` and then returns `x + 1` when
       `x.sum() > 0` and `x - 1` otherwise, and apply `jax.jit` to it.
       Explain the error it raises.
    1. Rewrite the branch once with `jnp.where` and once with
       `jax.lax.cond`, and print both jaxprs with `jax.make_jaxpr`. Which
       operations does each version execute at run time?
    1. Apply `jax.vmap` to the `jax.lax.cond` version over a batch of
       vectors and print its jaxpr. Explain what became of the conditional,
       and why.

2. [code] **Shape changes.** A jitted function is specialized to the shapes
   of its arguments (:numref:`subsec_comp-capture`).
    1. Jit a function that returns the mean of `jnp.tanh(x)` over a vector
       `x`, and let its Python body increment a global counter. Call it in
       a loop on vectors of lengths 100, 200, and 300, three times each.
       Count the traces, and time the first and the later calls at each
       length.
    1. Pad every input to length 512 and pass the function what it needs
       to exclude the padding from the mean, so that one trace serves all
       three lengths. Check the results against the unpadded version.
    1. Instead, pass the padded vector together with its length, mark the
       length with `static_argnums`, and slice inside the function. Count
       the traces and explain the count. Name an argument of a training
       step for which `static_argnums` is the right tool.
:end_tab:

3. **Padding buckets.** A compiled step that is specialized to static
   shapes (:numref:`subsec_comp-capture`) compiles once for every distinct
   input shape it sees. Consider a training run of $10^5$ steps in which
   every batch holds sequences of one length $\ell$, drawn uniformly from
   $\{1, \ldots, 512\}$, and is padded to a length $\ell' \geq \ell$. One
   compilation costs 20 s, and a step at padded length $\ell'$ costs
   $50\,\textrm{ms} \cdot \ell'/512$.
    1. Pad every batch to the next multiple of $w$, for $w$ dividing 512.
       Find the number of compilations and the expected padded length as
       functions of $w$, and write the total run time $T(w)$.
    1. Minimize $T(w)$ over $w$ treated as a continuous variable, and state
       how the optimal $w$ scales with the compilation cost and with the
       number of steps. Compare $T$ at $w = 1$, at the two powers of two
       nearest the optimum, and at $w = 512$.
    1. Padding to the next power of two compiles ten shapes. Compute its
       expected padded length and total run time, and compare them with
       those of $w = 64$. For which distribution of lengths would powers
       of two be the better choice?

1. **Counting round trips.** In eager execution, every operation of
   `gelu_ish` in :numref:`subsec_comp-fusion` runs as its own kernel, which
   reads its input arrays from memory and writes its output array back.
    1. List the kernels that eager execution launches for `gelu_ish`, in
       the order in which Python evaluates the expression, and count the
       arrays of the size of `x` that each kernel reads and writes. How
       many array-sized transfers does the chain make in total, and how
       many does a single fused kernel make?
    1. Predict the eager and the fused time for the $4000 \times 4000$
       fp32 input from these transfer counts and the memory bandwidth of
       your GPU, which is about 1.0 TB/s for the build GPU of
       :numref:`subsec_perf-counting`. Compare both predictions and their
       ratio with the times that the code of :numref:`subsec_comp-fusion`
       measures, and account for the differences.

1. [code] **Break-even step count.** The compiled training step of
   :numref:`subsec_comp-wholestep` costs a one-time compilation time $t_0$
   and lowers the steady-state time per step from $t_e$ to $t_c$.
    1. Derive the number of steps $N^*$ after which compilation has saved
       more time than it cost. Measure $t_0$, $t_e$, and $t_c$ with the
       section's code, making sure that $t_0$ excludes the execution of the
       step itself, and evaluate $N^*$.
    1. Repeat the measurement of $t_0$ in a fresh Python process without
       changing the code, and explain any change.

1. [code] **Fusion behind a matmul.** ● Compilation leaves a large matrix
   product unchanged but fuses the elementwise chain that follows it.
   Consider `gelu_ish(a @ a)` for an $n \times n$ matrix `a` in bf16 on a
   GPU with peak compute $P$, bandwidth $\beta$, and ridge point
   $I^* = P/\beta$ (:numref:`subsec_perf-counting`). Model the product as
   compute-bound, taking $2n^3/P$ seconds. Model the chain as
   bandwidth-bound: eager execution moves $m$ arrays of $n^2$ elements of
   $b$ bytes each through memory, and the fused kernel moves two such
   arrays.
    1. Derive the speedup bound $S(n)$ of the compiled over the eager
       computation in terms of $n$, $m$, $b$, and $I^*$, and show that it
       tends to 1 as $n$ grows. With $m$ counted for `gelu_ish`, evaluate
       $S(n)$ for bf16 on the build GPU at $n = 1024$, $4096$, and $16384$,
       and find the $n$ at which $S(n)$ falls to $1.1$.
    1. Measure the eager and the compiled time of `gelu_ish(a @ a)` with
       `d2l.Benchmark` for the sizes of :numref:`subsec_perf-sweep`. Plot
       the measured speedup together with $S(n)$, and explain where and
       why the two differ.

1. **Fusing a reduction.** ● A numerically safe softmax over each row of an
   $R \times d$ fp32 matrix can be written as five eager operations: the
   row maximum, the subtraction of that maximum, the exponential, the row
   sum, and the division by that sum.
    1. Count the elements that the five eager kernels read from and write
       to memory, keeping only terms proportional to $Rd$. Count the same
       for a single fused kernel that holds each row in on-chip memory, and
       state the speedup bound for a bandwidth-bound softmax.
    1. A row longer than the on-chip memory must be read from memory more
       than once. Find the smallest number of passes over a row, and the
       resulting traffic, for a fused kernel that computes the maximum and
       the sum in separate passes, and for one that maintains both online
       as in :eqref:`eq_online-softmax`.

:begin_tab:`pytorch`
8. [code] **Compiling the optimizer.** ● `torch.compile(net)` in
    :numref:`subsec_comp-wholestep` compiles the forward and the backward
    pass but leaves `opt.step()` eager.
    1. Apply `torch._dynamo.explain` to `train_step` itself, called with
       the eager `net`. How many graphs does it capture, where does it
       break, and which parts of the step fall into which graph?
    1. Count the parameter-sized arrays that one update must read and
       write at minimum, for plain SGD and for Adam (:numref:`sec_adam`).
       Then time `train_step(cnet)` against
       `torch.compile(train_step)(net)`, once with `torch.optim.SGD` and
       once with `torch.optim.Adam`. For which optimizer does compiling the
       update pay, and why?

    *Adapted from Michael Lazos,
    [Compiling the optimizer with torch.compile](https://docs.pytorch.org/tutorials/recipes/compiling_optimizer.html),
    PyTorch recipe.*
:end_tab:

:begin_tab:`jax`
8. [code] **Trace-time constants.** The Python body of a jitted function
    runs only while it is traced (:numref:`subsec_comp-capture`).
    1. Jit a function that returns `x + np.random.randn()`, call it twice,
       and print its jaxpr. Explain the two outputs.
    1. A dropout function draws its mask with `jax.random.bernoulli` from
       a key held in a global variable. Jit it, call it, assign a new key
       to the global variable, and call it again. Explain the result, and
       rewrite the function so that every step draws a fresh mask.
:end_tab:

9. [code] **Overhead crossover.** ● Use the stack of 60 layers of width
   256 at batch size 64 from :numref:`subsec_comp-overhead`, with its eager
   and its compiled version. Compilation removes the per-kernel launch
   latency of :numref:`fig_latency_ladder` from the critical path, but not
   the time each kernel takes to execute.
    1. Time both versions for depths $L$ from 10 to 200 at width 256, and
       fit each time as $t(L) = c + sL$. Compare the eager slope per kernel
       with the launch latency of :numref:`fig_latency_ladder`, and explain
       the compiled slope.
    1. Predict, from :eqref:`eq_roofline` and that launch latency, the
       width at which the matrix product of one layer takes as long to
       execute as to launch. Then time both versions at depth 60 for widths
       from 64 to 4096, plot the ratio of eager to compiled time against
       the width, and compare the width at which the ratio approaches 1
       with your prediction.

1. [extended] **A tracer of your own.** ●● Build a miniature tracing
   compiler in plain Python, with NumPy arrays as the values on which the
   captured program finally runs.
    1. Write a class `Tracer` whose instances carry a `shape` but no
       values, and functions `tanh`, `exp`, and `relu` that take a
       `Tracer`. Each operation on a `Tracer` appends a node to a shared
       graph and returns a new `Tracer`. A node holds the operation, its
       inputs, and the shape of its result. Support the arithmetic
       operators, the matrix product `@`, the comparison `>`, the methods
       `sum` and `max` with `axis` and `keepdims` arguments, and the three
       functions. Trace a copy of `gelu_ish` that calls your `tanh` on a
       placeholder of a given shape, and print the graph one node per line.
    1. Make `Tracer.__bool__` raise an error. Trace a function that
       returns `x + 1` when `x.shape[0] > 4` and `x - 1` otherwise, and a
       second function that branches the same way on `x.sum() > 0`.
       Explain which of the two your tracer can record, and what the
       recorded graph keeps of the branch.
    1. Write an interpreter that executes a graph node by node on NumPy
       arrays, and check its result against a direct NumPy evaluation of
       `gelu_ish`.
    1. Write a pass that partitions the graph into fused kernels under the
       rule that an elementwise node may join the kernel of its inputs, a
       matrix product forms a kernel of its own, and a reduction ends the
       kernel it belongs to. State how your pass treats a node whose inputs
       come from different kernels. For `gelu_ish`, for the five-operation
       softmax, and for the two-layer network
       `relu(x @ W1 + b1) @ W2 + b2`, count the kernel launches and the
       array elements moved through memory under one kernel per node and
       under your partition.

:begin_tab:`jax`
11. [code] **Unrolled loops.** ● `deep_fwd` of
    :numref:`subsec_comp-overhead` loops over a Python list of weight
    matrices, so tracing unrolls the loop into one long sequence of
    operations.
    1. For depths from 10 to 400, count the equations of the jaxpr of
       `deep_fwd` and time its ahead-of-time compilation with
       `jax.jit(deep_fwd).lower(deep_params, x).compile()`. Repeat for the
       gradient of `deep_fwd(deep_params, x).sum()` with respect to the
       weights. How do the count and the compile time grow with depth?
    1. Stack the weights into one array of shape $(L, 256, 256)$ for depth
       $L$, and rewrite the loop with `jax.lax.scan`. Repeat the
       measurements, check that both versions agree, and compare their
       steady-state times at depths 60 and 400. What does each version
       give up?
:end_tab:

<!-- slides -->

::: {.slide title="Computation Graphs"}
Autograd already builds a compute graph — to run *backward*.

![](../img/mdl-perf-compute-graph.svg){width=88%}

Eager: walk it one kernel at a time (a launch + a memory round
trip each). Capture it, and a compiler can rewrite across
nodes. Capturing the graph allows transformations across nodes.
:::

::: {.slide title="Graph Capture in PyTorch and JAX"}
![](../img/mdl-perf-compile-pipelines.svg){width=95%}

`torch.compile`: capture Python bytecode, **graph-break** to
Python when stuck. `jax.jit`: **trace** to a jaxpr — no breaks,
but **recompile on new shapes**.
:::

::: {.slide title="Breaks vs. Retraces"}
@compilation-capture-two-philosophies-1@pytorch

. . .

@compilation-capture-two-philosophies-2@jax

The print vanishes: tracing sees tensor ops only. Purity is the
price of having no graph breaks.
:::

::: {.slide title="Fusion Reduces Memory Traffic"}
The unfused chain from §13.1, compiled for fusion:

@compilation-what-the-compiler-does-fusion

Same answer first (the `allclose`), then faster: close to an
order of magnitude, entirely on the bytes side — one memory
round trip instead of one per op. FlashAttention (§10.5) is this
idea by hand.
:::

::: {.slide title="Compile the Training Step"}
@compilation-whole-step-compilation-measured

The first call pays a fixed compilation cost; later calls amortize it.
`torch.compile(net)` captures forward+backward (the optimizer
stays eager); `jax.jit` captures the whole step, including the update.
JAX's AOT `lower().compile()` also lets you *inspect* the plan.
:::

::: {.slide title="The Overhead Regime: Capture & Replay"}
Deep stack of thin layers — launches dominate arithmetic:

@compilation-the-overhead-regime-capture-and-replay

`reduce-overhead` records the launches into a CUDA graph and
replays them as one. XLA amortizes launches by construction —
no such knob needed.
:::

::: {.slide title="When Not to Compile"}
- short runs — compile cost never repaid
- graph breaks in the hot loop — little captured
- shape churn (JAX) — recompiles every step
- already compute-bound — nothing to fuse

Diagnose first (§13.1). Compile for bandwidth and overhead, not
reflexively.
:::
