# The Performance Model
:label:`sec_perf_model`

Every network we have trained so far spends most of its arithmetic in
matrix products, and we have implicitly assumed that a GPU computes them near
its peak rate. That assumption is often wrong. On the RTX 4090 that runs this
book's notebooks, a bf16 product of two $8192 \times 8192$ matrices runs
within a few percent of the 165 TFLOP/s hardware peak, whereas a
$256 \times 256$ product reaches less than two percent of it—same operation,
same chip, timed properly. A training step is a sequence of many operations,
some of them small. When most of them run at a fraction of peak, a faster GPU
or a second card changes little because the time goes to waiting rather than
computing.

Every operation on a GPU is limited by one of three resources: the arithmetic
units, the memory bandwidth that feeds them, or the fixed host cost of issuing
the operation from Python. Two paper counts—the FLOPs an operation performs
and the bytes it moves—decide between the first two. Adding one measured
constant, the host time $h$ to issue a call, lets us predict the third as
well. Our method throughout the chapter is to count FLOPs and bytes, predict
the time, measure it, and explain the difference.

```{.python .input #performance-model-the-performance-model}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #performance-model-the-performance-model}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #performance-model-the-performance-model-1}
%%tab pytorch
import os
import re
import time
# Detach PyTorch's profiler from the GPU when a profile ends; without this,
# every kernel launch after the first profile stays about 30% slower
os.environ['TEARDOWN_CUPTI'] = '1'
```

```{.python .input #performance-model-the-performance-model-1}
%%tab jax
import collections
import re
import time
```

```{.python .input #performance-model-the-performance-model-2}
%%tab pytorch
# Run fp32 matrix products on the tf32 tensor cores (PyTorch's default is
# full fp32). The bf16 products below are unaffected.
torch.set_float32_matmul_precision('high')
```

## Arithmetic Intensity and the Roofline
:label:`subsec_perf-counting`

Before timing anything, we can predict from an operation's shapes which
resource limits it. We used this idea briefly in
:numref:`subsec_kv-cache-prefill-decode` to explain why prefill processes
tokens much faster than autoregressive decoding; here we develop it in full.

### Counting FLOPs and Bytes

Every tensor operation makes two demands on a GPU: it performs arithmetic,
counted in *floating-point operations* (FLOPs, one addition or multiplication
each), and it moves bytes between the GPU's main memory and its on-chip
processors.

The GPU has a separate budget for each. The main memory of the RTX 4090
delivers at most $\beta = 1.008 \times 10^{12}$ bytes/s (~1 TB/s). Its peak
arithmetic rate $P$ depends on the number format and unit: bf16 matrix
products on the tensor cores reach $P = 165 \times 10^{12}$ FLOP/s, while tf32
matrix products and standard fp32 arithmetic run at half that rate (82.6
TFLOP/s) at the rated 2.52 GHz boost clock.

Which budget runs out first depends on their ratio, the *arithmetic intensity*

$$
I \;=\; \frac{\textrm{FLOPs performed}}{\textrm{bytes moved}}.
$$

Consider the matrix product $\mathbf{X}\mathbf{W}$ with
$\mathbf{X} \in \mathbb{R}^{B \times D}$ and
$\mathbf{W} \in \mathbb{R}^{D \times F}$, a linear layer applied to a batch
of $B$ rows. It performs $2BDF$ FLOPs (one multiply and one add per inner-product
term) and at best reads $\mathbf{X}$ and $\mathbf{W}$ once and writes the
output once ($BD + DF + BF$ numbers). With $b$ bytes per number, its
intensity is

$$
\frac{2BDF}{b\,(BD + DF + BF)}
\;=\; \frac{2}{b}\left(\frac{1}{F} + \frac{1}{B} + \frac{1}{D}\right)^{-1}
\;\approx\; \frac{2B}{b} \quad \textrm{for } B \ll D, F.
$$
:eqlabel:`eq_matmul_intensity`

The middle expression shows that the smallest of $(B, D, F)$ governs the
intensity. When the batch size $B$ is much smaller than the layer dimensions,
intensity grows linearly with $B$ because every weight fetched from memory is
reused across all $B$ rows.

For a 1B-parameter GPT with width $D = F = 2048$ in bf16 ($b = 2$), a training
minibatch of 8 sequences of 1,024 tokens has $B = 8192$ rows, giving
$I \approx 910$ FLOP/byte by :eqref:`eq_matmul_intensity` (limited by the
width $D$). During single-sequence autoregressive generation, $B = 1$ and
$I \approx 1$ FLOP/byte. The same layer spans nearly three orders of
magnitude in intensity between training and decoding. At the bottom end lies
an elementwise operation such as $y_i = x_i + 1$, which performs 1 FLOP per
$2b$ bytes moved ($I = 0.25$ FLOP/byte in bf16), regardless of tensor size.

### The Roofline Model

The *roofline model* :cite:`Williams.Waterman.Patterson.2009` turns intensity
into an upper bound on throughput: an operation runs no faster than the
compute roof $P$ and no faster than memory supplies data at rate $I\beta$:

$$
\textrm{performance} \;\leq\; \min(P,\; I \beta).
$$
:eqlabel:`eq_roofline`

On log–log axes, the bound is a sloped bandwidth line meeting a flat compute
roof at the *ridge point* $I^* = P/\beta$ (:numref:`fig_roofline`). Below
$I^*$ an operation is *bandwidth-bound*: memory cannot keep the arithmetic
units fed, so extra arithmetic on data already on chip is free. Above $I^*$ it
is *compute-bound*: the arithmetic units are saturated, and moving fewer bytes
does not speed it up. Throughout this chapter, $\beta$ denotes main-memory
(DRAM) bandwidth; operands small enough to reside in the 72 MB on-chip L2
cache can exceed this roof.

![The roofline of the RTX 4090. Attainable throughput is the smaller of the compute roof $P$ and the arithmetic intensity times the memory bandwidth $\beta$. The points mark the examples of the text: $x + 1$ in bf16, a linear layer of width 2048 applied to one token, and the same layer applied to a minibatch of 8,192 tokens. The ridge point lies near 164 FLOP/byte.](../img/mdl-perf-roofline.svg)
:label:`fig_roofline`

### The Ridge Point

For bf16 tensor-core products on the RTX 4090, the ridge point is
$I^* = 165.2 / 1.008 \approx 164$ FLOP/byte; for tf32 products ($P = 82.6$
TFLOP/s), it is $I^* \approx 82$ FLOP/byte. Modern data-center GPUs have even
higher ridge points (around 280–300 FLOP/byte) because tensor-core FLOP/s has
grown faster than HBM bandwidth. Only large matrix products clear these
thresholds; elementwise ops, normalizations, and single-token decoding all sit
far below the ridge.

For a square bf16 matrix product ($B = D = F = n$), $I = n / 3$, so the
roofline predicts that every product with $n \gtrsim 500$ is compute-bound.
Testing that prediction requires accurate GPU timing.

## Timing GPU Code
:label:`subsec_perf-measuring`

Because GPU dispatch is asynchronous (:numref:`subsec_async_dispatch`), Python
queues kernels and moves on immediately. Reading a wall clock around a kernel
call without synchronization measures how long Python took to enqueue the
work, not how long the GPU took to execute it. Accurate timing warms up the
device, synchronizes before starting and stopping the clock, and averages over
many repeats, as `d2l.Benchmark` does.

### Synchronization Points

Python blocks on the GPU only when it requests a value residing on the device:
converting a tensor to a Python scalar or NumPy array, printing it, branching
on it (`if loss < 0.1:`), or running an operation whose output shape depends
on tensor values (such as boolean masking `x[x > 0]`).

If the host takes time $h$ to issue a step and the GPU takes $t_\textrm{dev}$
to execute it, an unsynchronized loop overlaps host dispatch with device
execution, whereas reading a scalar every step serializes them and adds a
device-to-host round-trip latency $\sigma$:

$$
t_\textrm{once} \;\approx\; \max(h,\, t_\textrm{dev}),
\qquad
t_\textrm{every} \;\lesssim\; h + t_\textrm{dev} + \sigma.
$$

We compare two 1,000-step loops that multiply a $1024 \times 1024$ matrix by
itself and accumulate the sum on the GPU: one converts the running total to a
Python `float` every step, and the other reads it once at the end.

```{.python .input #performance-model-measuring-without-lying-2}
%%tab pytorch
x = torch.ones(1024, 1024, device=d2l.try_gpu())
s = torch.zeros((), device=x.device)
s += (x @ x).sum()
float(s)                          # Warm up: load the kernels

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000):
    s += (x @ x).sum()
    total = float(s)              # Reads the value: a barrier every step
read_every = time.perf_counter() - t0

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000):
    s += (x @ x).sum()            # Stays on the device: no barrier
total = float(s)                  # One read at the end
read_once = time.perf_counter() - t0
print(f'read every step: {read_every:.3f} s   read once: {read_once:.3f} s')
```

```{.python .input #performance-model-measuring-without-lying-2}
%%tab jax
x = jnp.ones((1024, 1024))
step = jax.jit(lambda s, x: s + (x @ x).sum())
s = step(jnp.zeros(()), x)
float(s)                          # Warm up: compile the step

t0 = time.perf_counter()
for _ in range(1000):
    s = step(s, x)
    total = float(s)              # Reads the value: a barrier every step
read_every = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(1000):
    s = step(s, x)                # Stays on the device: no barrier
total = float(s)                  # One read at the end
read_once = time.perf_counter() - t0
print(f'read every step: {read_every:.3f} s   read once: {read_once:.3f} s')
```

:begin_tab:`pytorch`
Without per-step reads, the three calls per step take ~40 µs of host dispatch
time (longer than the ~26 µs of tf32 device work), so the GPU runs behind the
host. Reading `float(s)` every step forces host and device to take turns,
slowing the loop by ~1.6×.
:end_tab:

:begin_tab:`jax`
Without per-step reads, each jitted call takes ~0.1 ms of host dispatch time.
Reading `float(s)` every step adds the device-to-host round-trip $\sigma$ on
every iteration, making the loop more than three times slower.
:end_tab:

Training loops should therefore accumulate metrics on the device and read them
only at logging intervals.

### A Benchmark Helper

From here on we time operations with `d2l.Benchmark`, which runs warm-up calls
to absorb compilation and kernel-loading overhead, then times `repeats` calls
between two device barriers. As a first check, a $4096 \times 4096$ bf16
product has intensity $I \approx 1365$ FLOP/byte, well above the ridge, so the
roofline predicts $2 \cdot 4096^3 / P \approx 0.83$ ms.

```{.python .input #performance-model-measuring-without-lying-4}
%%tab pytorch
a = torch.randn(4096, 4096, device=d2l.try_gpu(), dtype=torch.bfloat16)
d2l.Benchmark(lambda: torch.mm(a, a), desc='4096 x 4096 product')
```

```{.python .input #performance-model-measuring-without-lying-4}
%%tab jax
matmul = jax.jit(jnp.dot)
a = jax.random.normal(d2l.get_key(), (4096, 4096), dtype=jnp.bfloat16)
d2l.Benchmark(lambda: matmul(a, a), desc='4096 x 4096 product')
```

The measured time is within about 10% of the 0.83 ms prediction.

## Three Regimes
:label:`subsec_perf-regimes`

The wall-clock time of a GPU program divides into compute-bound kernels,
bandwidth-bound kernels, and idle gaps where the device waits for the host
:cite:`He.2022`. Three experiments isolate each regime.

### The Bandwidth Roof and the Compute Roof
:label:`subsec_perf-roofline-test`

First, we multiply an activation matrix of $B$ rows by a $d \times d$ bf16
weight matrix with $d = 16{,}384$ (512 MiB). By :eqref:`eq_matmul_intensity`,
its intensity is approximately $B$, crossing the ridge near $B = 164$. Below
the ridge, every call must stream 512 MiB of weights (~0.53 ms at
1.008 TB/s) regardless of $B$; above the ridge, the $2Bd^2$ FLOPs govern the
time, scaling linearly with $B$ (0.83 ms at $B = 256$, 3.32 ms at
$B = 1024$).

```{.python .input #hardware-batched-decoding}
%%tab pytorch
peak_tflops = 165.2    # RTX 4090: dense bf16 on the tensor cores
bandwidth_tbs = 1.008  # RTX 4090: GDDR6X memory
print(f'{torch.cuda.get_device_name()}: ridge point '
      f'{peak_tflops / bandwidth_tbs:.0f} FLOP/byte')
d, b = 16384, 2  # Width and bytes per bf16 number
W = torch.randn(d, d, dtype=torch.bfloat16, device=d2l.try_gpu())  # 0.54 GB
for B in (1, 4, 16, 64, 128, 256, 1024):
    X = torch.randn(B, d, dtype=torch.bfloat16, device=W.device)
    t = d2l.Benchmark(lambda: X @ W).time
    flops, nbytes = 2 * B * d * d, b * (B * d + d * d + B * d)
    t_mem, t_comp = nbytes / bandwidth_tbs / 1e12, flops / peak_tflops / 1e12
    roof = 'memory' if t_mem > t_comp else 'compute'
    print(f'B = {B:4d}: bound {1e3 * max(t_mem, t_comp):4.2f} ms ({roof:7s}), '
          f'measured {1e3 * t:4.2f} ms, {flops / t / 1e12:5.1f} TFLOP/s')
```

```{.python .input #hardware-batched-decoding}
%%tab jax
peak_tflops = 165.2    # RTX 4090: dense bf16 on the tensor cores
bandwidth_tbs = 1.008  # RTX 4090: GDDR6X memory
print(f'{jax.devices()[0].device_kind}: ridge point '
      f'{peak_tflops / bandwidth_tbs:.0f} FLOP/byte')
d, b = 16384, 2  # Width and bytes per bf16 number
W = jax.random.normal(d2l.get_key(), (d, d), dtype=jnp.bfloat16)  # 0.54 GB
for B in (1, 4, 16, 64, 128, 256, 1024):
    X = jax.random.normal(d2l.get_key(), (B, d), dtype=jnp.bfloat16)
    t = d2l.Benchmark(lambda: matmul(X, W)).time
    flops, nbytes = 2 * B * d * d, b * (B * d + d * d + B * d)
    t_mem, t_comp = nbytes / bandwidth_tbs / 1e12, flops / peak_tflops / 1e12
    roof = 'memory' if t_mem > t_comp else 'compute'
    print(f'B = {B:4d}: bound {1e3 * max(t_mem, t_comp):4.2f} ms ({roof:7s}), '
          f'measured {1e3 * t:4.2f} ms, {flops / t / 1e12:5.1f} TFLOP/s')
```

One weight matrix reaches both roofs: from $B = 1$ to $B = 128$ the time stays
flat near 0.6 ms despite $128\times$ more arithmetic, and from $B = 256$
upward it grows linearly with $B$ near the compute roof. Serving 1,024 tokens
at once performs $1{,}024\times$ the arithmetic of serving one token in less
than $6\times$ the time.

### Elementwise Operations

Next, we stay far below the ridge and vary the arithmetic per element on an
$8192 \times 8192$ fp32 tensor (256 MiB): addition, multiplication, sine, and
sigmoid. Each reads 256 MiB and writes 256 MiB (512 MiB total, or 0.53 ms at
1.008 TB/s).

```{.python .input #performance-model-three-regimes-1}
%%tab pytorch
x = torch.randn(8192, 8192, device=d2l.try_gpu())  # 256 MiB in fp32
for desc, f in [('add', lambda x: x + 1), ('mul', lambda x: x * 1.5),
                ('sin', torch.sin), ('sigmoid', torch.sigmoid)]:
    print(d2l.Benchmark(lambda: f(x), desc=desc))
```

```{.python .input #performance-model-three-regimes-1}
%%tab jax
x = jax.random.normal(d2l.get_key(), (8192, 8192))  # 256 MiB in fp32
for desc, f in [('add', lambda x: x + 1), ('mul', lambda x: x * 1.5),
                ('sin', jnp.sin), ('sigmoid', jax.nn.sigmoid)]:
    f = jax.jit(f)
    print(d2l.Benchmark(lambda: f(x), desc=desc))
```

All four take ~0.6 ms, achieving ~0.9 TB/s of effective memory bandwidth:
because memory traffic sets the time, transcendental arithmetic (`sin`,
`sigmoid`) costs no more time than `x + 1`.

In eager execution, chaining elementwise operations compounds this memory
traffic. The tanh approximation to GELU below (`gelu_ish`) chains 8
elementwise operations. Six unary operations each read and write one tensor
(12 tensors), and two binary operations (`x + 0.04 * x**3` and the outer
product with `x`) each read two tensors and write one (6 tensors), moving 18
tensors of 256 MiB in total—$9\times$ the bytes of a single addition.

```{.python .input #performance-model-three-regimes-2}
%%tab pytorch
def gelu_ish(x):  #@save
    """The tanh approximation of GELU with rounded constants: a chain of
    eight elementwise operations."""
    return 0.5 * x * (1 + torch.tanh(0.8 * (x + 0.04 * x**3)))

print(d2l.Benchmark(lambda: x + 1, desc='1 op'))
print(d2l.Benchmark(lambda: gelu_ish(x), desc='chain'))  # Eager: 8 kernels
```

```{.python .input #performance-model-three-regimes-2}
%%tab jax
def gelu_ish(x):  #@save
    """The tanh approximation of GELU with rounded constants: a chain of
    eight elementwise operations."""
    return 0.5 * x * (1 + jnp.tanh(0.8 * (x + 0.04 * x**3)))

print(d2l.Benchmark(lambda: x + 1, desc='1 op'))
print(d2l.Benchmark(lambda: gelu_ish(x), desc='chain'))  # No jit: 8 kernels
```

As predicted, the unfused chain takes $8\times$ to $9\times$ as long as `x + 1`.
*Kernel fusion* (:numref:`sec_compilation`) merges the chain into a single
kernel that reads `x` once and writes the output once.

### Matrix Products across Sizes
:label:`subsec_perf-sweep`

Does a square bf16 product reach the compute roof as soon as $I = n / 3$
crosses 164 near $n = 500$? Let us sweep $n$ from 256 to 8,192.

```{.python .input #performance-model-the-sweep-mapping-our-gpu}
%%tab pytorch
sizes = [256, 512, 1024, 2048, 4096, 8192]
times = []
for n in sizes:
    a = torch.randn(n, n, device=d2l.try_gpu(), dtype=torch.bfloat16)
    times.append(d2l.Benchmark(lambda: torch.mm(a, a), repeats=100).time)
```

```{.python .input #performance-model-the-sweep-mapping-our-gpu}
%%tab jax
sizes = [256, 512, 1024, 2048, 4096, 8192]
times = []
for n in sizes:
    a = jax.random.normal(d2l.get_key(), (n, n), dtype=jnp.bfloat16)
    times.append(d2l.Benchmark(lambda: matmul(a, a), repeats=100).time)
```

We plot the achieved TFLOP/s against the arithmetic intensity $n / 3$
alongside the roofline bound :eqref:`eq_roofline`.

```{.python .input #performance-model-matrix-products-across-sizes-1}
%%tab pytorch, jax
measured = [2 * n**3 / t / 1e12 for n, t in zip(sizes, times)]
intensity = [n / 3 for n in sizes]  # FLOP/byte of a square bf16 product
ridge = peak_tflops / bandwidth_tbs
roof_x = [intensity[0], ridge, intensity[-1]]
roof_y = [min(peak_tflops, bandwidth_tbs * i) for i in roof_x]
d2l.plot([roof_x, intensity], [roof_y, measured],
         'arithmetic intensity (FLOP/byte)', 'TFLOP/s',
         legend=['roofline bound', 'measured'],
         xscale='log', yscale='log', fmts=('-', 'o'))
```

```{.python .input #performance-model-matrix-products-across-sizes-2}
%%tab pytorch, jax
for n, t, tf in zip(sizes, times, measured):
    print(f'n = {n:4d}: {1e6 * t:7.1f} us per call, {tf:5.1f} TFLOP/s')
```

At $n = 4096$ and $8192$ the product runs near the 165 TFLOP/s peak. At small
sizes, however, the points lie far below the roofline and climb with a log–log
slope of 3—the signature of a constant time per call while work scales as
$n^3$. From $n = 256$ to $1024$ the work grows $64\times$, yet the time per
call changes by less than $2\times$.

To see whether that constant time is spent on the host or the device, we time
a $256 \times 256$ product alongside `x + 1` of the same shape, and use
`d2l.kernel_times` (defined in :numref:`subsec_perf-profiler`) to measure how
long the GPU itself is busy per call.

```{.python .input #performance-model-matrix-products-across-sizes-3}
%%tab pytorch
a = torch.randn(256, 256, device=d2l.try_gpu(), dtype=torch.bfloat16)
for desc, f in [('256 x 256 product', lambda: torch.mm(a, a)),
                ('x + 1 on 256 x 256', lambda: a + 1)]:
    wall = d2l.Benchmark(f, repeats=100).time
    busy = sum(d2l.kernel_times(f, steps=100).values()) / 1e3  # Seconds
    print(f'{desc:18s}: {1e6 * wall:5.1f} us per call, '
          f'device busy {1e6 * busy:4.1f} us')
```

```{.python .input #performance-model-matrix-products-across-sizes-3}
%%tab jax
add_one = jax.jit(lambda x: x + 1)
a = jax.random.normal(d2l.get_key(), (256, 256), dtype=jnp.bfloat16)
for desc, f in [('256 x 256 product', lambda: matmul(a, a)),
                ('x + 1 on 256 x 256', lambda: add_one(a))]:
    wall = d2l.Benchmark(f, repeats=100).time
    busy = sum(d2l.kernel_times(f, steps=100).values()) / 1e3  # Seconds
    print(f'{desc:18s}: {1e6 * wall:5.1f} us per call, '
          f'device busy {1e6 * busy:4.1f} us')
```

In both cases the GPU works for only a few microseconds; the rest is the host
dispatch time $h$ (Python, framework, and driver). For a $256 \times 256$
matrix multiplication, the most expensive part of the operation is the line
of Python that asks for it. When calls are queued back-to-back, host and
device overlap and each call takes $\max(h, 2n^3/P)$. For small $n$, the
device finishes before the host issues the next call, placing the program in
the *overhead-bound* regime. The knee where compute overtakes dispatch lies
at $n \approx (hP/2)^{1/3}$: near $n = 1{,}000$ in PyTorch ($h \approx 13$ µs)
and near $n = 1{,}600\text{--}2{,}000$ for single-op JAX dispatch
($h \approx 50\text{--}100$ µs, paid once for an entire jitted training step).

Combining all three floors gives our fundamental time bound for a single
dispatched operation:

$$
t \;\gtrsim\; \max\left(h,\; \frac{\textrm{FLOPs}}{P},\;
\frac{\textrm{bytes}}{\beta}\right).
$$
:eqlabel:`eq_perf-time`

With $h = 10$ µs, any operation with fewer than $hP \approx 1.6 \times 10^9$
FLOPs and fewer than $h\beta \approx 10$ MB moved takes the time of its launch
alone. Moreover, the knee $n \approx (hP/2)^{1/3}$ grows with $P$: faster GPUs
require larger operations to avoid becoming overhead-bound.

### Diagnosis and Fixes

In a full training step the three regimes mix, and the dominant share names
the step's regime (:numref:`fig_regimes`). :numref:`tab_perf_regimes`
summarizes the symptom and standard remedies for each regime.

![The three regimes, schematically. Where the time of a step goes, among
compute-bound kernels, bandwidth-bound kernels, and idle gaps, determines
which fix applies; each bar names one typical
fix.](../img/mdl-perf-regimes.svg)
:label:`fig_regimes`

:Symptoms and standard fixes of the three regimes.
:label:`tab_perf_regimes`

| regime | symptom | standard fix |
|---|---|---|
| compute-bound | device busy; intensity above the ridge; time grows with the FLOPs | less arithmetic; a format with faster tensor cores; more devices |
| bandwidth-bound | device busy; intensity below the ridge; time grows with the bytes moved | a larger batch, which raises the intensity of the products; a narrower format; fusion; recomputation instead of storage; algorithms that reuse data on the chip |
| overhead-bound | device idle between kernels; time per call independent of size | a larger batch, which means fewer calls per example; capturing the whole program and replaying it without Python |

## Profiling a Training Step
:label:`subsec_perf-profiler`

A *profiler* records the duration of every GPU kernel so we can compare busy
device time against wall-clock time and break down device time by kernel type.
Before profiling, let us predict the time of one SGD step of an MLP
($4096 \to 4096 \to 4096 \to 10$) on a minibatch of 1,024 examples in fp32
with tf32 matrix products ($P = 82.6$ TFLOP/s).

### Predicting the Step

Each of the two $4096 \times 4096$ layers performs
$2 \cdot 1024 \cdot 4096^2 \approx 3.44 \times 10^{10}$ FLOPs per forward
product. A training step runs five such products (two forward, two weight-gradient,
and one input-gradient for the second hidden layer). At $I \approx 340$
FLOP/byte (above the tf32 ridge of 82), each product takes
$3.44 \times 10^{10} / (82.6 \times 10^{12}) \approx 0.42$ ms. The SGD update
reads each of the 33.6M fp32 weights and its gradient and writes the new
weight ($12 \times 33.6 \times 10^6 \approx 0.40$ GB), taking ~0.40 ms at
1 TB/s. Neglecting the small final layer and ReLUs, the step should take

$$
5 \times 0.42 \textrm{ ms} + 0.40 \textrm{ ms} \;\approx\; 2.5 \textrm{ ms}.
$$

```{.python .input #performance-model-profiling-a-training-step-1}
%%tab pytorch
net = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(),
                    nn.Linear(4096, 4096), nn.ReLU(),
                    nn.Linear(4096, 10)).to(d2l.try_gpu())
X = torch.randn(1024, 4096, device=d2l.try_gpu())
opt = torch.optim.SGD(net.parameters(), lr=0.01)

def step():
    opt.zero_grad()
    net(X).mean().backward()
    opt.step()

wall = d2l.Benchmark(step).time
```

```{.python .input #performance-model-profiling-a-training-step-1}
%%tab jax
def mlp(params, X):
    for W, b in params[:-1]:
        X = jax.nn.relu(X @ W + b)
    W, b = params[-1]
    return X @ W + b

@jax.jit
def step(params, X):
    grads = jax.grad(lambda p: mlp(p, X).mean())(params)
    return jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)

widths = [4096, 4096, 4096, 10]
params = [(jax.random.normal(d2l.get_key(), (m, n)) / m**0.5, jnp.zeros(n))
          for m, n in zip(widths[:-1], widths[1:])]
X = jax.random.normal(d2l.get_key(), (1024, 4096))
wall = d2l.Benchmark(lambda: step(params, X)).time
```

`kernel_times` runs a function `steps` times under the framework profiler and
returns the average device milliseconds per call for every kernel launched.

```{.python .input #performance-model-the-profiler}
%%tab pytorch
def kernel_times(f, steps=10):  #@save
    """Device milliseconds per call of f, for every kernel that f runs."""
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            acc_events=True) as prof:
        for _ in range(steps):
            f()
        torch.cuda.synchronize()
    return {e.key: e.device_time_total / 1e3 / steps
            for e in prof.key_averages()
            if e.device_type == torch.autograd.DeviceType.CUDA}
```

```{.python .input #performance-model-the-profiler}
%%tab jax
#@save
import glob
import tempfile

def kernel_times(f, steps=10):  #@save
    """Device milliseconds per call of f, for every kernel that f runs."""
    kernels = collections.Counter()
    with tempfile.TemporaryDirectory() as trace_dir:  # Removed afterwards
        with jax.profiler.trace(trace_dir):
            for _ in range(steps):
                out = f()
            jax.block_until_ready(out)
        xspace = glob.glob(f'{trace_dir}/**/*.xplane.pb', recursive=True)[0]
        gpu = jax.profiler.ProfileData.from_file(
            xspace).find_plane_with_name('/device:GPU:0')
        for line in gpu.lines:
            if line.name.startswith('Stream'):  # One line per CUDA stream
                for e in line.events:
                    kernels[e.name] += e.duration_ns / 1e6 / steps
    return kernels
```

Both profilers can also export timeline traces (`prof.export_chrome_trace` in
PyTorch; `jax.profiler.trace` directories in JAX) viewable at
<https://ui.perfetto.dev>.

### Reading the Profile

`show_profile` compares busy device time against wall-clock step time, computes
the share spent in matrix-product kernels, and lists the top kernels.

```{.python .input #performance-model-reading-the-profile-1}
%%tab pytorch, jax
PRODUCT_KERNELS = 'gemm|xmma|cutlass|matmul|sdpa'  #@save
# Kernel names that mark a matrix product: cuBLAS and CUTLASS kernels (gemm,
# xmma, cutlass, which also names PyTorch's fused attention), XLA's fused
# products (gemm_fusion_dot), and matmul or sdpa kernels

def show_profile(kernels, step_ms, rows=6):  #@save
    """Busy device time against the wall-clock time step_ms of a step, the
    share of the matrix products, and the kernels with the most time."""
    busy = sum(kernels.values())
    products = sum(ms for name, ms in kernels.items()
                   if re.search(PRODUCT_KERNELS, name, re.I))
    print(f'wall clock {step_ms:.2f} ms per step, device busy {busy:.2f} ms '
          f'({busy / step_ms:.0%}), matrix products {products / busy:.0%}')
    for name, ms in sorted(kernels.items(), key=lambda kv: -kv[1])[:rows]:
        name = re.sub(r'^void |\(anonymous namespace\)::|\w+::|\(.*', '', name)
        print(f'{ms:6.2f} ms {ms / busy:4.0%}  {name[:52]}')
```

```{.python .input #performance-model-profiling-a-training-step-2}
%%tab pytorch
show_profile(kernel_times(step), 1000 * wall)
```

```{.python .input #performance-model-profiling-a-training-step-2}
%%tab jax
show_profile(kernel_times(lambda: step(params, X)), 1000 * wall)
```

The measured step comes within ~10% of the predicted 2.5 ms, with the GPU busy
for nearly the entire step. Matrix products (*gemm* kernels) account for over
three quarters of the device time.

:begin_tab:`pytorch`
The three `gemm` lines aggregate the five large matrix products, and
`multi_tensor_apply_kernel` is the batched SGD update at the predicted 0.4 ms.
:end_tab:

:begin_tab:`jax`
XLA fuses the SGD update directly into the weight-gradient products
(`gemm_fusion_dot`), subtracting the scaled gradient before writing the
updated weight so the gradient never makes a separate round trip to memory.
:end_tab:

Notice that an unfused optimizer update takes a fixed time independent of
batch size: at batch size 128, the five products shrink eightfold to ~0.07 ms
each, leaving the 0.4 ms update larger than all five products combined.

## Summary

A GPU operation is bounded by
$t \gtrsim \max(h,\, \textrm{FLOPs}/P,\, \textrm{bytes}/\beta)$, where $h$ is
host dispatch overhead, $P$ is peak arithmetic throughput, and $\beta$ is
memory bandwidth. Arithmetic intensity $I = \textrm{FLOPs}/\textrm{bytes}$
compares the last two: below the ridge point $I^* = P/\beta$ (~164 FLOP/byte
for bf16 on the RTX 4090), an operation is bandwidth-bound; above it,
compute-bound. For a linear layer, intensity scales with batch size from ~1
FLOP/byte during single-token decoding to hundreds of FLOP/byte during
training. Accurate benchmarking requires device synchronization outside loops
and warm-up runs (`d2l.Benchmark`), while `d2l.kernel_times` and
`d2l.show_profile` verify paper predictions kernel by kernel.

## Exercises

1. **The smallest dimension.** The exact expression for the intensity of
   the product $\mathbf{X}\mathbf{W}$ in :eqref:`eq_matmul_intensity`, before
   the $B \ll D, F$ approximation, holds for all shapes. Let
   $m = \min(B, D, F)$.
    1. Show that the intensity lies between $2m/(3b)$ and $2m/b$. Conclude
       that in bf16, on a GPU with ridge point 164 FLOP/byte, every matrix
       product with a dimension below 164 is bandwidth-bound in the roofline
       model.
    1. Apply the bound to three bf16 products: the final layer of the MLP
       of :numref:`subsec_perf-profiler` ($4096 \to 10$ on a batch of 4,096);
       the attention scores $\mathbf{Q}\mathbf{K}^\top$ of one head with
       $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{T \times 64}$; and a
       $1 \times 1$ convolution (:numref:`subsec_1x1`) from 64 to 64 channels
       on a batch of 128 images of $32 \times 32$ pixels. Which dimensions
       would have to grow for each to reach the ridge point?
    1. A multi-head attention layer computes the scores of all its heads in
       one batched kernel. Does batching the heads change the intensity of
       the scores?

2. [code] **Batch size and the ridge.** Consider the product
   $\mathbf{X}\mathbf{W}$ with $\mathbf{X} \in \mathbb{R}^{B \times 8192}$
   and $\mathbf{W} \in \mathbb{R}^{8192 \times 8192}$ in bf16.
    1. Using the exact expression in :eqref:`eq_matmul_intensity`, find the
       batch size at which the product reaches the ridge point of your GPU,
       and compare it with the small-$B$ approximation.
    1. Predict from :eqref:`eq_perf-time` the time of the product for $B = 1$
       and $B = 64$ concurrent decoding sequences, measure both with
       `d2l.Benchmark`, and determine up to what batch size serving more
       sequences costs almost no extra time per step.

:begin_tab:`pytorch`
3. [code] **Hidden barriers.** Each of the following lines runs once per
    step of a training loop, where `x` and `g` are GPU tensors and `c` is a
    Python float: `x[x > 0].sum()`, `print(x.shape)`, `x.argmax()`,
    `if g.norm() > c: g *= c / g.norm()`, and
    `x * torch.tensor(0.5, device=x.device)`.
    1. Decide which lines make the host wait for the device, and justify each.
    1. Rewrite every synchronizing line so that it computes the same result
       without a host barrier.
    1. Check your answers with `torch.cuda.set_sync_debug_mode('warn')`.
:end_tab:

:begin_tab:`jax`
3. [code] **Hidden barriers.** Each of the following lines runs once per
    step of a training loop, where `x` and `g` are GPU arrays and `c` is a
    Python float: `x[x > 0].sum()`, `print(x.shape)`, `jnp.argmax(x)`,
    `if jnp.linalg.norm(g) > c: g = g * c / jnp.linalg.norm(g)`, and
    `x * jnp.asarray(0.5)`.
    1. Decide which lines make the host wait for the device, and justify each.
    1. Rewrite every synchronizing line so that it computes the same result
       without a host barrier.
    1. Check your answers inside
       `with jax.transfer_guard_device_to_host('disallow'):`.
:end_tab:

4. **Diagnosis from timings.** Suppose `d2l.Benchmark` reports the following
   times per call for four bf16 operations on the RTX 4090 ($P = 165$ TFLOP/s,
   $\beta = 1.008$ TB/s): Operation A multiplies two $8192 \times 8192$
   matrices in 7.4 ms; Operation B applies row-wise softmax to an
   $8192 \times 8192$ matrix in 0.30 ms (~5 FLOPs/element, one read and one
   write); Operation C multiplies two $256 \times 256$ matrices in 10 µs;
   Operation D multiplies two $1536 \times 1536$ matrices in 0.10 ms.
    1. For each operation, compute its arithmetic intensity and the fractions
       of $P$ and $\beta$ it achieves. Classify each operation by its
       intensity relative to the ridge point and by the roof it approaches.
       Where do the two classifications disagree?
    1. For each operation close to neither roof, distinguish between host
       dispatch latency $h$ and insufficient thread-block tiles to occupy all
       streaming multiprocessors, and state what a profiler timeline would show
       for each.

5. [code] **A pipeline model of dispatch.** ● Model one GPU call by three
   durations: host enqueue time $h$, device execution time $t_\textrm{dev}$,
   and final barrier latency $\sigma$.
    1. Show that $R$ calls enqueued back-to-back followed by one barrier take
       $R \max(h, t_\textrm{dev}) + \min(h, t_\textrm{dev}) + \sigma$. What
       does `d2l.Benchmark` report when $t_\textrm{dev} < h$?
    1. For the loops of :numref:`subsec_perf-measuring`, derive the ratio of
       the per-step-read loop time to the single-read loop time in terms of
       $h$, $t_\textrm{dev}$, and $\sigma$.
    1. Run both loops for matrix sides $256, 512, 1024, 2048, 4096$, estimate
       $h$, $\sigma$, and $t_\textrm{dev}$, and determine above what device
       time per step reading a scalar every step costs less than 10%.

6. [code] **Tiles and waves.** ● A GPU consists of $n_\textrm{SM}$ streaming
   multiprocessors ($n_\textrm{SM} = 128$ on the RTX 4090). Suppose a matrix-product
   kernel partitions an $n \times n$ output into $128 \times 128$ tiles, each
   SM computes one tile at a time at rate $P / n_\textrm{SM}$, and tiles run
   in waves of at most $n_\textrm{SM}$.
    1. Derive the fraction of peak throughput this model allows as a function
       of $n$ and $n_\textrm{SM}$, and compare it with your sweep measurements
       in :numref:`subsec_perf-sweep`.
    1. Predict and measure the fraction of peak for $n = 2048, 2176, 2304,
       2432, 2560$ (consecutive multiples of 128), and explain where the
       measurements depart from the simple wave model.
    1. Describe two ways a kernel can split an $n = 512$ product into more work
       items so all SMs participate, and state the overhead each adds.

[Discussions](https://d2l.discourse.group/t/36787)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.1]{.kicker}

The Performance Model<br>
**arithmetic intensity · the roofline · GPU timing · three regimes · profiling**
:::
:::

::: {.slide title="The Same Product Runs Below 2% or Near 100% of the Peak"}
[The problem]{.kicker}

RTX 4090, bf16 tensor cores, peak $P$ = 165 TFLOP/s (derived from the
card's design).

- $256 \times 256$ times $256 \times 256$: under 2% of $P$.
- $8192 \times 8192$ times $8192 \times 8192$: within a few percent of $P$.

Same operation, same chip, timed correctly.

::: {.d2l-note}
A training step is a long sequence of such operations. **What limits the
small one, and can we predict it before running?**
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- count the FLOPs and bytes of an operation and compute its intensity;
- bound its time by $\max(h,\ \textrm{FLOPs}/P,\ \textrm{bytes}/\beta)$ and
  name its regime;
- time GPU code correctly and find hidden synchronization points;
- predict a training step and check the prediction against a profile.

**Running example:** one RTX 4090 ($P$ = 165 TFLOP/s in bf16,
$\beta$ = 1.008 TB/s) and the width-2048 linear layers of a 1B-parameter
GPT.

**Idea:** two counts and one measured constant predict which resource
limits an operation.
:::

::: {.slide title="The Performance Model Opens the Chapter"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| **The Performance Model** | **roofline, host time, timing, profiling** |
| Hardware | where peak arithmetic and bandwidth come from |
| Computation Graphs and Compilation | fusion and kernel replay |
| Memory and Precision | what fits: bytes per parameter and per token |
| Multi-GPU from First Principles | data parallelism by hand, its cost model |
| Data Parallelism in Practice | one process per GPU, overlap |
| Sharding the Training State | training state divided among GPUs |
| Case Study: Optimizing a Transformer | the whole method on one GPT |
:::

::: {.col .narrow}
- **Uses:** matrix products, bf16 and tf32, the GPU's asynchronous queue.
- **Skips:** writing GPU kernels; machines with more than one node.
:::
:::
:::

::: {.slide title="Intensity Is FLOPs per Byte Moved"}
$$I = \frac{\textrm{FLOPs performed}}{\textrm{bytes moved}}$$

Product $\mathbf{X}\mathbf{W}$, $\mathbf{X}$: $B \times D$, $\mathbf{W}$:
$D \times F$, $b$ bytes per number:

$$I = \frac{2BDF}{b\,(BD + DF + BF)} \;\approx\; \frac{2B}{b}
\quad \textrm{for } B \ll D, F$$

Elementwise $y_i = x_i + 1$: one FLOP per $2b$ bytes, $I = 0.25$ in bf16.

::: {.d2l-note}
Each weight fetched from memory is used once per row of the batch.
:::
:::

::: {.slide title="One GPT Layer Spans a Factor of a Thousand in Intensity"}
Linear layer of width $D = F = 2048$, bf16 ($b = 2$):

| use | rows $B$ | intensity |
|---|---|---|
| training, 8 sequences of 1,024 tokens | 8,192 | $\approx 910$ FLOP/byte |
| generating text, one token | 1 | $\approx 1$ FLOP/byte |

- Training: the width, not the batch, is the smallest dimension.
- Generation: every weight is read to be used once.
:::

::: {.slide title="The Roofline Caps Throughput at min(P, Iβ)"}
![](../img/mdl-perf-roofline.svg){width=72%}

$P$: peak FLOP/s; $\beta$: memory bandwidth; ridge $I^* = P/\beta$.
Below it: **bandwidth-bound**, extra arithmetic costs no time. Above it:
**compute-bound**, fewer bytes do not help.
:::

::: {.slide title="The RTX 4090 Needs 164 FLOPs per Byte"}
$$I^* = \frac{P}{\beta} = \frac{165.2 \textrm{ TFLOP/s}}{1.008 \textrm{ TB/s}}
\approx 164 \textrm{ FLOP/byte}$$

- $P$ depends on the format: tf32 products run at 82.6 TFLOP/s, ridge 82.
- Square bf16 product, $B = D = F = n$: $I = n/3$, compute-bound from
  $n \approx 500$, on paper.
- $\beta$ is main memory: the model covers operands larger than the 72 MB
  on-chip cache.
:::

::: {.slide title="Check: Bound the Time of Each Operation"}
[Check]{.kicker}

RTX 4090, bf16, ridge 164 FLOP/byte. Bound the time of each operation.

1. $x + 1$ on $10^8$ elements.
2. The width-2048 layer applied to one token.
3. A product of two $4096 \times 4096$ matrices.

::: {.fragment .fade-up}
1. $I = 0.25$: moves $4 \times 10^8$ bytes, at least 0.4 ms.
2. $I \approx 1$: reads 8.4 MB of weights, at least 8 µs.
3. $I \approx 1365$: $1.4 \times 10^{11}$ FLOPs at 165 TFLOP/s, at least
   0.83 ms.
:::
:::

::: {.slide title="A Clock Without a Barrier Measures the Enqueue"}
![](../img/mdl-perf-async-timeline.svg){width=82%}

- Python queues kernels and moves on; the GPU runs them behind it.
- Correct timing: warm up, wait for the device before starting and before
  stopping, average many calls.
:::

::: {.slide title="Reading a Value Every Step Makes Host and Device Take Turns"}
$h$: host time to issue a call (a step of several calls: their sum);
$t_\textrm{dev}$: device time of a step; $\sigma$: round trip of a read.

$$t_\textrm{once} \approx \max(h, t_\textrm{dev}), \qquad
t_\textrm{every} \lesssim h + t_\textrm{dev} + \sigma$$

1,000 steps of a $1024 \times 1024$ product, total read every step or once:

@!performance-model-measuring-without-lying-2

::: {.d2l-note}
Hidden reads: `float(x)`, `print(x)`, `if loss < 0.1:`, `x[x > 0]`.
:::
:::

::: {.slide title="Benchmark Packages the Timing Rules"}
```{.python #performance-model-benchmark-rules}
for _ in range(warmup):       # load kernels, compile
    f()
wait_for_device()
t0 = time.perf_counter()
for _ in range(repeats):
    f()
wait_for_device()
seconds_per_call = (time.perf_counter() - t0) / repeats
```

@!performance-model-measuring-without-lying-4

Predicted: $2 \cdot 4096^3 / P$ = 0.83 ms. Measured: within about a tenth.
:::

::: {.slide title="One Matrix Reaches Both Roofs"}
$16{,}384 \times 16{,}384$ bf16 weight (512 MiB) times $B$ rows. Predicted:
0.53 ms for every $B$ below the ridge ($B \approx 164$), then $\propto B$.

@!hardware-batched-decoding

::: {.d2l-note}
Up to $B = 128$ the time hardly moves. A thousand times the work costs
five to six times the time.
:::
:::

::: {.slide title="Check: Which Elementwise Kernel Is Slowest?"}
[Check]{.kicker}

$8192 \times 8192$ fp32 (256 MiB): `x + 1`, `x * 1.5`, `sin(x)`,
`sigmoid(x)`. Which takes longest, and how long?

::: {.fragment .fade-up}
@!performance-model-three-regimes-1

None: each moves 512 MiB, at least 0.53 ms at $\beta$. Measured: about
0.9 TB/s, whatever the kernel computes.
:::
:::

::: {.slide title="An Unfused Chain Moves Nine Times the Bytes"}
```{.python #performance-model-gelu-chain}
0.5 * x * (1 + tanh(0.8 * (x + 0.04 * x**3)))   # 8 operations
```

Predicted: 8 kernels read and write 18 tensors of 256 MiB; one addition
moves 2.

@!performance-model-three-regimes-2

::: {.d2l-note}
One fused kernel would move 2 tensors instead of 18.
:::
:::

::: {.slide title="Small Products Run Far Below the Roofline"}
@!performance-model-matrix-products-across-sizes-1

Up to $n = 1024$ the time per call grows far less than the work, and the
points climb with slope three, the mark of a fixed time per call. By
$n = 4096$ the products reach the roof.

::: {.d2l-note}
On paper compute-bound from $n \approx 500$; in fact from 1,000 to 2,000,
depending on the host time per call.
:::
:::

::: {.slide title="The Device Works a Few Microseconds of Each Small Call"}
$256 \times 256$ bf16: wall-clock time per call against the device's busy
time, from the profiler.

@!performance-model-matrix-products-across-sizes-3

The rest of each call is the host time $h$: Python, the framework, and the
driver that launches the kernel.

::: {.d2l-note}
For a 256 × 256 product, the most expensive part of the computation is the
line of Python that asks for it.
:::
:::

::: {.slide title="Each Call Costs at Least the Largest of Three Floors"}
$$t \;\gtrsim\; \max\left(h,\ \frac{\textrm{FLOPs}}{P},\
\frac{\textrm{bytes}}{\beta}\right)$$

- The largest term names the regime: **overhead-**, compute-, or
  bandwidth-bound.
- $h$ is measured: from about 10 µs to 0.1 ms per call on our machine,
  depending on the software stack.
- Square product: knee at $n \approx (hP/2)^{1/3}$; near 1,000 for
  $h$ = 13 µs, near 2,000 for $h$ = 0.1 ms.
- $h$ = 10 µs: under $1.6 \times 10^9$ FLOPs and 10 MB, an operation costs
  one launch. The width-2048 layer on one token moves 8.4 MB in about
  8 µs: limited by the launch before the bandwidth.
:::

::: {.slide title="Check: Faster GPUs Need Larger Operations"}
[Check]{.kicker}

Where is the knee $n \approx (hP/2)^{1/3}$ of a square bf16 product?

1. RTX 4090 ($P$ = 165 TFLOP/s) with a host time of $h$ = 5 µs.
2. B200 ($P$ = 2,250 TFLOP/s) with the same $h$ = 5 µs.

::: {.fragment .fade-up}
1. $(5 \times 10^{-6} \cdot 1.65 \times 10^{14} / 2)^{1/3} \approx 740$.
2. $(5 \times 10^{-6} \cdot 2.25 \times 10^{15} / 2)^{1/3} \approx 1800$.

With the same host, a 14 times faster GPU needs products
$14^{1/3} \approx 2.4$ times wider to stay busy.
:::
:::

::: {.slide title="Each Regime Has Its Standard Fixes"}
![](../img/mdl-perf-regimes.svg){width=70%}

| regime | symptom | fix |
|---|---|---|
| compute | busy, $I$ above the ridge | less arithmetic, a format with faster tensor cores, more GPUs |
| bandwidth | busy, $I$ below the ridge | larger batch, narrower format, fusion |
| overhead | idle, time independent of size | larger batch, replay the whole step without Python |
:::

::: {.slide title="Check: Predict the Time of the MLP Step"}
[Check]{.kicker}

MLP $4096 \to 4096 \to 4096 \to 10$, minibatch 1,024, fp32 with tf32
products ($P$ = 82.6 TFLOP/s), one SGD step. Predict its time.

::: {.fragment .fade-up}
- Five large products (2 forward, 3 backward), $3.4 \times 10^{10}$ FLOPs
  each: $5 \times 0.42$ ms.
- SGD: 12 bytes × 34 M weights = 0.4 GB, 0.4 ms.
- Step: about **2.5 ms**.
:::
:::

::: {.slide title="The Profile Confirms the Prediction within a Tenth"}
@!performance-model-profiling-a-training-step-2

- Device busy almost all the time: not overhead-bound.
- Products: three quarters or more of the device time, about 0.42 ms per
  product, more where the update is fused in.
- The update: its own 0.4 ms kernel, or fused into the products.
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- Method: **count** FLOPs and bytes, **predict** with
  $\max(h,\ \textrm{FLOPs}/P,\ \textrm{bytes}/\beta)$, **measure** with
  `Benchmark` and a profiler, **explain** the difference. A failed
  prediction points at what we did not count.
- Roofline: performance $\leq \min(P, I\beta)$; ridge 164 FLOP/byte in
  bf16, 82 in tf32.
- One $16{,}384^2$ weight: flat at about 0.6 ms up to $B = 128$, then
  $\propto B$ at the compute roof.
- Elementwise kernels run at about 0.9 TB/s; an unfused chain of 8 moves
  9 times the bytes of one.
- Small calls cost $h$: a $256 \times 256$ product reaches under 2% of the
  peak.
- The MLP step: predicted 2.5 ms, measured within a tenth.

**Next:** where $P$ and $\beta$ come from: the hardware.
:::
