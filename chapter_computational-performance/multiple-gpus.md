# Multi-GPU from First Principles
:label:`sec_multi_gpu`

Every training run so far has used one GPU, even though our build workstation
holds four. The simplest way to use all $k$ GPUs is *data parallelism*: give
every GPU a full replica of the model and $1/k$ of each minibatch, sum the
resulting gradients across GPUs with an *allreduce*, and apply the same
parameter update on every card.

On our workstation, inter-GPU communication travels over PCIe 4.0 staged
through host DRAM (:numref:`subsec_hw-interconnects`), capping each link at a
nominal 32 GB/s—roughly $30\times$ slower than a card's 1 TB/s on-board
memory. A second GPU can shorten a step only if the computation it takes over
lasts longer than the gradient exchange (and host dispatch overhead) it adds.
In this section we build data parallelism by hand, derive the *ring allreduce*
algorithm used by collective communication libraries, and construct a cost
model that predicts multi-GPU step times from individually measured constants.
We test our model on LeNet on Fashion-MNIST, whose tiny steps expose fixed
host and communication overheads starkly.

```{.python .input #multiple-gpus-multi-gpu-from-first-principles}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #multiple-gpus-multi-gpu-from-first-principles}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #multiple-gpus-multi-gpu-from-first-principles-1}
%%tab pytorch
import os
import time
# Detach PyTorch's profiler from the GPU when a profile ends; without this,
# every kernel launch after the first profile stays about 30% slower
os.environ['TEARDOWN_CUPTI'] = '1'
```

```{.python .input #multiple-gpus-multi-gpu-from-first-principles-1}
%%tab jax
import time
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.sharding import NamedSharding, PartitionSpec as P
```

```{.python .input #multiple-gpus-multi-gpu-from-first-principles-2}
%%tab pytorch
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True  # Time convolution algorithms per shape
```

## Three Ways to Split
:label:`subsec_mg-splitting`

A model and its minibatch can be partitioned across devices in three basic ways
(:numref:`fig_splitting`):

* **Data parallelism** replicates the model on every device, splits the
  minibatch across devices, and sums gradients once per step. It works for any
  model that fits on a single GPU, though every device still holds all $16N$
  bytes of fp32 Adam training state (:numref:`sec_memory_precision`).
* **Pipeline parallelism** splits the layer stack across devices, passing
  activations forward and gradients backward at stage boundaries
  (Exercise :exref:`ex_mg-pipeline`).
* **Tensor parallelism** splits individual weight matrices across devices
  (like AlexNet across two 3 GB GPUs in 2012
  :cite:`Krizhevsky.Sutskever.Hinton.2012`), communicating several times per
  layer (Exercise :exref:`ex_mgp-tensor` in :numref:`sec_sharding`).

![Three ways to split a model across two devices, with the cells of each
device in its own color: replicate the model and split the batch (data
parallel); split the layer stack (pipeline parallel); split every layer's
width (tensor parallel).](../img/mdl-perf-splitting.svg)
:label:`fig_splitting`

## Data Parallelism by Hand
:label:`subsec_mg-byhand`

One data-parallel step on $k$ devices has four stages
(:numref:`fig_data_parallel`): split the minibatch into $k$ equal shards;
compute each shard's gradient on its own device; sum the $k$ gradients via an
allreduce so every device holds the sum; and apply the identical parameter
update on every device. Because all replicas start from identical parameters
and apply identical updates, they remain synchronized without ever re-copying
weights.

![One data-parallel step on two GPUs. Split the minibatch, run identical
forward and backward passes on each replica, sum the gradients with an
allreduce, then apply the identical update
everywhere.](../img/mdl-perf-data-parallel.svg)
:label:`fig_data_parallel`

### The Model and the Data

We write LeNet (:numref:`sec_lenet`, with ReLU activations and He
initialization) as a function of an explicit list of parameter tensors so
every cross-device copy is visible. Its 61,706 fp32 parameters occupy 247 KB,
and the 60,000 Fashion-MNIST training images occupy 188 MB—small enough to
keep a copy of the dataset on each GPU so host data loading never enters our
timings.

```{.python .input #multiple-gpus-data-parallelism-by-hand-1}
%%tab pytorch
def new_params():
    """LeNet's parameters on the CPU: He-initialized weights, zero biases."""
    params = []
    for shape in [(6, 1, 5, 5), (16, 6, 5, 5), (120, 400), (84, 120),
                  (10, 84)]:
        fan_in = math.prod(shape[1:])
        params += [torch.randn(shape) * math.sqrt(2 / fan_in),
                   torch.zeros(shape[0])]
    return params

def lenet(params, X):
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = params
    H = F.avg_pool2d(F.relu(F.conv2d(X, W1, b1, padding=2)), 2)
    H = F.avg_pool2d(F.relu(F.conv2d(H, W2, b2)), 2).flatten(1)
    H = F.relu(F.linear(H, W3, b3))
    H = F.relu(F.linear(H, W4, b4))
    return F.linear(H, W5, b5)

def to_device(split, device):
    """A whole Fashion-MNIST split on `device`: images in [0, 1], labels."""
    return split.data.to(device).unsqueeze(1) / 255, split.targets.to(device)
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-1}
%%tab jax
def new_params(key):
    """LeNet's parameters: He-initialized weights, zero biases."""
    shapes = [(5, 5, 1, 6), (5, 5, 6, 16), (400, 120), (120, 84), (84, 10)]
    params = []
    for key, shape in zip(jax.random.split(key, len(shapes)), shapes):
        fan_in = math.prod(shape[:-1])
        params += [jax.random.normal(key, shape) * math.sqrt(2 / fan_in),
                   jnp.zeros(shape[-1])]
    return params

def lenet(params, X):
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = params
    conv = lambda X, W, padding: jax.lax.conv_general_dilated(
        X, W, (1, 1), padding, dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    pool = lambda X: nnx.avg_pool(X, (2, 2), strides=(2, 2))
    H = pool(jax.nn.relu(conv(X, W1, ((2, 2), (2, 2))) + b1))
    H = pool(jax.nn.relu(conv(H, W2, 'VALID') + b2)).reshape(X.shape[0], -1)
    H = jax.nn.relu(H @ W3 + b3)
    H = jax.nn.relu(H @ W4 + b4)
    return H @ W5 + b5

def to_device(split, sharding):
    """A whole Fashion-MNIST split placed by `sharding`: images in [0, 1],
    labels."""
    X, y = split
    return jax.device_put((X[..., None] / np.float32(255),
                           y.astype(np.int32)), sharding)
```

```{.python .input #multiple-gpus-the-model-and-the-data-1}
%%tab pytorch
assert d2l.num_gpus() >= 2, 'This section needs two GPUs'
data = d2l.FashionMNIST()
devices = [d2l.try_gpu(i) for i in range(2)]
train_data = [to_device(data.train, d) for d in devices]  # A copy each
test_data = to_device(data.val, devices[0])
base = new_params()          # One initialization for every run below
print(f'{sum(p.numel() for p in base)} parameters in {len(base)} tensors; '
      f'images {tuple(train_data[0][0].shape)} on each of 2 devices')
```

```{.python .input #multiple-gpus-the-model-and-the-data-1}
%%tab jax
assert jax.device_count() >= 2, 'This section needs two GPUs'
data = d2l.FashionMNIST()
meshes = [jax.make_mesh((k,), ('data',), devices=jax.devices()[:k])
          for k in (1, 2)]
train_data = [to_device(data.train, NamedSharding(mesh, P()))
              for mesh in meshes]  # A copy on every device of each mesh
test_data = (data.val[0][..., None] / np.float32(255),
             data.val[1].astype(np.int32))
base = new_params(d2l.get_key())  # One initialization for every run
print(f'{sum(p.size for p in base)} parameters in {len(base)} tensors; '
      f'images {train_data[1][0].shape} on each of 2 devices')
```

:begin_tab:`jax`
In JAX, multi-device arrays are placed on a named device *mesh* via
`NamedSharding` and `PartitionSpec` (`P('data')` shards the leading axis
across the `'data'` mesh axis; `P()` replicates the full array on every
device).
:end_tab:

### Broadcast and Allreduce

*Broadcast* copies initial parameters to every device once; *allreduce* sums
one tensor per device and leaves the total on every device. In PyTorch we
write a simple *star* allreduce (device 0 gathers and sums all tensors, then
copies the sum back); in JAX we use `jax.shard_map` with `jax.lax.psum`
(which delegates to NCCL).

```{.python .input #multiple-gpus-data-parallelism-by-hand-2}
%%tab pytorch
def broadcast(params, device):
    """A copy of every tensor in `params` on `device`, ready for autograd."""
    return [p.to(device, copy=True).requires_grad_() for p in params]

def allreduce(data):
    """Sum the tensors in `data`, one per device, onto every device."""
    for i in range(1, len(data)):    # Star: collect and add on device 0 ...
        data[0] += data[i].to(data[0].device)
    for i in range(1, len(data)):    # ... then copy the sum back
        data[i].copy_(data[0])
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-2}
%%tab jax
def broadcast(params, mesh):
    """A copy of every array in `params` on every device of `mesh`."""
    return jax.device_put(params, NamedSharding(mesh, P()))

@partial(jax.jit, static_argnames='mesh')
def allreduce(x, mesh):
    """Sum the rows of `x` (row j on device j); every device gets the sum."""
    return jax.shard_map(lambda x_j: jax.lax.psum(x_j, 'data'), mesh=mesh,
                         in_specs=P('data'), out_specs=P('data'))(x)
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-3}
%%tab pytorch
x = [torch.full((1, 2), i + 1.0, device=d) for i, d in enumerate(devices)]
print('before:', [x_j.tolist() for x_j in x])
allreduce(x)
print('after: ', [x_j.tolist() for x_j in x])
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-3}
%%tab jax
x = jax.device_put(jnp.array([[1.0, 1.0], [2.0, 2.0]]),
                   NamedSharding(meshes[1], P('data')))
print('before:', x.tolist())
print('after: ', allreduce(x, meshes[1]).tolist())
```

### The Data-Parallel Step

With $k$ equal-sized shards, the average of the $k$ per-shard mean-loss
gradients equals the full-minibatch gradient (Exercise
:exref:`ex_mg-normalization`). Because every allreduce call incurs a fixed
latency regardless of message size, we concatenate LeNet's 10 gradient tensors
into a single flat vector and perform one allreduce per step. Passing
`communicate=False` skips the allreduce so we can time pure device computation.

```{.python .input #multiple-gpus-data-parallelism-by-hand-4}
%%tab pytorch
def shard_grads(params, data, idx):
    """Gradient of the mean loss on each device's share of the minibatch
    `idx` (example indices), as one flat vector per device."""
    grads = []
    for ps, (X, y), i in zip(params, data, idx.chunk(len(params))):
        i = i.to(X.device, non_blocking=True)  # Don't wait for the device
        F.cross_entropy(lenet(ps, X[i]), y[i]).backward()
        grads.append(torch.cat([p.grad.flatten() for p in ps]))
        for p in ps:
            p.grad = None              # The next backward starts from zero
    return grads

def train_step(params, data, idx, lr, communicate=True):
    """One data-parallel SGD step on the minibatch `idx`; without
    communication, every device steps on its own shard's gradient."""
    grads = shard_grads(params, data, idx)
    if communicate:
        allreduce(grads)               # One message per device and step
    with torch.no_grad():
        for ps, g in zip(params, grads):
            for p, g_p in zip(ps, g.split([p.numel() for p in ps])):
                p -= lr / len(params) * g_p.view_as(p)  # Mean of k grads
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-4}
%%tab jax
def shard_grad(params, X, y, i):
    """Inside shard_map: the gradient of the mean loss on this device's
    share `i` of the minibatch, as one flat vector."""
    def loss(p):
        return optax.softmax_cross_entropy_with_integer_labels(
            lenet(p, X[i]), y[i]).mean()
    own = jax.lax.pcast(params, 'data', to='varying')  # This device's copy
    return ravel_pytree(jax.grad(loss)(own))[0]

@partial(jax.jit, static_argnames=('mesh', 'communicate'))
def train_step(params, data, idx, lr, mesh, communicate=True):
    """One data-parallel SGD step on the minibatch `idx`; without
    communication, every device steps on its own shard's gradient."""
    unravel = ravel_pytree(params)[1]
    def step(params, X, y, i):
        g = shard_grad(params, X, y, i)
        if communicate:
            g = jax.lax.psum(g, 'data')              # Allreduce
        k = jax.lax.axis_size('data')
        return jax.tree.map(lambda p, g_p: p - lr / k * g_p, params,
                            unravel(g))
    out = P() if communicate else P('data')  # Copies differ without the sum
    return jax.shard_map(step, mesh=mesh, in_specs=(P(), P(), P(), P('data')),
                         out_specs=out)(params, *data, idx)
```

### A One-Step Check

Before timing anything, we verify that one data-parallel step on two GPUs
produces the same parameter update as one step on a single GPU over the same
256 examples (using strict fp32 arithmetic so convolution algorithm differences
under tf32 do not mask bugs such as forgetting to divide by $k$).

```{.python .input #multiple-gpus-data-parallelism-by-hand-7}
%%tab pytorch
idx = torch.randperm(len(data.train))[:256]
updates = []
torch.set_float32_matmul_precision('highest')  # True fp32 in matmuls ...
with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):  # ... convs
    for k in (1, 2):
        params = [broadcast(base, d) for d in devices[:k]]
        train_step(params, train_data[:k], idx, lr=0.2)
        updates.append([p.detach().cpu() - b
                        for p, b in zip(params[0], base)])
torch.set_float32_matmul_precision('high')
largest = max(u.abs().max() for u in updates[0])
diff = max((u1 - u2).abs().max() for u1, u2 in zip(*updates))
print(f'largest update {largest:.1e}, largest difference {diff:.1e}')
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-7}
%%tab jax
idx = np.random.default_rng().permutation(len(data.train[1]))[:256]
updates = []
with jax.default_matmul_precision('highest'):  # True fp32
    for mesh, data_k in zip(meshes, train_data):
        i = jax.device_put(idx, NamedSharding(mesh, P('data')))
        params = train_step(broadcast(base, mesh), data_k, i, 0.2, mesh)
        updates.append([np.asarray(p) - np.asarray(b)
                        for p, b in zip(params, base)])
largest = max(np.abs(u).max() for u in updates[0])
diff = max(np.abs(u1 - u2).max() for u1, u2 in zip(*updates))
print(f'largest update {largest:.1e}, largest difference {diff:.1e}')
```

The single-GPU and two-GPU updates agree to fp32 rounding error (~$10^{-7}$).

## Ring Allreduce
:label:`subsec_mg-ring`

A star allreduce routes all traffic through device 0, forcing the hub to send
and receive $(k-1)S$ bytes for a gradient of size $S$. *Ring allreduce*
:cite:`Patarasuk.Yuan.2009,Sergeev.Del-Balso.2018` distributes traffic evenly
across all $k$ devices arranged in a logical ring, splitting each device's
$S$-byte gradient into $k$ chunks of $S/k$ bytes and running two phases of
$k - 1$ steps each (:numref:`fig_ring_allreduce`):

1. **Reduce-scatter:** In each of $k - 1$ steps, every device sends one chunk
   to its successor and adds the chunk received from its predecessor into its
   local copy. After $k - 1$ steps, each device holds the complete sum for one
   distinct chunk.
2. **All-gather:** Over another $k - 1$ steps, the finished chunks circulate
   around the ring until every device holds all $k$ summed chunks.

Each device sends $(k-1)S/k$ bytes per phase, or in total

$$
\frac{2(k-1)}{k}\, S \quad\textrm{bytes per device.}
$$
:eqlabel:`eq_ring_traffic`

Per-device traffic is $S$ at $k = 2$ and stays strictly below $2S$ no matter
how many devices join the ring. Moreover, the two phases are useful
independently: sharded training (:numref:`sec_sharding`) uses reduce-scatter
to partition summed gradients and all-gather to assemble sharded weights.

![Ring allreduce on four GPUs, computed step by step. Reduce-scatter
accumulates one chunk per device around the ring; all-gather then
distributes the finished chunks. Each cell counts how many of the four
contributions a chunk holds, so a chunk holding 4 is complete. An
outlined cell is the chunk that a GPU received in that step, and the ring
in the last panel shows which GPU sends to which.](../img/mdl-perf-ring-allreduce.svg)
:label:`fig_ring_allreduce`

We verify the ring indexing and byte count in a NumPy simulation across
$k \in \{2, 4, 8, 16\}$:

```{.python .input #multiple-gpus-ring-allreduce-1}
%%tab pytorch, jax
def ring_allreduce(xs):
    """Ring allreduce of the vectors xs, one per device; returns every
    device's result and the bytes that each device sent."""
    k = len(xs)
    chunks = [np.array_split(x.copy(), k) for x in xs]  # chunks[j][c]
    sent = 0
    for s in range(k - 1):  # Reduce-scatter: add the received chunk
        msgs = [chunks[j][(j - s) % k].copy() for j in range(k)]
        for j, msg in enumerate(msgs):
            chunks[(j + 1) % k][(j - s) % k] += msg
        sent += msgs[0].nbytes
    for s in range(k - 1):  # All-gather: replace by the received chunk
        msgs = [chunks[j][(j + 1 - s) % k].copy() for j in range(k)]
        for j, msg in enumerate(msgs):
            chunks[(j + 1) % k][(j + 1 - s) % k] = msg
        sent += msgs[0].nbytes
    return [np.concatenate(c) for c in chunks], sent
```

```{.python .input #multiple-gpus-ring-allreduce-2}
%%tab pytorch, jax
rng = np.random.default_rng()
for k in (2, 4, 8, 16):
    xs = [rng.standard_normal(1024, dtype=np.float32) for _ in range(k)]
    results, sent = ring_allreduce(xs)
    correct = all(np.allclose(r, sum(xs), atol=1e-5) for r in results)
    print(f'k = {k:2d}: every device holds the sum: {correct}; '
          f'sent per device: {sent / xs[0].nbytes:.2f} S')
```

While per-device bytes stay bounded by $2S$, a ring executes $2(k-1)$
sequential hops, each paying per-hop start-up latency. For small messages on
many GPUs, collective libraries switch to $\log_2 k$-depth tree algorithms
(Exercise :exref:`ex_mg-latency`).

## Communication Cost and Scaling
:label:`subsec_mg-accounting`

### A Cost Model for One Step

Using the latency–bandwidth model :cite:`Hockney.1994,Thakur.Rabenseifner.Gropp.2005`,
each allreduce call pays a fixed start-up latency $\alpha$ plus a time
$\beta_{\textrm{link}}$ per byte sent by each device (the inverse of the
allreduce *bus bandwidth*). Without overlap, a step on $k$ devices with global
batch $B$ and $c$ allreduce calls takes

$$
t_{\textrm{step}}(k) \approx t_{\textrm{compute}}(B/k) + c\,\alpha +
\frac{2(k-1)}{k}\,S\,\beta_{\textrm{link}},
$$
:eqlabel:`eq_dp_cost`

where $c = 0$ at $k = 1$. A second device shortens the step only if
$t_{\textrm{compute}}(B) - t_{\textrm{compute}}(B/2) > c\alpha + S\,\beta_{\textrm{link}}$.

### Latency and Bandwidth of the Allreduce

We measure $\alpha$ with a 1-float allreduce and $\beta_{\textrm{link}}$ with
a $2^{26}$-float (268 MB) allreduce on two GPUs:

```{.python .input #multiple-gpus-the-accounting}
%%tab pytorch
def allreduce_time(n, devices, calls):
    """Seconds per allreduce of n floats per device."""
    x = [torch.zeros(n, device=d) for d in devices]
    return d2l.Benchmark(lambda: allreduce(x), warmup=10 * calls,
                         repeats=3 * calls, devices=devices).time

alpha = allreduce_time(1, devices, calls=100)  # Our star: two copies in turn
n = 2**26                                # 64M floats ...
S = 4 * n                                # ... 268 MB (256 MiB) per device
beta_link = (allreduce_time(n, devices, calls=3) - alpha) / S  # s per byte
print(f'alpha = {1e6 * alpha:.0f} us, beta_link = {1e9 * beta_link:.2f} '
      f'ns/byte (bus bandwidth {1e-9 / beta_link:.1f} GB/s)')
```

```{.python .input #multiple-gpus-the-accounting}
%%tab jax
def allreduce_time(n, mesh, calls):
    """Seconds per allreduce of n floats per device."""
    x = jnp.zeros((mesh.size, n), device=NamedSharding(mesh, P('data')))
    return d2l.Benchmark(lambda: allreduce(x, mesh), repeats=calls).time

def program_time(m, mesh, calls=300):
    """Seconds per call of one compiled program that makes m allreduces of
    one float, each on the result of the one before."""
    def chain(x):
        for _ in range(m):               # Each allreduce waits for the last
            x = jax.lax.pmean(x, 'data')
        return x
    f = jax.jit(jax.shard_map(chain, mesh=mesh, in_specs=P('data'),
                              out_specs=P('data')))
    x = jnp.zeros((mesh.size, 1), device=NamedSharding(mesh, P('data')))
    return d2l.Benchmark(lambda: f(x), warmup=10, repeats=calls).time

# The training step's psum runs inside a program that is dispatched anyway,
# so alpha is what each further allreduce adds to one program
alpha = (program_time(10, meshes[1]) - program_time(1, meshes[1])) / 9
n = 2**26                                # 64M floats ...
S = 4 * n                                # ... 268 MB (256 MiB) per device
# Both calls dispatch a program of their own; their difference is the bytes
beta_link = (allreduce_time(n, meshes[1], calls=3)
             - allreduce_time(1, meshes[1], calls=100)) / S  # s per byte
print(f'alpha = {1e6 * alpha:.0f} us, beta_link = {1e9 * beta_link:.2f} '
      f'ns/byte (bus bandwidth {1e-9 / beta_link:.1f} GB/s)')
```

To test whether :eqref:`eq_dp_cost` predicts the benefit of gradient fusion, we
allreduce LeNet's 10 gradient tensors ($S = 247$ KB total) separately
($c = 10$) versus concatenated into a single flat vector ($c = 1$), which
should differ by $9\alpha$:

```{.python .input #multiple-gpus-latency-and-bandwidth-of-the-allreduce-1}
%%tab pytorch
grads = [[torch.zeros_like(p, device=d) for p in base] for d in devices]
flat = [torch.cat([g.flatten() for g in G]) for G in grads]
def separately():
    for g in zip(*grads):              # One allreduce per tensor
        allreduce(list(g))
n_bytes, c = flat[0].nbytes, len(base)
for name, f, calls in (('separately', separately, c),
                       ('at once', lambda: allreduce(flat), 1)):
    t = d2l.Benchmark(f, warmup=1000, repeats=300, devices=devices).time
    print(f'{name:10s}: {1e6 * t:4.0f} us measured, '
          f'{1e6 * (calls * alpha + n_bytes * beta_link):4.0f} us predicted')
```

```{.python .input #multiple-gpus-latency-and-bandwidth-of-the-allreduce-1}
%%tab jax
mesh = meshes[1]
sizes = [p.size for p in base]
flat = jnp.zeros((mesh.size, sum(sizes)), device=NamedSharding(mesh, P('data')))
n_bytes, c = flat.nbytes // mesh.size, len(base)  # Bytes per device

def separately(g):
    """One psum per gradient tensor, each after the one before; without the
    barrier, XLA would merge the ten into one allreduce."""
    out = []
    for g_i in jnp.split(g, np.cumsum(sizes)[:-1], axis=1):
        if out:
            g_i, _ = jax.lax.optimization_barrier((g_i, out[-1]))
        out.append(jax.lax.psum(g_i, 'data'))
    return jnp.concatenate(out, axis=1)

t = {}
for name, f in (('separately', separately),
                ('at once', lambda g: jax.lax.psum(g, 'data'))):
    f = jax.jit(jax.shard_map(f, mesh=mesh, in_specs=P('data'),
                              out_specs=P('data')))
    t[name] = d2l.Benchmark(lambda: f(flat), warmup=10, repeats=300).time
    print(f'{name:10s}: {1e6 * t[name]:4.0f} us measured')
# Both programs are dispatched once and move the same bytes
print(f'difference: {1e6 * (t["separately"] - t["at once"]):4.0f} us '
      f'measured, {1e6 * (c - 1) * alpha:4.0f} us predicted by {c - 1} alpha')
```

For a 247 KB gradient, nine extra message envelopes ($9\alpha$) cost more than
all the bytes inside them—which is why production data-parallel libraries pack
small gradients into large buckets before communicating
(:numref:`sec_multi_gpu_concise`).

### LeNet on One and Two GPUs

Before running LeNet on two GPUs, let us profile its local work on one GPU for
a minibatch of 256 examples:

```{.python .input #multiple-gpus-lenet-on-one-and-two-gpus-2}
%%tab pytorch
params = [broadcast(base, devices[0])]
step = lambda: train_step(params, train_data[:1], idx, 0.2, communicate=False)
t = d2l.Benchmark(step, warmup=500, repeats=150, devices=devices[:1]).time
d2l.show_profile(d2l.kernel_times(step, steps=100), 1e3 * t, rows=4)
```

```{.python .input #multiple-gpus-lenet-on-one-and-two-gpus-2}
%%tab jax
mesh, params = meshes[0], broadcast(base, meshes[0])
i = jax.device_put(idx, NamedSharding(mesh, P('data')))
step = lambda: train_step(params, train_data[0], i, 0.2, mesh,
                          communicate=False)
t = d2l.Benchmark(step, repeats=500).time
d2l.show_profile(d2l.kernel_times(step, steps=100), 1e3 * t, rows=4)
```

The GPU is idle for most of the step: LeNet at batch size 256 is
overhead-bound, so its step time equals the host dispatch time
$h_{\textrm{step}}$. Consequently, halving the batch from 256 to 128 on one
GPU barely changes the step time, whereas driving two GPUs from one host thread
takes roughly $2 h_{\textrm{step}}$ (:numref:`fig_one_launcher`):

```{.python .input #multiple-gpus-lenet-on-one-and-two-gpus-1}
%%tab pytorch
params, t_local = [broadcast(base, d) for d in devices], {}
for k, b in ((1, 256), (1, 128), (2, 128)):
    t_local[k, b] = d2l.Benchmark(
        lambda: train_step(params[:k], train_data[:k], idx[:k * b], 0.2,
                           communicate=False),
        warmup=500, repeats=150, devices=devices[:k]).time
    print(f'{k} GPU(s), {b} examples each: {1e3 * t_local[k, b]:.2f} ms')
```

```{.python .input #multiple-gpus-lenet-on-one-and-two-gpus-1}
%%tab jax
t_local = {}
for k, b in ((1, 256), (1, 128), (2, 128)):
    mesh, data_k = meshes[k - 1], train_data[k - 1]
    params = broadcast(base, mesh)
    i = jax.device_put(idx[:k * b], NamedSharding(mesh, P('data')))
    t_local[k, b] = d2l.Benchmark(lambda: train_step(
        params, data_k, i, 0.2, mesh, communicate=False), repeats=500).time
    print(f'{k} GPU(s), {b} examples each: {1e3 * t_local[k, b]:.2f} ms')
```

Incorporating the single-host dispatch floor $k\,h_{\textrm{step}}$ alongside
the device compute time $t_{\textrm{dev}}(B/k)$ refines :eqref:`eq_dp_cost` to

$$
t_{\textrm{step}}(k) \approx \max\bigl(k\,h_{\textrm{step}},\;
t_{\textrm{dev}}(B/k)\bigr) + c\,\alpha +
\frac{2(k-1)}{k}\,S\,\beta_{\textrm{link}}.
$$
:eqlabel:`eq_dp_one-launcher`

![A host-bound step on two GPUs, drawn as one host that issues the work of
GPU 0 and then that of GPU 1. Each GPU idles while the host serves the
other, so a step whose kernels are short takes about $2h_{\textrm{step}}$
of host time plus the exchange.](../img/mdl-perf-one-launcher.svg)
:label:`fig_one_launcher`

We test both predictions over a 5-epoch training run at global batch size 256
on 1 and 2 GPUs:

```{.python .input #multiple-gpus-data-parallelism-by-hand-5}
%%tab pytorch
def train(params, train_data, test_data, orders, batch_size=256, lr=0.2):
    """Data-parallel SGD on the devices that hold `params` and `train_data`,
    one epoch per example order; returns seconds per step, test accuracy."""
    devices = [X.device for X, _ in train_data]
    times = []
    for order in orders:
        t0 = time.perf_counter()
        for idx in order.split(batch_size):
            train_step(params, train_data, idx, lr)
        for d in devices:  # Wait until every device has finished
            torch.cuda.synchronize(d)
        times.append((time.perf_counter() - t0)
                     / math.ceil(len(order) / batch_size))
    X, y = test_data
    with torch.no_grad():
        acc = (lenet(params[0], X).argmax(1) == y).float().mean().item()
    return sum(times[1:]) / (len(times) - 1), acc
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-5}
%%tab jax
def train(params, train_data, test_data, mesh, orders, batch_size=256,
          lr=0.2):
    """Data-parallel SGD on the devices of `mesh`, one epoch per example
    order; returns seconds per step, test accuracy."""
    times = []
    for order in orders:
        starts = range(0, len(order), batch_size)
        batches = jax.device_put(  # The epoch's minibatches, in one call
            [order[s:s + batch_size] for s in starts],
            NamedSharding(mesh, P('data')))
        t0 = time.perf_counter()
        for idx in batches:
            params = train_step(params, train_data, idx, lr, mesh)
        jax.block_until_ready(params)
        times.append((time.perf_counter() - t0) / len(batches))
    X, y = test_data
    acc = float((jax.jit(lenet)(params, X).argmax(1) == y).mean())
    return sum(times[1:]) / (len(times) - 1), acc
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-6}
%%tab pytorch
orders, t_step = [torch.randperm(len(data.train)) for _ in range(5)], {}
for k in (1, 2):
    params = [broadcast(base, d) for d in devices[:k]]
    t_step[k], acc = train(params, train_data[:k], test_data, orders)
    print(f'{k} GPU(s): {1e3 * t_step[k]:.2f} ms per step, '
          f'test accuracy {acc:.2f}')
exchange = alpha + n_bytes * beta_link    # One allreduce of the gradient
print(f'2 GPUs predicted: {1e3 * (t_local[1, 128] + exchange):.2f} ms per '
      f'step by eq_dp_cost, {1e3 * (2 * t_step[1] + exchange):.2f} ms with '
      f'one launcher')
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-6}
%%tab jax
rng = np.random.default_rng()
orders, t_step = [rng.permutation(len(data.train[1])) for _ in range(5)], {}
for k, mesh, data_k in zip((1, 2), meshes, train_data):
    t_step[k], acc = train(broadcast(base, mesh), data_k, test_data, mesh,
                           orders)
    print(f'{k} GPU(s): {1e3 * t_step[k]:.2f} ms per step, '
          f'test accuracy {acc:.2f}')
exchange = alpha + n_bytes * beta_link    # One allreduce of the gradient
print(f'2 GPUs predicted: {1e3 * (t_local[1, 128] + exchange):.2f} ms per '
      f'step by eq_dp_cost, {1e3 * (2 * t_step[1] + exchange):.2f} ms with '
      f'one launcher')
```

Both runs reach the same test accuracy, and :eqref:`eq_dp_one-launcher`
accurately predicts the two-GPU slowdown: at batch 256, LeNet's gradient
exchange is tiny (~0.1 ms), and the bottleneck is feeding two GPUs from a
single host launcher.

### Strong and Weak Scaling, and Larger Minibatches

Multi-GPU benchmarks follow one of two conventions: *strong scaling* holds the
global batch $B$ fixed as $k$ grows (aiming to finish the same step faster),
whereas *weak scaling* holds the per-GPU batch $b$ fixed so the global batch
grows as $kb$ (aiming for $k\times$ throughput in examples/s, though larger
global batches alter optimization dynamics, :numref:`sec_batch_size`).

By :eqref:`eq_dp_one-launcher`, a second GPU improves weak-scaling throughput
($2\,t(1)/t(2) > 1$) when

$$
\max\bigl(2h_{\textrm{step}},\; t_{\textrm{dev}}(b)\bigr) + \alpha +
S\,\beta_{\textrm{link}} \;<\;
2\max\bigl(h_{\textrm{step}},\; t_{\textrm{dev}}(b)\bigr).
$$

A second GPU never helps when $t_{\textrm{dev}}(b) \le h_{\textrm{step}}$; once
$t_{\textrm{dev}}(b)$ outgrows $2h_{\textrm{step}}$ and the exchange time, the
speedup approaches 2. We test this across $b \in \{256, 4096, 32768\}$
examples per GPU:

```{.python .input #multiple-gpus-larger-minibatches}
%%tab pytorch
for b in (256, 4096, 32768):                # Examples per GPU
    batch = torch.randint(len(data.train), (2 * b,))  # With replacement
    steps = [lambda k=k: train_step(params[:k], train_data[:k],
                                    batch[:k * b], 0.2) for k in (1, 2)]
    times = [[], []]                        # Three timed windows per k,
    for w in range(3):                      # taken in turn for k = 1 and 2
        for k in (1, 2):
            params = [broadcast(base, d) for d in devices]  # Equal replicas
            times[k - 1].append(d2l.Benchmark(
                steps[k - 1], warmup=10 if w == 0 else 0,
                repeats=max(20, 2**18 // b), devices=devices[:k]).time)
    t = np.median(times, axis=1)
    t_dev = sum(d2l.kernel_times(steps[0]).values()) / 1e3
    if b == 256:              # The host sets this step's time (profile above)
        h_step = t[0]
    predicted = 2 * t[0] / (max(2 * h_step, t_dev) + exchange)
    speedups = 2 * np.array(times[0]) / np.array(times[1])
    print(f'b = {b:5d}: 1 GPU {1e3 * t[0]:6.2f} ms (device busy '
          f'{1e3 * t_dev:5.2f} ms), 2 GPUs {1e3 * t[1]:6.2f} ms\n'
          f'           speedup {predicted:.2f} predicted, '
          f'{2 * t[0] / t[1]:.2f} measured '
          f'(windows {speedups.min():.2f} to {speedups.max():.2f})')
```

```{.python .input #multiple-gpus-larger-minibatches}
%%tab jax
rng = np.random.default_rng()
for b in (256, 4096, 32768):                # Examples per GPU
    batch = rng.integers(len(data.train[1]), size=2 * b)  # With replacement
    steps = []
    for k, mesh, data_k in zip((1, 2), meshes, train_data):
        params = broadcast(base, mesh)
        i = jax.device_put(batch[:k * b], NamedSharding(mesh, P('data')))
        steps.append(partial(train_step, params, data_k, i, 0.2, mesh))
    times = [[], []]                        # Three timed windows per k,
    for w in range(3):                      # taken in turn for k = 1 and 2
        for j, step in enumerate(steps):
            times[j].append(d2l.Benchmark(
                step, warmup=10 if w == 0 else 0,
                repeats=max(20, 2**18 // b)).time)
    t = np.median(times, axis=1)
    t_dev = sum(d2l.kernel_times(steps[0]).values()) / 1e3
    if b == 256:              # The host sets this step's time (profile above)
        h_step = t[0]
    predicted = 2 * t[0] / (max(2 * h_step, t_dev) + exchange)
    speedups = 2 * np.array(times[0]) / np.array(times[1])
    print(f'b = {b:5d}: 1 GPU {1e3 * t[0]:6.2f} ms (device busy '
          f'{1e3 * t_dev:5.2f} ms), 2 GPUs {1e3 * t[1]:6.2f} ms\n'
          f'           speedup {predicted:.2f} predicted, '
          f'{2 * t[0] / t[1]:.2f} measured '
          f'(windows {speedups.min():.2f} to {speedups.max():.2f})')
```

At $b = 256$ the second GPU gives no speedup; at $b = 32{,}768$ device work
far exceeds $2h_{\textrm{step}}$ and the 247 KB gradient exchange, nearly
doubling throughput as predicted.

## Summary

Data parallelism replicates the model on $k$ devices, computes gradients on
$1/k$ of the minibatch per device, and sums gradients via an allreduce before
updating. Ring allreduce decomposes the sum into a reduce-scatter followed by
an all-gather, sending $2(k-1)S/k < 2S$ bytes per device across $2(k-1)$ hops.
Our step cost model :eqref:`eq_dp_one-launcher` combines device compute time,
single-host dispatch overhead $k\,h_{\textrm{step}}$, per-call latency
$c\alpha$, and bandwidth cost $2(k-1)S\beta_{\textrm{link}}/k$, explaining why
concatenating small gradient tensors saves $9\alpha$ and why a second GPU only
speeds up training once per-GPU device compute outlasts both host dispatch and
communication.

## Exercises

1. **Data versus pipeline parallelism.** Consider a stack of $L$ bias-free
   linear layers of width $d$ in fp32 trained on two GPUs with batch size $B$.
   Data parallelism allreduces all weight gradients; two-stage pipeline
   parallelism (:numref:`fig_splitting`) places $L/2$ layers on each GPU and
   sends $B \times d$ activations forward and gradients backward across the
   cut.
    1. Count the bytes transferred per step in each scheme, evaluate both for
       $L = 32, d = 4096, B = 256$, and find the condition on $(B, L, d)$ for
       data parallelism to move fewer bytes.
    1. When the batch is split into $m$ pipeline micro-batches
       :cite:`Huang.Cheng.Bapna.ea.2019`, derive the bubble (idle) fraction of
       each GPU in a two-stage pipeline and evaluate it for $m = 1$ and
       $m = 8$.
   :exlabel:`ex_mg-pipeline`

2. **Gradient normalization.** Compare two data-parallel gradient recipes over
   $k$ shards of a batch of $B$ examples: Recipe A sums per-example losses on
   each device, sums gradients across devices, and divides by $B$; Recipe B
   averages per-example losses on each device and averages the $k$ device
   gradients.
    1. Show that both yield the exact full-batch mean gradient when all shards
       hold $B/k$ examples.
    1. For a language model averaging loss over tokens, suppose two shards hold
       3,000 and 1,000 tokens. Find the relative weight Recipe B assigns to a
       token in each shard, and modify Recipe B so it is exact for unequal
       token counts.
   :exlabel:`ex_mg-normalization`

3. **Strong-scaling speedup.** Model shard compute time as
   $t_{\textrm{compute}}(s) = t_0 + s\,t_1$ and neglect $\alpha$, so by
   :eqref:`eq_dp_cost` at fixed global batch $B$:
   $$t_{\textrm{step}}(k) = t_0 + \frac{B\,t_1}{k} + \frac{2(k-1)}{k}\,S\,\beta_{\textrm{link}}.$$
    1. Show that the strong-scaling speedup $R(k) = t_{\textrm{step}}(1) /
       t_{\textrm{step}}(k) > 1$ holds for one $k > 1$ if and only if it holds
       for all $k > 1$, and find $\lim_{k \to \infty} R(k)$.
    1. For $N = 11.2$M fp32 parameters, $t_0 = 2$ ms, $t_1 = 0.2$ ms/example,
       and $B = 256$, evaluate $R(2)$, $R(4)$, and $\lim_{k \to \infty} R(k)$
       at bus bandwidths of 20 GB/s and 5 GB/s, and find the minimum bus
       bandwidth for any speedup.
    1. Recompute $R(4)$ and $\lim_{k \to \infty} R(k)$ at 5 GB/s when gradients
       are communicated in bf16 ($S$ halved).

:begin_tab:`pytorch`
4. [code] **Ring allreduce by hand.** ●
    1. For $k$ devices holding $S$ bytes each, compare the total bytes moved
       and the bytes sent/received by the busiest device for star versus ring
       allreduce. Predict the ratio of their times when all $k$ bidirectional
       links operate concurrently versus when all transfers share one single
       bus sequentially.
    1. Implement `ring_allreduce(data)` in PyTorch using `chunk` and explicit
       `.to()` copies, verify it against `allreduce` for $k \in \{2, 3, 4\}$,
       and benchmark both on 64M floats per device at $k = 2$ and $k = 4$.
:end_tab:

:begin_tab:`jax`
4. [code] **Ring allreduce by hand.** ●
    1. For $k$ devices holding $S$ bytes each, compare the total bytes moved
       and the bytes sent/received by the busiest device for star versus ring
       allreduce. Predict the ratio of their times when all $k$ bidirectional
       links operate concurrently versus when all transfers share one single
       bus sequentially.
    1. Implement `ring_allreduce` inside `jax.shard_map` using
       `jax.lax.ppermute` and `jax.lax.axis_index`, verify it against
       `jax.lax.psum` for $k \in \{2, 4\}$, and benchmark it on 64M floats per
       device.
:end_tab:

5. **Latency in the allreduce.** ● If one hop of $n$ bytes takes
   $\alpha_{\textrm{hop}} + n\,\beta_{\textrm{link}}$, a $2(k-1)$-step ring
   allreduce takes
   $$t_{\textrm{ring}} = 2(k-1)\left(\alpha_{\textrm{hop}} + \frac{S}{k}\,\beta_{\textrm{link}}\right),$$
   whereas a binary-tree allreduce (sum up, broadcast down in $2\log_2 k$
   steps of $S$ bytes) takes
   $$t_{\textrm{tree}} = 2\log_2 k \left(\alpha_{\textrm{hop}} + S\,\beta_{\textrm{link}}\right).$$
    1. Compare how the latency and bandwidth terms of each algorithm scale with
       $k$. For $k = 8$, $\alpha_{\textrm{hop}} = 10$ µs, and bus bandwidth
       20 GB/s, find the crossover message size $S$ below which the tree is
       faster.
    1. At $k = 2$, express the half-performance message size $S_{1/2}$ (where
       fixed latency equals transfer time) in terms of $\alpha$ and
       $\beta_{\textrm{link}}$, evaluate it from your measurements in
       :numref:`subsec_mg-accounting`, and compare it with PyTorch DDP's 25 MB
       default bucket size.
   :exlabel:`ex_mg-latency`

[Discussions](https://d2l.discourse.group/t/36791)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.5]{.kicker}

Distributed Collectives and Data Parallelism from Scratch<br>**parallelism taxonomy · broadcast & allreduce · ring allreduce · step cost model · strong & weak scaling**
:::
:::

::: {.slide title="When Does a Second GPU Actually Shorten a Step?"}
[Motivation]{.kicker}

Our workstation contains four RTX 4090 GPUs, each reading its own GDDR6X memory at $1.0\text{ TB/s}$. Suppose we split every training minibatch across $k$ GPUs using **data parallelism**:

- **The compute dividend:** Each GPU processes only $1/k$ of the minibatch examples during the forward and backward passes.
- **The communication tax:** Before updating weights, all $k$ GPUs must sum their parameter gradients across the PCIe interconnect staged through host RAM ($\le 32\text{ GB/s}$ per direction—roughly **$30\times$ slower** than on-card memory).
- **The host-dispatch tax:** If a single Python thread dispatches kernels to all $k$ GPUs sequentially, CPU launch work scales as $k \cdot h_{\textrm{step}}$.

::: {.d2l-note .warn}
Adding a second GPU speeds up training **if and only if** the device compute time saved on half a minibatch exceeds both the inter-GPU gradient exchange and the added host dispatch overhead.
:::
:::

::: {.slide title="Three Ways to Partition Training Across Devices"}
[Parallelism Taxonomy]{.kicker}

![](../img/mdl-perf-splitting.svg){width=88%}

- **Data Parallelism (DP):** Replicates the full model ($16N$ bytes for `fp32` Adam) on every GPU, shards the minibatch along the batch axis, and exchanges gradients once per step via an **allreduce**.
- **Pipeline & Tensor Parallelism (PP / TP):** Partition the model's depth (layer stages) or width (individual weight matrices) across GPUs when a model is too large for one card, communicating activations at layer boundaries.
:::

::: {.slide title="Anatomy of a Data-Parallel Training Step"}
[Data Parallelism]{.kicker}

![](../img/mdl-perf-data-parallel.svg){width=58%}

- **Four stages per step:** (1) Split the minibatch of $B$ examples into $k$ disjoint shards of $B/k$ examples; (2) run forward and backward passes independently on each GPU; (3) sum the $k$ local gradients via an **allreduce**; (4) apply the identical optimizer step on every GPU.
- **Lockstep weight invariant:** Because all replicas start from the same broadcast parameters and apply identical gradient updates, their weights stay identical forever without re-broadcasting.
:::

::: {.slide title="Broadcast and Allreduce Primitives"}
[Collective Primitives]{.kicker}

Data parallelism relies on two collective communication operations across $k$ devices:

- **`broadcast(params, devices)`:** Copies initial model parameters from device $0$ to all $k$ devices once at the start of training.
- **`allreduce(grads)`:** Sums one tensor $x_j$ from each device $j \in \{0, \ldots, k-1\}$ and leaves the full sum $\sum_{j=0}^{k-1} x_j$ on **every** device.

@!multiple-gpus-data-parallelism-by-hand-3
:::

::: {.slide title="Exact Gradient Averaging Across k Equal Shards"}
[Mathematical Invariant]{.kicker}

Suppose a global minibatch $\mathcal{B}$ of $B$ examples is partitioned into $k$ equal shards $\mathcal{B}_1, \ldots, \mathcal{B}_k$ of size $B/k$. Device $j$ computes the mean-loss gradient $\mathbf{g}_j$ on its shard, and the allreduce sums $\mathbf{g}_1 + \cdots + \mathbf{g}_k$:

$$\mathbf{g}_j \;=\; \nabla_{\boldsymbol{\theta}} \left(\frac{k}{B} \sum_{i \in \mathcal{B}_j} \ell_i(\boldsymbol{\theta})\right) \quad \Longrightarrow \quad \frac{1}{k} \sum_{j=1}^{k} \mathbf{g}_j \;=\; \nabla_{\boldsymbol{\theta}} \left(\frac{1}{B} \sum_{i \in \mathcal{B}} \ell_i(\boldsymbol{\theta})\right)$$

- **Dividing by $k$ is mandatory:** Because the allreduce computes a **sum** of $k$ per-shard averages, dividing the summed gradient (or the learning rate $\eta$) by $k$ recovers the exact full-batch gradient.
- **Pitfall of omitting $1/k$:** Forgetting to divide by $k$ silently multiplies the effective learning rate by $k$ ($k\eta$)—a bug that may still converge to a worse optimum without crashing!
:::

::: {.slide title="The One-Step Gradient Check: 1 GPU vs. 2 GPUs"}
[Verification Discipline]{.kicker}

Before timing any multi-GPU loop, we run one SGD step on $B = 256$ Fashion-MNIST images on $1$ GPU and on $2$ GPUs (in strict `float32` without `tf32` rounding) from the exact same initial weights:

@!multiple-gpus-data-parallelism-by-hand-7

::: {.d2l-note .rule}
The single-GPU and two-GPU parameter updates agree to `float32` floating-point rounding ($\sim 10^{-7}$ vs. update magnitude $\sim 10^{-2}$), proving that our shard slicing, loss scaling, allreduce, and $1/k$ division are exact.
:::
:::

::: {.slide title="Why Star Allreduce Bottlenecks at the Hub Device"}
[Star vs. Ring Topology]{.kicker}

Suppose each of $k$ devices holds a gradient vector of $S$ bytes (e.g., $S = 4\text{ GB}$ for our 1B-parameter GPT in `float32`):

| Collective Topology | Bytes Sent by Busiest Device | Scaling as $k \to \infty$ | 1B GPT ($S = 4\text{ GB}$) at $k = 4$ |
| :--- | :---: | :---: | :---: |
| **Star Allreduce** (device 0 hub) | $(k - 1)\, S$ | Grows linearly: **$O(k S)$** | Hub sends **$12\text{ GB}$** & receives **$12\text{ GB}$** |
| **Ring Allreduce** (logical ring) | $\frac{2(k - 1)}{k}\, S$ | Bounded above by **$2S$** | Every device sends **$6\text{ GB}$** & receives **$6\text{ GB}$** |

- **Why the star fails at scale:** In a star topology, device $0$ must gather $(k-1)$ tensors of size $S$, add them, and scatter $(k-1)$ copies of the sum back out, leaving $(k-1)$ peer links idle while the hub link saturates.
:::

::: {.slide title="Ring Allreduce: Reduce-Scatter Followed by All-Gather"}
[Ring Allreduce]{.kicker}

![](../img/mdl-perf-ring-allreduce.svg){width=86%}

- **Two phases of $k - 1$ steps on chunks of size $S/k$:**
  1. **Reduce-scatter ($k-1$ hops):** Each GPU sends one $S/k$ chunk to its ring neighbor and adds the incoming chunk into its local buffer; after $k-1$ hops, each GPU holds the **complete sum for one unique $1/k$ shard**.
  2. **All-gather ($k-1$ hops):** The completed $S/k$ shards circulate around the ring without addition until every GPU holds all $k$ summed shards.
:::

::: {.slide title="Verifying Ring Traffic (< 2S) and Sequential Hop Count"}
[Ring Simulation]{.kicker}

Each of the $2(k-1)$ ring steps transmits one chunk of $S/k$ bytes per GPU, so total bytes sent per GPU is $\frac{2(k-1)}{k} S < 2S$:

@!multiple-gpus-ring-allreduce-2

- **Bandwidth vs. latency trade-off:** Ring allreduce is bandwidth-optimal for large tensors ($S$ independent of $k$ as $k$ grows), but executes $2(k-1)$ sequential hops (each paying start-up latency $\alpha_{\textrm{hop}}$). For small messages on many GPUs, NCCL switches to $2\log_2 k$-hop trees.
:::

::: {.slide title="The Latency-Bandwidth Cost Model for a Parallel Step"}
[Step Cost Model]{.kicker}

Using Hockney's latency–bandwidth model, each allreduce call pays a fixed start-up latency $\alpha$ plus a transfer time $\beta_{\textrm{link}}$ per byte sent per device (the inverse of the allreduce *bus bandwidth*):

$$t_{\textrm{step}}(k) \;\approx\; t_{\textrm{compute}}(B/k) \;+\; c\,\alpha \;+\; \frac{2(k-1)}{k}\,S\,\beta_{\textrm{link}}$$

- **$t_{\textrm{compute}}(B/k)$:** Local forward, backward, and optimizer time on a shard of $B/k$ examples.
- **$c\,\alpha$:** Fixed start-up overhead across $c$ separate allreduce calls per step ($c = 0$ when $k = 1$).
- **$\frac{2(k-1)}{k} S \beta_{\textrm{link}}$:** Byte transfer time for $S$ total gradient bytes per GPU.
- **When $k = 2$ beats $k = 1$:** A second GPU shortens the step only if $t_{\textrm{compute}}(B) - t_{\textrm{compute}}(B/2) > c\alpha + S\beta_{\textrm{link}}$.
:::

::: {.slide title="Measuring Collective Latency α and Bandwidth β_link"}
[Link Calibration]{.kicker}

We calibrate $\alpha$ with a 1-float ($4\text{ B}$) allreduce and $\beta_{\textrm{link}}$ with a $2^{26}$-float ($268\text{ MB}$) allreduce across two RTX 4090s:

@!multiple-gpus-the-accounting

- **Why bus bandwidth is well below $32\text{ GB/s}$:** Because consumer RTX 4090s lack NVLink and PCIe peer-to-peer access, every inter-GPU byte traverses PCIe into host DRAM and back out over PCIe to the destination GPU.
:::

::: {.slide title="Why Small Gradient Tensors Must Be Concatenated"}
[Envelopes vs. Bytes]{.kicker}

LeNet has $10$ parameter tensors (5 weights, 5 biases) totaling $61{,}706$ `float32` parameters ($S = 247\text{ KB}$). Allreducing them one by one ($c = 10$) versus concatenated into a single flat vector ($c = 1$) moves the exact same $247\text{ KB}$:

@!multiple-gpus-latency-and-bandwidth-of-the-allreduce-1

::: {.d2l-note .rule}
**The price is in the envelopes ($9\alpha$):** For a $247\text{ KB}$ gradient, $S\beta_{\textrm{link}} \approx 100\,\mu\text{s}$, so launching $9$ extra allreduces costs more time in start-up latency than transferring all the gradient bytes! Always fuse or bucket small gradients before communicating.
:::
:::

::: {.slide title="Profiling LeNet: An Overhead-Bound Single-GPU Step"}
[Local Step Profile]{.kicker}

Before timing LeNet on two GPUs at global batch size $B = 256$, we profile its local computation (`communicate=False`) on one GPU:

@!multiple-gpus-lenet-on-one-and-two-gpus-2

- **Host dispatch dominates:** The GPU is busy for only a small fraction of the step; LeNet at $B = 256$ is **overhead-bound**, so its single-GPU step time is governed by host launch time $h_{\textrm{step}}$ rather than GPU arithmetic.
:::

::: {.slide title="The Single-Launcher Bottleneck: Host Work Doubles"}
[Host Dispatch Ceiling]{.kicker}

::: {.cols .vc}
::: {.col .fig}
![](../img/mdl-perf-one-launcher.svg){width=100%}
:::

::: {.col}
- **Halving an overhead-bound batch saves nothing:** On $1$ GPU, $b = 128$ takes nearly the exact same time $h_{\textrm{step}}$ as $b = 256$.
- **One host thread driving $k$ GPUs:** When a single Python loop dispatches kernels to GPU 0 and then to GPU 1, host launch time scales as $k \cdot h_{\textrm{step}}$!

@!multiple-gpus-lenet-on-one-and-two-gpus-1
:::
:::
:::

::: {.slide title="Predicting and Measuring the Two-GPU LeNet Slowdown"}
[Accounting for Slowdown]{.kicker}

Replacing $t_{\textrm{compute}}(B/k)$ by $\max(k\,h_{\textrm{step}},\, t_{\textrm{dev}}(B/k))$ accounts for both single-thread host dispatch and device execution:

$$t_{\textrm{step}}(k) \;\approx\; \max\!\bigl(k\,h_{\textrm{step}},\; t_{\textrm{dev}}(B/k)\bigr) \;+\; c\,\alpha \;+\; \frac{2(k-1)}{k}\,S\,\beta_{\textrm{link}}$$

@!multiple-gpus-data-parallelism-by-hand-6

::: {.d2l-note .warn}
At global batch $B = 256$, two GPUs run **nearly $2\times$ slower** than one GPU—not because of the $247\text{ KB}$ gradient exchange ($\sim 0.1\text{ ms}$), but because the single host launcher takes $2h_{\textrm{step}}$ to issue work to two cards!
:::
:::

::: {.slide title="Strong Scaling vs. Weak Scaling Conventions"}
[Scaling Laws]{.kicker}

Every multi-GPU benchmark follows one of two scaling conventions, and confusing them misinterprets reported speedups:

| Scaling Mode | Batch Size Convention | Ideal $k$-GPU Step Time | Ideal Throughput ($\textrm{ex/s}$) | Optimization Impact |
| :--- | :--- | :---: | :---: | :--- |
| **Strong Scaling** | Fixed global batch $B$ ($b = B/k$ per GPU) | $t_{\textrm{step}}(1) / k$ | $k \times$ faster to finish $B$ | Identical gradient statistics & convergence |
| **Weak Scaling** | Fixed per-GPU batch $b$ ($B = k b$ global) | $t_{\textrm{step}}(1)$ | $k \times$ examples/s | Alters optimization dynamics at very large $kb$ |

- **Weak-scaling speedup condition ($k = 2$ with one launcher):** Two GPUs achieve higher throughput ($2\,t(1)/t(2) > 1$) only when:
  $$\max\bigl(2h_{\textrm{step}},\; t_{\textrm{dev}}(b)\bigr) + \alpha + S\,\beta_{\textrm{link}} \;<\; 2\max\bigl(h_{\textrm{step}},\; t_{\textrm{dev}}(b)\bigr)$$
:::

::: {.slide title="Crossing the Scaling Threshold with Larger Minibatches"}
[Amortizing Overheads]{.kicker}

Increasing the per-GPU batch size $b$ from $256 \to 4{,}096 \to 32{,}768$ pushes device execution time $t_{\textrm{dev}}(b)$ past both $2h_{\textrm{step}}$ and the gradient exchange $\alpha + S\beta_{\textrm{link}}$:

@!multiple-gpus-larger-minibatches

- **Three distinct regimes:** At $b = 256$, $t_{\textrm{dev}} \ll h_{\textrm{step}}$ (no gain); at $b = 4{,}096$, $t_{\textrm{dev}}$ begins to overtake $h_{\textrm{step}}$; at $b = 32{,}768$, device compute dominates both host dispatch and the $247\text{ KB}$ allreduce, achieving **$\sim 1.9\times$ weak-scaling speedup** on two GPUs.
:::

::: {.slide title="Concept Check: Allreduce Traffic and Multi-GPU Speedup"}
[Check Your Understanding]{.kicker}

Use our multi-GPU cost model to answer two concrete scaling questions:

1. A 1B-parameter GPT ($N = 10^9$, `float32` gradients $S = 4\text{ GB}$) trains on $k = 8$ GPUs. How many bytes does the busiest GPU send per step in a **star allreduce** versus a **ring allreduce**?
2. Suppose one step of the 1B GPT processes $32{,}768$ tokens globally ($6N\text{ FLOPs/token}$) at an achieved rate of $35\text{ TFLOP/s}$ per GPU in `bfloat16`. Without communication overlap, what is the strong-scaling speedup of $k = 2$ GPUs over $1$ GPU when the inter-GPU bus bandwidth is (a) $10\text{ GB/s}$ vs. (b) $2.5\text{ GB/s}$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Star hub sends $(k-1)S = 7 \times 4 = \mathbf{28\text{ GB}}$** (and receives $28\text{ GB}$). **Ring sends $\frac{2(k-1)}{k}S = \frac{14}{8} \times 4 = \mathbf{7\text{ GB}}$ per GPU** ($4\times$ less peak per-device traffic, strictly $< 2S = 8\text{ GB}$).
2. **1 GPU compute time:** $\frac{6 \times 10^9 \times 32{,}768}{3.5 \times 10^{13}} \approx \mathbf{5.6\text{ s}}$. **2 GPUs compute time:** $2.8\text{ s}$, plus $S\beta_{\textrm{link}}$ ($k=2$ sends $\frac{2(1)}{2}S = 4\text{ GB}$):
   - **(a) At $10\text{ GB/s}$:** $t_{\textrm{comm}} = 0.4\text{ s} \implies t_{\textrm{step}}(2) = 3.2\text{ s} \implies \textrm{Speedup} = 5.6 / 3.2 = \mathbf{1.75\times}$.
   - **(b) At $2.5\text{ GB/s}$:** $t_{\textrm{comm}} = 1.6\text{ s} \implies t_{\textrm{step}}(2) = 4.4\text{ s} \implies \textrm{Speedup} = 5.6 / 4.4 = \mathbf{1.27\times}$.
:::
:::

::: {.slide title="Summary: Collectives, Ring Allreduce, and Cost Models"}
[Key Takeaways]{.kicker}

| Concept / Mechanism | Mathematical Form / Rule | Value / Behavior | Role in Multi-GPU Training |
| :--- | :--- | :--- | :--- |
| **Data Parallelism** | $\mathbf{g} = \frac{1}{k}\sum_{j=1}^k \mathbf{g}_j$ | Replicates $16N$ state on $k$ GPUs | Splits batch $B$ into $k$ shards of $B/k$; sums grads once/step |
| **One-Step Check** | $\|\Delta \boldsymbol{\theta}_{k=2} - \Delta \boldsymbol{\theta}_{k=1}\|_\infty$ | $\le 10^{-7}$ in true `fp32` | Catches missing $1/k$ division or shard-slicing bugs |
| **Ring Allreduce** | Reduce-scatter + All-gather | $\frac{2(k-1)}{k}S < 2S$ bytes/GPU | Distributes traffic evenly across $k$ GPUs in $2(k-1)$ hops |
| **Step Cost Model** | $\max(kh_{\textrm{step}}, t_{\textrm{dev}}) + c\alpha + \frac{2(k-1)}{k}S\beta_{\textrm{link}}$ | $\beta_{\textrm{link}} = 1 / \textrm{bus BW}$ | Predicts when adding GPUs speeds up or slows down a step |
| **Gradient Fusion** | Concatenate $c$ tensors $\to 1$ buffer | Saves $(c - 1)\alpha$ latency | Eliminates per-tensor start-up overhead for small gradients |
| **Strong vs. Weak** | Fixed $B$ vs. fixed $b = B/k$ | Speedup $t(1)/t(k)$ vs. $k t(1)/t(k)$ | Distinguishes faster single steps from higher total throughput |
:::
