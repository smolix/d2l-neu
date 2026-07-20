# Multi-GPU from First Principles
:label:`sec_multi_gpu`

The ladder of :numref:`sec_memory_precision` ended with the one rung that
buys both more compute and more memory at once: another GPU. This section
adds it — but honestly, and from the ground up. We build data-parallel
training by hand, with explicit device-to-device copies and a
hand-rolled gradient sum, so that when :numref:`sec_multi_gpu_concise`
replaces our loop with production machinery, you know exactly what that
machinery does. We derive the communication algorithm the professionals
use (ring allreduce) and its cost, and — because this book's build box
has no fast inter-GPU fabric — we *measure* what communication actually
costs and discover the central fact of parallel training: **a second GPU
is not free, and whether it pays is an accounting question you can answer
before you run.**

That last point is why this section is built on a machine with no NVLink
and no peer-to-peer transfer (:numref:`sec_hardware`). On a datacenter
rack the communication cost hides under a terabyte-per-second fabric and
the accounting feels academic; on our box it is impossible to ignore,
which makes it the better teacher. Every conclusion here holds at two
GPUs as well as four — the number of devices is a variable, never a
constant.

*Prerequisites: minibatch SGD and the effect of batch size on the gradient
estimate (*:numref:`sec_minibatch_sgd`*); LeNet (*:numref:`sec_lenet`*); the
interconnect measurements of* :numref:`subsec_hw-interconnects`*.*

```{.python .input #multiple-gpus-multi-gpu-from-first-principles}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

torch.set_float32_matmul_precision('high')
```

```{.python .input #multiple-gpus-multi-gpu-from-first-principles}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import optax
from functools import partial
```

## Three Ways to Split
:label:`subsec_mg-splitting`

Given more than one GPU and a model to train, there are three ways to
divide the work (:numref:`fig_splitting`).

![Three ways to split a model across two devices: replicate it and split
the batch (data parallel); split the layer stack (pipeline parallel);
split every layer's width (tensor parallel).](../img/mdl-perf-splitting.svg)
:label:`fig_splitting`

**Data parallelism** replicates the whole model on every GPU and splits the
*batch*: each device runs the full network on a different slice of the
minibatch, then the devices sum their gradients so every replica takes the
same optimizer step. It is the simplest, works for any model that fits on
one GPU, and needs communication only once per step — so it is what this
chapter builds. **Pipeline parallelism** splits the *layer stack* across
devices, each holding a few consecutive layers; it lets a model too deep
for one GPU fit, at the cost of tight inter-stage synchronization.
**Tensor parallelism** splits *within* each layer — each device holds a
slice of every weight matrix — and communicates several times per layer.
The latter two matter only at scales this part defers to the Language
Models chapters, which have models large enough to warrant them; a single
historical note is that the very first of them appeared in 2012, when
AlexNet was split across two GPUs simply because its weights did not fit
in one card's 3 GB :cite:`Krizhevsky.Sutskever.Hinton.2012`. Data
parallelism is our subject, and — a warning that :numref:`sec_memory_precision`
already made — it does *not* let you train a bigger model: every GPU still
holds a full copy.

## Data Parallelism by Hand
:label:`subsec_mg-byhand`

Data-parallel training on $k$ GPUs is five steps per minibatch: split the
batch into $k$ shards; run forward and backward on each shard against that
device's copy of the parameters; sum the $k$ gradient sets so all devices
agree; and let every device apply the same update
(:numref:`fig_data_parallel`). We build each piece for a small LeNet, then
run it — and watch it lose.

![One data-parallel step on two GPUs. Split the minibatch, run identical
forward and backward passes on each replica, sum the gradients with an
allreduce, then apply the identical update
everywhere.](../img/mdl-perf-data-parallel.svg)
:label:`fig_data_parallel`

We define LeNet from raw tensors (not a module) so that parameters,
gradients, and their movement between devices are fully visible:

```{.python .input #multiple-gpus-data-parallelism-by-hand-1}
%%tab pytorch
scale = 0.01
def new_params(device):
    def p(*shape):
        return (torch.randn(*shape, device=device) * scale).requires_grad_()
    return [p(20, 1, 3, 3), torch.zeros(20, device=device, requires_grad=True),
            p(50, 20, 5, 5), torch.zeros(50, device=device, requires_grad=True),
            p(800, 128), torch.zeros(128, device=device, requires_grad=True),
            p(128, 10), torch.zeros(10, device=device, requires_grad=True)]

def lenet(X, params):
    h1 = F.avg_pool2d(F.relu(F.conv2d(X, params[0], params[1])), 2, 2)
    h2 = F.avg_pool2d(F.relu(F.conv2d(h1, params[2], params[3])), 2, 2)
    h2 = h2.reshape(h2.shape[0], -1)
    h3 = F.relu(torch.mm(h2, params[4]) + params[5])
    return torch.mm(h3, params[6]) + params[7]

loss = nn.CrossEntropyLoss(reduction='none')
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-1}
%%tab jax
scale = 0.01
def new_params(key):
    ks = jax.random.split(key, 4)
    return [jax.random.normal(ks[0], (20, 1, 3, 3)) * scale, jnp.zeros(20),
            jax.random.normal(ks[1], (50, 20, 5, 5)) * scale, jnp.zeros(50),
            jax.random.normal(ks[2], (800, 128)) * scale, jnp.zeros(128),
            jax.random.normal(ks[3], (128, 10)) * scale, jnp.zeros(10)]

def lenet(params, X):
    conv = lambda x, W: jax.lax.conv_general_dilated(
        x, W, (1, 1), 'VALID', dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    pool = lambda x: jax.lax.reduce_window(
        x, 0.0, jax.lax.add, (1, 1, 2, 2), (1, 1, 2, 2), 'VALID') / 4.0
    h1 = pool(jax.nn.relu(conv(X, params[0]) + params[1].reshape(1, -1, 1, 1)))
    h2 = pool(jax.nn.relu(conv(h1, params[2]) + params[3].reshape(1, -1, 1, 1)))
    h2 = h2.reshape(h2.shape[0], -1)
    h3 = jax.nn.relu(jnp.dot(h2, params[4]) + params[5])
    return jnp.dot(h3, params[6]) + params[7]
```

The two operations that make it parallel are *broadcasting* parameters to
each device and *summing* gradients across them. We build the PyTorch
version by hand, because the hand-rolled version is the lesson; the
`#@save`d `split_batch` (which chops a minibatch into per-device shards)
rounds out the toolkit:

```{.python .input #multiple-gpus-data-parallelism-by-hand-2}
%%tab pytorch
def get_params(params, device):
    return [p.clone().to(device).detach().requires_grad_() for p in params]

def allreduce(data):
    # Star pattern: sum everything onto device 0, then broadcast back.
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

def split_batch(X, y, devices):  #@save
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-2}
%%tab jax
def get_params(params, num_devices):
    """Data parallelism replicates the full parameter set on every device;
    shard_map's P() in-spec (below) broadcasts one copy to each."""
    return params

def split_batch(X, y, num_devices):  #@save
    """Reshape `X` and `y` onto a leading device axis of size num_devices."""
    assert X.shape[0] % num_devices == 0
    reshape = lambda a: a.reshape(num_devices, -1, *a.shape[1:])
    return reshape(X), reshape(y)
```

`allreduce` is the heart of it, and the naive version above is
deliberately clumsy: it gathers every device's gradient onto device 0,
sums there, and broadcasts back — a *star* topology, with device 0 as a
hub that handles all $k-1$ inbound and $k-1$ outbound transfers. Watch it
work on two vectors living on two GPUs:

```{.python .input #multiple-gpus-data-parallelism-by-hand-3}
%%tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1)
        for i in range(min(2, d2l.num_gpus()))]
print('before:', [d.cpu().numpy().tolist() for d in data])
allreduce(data)
print('after: ', [d.cpu().numpy().tolist() for d in data])
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-3}
%%tab jax
# JAX expresses the same sum as a collective inside shard_map (below);
# here we show the result the collective must produce.
data = jnp.stack([jnp.ones((1, 2)) * (i + 1)
                  for i in range(min(2, jax.local_device_count()))])
print('before:', data.tolist())
print('summed:', jnp.broadcast_to(data.sum(0), data.shape).tolist())
```

The training step assembles the pieces. Each device computes its shard's
gradient; we allreduce parameter by parameter; each device applies plain
SGD. The whole thing runs in one Python process — nothing here needs
multiple processes, because we move tensors explicitly:

```{.python .input #multiple-gpus-data-parallelism-by-hand-4}
%%tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    ls = [loss(lenet(Xs, dev_W), ys).sum()
          for Xs, ys, dev_W in zip(X_shards, y_shards, device_params)]
    for l in ls:
        l.backward()
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
        for param in device_params:
            d2l.sgd(param, lr, X.shape[0])
            for p in param:
                p.grad = None
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-4}
%%tab jax
@partial(jax.jit, static_argnames=('lr', 'mesh'))
def train_step(params, X, y, lr, mesh):
    """One data-parallel step: shard_map makes the psum collective explicit.
    `X`, `y` arrive with the batch sharded across devices (P('data')) and
    `params` replicated (P()) -- see `train` below; shard_map hands each device
    the full parameter replica and its own batch shard, computes that shard's
    gradient, and psum sums the gradients across devices."""
    P = jax.sharding.PartitionSpec

    def per_device(params, X, y):
        def loss_fn(p):
            logits = lenet(p, X[0])   # X[0]: strip the size-1 sharded axis
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, y[0]).mean()
        grads = jax.grad(loss_fn)(params)
        grads = jax.lax.psum(grads, 'data')     # The allreduce, in one line
        return jax.tree.map(lambda p, g: p - lr * g, params, grads)

    step = jax.shard_map(per_device, mesh=mesh,
                         in_specs=(P(), P('data'), P('data')),
                         out_specs=P())
    return step(params, X, y)
```

The two tabs make the same computation visible in two idioms. PyTorch
moves gradients between devices with explicit `.to(device)` copies inside
`allreduce`; JAX writes the collective as a single `jax.lax.psum` inside a
`jax.shard_map`, where the `PartitionSpec('data')` annotation tells XLA
that the leading axis is sharded across devices. (Note the top-level
`jax.shard_map` — the older `jax.pmap` is a compatibility shim as of JAX
0.8 and we do not use it.) The collective is *visible in the code* in both
— exactly the point of building it by hand. The training and evaluation
loops that wrap `train_batch`/`train_step` are the multi-GPU cousins of
:numref:`sec_lenet`'s; we elide them here (they live in the notebook) and
run the result. First, one GPU:

```{.python .input #multiple-gpus-data-parallelism-by-hand-5}
%%tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    base = new_params(devices[0])
    device_params = [get_params(base, d) for d in devices]
    timer, num_epochs = d2l.Timer(), 5
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
    acc = d2l.evaluate_accuracy_gpu(
        lambda x: lenet(x, device_params[0]), test_iter, devices[0])
    print(f'test acc {acc:.2f}, {timer.avg():.2f} sec/epoch on {num_gpus} GPU(s)')

train(num_gpus=1, batch_size=256, lr=0.2)
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-5}
%%tab jax
def train(num_gpus, batch_size, lr):
    P = jax.sharding.PartitionSpec
    data = d2l.FashionMNIST(batch_size=batch_size)
    train_iter = data.get_dataloader(train=True)
    mesh = jax.make_mesh((num_gpus,), ('data',))
    place = lambda a, spec: jax.device_put(
        a, jax.sharding.NamedSharding(mesh, spec))
    params = get_params(new_params(jax.random.PRNGKey(0)), num_gpus)
    params = jax.tree.map(lambda p: place(p, P()), params)  # a replica per device
    timer, num_epochs = d2l.Timer(), 5
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            X = jnp.array(X).transpose(0, 3, 1, 2)
            if X.shape[0] % num_gpus:       # drop a ragged final batch
                continue
            Xs, ys = split_batch(X, jnp.array(y), num_gpus)
            Xs, ys = place(Xs, P('data')), place(ys, P('data'))  # shard the batch
            params = train_step(params, Xs, ys, lr, mesh)
        jax.block_until_ready(params)
        timer.stop()
    print(f'{timer.avg():.2f} sec/epoch on {num_gpus} GPU(s)')

train(num_gpus=1, batch_size=256, lr=0.2)
```

Now two GPUs, same batch size and learning rate. The accuracy is
essentially unchanged — the optimization is mathematically identical — but
the wall-clock is **worse**:

```{.python .input #multiple-gpus-data-parallelism-by-hand-6}
%%tab pytorch
train(num_gpus=min(2, d2l.num_gpus()), batch_size=256, lr=0.2)
```

```{.python .input #multiple-gpus-data-parallelism-by-hand-6}
%%tab jax
train(num_gpus=min(2, jax.local_device_count()), batch_size=256, lr=0.2)
```

The second GPU buys **no speedup** — the two-GPU run is no faster than one,
and (depending on each framework's per-step overhead) can be outright
slower — and this is not a bug, it is the syllabus.
The reason is *not* slow communication: LeNet's parameters are tiny, so
their gradient allreduce is negligible, and we will measure the raw copy
path at tens of GB/s below. The reason is that LeNet is simply too small
to parallelize. Splitting a 256-example batch into two 128-example shards
underfeeds each GPU — a convolution on half the batch does not run at half
the time, because a small batch never filled the device to begin with —
and the Python orchestration and per-step synchronization are not amortized
by microsecond-scale compute. The technique is not wrong; the *regime* is
wrong. This tiny model is the worst case for data parallelism, and
diagnosing exactly why — separating the communication cost, which is small
here, from the underutilization cost, which is not — is the next two
subsections' work.

## Doing Better: Ring Allreduce
:label:`subsec_mg-ring`

Our star `allreduce` has an obvious flaw: device 0 is a hub through which
everything passes, so it moves $(k-1)N$ bytes in and $(k-1)N$ out for a
parameter set of $N$ bytes, and the other devices' links sit idle while it
works. The algorithm the professionals use — *ring allreduce*
:cite:`Patarasuk.Yuan.2009` — removes the hub entirely, and it is worth
deriving because the same identity reappears as the seed of FSDP in
:numref:`sec_multi_gpu_concise`.

Arrange the $k$ devices in a ring, each talking only to its neighbor.
Split each device's gradient vector into $k$ equal chunks. The allreduce
runs in two phases (:numref:`fig_ring_allreduce`). In **reduce-scatter**,
over $k-1$ steps, device $i$ sends one chunk to its neighbor while
receiving and accumulating another; after the phase, each device owns one
fully-summed chunk (a different chunk per device). In **all-gather**,
another $k-1$ steps pass those finished chunks around the ring until every
device has all $k$. At every step, every link carries exactly one chunk of
size $N/k$, so each device sends $(k-1) \cdot N/k$ bytes per phase, and
both phases together move

$$
\frac{2(k-1)}{k}\, N \quad\text{bytes per device,}
$$
:eqlabel:`eq_ring_traffic`

which approaches $2N$ as $k$ grows and is **independent of $k$** — adding
GPUs does not increase the per-device communication. The star pattern's
hub moved $(k-1)N$; the ring moves a constant. That is why every
production collective library is built on rings (and trees for small
messages).

![Ring allreduce on four GPUs, computed step by step. Reduce-scatter
accumulates one chunk per device around the ring; all-gather then
distributes the finished chunks. Each cell counts how many of the four
contributions a chunk holds; a full chunk (4) is
complete.](../img/mdl-perf-ring-allreduce.svg)
:label:`fig_ring_allreduce`

The catch, on our box, is that the elegant $2(k-1)/k$ accounting assumes
the links are the bottleneck — and they are, but *which* links? With no
peer-to-peer transfer, every "neighbor to neighbor" hop is really a
round trip through host memory (:numref:`subsec_hw-interconnects`), so the
ring's theoretical advantage over the star is largely erased: the
transport, not the topology, is the ceiling. This is the theory-versus-
practice lesson in miniature — NCCL will still pick a ring or tree per
message size, but on this hardware the constant in front of $N$ is what
hurts, and no algorithm fixes a slow wire.

## The Accounting
:label:`subsec_mg-accounting`

We can now answer the question data parallelism always poses — *does the
next GPU pay?* — with a cost model rather than a guess. One step on $k$
GPUs takes roughly

$$
t_{\text{step}}(k) \;\approx\;
\underbrace{t_{\text{compute}}(B/k)}_{\text{forward+backward on a shard}}
\;+\;
\underbrace{t_{\text{comm}}}_{\approx\, 2N / \beta},
$$
:eqlabel:`eq_dp_cost`

where $B$ is the global batch, $N$ the bytes of gradients to reduce, and
$\beta$ the achievable allreduce bandwidth (:eqref:`eq_ring_traffic` gives
the $2N$; the per-device traffic is independent of $k$). The compute term
*shrinks* with more GPUs (each does less of the batch); the communication
term does not. Parallelism pays exactly when the compute you offload
exceeds the communication you take on — big models (large $N$-relative-
to-work... no: large *compute* per byte communicated), big per-device
batches, and fast links all push in your favor; a tiny model on a slow
link, like LeNet on our box, is the case where it never pays. Let's plug
in real numbers by measuring $\beta$ directly, the effective bandwidth of
our hand-rolled allreduce:

```{.python .input #multiple-gpus-the-accounting}
%%tab pytorch
if d2l.num_gpus() >= 2:
    N = 64 * 1024 * 1024  # 64M floats = 256 MB per replica
    data = [torch.randn(N, device=d2l.try_gpu(i)) for i in range(2)]
    t = d2l.Benchmark(lambda: allreduce(data), warmup=2, repeats=5).time
    # Ring traffic per device is ~2N bytes; report effective busbw
    print(f'allreduce {2 * N * 4 / t / 1e9:.2f} GB/s effective '
          f'over {1000 * t:.1f} ms')
else:
    print('needs 2 GPUs')
```

```{.python .input #multiple-gpus-the-accounting}
%%tab jax
if jax.local_device_count() >= 2:
    mesh = jax.make_mesh((2,), ('data',))
    P = jax.sharding.PartitionSpec
    N = 64 * 1024 * 1024
    x = jax.device_put(jnp.ones((2, N)),
                       jax.sharding.NamedSharding(mesh, P('data')))
    psum = jax.jit(jax.shard_map(
        lambda a: jax.lax.psum(a, 'data'), mesh=mesh,
        in_specs=P('data'), out_specs=P('data')))
    t = d2l.Benchmark(lambda: psum(x), warmup=2, repeats=5).time
    print(f'psum {2 * N * 4 / t / 1e9:.2f} GB/s effective over {1000*t:.1f} ms')
else:
    print('needs 2 GPUs')
```

The hand-rolled copy sustains on the order of ten GB/s — this is a plain
PCIe transfer, staged through host memory, running near the bus limit, and
already one to two orders of magnitude below an NVLink domain's ~1.8 TB/s
per GPU (:numref:`tab_gpu_specs`). A theory-versus-practice aside worth
noticing: a *collective library* like NCCL, whose ring/tree chunking is
tuned for peer-to-peer fabrics, extracts noticeably *less* effective
bandwidth than this naive one-shot copy on our P2P-less box — its busbw
here is only a couple of GB/s. On this hardware the transport, not the
algorithm, is the ceiling, and a clever ring cannot beat a slow wire.

Now read the cost model honestly against what we measured. LeNet's
parameters are tiny, so $2N/\beta$ is a fraction of a millisecond — the
communication term is *not* what denies the speedup. The culprit is the
other term: $t_{\text{compute}}(B/k)$ does not actually fall like $1/k$
for a small model, because halving an already-small batch leaves each GPU
underutilized, so $t_{\text{compute}}(B/2) \approx t_{\text{compute}}(B)$
and the second GPU does redundant-feeling work for no wall-clock gain. The
model pays off only when compute genuinely scales with the batch — a
compute-dense network with a large per-device batch, where
$t_{\text{compute}}(B/k) \approx t_{\text{compute}}(B)/k$ dominates the
small $t_{\text{comm}}$. That is exactly the regime
:numref:`sec_multi_gpu_concise` moves to, and where the second GPU finally
earns its keep.

A closing word of history, because it names the lineage. Before
synchronous ring allreduce won for dense training, large-scale learning
organized this same communication through *parameter servers*: dedicated
storage nodes exposed a `push` (accumulate my gradient) and `pull`
(give me the current sum) interface, sharding the parameters across
servers so aggregation bandwidth scaled with the fleet
:cite:`Li.Andersen.Park.ea.2014`. That push/pull abstraction is the
authors' own lineage, and it lives on today mainly in recommender-system
embedding tables (the territory of :numref:`chap_recsys`), where the
parameters are too large and too sparse for every worker to hold a full
replica. For dense training, collectives won; the modern production map is
:numref:`sec_training_systems`.

## Summary

* Data parallelism replicates the model, splits the batch, sums gradients
  with an allreduce, and applies the identical update on every device.
  It is the simplest form of multi-GPU training and does not enlarge the
  model that fits.
* Built by hand, the gradient sum is a star: device 0 is a hub moving
  $(k-1)N$ bytes each way. Ring allreduce removes the hub — reduce-scatter
  then all-gather — moving $2(k-1)/k \cdot N$ bytes per device,
  *independent of $k$*. That identity reappears as the basis of FSDP.
* The cost model $t_{\text{step}}(k) \approx t_{\text{compute}}(B/k) +
  2N/\beta$ decides whether a GPU pays — but only when compute genuinely
  scales with the batch. LeNet is the honest worst case: its gradients are
  tiny (communication negligible, and the hand-rolled copy runs at tens of
  GB/s anyway), yet a second GPU buys no speedup because halving an
  already-small batch underutilizes each device, so $t_{\text{compute}}(B/k)$
  does not fall. Inter-GPU bandwidth on this P2P-less box is PCIe-limited
  (tens of GB/s for a raw copy, a couple of GB/s of NCCL busbw) — one to two
  orders of magnitude below an NVLink domain.
* Parameter servers (push/pull) organized this communication for the
  asynchronous, multi-machine era; synchronous collectives won for dense
  training, and the pattern survives in recsys embedding systems.

## Exercises

1. Extend the hand-rolled version to $k = 4$ (guard with `d2l.num_gpus()`)
   and measure `sec/epoch` at $k \in \{1, 2, 4\}$. Does the slowdown grow,
   shrink, or hold? Explain using :eqref:`eq_dp_cost`.
1. Implement ring allreduce with explicit `.to()` copies (reduce-scatter
   then all-gather) and test whether it beats the star `allreduce` on our
   host-staged box. It barely can — explain why in terms of
   :numref:`subsec_hw-interconnects`.
1. Compute the ring's per-device traffic (:eqref:`eq_ring_traffic`) for
   ResNet-18's ~11M parameters in fp32, and, using your measured $\beta$,
   predict $t_{\text{comm}}$ per step. :numref:`sec_multi_gpu_concise`
   will measure the real thing — how close is your prediction?
1. A thought experiment: sending gradients in bf16 instead of fp32 halves
   $N$ and hence $t_{\text{comm}}$. What could break, and which term of
   :eqref:`eq_dp_cost` does it help — the one that shrinks with $k$ or the
   one that does not?
1. Scale the batch size with $k$ (from $B$ to $kB$) so each device keeps a
   full $B$-sized shard. Now which term of :eqref:`eq_dp_cost` dominates,
   and does the second GPU pay? Relate your answer to the batch-size
   discussion of :numref:`sec_batch_size`.

<!-- slides -->

::: {.slide title="The Next Rung: Another GPU"}
More GPUs buy more compute *and* more memory. The catch:
communication is not free, and on a box with no NVLink it is
loud enough to hear.

Plan: build data parallelism by hand, derive the collective
the professionals use, then *measure* what a second GPU costs —
and predict, before running, whether it pays.
:::

::: {.slide title="Three Ways to Split"}
![](../img/mdl-perf-splitting.svg){width=95%}

Data parallel is our subject: simplest, one sync per step,
works for any model that fits. Pipeline and tensor parallel
wait for the Language Models part.
:::

::: {.slide title="Data Parallelism by Hand"}
![](../img/mdl-perf-data-parallel.svg){width=62%}

Split batch → forward/backward per replica → **allreduce
gradients** → identical update. One process; tensors moved
explicitly.

@multiple-gpus-data-parallelism-by-hand-4
:::

::: {.slide title="Two GPUs, No Speedup"}
@multiple-gpus-data-parallelism-by-hand-6

Not a bug — the syllabus. LeNet is too small: halving a small
batch underutilizes each GPU. *Not* a bandwidth problem — the
hand-rolled copy runs at tens of GB/s. Wrong regime, not wrong
technique.
:::

::: {.slide title="Ring Allreduce"}
![](../img/mdl-perf-ring-allreduce.svg){width=95%}

Star: hub moves $(k-1)N$. Ring (reduce-scatter + all-gather):
$\frac{2(k-1)}{k}N$ per device — **independent of $k$**. The
identity that becomes FSDP.
:::

::: {.slide title="The Accounting"}
$$t_{\text{step}}(k) \approx t_{\text{compute}}(B/k) + 2N/\beta$$

@multiple-gpus-the-accounting

The copy runs at tens of GB/s (PCIe-limited; NCCL busbw is
lower). So LeNet's no-speedup isn't communication — it's
$t_{\text{compute}}(B/k)$ *not* shrinking when a small batch is
halved. Big model + big batch → the second GPU pays (next
section).
:::

::: {.slide title="Lineage"}
- **Parameter servers** (push/pull): the asynchronous,
  multi-machine era; alive today in recsys embeddings.
- **Synchronous collectives** (ring allreduce): won for dense
  training; what DDP runs.

Production map → the Tools appendix. Next: let the library
run the ring for us.
:::
