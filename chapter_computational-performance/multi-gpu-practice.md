# Data Parallelism in Practice
:label:`sec_multi_gpu_concise`

<!-- WRITER: retitled after the split (structure.md §6.1). The opening question "what can several GPUs do for a model that is too large for any one of them?" and the roadmap sentence "We then divide the training state among the GPUs" now belong to sec_sharding; the 36-block GPT paragraph moved there. -->

In :numref:`sec_multi_gpu` we built data parallelism by hand, and a second
GPU made the training steps of LeNet slower: each GPU had too little work
to outweigh even a small exchange of gradients. A larger model gives each
GPU more work for every byte it exchanges, so it should fare better. How
much better, once the exchange is organized well? And what can several
GPUs do for a model that is too large for any one of them?

A ResNet-18 on Fashion-MNIST images enlarged to 64×64 pixels gives each
GPU far more work than LeNet. It performs 2.2 billion multiply-adds per image, about
fifty for every byte of its 45 MB of gradients, so the work that a second
GPU takes over is large compared with the gradients that the two GPUs must
exchange. Whether this advantage survives depends on how the exchange is
organized. The hand-written loop of :numref:`sec_multi_gpu` waited for
the whole backward pass before it communicated, although the gradients of
the last layers were ready long before those of the first.

:begin_tab:`pytorch`
It also issued the work of every GPU from one Python thread and routed all
gradients through device 0.
:end_tab:

:begin_tab:`jax`
It also spelled out the communication itself, a `shard_map` with a `psum`
over the flattened gradients, which every new model would need again.
:end_tab:

We first replace the hand-written loop with each framework's
production tool for data parallelism, which starts communicating while the
backward pass is still running. We measure how ResNet-18 then scales on two
and four GPUs, and compare the measurement with the cost model of
:numref:`sec_multi_gpu`. We then divide the training state among the GPUs
instead of copying it, which reduces the memory per GPU but adds
communication.
:numref:`sec_fast_transformer` applies both to a transformer.

```{.python .input #multi-gpu-practice-setup}
%%tab jax
# JAX and NCCL read these settings once, when they start, so this cell runs
# before the imports. XLA moves a collective away from the computation
# around it, so that the two overlap, only when its latency-hiding scheduler
# is on. The 'vmm' allocator reserves nothing up front and maps each array
# into one contiguous range, where JAX's default allocator reserves most of
# every GPU at start-up and carves every array out of that block: one step
# below needs about 10 GB in one piece after many smaller arrays have come
# and gone, and a later cell starts a second JAX process on the same GPUs.
# The notebook also runs with NCCL_LOCAL_REGISTER=0: NCCL's buffer
# registration fails on GPUs without peer-to-peer access, such as ours, and
# would print a warning at every collective.
import os
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                           ' --xla_gpu_enable_latency_hiding_scheduler=true')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'vmm')
```

```{.python .input #multi-gpu-practice-multi-gpu-in-practice}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #multi-gpu-practice-multi-gpu-in-practice}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #multi-gpu-practice-multi-gpu-in-practice-1}
%%tab jax
import json
import re
import subprocess
import sys
from jax.sharding import NamedSharding, PartitionSpec as P
```

```{.python .input #multi-gpu-practice-multi-gpu-in-practice-1}
%%tab pytorch
import json
import os
import random
import sys
```

## Production Data Parallelism
:label:`subsec_mgp-production`

Production data parallelism computes what the loop of
:numref:`subsec_mg-byhand` computes. Every GPU holds a copy of the model,
computes the gradient of the loss on its share of each minibatch, and
applies the average of the gradients of all shares. What changes is how the
work is organized, and above all when the communication starts.

:begin_tab:`pytorch`
PyTorch changes two more things. Every GPU gets a process of its own,
and NCCL replaces our star with the ring of :numref:`subsec_mg-ring`,
or with a tree for small messages, so that no device carries the traffic of
all the others.
:end_tab:

:begin_tab:`jax`
Our loop was already one compiled program, and its `psum` already ran
through NCCL, which uses the ring of :numref:`subsec_mg-ring`. Besides the
timing, the change is that we no longer write the communication. There is
no `shard_map` and no `psum`: the placement of the inputs alone makes the
step data-parallel, and the compiler writes the collectives.
:end_tab:

### Overlapping Communication with the Backward Pass

The backward pass computes gradients from the last layer to the first, so
the gradients of the last layers are complete long before those of the
first. Our loop in :numref:`subsec_mg-byhand` nevertheless waited for the
whole backward pass before it started the allreduce, and the links between
the GPUs sat idle in the meantime. The production tools group the
gradients into *buckets* of a few tens of megabytes, in the order in which
the backward pass produces them, and start the allreduce of a bucket as
soon as its last gradient exists :cite:`Li.Zhao.Varma.ea.2020`. While one
bucket travels between the GPUs, the GPUs compute the gradients of the
next (:numref:`fig_ddp_overlap`).

![Top: the allreduce waits for the whole backward pass. Bottom: the
gradients are grouped into buckets, and each bucket's allreduce starts as
soon as the backward pass has produced it. Only the last bucket runs after
the backward pass has ended.](../img/mdl-perf-ddp-overlap.svg)
:label:`fig_ddp_overlap`

The size of a bucket is a compromise between the two parts of an
allreduce's time, a fixed cost $\alpha$ per call and the bytes divided by
the bandwidth $\beta$. Small buckets start early but incur $\alpha$ many
times. A single bucket for all gradients incurs $\alpha$ once, but it waits
for the whole backward pass. Whatever the size, the last bucket holds the
gradients of the first layers, which the backward pass computes last. Its
allreduce has nothing left to overlap with, so its time adds to every step.
Overlap therefore shortens the communication of a step, but it cannot
remove it.

:begin_tab:`pytorch`
In PyTorch, `DistributedDataParallel` (DDP) implements bucketing. It
registers a hook on every parameter that marks the parameter's gradient as
ready, and it starts the allreduce of a bucket once all of the bucket's
gradients are ready. A bucket holds up to 25 MiB (the argument
`bucket_cap_mb`), and the buckets are filled in the reverse order of the
model's parameters, which is close to the order of the backward pass.
ResNet-18's 45 MB of gradients form three buckets of about 9, 26, and 9 MB.
The last of them holds the gradients of the first two thirds of the layers,
which have few channels and therefore few weights. The allreduces run on a
CUDA stream of their own, so they proceed while the backward kernels run.
:end_tab:

:begin_tab:`jax`
In JAX, the compiler implements bucketing. `jax.jit` compiles the whole
training step, the forward pass, the backward pass, and the update, into
one program, so XLA knows when each gradient is ready. It merges the many
small gradient allreduces into buffers of up to 30 MiB, its counterpart of
buckets; the flag `--xla_gpu_all_reduce_combine_threshold_bytes` sets this
limit. ResNet-18's 45 MB of gradients thus travel in two allreduces. XLA
also compiles each allreduce as an asynchronous pair of
operations, `all-reduce-start` and `all-reduce-done`, so that the GPU can
run computation that does not need the sum while the sum is in transit.
It schedules the pair apart, with computation in between, only when its
latency-hiding scheduler is on; the first cell of this section turns it
on with `--xla_gpu_enable_latency_hiding_scheduler=true`. Without it, a
profile of the step shows the gradient allreduces running on the stream
of the computation, between two of its kernels.
:end_tab:

### The Execution Model

:begin_tab:`pytorch`
In :numref:`sec_multi_gpu` one Python thread issued the kernels of every
GPU, one GPU after the other. The time that the host spends on a step then
grows with the number of GPUs, and for a model with many small kernels the
host becomes the bottleneck, as it did for LeNet. PyTorch instead starts
one process per GPU. Every process runs an ordinary training script on its
own GPU, so no GPU waits for the host to serve the others. The older
`nn.DataParallel` drives all GPUs from one process, as our loop did, and
PyTorch's documentation recommends DDP even within one machine.

The processes find each other through
`torch.distributed.init_process_group`, which connects them through NCCL,
NVIDIA's library of collective operations, and DDP sends its allreduces
through NCCL. The launcher `torchrun` starts $k$ copies of a script and
tells each copy its *rank*, its index from 0 to $k-1$, through environment
variables. A notebook is a single process, so we write each script as a
string and start it with the following helper. It collects the lines of
JSON that the processes print, process by process in the order of their
ranks. Every script begins with `init_distributed`: it pins its process to
the GPU that `torchrun` assigned to it and joins the process group.
:end_tab:

:begin_tab:`jax`
JAX runs one process that sees all GPUs, as in :numref:`sec_multi_gpu`.
`jax.jit` compiles a step into a single program, and every GPU of a mesh
runs this program on its own part of the data, so the host dispatches one
program per step however many GPUs take part. This scheme is called
*single program, multiple data* (SPMD).

In :numref:`sec_multi_gpu` the program came from a `shard_map`, in which
we wrote the per-device computation and its `psum`. Now the program is
written for one device. What makes it data-parallel is the placement of
its inputs: the parameters are replicated on every GPU of the mesh
(`P()`), and the minibatch is split along its leading axis
(`P('data')`). The compiler partitions the computation to match this
placement, so every GPU computes the gradient of
its shard. Each parameter, however, is a copy on every GPU, and the
derivative of "copy to every device" is the sum over the devices
(:numref:`sec_multi_gpu`). The compiler therefore sums every parameter's
gradient across the GPUs, which is the allreduce that we wrote by hand.
JAX's partitioner, Shardy, implements the SPMD partitioning of GSPMD
:cite:`Xu.Lee.Chen.ea.2021`.
:end_tab:

```{.python .input #multi-gpu-practice-one-program-per-gpu}
%%tab pytorch
#@save
import glob
import subprocess
import tempfile

def torchrun(script, nproc, *args, env=None):  #@save
    """Run Python source in nproc processes; return their JSON lines."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'script.py')
        with open(path, 'w') as f:
            f.write(script)
        launcher = os.path.join(os.path.dirname(sys.executable), 'torchrun')
        # Each process prints to a file of its own (--redirects 1), so that
        # the lines of different processes cannot interleave
        run = subprocess.run(
            [launcher, '--standalone', f'--nproc-per-node={nproc}',
             '--log-dir', tmp, '--redirects', '1', path, *map(str, args)],
            capture_output=True, text=True, env={**os.environ, **(env or {})})
        if run.returncode != 0:
            raise RuntimeError(run.stderr[-3000:])
        logs = sorted(glob.glob(f'{tmp}/*/attempt_0/*/stdout.log'),
                      key=lambda log: int(log.split(os.sep)[-2]))  # By rank
        return [json.loads(line) for log in logs for line in open(log)
                if line.startswith('{')]

def init_distributed():  #@save
    """Join the process group of torchrun; return the rank, the number of
    processes, and the GPU of this process."""
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group('nccl', device_id=device)
    return (torch.distributed.get_rank(), torch.distributed.get_world_size(),
            device)
```

### The Training Step

Our model is the ResNet-18 of :numref:`sec_resnet` with a stem for small
images: a 3×3 convolution without max-pooling, followed by four stages of
two residual blocks each, with He initialization as in
:numref:`sec_resnet`. We save it, since the Image Models part
(:numref:`chap_cv`) trains it on several GPUs.

```{.python .input #multi-gpu-practice-ddp-really-run-1}
%%tab pytorch
def resnet18(num_classes, in_channels=1):  #@save
    """ResNet-18 for small images: a 3x3 stem without max-pooling, He
    initialization."""
    def stage(num_channels, first=False):
        return [d2l.Residual(num_channels, use_1x1conv=not first,
                             strides=1 if first else 2),
                d2l.Residual(num_channels)]
    net = nn.Sequential(
        # No stem bias: the batch normalization after the conv removes it
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(),
        *stage(64, first=True), *stage(128), *stage(256), *stage(512),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
    with torch.no_grad():  # Create the lazy parameters of the blocks
        net.eval()(torch.zeros(1, in_channels, 32, 32))
    for m in net.modules():  # He initialization, zero biases
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net.train()
```

```{.python .input #multi-gpu-practice-ddp-really-run-1}
%%tab jax
def resnet18(num_classes, in_channels=1, *, rngs):  #@save
    """ResNet-18 for small images: a 3x3 stem without max-pooling, He
    initialization."""
    he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
    def stage(num_channels, in_channels, first=False):
        return [d2l.Residual(num_channels, use_1x1conv=not first,
                             strides=1 if first else 2,
                             in_channels=in_channels, rngs=rngs),
                d2l.Residual(num_channels, in_channels=num_channels,
                             rngs=rngs)]
    return nnx.Sequential(
        # No stem bias: the batch normalization after the conv removes it.
        # Running averages decay 0.9, two-pass variance
        nnx.Conv(in_channels, 64, (3, 3), padding='same', use_bias=False,
                 kernel_init=he, rngs=rngs),
        nnx.BatchNorm(64, momentum=0.9, use_fast_variance=False, rngs=rngs),
        nnx.relu,
        *stage(64, 64, first=True), *stage(128, 64), *stage(256, 128),
        *stage(512, 256),
        lambda X: X.mean(axis=(1, 2)),  # Global average pooling
        nnx.Linear(512, num_classes, kernel_init=he, rngs=rngs))
```

As in :numref:`sec_multi_gpu`, every GPU keeps its own copy of the
training set, here enlarged once to 64×64 pixels, which takes 983 MB in
fp32. The input pipeline then stays out of our measurements, and the only
data that cross between GPUs are gradients. We also count the gradient
bytes $N$ that every step must sum across the GPUs:

<!-- WRITER: the cell now names the gradient bytes S (structure.md notation: N counts parameters only, S the bytes of a message) and prints "11.2M parameters, S = 45 MB"; the sentence and every later $N$ meaning bytes in this file (prose, exercises 2 and 3) become $S$. -->

```{.python .input #multi-gpu-practice-the-training-step-1}
%%tab pytorch
d2l.FashionMNIST()                   # Download once, before any process runs
num_params = d2l.num_params(resnet18(10))
S = 4 * num_params                   # Gradient bytes in fp32
print(f'{num_params / 1e6:.1f}M parameters, S = {S / 1e6:.0f} MB')
```

```{.python .input #multi-gpu-practice-the-training-step-1}
%%tab jax
data = d2l.FashionMNIST()
X_train = jax.image.resize(data.train[0][..., None] / np.float32(255),
                           (len(data.train[0]), 64, 64, 1), 'bilinear')
y_train = jnp.asarray(data.train[1], jnp.int32)
num_params = d2l.num_params(resnet18(10, rngs=nnx.Rngs(d2l.get_key())))
S = 4 * num_params                   # Gradient bytes in fp32
print(f'{num_params / 1e6:.1f}M parameters, S = {S / 1e6:.0f} MB')
```

:begin_tab:`pytorch`
The training script follows. Two parts make it data-parallel:
`init_distributed`, which joins the process to the others, and the `DDP`
wrapper around the model. The rest is a single-GPU training loop. Every
process takes its own slice of each global minibatch of $kb$ examples.
All processes draw the same random permutation, because they share a seed,
and the process of rank $r$ takes the $r$-th of the $k$ slices of $b$
examples. For data that stay in host memory, `DistributedSampler` performs
the same split for a `DataLoader`. The script times its steps with
`d2l.Benchmark` (:numref:`sec_perf_model`) and prints the seconds per step.
:end_tab:

:begin_tab:`jax`
The training step itself contains no parallel code. Its argument `idx`, the
indices of the minibatch's examples, arrives split across the GPUs of the
mesh, so every GPU gathers its own examples from its copy of the training
set, and the compiler takes care of the rest. `setup` builds the model and
the optimizer and replicates them, with the training set, on the first $k$
GPUs; `minibatches` draws random minibatches and splits their indices across
the mesh. The mesh axis is of type `Auto`, which lets the compiler derive
the placement of every intermediate array, here the gathered examples, from
the placement of the inputs.
:end_tab:

```{.python .input #multi-gpu-practice-ddp-really-run-2}
%%tab pytorch
TRAIN_DDP = r'''
import contextlib, json, sys, torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from d2l import torch as d2l

rank, k, device = d2l.init_distributed()
torch.backends.cudnn.benchmark = True  # Tune convolutions for fixed shapes
# b examples per GPU and step; sync=0 skips the gradient allreduce, and
# profile=1 also reports the device time per step of NCCL and other kernels
b, num_steps, seed, sync, profile = map(int, sys.argv[1:])
torch.manual_seed(seed)                # The same permutations everywhere

data = d2l.FashionMNIST().train        # A copy on every GPU, at 64 x 64
X = F.interpolate(data.data.to(device)[:, None] / 255, 64, mode='bilinear')
y = data.targets.to(device)
model = DDP(d2l.resnet18(10).to(device), device_ids=[device])
opt = torch.optim.SGD(model.parameters(), lr=0.1)

def minibatches():
    """This process's b examples of every global minibatch of k * b."""
    while True:
        perm = torch.randperm(len(y), device=device)
        for s in range(len(y) // (k * b)):
            yield perm[(s * k + rank) * b:][:b]

def train_step(i):
    opt.zero_grad(set_to_none=True)
    # Under no_sync, which must enclose the forward pass as well, each GPU
    # keeps its own gradient
    with model.no_sync() if not sync else contextlib.nullcontext():
        F.cross_entropy(model(X[i]), y[i]).backward()
    opt.step()

batches = minibatches()
t = d2l.Benchmark(lambda: train_step(next(batches)), warmup=10,
                  repeats=num_steps).time
result = {'rank': rank, 'step_time': t}
if profile:
    kernels = d2l.kernel_times(lambda: train_step(next(batches)))
    result['comm'] = sum(ms for name, ms in kernels.items() if 'nccl' in name)
    result['compute'] = sum(kernels.values()) - result['comm']
print(json.dumps(result))
dist.destroy_process_group()
'''
```

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-1}
%%tab jax
def setup(k, key):
    """ResNet-18 with SGD and the training set, replicated on k GPUs."""
    mesh = jax.make_mesh((k,), ('data',), devices=jax.devices()[:k],
                         axis_types=(jax.sharding.AxisType.Auto,))
    replicated = NamedSharding(mesh, P())
    model = resnet18(10, rngs=nnx.Rngs(key))
    opt = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    for m in (model, opt):
        nnx.update(m, jax.device_put(nnx.state(m), replicated))
    return mesh, model, opt, jax.device_put((X_train, y_train), replicated)

def minibatches(mesh, batch_size):
    """Indices of random minibatches, split across the GPUs of the mesh."""
    rng, n = np.random.default_rng(), len(y_train)
    while True:
        perm = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            yield jax.device_put(perm[i:i + batch_size],
                                 NamedSharding(mesh, P('data')))

@nnx.jit
def train_step(model, opt, X, y, idx):
    """One SGD step on the examples idx, written for a single device."""
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(
            model(X[idx]), y[idx]).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)
    return loss
```

### A One-Step Check

Before timing anything, we check, as in :numref:`subsec_mg-byhand`, that a
step on two GPUs computes what we expect. We compare it with steps on one
GPU, from the same initialization and on the same 256 examples, in full
fp32 arithmetic so that rounding differences stay small. The layer to
watch is batch normalization (:numref:`sec_batch_norm`). It normalizes
every channel with the mean and the variance over the minibatch, and when
the minibatch is split across GPUs, these statistics can cover either one
GPU's shard or the whole minibatch.

:begin_tab:`pytorch`
The script below takes one DDP step on $k$ GPUs, each GPU on its half of
256 random images. Process 0 also keeps an unwrapped copy of the model and
takes two reference steps on its GPU: one on all 256 examples at once, and
one that accumulates the gradients of the same two halves before it
updates the parameters.
:end_tab:

:begin_tab:`jax`
We take one step on one GPU and one step on two GPUs, from the same
initialization and on the same 256 examples, and compare the changes of the
parameters.
:end_tab:

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5}
%%tab pytorch
CHECK_DDP = r'''
import copy, json, sys, torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from d2l import torch as d2l

rank, k, device = d2l.init_distributed()
torch.backends.cudnn.allow_tf32 = False  # Full fp32 convolutions
torch.manual_seed(int(sys.argv[1]))      # The same model and data everywhere
X = torch.rand(256, 1, 64, 64, device=device)
y = torch.randint(0, 10, (256,), device=device)
model = d2l.resnet18(10).to(device)
ref = copy.deepcopy(model)               # An unwrapped copy for one GPU

def update(net, shards):
    """Parameter change of one SGD step on the mean loss over the shards."""
    before = [p.detach().clone() for p in net.parameters()]
    for X_j, y_j in shards:
        (F.cross_entropy(net(X_j), y_j) / len(shards)).backward()
    with torch.no_grad():
        for p, b in zip(net.parameters(), before):
            p -= 0.1 * p.grad
    return [p.detach() - b for p, b in zip(net.parameters(), before)]

ddp = update(DDP(model, device_ids=[device]),
             [(X.chunk(k)[rank], y.chunk(k)[rank])])
if rank == 0:
    gap = lambda u: max((a - b).abs().max().item() for a, b in zip(ddp, u))
    halves = update(copy.deepcopy(ref), list(zip(X.chunk(k), y.chunk(k))))
    print(json.dumps({'largest': max(u.abs().max().item() for u in ddp),
                      'full': gap(update(ref, [(X, y)])),
                      'halves': gap(halves)}))
dist.destroy_process_group()
'''
if d2l.num_gpus() >= 2:
    r = d2l.torchrun(CHECK_DDP, 2, random.getrandbits(32))[0]
    print(f"largest update {r['largest']:.1e}\n"
          f"largest difference from one GPU on all 256: {r['full']:.1e}\n"
          f"largest difference from one GPU on two accumulated halves: "
          f"{r['halves']:.1e}")
```

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5}
%%tab jax
key = d2l.get_key()                   # One initialization for both steps
idx = np.random.default_rng().permutation(len(y_train))[:256]
updates = []
with jax.default_matmul_precision('highest'):  # Full fp32 convolutions
    for k in (1, min(2, jax.device_count())):
        mesh, model, opt, (X, y) = setup(k, key)
        before = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
        train_step(model, opt, X, y,
                   jax.device_put(idx, NamedSharding(mesh, P('data'))))
        updates.append(jax.tree.map(lambda p, b: np.asarray(p) - b,
                                    nnx.state(model, nnx.Param), before))
u1, u2 = (jax.tree.leaves(u) for u in updates)
print(f'largest update {max(np.abs(u).max() for u in u1):.1e}\n'
      f'largest difference between one and two GPUs: '
      f'{max(np.abs(a - b).max() for a, b in zip(u1, u2)):.1e}')
```

:begin_tab:`pytorch`
The DDP step agrees with the accumulated halves to rounding error, well
under one percent of the largest update, but it differs from the step on
all 256 examples by more than one percent of the largest update. DDP normalizes
each GPU's shard with the statistics of that shard, so it computes what
one GPU computes when it accumulates the gradients of the same shards
(:numref:`subsec_mp-accumulation`). The mean and the variance over 128
examples are already close to those over the whole minibatch. Over a few
examples per GPU they become noisy, and `nn.SyncBatchNorm` then computes
them over the whole minibatch, which adds collectives to every batch
normalization layer (exercise :exref:`ex_mgp-syncbn`).
:end_tab:

:begin_tab:`jax`
The two steps agree to rounding error, well under one percent of the
largest update. Under `jit`, the mean and the variance of a batch
normalization layer are means over an axis that is split across the GPUs,
so the compiler computes them over the whole minibatch and inserts an
allreduce for each. The sharded step therefore
computes the single-GPU step on all 256 examples. Lowering the step and
searching the compiled program for its allreduces shows both kinds.
`allreduce_megabytes` does the search: it finds every allreduce
instruction in the text of the compiled program and adds up the sizes of
the fp32 arrays that it returns.
:end_tab:

```{.python .input #multi-gpu-practice-a-one-step-check-1}
%%tab jax
def allreduce_megabytes(compiled):  #@save
    """Megabytes of fp32 results of each allreduce in a compiled program,
    largest first."""
    sizes = []
    for line in compiled.as_text().splitlines():
        # A synchronous all-reduce, or the -done half of an asynchronous
        # pair, whose result type lists the summed arrays only
        if re.search(r' all-reduce(-done)?\(', line):
            result = line.split(' = ')[1].split(' all-reduce')[0]
            sizes.append(sum(4 * math.prod(map(int, dims.split(',')))
                             for dims in re.findall(r'f32\[([\d,]+)\]',
                                                    result)) / 1e6)
    return sorted(sizes, reverse=True)
```

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-3}
%%tab jax
mesh, model, opt, (X, y) = setup(jax.device_count(), d2l.get_key())
idx = next(minibatches(mesh, 256 * mesh.size))
compiled = train_step.lower(model, opt, X, y, idx).compile()
sizes = allreduce_megabytes(compiled)  # One entry per allreduce
big = [m for m in sizes if m > 1]      # Gradient buffers, above 1 MB
print(f'{len(sizes)} allreduces: {len(big)} of '
      f'{" and ".join(f"{m:.1f}" for m in big)} MB, the other '
      f'{len(sizes) - len(big)} of {sum(sizes) - sum(big):.2f} MB together')
```

:begin_tab:`jax`
Two large allreduces carry almost all of the 45 MB of gradients: XLA
merged the gradients of the many parameter tensors into two buffers. The
many small allreduces carry the statistics of the batch normalization
layers, each a vector of one number per channel. They are cheap in bytes,
but each of them incurs the fixed cost $\alpha$, and each lies on the path
of the forward or the backward pass, because the next layer needs its
result.
:end_tab:

## Scaling on Four GPUs
:label:`subsec_mgp-scaling`

We measure both conventions of :numref:`subsec_mg-accounting`: weak
scaling, which holds the minibatch per GPU fixed while the number of GPUs
$k$ grows and reports examples per second, and strong scaling, which holds
the global minibatch fixed and reports how much sooner a step finishes. The
function `step_time(k, b)` returns the seconds per step on $k$
GPUs with $b$ examples per GPU, averaged over 50 steps after a warm-up.

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-4}
%%tab pytorch
def step_time(k, b, num_steps=50):
    """Seconds per DDP step on k GPUs with b examples per GPU."""
    ranks = d2l.torchrun(TRAIN_DDP, k, b, num_steps, random.getrandbits(32),
                         1, 0)          # Allreduce the gradients, no profile
    return max(r['step_time'] for r in ranks)
```

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-4}
%%tab jax
def stepper(k, b):
    """One data-parallel step on k GPUs with b examples per GPU, as a
    function without arguments."""
    mesh, model, opt, (X, y) = setup(k, d2l.get_key())
    batches = minibatches(mesh, k * b)
    return lambda: train_step(model, opt, X, y, next(batches))

def step_time(k, b, num_steps=50):
    """Seconds per data-parallel step on k GPUs with b examples per GPU."""
    return d2l.Benchmark(stepper(k, b), warmup=10, repeats=num_steps).time
```

Weak scaling first, with 256 examples per GPU:

```{.python .input #multi-gpu-practice-ddp-really-run-3}
%%tab pytorch, jax
ks = [k for k in (1, 2, 4) if k <= d2l.num_gpus()]
weak = {k: step_time(k, 256) for k in ks}
for k in ks:
    print(f'{k} GPU(s): {k * 256 / weak[k]:5.0f} examples/s, speedup '
          f'{k * weak[1] / weak[k]:.2f}, efficiency {weak[1] / weak[k]:.0%}')
```

Two GPUs process about 1.7 times as many examples per second as one, and
four GPUs a little over three times as many. The *efficiency*, the speedup divided
by $k$, falls gradually as GPUs are added. Under weak
scaling every GPU does the same work at every $k$. The efficiency is
therefore the ratio of the single-GPU step time to the step time on $k$
GPUs, and what it loses is the communication that the step does not hide.

Strong scaling holds the global minibatch at 512, so each of the $k$ GPUs
processes $512/k$ examples while the allreduce keeps its size:

```{.python .input #multi-gpu-practice-ddp-really-run-4}
%%tab pytorch, jax
strong = {k: step_time(k, 512 // k) for k in ks}
for k in ks:
    print(f'{k} GPU(s), {512 // k} examples each: speedup '
          f'{strong[1] / strong[k]:.2f}, efficiency '
          f'{strong[1] / strong[k] / k:.0%}')
d2l.plot(ks, [ks, [k * weak[1] / weak[k] for k in ks],
              [strong[1] / strong[k] for k in ks]], 'GPUs', 'speedup',
         legend=['linear', 'weak (256 per GPU)', 'strong (512 in all)'],
         fmts=('k:', '-o', '--s'))
```

At two GPUs the two conventions give about the same speedup, because 256
and 512 examples keep an RTX 4090 about equally busy. At four GPUs they
diverge. Strong scaling leaves each GPU 128 examples per step, so the
step's computation shrinks while its communication does not, and the
efficiency falls well below that of weak scaling. The same mechanism made
LeNet slower on two GPUs; here its effect is much weaker.

<!-- WRITER (Scaling): new dev numbers (GPUs 2 and 3, 2026-11-19, not captured; the capture uses GPUs 0 and 1 and four GPUs): PyTorch weak 1.82 at k = 2 (91%), strong 1.83; JAX weak 1.73 (87%), strong 1.75. The step functions changed shape but not meaning: PyTorch `step_time` passes TRAIN_DDP two more arguments (sync=1, profile=0); JAX `step_time` is built on a new `stepper(k, b)` that returns the step as a function without arguments. -->

### Predicted and Measured Communication

The cost model :eqref:`eq_dp_cost` of :numref:`subsec_mg-accounting`
predicts the time of a step on $k$ GPUs from the compute time of a shard of
$B/k$ examples, the number $m$ of allreduce calls with fixed cost $\alpha$
each, and the gradient bytes $N$ at bandwidth $\beta$:

$$
t_{\textrm{step}}(k) \approx t_{\textrm{compute}}(B/k) + m\,\alpha +
\frac{2N}{\beta}.
$$

Under weak scaling the first term does not change with $k$, so the
difference $t_{\textrm{step}}(2) - t_{\textrm{step}}(1)$ is the
communication that a step exposes. For 45 MB of gradients in a few calls,
the bandwidth term dominates. The model counts the whole allreduce, so
$2N/\beta$ is the exposed time if nothing overlaps. Overlap as in
:numref:`fig_ddp_overlap` lowers the exposed time: only the allreduce of
the last bucket remains exposed, provided that the earlier buckets finish
while the backward pass still runs.

As in :numref:`subsec_mg-accounting`, we define $\beta$ as $2N$ divided by
the time of a large allreduce on two GPUs, so $\beta$ counts the bytes
that leave and enter a GPU together. We measure it with a bare allreduce of
256 MB per GPU, the same size as there, through the library that the
training uses:

:begin_tab:`pytorch`
`ALLREDUCE` times `torch.distributed.all_reduce` through NCCL in every
process of a two-process launch.
:end_tab:

:begin_tab:`jax`
`ALLREDUCE` times the `psum` of :numref:`sec_multi_gpu` on two GPUs. NCCL
reads its configuration once per process, and we will change that
configuration below, so the measurement runs in a fresh Python process of
its own.
:end_tab:

```{.python .input #multi-gpu-practice-communication-predicted-and-measured-1}
%%tab pytorch
ALLREDUCE = r'''
import json, torch
import torch.distributed as dist
from d2l import torch as d2l

_, _, device = d2l.init_distributed()
x = torch.zeros(2**26, device=device)   # 64M floats: 256 MB
t = d2l.Benchmark(lambda: dist.all_reduce(x), repeats=10).time
print(json.dumps({'beta_link': x.nbytes / t}))  # Each GPU sends S at k = 2
dist.destroy_process_group()
'''
def allreduce_beta(env=None):
    """beta_link of a bare 256 MB allreduce on two GPUs: the bytes per
    second that each GPU sends."""
    return min(r['beta_link'] for r in d2l.torchrun(ALLREDUCE, 2, env=env))
```

```{.python .input #multi-gpu-practice-communication-predicted-and-measured-1}
%%tab jax
ALLREDUCE = r'''
import json, jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from d2l import jax as d2l

mesh = jax.make_mesh((2,), ('data',), devices=jax.devices()[:2])
x = jnp.zeros((2, 2**26), device=NamedSharding(mesh, P('data')))  # 256 MB
psum = jax.jit(jax.shard_map(lambda x: jax.lax.psum(x, 'data'), mesh=mesh,
                             in_specs=P('data'), out_specs=P('data')))
t = d2l.Benchmark(lambda: psum(x), repeats=10).time
print(json.dumps({'beta_link': x.nbytes / mesh.size / t}))  # S per GPU
'''
def allreduce_beta(env=None):
    """beta_link of a bare 256 MB allreduce on two GPUs: the bytes per
    second that each GPU sends."""
    run = subprocess.run([sys.executable, '-c', ALLREDUCE], check=True,
                         capture_output=True, text=True,
                         env={**os.environ, **(env or {})})
    return json.loads(run.stdout.splitlines()[-1])['beta_link']
```

```{.python .input #multi-gpu-practice-ddp-really-run-5}
%%tab pytorch, jax
if d2l.num_gpus() >= 2:
    beta_link = allreduce_beta()
    print(f'beta_link = {beta_link / 1e9:.2f} GB/s')
    for k in ks[1:]:
        ring = 2 * (k - 1) / k     # Bytes that each GPU sends, per byte of S
        print(f'k={k}: predicted {ring:.1f} S/beta_link = '
              f'{1e3 * ring * S / beta_link:.0f} ms, measured exposed '
              f'{1e3 * (weak[k] - weak[1]):.0f} ms per step')
```

<!-- WRITER: the cell now measures beta_link, the bytes per second that each GPU sends during a bare 256 MB allreduce at k = 2 (= S/t there; half the old beta = 2N/t), and prints for every k > 1 the prediction 2(k-1)/k * S/beta_link (eq_ring_traffic, eq_dp_cost) against the measured exposed time weak[k] - weak[1]. The equation display and the definition of beta above go (cite eq_dp_cost and beta_link of sec_multi_gpu, MGP-11); "2N/beta" becomes "S/beta_link" throughout. Dev, GPUs 2 and 3: PyTorch beta_link 3.09 GB/s, predicted 14 ms, measured exposed 11 ms; JAX 3.04 GB/s, 15 ms, 15 ms. Store (GPUs 0 and 1, old beta 4.9 GB/s, so beta_link about 2.45): predicted 18 ms, measured 15 ms (P) and 19 ms (J); k = 4 measured 23 ms (P), 30 ms (J) against 1.5 S/beta_link of about 27 ms, which the capture will print. A measured beta_link belongs to a pair of GPUs (2.43 GB/s for GPUs 0 and 1 in an earlier dev run, 3.09 for 2 and 3). -->

The measured exposed time is close to $2N/\beta$, the time of an
allreduce that nothing overlaps. Overlap hides far less of the
communication on this machine than the bucket model predicts.

:begin_tab:`pytorch`
DDP's last bucket holds 8.7 of the 45 MB of gradients, so with full overlap
a step would expose a fifth of $2N/\beta$, about 4 ms.
:end_tab:

:begin_tab:`jax`
With full overlap a step would expose only the last of XLA's two gradient
allreduces. Their sizes, about 26 and 19 MB, put that bound at about two
fifths or three fifths of $2N/\beta$, depending on which comes last.
:end_tab:

The bucket model makes three assumptions: every allreduce starts as soon
as its bucket is complete, it runs as fast as the bare benchmark, and the
backward kernels beside it run as fast as they do alone. A trace of one
step, which records when every kernel starts and ends, tests them
(exercise :exref:`ex_mgp-buckets`). On this machine the third fails: the
kernels of an allreduce take a share of the GPU's processors and memory
bandwidth, and the backward kernels beside them run more slowly.

```{.python .input #multi-gpu-practice-predicted-and-measured-communication-2}
%%tab pytorch
def profile_step(k, b, communicate=True):
    """Seconds per DDP step on k GPUs with b examples per GPU, and device
    milliseconds per step on GPU 0 of computation and of NCCL's kernels;
    without communication, every GPU steps on its own shard's gradient."""
    ranks = d2l.torchrun(TRAIN_DDP, k, b, 50, random.getrandbits(32),
                         int(communicate), 1)
    return (max(r['step_time'] for r in ranks), ranks[0]['compute'],
            ranks[0]['comm'])
```

```{.python .input #multi-gpu-practice-predicted-and-measured-communication-2}
%%tab jax
def local_stepper(k, b):
    """The step of `stepper` without communication: inside shard_map every
    GPU sees only its shard, normalizes it with the shard's own batch
    statistics, and steps on its own shard's gradient."""
    mesh, model, opt, (X, y) = setup(k, d2l.get_key())
    batches = minibatches(mesh, k * b)
    graphdef, state = nnx.split((model, opt))
    def local(state, X, y, idx):
        model, opt = nnx.merge(graphdef, state)
        train_step(model, opt, X, y, idx)
        return nnx.state((model, opt))
    # With check_vma=False autodiff adds no sum over the GPUs, and the
    # copies, which then drift apart, are returned as if they were equal
    local = jax.jit(jax.shard_map(local, mesh=mesh, check_vma=False,
                                  in_specs=(P(), P(), P(), P('data')),
                                  out_specs=P()))
    def step():
        nonlocal state
        state = local(state, X, y, next(batches))
        return state
    return step

def profile_step(k, b, communicate=True):
    """Seconds per step on k GPUs with b examples per GPU, and device
    milliseconds per step on GPU 0 of computation and of NCCL's kernels:
    the two gradient allreduces and the small allreduces of the batch
    normalization statistics."""
    step = stepper(k, b) if communicate else local_stepper(k, b)
    t = d2l.Benchmark(step, warmup=10, repeats=50).time
    kernels = d2l.kernel_times(step)
    comm = sum(ms for name, ms in kernels.items() if 'nccl' in name.lower())
    return t, sum(kernels.values()) - comm, comm
```

<!-- WRITER: new measurement (MGP-12), replaces the trace numbers of the two tab notes after it. `profile_step(k, b, communicate)` returns the seconds per step and the device milliseconds per step on GPU 0 of NCCL's kernels ("communicates") and of all other kernels ("computes"). Without communication every GPU steps on its own shard's gradient: PyTorch runs the forward and backward pass inside DDP's `no_sync()`; JAX runs `train_step` inside `shard_map` with `check_vma=False`, so autodiff inserts no sum (the compiled step contains no collective; checked in dev). Dev, GPUs 2 and 3, repeated three times with a spread of 0.5 ms or less:
PyTorch: 1 GPU 110.7 ms (computes 110.5); 2 GPUs without communication 110.2 ms (110.1); 2 GPUs 122.3 ms (computes 121.5, communicates 16.0). Two processes cost each other nothing; the allreduce kernels run 16 ms per step, almost all of it beside the backward kernels, but the computation kernels take 11 ms longer than alone, which is the whole exposed time. Overlap works as designed; the assumption that fails is the third one (the kernels beside an allreduce run as fast as alone).
JAX: 1 GPU 98.8 ms (98.6); 2 GPUs without communication 98.8 ms (98.6); 2 GPUs 115.2 ms (computes 105.3, communicates 16.9; other runs 112.8 to 113.4 ms per step). One program on two GPUs costs nothing by itself; the allreduce kernels take 17 ms and the computation beside them 7 ms more than alone, and the step grows by 16 ms: little of the allreduce is hidden, and what is hidden slows the computation.
Shared prose can state the decomposition (no cost from running two GPUs; the communication costs about S/beta_link per step); each tab reads its own split (P: hidden but slows the kernels beside it; J: mostly not hidden). Exercise ex_mgp-buckets part 4 asks for the no_sync trace that this cell now partly shows: re-aim it at when each allreduce starts and ends; its wording "with the backward pass inside model.no_sync()" is wrong as written: DDP decides in the forward pass whether to synchronize, so no_sync must enclose the forward pass too (a first dev version that wrapped only backward still ran the 16 ms allreduce). -->

```{.python .input #multi-gpu-practice-predicted-and-measured-communication-1}
%%tab pytorch, jax
if d2l.num_gpus() >= 2:
    for k, communicate in ((1, True), (2, False), (2, True)):
        t, compute, comm = profile_step(k, 256, communicate)
        name = f'{k} GPU' + ('s' if k > 1 else '') + (
            '' if communicate else ' without communication')
        print(f'{name + ":":30s}{1e3 * t:6.1f} ms per step; GPU 0 computes '
              f'{compute:6.1f} ms, communicates {comm:5.1f} ms')
```

<!-- WRITER: the two tab notes below state trace numbers that no cell prints; the cell above replaces them (see its note).
JAX tab (lib review, finding 3): the communicating JAX step also carries one small allreduce per batch-norm statistic (the -3 cell counts them), and the step without communication normalizes per shard, so the JAX "communicates" column and the difference "2 GPUs" minus "2 GPUs without communication" include those latency-bound collectives, while 2(k-1)/k S/beta_link counts gradient bytes only. The JAX tab note must say so; the PyTorch difference is the gradient allreduce alone (BN is per GPU in both PyTorch runs). -->

:begin_tab:`jax`
In the JAX rows, the step with communication also runs the small
allreduces of the batch normalization statistics, and the step without
it normalizes each shard with its own statistics. The difference between
the two steps, and the time of NCCL's kernels, therefore include those
collectives as well as the two gradient allreduces.
:end_tab:

:begin_tab:`pytorch`
In a trace of the step on two GPUs, the backward kernels take about a
tenth longer than in the same step without communication, and the
allreduces take more than twice as long as the bare benchmark. Together
they lengthen the step by about 15 ms.
:end_tab:

:begin_tab:`jax`
The first assumption fails as well. A merged buffer is complete only when
the last of its gradients is, so in XLA's schedule of the step on two GPUs
the larger allreduce starts two convolutions before the end of the
backward pass, and the smaller one after it.
:end_tab:

On a machine whose GPUs share an NVLink domain, each GPU exchanges data
with the others at 450 GB/s in each direction on the H100 (NVLink 4) and
at 900 GB/s on the B200 (NVLink 5), hundreds of times the rate measured
here (:numref:`subsec_hw-interconnects`). The allreduce of ResNet-18's
gradients then takes well under a millisecond, and the cost model predicts
near-linear scaling for this model with or without overlap.

### The Collective Transport

<!-- WRITER: beta_link is printed directly now (2.4 to 3.1 GB/s per pair; "beta/2 in each direction" goes). The NCCL version facts are in the cell's code comment and verified against the installed libraries: the string NCCL_SHM_USE_CUDA_MEMCPY is in NCCL 2.28.9 (PyTorch 2.11) and 2.30.4 (jaxlib 0.12) and absent from the NCCL 2.31.2 library; the tab notes drop the versions. Dev, GPUs 2 and 3: 3.09 -> 11.80 GB/s (3.8x, P), 3.04 -> 11.45 GB/s (3.8x, J); store, GPUs 0 and 1: 4.7x (P), 4.5x (J). Read the factor as about four to five, per pair. -->

At $k = 2$ each GPU sends and receives at $\beta/2$ in each direction,
about 2.5 GB/s. A copy between two of our GPUs, staged through host memory,
can run at up to 23 GB/s (:numref:`subsec_hw-interconnects`), about nine
times as fast. The cause lies in how NCCL moves data between GPUs that lack
peer-to-peer access. It falls back to buffers in host memory, and by
default GPU kernels copy the data into and out of these buffers with
ordinary loads and stores across PCIe. The copy engines, the units that
perform a GPU's memory copies, move the same bytes faster. An undocumented
NCCL parameter, `NCCL_SHM_USE_CUDA_MEMCPY=1`, makes NCCL use them. It
exists in the NCCL versions that our pinned frameworks bundle, and NCCL
2.31 removed it. We measure the bare allreduce with the parameter set:

```{.python .input #multi-gpu-practice-ddp-really-run-6}
%%tab pytorch, jax
if d2l.num_gpus() >= 2:
    # An undocumented parameter of NCCL's transport through host memory. It
    # exists in NCCL 2.28.9 (bundled with PyTorch 2.11) and 2.30.4 (with
    # jaxlib 0.12); the library of NCCL 2.31.2 no longer contains it
    beta_ce = allreduce_beta({'NCCL_SHM_USE_CUDA_MEMCPY': '1'})
    print(f'beta_link = {beta_link / 1e9:.2f} GB/s by default, '
          f'{beta_ce / 1e9:.2f} GB/s with the copy engines '
          f'({beta_ce / beta_link:.1f}x)')
```

The bandwidth rises nearly five-fold, and with it the predicted
communication time of ResNet-18 falls from about 18 ms to about 4 ms.

:begin_tab:`pytorch`
PyTorch 2.11 bundles NCCL 2.28.9. On our machine, DDP training with the
parameter set deadlocks within seconds, although the same collectives
complete in the bare benchmark. Exercise :exref:`ex_mgp-transport`
reproduces both outcomes under a time limit. We therefore train with
NCCL's default.
:end_tab:

:begin_tab:`jax`
jaxlib 0.12 bundles NCCL 2.30.4. Whether the training step of this
section completes with the parameter set, and how much faster it runs, is
the subject of exercise :exref:`ex_mgp-transport`. We train with NCCL's
default, because the parameter is specific to these NCCL versions and to
machines without peer-to-peer access.
:end_tab:

One environment variable changed $\beta$ nearly five-fold here. Whether such a
setting is safe has to be checked on the training run itself, not only on
a benchmark of bare collectives.

<!-- WRITER: the sharding half moved to sharding.md (sec_sharding). This "We can now" paragraph goes (MGP-18; its second sentence into the Summary), and the body ends with a one-sentence hand-off to sec_sharding. -->

We can now train a model on all GPUs of one machine with the tools that
practitioners use, predict from the cost model what a step will cost, and
check the prediction against a measurement. Our measurements also show how
much depends on the connection between the GPUs: when they exchange data
through host memory, communication sets the pace long before the
arithmetic does. :numref:`sec_fast_transformer` applies the method of this
chapter to a transformer, where precision, compilation, and data
parallelism interact.

## Summary

Production data parallelism computes what the hand-written loop computes,
but the framework writes the collectives, and the allreduce of each bucket
of gradients starts while the backward pass is still running. A one-step
check against a single GPU shows what the data-parallel step computes,
including how batch normalization treats a split minibatch. On our four
RTX 4090s ResNet-18 gains from every added GPU, but its exposed
communication is close to the cost model's time for an allreduce that
nothing overlaps: on this machine, overlap hides little of it.

## Exercises

:begin_tab:`pytorch`
1. [code] **Splitting the data.** In `TRAIN_DDP` every process takes its
   own slice of each global minibatch: all processes draw the same
   permutation, and the process of rank $r$ takes the slice that starts at
   $(sk + r)b$ in step $s$.
    1. Remove the rank from this offset, so that every process takes the
       same $b$ examples. Predict what `step_time` reports at $k = 2$
       relative to the correct script, and which global minibatch the
       averaged gradient corresponds to. Then check, with each process
       printing a checksum of its first minibatch.
    1. Restore the offset and give each process a seed of its own. Which
       property of the split is lost, and does it matter for training?
    1. Name a check that catches the first mistake and is cheap enough to
       run at every launch.
:end_tab:

:begin_tab:`jax`
1. [code] **Splitting the data.** `minibatches` places the indices with
   `P('data')`, which splits them across the GPUs of the mesh.
    1. Place them with `P()` instead, which gives every GPU all indices.
       Predict which collectives the compiled `train_step` then contains
       and how `step_time(2, 256)` compares with `step_time(1, 256)`, and
       check both.
    1. Restore `P('data')` and, inside `train_step`, reshard the gathered
       images `X[idx]` to `P(None, 'data')` with `jax.sharding.reshard`,
       which splits the height of every image across the GPUs. Which values
       must neighboring GPUs exchange for a $3 \times 3$ convolution, and
       which collective does that take? Find it in the compiled
       `train_step`.
:end_tab:

2. **Weak-scaling extrapolation.** Suppose that a step on $k$ GPUs takes
   the single-GPU step time $t_1$ plus an exposed communication time that
   is proportional to the per-GPU ring traffic of
   :numref:`subsec_mg-ring`, $2(k-1)N/k$.
    1. Express the weak-scaling efficiency $E(k)$ through a single
       constant, fit the constant to your measurement at $k = 2$, and
       predict $E(4)$. Compare your prediction with the measured $E(4)$
       and with the prediction of a model in which the exposed time is the
       same for every $k \geq 2$.
    1. Predict $E(8)$ for an eight-GPU machine of the same design as ours,
       on which every transfer between GPUs is staged through host memory
       (:numref:`subsec_hw-interconnects`). How does the total ring traffic
       through host memory grow with $k$, and in which direction should
       this move a measured $E(8)$ relative to your prediction?
    1. With gradient accumulation (:numref:`subsec_mp-accumulation`), each
       GPU runs $a$ micro-batches of 256 examples per optimizer step and
       communicates only after the last one. Starting from your measured
       $E(4)$, find the smallest $a$ that raises the efficiency at $k = 4$
       to 95%. What happens to the global minibatch, and why does that
       matter (:numref:`sec_batch_size`)?

:begin_tab:`pytorch`
3. [code] **Bucket size.** ● Model the overlap of :numref:`fig_ddp_overlap`
   as follows. The backward pass takes time $t_b$ and releases the
   gradients in $n$ buckets of equal size, bucket $i$ at time $i\,t_b/n$.
   The allreduces of the buckets run one at a time and in order, and each
   takes $\alpha + \tau/n$, where $\tau = 2N/\beta$ is the communication
   time of all gradients and $\alpha$ is a fixed cost per collective. The
   exposed communication is the time from the end of the backward pass to
   the end of the last allreduce.
    1. Show that the exposed communication is

        $$t_{\text{exposed}}(n) = \max\Bigl(\alpha + \frac{\tau}{n},\;
        n\alpha + \tau - \frac{n-1}{n}\,t_b\Bigr).$$

    1. Find the $n$ that minimizes $t_{\text{exposed}}$ when $\tau$ is well
       below $t_b$, and again when $\tau > t_b$. What does a bucket that is
       too small cost, and what does one that is too large cost?
    1. Pass `bucket_cap_mb` to `DDP` in `TRAIN_DDP`, sweep it from 1 MB to
       100 MB at $k = 2$ with `step_time`, and plot the exposed
       communication against the bucket size. Estimate $t_b$ as two thirds
       of the single-GPU step time and $\tau$ as $2N/\beta$. Is the
       measured optimum consistent with your model, and which $\alpha$ does
       it imply?
    1. At the default buckets, the exposed communication is close to
       $\tau$ rather than to the time of the last bucket. Record a few
       steps at $k = 2$ with `torch.profiler.profile`, once as they are and
       once with the backward pass inside `model.no_sync()`, which skips
       the allreduces. Compare the time from the first to the last backward
       kernel in the two runs, and find when each allreduce kernel starts
       and ends. Which assumption of the model fails on our machine?
   :exlabel:`ex_mgp-buckets`

4. [code] **Transport configuration.** ● `allreduce_beta` compares NCCL's
   default transport with the copy engines that `NCCL_SHM_USE_CUDA_MEMCPY=1`
   selects, at a single payload of 256 MB. The parameter affects only the
   transport through host memory that NCCL uses between GPUs that lack
   peer-to-peer access.
    1. Launch `ALLREDUCE` once with `NCCL_DEBUG=INFO` and find the transport
       that NCCL reports for each channel. Predict whether the parameter can
       change anything on your machine.
    1. Extend the comparison to payloads from 1 MB to 256 MB and plot the
       bandwidth of both transports against the payload. Where does each
       saturate, and does the five-fold gap hold at every size?
    1. Launch `TRAIN_DDP` on two GPUs with the parameter set (the `env`
       argument of `d2l.torchrun`), once with the default buckets and once
       with `bucket_cap_mb` large enough that all gradients form a single
       bucket, each launch under a time limit that also terminates the
       processes. Does the deadlock depend on the
       overlap of communication with the backward pass, and what does the
       outcome tell you about which workloads the bare benchmark validates?
   :exlabel:`ex_mgp-transport`
:end_tab:

:begin_tab:`jax`
3. [code] **Bucket size.** ● Model the overlap of :numref:`fig_ddp_overlap`
   as follows. The backward pass takes time $t_b$ and releases the
   gradients in $n$ buckets of equal size, bucket $i$ at time $i\,t_b/n$.
   The allreduces of the buckets run one at a time and in order, and each
   takes $\alpha + \tau/n$, where $\tau = 2N/\beta$ is the communication
   time of all gradients and $\alpha$ is a fixed cost per collective. The
   exposed communication is the time from the end of the backward pass to
   the end of the last allreduce.
    1. Show that the exposed communication is

        $$t_{\text{exposed}}(n) = \max\Bigl(\alpha + \frac{\tau}{n},\;
        n\alpha + \tau - \frac{n-1}{n}\,t_b\Bigr).$$

    1. Find the $n$ that minimizes $t_{\text{exposed}}$ when $\tau$ is well
       below $t_b$, and again when $\tau > t_b$. What does a bucket that is
       too small cost, and what does one that is too large cost?
    1. XLA merges gradient allreduces up to a byte threshold, which the
       flag `--xla_gpu_all_reduce_combine_threshold_bytes` sets. In fresh
       processes, with `XLA_FLAGS` set before JAX initializes its backend,
       sweep the threshold from 1 MB to 100 MB. For each value, count the
       gradient allreduces in the compiled `train_step` and measure the
       exposed communication `step_time(2, 256) - step_time(1, 256)`.
       Estimate $t_b$ as two thirds of the single-GPU step time and $\tau$
       as $2N/\beta$. Is the measured optimum consistent with your model,
       and which $\alpha$ does it imply?
    1. At the default threshold, the exposed communication is close to
       $\tau$ rather than to the time of the last allreduce. Record a few
       steps at $k = 1$ and $k = 2$ inside `jax.profiler.trace` and open
       the traces in Perfetto. Compare the time from the first to the last
       kernel of the backward pass in the two traces, and find when each
       gradient allreduce starts and ends. Which assumption of the model
       fails on our machine?
   :exlabel:`ex_mgp-buckets`

4. [code] **Transport configuration.** ● XLA runs its collectives through
   NCCL, so the parameter `NCCL_SHM_USE_CUDA_MEMCPY=1` applies to JAX
   programs as well. It affects only the transport through host memory that
   NCCL uses between GPUs that lack peer-to-peer access. NCCL reads its
   configuration once per process, so compare settings in fresh processes,
   as `allreduce_beta` does.
    1. Run `ALLREDUCE` once with `NCCL_DEBUG=INFO`, find the transport that
       NCCL reports for each channel, and predict whether the parameter can
       change anything on your machine.
    1. Time the `psum` for payloads from 1 MB to 256 MB under both settings
       and plot the bandwidth against the payload. Where does each
       saturate, and does the five-fold gap hold at every size?
    1. Write the training of this section into a script and run it on two
       GPUs in a fresh process with the parameter set, under a time limit.
       Does training complete, and at what speed relative to the default
       transport?
    1. Read the sizes of the allreduces in the compiled `train_step`. At
       which of them does the parameter change the bandwidth?
   :exlabel:`ex_mgp-transport`
:end_tab:

:begin_tab:`pytorch`
5. [code] **Synchronized batch normalization.** ● The one-step check of
   :numref:`subsec_mgp-production` shows that DDP normalizes each GPU's
   shard with the statistics of that shard.
    1. Convert the model in `CHECK_DDP` with
       `nn.SyncBatchNorm.convert_sync_batchnorm` after creating its
       parameters and before wrapping it in `DDP`, and repeat the check.
       Which of the two references does the DDP step now match?
    1. How many collectives per step does the conversion add to
       `resnet18`? Measure `step_time` with the converted model at $k = 2$
       and $k = 4$ under weak scaling, and compare the added time with the
       time that the added bytes take at bandwidth $\beta$. Which
       properties of these collectives, other than their size, set their
       cost?
   :exlabel:`ex_mgp-syncbn`
:end_tab:

:begin_tab:`jax`
5. [code] **Batch-norm statistics.** ● The one-step check of
   :numref:`subsec_mgp-production` shows that under `jit` the statistics
   of batch normalization are computed over the global minibatch, as on a
   single GPU.
    1. Count the small allreduces in the compiled `train_step`, and account
       for their number from the `BatchNorm` layers of `resnet18` and the
       passes in which their statistics are needed. What does each of them
       carry?
    1. Call `model.eval()` before compiling, so that every `BatchNorm`
       normalizes with its running averages, and count the allreduces
       again. Measure `step_time` with and without this change at $k = 1$,
       $k = 2$ and $k = 4$. Separate what making the statistics global
       costs per step from what computing batch statistics costs on a
       single GPU, and compare the cost of global statistics with the time
       that the added bytes take at bandwidth $\beta$. Which properties of
       these collectives, other than their size, set their cost?
   :exlabel:`ex_mgp-syncbn`
:end_tab:

[Discussions](https://d2l.discourse.group/t/36792)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.6]{.kicker}

Multi-GPU in Practice<br>**bucketing and overlap · the execution model · scaling and communication time · the collective transport · ZeRO and FSDP**
:::
:::

::: {.slide title="ResNet-18 Has the Work That LeNet Lacked; the Exchange Still Costs" only="pytorch"}
[The problem]{.kicker}

- LeNet: a second GPU made each step slower (previous deck).
- ResNet-18, 64×64 Fashion-MNIST: 2.2 G multiply-adds per image,
  45 MB of fp32 gradients, about 50 multiply-adds per gradient byte.
- Our loop: allreduce after the whole backward, one Python thread for
  all GPUs, a star through device 0.
- GPT with 36 blocks, 1.9B parameters, Adam in fp32: 29 GiB of state.

::: {.d2l-note .warn}
How close to $k\times$ does DDP get, and how do we train a model that
fits on no single card?
:::
:::

::: {.slide title="ResNet-18 Has the Work That LeNet Lacked; the Exchange Still Costs" only="jax"}
[The problem]{.kicker}

- LeNet: a second GPU made each step slower (previous deck).
- ResNet-18, 64×64 Fashion-MNIST: 2.2 G multiply-adds per image,
  45 MB of fp32 gradients, about 50 multiply-adds per gradient byte.
- Our loop: one `psum` after the whole backward, and a hand-written
  `shard_map` for every model.
- GPT with 36 blocks, 1.9B parameters, Adam in fp32: 29 GiB of state.

::: {.d2l-note .warn}
How close to $k\times$ does `jit` with placement get, and how do we
train a model that fits on no single card?
:::
:::

<!-- WRITER: deck split (worklist Part 7 Deck): the four sharding slides moved to sharding.md; the ZeRO goal, the GPT running example, "Production Tools, Then Sharding", the opening slides' GPT bullet and the ZeRO wrap-up bullets still describe the old section. The predicted-and-measured slide's formula becomes 2(k-1)/k S/beta_link (its output now prints beta_link and one line per k); the transport slide becomes "A Workaround Is Not a Fix" (its output prints beta_link); add a slide for the decomposition cell multi-gpu-practice-predicted-and-measured-communication-1 (one GPU, two without communication, two). -->

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- explain how buckets overlap the allreduce with the backward pass;
- train with production data parallelism, checked against one GPU;
- predict the exposed communication from $2N/\beta$ and test it;
- size the memory and communication of ZeRO stages 1–3.

**Running example:** ResNet-18 on our four RTX 4090s; the 36-block GPT
(1.9B parameters) for sharding.
:::

::: {.slide title="Production Tools, Then Sharding"}
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
| **Multi-GPU in Practice** | **production data parallelism, sharding** |
| Case Study: Optimizing a Transformer | the whole method on one GPT |
:::

::: {.col .narrow}
- **Uses:** the ring allreduce, the cost model, 16 bytes per
  parameter.
- **Skips:** tensor, pipeline, and expert parallelism; several machines
  (Language Models part).
:::
:::
:::

::: {.slide title="Same Algorithm, Organized Differently" only="pytorch"}
[Production data parallelism]{.kicker}

| | hand-written loop | DDP |
|---|---|---|
| allreduce starts | after the whole backward | per bucket, during the backward |
| work issued by | one Python thread for all GPUs | one process per GPU |
| allreduce algorithm | star through device 0 | ring or tree (NCCL) |

Each GPU still: full model copy, its shard of the minibatch, the average
gradient, the same update.
:::

::: {.slide title="Same Algorithm, Organized Differently" only="jax"}
[Production data parallelism]{.kicker}

| | hand-written loop | `jit` with placement |
|---|---|---|
| allreduce starts | after the whole backward | per combined buffer, scheduled by XLA |
| collectives written by | us: `shard_map` and `psum` | the compiler |
| unchanged | one jitted program, NCCL's ring | one jitted program, NCCL's ring |

Each GPU still: full model copy, its shard of the minibatch, the average
gradient, the same update.
:::

::: {.slide title="Buckets Let the Allreduce Run During the Backward Pass"}
[Overlap]{.kicker}

![](../img/mdl-perf-ddp-overlap.svg){width=86%}

Gradients appear last layer first. A bucket of tens of MB starts its
allreduce as soon as its last gradient exists.
:::

::: {.slide title="The Last Bucket Never Overlaps"}
[Overlap]{.kicker}

- Small buckets start early, but each allreduce costs a fixed $\alpha$.
- One bucket costs $\alpha$ once, but waits for the whole backward.
- The last bucket holds the first layers' gradients, computed last:
  nothing is left to hide it.

::: {.d2l-note .rule}
Overlap shortens the communication of a step; it cannot remove it.
:::
:::

::: {.slide title="DDP: One Process per GPU, Launched by torchrun" only="pytorch"}
[One process per GPU]{.kicker}

```{.python #multi-gpu-practice-exercises-1}
rank, k, device = d2l.init_distributed()   # join the process group
model = d2l.resnet18(10).to(device)
model = DDP(model, device_ids=[device])
for s in range(len(y) // (k * b)):        # this rank's slice of a
    i = perm[(s * k + rank) * b:][:b]     # global minibatch of k * b
    F.cross_entropy(model(X[i]), y[i]).backward()  # hooks: allreduce
    opt.step(); opt.zero_grad()
```

- `torchrun --nproc-per-node=k` starts $k$ processes, rank $0 \ldots k-1$.
- Buckets of up to 25 MiB: ResNet-18's 45 MB form three.
:::

::: {.slide title="JAX: Place the Inputs, and jit Writes the Allreduce" only="jax"}
[Placement]{.kicker}

```{.python #multi-gpu-practice-exercises-2}
replicated = NamedSharding(mesh, P())      # parameters, data: copies
split = NamedSharding(mesh, P('data'))     # minibatch indices: shards

@nnx.jit
def train_step(model, opt, X, y, idx):     # written for one device
    def loss_fn(model):
        return softmax_cross_entropy(model(X[idx]), y[idx]).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)               # grads summed across GPUs
```

A replicated parameter's gradient is the sum over the GPUs: the compiler
inserts the allreduce.
:::

::: {.slide title="DDP Equals One GPU Accumulating Over the Same Shards" only="pytorch"}
[A one-step check]{.kicker}

Same initialization, same 256 images, full fp32; DDP on two GPUs against
two single-GPU references:

@!multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5

- Each GPU normalizes its shard with its own batch-norm statistics.
- `nn.SyncBatchNorm`: global statistics, extra collectives per layer.
:::

::: {.slide title="The Sharded Step Equals the Single-GPU Step" only="jax"}
[A one-step check]{.kicker}

Same initialization, same 256 images, full fp32; one GPU against two:

@!multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5

- Batch-norm means over the split batch axis span all GPUs: the compiler
  computes global statistics, as one GPU would.
:::

::: {.slide title="Two Large Allreduces Carry the Gradients; Many Small Ones the Statistics" only="jax"}
[Where the collectives are]{.kicker}

@!multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-3

- XLA merges gradients into buffers of up to 30 MiB: 45 MB in two.
- Each batch-norm allreduce is tiny, but costs $\alpha$ and sits on the
  forward or backward path.
:::

::: {.slide title="Check: Forget the Rank in the Slice" only="pytorch"}
[Check]{.kicker}

Every GPU takes the same $b$ examples instead of its own slice. What do
the throughput and the gradient report, and what is wrong?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Throughput looks unchanged: $k$ GPUs, $kb$ examples counted
per step. But the average gradient is that of $b$ examples, not $kb$: the
effective minibatch is $k$ times smaller, and no error is raised.
:::
:::
:::

::: {.slide title="Check: Replicate the Indices" only="jax"}
[Check]{.kicker}

The minibatch indices are placed with `P()` instead of `P('data')`, so
every GPU receives all of them. Is the step still correct, and what does
the second GPU contribute?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Correct: every GPU computes the gradient of the whole
minibatch, so no allreduce is needed. But every GPU does all the work:
the step takes as long as on one GPU, and the speedup is 1.
:::
:::
:::

::: {.slide title="Weak Scaling: About 1.7× on Two GPUs, over 3× on Four"}
[Scaling]{.kicker}

256 examples per GPU, the global minibatch grows with $k$:

@!multi-gpu-practice-ddp-really-run-3

Efficiency $= t_{\textrm{step}}(1)/t_{\textrm{step}}(k)$: what it loses is
the communication that the step does not hide.
:::

::: {.slide title="Strong Scaling: The Global Minibatch Stays at 512"}
[Scaling]{.kicker}

@!multi-gpu-practice-ddp-really-run-4

At $k = 4$, 128 examples per GPU: compute shrinks, the allreduce does not.
:::

::: {.slide title="The Exposed Time Is Close to an Allreduce with No Overlap"}
[Predicted and measured]{.kicker}

$t_{\textrm{step}}(k) \approx t_{\textrm{compute}}(B/k) + m\,\alpha +
2N/\beta$. Weak scaling keeps the first term, so
$t_{\textrm{step}}(2) - t_{\textrm{step}}(1)$ is the exposed communication.
No overlap: $2N/\beta$. Full overlap: the last bucket only.

@!multi-gpu-practice-ddp-really-run-5

::: {.d2l-note .warn}
Overlap hides far less here than the bucket model predicts.
:::
:::

::: {.slide title="Check: The Same Step on an NVLink Machine"}
[Check]{.kicker}

$N$ = 45 MB. On our machine each GPU sends at about 2.5 GB/s; NVLink 5
moves 900 GB/s per direction. What happens to the allreduce and to the
weak-scaling efficiency?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The allreduce falls from about 18 ms to well under 1 ms,
against a step of about 100 ms: the efficiency approaches 100%, with or
without overlap. Same model, same cost model, a different $\beta$.
:::
:::
:::

::: {.slide title="One Undocumented NCCL Parameter Changes β Nearly Five-Fold"}
[The collective transport]{.kicker}

No peer-to-peer access: NCCL stages through host memory, copying with
GPU kernels by default. `NCCL_SHM_USE_CUDA_MEMCPY=1` uses the copy
engines instead:

@!multi-gpu-practice-ddp-really-run-6

::: {.d2l-note .warn}
Undocumented, removed in NCCL 2.31. We train with the default: validate
a setting on the workload, not only on the benchmark.
:::
:::

::: {.slide title="Wrap-up" only="pytorch"}
[Wrap-up]{.kicker}

- Buckets overlap allreduces with the backward; the last stays exposed.
- DDP: one process per GPU, NCCL's ring instead of our star.
- One-step check: DDP equals one GPU accumulating over the shards.
- ResNet-18, weak scaling: about 1.7× on two GPUs, over 3× on four.
- Exposed time near $2N/\beta$: overlap hides little through host memory.
- One NCCL variable: $\beta$ nearly five-fold, but DDP training deadlocks.
- ZeRO-1/2: less memory, same bytes; FSDP: $16/k$ bytes per parameter, 1.5× the traffic.

**Next:** the method of this chapter applied to a transformer, in Case
Study: Optimizing a Transformer.
:::

::: {.slide title="Wrap-up" only="jax"}
[Wrap-up]{.kicker}

- Buckets overlap allreduces with the backward; the last stays exposed.
- `jit` with placement: the compiler writes and combines the allreduces.
- One-step check: the sharded step equals the single-GPU step.
- ResNet-18, weak scaling: about 1.7× on two GPUs, over 3× on four.
- Exposed time near $2N/\beta$: overlap hides little through host memory.
- One NCCL variable: $\beta$ nearly five-fold; validate it on the training run.
- ZeRO-1/2: less memory, same bytes; FSDP: $16/k$ bytes per parameter, 1.5× the traffic.

**Next:** the method of this chapter applied to a transformer, in Case
Study: Optimizing a Transformer.
:::
