# Data Parallelism in Practice
:label:`sec_multi_gpu_concise`

In :numref:`sec_multi_gpu`, our hand-written data-parallel loop slowed LeNet
down on two GPUs for two reasons: a single Python thread dispatched kernels to
both GPUs sequentially, and the allreduce waited until the entire backward pass
had finished even though the last layers' gradients were ready much earlier.

Production data-parallel tools eliminate both bottlenecks: they either run one
host process per GPU or
compile a single sharded program across all GPUs, and they group gradients into *buckets* whose allreduces start as
soon as each bucket fills during the backward pass. To test how much of the
gradient exchange can be hidden behind backward compute, we move from LeNet to
ResNet-18 on $64 \times 64$ Fashion-MNIST images (2.2G multiply-adds per
image, ~50 FLOPs per byte of its 45 MB gradient, keeping one RTX 4090 busy for
~100 ms per 256-image step). We verify what the parallel step computes via a
one-step check and measure weak and strong scaling across one, two, and four
GPUs against our cost model.

```{.python .input #multi-gpu-practice-setup}
%%tab jax
# JAX and NCCL read these settings once, when they start, so this cell runs
# before the imports. XLA overlaps a collective with computation only when
# its latency-hiding scheduler is on. The 'vmm' allocator reserves nothing
# up front and maps each array into one contiguous range, where JAX's
# default allocator reserves most of every GPU at start-up and carves every
# array out of that block: one step below needs about 10 GB in one piece
# after many smaller arrays have come and gone, and a later cell starts a
# second JAX process on the same GPUs.
# On GPUs without peer-to-peer access, such as ours, set
# NCCL_LOCAL_REGISTER=0 in the environment before starting: NCCL's buffer
# registration fails there and prints a warning at every collective.
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'vmm')
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                           ' --xla_gpu_enable_latency_hiding_scheduler=true')
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

### The Execution Model

Every modern framework drives $k$ GPUs using one of two *single program,
multiple data* (SPMD) designs:

1. **Multi-process ($k$ copies + collective library):** A launcher starts $k$
   independent Python processes—one per GPU, each with its own *rank* in
   $\{0, \ldots, k-1\}$ and its own CPU thread—that synchronize gradients
   through NCCL. Any Python code can run inside each process.
2. **Compiler-partitioned single program:** The user writes a single-device
   step and annotates the sharding of its inputs on a device mesh; the
   compiler partitions the computation and inserts and schedules the collective
   operations automatically :cite:`Xu.Lee.Chen.ea.2021`.

:begin_tab:`pytorch`
PyTorch uses the multi-process design: `torchrun` launches $k$ processes and
`DistributedDataParallel` (DDP) wraps the model. Because a notebook is a
single process, our helper `torchrun` writes a self-contained script to a
temporary file, runs `torchrun --standalone --nproc-per-node=k`, and parses
the JSON output of each rank.
:end_tab:

:begin_tab:`jax`
JAX uses the compiler-partitioned design: `nnx.jit` compiles a single-device
training step, and placing replicated parameters (`P()`) alongside a
batch-sharded input (`P('data')`) causes the Shardy/GSPMD partitioner to
insert the gradient allreduce automatically.
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

### Overlapping Communication with the Backward Pass

Production frameworks pack gradients into *buckets* of a few tens of megabytes
in reverse layer order (the order in which backpropagation produces them) and
launch each bucket's asynchronous allreduce as soon as its last gradient is
ready :cite:`Li.Zhao.Varma.ea.2020` (:numref:`fig_ddp_overlap`).

![Top: the allreduce waits for the whole backward pass. Bottom: the
gradients are grouped into buckets, and each bucket's allreduce starts as
soon as the backward pass has produced it. Only the last bucket runs after
the backward pass has ended, and the earlier ones are hidden only if each
allreduce finishes while the backward pass is still
running.](../img/mdl-perf-ddp-overlap.svg)
:label:`fig_ddp_overlap`

Bucket size trades off per-call latency $\alpha$ against overlap: tiny buckets
pay $\alpha$ many times, while a single giant bucket cannot start until the
backward pass finishes. Under the ideal *bucket model* of overlap—assuming
inter-GPU links are fast enough to finish each bucket before the next fills
and that communication does not slow down concurrent backward kernels—every
bucket except the last is hidden behind the backward pass.

:begin_tab:`pytorch`
DDP closes the first bucket at 1 MiB and subsequent buckets at 25 MiB
(`bucket_cap_mb`). For ResNet-18's 45 MB of gradients, DDP forms three buckets
of ~9, 26, and 9 MB (the final 9 MB bucket holds the early layers, computed
last).
:end_tab:

:begin_tab:`jax`
XLA combines gradient allreduces up to a 30 MiB threshold
(`--xla_gpu_all_reduce_combine_threshold_bytes`), packing ResNet-18's 45 MB of
gradients into two asynchronous `all-reduce-start` / `all-reduce-done` pairs
scheduled concurrently with backward compute.
:end_tab:

### The Training Step

We define the small-image ResNet-18 of :numref:`sec_resnet` (`#@save`d for
:numref:`chap_cv`) and count its gradient bytes $S$ (45 MB in fp32):

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

Next, we load Fashion-MNIST and compute ResNet-18's parameter count
`num_params` and fp32 gradient volume `S`:

:begin_tab:`pytorch`
We also compute the reverse-order 1 MiB and 25 MiB gradient `buckets` that DDP
packs before launching allreduces.
:end_tab:

```{.python .input #multi-gpu-practice-the-training-step-1}
%%tab pytorch
d2l.FashionMNIST()                   # Download once, before any process runs
net = resnet18(10)
num_params = d2l.num_params(net)
S = 4 * num_params                   # Gradient bytes in fp32
# DDP's buckets: the gradients in the order in which the backward pass
# produces them, the reverse of the parameters', the first bucket closed
# at 1 MiB and every later one at 25 MiB
buckets, size = [], 0
for p in reversed(list(net.parameters())):
    size += 4 * p.numel()
    if size >= (25 if buckets else 1) * 2**20:
        buckets, size = buckets + [size], 0
buckets += [size] if size else []
last_bucket = buckets[-1]            # Bytes of the bucket sent last
print(f'{num_params / 1e6:.1f}M parameters, S = {S / 1e6:.0f} MB in buckets '
      f'of {", ".join(f"{b / 1e6:.1f}" for b in buckets)} MB')
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

We now define the data-parallel training step on $64 \times 64$ Fashion-MNIST
images:

:begin_tab:`pytorch`
`TRAIN_DDP` wraps `resnet18` in `DDP` and times `train_step` across processes
(with optional `model.no_sync()` and kernel profiling).
:end_tab:

:begin_tab:`jax`
`setup` replicates the model and dataset across the `'data'` mesh, and
`train_step` runs a jitted SGD update on sharded minibatch indices `idx`.
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

Unlike LeNet, ResNet-18 contains batch normalization (:numref:`sec_batch_norm`),
which normalizes each channel by its minibatch mean and variance. When a
minibatch of 256 examples is split across two GPUs, does each GPU normalize by
its own 128-example shard statistics (matching gradient accumulation over two
128-example micro-batches on one GPU), or do the GPUs synchronize global
256-example statistics across cards? Let us test the two-GPU update against
both single-GPU references.

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
key = d2l.get_key()                   # One initialization for every step
idx = np.random.default_rng().permutation(len(y_train))[:256]

def update(k, idx):
    """Parameter change of one step on k GPUs on the examples idx."""
    mesh, model, opt, (X, y) = setup(k, key)
    before = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
    train_step(model, opt, X, y,
               jax.device_put(idx, NamedSharding(mesh, P('data'))))
    return jax.tree.leaves(jax.tree.map(lambda p, b: np.asarray(p) - b,
                                        nnx.state(model, nnx.Param), before))

with jax.default_matmul_precision('highest'):  # Full fp32 convolutions
    two = update(min(2, jax.device_count()), idx)
    full = update(1, idx)
    # For SGD, the mean of the steps on the two halves is the step on
    # their accumulated gradients
    halves = [(a + b) / 2 for a, b in zip(*(update(1, half)
                                            for half in np.split(idx, 2)))]
gap = lambda u: max(np.abs(a - b).max() for a, b in zip(two, u))
print(f'largest update {max(np.abs(u).max() for u in two):.1e}\n'
      f'largest difference from one GPU on all 256: {gap(full):.1e}\n'
      f'largest difference from one GPU on two accumulated halves: '
      f'{gap(halves):.1e}')
```

:begin_tab:`pytorch`
By default, PyTorch DDP normalizes each GPU's shard with local per-shard
statistics—matching the two accumulated halves to ~$10^{-7}$ while differing
from full-batch statistics by several percent (converting to
`nn.SyncBatchNorm` synchronizes global statistics at the cost of extra
collectives; Exercise :exref:`ex_mgp-syncbn`).
:end_tab:

:begin_tab:`jax`
Because the batch axis is sharded across the `'data'` mesh axis in `Auto`
mode, XLA's partitioner computes batch-norm means and variances globally
across the entire minibatch—matching the single-GPU 256-example step and
inserting small allreduces for the per-channel batch-norm statistics alongside
the two large gradient buffers:
:end_tab:

```{.python .input #multi-gpu-practice-a-one-step-check-1}
%%tab jax
def allreduce_megabytes(compiled):  #@save
    """Megabytes of fp32 results of each allreduce in a compiled program,
    in the order in which the program runs them."""
    sizes = []
    for line in compiled.as_text().splitlines():
        # A synchronous all-reduce, or the -done half of an asynchronous
        # pair, whose result type lists the summed arrays only
        if re.search(r' all-reduce(-done)?\(', line):
            result = line.split(' = ')[1].split(' all-reduce')[0]
            sizes.append(sum(4 * math.prod(map(int, dims.split(',')))
                             for dims in re.findall(r'f32\[([\d,]+)\]',
                                                    result)) / 1e6)
    return sizes
```

:begin_tab:`jax`
Compiling `train_step` on the multi-GPU mesh and applying `allreduce_megabytes`
reveals two large gradient allreduces (`big`, with the final buffer recorded as
`last_bucket`) alongside the small per-channel batch-norm allreduces:
:end_tab:

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-3}
%%tab jax
mesh, model, opt, (X, y) = setup(jax.device_count(), d2l.get_key())
idx = next(minibatches(mesh, 256 * mesh.size))
compiled = train_step.lower(model, opt, X, y, idx).compile()
sizes = allreduce_megabytes(compiled)  # One entry per allreduce
big = [m for m in sizes if m > 1]      # Gradient buffers, above 1 MB
last_bucket = 1e6 * big[-1]            # Bytes of the buffer sent last
print(f'{len(sizes)} allreduces: {len(big)} of '
      f'{" and ".join(f"{m:.1f}" for m in big)} MB, the other '
      f'{len(sizes) - len(big)} of {sum(sizes) - sum(big):.2f} MB together')
```

## Scaling on Four GPUs
:label:`subsec_mgp-scaling`

We now measure weak scaling (256 examples per GPU) and strong scaling (512
examples globally) across $k \in \{1, 2, 4\}$ GPUs.

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

First we measure $\beta_\textrm{link}$ on a bare 256 MB allreduce through NCCL
and compare two predictions for the exposed communication time
$t_{\textrm{step}}(k) - t_{\textrm{step}}(1)$ under weak scaling:

1. **No overlap:** the entire ring allreduce of $S = 45$ MB is exposed
   ($\frac{2(k-1)}{k} S \beta_\textrm{link}$).
2. **Full bucket overlap:** only the final bucket/buffer ($last\_bucket$) is
   exposed after the backward pass ends.

```{.python .input #multi-gpu-practice-communication-predicted-and-measured-1}
%%tab pytorch
ALLREDUCE = r'''
import json, torch
import torch.distributed as dist
from d2l import torch as d2l

_, _, device = d2l.init_distributed()
x = torch.zeros(2**26, device=device)   # 64M floats: 256 MB
t = d2l.Benchmark(lambda: dist.all_reduce(x), repeats=10).time
print(json.dumps({'beta_link': t / x.nbytes}))  # Each GPU sends S at k = 2
dist.destroy_process_group()
'''
def allreduce_beta(env=None):
    """beta_link of a bare 256 MB allreduce on two GPUs: the seconds
    per byte that each GPU sends."""
    return max(r['beta_link'] for r in d2l.torchrun(ALLREDUCE, 2, env=env))
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
print(json.dumps({'beta_link': t / (x.nbytes / mesh.size)}))  # S per GPU
'''
def allreduce_beta(env=None):
    """beta_link of a bare 256 MB allreduce on two GPUs: the seconds
    per byte that each GPU sends."""
    run = subprocess.run([sys.executable, '-c', ALLREDUCE], check=True,
                         capture_output=True, text=True,
                         env={**os.environ, **(env or {})})
    return json.loads(run.stdout.splitlines()[-1])['beta_link']
```

Calling `allreduce_beta()` yields the measured link cost `beta_link`, from
which we compute `exposed[k]` for $k \in \{2, 4\}$ GPUs under both the
un-overlapped ($2(k-1)S\beta_\textrm{link}/k$) and full-overlap (`last_bucket`)
models:

```{.python .input #multi-gpu-practice-ddp-really-run-5}
%%tab pytorch, jax
if d2l.num_gpus() >= 2:
    beta_link = allreduce_beta()
    print(f'bus bandwidth = {1e-9 / beta_link:.2f} GB/s '
          f'(beta_link = {1e9 * beta_link:.2f} ns/byte)')
    exposed = {}                   # Predicted exposed seconds per step
    for k in (2, 4):
        ring = 2 * (k - 1) / k     # Bytes that each GPU sends, per byte of S
        exposed[k] = (ring * S * beta_link, ring * last_bucket * beta_link)
        print(f'k={k}: exposed {1e3 * exposed[k][0]:.0f} ms without overlap '
              f'({ring:.1f} S beta_link), {1e3 * exposed[k][1]:.0f} ms with '
              f'full overlap (last bucket)')
```

We now benchmark weak scaling at 256 examples per GPU (`step_time(k, 256)`)
across $k \in \{1, 2, 4\}$ GPUs and compare the measured throughput, scaling
efficiency, and exposed overhead `weak[k] - weak[1]` against both predictions:

```{.python .input #multi-gpu-practice-ddp-really-run-3}
%%tab pytorch, jax
ks = [k for k in (1, 2, 4) if k <= d2l.num_gpus()]
weak = {k: step_time(k, 256) for k in ks}
for k in ks:
    print(f'{k} GPU(s): {k * 256 / weak[k]:5.0f} examples/s, speedup '
          f'{k * weak[1] / weak[k]:.2f}, efficiency {weak[1] / weak[k]:.0%}')
    if k > 1:  # Predicted: the one-GPU step plus the exposed communication
        none, full = (weak[1] / (weak[1] + e) for e in exposed[k])
        print(f'{"":10s}predicted {none:.0%} without overlap, {full:.0%} '
              f'with full overlap; exposed {1e3 * (weak[k] - weak[1]):.0f} ms')
```

At ~2.4 GB/s bus bandwidth, an un-overlapped allreduce of 45 MB takes ~18 ms
on two GPUs and ~28 ms on four. In our weak-scaling measurements, four GPUs
achieve $3.1\text{--}3.3\times$ the throughput of one GPU (77–83% scaling
efficiency)—matching the *no-overlap* prediction closely and falling well
short of the full-overlap bucket prediction.

Under strong scaling (fixed global batch of 512 examples, so each of the $k$
GPUs processes $512/k$ examples while communication volume stays fixed),
efficiency drops further at $k = 4$:

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

### Assumptions of the Bucket Model

Why did overlap hide so little of the communication? The bucket model assumes
that (1) each bucket's allreduce starts during the backward pass, (2) NCCL
runs at its bare benchmark rate, and (3) backward compute kernels do not slow
down when NCCL runs alongside them. Profiling GPU 0 across three configurations—1
GPU, 2 GPUs without communication, and 2 GPUs with communication—tests all
three assumptions:

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

Running `profile_step` at $b = 256$ examples per GPU across the three
configurations—one GPU, two GPUs without gradient synchronization, and two GPUs
with allreduce—separates wall-clock step time into device compute and NCCL
kernel time:

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

Two GPUs without communication step at the exact same speed as one GPU, and
NCCL's kernels run for 20–22 ms (matching the 18 ms bare benchmark). What
fails is the third assumption: **when NCCL runs concurrently with the backward
pass, the compute kernels themselves slow down by 8–13 ms**. On GPUs without
peer-to-peer access, NCCL copies data through host memory using its own GPU
SM kernels, which occupy streaming multiprocessors and memory bandwidth that
the backward pass would otherwise use.

An undocumented NCCL flag (`NCCL_SHM_USE_CUDA_MEMCPY=1`, present in
NCCL 2.28–2.30) can offload those host-memory copies to the GPU's DMA copy
engines in bare microbenchmarks:

:begin_tab:`pytorch`
The same flag deadlocks DDP when its allreduces overlap with backward compute.
:end_tab:

```{.python .input #multi-gpu-practice-ddp-really-run-6}
%%tab pytorch, jax
if d2l.num_gpus() >= 2:
    # An undocumented parameter of NCCL's transport through host memory. It
    # exists in NCCL 2.28.9 (bundled with PyTorch 2.11) and 2.30.4 (with
    # jaxlib 0.12); the library of NCCL 2.31.2 no longer contains it
    beta_ce = allreduce_beta({'NCCL_SHM_USE_CUDA_MEMCPY': '1'})
    print(f'bus bandwidth {1e-9 / beta_link:.2f} GB/s by default, '
          f'{1e-9 / beta_ce:.2f} GB/s with the copy engines '
          f'({beta_link / beta_ce:.1f}x)')
```

On data-center GPUs connected by NVLink (450–900 GB/s per direction,
:numref:`subsec_hw-interconnects`), ResNet-18's 45 MB allreduce takes well
under a millisecond and achieves near-linear scaling. Even with ideal scaling,
however, data parallelism still replicates all $16N$ bytes of parameter and
optimizer state on every card; :numref:`sec_sharding` partitions that state
across GPUs.

## Summary

Production data parallelism drives $k$ GPUs either via $k$ independent
processes synchronized through NCCL or via a single
compiler-partitioned program, grouping gradients
into buckets of 25–30 MiB to overlap allreduces with the backward pass. A
one-step check against single-GPU references reveals whether batch
normalization uses per-shard statistics or global
statistics via extra collectives. On our four RTX 4090s (communicating through host memory at
2.4 GB/s bus bandwidth), four GPUs deliver $3.1\text{--}3.3\times$ single-GPU
throughput under weak scaling, and kernel profiling shows that SM contention
from host-staged NCCL kernels claws back most of the time saved by overlapping
communication with backward computation.

## Exercises

:begin_tab:`pytorch`
1. [code] **Splitting the data.** In `TRAIN_DDP`, rank $r$ takes the slice
   starting at $(sk + r)b$ in step $s$ from a shared random permutation.
    1. Remove `rank` from this offset so every process takes the same $b$
       examples. Predict how `step_time` at $k = 2$ and the averaged gradient
       behave, and verify by printing a minibatch checksum on each rank.
    1. Restore the offset but give each rank a different random seed. Which
       property of the minibatch split is lost?
    1. Propose a lightweight runtime assertion that catches identical data
       shards across ranks on the first step.
:end_tab:

:begin_tab:`jax`
1. [code] **Splitting the data.** In `minibatches`, indices are placed with
   `P('data')`.
    1. Place them with `P()` instead so every GPU receives all indices.
       Predict which collectives the compiled `train_step` contains and how
       `step_time(2, 256)` compares with `step_time(1, 256)`, then check both.
    1. Restore `P('data')` and inside `train_step` constrain `X[idx]` to
       `NamedSharding(mesh, P(None, 'data'))`, splitting image height across
       GPUs. Which boundary values must neighboring GPUs exchange for a
       $3 \times 3$ convolution, and which collective appears in the compiled
       HLO?
:end_tab:

2. **Weak-scaling extrapolation.** Suppose a step on $k$ GPUs takes single-GPU time $t_1$ plus exposed communication proportional to the ring traffic $2(k-1)S/k$.
    1. Express the weak-scaling efficiency $E(k) = t_1 / t_{\textrm{step}}(k)$ in terms of a single constant, calibrate it on the measured $E(2)$, and compare its prediction for $E(4)$ with the measured value in :numref:`subsec_mgp-scaling`.
    1. With gradient accumulation over $a$ micro-batches of 256 examples per GPU (communicating only on the last micro-batch), find the smallest $a$ that raises $E(4)$ to 95%, and state how the global batch size changes.

:begin_tab:`pytorch`
3. [code] **Bucket size.** ● Model bucket overlap (:numref:`fig_ddp_overlap`)
   by assuming a backward pass of duration $t_b$ releases $n$ equal buckets at
   times $i\,t_b/n$ ($i = 1, \ldots, n$), and each bucket's allreduce takes
   $\alpha + \tau/n$ sequentially, where $\tau = S\,\beta_\textrm{link}$.
    1. Show that the exposed communication after the backward pass finishes is
       $$t_{\textrm{exposed}}(n) = \max\Bigl(\alpha + \frac{\tau}{n},\;
       n\alpha + \tau - \frac{n-1}{n}\,t_b\Bigr),$$
       and find the optimal $n$ when $\tau < t_b$ versus $\tau > t_b$.
    1. Sweep `bucket_cap_mb` in `DDP` from 1 MB to 100 MB at $k = 2$ and
       compare the measured exposed communication against the model.
   :exlabel:`ex_mgp-buckets`

4. [code] **Synchronized batch normalization.** ● The one-step check of
   :numref:`subsec_mgp-production` showed that DDP normalizes each shard with
   local statistics by default.
    1. Convert `model` in `CHECK_DDP` with
       `nn.SyncBatchNorm.convert_sync_batchnorm` before wrapping in `DDP` and
       repeat the check. Which single-GPU reference does it match now?
    1. Count the extra collectives per step added by `SyncBatchNorm`, measure
       `step_time` at $k = 2$ and $k = 4$, and explain why their time exceeds
       their byte volume times $\beta_\textrm{link}$.
   :exlabel:`ex_mgp-syncbn`
:end_tab:

:begin_tab:`jax`
3. [code] **Bucket size.** ● Model bucket overlap (:numref:`fig_ddp_overlap`)
   by assuming a backward pass of duration $t_b$ releases $n$ equal buckets at
   times $i\,t_b/n$ ($i = 1, \ldots, n$), and each bucket's allreduce takes
   $\alpha + \tau/n$ sequentially, where $\tau = S\,\beta_\textrm{link}$.
    1. Show that the exposed communication after the backward pass finishes is
       $$t_{\textrm{exposed}}(n) = \max\Bigl(\alpha + \frac{\tau}{n},\;
       n\alpha + \tau - \frac{n-1}{n}\,t_b\Bigr),$$
       and find the optimal $n$ when $\tau < t_b$ versus $\tau > t_b$.
    1. Sweep `--xla_gpu_all_reduce_combine_threshold_bytes` from 1 MB to
       100 MB in fresh processes, count the resulting gradient allreduces, and
       compare the measured exposed communication against the model.
   :exlabel:`ex_mgp-buckets`

4. [code] **Batch-norm statistics.** ● Under `nnx.jit` with `Auto` mesh axes,
   batch normalization computes global statistics across all GPUs.
    1. Account for the 74 small allreduces in `allreduce_megabytes(compiled)`
       from the number of `BatchNorm` layers in `resnet18` and the forward and
       backward passes.
    1. Call `model.eval()` before compiling so `BatchNorm` uses running
       averages instead of batch statistics, verify that the small allreduces
       disappear, and measure how much step time they cost at $k = 1, 2, 4$.
   :exlabel:`ex_mgp-syncbn`
:end_tab:

[Discussions](https://d2l.discourse.group/t/36792)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.6]{.kicker}

Distributed Data Parallel (DDP) in Practice: Bucketing and Overlap<br>**multi-process vs. compiler SPMD · gradient bucketing · backward overlap · BatchNorm semantics · scaling & SM contention**
:::
:::

::: {.slide title="From Toy LeNet to Compute-Heavy ResNet-18" only="pytorch"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why LeNet stalled in §13.5:** One Python thread dispatched sequentially to both GPUs ($k \cdot h_{\textrm{step}}$), and the allreduce waited until `backward()` finished.
- **Production DDP remedies:** Spawns **1 process per GPU** (`torchrun`), launching bucketed allreduces *during* `backward()`.
- **ResNet-18 benchmark ($64 \times 64$, $b=256$):** Keeps each RTX 4090 busy for $\sim 100\text{ ms/step}$ across $11.2\text{M}$ params ($S = 45\text{ MB}$), exposing true communication overlap.
:::

::: {.col .fig .big}
@fig:ch13-perf-lenet-vs-resnet18
:::
:::
:::

::: {.slide title="From Toy LeNet to Compute-Heavy ResNet-18" only="jax"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why LeNet stalled in §13.5:** One Python thread dispatched sequentially to both GPUs ($k \cdot h_{\textrm{step}}$), and the allreduce waited until `backward()` finished.
- **Compiled data-parallel remedy:** Compiles **1 multi-GPU program** (`jax.jit`) whose XLA-inserted allreduces run *during* the backward pass.
- **ResNet-18 benchmark ($64 \times 64$, $b=256$):** Keeps each RTX 4090 busy for $\sim 100\text{ ms/step}$ across $11.2\text{M}$ params ($S = 45\text{ MB}$), exposing true communication overlap.
:::

::: {.col .fig .big}
@fig:ch13-perf-lenet-vs-resnet18
:::
:::
:::

::: {.slide title="Multi-Process SPMD: One Python Process per GPU" only="pytorch"}
[Execution Models]{.kicker}

::: {.cols .vc}
::: {.col}
- **Multi-Process SPMD (`torchrun` + `DDP`):** Spawns $k$ independent Python processes (1 per GPU); autograd hooks fire asynchronous NCCL bucket allreduces during `backward()`.
:::

::: {.col .fig .big}
@fig:ch13-perf-spmd-models@pytorch
:::
:::
:::

::: {.slide title="Compiler-Partitioned SPMD: One Program for k GPUs" only="jax"}
[Execution Models]{.kicker}

::: {.cols .vc}
::: {.col}
- **Compiler-Partitioned SPMD (`jax.jit` + `Mesh`):** One process traces a single-device function annotated with `NamedSharding`; XLA (GSPMD/Shardy) inserts and overlaps NCCL collectives in HLO.
:::

::: {.col .fig .big}
@fig:ch13-perf-spmd-models@jax
:::
:::
:::

::: {.slide title="How Production DDP Fixes the Hand-Written Loop" only="pytorch"}
[PyTorch DDP]{.kicker}

::: {.cols .vc}
::: {.col}
- **Parallel host dispatch ($1 \cdot h_{\textrm{step}}$):** Each of the $k$ `torchrun` ranks drives its own GPU concurrently, eliminating the $k \cdot h_{\textrm{step}}$ single-thread bottleneck.
- **Ring/Tree NCCL collectives:** Replaces the $O(kS)$ hub-bottlenecked star gather/scatter with bandwidth-optimal $\frac{2(k-1)}{k}S$ NCCL collectives.
- **Pipelined backward hooks:** Fires asynchronous allreduces on $25\text{ MiB}$ reverse-order buckets while earlier layers are still backpropagating.
:::

::: {.col .fig .big}
@fig:ch13-perf-torchrun-ddp
:::
:::
:::

::: {.slide title="How Compiler Partitioning Automates Data Parallelism" only="jax"}
[JAX Sharded JIT]{.kicker}

::: {.cols .vc}
::: {.col}
- **Single-device user code:** You write a standard `train_step(model, opt, X, y, idx)` without manual `shard_map`, `ravel_pytree`, or `lax.psum`.
- **Automatic collective synthesis:** Marking weights replicated (`P()`) and batch indices sharded (`P('data')`) tells XLA to insert gradient allreduces automatically.
- **30 MiB asynchronous combining:** XLA merges allreduces up to $30\text{ MiB}$ and schedules backward compute between `all-reduce-start` and `all-reduce-done`.
:::

::: {.col .fig .big}
@fig:ch13-perf-jax-mesh-sharding
:::
:::
:::

::: {.slide title="One Process per GPU with torchrun and DDP" only="pytorch"}
[Multi-Process Workflow]{.kicker}

`torchrun --standalone --nproc-per-node=k` spawns $k$ processes; each binds to its assigned GPU via `LOCAL_RANK` and wraps the model in `DDP`:

```{.python #multi-gpu-practice-exercises-1}
rank, k, device = d2l.init_distributed()   # join the NCCL process group
model = d2l.resnet18(10).to(device)
model = DDP(model, device_ids=[device])    # broadcasts weights from rank 0
for s in range(len(y) // (k * b)):         # rank r takes slice (s*k + r)*b
    i = perm[(s * k + rank) * b:][:b]      # disjoint shard of global batch k*b
    F.cross_entropy(model(X[i]), y[i]).backward()  # hooks launch bucket allreduces
    opt.step(); opt.zero_grad(set_to_none=True)
```

::: {.d2l-note .rule}
DDP broadcasts initial parameters and buffers from rank $0$ at construction time and registers autograd hooks on every parameter to fire bucketed NCCL allreduces during `backward()`.
:::
:::

::: {.slide title="Declarative Mesh Placement with NamedSharding" only="jax"}
[Mesh Sharding Workflow]{.kicker}

In JAX, we write `train_step` as if for a single GPU, replicate the model (`P()`), and shard the minibatch indices (`P('data')`) across an `Auto` mesh:

```{.python #multi-gpu-practice-exercises-2}
replicated = NamedSharding(mesh, P())      # parameters & dataset: replicated
split = NamedSharding(mesh, P('data'))     # minibatch indices: sharded across GPUs

@nnx.jit
def train_step(model, opt, X, y, idx):     # written for a single logical device
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(
            model(X[idx]), y[idx]).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)               # XLA inserts gradient allreduces
```

::: {.d2l-note .rule}
Because `model` is replicated (`P()`) while `idx` is sharded (`P('data')`), differentiating the mean loss over `'data'` automatically requires an allreduce across `'data'` to keep the updated parameters replicated.
:::
:::

::: {.slide title="Overlapping Bucketed Allreduces with Backpropagation"}
[Communication Overlap]{.kicker}

![](../img/mdl-perf-ddp-overlap.svg){width=84%}

- **Reverse-order gradient production:** Backpropagation computes gradients from the final output layer back toward the first input layer.
- **Pipelining communication and compute:** Packing gradients into **buckets** of $25\text{--}30\text{ MiB}$ in reverse parameter order lets each bucket's allreduce run on a separate communication stream while earlier layers are still backpropagating.
:::

::: {.slide title="The Bucket Size Trade-Off and the Ideal Bucket Model"}
[Bucket Model]{.kicker}

::: {.cols .vc}
::: {.col}
$$t_{\textrm{exposed (ideal)}} \;=\; \frac{2(k-1)}{k}\, S_{\textrm{last\_bucket}}\, \beta_{\textrm{link}}$$

- **Tiny buckets ($c \gg 1$) vs. 1 giant bucket ($c = 1$):** Tiny buckets pay latency $c\alpha$ dozens of times; one giant bucket waits until the entire backward pass finishes.
- **25–30 MiB sweet spot:** Under ideal overlap (fast links, zero SM contention), every bucket except the final bucket $S_{\textrm{last}}$ is hidden behind backpropagation.
:::

::: {.col .fig}
@fig:ch13-perf-bucket-tradeoff
:::
:::
:::

::: {.slide title="Reverse-Order Bucketing in PyTorch DDP (1 & 25 MiB)" only="pytorch"}
[DDP Buckets]{.kicker}

PyTorch DDP closes the first bucket at $1\text{ MiB}$ (so communication starts almost immediately) and subsequent buckets at `bucket_cap_mb = 25` ($25\text{ MiB} = 26.2\text{ MB}$):

- **ResNet-18 bucket layout ($S = 44.7\text{ MB}$ total):** Walking `reversed(list(net.parameters()))` forms three buckets of **$9.4\text{ MB}$, $26.2\text{ MB}$, and $9.0\text{ MB}$** (a single $9.4\text{ MB}$ late-stage `Conv2d` weight tensor overshoots the $1\text{ MiB}$ first cap on its own).
- **Ideal overlap prediction:** If the first two buckets ($9.4$ and $26.2\text{ MB}$) finish behind backpropagation, only the final **$S_{\textrm{last}} = 9.0\text{ MB}$** bucket (holding the earliest layers, computed last) remains exposed—cutting exposed communication by **$5\times$**!
:::

::: {.slide title="Asynchronous Allreduce Combining in XLA (30 MiB)" only="jax"}
[XLA Collective Combining]{.kicker}

In JAX, XLA's `all-reduce-combiner` pass merges gradient allreduces up to a $30\text{ MiB}$ threshold (`--xla_gpu_all_reduce_combine_threshold_bytes`):

- **ResNet-18 buffer layout ($S = 44.7\text{ MB}$ total):** XLA packs the parameter gradients into two asynchronous `all-reduce-start` / `all-reduce-done` pairs of **$25.8\text{ MB}$** and **$18.9\text{ MB}$**, scheduling backward convolutions between `start` and `done`.
- **Ideal overlap prediction:** If the first $25.8\text{ MB}$ buffer finishes while earlier layers backpropagate, only the final **$S_{\textrm{last}} = 18.9\text{ MB}$** buffer remains exposed after backward compute ends.
:::

::: {.slide title="One-Step Check: Per-Shard BatchNorm in PyTorch DDP" only="pytorch"}
[BatchNorm Semantics]{.kicker}

Unlike LeNet, ResNet-18 contains `BatchNorm2d` layers. Comparing one 2-GPU DDP step on $256$ images against (a) one GPU on all $256$ images and (b) one GPU accumulating two $128$-image halves reveals what `BatchNorm2d` computes:

@!multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5

- **Local per-shard statistics by default:** PyTorch DDP normalizes each GPU's $128$-image shard using its **own local mean and variance**, matching the two accumulated halves to $10^{-7}$ while differing from global 256-image BatchNorm by $\sim 6\%$ of the update!
:::

::: {.slide title="One-Step Check: Global BatchNorm Under JAX Auto Mesh" only="jax"}
[BatchNorm Semantics]{.kicker}

Comparing one 2-GPU `nnx.jit` step on $256$ images against (a) one GPU on all $256$ images and (b) one GPU accumulating two $128$-image halves reveals which statistics BatchNorm uses:

@!multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5

- **Global batch statistics under `Auto` sharding:** Because the batch dimension is sharded across `'data'` in `AxisType.Auto` mode, XLA partitions the BatchNorm mean and variance reductions across **all GPUs globally**, matching the single-GPU 256-image update to `float32` rounding ($\sim 10^{-7}$)!
:::

::: {.slide title="Local BatchNorm vs. SyncBatchNorm Trade-Offs" only="pytorch"}
[Synchronizing Statistics]{.kicker}

When does per-shard BatchNorm matter, and what does synchronizing it cost?

- **When local BatchNorm works well:** When the per-GPU shard size $b = B/k$ is large ($b \ge 32\text{--}64$ images/GPU), per-shard means and variances are low-variance estimates and avoid any forward-pass communication barrier.
- **When `nn.SyncBatchNorm` is needed:** In object detection or segmentation with tiny per-GPU batches ($b = 1\text{--}4$ high-resolution images/GPU), local statistics are too noisy; `nn.SyncBatchNorm.convert_sync_batchnorm(model)` inserts an allreduce at **every BatchNorm layer** during the forward and backward passes (adding dozens of latency-bound $\alpha$ synchronization points).
:::

::: {.slide title="Inspecting Emitted Collectives in the Compiled HLO" only="jax"}
[Compiler Collectives]{.kicker}

Parsing the compiled HLO text (`compiled.as_text()`) with `allreduce_megabytes` confirms both the two combined gradient buffers and the global BatchNorm reductions:

@!multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-3

- **Two large buffers + 74 tiny calls:** The two large allreduces ($25.8\text{ MB}$ and $18.9\text{ MB}$) carry the $44.7\text{ MB}$ of parameter gradients; the $74$ small allreduces ($0.24\text{ MB}$ total) synchronize per-channel BatchNorm means, variances, and activation gradients across all 20 BatchNorm layers in the forward and backward passes.
:::

::: {.slide title="Silent Data-Sharding Bugs in Distributed Training"}
[Practical Pitfalls]{.kicker}

::: {.cols .vc}
::: {.col}
- **Bug 1 — Identical shards across ranks:** Omitting `rank` in the slice offset (or sharding indices with `P()`) makes all $k$ GPUs compute on the exact same $b$ examples—wasting $k-1$ GPUs!
- **Bug 2 — Unseeded per-rank shuffles:** Independent random permutations sample examples *with replacement* across ranks instead of partitioning the epoch.
- **Step-1 defense:** Assert that minibatch checksums differ across ranks while sharing the epoch seed.
:::

::: {.col .fig .big}
@fig:ch13-perf-data-shard-bugs
:::
:::
:::

::: {.slide title="Bounding Exposed Communication: No Overlap vs. Bucket"}
[Paper Predictions]{.kicker}

First, we measure $\beta_{\textrm{link}}$ on a bare $256\text{ MB}$ NCCL allreduce ($\sim 2.4\text{ GB/s}$ bus bandwidth, or $\beta_{\textrm{link}} \approx 0.42\text{ ns/B}$) and compute our two theoretical bounds on exposed communication $t_{\textrm{step}}(k) - t_{\textrm{step}}(1)$ under weak scaling:

@!multi-gpu-practice-ddp-really-run-5

- **Two clear predictions:** Without overlap, exposed communication should be **$\sim 18\text{ ms}$** ($k=2$) and **$\sim 28\text{ ms}$** ($k=4$); with ideal bucket overlap, only the last bucket/buffer should be exposed (**$4\text{--}8\text{ ms}$** at $k=2$, **$6\text{--}12\text{ ms}$** at $k=4$).
:::

::: {.slide title="Weak Scaling on Four GPUs: 3.1–3.3× Throughput"}
[Weak Scaling Measured]{.kicker}

Fixing the per-GPU batch size at $b = 256$ images/GPU across $k \in \{1, 2, 4\}$ GPUs measures weak-scaling throughput and exposed communication $t_{\textrm{step}}(k) - t_{\textrm{step}}(1)$:

@!multi-gpu-practice-ddp-really-run-3

::: {.d2l-note .warn}
Four GPUs achieve **$3.1\text{--}3.3\times$ single-GPU throughput** ($77\text{--}83\%$ scaling efficiency), and the measured exposed communication matches the **no-overlap prediction** ($\sim 18\text{ ms}$ at $k=2$, $\sim 28\text{ ms}$ at $k=4$) rather than the ideal bucket model!
:::
:::

::: {.slide title="Strong Scaling on Four GPUs: Shrinking Compute Share"}
[Strong Scaling Measured]{.kicker}

Fixing the global batch size at $B = 512$ images ($b = 512/k$ per GPU) measures strong scaling across $k \in \{1, 2, 4\}$ GPUs:

@!multi-gpu-practice-ddp-really-run-4

- **Why strong scaling drops off faster at $k = 4$:** At $k = 4$, each GPU processes only $128$ images (halving local backward compute time relative to $b = 256$), while the $45\text{ MB}$ gradient allreduce volume $\frac{2(k-1)}{k}S = 1.5\,S$ stays full-sized.
:::

::: {.slide title="Why Did Overlap Hide So Little? Profiling GPU 0"}
[Testing Bucket Assumptions]{.kicker}

To discover why exposed communication matched the no-overlap formula even though buckets launched during `backward()`, we profile GPU 0 across three runs: (1) 1 GPU, (2) 2 GPUs without communication (`no_sync` / `shard_map`), and (3) 2 GPUs with full communication:

@!multi-gpu-practice-predicted-and-measured-communication-1

::: {.d2l-note .rule}
**Assumption 3 of the bucket model failed:** While NCCL's communication kernels overlap in time with backpropagation, **the backward compute kernels running alongside NCCL slow down by $8\text{--}13\text{ ms}$**, clawing back most of the time saved by overlap!
:::
:::

::: {.slide title="SM Contention From Host-Staged NCCL Copy Kernels" only="pytorch"}
[Transport Bottleneck]{.kicker}

Why does running an NCCL allreduce slow down concurrent convolution kernels on our workstation?

- **How NCCL moves data without peer-to-peer (P2P):** Because our consumer RTX 4090s lack NVLink and PCIe P2P, NCCL stages every chunk through shared host DRAM (`SHM`) by running **dedicated GPU SM kernels** that read/write over PCIe and sum chunks on the SMs.
- **Resource contention:** Those NCCL SM kernels compete directly with backward convolution and GEMM kernels for **Streaming Multiprocessors** and **DRAM memory bandwidth**.

@!multi-gpu-practice-ddp-really-run-6

::: {.d2l-note .warn}
Offloading host-staged copies to DMA copy engines (`NCCL_SHM_USE_CUDA_MEMCPY=1`) boosts bare allreduce bandwidth $4\text{--}5\times$, but deadlocks PyTorch DDP when overlapped with backward compute—always verify transport flags on the actual training loop!
:::
:::

::: {.slide title="SM Contention From Host-Staged NCCL Copy Kernels" only="jax"}
[Transport Bottleneck]{.kicker}

Why does running an NCCL allreduce slow down concurrent convolution kernels on our workstation?

- **How NCCL moves data without peer-to-peer (P2P):** Because our consumer RTX 4090s lack NVLink and PCIe P2P, NCCL stages every chunk through shared host DRAM (`SHM`) by running **dedicated GPU SM kernels** that read/write over PCIe and sum chunks on the SMs.
- **Resource contention:** Those NCCL SM kernels compete directly with backward convolution and GEMM kernels for **Streaming Multiprocessors** and **DRAM memory bandwidth**.

@!multi-gpu-practice-ddp-really-run-6

::: {.d2l-note .warn}
Offloading host-staged copies to DMA copy engines (`NCCL_SHM_USE_CUDA_MEMCPY=1`) boosts bare allreduce bandwidth $4\text{--}5\times$, but a bare microbenchmark says nothing about the overlapped training loop—always verify transport flags on the actual training loop!
:::
:::

::: {.slide title="PCIe Host Staging vs. NVLink Fabrics"}
[Interconnect Impact]{.kicker}

::: {.cols .vc}
::: {.col}
- **Consumer PCIe 4.0 via Host DRAM ($\sim 2.4\text{ GB/s}$ bus):** A $45\text{ MB}$ allreduce at $k=4$ takes $\sim 28\text{ ms}$ and competes for SMs, yielding $77\text{--}83\%$ weak-scaling efficiency ($3.1\text{--}3.3\times$).
- **Data-Center NVLink 4/5 ($450\text{--}900\text{ GB/s}$):** The same $45\text{ MB}$ allreduce finishes in $< 0.2\text{ ms}$ via dedicated fabric engines, achieving $>99\%$ scaling efficiency ($\approx 4.0\times$).
- **Next wall:** Even on NVLink, DDP replicates all $16N$ bytes of state on every GPU (§13.7).
:::

::: {.col .fig .big}
@fig:ch13-perf-pcie-vs-nvlink-ddp
:::
:::
:::

::: {.slide title="Concept Check: Bucketing, BatchNorm, and Scaling" only="pytorch"}
[Check Your Understanding]{.kicker}

Test your grasp of production data parallelism with three diagnostic questions:

1. In what parameter order does DDP pack gradients into buckets, and why would packing them in forward layer order ($1, 2, \ldots, L$) defeat communication overlap?
2. On our four RTX 4090s, weak scaling at $b = 256$ images/GPU gave $E(4) = t_1 / t_{\textrm{step}}(4) \approx 80\%$ efficiency ($t_1 \approx 100\text{ ms}$ compute + $28\text{ ms}$ exposed allreduce). If we accumulate gradients over $a = 5$ micro-batches of $256$ images per GPU before running the allreduce (`model.no_sync()` on the first $4$ micro-batches), what does the weak-scaling efficiency $E(4)$ become?
3. When you run a one-step check comparing 2-GPU training against 1-GPU training on a network with `BatchNorm`, which single-GPU reference does default `DDP` match?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Reverse parameter order ($L, L-1, \ldots, 1$):** Because `backward()` produces gradients from layer $L$ back to layer $1$. A forward-ordered bucket containing layer $1$ could not launch until the very end of `backward()`.
2. **$\mathbf{E(4) \approx 95\%}$:** Local compute grows $5\times$ to $5 \times 100 = 500\text{ ms}$ while the single allreduce stays at $28\text{ ms}$, giving $E(4) = \frac{500}{500 + 28} \approx 94.7\%$ (at the cost of a $5\times$ larger global batch $B = 5 \times 4 \times 256 = 5{,}120$).
3. **`DDP`** uses per-shard statistics by default (matching **1 GPU accumulating two $B/2$ halves**).
:::
:::

::: {.slide title="Concept Check: Bucketing, BatchNorm, and Scaling" only="jax"}
[Check Your Understanding]{.kicker}

Test your grasp of production data parallelism with three diagnostic questions:

1. In what parameter order should gradients be packed into buckets, and why would packing them in forward layer order ($1, 2, \ldots, L$) defeat communication overlap?
2. On our four RTX 4090s, weak scaling at $b = 256$ images/GPU gave $E(4) = t_1 / t_{\textrm{step}}(4) \approx 80\%$ efficiency ($t_1 \approx 100\text{ ms}$ compute + $28\text{ ms}$ exposed allreduce). If we accumulate gradients over $a = 5$ micro-batches of $256$ images per GPU before running the allreduce, what does the weak-scaling efficiency $E(4)$ become?
3. When you run a one-step check comparing 2-GPU training against 1-GPU training on a network with `BatchNorm`, which single-GPU reference does `Auto` sharding match?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Reverse parameter order ($L, L-1, \ldots, 1$):** Because `backward()` produces gradients from layer $L$ back to layer $1$. A forward-ordered bucket containing layer $1$ could not launch until the very end of `backward()`.
2. **$\mathbf{E(4) \approx 95\%}$:** Local compute grows $5\times$ to $5 \times 100 = 500\text{ ms}$ while the single allreduce stays at $28\text{ ms}$, giving $E(4) = \frac{500}{500 + 28} \approx 94.7\%$ (at the cost of a $5\times$ larger global batch $B = 5 \times 4 \times 256 = 5{,}120$).
3. **`Auto` sharding** inserts cross-GPU reductions for BatchNorm statistics (matching **1 GPU on the full batch $B$**).
:::
:::

::: {.slide title="Summary: Production Data Parallelism and Overlap" only="pytorch"}
[Key Takeaways]{.kicker}

::: {.cols .vc}
::: {.col}
- **SPMD Execution:** `torchrun` + `DDP` ($k$ processes) eliminates the single-thread $k \cdot h_{\textrm{step}}$ bottleneck.
- **Reverse-Order Bucketing:** Packing gradients into $25\text{--}30\text{ MiB}$ reverse-order buckets pipelines NCCL allreduces with backpropagation.
- **Verification & Contention:** Always run a 1-step check (catching BatchNorm and sharding bugs) and profile SM contention on host-staged PCIe links.
:::

::: {.col .fig .big}
@fig:ch13-perf-ddp-summary@pytorch
:::
:::
:::

::: {.slide title="Summary: Production Data Parallelism and Overlap" only="jax"}
[Key Takeaways]{.kicker}

::: {.cols .vc}
::: {.col}
- **SPMD Execution:** `jax.jit` + `NamedSharding` (1 compiled program) eliminates the single-thread $k \cdot h_{\textrm{step}}$ bottleneck.
- **Reverse-Order Bucketing:** Packing gradients into $25\text{--}30\text{ MiB}$ reverse-order buckets pipelines NCCL allreduces with backpropagation.
- **Verification & Contention:** Always run a 1-step check (catching BatchNorm and sharding bugs) and profile SM contention on host-staged PCIe links.
:::

::: {.col .fig .big}
@fig:ch13-perf-ddp-summary@jax
:::
:::
:::
