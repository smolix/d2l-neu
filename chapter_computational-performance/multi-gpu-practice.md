# Data Parallelism in Practice
:label:`sec_multi_gpu_concise`

In :numref:`sec_multi_gpu`, our hand-written data-parallel loop slowed LeNet
down on two GPUs for two reasons: a single Python thread dispatched kernels to
both GPUs sequentially, and the allreduce waited until the entire backward pass
had finished even though the last layers' gradients were ready much earlier.

Production data-parallel tools eliminate both bottlenecks: they either run one
host process per GPU (`torchrun` + `DistributedDataParallel` in PyTorch) or
compile a single sharded program across all GPUs (`jax.jit` + `NamedSharding`
in JAX), and they group gradients into *buckets* whose allreduces start as
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

Even though an undocumented NCCL flag (`NCCL_SHM_USE_CUDA_MEMCPY=1`, present in
NCCL 2.28–2.30) can offload those host-memory copies to the GPU's DMA copy
engines in bare microbenchmarks, it deadlocks PyTorch DDP when overlapped with
backward compute:

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
processes synchronized through NCCL (`torchrun` + DDP) or via a single
compiler-partitioned program (`jax.jit` + mesh shardings), grouping gradients
into buckets of 25–30 MiB to overlap allreduces with the backward pass. A
one-step check against single-GPU references reveals whether batch
normalization uses per-shard statistics (PyTorch DDP by default) or global
statistics via extra collectives (JAX `Auto` sharding and PyTorch
`SyncBatchNorm`). On our four RTX 4090s (communicating through host memory at
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

Data Parallelism in Practice<br>**one process or one program · buckets and overlap · a one-step check · scaling on four GPUs · where the time goes**
:::
:::

::: {.slide title="LeNet Lost on Two GPUs; ResNet-18 Has About 30 Times Its Work per Gradient Byte"}
[The problem]{.kicker}

- LeNet (previous deck): a second GPU slowed each step, because one
  Python thread fed both GPUs and the steps were tiny.
- Our loop sent no gradient before the whole backward pass ended.
- ResNet-18 at 64×64: 2.2 G multiply-adds per image, $S$ = 45 MB of
  gradients, about 50 per byte (LeNet: under 2).
- A step on 256 images keeps an RTX 4090 busy for about 0.1 s: plenty
  of time for any host.

::: {.d2l-note .warn}
How close to $k\times$ the throughput do the production tools get, and
how much of the exchange does overlap hide?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- name the two ways to drive $k$ GPUs, what each needs, and when the
  host's organization matters;
- explain how buckets overlap the allreduce with the backward pass, and
  when they cannot;
- check a data-parallel step against one GPU, batch norm included;
- predict the exposed communication from $S\,\beta_\textrm{link}$
  ($\beta_\textrm{link}$: the time per byte that each GPU sends in an
  allreduce), and split the measured time into its causes.

**Running example:** ResNet-18 on our four RTX 4090s, which have no
peer-to-peer access. Its 45 MB of gradients are small enough to study
overlap; the chapter's GPT returns in the next deck.
:::

::: {.slide title="The Hand-Written Loop, Now with Production Tools"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Performance Model | roofline, regimes, timing, profiles |
| Hardware | where peak arithmetic and bandwidth come from |
| Computation Graphs and Compilation | fusion and kernel replay |
| Memory and Precision | bytes per parameter and per token |
| Multi-GPU from First Principles | data parallelism by hand, cost model |
| **Data Parallelism in Practice** | **production tools on four GPUs** |
| Sharding the Training State | training state split across GPUs |
| Case Study: Optimizing a Transformer | the whole method on one GPT |
:::

::: {.col .narrow}
- **Uses:** the ring allreduce; $\alpha$ and $\beta_\textrm{link}$;
  batch normalization.
- **Skips:** sharding (next deck); tensor, pipeline, and expert
  parallelism; several machines (Language Models part).
:::
:::
:::

::: {.slide title="DDP Runs Our Algorithm with a Process per GPU and Early Allreduces" only="pytorch"}
[Production data parallelism]{.kicker}

| | hand-written loop | DDP |
|---|---|---|
| work issued by | one Python thread for all GPUs | one process per GPU |
| allreduce starts | after the whole backward | per bucket, during the backward |
| allreduce algorithm | star through device 0 | ring or tree (NCCL) |

::: {.d2l-note .rule}
Each GPU still holds a full model copy, takes its shard of the
minibatch, and applies the average gradient.
:::
:::

::: {.slide title="jit with Placement Runs Our Algorithm, and the Compiler Writes the Collectives" only="jax"}
[Production data parallelism]{.kicker}

| | hand-written loop | `jit` with placement |
|---|---|---|
| collectives written by | us: `shard_map` and `psum` | the compiler |
| allreduce starts | after the whole backward | per combined buffer, scheduled by XLA |
| unchanged | one jitted program, NCCL's ring | one jitted program, NCCL's ring |

::: {.d2l-note .rule}
Each GPU still holds a full model copy, takes its shard of the
minibatch, and applies the average gradient.
:::
:::

::: {.slide title="Two Ways to Drive k GPUs: k Copies of a Program, or One Partitioned Program"}
[The execution model]{.kicker}

| | $k$ copies + collective library | one program, compiler-partitioned |
|---|---|---|
| host | one thread per GPU | one dispatch per step |
| code | any Python | traced, static shapes |
| communication | explicit calls, or a wrapper | inserted and scheduled by the compiler |
| started by | a launcher that assigns ranks | one process that sees all GPUs |

Both are *single program, multiple data* (SPMD).

::: {.d2l-note .rule}
They differ in who partitions the program and writes the communication.
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

::: {.d2l-note .rule}
`torchrun --nproc-per-node=k script.py` starts $k$ processes, ranks
$0 \ldots k-1$; each script string in the notebook runs this way.
:::
:::

::: {.slide title="JAX: Place the Inputs, and jit Writes the Allreduce" only="jax"}
[Placement]{.kicker}

```{.python #multi-gpu-practice-exercises-2}
replicated = NamedSharding(mesh, P())      # parameters, data: copies
split = NamedSharding(mesh, P('data'))     # minibatch indices: shards

@nnx.jit
def train_step(model, opt, X, y, idx):     # written for one device
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(
            model(X[idx]), y[idx]).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)               # grads summed across GPUs
```

::: {.d2l-note .rule}
A replicated parameter's gradient is the sum over the GPUs, so the
compiler inserts the allreduce.
:::
:::

::: {.slide title="Buckets Let the Allreduce Run During the Backward Pass"}
[Overlap]{.kicker}

![](../img/mdl-perf-ddp-overlap.svg){width=86%}

::: {.d2l-note .rule}
Gradients appear last layer first. A bucket of tens of MB starts its
allreduce as soon as its last gradient exists.
:::
:::

::: {.slide title="Overlap Hides All but the Last Bucket, and Only If the Links Keep Up"}
[Overlap]{.kicker}

- Small buckets start early, but each allreduce costs a fixed $\alpha$.
- One bucket costs $\alpha$ once, but waits for the whole backward.
- The last bucket holds the first layers' gradients, computed last.
- Slow links: allreduces queue up and run on after the backward ends.

The *bucket model*: every allreduce but the last runs unseen beside the
backward pass.

::: {.d2l-note .rule}
Overlap shortens the communication of a step; it cannot remove it.
:::
:::

::: {.slide title="DDP's Last Bucket Holds the First Two Thirds of the Layers, 9 MB" only="pytorch"}
[Overlap in DDP]{.kicker}

- Caps: 1 MiB for the first bucket, 25 MiB (26.2 MB) for the rest; a
  bucket closes once it reaches its cap, so it can overshoot by one
  tensor.
- ResNet-18: 9, 26, 9 MB. A 9 MB convolution weight overshoots the first
  cap on its own.
- The allreduces run on a CUDA stream of their own.

::: {.d2l-note .rule}
With full overlap, only the last 9 MB would be exposed: about 4 ms here.
:::
:::

::: {.slide title="XLA Merges the Gradients into Two Buffers of up to 30 MiB" only="jax"}
[Overlap in XLA]{.kicker}

- One compiled step: XLA knows when each gradient is ready.
- Small allreduces merged up to 30 MiB: ResNet-18's 45 MB in two.
- Each allreduce is a `start`/`done` pair; the latency-hiding scheduler
  places computation between them.

::: {.d2l-note .rule}
With full overlap, only the last buffer, 19 MB, would be exposed: about
8 ms here.
:::
:::

::: {.slide title="DDP Normalizes Each GPU's Shard with Its Own Statistics" only="pytorch"}
[A one-step check]{.kicker}

Same initialization, same 256 images, full fp32; DDP on two GPUs against
two single-GPU references:

@!multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5

- DDP = one GPU accumulating the two halves: shard statistics.
- `nn.SyncBatchNorm`: global statistics, extra collectives per layer.

::: {.d2l-note .rule}
Both are valid batch normalization; nothing warns you which one you got.
:::
:::

::: {.slide title="Under jit, Batch Norm Uses Global Statistics" only="jax"}
[A one-step check]{.kicker}

Same initialization, same 256 images, full fp32; two GPUs against two
single-GPU references:

@!multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5

- Matches one GPU on all 256 to the size of a change in summation order;
  per-shard statistics would miss by a tenth of the update or more.
- Why: batch-norm means over the split batch axis span all GPUs, so the
  compiler inserts allreduces for them.

::: {.d2l-note .rule}
The compiled program, on the next slide, shows those allreduces.
:::
:::

::: {.slide title="Two Large Allreduces Carry the Gradients; Many Small Ones the Statistics" only="jax"}
[Where the collectives are]{.kicker}

@!multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-3

- Two buffers: almost all of the 45 MB of gradients.
- 74 small calls: per-channel vectors, a few per call, mostly batch-norm
  statistics. Their presence shows that the statistics are global.

::: {.d2l-note .rule}
Each batch-norm allreduce is tiny, but costs $\alpha$ and sits on the
forward or backward path.
:::
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
the step takes as long as one GPU takes on the whole minibatch, and the
weak-scaling speedup is 1.
:::
:::
:::

::: {.slide title="Check: Predict the Exposed Time with No Overlap"}
[Check]{.kicker}

$S$ = 44.7 MB of gradients. Each GPU sends $2(k-1)/k \cdot S$ in a ring,
at a bus bandwidth of about 2.4 GB/s on our machine. How long is the
allreduce at $k = 2$ and at $k = 4$, if nothing overlaps?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $44.7/2.4 \approx 18.6$ ms at $k = 2$; $1.5 \times 18.6
\approx 28$ ms at $k = 4$. With full overlap, only the last bucket's
share.
:::
:::
:::

::: {.slide title="Two Predictions: No Overlap, or Only the Last Bucket"}
[Predicted]{.kicker}

Weak scaling keeps the computation per GPU, so
$t_{\textrm{step}}(k) - t_{\textrm{step}}(1)$ is the exposed
communication. No overlap: $\frac{2(k-1)}{k} S\,\beta_\textrm{link}$;
full overlap: the same for the last bucket alone.

@!multi-gpu-practice-ddp-really-run-5
:::

::: {.slide title="Weak Scaling: Four GPUs Do the Work of a Little Over Three"}
[Scaling]{.kicker}

256 examples per GPU; efficiency $= t_{\textrm{step}}(1)/t_{\textrm{step}}(k)$,
predicted as $t_{\textrm{step}}(1)/(t_{\textrm{step}}(1) + \textrm{exposed})$:

@!multi-gpu-practice-ddp-really-run-3

::: {.d2l-note .warn}
The exposed time is close to an allreduce with no overlap: overlap hides
far less here than the bucket model predicts.
:::
:::

::: {.slide title="Strong Scaling: A Run with a Fixed Minibatch of 512 Finishes Less Than 3× Sooner"}
[Scaling]{.kicker}

@!multi-gpu-practice-ddp-really-run-4

::: {.d2l-note .rule}
At $k = 4$, 128 examples per GPU: the computation shrinks, the allreduce
does not.
:::
:::

::: {.slide title="Two GPUs Cost Nothing by Themselves; Computation Slows Beside an Allreduce"}
[Where the time goes]{.kicker}

One GPU; two GPUs that skip the allreduce; two GPUs. Device time on GPU 0:

@!multi-gpu-practice-predicted-and-measured-communication-1

- NCCL's kernels: about $S\,\beta_\textrm{link}$, as predicted.
- The kernels beside them run slower: NCCL copies through host memory with
  GPU kernels of its own.

::: {.d2l-note .rule}
The time that overlap hides comes back, mostly, as slower computation.
:::
:::

::: {.slide title="A Workaround Is Not a Fix"}
[The collective transport]{.kicker}

No peer-to-peer access: NCCL stages through host memory, copying with GPU
kernels. `NCCL_SHM_USE_CUDA_MEMCPY=1` uses the copy engines instead:

@!multi-gpu-practice-ddp-really-run-6

::: {.d2l-note .warn}
Undocumented and since removed. A setting that speeds up a bare
benchmark must be checked on the training run.
:::
:::

::: {.slide title="Check: The Same Step on an NVLink Machine"}
[Check]{.kicker}

$S$ = 45 MB. On our machine each GPU sends at about 2.4 GB/s; NVLink 5
moves 900 GB/s per direction. What happens to the allreduce and to the
weak-scaling efficiency?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The allreduce falls from about 18 ms to well under 1 ms,
against a step of about 100 ms: the cost model predicts an efficiency
near 100%, with or without overlap. Same model, same cost model, a
different $\beta_\textrm{link}$.
:::
:::
:::

::: {.slide title="Wrap-up" only="pytorch"}
[Wrap-up]{.kicker}

- DDP: one process per GPU, NCCL's ring instead of our star.
- Buckets overlap allreduces with the backward; the last stays exposed.
- One-step check: DDP equals one GPU accumulating over the shards.
- ResNet-18, weak scaling: 1.8× on two GPUs, 3.3× on four.
- Exposed time near $S\,\beta_\textrm{link}$: the allreduce overlaps,
  but the computation beside it slows down.
- Copy engines: bus bandwidth four to five times higher, but
  DDP training deadlocks.

**Next:** every GPU still holds the whole training state, 16 bytes per
parameter for Adam in fp32. Sharding the Training State divides it among
the GPUs.
:::

::: {.slide title="Wrap-up" only="jax"}
[Wrap-up]{.kicker}

- `jit` with placement: one program; the compiler writes and merges the
  allreduces.
- Buckets overlap allreduces with the backward; the last stays exposed.
- One-step check and compiled program: global batch-norm statistics.
- ResNet-18, weak scaling: 1.7× on two GPUs, 3.1× on four.
- Exposed time near $S\,\beta_\textrm{link}$: half of it runs alone, and
  the computation beside the rest slows down.
- Copy engines: bus bandwidth four to five times higher; check
  it on the training run.

**Next:** every GPU still holds the whole training state, 16 bytes per
parameter for Adam in fp32. Sharding the Training State divides it among
the GPUs.
:::
