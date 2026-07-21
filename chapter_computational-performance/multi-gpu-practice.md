# Multi-GPU in Practice
:label:`sec_multi_gpu_concise`

The hand-rolled loop of :numref:`sec_multi_gpu` taught the mechanism and
then lost the race: on a tiny model over a host-staged wire, the second
GPU made things slower. Two things were wrong, and only one was the
hardware. This section fixes the other — the *software* — by replacing our
loop with the production machinery, and in doing so meets the chapter's
sharpest framework contrast: PyTorch makes you launch processes and the
collectives are explicit; JAX runs in one process and you merely *annotate
the layout*, letting the compiler write the collectives for you. We
measure real scaling on 2–4 GPUs, sketch how the same ideas shard a model
too big to replicate (FSDP), and stop at the edge of the single node.

*Prerequisites: the from-scratch data-parallel loop, the ring-allreduce
identity, and the cost model of* :numref:`sec_multi_gpu`*; the memory
anatomy of* :numref:`sec_memory_precision`*. The multi-process idiom below
was verified to run under this book's notebook build; why this box's
fabric is slow is the topology story of* :numref:`subsec_hw-interconnects`*,
and what a collective costs on it is* :numref:`sec_multi_gpu`*'s
measurement.*

```{.python .input #multi-gpu-practice-multi-gpu-in-practice}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import torch
from torch import nn
from torchvision import datasets

torch.set_float32_matmul_precision('high')
```

```{.python .input #multi-gpu-practice-multi-gpu-in-practice}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
import optax
import os
import re
import time

# XLA's NCCL buffer registration fails harmlessly on this P2P-less box;
# opt out rather than let every collective print the warning.
os.environ.setdefault('NCCL_LOCAL_REGISTER', '0')
```

## What Our Hand-Rolled Loop Lacked
:label:`subsec_mgp-lacked`

Our :numref:`sec_multi_gpu` implementation had three deficits, and modern
data parallelism repairs each.

* **No overlap.** Our loop finished the *entire* backward pass, then called
  `allreduce`. But gradients become available layer by layer, back to
  front, so the last layers' gradients could be communicated while the
  earlier layers are still computing. Serializing compute-then-communicate
  wastes exactly the time the fabric is idle during backward.
* **One Python process.** A single interpreter drove all $k$ GPUs, so one
  GIL-bound thread dispatched every kernel — the overhead regime of
  :numref:`sec_perf_model`, multiplied by $k$.
* **A star topology.** Our `allreduce` funneled everything through device
  0; :numref:`subsec_mg-ring` showed the ring moves a constant per device
  instead.

PyTorch's `DistributedDataParallel` (DDP) fixes all three
:cite:`Li.Zhao.Varma.ea.2020`: one **process per GPU** (no shared GIL),
NCCL's **ring/tree collectives** (no hub), and — the headline —
gradient **bucketing that overlaps communication with the backward pass**
(:numref:`fig_ddp_overlap`). As each bucket of gradients fills, DDP kicks
off its allreduce immediately, so by the time the backward pass reaches the
first layer, the last layers' gradients are already summed. This is
compute–communication overlap — independent work scheduled onto the fabric
while the GPUs keep computing — finally shown where it pays.

![What DDP buys over the hand-rolled loop: gradient bucketing lets each
bucket's allreduce overlap the rest of the backward pass, instead of
waiting for all of it.](../img/mdl-perf-ddp-overlap.svg)
:label:`fig_ddp_overlap`

## DDP, Really Run
:label:`subsec_mgp-ddp`

DDP needs multiple processes, and a notebook is one process — so we launch
the extra ones. The idiom, verified to work under this book's build: write
the training script to a sidecar file, then launch it with `torchrun`,
which spawns one process per GPU, sets up the rendezvous, and runs the
script under `init_process_group`. Each rank writes its result to a scratch
directory the notebook reads back (cleared before every launch, so a
crashed run can never serve stale numbers). We keep the script minimal; the
*loop body* is unchanged from single-GPU training, and the scaffolding
around it is what the launcher's multi-process world requires:

```{.python .input #multi-gpu-practice-ddp-really-run-1}
%%tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

```{.python .input #multi-gpu-practice-ddp-really-run-1}
%%tab jax
#@save
class ResNet18(nnx.Module):
    """A slightly modified ResNet-18 (small stem, no max-pool)."""
    def __init__(self, num_classes=10, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Sequential(
            nnx.Conv(1, 64, (3, 3), (1, 1), padding='same', rngs=rngs),
            nnx.BatchNorm(64, rngs=rngs), nnx.relu,
            d2l.Residual(64, in_channels=64, rngs=rngs),
            d2l.Residual(64, in_channels=64, rngs=rngs),
            d2l.Residual(128, use_1x1conv=True, strides=(2, 2),
                         in_channels=64, rngs=rngs),
            d2l.Residual(128, in_channels=128, rngs=rngs),
            d2l.Residual(256, use_1x1conv=True, strides=(2, 2),
                         in_channels=128, rngs=rngs),
            d2l.Residual(256, in_channels=256, rngs=rngs),
            d2l.Residual(512, use_1x1conv=True, strides=(2, 2),
                         in_channels=256, rngs=rngs),
            d2l.Residual(512, in_channels=512, rngs=rngs),
            lambda x: x.mean(axis=(1, 2)),
            nnx.Linear(512, num_classes, rngs=rngs))

    def __call__(self, x):
        return self.net(x)
```

The DDP training script, written to disk from a cell. The two lines that
make training data-parallel are `init_process_group("nccl")` once per
process and the `DDP(model)` wrap — after those, *the loop body is
unchanged from single-GPU*. The rest is the launcher's housekeeping, shown
in full so none of it is magic: a rank-local device, a
`DistributedSampler` that hands each rank a disjoint shard of the data
(re-shuffled per epoch by `set_epoch`), and process-group teardown. One
practicality earns its own line: the parent notebook downloads
Fashion-MNIST once, quietly, *before* any rank exists, so the ranks never
race on the download:

```{.python .input #multi-gpu-practice-ddp-really-run-2}
%%tab pytorch
with contextlib.redirect_stdout(io.StringIO()), \
     contextlib.redirect_stderr(io.StringIO()):     # download once, quietly
    datasets.FashionMNIST('./data', train=True, download=True)

DDP_SCRIPT = r'''
import json, os, sys, time, torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from d2l import torch as d2l

rank = int(os.environ["LOCAL_RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dist.init_process_group("nccl")
torch.set_float32_matmul_precision("high"); torch.backends.cudnn.benchmark = True

model = d2l.resnet18(10, 1).to(rank)
model(torch.zeros(1, 1, 64, 64, device=rank))     # materialize Lazy params
model = DDP(model, device_ids=[rank])
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss = nn.CrossEntropyLoss()

B = int(sys.argv[1]) if len(sys.argv) > 1 else 256   # per-rank batch size
tf = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
ds = datasets.FashionMNIST("./data", train=True, transform=tf)
sampler = torch.utils.data.distributed.DistributedSampler(ds, world, rank)
loader = torch.utils.data.DataLoader(ds, B, sampler=sampler, num_workers=2)

for epoch in range(2):                             # epoch 0 warms up
    sampler.set_epoch(epoch); n = 0; torch.cuda.synchronize(); t0 = time.time()
    for X, y in loader:
        X, y = X.to(rank), y.to(rank)
        opt.zero_grad(set_to_none=True)
        loss(model(X), y).backward(); opt.step(); n += X.shape[0]
    torch.cuda.synchronize(); dt = time.time() - t0
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, f"rank{rank}.json"), "w") as f:
    json.dump({"rank": rank, "samples_per_s": n / dt}, f)
dist.destroy_process_group()
'''
DDP_DIR = pathlib.Path('ddp_scratch')
DDP_DIR.mkdir(exist_ok=True)
_ = (DDP_DIR / 'train_ddp.py').write_text(DDP_SCRIPT)
```

One word before the numbers, because every published speedup quietly picks
a convention (:numref:`subsec_mg-accounting`). This sweep holds the
*per-rank* batch fixed at 256, so the global batch grows to $256k$ —
**weak scaling**: the efficiencies it prints are throughput per GPU
relative to one GPU. Remember that a grown global batch also changes the
optimization trajectory itself (:numref:`sec_batch_size`); the
strong-scaling question — same global batch, finishing sooner — comes
right after.

```{.python .input #multi-gpu-practice-ddp-really-run-3}
%%tab pytorch
def ddp_throughput(k, batch_size=256):
    """Launch k-process DDP via torchrun; return aggregate samples/s."""
    for stale in DDP_DIR.glob('rank*.json'):   # a crashed run must not
        stale.unlink()                         # serve old results
    torchrun = str(pathlib.Path(sys.executable).parent / 'torchrun')
    subprocess.run([torchrun, '--standalone', f'--nproc-per-node={k}',
                    str(DDP_DIR / 'train_ddp.py'), str(batch_size)],
                   check=True)
    per_rank = []
    for r in range(k):
        with open(DDP_DIR / f'rank{r}.json') as f:
            per_rank.append(json.load(f)['samples_per_s'])
    return sum(per_rank)

ks = [k for k in (1, 2, 4) if k <= d2l.num_gpus()]
tput = [ddp_throughput(k) for k in ks]
for k, t in zip(ks, tput):
    print(f'{k} GPU(s): {t:.0f} samples/s, {t / tput[0]:.2f}x, '
          f'{100 * t / tput[0] / k:.0f}% weak-scaling efficiency')
d2l.plot(ks, [tput], 'GPUs', 'samples/s')
```

On our four-GPU box, ResNet-18 on Fashion-MNIST-64 (11.2M parameters)
scales the way the accounting of :numref:`sec_multi_gpu` says a
*compute-dense* model should: about 1.8× at two GPUs and about 3.3× at
four — the cell prints 88% and 82% weak-scaling efficiency. The efficiency
sags gently — no cliff — because each step's compute is large enough to
hide most of the communication, unlike LeNet. Strong scaling asks a harder
question of the same hardware: hold the global batch at 512 and split it
ever thinner, so the per-rank batch shrinks as $512/k$ while the allreduce
stays the same size:

```{.python .input #multi-gpu-practice-ddp-really-run-4}
%%tab pytorch
tput_strong = [ddp_throughput(k, batch_size=512 // k) for k in ks]
for k, t in zip(ks, tput_strong):
    print(f'{k} GPU(s): {t:.0f} samples/s, {t / tput_strong[0]:.2f}x, '
          f'{100 * t / tput_strong[0] / k:.0f}% strong-scaling efficiency')
d2l.plot(ks, [tput, tput_strong], 'GPUs', 'samples/s',
         legend=['weak (per-rank 256)', 'strong (global 512)'])
```

At two GPUs the two conventions nearly coincide on this model — a 4090 is
about equally saturated by a per-rank batch of 256 or 512, so the $k=1$
baselines barely differ — but they part company as $k$ grows: by $k=4$ the
strong-scaling run gives each GPU only 128 examples per step, and any gap
that opens between the strong and weak efficiencies is pure
underutilization, the mechanism that sank LeNet, now in miniature. That is
:eqref:`eq_dp_cost` read twice: weak scaling holds $t_{\text{compute}}$
per device constant and asks the fabric to keep up; strong scaling shrinks
$t_{\text{compute}}(B/k)$ toward the fixed $t_{\text{comm}}$ floor.

Now confront the sweep with the cost model by measurement rather than by
assertion. :eqref:`eq_dp_cost` prices a step's communication at
$2N/\beta$, and DDP itself provides the instrument to check it:
`no_sync()` runs backward with gradient synchronization turned off, so the
difference between synced and unsynced step times estimates what
communication really costs — an estimate, not a truth, since turning sync
off also changes what overlaps what:

```{.python .input #multi-gpu-practice-ddp-really-run-5}
%%tab pytorch
COMM_SCRIPT = r'''
import contextlib, json, os, time, torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from d2l import torch as d2l

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
torch.set_float32_matmul_precision("high"); torch.backends.cudnn.benchmark = True

model = d2l.resnet18(10, 1).to(rank)
model(torch.zeros(1, 1, 64, 64, device=rank))
model = DDP(model, device_ids=[rank])
loss = nn.CrossEntropyLoss()
X = torch.randn(256, 1, 64, 64, device=rank)
y = torch.randint(0, 10, (256,), device=rank)

def fwd_bwd():
    model.zero_grad(set_to_none=True)
    loss(model(X), y).backward()

def step_time(sync_grads, steps=30):
    ctx = contextlib.nullcontext if sync_grads else model.no_sync
    for _ in range(10):                            # warmup
        with ctx(): fwd_bwd()
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    for _ in range(steps):
        with ctx(): fwd_bwd()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps

synced, silent = step_time(True), step_time(False)
if rank == 0:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "comm.json"), "w") as f:
        json.dump({"synced_ms": 1e3 * synced, "no_sync_ms": 1e3 * silent}, f)
dist.destroy_process_group()
'''
_ = (DDP_DIR / 'comm_probe.py').write_text(COMM_SCRIPT)

net = resnet18(10, 1)                     # the gradient bytes, N
net(torch.zeros(1, 1, 64, 64))            # materialize Lazy params
n_bytes = 4 * sum(p.numel() for p in net.parameters())
beta = 4.5e9   # NCCL allreduce, effective bytes/device/s on this box (13.5)
if d2l.num_gpus() >= 2:
    torchrun = str(pathlib.Path(sys.executable).parent / 'torchrun')
    subprocess.run([torchrun, '--standalone', '--nproc-per-node=2',
                    str(DDP_DIR / 'comm_probe.py')], check=True)
    with open(DDP_DIR / 'comm.json') as f:
        comm = json.load(f)
    print(f"k=2: predicted comm 2N/beta = {2 * n_bytes / beta * 1e3:.0f} "
          f"ms/step; measured {comm['synced_ms'] - comm['no_sync_ms']:.0f} ms "
          f"(synced {comm['synced_ms']:.0f}, no_sync {comm['no_sync_ms']:.0f})")
```

Prediction and measurement land within tens of percent of each other, and
**that agreement is the result**: a scaling curve you can price before you
buy, not a marketing "N× faster" claim — and it holds at two GPUs as
clearly as at four. One line on what a datacenter box changes: an NVLink
fabric shrinks $t_{\text{comm}}$ by roughly two orders of magnitude
(:numref:`tab_gpu_specs`), so the same accounting predicts near-linear
scaling — same model, different constant. (The legacy `nn.DataParallel` is
single-process and GIL-bound; use DDP even on one node, as PyTorch's own
docs advise.)

One loose end from :numref:`sec_multi_gpu` deserves to be closed by
measurement rather than left as an assertion. The diagnosis there was
that NCCL's P2P-less fallback moves bytes with a latency-bound GPU-kernel
copy — in effect a performance bug in how the library's default transport
interacts with this particular platform — and that one documented switch,
`NCCL_SHM_USE_CUDA_MEMCPY=1`, re-routes the same transfer over the DMA
copy engines. Measure the bare collective both ways:

```{.python .input #multi-gpu-practice-ddp-really-run-6}
%%tab pytorch
BENCH_SCRIPT = r'''
import json, os, time, torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
x = torch.randn(16 * 1024 * 1024, device=rank)     # 64 MB of fp32
for _ in range(5):                                 # warmup
    dist.all_reduce(x)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(15):
    dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.time() - t0) / 15
if rank == 0:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "bench.json"), "w") as f:
        json.dump({"gbs": 2 * x.numel() * 4 / dt / 1e9}, f)
dist.destroy_process_group()
'''
_ = (DDP_DIR / 'allreduce_bench.py').write_text(BENCH_SCRIPT)

def bare_allreduce_gbs(**env):
    (DDP_DIR / 'bench.json').unlink(missing_ok=True)
    torchrun = str(pathlib.Path(sys.executable).parent / 'torchrun')
    subprocess.run([torchrun, '--standalone', '--nproc-per-node=2',
                    str(DDP_DIR / 'allreduce_bench.py')],
                   check=True, env={**os.environ, **env})
    with open(DDP_DIR / 'bench.json') as f:
        return json.load(f)['gbs']

if d2l.num_gpus() >= 2:
    slow = bare_allreduce_gbs()
    # Works around an NCCL transport bottleneck on this P2P-less box
    # (13.5's diagnosis) -- a platform-specific fix; validate on yours.
    fast = bare_allreduce_gbs(NCCL_SHM_USE_CUDA_MEMCPY='1')
    print(f'bare allreduce, effective bytes/device/s: '
          f'{slow:.1f} GB/s at the default, {fast:.1f} GB/s configured '
          f'({fast / slow:.1f}x)')
```

Roughly five-fold, from configuration alone. First the general lesson: a
collective library's configuration — which transport it picks, what
topology it assumes — can move communication performance by *factors*,
not percent, so measure yours against what the wire demonstrably carries
(:numref:`sec_multi_gpu`'s raw copy) before trusting it. Then the
specific one, which is why every training run above still uses the
library's defaults: this workaround wins the microbenchmark and loses
the workload. On our box, with this NCCL build, the copy-engine mode
deadlocks DDP's overlapped training path within seconds — the very
collectives that just ran flawlessly in isolation wedge inside the
training loop — so adopting it here would trade a five-fold bandwidth
win for a hung notebook. An escape hatch is platform-specific twice over:
whether you need it, and whether it survives your workload, are both
measurements (the exercises have you reproduce both halves). The cost
model is indifferent either way — :eqref:`eq_dp_cost` simply takes
whatever $\beta$ your fabric, as configured, sustains.

## Sharding the Redundant: the FSDP Idea
:label:`subsec_mgp-fsdp`

DDP replicates *everything* on every rank: $k$ identical copies of the
parameters, the gradients, and the optimizer states. For the $16P$-byte
training footprint of :numref:`sec_memory_precision`, that is $k-1$ copies
of everything, wasted — and it caps the model size at what one GPU holds,
the limitation :numref:`sec_multi_gpu` flagged. **Fully Sharded Data
Parallel** (FSDP) removes the redundancy by *sharding* those tensors across
ranks, each rank owning $1/k$ of each, and materializing a full layer only
for the moment it is needed :cite:`Zhao.Gu.Varma.ea.2023`. The idea is the
ZeRO ladder :cite:`Rajbhandari.Rasley.Ruwase.ea.2020`: shard the
optimizer states first (they are the biggest, $8P$), then the gradients,
then the parameters — each rung cutting memory toward $1/k$ at the cost of
more communication.

The mechanism is the :numref:`sec_multi_gpu` identity, cashed in. Recall
that allreduce = reduce-scatter + all-gather. FSDP simply *keeps the two
halves separate*: an **all-gather** reconstructs a layer's full parameters
just before it computes, and frees them just after; a **reduce-scatter**
sums each layer's gradients but leaves each rank holding only its own
shard (:numref:`fig_fsdp_lifecycle`). No tensor's full replica ever lives
longer than the layer that needs it.

![The FSDP lifecycle of one block, under one simplified
reshard-after-forward policy: parameters live sharded ($P/k$ per rank); an
all-gather materializes the full block only while it computes, then frees
it; a reduce-scatter leaves each rank with its gradient
shard.](../img/mdl-perf-fsdp-lifecycle.svg)
:label:`fig_fsdp_lifecycle`

That completes the small family of collectives this chapter needs — worth
one table, since the rest of the book will name them without ceremony:

| collective | what every rank ends with | where it appears |
|---|---|---|
| allreduce | the full sum | DDP's gradient buckets; :numref:`sec_multi_gpu` |
| reduce-scatter | one shard of the sum | FSDP gradients |
| all-gather | every shard, concatenated | FSDP parameters, just-in-time |
| all-to-all | a different shard from each peer | expert parallelism (:numref:`sec_training_systems`) |
:label:`tab_collectives`

FSDP's payoff — fitting a model that does not fit — is invisible on our
11.2M-parameter demo, which occupies a few hundred MB of a 24 GB card, so
we show the *shape* of the code rather than run it. The modern API is
`fully_shard` over a `DeviceMesh`; the original `FullyShardedDataParallel`
wrapper class still imports at our pin, but it is the deprecated legacy
path:

```{.python .input #multi-gpu-practice-sharding-the-redundant-the-fsdp-idea}
%%tab pytorch
# Code sketch (not executed): shard each block, then the whole model.
# from torch.distributed.fsdp import fully_shard
# from torch.distributed.device_mesh import init_device_mesh
# mesh = init_device_mesh("cuda", (world_size,))
# model = build_large_model()
# for block in model.transformer_blocks:   # shard per repeated block
#     fully_shard(block, mesh=mesh)
# fully_shard(model, mesh=mesh)            # shard the remainder
print('FSDP sketch: reach for it past a few billion parameters, '
      'not on an 11M-param demo')
```

```{.python .input #multi-gpu-practice-sharding-the-redundant-the-fsdp-idea}
%%tab jax
# In JAX the same sharding is a PartitionSpec, not a wrapper — see below.
print('JAX shards by annotation; the next subsection is the demo')
```

You reach for FSDP when the training state at your precision — the
parameters, gradients, and optimizer states of
:numref:`sec_memory_precision`'s anatomy, plus activations — no longer
fits on one GPU, or when that redundancy is worth trading away for
communication; for this card class the threshold arrives at a few billion
parameters. The production distributed-training map, and how to combine
FSDP with the other parallelism axes, lives in
:numref:`sec_training_systems`.

## JAX: Annotate the Layout, the Compiler Writes the Collectives
:label:`subsec_mgp-jax`

Everything above was PyTorch's world: multiple processes, explicit
collectives, a launcher. JAX offers a different deal, and it is the
chapter's cleanest framework contrast. One process sees all the GPUs
(:numref:`subsec_hw-interconnects`); you describe *how the data is laid out*
across them with a `Mesh` and a `NamedSharding`, `device_put` the arrays
onto that layout — parameters replicated (`P()`), the batch sharded along
its leading axis (`P('data')`) — and `jit` the **unchanged** single-device
training step. XLA's automatic-parallelization pass (GSPMD
:cite:`Xu.Lee.Chen.ea.2021`) partitions the computation to match the data
layout and *inserts the very allreduce* that :numref:`sec_multi_gpu` wrote
by hand — you never write a collective:

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-1}
%%tab jax
def make_mesh(k):
    return jax.make_mesh((k,), ('data',))

def shard_batch(X, y, mesh):
    P = jax.sharding.PartitionSpec('data')      # leading axis across devices
    sh = jax.sharding.NamedSharding(mesh, P)
    return jax.device_put(X, sh), jax.device_put(y, sh)

def replicate(module, mesh):
    """Give every device of the mesh a full copy of a module's state."""
    sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    nnx.update(module, jax.device_put(nnx.state(module), sh))

# The training step is written for one device; jit + sharded inputs make it
# data-parallel with no change to the body.
@nnx.jit
def train_step(model, opt, X, y):
    def loss_fn(m):
        return optax.softmax_cross_entropy_with_integer_labels(
            m(X), y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)                     # XLA inserts the allreduce
    return loss
```

Now run it — a real `ResNet18`, a real optimizer, a real Fashion-MNIST
batch (resized once to 64×64 and kept as host arrays). The reveal is
`visualize_array_sharding`, which draws where a tensor actually lives: the
batch split across the mesh before the step, and a weight *after* the step
still replicated everywhere — the compiler's allreduce is what kept every
copy identical:

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-2}
%%tab jax
imgs, labels = d2l.FashionMNIST().train         # raw arrays; resize once
X_all = np.asarray(jax.image.resize(
    jnp.float32(imgs[:, :, :, None]) / 255, (len(imgs), 64, 64, 1),
    'bilinear'))
y_all = np.asarray(labels, np.int32)

k = min(4, jax.local_device_count())
mesh = make_mesh(k)
model = ResNet18(rngs=nnx.Rngs(0))
opt = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
replicate(model, mesh); replicate(opt, mesh)
Xs, ys = shard_batch(X_all[:256], y_all[:256], mesh)
loss = train_step(model, opt, Xs, ys)           # one data-parallel step
print(f'one sharded step: loss {float(loss):.2f}')
print('the batch, split along its leading axis:')
jax.debug.visualize_array_sharding(Xs.reshape(len(Xs), -1))
print('a weight after the step, still replicated:')
jax.debug.visualize_array_sharding(model.net.layers[-1].kernel[...])
```

Take the receipt. :numref:`sec_multi_gpu` wrote its `pmean` by hand; here
nobody wrote a collective — so where is it? In the compiled program. Lower
and compile the step, and search the XLA text for what GSPMD inserted:

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-3}
%%tab jax
hlo = train_step.lower(model, opt, Xs, ys).compile().as_text()
ops = [line.strip() for line in hlo.splitlines()
       if ' all-reduce(' in line or ' all-reduce-start(' in line]
sizes = [int(re.search(r'\[(\d+)\]', op).group(1)) for op in ops]
print(f'{len(ops)} all-reduce ops in the compiled step; the largest '
      f'sums {max(sizes) / 1e6:.1f}M floats:')
print(ops[sizes.index(max(sizes))][:88] + ' ...')
```

Dozens of allreduces, not one — and reading them teaches. The two largest
together carry the ~11M gradient floats (XLA bucketed them into two fused
sums, much as DDP buckets); the many small ones are batch-norm statistics,
because under `jit` even normalization is computed over the *global*
batch — the compiler preserves single-device semantics exactly, where DDP
leaves each replica its own batch-norm statistics unless you ask for
`SyncBatchNorm`. (The exact count is a compiler artifact; the two big ones
are the point.)

Measured, under the same weak-scaling convention as the DDP sweep
(per-device batch 256) — one process, no launcher, no sidecar files:

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-4}
%%tab jax
def batches(global_batch):
    n = (len(X_all) // global_batch) * global_batch
    for i in range(0, n, global_batch):
        yield X_all[i:i + global_batch], y_all[i:i + global_batch]

def jax_throughput(k, per_device_batch=256):
    mesh = make_mesh(k)
    model = ResNet18(rngs=nnx.Rngs(0))
    opt = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    replicate(model, mesh); replicate(opt, mesh)
    B = per_device_batch * k
    for epoch in range(2):                 # epoch 0 warms up and compiles
        t0 = time.time(); n = 0
        for X, y in batches(B):
            loss = train_step(model, opt, *shard_batch(X, y, mesh))
            n += X.shape[0]
        loss.block_until_ready()           # completion timing
        dt = time.time() - t0
    return n / dt

ks = [k for k in (1, 2, 4) if k <= jax.local_device_count()]
tput = [jax_throughput(k) for k in ks]
for k, t in zip(ks, tput):
    print(f'{k} GPU(s): {t:.0f} samples/s, {t / tput[0]:.2f}x, '
          f'{100 * t / tput[0] / k:.0f}% weak-scaling efficiency')
```

The shape matches the DDP sweep — real gains, efficiency sagging as $k$
grows, no cliff — with one instructive difference: the compiled XLA step
is faster per device, so the same allreduce is a *larger fraction* of each
step and the efficiency reads a few points lower (82% already at two GPUs
in our runs). That is :eqref:`eq_dp_cost` again: speed up the compute and
the fabric gets *relatively* slower. And note what these numbers are not:
this loop feeds pre-staged host arrays while the DDP script pays a real
`DataLoader`, so the absolute samples/s are not a framework shoot-out —
the scaling curve is the comparison.

One more check, because we are trusting a collective nobody wrote.
:numref:`sec_multi_gpu` ended its hand-built loop with a one-step equality
test, and the declarative version deserves the same scrutiny — same
initialization, same 256-example batch, one step on one device versus one
step sharded across two:

```{.python .input #multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-5}
%%tab jax
def one_step_delta(k):
    mesh = make_mesh(k)
    model = ResNet18(rngs=nnx.Rngs(0))
    opt = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    replicate(model, mesh); replicate(opt, mesh)
    before = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
    train_step(model, opt, *shard_batch(X_all[:256], y_all[:256], mesh))
    after = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
    return jax.tree.map(lambda a, b: a - b, after, before)

d1 = one_step_delta(1)
d2 = one_step_delta(min(2, jax.local_device_count()))
gaps = jax.tree.map(lambda a, b: np.abs(a - b).max(), d1, d2)
print('max update difference, k=1 vs k=2: '
      f'{max(jax.tree.leaves(gaps)):.1e}')
```

The two runs agree to about $10^{-4}$ against update magnitudes near
$10^{-2}$ — not the $10^{-9}$ of :numref:`sec_multi_gpu`'s LeNet check,
and the gap is itself informative: these are two *different compiled
programs*, whose tf32 convolutions and batch-norm reductions associate
their arithmetic differently, so the residue is rounding. What the check
exists to catch — a summed-instead-of-averaged gradient, an accidental
$k\times$ learning rate — would announce itself at the scale of the update
itself.

And here is the punchline. To move between *data* parallelism, *tensor*
parallelism, and *FSDP*-style sharding in JAX, you change the
`PartitionSpec` — not the model code. Sharding the batch axis gives data
parallelism (above); sharding a weight's feature axis gives tensor
parallelism; sharding the parameters and letting XLA all-gather them
just-in-time gives the FSDP pattern. **One sharding vocabulary — annotate
the layout, the compiler writes the collectives — spans what PyTorch
exposes as three different APIs** (DDP, tensor-parallel wrappers, FSDP) —
though an effective sharding plan is still model-aware: someone has to
choose the layouts and the constraint points. The manual end of the same
spectrum is the `jax.shard_map` + `lax.psum` of :numref:`sec_multi_gpu`,
where you write the collective yourself; `jit` + sharding is the
automatic end.

| | PyTorch | JAX |
|---|---|---|
| processes | one per GPU (`torchrun`) | one, sees all GPUs |
| collectives | explicit (NCCL, or DDP's buckets) | inserted by XLA/GSPMD |
| data parallel | `DistributedDataParallel` | shard the batch axis |
| tensor parallel | separate wrappers / DTensor | change the `PartitionSpec` |
| sharded (FSDP) | `fully_shard` | change the `PartitionSpec` |
| control | imperative, visible | declarative, compiler-driven |
:label:`tab_pt_jax_parallel`

Neither deal is strictly better: PyTorch's explicitness makes the
communication legible and debuggable; JAX's declarativeness makes the same
code span parallelism strategies by editing an annotation. Knowing both is
knowing the design space.

## When One Node Is Not Enough
:label:`subsec_mgp-bridge`

This chapter stops at the boundary of a single machine, and it is worth
naming what lies past it. When a model is too large for even sharded data
parallelism on one node — when the parameters, or the batch, or the
sequence length outgrow what $k$ local GPUs can hold — the answer is to
shard across *machines*, combining data parallelism with tensor, pipeline,
and expert parallelism into the "3D parallelism" of frontier training. The
collectives then run over a network fabric measured in tens of GB/s
between nodes rather than an NVLink domain within one, and the cost model
of :numref:`sec_multi_gpu` acquires a second, slower bandwidth term. That
is the province of the Language Models part, which has models and datasets
large enough to warrant it; the production library map — Megatron, the
FSDP/DTensor stack, DeepSpeed, and how to launch and checkpoint them across
a cluster — is :numref:`sec_training_systems`. From here the communication
*algebra* stays the same; what multi-node adds is engineering on top of
it — a hierarchy of fabrics (NVLink inside a node, a network between
nodes), rendezvous and elastic restart when machines fail, and stragglers
that turn a synchronous step into a queueing problem.

## Summary

* Production data parallelism (DDP) fixes the three deficits of a
  hand-rolled loop: one process per GPU (no shared GIL), ring/tree
  collectives (no hub), and gradient bucketing that **overlaps
  communication with the backward pass**.
* Launched from a notebook via `torchrun` on a sidecar script, DDP scales
  a compute-dense ResNet-18 by roughly 1.8× on two GPUs and 3.3× on four
  on our host-staged box (88% and 82% *weak-scaling* efficiency; a
  fixed-global-batch strong-scaling sweep asks the harder question), and a
  `no_sync()` measurement confirms :eqref:`eq_dp_cost`'s communication
  price to within tens of percent. An NVLink box changes only the
  constant.
* The constant is also configuration-sensitive: one documented NCCL
  switch quintuples the bare-collective bandwidth on this box — yet
  deadlocks DDP's overlapped training path, so the notebooks keep the
  library's defaults and teach the pair of measurements instead.
  Collective-library configuration moves communication by factors, not
  percent; a workaround is validated per platform *and* per workload.
* FSDP shards the $16P$-byte training state across ranks — the ZeRO ladder
  — by splitting allreduce back into its reduce-scatter and all-gather
  halves and materializing each layer just-in-time. It is for models whose
  training state outgrows one GPU — a few billion parameters on this card
  class.
* JAX inverts the model: one process, annotate the data layout with a
  `Mesh` and `PartitionSpec`, `jit` the unchanged step, and XLA inserts
  the collectives — visible in the compiled program's `all-reduce` ops,
  measured in a k-sweep, and verified by a k=1-versus-k=2 equality check.
  Changing the `PartitionSpec` — not the code — moves between data,
  tensor, and FSDP-style sharding, spanning what PyTorch exposes as three
  APIs.
* Multi-node 3D parallelism and the production library map are the
  Language Models part and :numref:`sec_training_systems`.

## Exercises

1. Vary DDP's `bucket_cap_mb` (the gradient bucket size) and measure
   throughput at $k = 2$. Why is there an optimum — what does a too-small
   bucket cost, and a too-large one?
1. Extend the `no_sync()` measurement to $k = 4$. How much does the
   per-step communication time grow from $k=2$? Compare against
   :eqref:`eq_ring_traffic`'s $2(k-1)/k$ growth and the flat $2N/\beta$
   of :eqref:`eq_dp_cost` — which fits better, and what does that tell
   you about how NCCL schedules the transfer on this P2P-less box?
1. Reproduce both halves of the fabric-configuration story. First extend
   the bare-collective comparison across payloads from 1 MB to 256 MB:
   where does each transport's effective bandwidth saturate, and does
   the five-fold gap persist? Then set `NCCL_SHM_USE_CUDA_MEMCPY=1`
   inside `train_ddp.py` itself and relaunch the $k=2$ sweep — on our
   box the run wedges within seconds (be ready to kill the launcher).
   What does the pair teach about trusting a library's defaults on a
   topology it was not tuned for — and about trusting a workaround
   anywhere you have not re-validated it, workload and all?
1. Write the `PartitionSpec` that shards a weight matrix's *output*
   features across the mesh (tensor parallelism) and
   `visualize_array_sharding` the result. How does the communication
   pattern differ from the batch-sharded (data-parallel) case?
1. Size ZeRO stage 3 (parameters, gradients, and optimizer states all
   sharded) for a 7-billion-parameter model on 8 GPUs with 80 GB each:
   what is the per-GPU training footprint, and does it fit? Redo the
   arithmetic for DDP (no sharding) and explain the difference.
1. The weak-scaling efficiency in the DDP sweep fell from 88% at $k=2$ to
   82% at $k=4$. Using :eqref:`eq_dp_cost`, predict the efficiency at
   $k=8$ on the same fabric, and state the assumption your prediction
   makes about how $t_{\text{comm}}$ grows.

<!-- slides -->

::: {.slide title="What the Hand-Rolled Loop Lacked"}
Three deficits, all software:

- **no overlap** — communicate only after the whole backward
- **one process** — one GIL dispatching $k$ GPUs
- **star topology** — device 0 as a hub

DDP fixes all three: process per GPU, ring collectives, and
buckets that overlap comm with backward.
:::

::: {.slide title="DDP Overlap"}
![](../img/mdl-perf-ddp-overlap.svg){width=92%}

Gradients arrive back-to-front; bucket them and allreduce each
as it fills, hiding communication under compute.
:::

::: {.slide title="DDP, Really Run"}
Multiple processes from a notebook: write a sidecar script,
launch with `torchrun`, read back per-rank results.

@multi-gpu-practice-ddp-really-run-2@pytorch

`init_process_group` + `DDP(model)` — the *loop body* is
unchanged from single-GPU; the scaffolding is the launcher's.
:::

::: {.slide title="Weak Scaling, Measured"}
Convention first: per-rank batch fixed at 256 ⇒ global batch
grows with $k$ — **weak scaling**.

@multi-gpu-practice-ddp-really-run-3@pytorch

~1.8× at 2 GPUs (88%), ~3.3× at 4 (82%) — sublinear, no cliff.
Strong scaling (global batch 512, split thinner) parts company
as $k$ grows. NVLink changes only the constant.
:::

::: {.slide title="Price the Fabric, Then Check the Bill"}
`no_sync()` turns gradient sync off ⇒ synced − unsynced steps
estimate the communication time.

@multi-gpu-practice-ddp-really-run-5@pytorch

The cost model's $2N/\beta$ lands within tens of percent of
the measurement. *Prediction agreeing with measurement is the
result.*
:::

::: {.slide title="Configured vs. Default: Factors, Not Percent"}
§13.5 diagnosed the fallback transport; one documented switch
re-routes it over the DMA copy engines:

@multi-gpu-practice-ddp-really-run-6@pytorch

**~5× from configuration alone** — yet the same mode *deadlocks*
DDP's overlapped training on this box, so the notebooks keep
the defaults. A workaround is validated per platform *and* per
workload — measure the workload, not just the wire.
:::

::: {.slide title="Sharding the Redundant: FSDP"}
![](../img/mdl-perf-fsdp-lifecycle.svg){width=88%}

DDP replicates the whole $16P$-byte state $k$ times. FSDP
shards it: allreduce = reduce-scatter + all-gather, kept
*separate* — gather a layer just-in-time, free it after. The
§13.5 identity, cashed in.
:::

::: {.slide title="JAX: Annotate the Layout"}
One process. Describe the layout; `jit` the unchanged step;
XLA writes the collectives.

@multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-2@jax

Batch sharded, weights replicated — and still replicated
*after* the step: the compiler's allreduce kept every copy
identical.
:::

::: {.slide title="The Receipt"}
Nobody wrote a collective — so find it in the compiled program:

@multi-gpu-practice-jax-annotate-the-layout-the-compiler-writes-the-collectives-3@jax

Two big allreduces carry the gradients (XLA bucketed them);
the small ones are batch-norm statistics — global under `jit`.
Measured k-sweep + a k=1-vs-k=2 equality check close the loop.
Change the `PartitionSpec`, not the code, to move between
data, tensor, and FSDP sharding.
:::

::: {.slide title="Explicit vs. Declarative"}
| | PyTorch | JAX |
|---|---|---|
| processes | one per GPU | one, all GPUs |
| collectives | explicit | XLA inserts |
| switch strategy | different API | different `PartitionSpec` |

Neither is better; knowing both is knowing the design space.
Past one node → 3D parallelism, the Language Models part.
:::
