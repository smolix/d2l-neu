# Sharding the Training State
:label:`sec_sharding`

In :numref:`sec_memory_precision`, we saw that training with Adam in fp32
stores 16 bytes per parameter—four bytes each for the weight, its gradient,
and Adam's two moment estimates. For our 18-block, 1B-parameter GPT, that
fixed state fills 15.1 GiB of a 24 GiB card before reading a single token, and
data parallelism (:numref:`sec_multi_gpu`, :numref:`sec_multi_gpu_concise`)
replicates all 16 bytes on every GPU. If we double the depth of our GPT to 36
blocks (1.92B parameters, 28.6 GiB of state), data parallelism on four 24 GiB
GPUs cannot train it any more than one GPU can.

Can $k$ GPUs train a model that no single card can hold? They can, because a
GPU needs the full weights of a block only while computing that block's
forward or backward pass, and it never needs the full gradient or full
optimizer state at all. Each of the $k$ GPUs can therefore store $1/k$ of the
weights, gradients, and optimizer moments, gather each block's weights on the
fly just before the block runs, and reduce-scatter each block's gradients as
soon as they are produced.

```{.python .input #sharding-setup}
%%tab jax
# XLA reads its flags once, when JAX starts, so this cell runs before the
# imports. The latency-hiding scheduler orders the step with its memory in
# view; without it, the sharded step needs more memory than JAX may
# allocate by default (see the note on stage 2). On GPUs without
# peer-to-peer access, such as ours, set NCCL_LOCAL_REGISTER=0 in the
# environment before starting: NCCL's buffer registration fails there and
# prints a warning at every collective.
import os
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                           ' --xla_gpu_enable_latency_hiding_scheduler=true')
```

```{.python .input #sharding-sharding-the-training-state}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #sharding-sharding-the-training-state}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #sharding-sharding-the-training-state-1}
%%tab pytorch
import json
```

```{.python .input #sharding-sharding-the-training-state-1}
%%tab jax
import collections
import re
from jax.sharding import NamedSharding, PartitionSpec as P
```

We first verify on a meta/shape-only device that the 36-block GPT's $16N$
bytes of training state exceed the 24 GiB capacity of a single RTX 4090:

```{.python .input #sharding-sharding-the-training-state-2}
%%tab pytorch
config = dict(vocab_size=50257, num_hiddens=2048, num_heads=16,
              num_blks=36, pos='learned')
k = 2                       # GPUs, from the all-gather below on
with torch.device('meta'):  # Shapes and dtypes only, no data
    N = d2l.num_params(d2l.GPT(**config))
print(f'{N / 1e9:.2f} billion parameters, 16N = {16 * N / 2**30:.1f} GiB')
if d2l.num_gpus():
    total = torch.cuda.mem_get_info()[1]
    print(f'one GPU holds {total / 2**30:.1f} GiB')
```

```{.python .input #sharding-sharding-the-training-state-2}
%%tab jax
config = dict(vocab_size=50257, num_hiddens=2048, num_heads=16,
              num_blks=36, pos='learned')
k = 2                       # GPUs, from the all-gather below on
key = d2l.get_key()
N = d2l.num_params(nnx.eval_shape(lambda: d2l.GPT(**config,
                                                  rngs=nnx.Rngs(key))))
print(f'{N / 1e9:.2f} billion parameters, 16N = {16 * N / 2**30:.1f} GiB')
if jax.default_backend() == 'gpu':
    limit = jax.devices()[0].memory_stats()['bytes_limit']
    print(f'JAX may allocate {limit / 2**30:.1f} GiB of one GPU')
```

## Partitioning the Optimizer State, Gradients, and Weights
:label:`subsec_mgp-fsdp`

In standard data parallelism, all $k$ GPUs hold identical copies of Adam's
moment estimates and perform identical optimizer updates. The *ZeRO*
(*Zero Redundancy Optimizer*) family :cite:`Rajbhandari.Rasley.Ruwase.ea.2020`
eliminates this redundancy in three cumulative stages (:numref:`tab_mgp-zero`):

* **ZeRO Stage 1** partitions only the optimizer state: each GPU holds Adam's
  two moments for $1/k$ of the parameters ($8 + 8/k$ bytes/parameter). Instead
  of an allreduce ($=\text{reduce-scatter} + \text{all-gather}$,
  :numref:`subsec_mg-ring`), the GPUs reduce-scatter the gradients so each GPU
  receives the summed gradient for its $1/k$ shard, updates its own parameters,
  and all-gathers the updated weights—moving the exact same bytes as data
  parallelism.
* **ZeRO Stage 2** also partitions the gradients ($4 + 12/k$ bytes/parameter):
  as each gradient bucket fills during the backward pass, it is immediately
  reduce-scattered and the non-owned $1 - 1/k$ gradient shards are freed,
  again at zero extra byte cost over data parallelism.
* **ZeRO Stage 3**, or *Fully Sharded Data Parallelism* (FSDP)
  :cite:`Zhao.Gu.Varma.ea.2023`, shards the weights as well ($16/k$
  bytes/parameter). Before a block runs in the forward or backward pass, an
  all-gather assembles its full weights; immediately after the block finishes,
  the non-owned weight shards are freed, and the block's gradients are
  reduce-scattered (:numref:`fig_fsdp_lifecycle`).

:Training state per GPU at the end of the backward pass, with Adam in fp32 on $k$ GPUs, and for the GPT with 36 blocks and 1.92 billion parameters on $k = 2$.
:label:`tab_mgp-zero`

| each GPU holds | weights | gradients | Adam state | bytes per parameter | GPT, $k = 2$ |
|---|---|---|---|---|---|
| data parallelism | all | all | all | 16 | 28.6 GiB |
| ZeRO stage 1 | all | all | $1/k$ | $8 + 8/k$ | 21.4 GiB |
| ZeRO stage 2 | all | $1/k$ | $1/k$ | $4 + 12/k$ | 17.9 GiB |
| ZeRO stage 3 (FSDP) | $1/k$ | $1/k$ | $1/k$ | $16/k$ | 14.3 GiB |

![The life of one block's weights under FSDP on one GPU. Between uses the
block is sharded, and each GPU holds $1/k$ of it; the other blocks stay
sharded throughout. An all-gather assembles the whole block for its forward
pass, after which it is released. A second all-gather assembles it for the
backward pass, after which it is released again, and a reduce-scatter
leaves each GPU with its share of the
gradients.](../img/mdl-perf-fsdp-lifecycle.svg)
:label:`fig_fsdp_lifecycle`

Because FSDP performs two all-gathers (forward and backward) and one
reduce-scatter of $S = 4N$ bytes per step—three collective phases where a ring
allreduce performs two—it moves **$1.5\times$ the communication volume of data
parallelism**. :numref:`tab_collectives` summarizes the four primary
collectives used across parallel training.

:The collective operations behind data-parallel and sharded training.
:label:`tab_collectives`

| collective | what every GPU ends with | where it appears |
|---|---|---|
| allreduce | the full sum | gradients in data parallelism |
| reduce-scatter | one $1/k$ shard of the sum | gradients in ZeRO and FSDP |
| all-gather | every shard, concatenated | weights in ZeRO and FSDP |
| all-to-all | a distinct slice from each GPU | token routing in mixtures of experts (:numref:`sec_moe`) |

## Memory and Communication

Let us predict the peak memory and step time of our 36-block GPT under FSDP on
$k = 2$ GPUs, with 2 sequences of 1,024 tokens per GPU and per-block activation
checkpointing (:numref:`subsec_mp-checkpointing`).

* **Memory:** The sharded training state takes $16N/2 = 14.3$ GiB per GPU.
  Checkpointed activations take ~1 MiB/token (~2 GiB for 2,048 tokens), and one
  or two currently gathered blocks (~0.2 GB each) bring peak memory to at most
  **~17 GiB** per GPU—comfortably below 24 GiB.
* **Communication:** Each of the three collective phases sends $(k-1)S/k$
  bytes per GPU, giving a total communication time of

  $$
  t_\textrm{comm} = \frac{3(k-1)}{k}\,S\,\beta_\textrm{link}.
  $$
  :eqlabel:`eq_fsdp_comm`

  Notice the key scaling property of FSDP: **adding more GPUs divides the
  state memory per GPU by $k$, but the bytes each GPU sends per step grow from
  $1.5S$ (at $k = 2$) toward $3S$ (at large $k$)**. Only processing more tokens
  per GPU or using a faster interconnect reduces the communication share of a
  step.

We first measure $\beta_\textrm{link}$ for an all-gather of 256 MiB across
$k = 2$ GPUs:

```{.python .input #sharding-memory-and-communication-1}
%%tab pytorch
GATHER = r'''
import json, torch
import torch.distributed as dist
from d2l import torch as d2l

_, k, device = d2l.init_distributed()
x = torch.zeros(2**26 // k, device=device)  # This GPU's part
y = torch.empty(2**26, device=device)       # 256 MiB, gathered
t = d2l.Benchmark(lambda: dist.all_gather_into_tensor(y, x), repeats=10).time
print(json.dumps({'beta_link': t / ((k - 1) / k * y.nbytes)}))  # Per byte sent
dist.destroy_process_group()
'''
assert d2l.num_gpus() >= k, 'The cells from here on need two GPUs'
beta_link = max(r['beta_link'] for r in d2l.torchrun(GATHER, k))
print(f'bus bandwidth = {1e-9 / beta_link:.1f} GB/s in an all-gather on {k} '
      f'GPUs (beta_link = {1e9 * beta_link:.2f} ns/byte)')
```

```{.python .input #sharding-memory-and-communication-1}
%%tab jax
assert jax.device_count() >= k, 'The cells from here on need two GPUs'
mesh = jax.make_mesh((k,), ('data',), devices=jax.devices()[:k],
                     axis_types=(jax.sharding.AxisType.Auto,))
x = jnp.zeros(2**26, device=NamedSharding(mesh, P('data')))  # 256 MiB
# Asking for the result on every GPU makes the compiler gather the parts
gather = jax.jit(lambda x: x, out_shardings=NamedSharding(mesh, P()))
t = d2l.Benchmark(lambda: gather(x), repeats=10).time
beta_link = t / ((k - 1) / k * x.nbytes)  # Seconds per byte sent per GPU
del x                       # Frees its memory before the training step
print(f'bus bandwidth = {1e-9 / beta_link:.1f} GB/s in an all-gather on {k} '
      f'GPUs (beta_link = {1e9 * beta_link:.2f} ns/byte)')
```

Using the measured $\beta_\textrm{link}$ and the RTX 4090's 82.6 TFLOP/s tf32
peak (`peak_flops`), we can evaluate our paper predictions for per-GPU state
memory ($16N/k$), communication time ($3(k-1)S\beta_\textrm{link}/k$), and
checkpointed compute time ($8NbT/P$):

```{.python .input #sharding-memory-and-communication-2}
%%tab pytorch, jax
b, T = 2, 1024        # Sequences of T tokens per GPU and step
S = 4 * N             # Bytes of the fp32 weights, and of the gradients
peak_flops = 82.6e12  # tf32 FLOP/s of the RTX 4090
print(f'state per GPU, 16N/k: {16 * N / k / 2**30:.1f} GiB')
print(f'communication, 3(k-1)S beta_link/k: '
      f'{3 * (k - 1) / k * S * beta_link:.1f} s per step')
print(f'arithmetic at the peak, 8N bT/P: '
      f'{8 * N * b * T / peak_flops:.2f} s per step')
```

At ~2.2 GB/s bus bandwidth over host memory, moving $1.5S = 11.5$ GB per step
takes ~5 s, whereas the 2,048 tokens of arithmetic take only ~0.4 s. Because
0.4 s of compute cannot hide 5 s of communication, the step should take ~5 s.

## Sharded Training on Two GPUs

Now we run the sharded 36-block GPT step on two GPUs, constructing the model
so that no single GPU ever materializes the full 28.6 GiB state at once.

:begin_tab:`pytorch`
In PyTorch, we construct the model on the `'meta'` device, wrap each
checkpointed block and the top-level model with `fully_shard` from
`torch.distributed.fsdp`, materialize only this rank's shards via
`model.to_empty(device=device)`, and initialize them in place.
:end_tab:

```{.python .input #sharding-memory-and-communication-3}
%%tab pytorch
TRAIN_FSDP = r'''
import json, sys, torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint
from d2l import torch as d2l

class Checkpointed(nn.Module):
    """A block whose activations the backward pass recomputes."""
    def __init__(self, blk):
        super().__init__()
        self.blk = blk

    def forward(self, X, **kwargs):
        return checkpoint(self.blk, X, use_reentrant=False, **kwargs)

config, b, T = json.loads(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
_, k, device = d2l.init_distributed()
torch.set_float32_matmul_precision('high')  # fp32 products on tf32 cores
torch.manual_seed(0)        # The same seed in every process
with torch.device('meta'):  # Shapes only: no GPU holds the whole model
    model = d2l.GPT(**config)
model.blks = nn.ModuleList(Checkpointed(blk) for blk in model.blks)
for blk in model.blks:      # Each block is gathered and released on its own
    fully_shard(blk)
fully_shard(model)          # The rest: the embeddings and the final norm
model.to_empty(device=device)  # This GPU's shards, not yet initialized
with torch.no_grad():       # Initialize the shards as d2l.GPT does
    for m in model.modules():
        if isinstance(m, nn.RMSNorm):
            m.reset_parameters()
    for emb in (model.token_emb, model.pos_emb):
        nn.init.normal_(emb.weight, std=0.02)
    model.init_weights()
opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
# The same tokens on every GPU: they only set the shapes of the step
X = torch.randint(0, config['vocab_size'], (b, T), device=device)

def step():
    loss = F.cross_entropy(model(X).flatten(0, 1), X.flatten())
    loss.backward()
    opt.step()
    opt.zero_grad()

step()                      # Warm-up; creates Adam's moment estimates
times = [d2l.Benchmark(step, warmup=0, repeats=1).time for _ in range(3)]
kernels = d2l.kernel_times(step, steps=1)
comm = sum(ms for name, ms in kernels.items() if 'nccl' in name.lower())
print(json.dumps({'peak': torch.cuda.max_memory_allocated(), 'times': times,
                  'comm': comm / 1e3}))
dist.destroy_process_group()
'''
```

:begin_tab:`pytorch`
We now launch `TRAIN_FSDP` across $k$ processes with `d2l.torchrun` and collect
the peak allocated memory (`peak_bytes`), step durations (`times`), and NCCL
communication time (`comm`) from rank 0:
:end_tab:

```{.python .input #sharding-memory-and-communication-4}
%%tab pytorch
ranks = d2l.torchrun(TRAIN_FSDP, k, json.dumps(config), b, T)
peak_bytes = max(r['peak'] for r in ranks)
times, comm = ranks[0]['times'], ranks[0]['comm']  # GPU 0
```

:begin_tab:`jax`
In JAX, `fsdp_sharding` shards each parameter tensor along its first axis
divisible by $k$ (`P('data')`), and `create` applies this sharding constraint
so each GPU initializes only its own shards of `model` and `opt`:
:end_tab:

```{.python .input #sharding-memory-and-communication-5}
%%tab jax
def fsdp_sharding(x):
    """Split the first axis of x whose length k divides across the GPUs;
    replicate x if there is none."""
    for i, n in enumerate(x.shape):
        if n % k == 0:
            return NamedSharding(mesh, P(*[None] * i, 'data'))
    return NamedSharding(mesh, P())

@nnx.jit
def create(key):
    # The key is an argument: XLA would fold a constant key into the program,
    # together with every weight drawn from it, as large constants
    model = d2l.GPT(**config, rngs=nnx.Rngs(key))
    opt = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
    state = nnx.state((model, opt))
    nnx.update((model, opt), jax.lax.with_sharding_constraint(
        state, jax.tree.map(fsdp_sharding, state)))
    return model, opt

model, opt = create(d2l.get_key())  # Each GPU computes only its shards
```

:begin_tab:`jax`
Next, we wrap each block in `nnx.remat` for activation checkpointing, constrain
every block's activation output in `loss_fn` to stay batch-sharded (`data`) so
XLA emits FSDP's all-gathers and reduce-scatters automatically, and donate
`model` and `opt` in `train_step` so updated state overwrites the old shards in
place:
:end_tab:

```{.python .input #sharding-memory-and-communication-6}
%%tab jax
data = NamedSharding(mesh, P('data'))     # Split along the batch
block = nnx.remat(lambda blk, H: blk(H))  # Recomputed in the backward pass

def loss_fn(model, X):
    H = model.token_emb(X) + model.pos_emb(jnp.arange(X.shape[1]))
    for blk in model.blks:  # Every block's output stays split by batch
        H = jax.lax.with_sharding_constraint(block(blk, H), data)
    logits = model.token_emb.attend(model.norm(H))
    return optax.softmax_cross_entropy_with_integer_labels(logits, X).mean()

# Donated, the old weights and moment estimates make room for the new ones
@nnx.jit(donate_argnums=(0, 1))
def train_step(model, opt, X):
    loss, grads = nnx.value_and_grad(loss_fn)(model, X)
    opt.update(model, grads)
    return loss
```

:begin_tab:`jax`
Before running the step, we compile `train_step` on a batch-sharded token batch
`X` of shape $(kb, T)$ and query `compiled.memory_analysis()` to check the
per-GPU footprint of donated state and temporary buffers:
:end_tab:

```{.python .input #sharding-memory-and-communication-7}
%%tab jax
X = jax.device_put(jax.random.randint(d2l.get_key(), (k * b, T), 0,
                                      config['vocab_size']), data)
compiled = train_step.lower(model, opt, X).compile()
mem = compiled.memory_analysis()  # Bytes per GPU
print(f'per GPU: weights and moments '
      f'{mem.argument_size_in_bytes / 2**30:.1f} GiB (donated: '
      f'{mem.alias_size_in_bytes / 2**30:.1f} GiB), temporaries '
      f'{mem.temp_size_in_bytes / 2**30:.1f} GiB')
```

:begin_tab:`jax`
To verify that XLA emitted the expected FSDP collectives, `collective_megabytes`
parses the compiled HLO text and totals the result bytes of each collective
operation per GPU:
:end_tab:

```{.python .input #sharding-memory-and-communication-8}
%%tab jax
def collective_megabytes(compiled):
    """Megabytes of the results of each kind of collective in a compiled
    program, on one GPU."""
    sizes = collections.Counter()
    for line in compiled.as_text().splitlines():
        # A synchronous collective, or the -done half of an asynchronous
        # pair, whose result type lists only the arrays that it returns
        op = re.search(r' = (.*?) (all-gather|reduce-scatter|all-reduce|'
                       r'all-to-all|collective-permute)(-done)?\(', line)
        for bits, dims in re.findall(r'[a-z]+(\d+)\[([\d,]*)\]',
                                     op[1] if op else ''):
            n = math.prod(int(d) for d in dims.split(',') if d)
            sizes[op[2]] += int(bits) // 8 * n / 1e6
    return dict(sizes)
```

:begin_tab:`jax`
Tallying the collectives in `compiled` confirms that `all-gather` (twice per
step) and `reduce-scatter` (once per step) move the predicted byte volumes:
:end_tab:

```{.python .input #sharding-memory-and-communication-9}
%%tab jax
for kind, mb in sorted(collective_megabytes(compiled).items(),
                       key=lambda kv: -kv[1]):
    print(f'{kind}: {mb / 1e3:.2f} GB')
```

:begin_tab:`jax`
Now we warm up `train_step`, time three repetitions, and profile the step to
record NCCL communication time (`comm`) and peak device memory (`peak_bytes`):
:end_tab:

```{.python .input #sharding-memory-and-communication-10}
%%tab jax
train_step(model, opt, X).block_until_ready()  # Warm-up, finished
times = [d2l.Benchmark(lambda: train_step(model, opt, X), warmup=0,
                       repeats=1).time for _ in range(3)]
kernels = d2l.kernel_times(lambda: train_step(model, opt, X), steps=1)
comm = sum(ms for name, ms in kernels.items() if 'nccl' in name.lower()) / 1e3
peak_bytes = jax.devices()[0].memory_stats()['peak_bytes_in_use']
```

Finally, we report the measured per-GPU peak memory, median step duration, and
time spent in NCCL communication kernels on GPU 0:

```{.python .input #sharding-memory-and-communication-11}
%%tab pytorch, jax
print(f'per GPU: peak memory {peak_bytes / 2**30:.1f} GiB; '
      f'{np.median(times):.2f} s per step ({min(times):.2f} to '
      f'{max(times):.2f})\nof each step, GPU 0 spends {comm:.2f} s in '
      f'communication kernels')
```

Both GPUs hold the 36-block model at a peak of ~16 GiB per card (under our 17
GiB bound), and each step takes ~5.5 s, with GPU 0 spending over 90% of the
step in NCCL communication kernels—within a sixth of our 5 s paper prediction.
Over NVLink 5 (900 GB/s per direction), the same 11.5 GB of collectives takes
only ~13 ms, less than a thirtieth of the 0.4 s compute time.

This comparison also shows *when* to shard: a model whose training state fits
on a single GPU alongside a reasonable micro-batch (such as our 18-block GPT)
trains much faster with data parallelism and gradient accumulation (which
communicates once per global step) than with FSDP over PCIe (which re-gathers
the weights twice on every micro-batch).

## Summary

Data parallelism replicates all $16N$ bytes of fp32 Adam state on every GPU.
ZeRO partitions this state across $k$ GPUs: Stage 1 shards the optimizer
moments ($8 + 8/k$ bytes/parameter) and Stage 2 also shards the gradients
($4 + 12/k$ bytes/parameter), both moving the exact same $2(k-1)S/k$ bytes per
step as a ring allreduce. Stage 3 (FSDP) shards the weights as well ($16/k$
bytes/parameter), gathering each block's weights for the forward and backward
pass and reduce-scattering gradients at $1.5\times$ the communication volume
of data parallelism ($3(k-1)S/k$ bytes per GPU). Sharding enables two 24 GiB
GPUs to train a 36-block GPT whose 28.6 GiB state fits on neither card alone.

## Exercises

1. **Cost of sharding.** ● Count communication per GPU and step in units
   of $S = 4N$, the bytes of one full set of fp32 weights or gradients.
    1. A variant of ZeRO Stage 3 keeps every gathered block in memory from its
       forward pass until its backward pass finishes. Which collective does
       this eliminate, and how much memory saving from :numref:`tab_mgp-zero`
       does it give up?
    1. Mixed-precision FSDP all-gathers weights in bf16 and reduce-scatters gradients in bf16 while keeping each GPU's master weight and Adam shards in fp32. Compare its bytes transferred per step with fp32 data parallelism, and its memory per parameter with the last row of :numref:`tab_mgp-zero`.
    1. For the 36-block GPT on $k = 2$ GPUs at the measured
       $\beta_\textrm{link}$, how many tokens per GPU would a step need for
       checkpointed compute ($8N$ FLOPs/token at 83 TFLOP/s) to match the
       communication time :eqref:`eq_fsdp_comm`? Do those activations fit on
       the card, and why does gradient accumulation not help when FSDP
       re-gathers weights on every micro-batch?

:begin_tab:`pytorch`
2. [extended] **Sharded optimizer state.** ● Implement ZeRO Stage 1
    :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` by hand starting from
    `TRAIN_DDP` (:numref:`sec_multi_gpu_concise`) with `torch.optim.Adam`.
    1. Partition the parameter tensors across ranks so each rank owns roughly
       $1/k$ of the parameters and runs Adam only on its owned tensors, then
       broadcast each updated tensor from its owner with `dist.broadcast`.
       Verify that the parameters match full DDP Adam after several steps, and
       compare with `torch.distributed.optim.ZeroRedundancyOptimizer`.
    1. Measure allocated and peak GPU memory at $k = 2$ and $k = 4$ with and
       without sharding, and compare the total bytes sent by DDP allreduce plus
       owner broadcasts against Stage 1's reduce-scatter plus all-gather.

3. [code] **Tensor versus data parallelism.** ● Consider the MLP block
    $\mathbf{Y} = \phi(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$ with
    $\mathbf{X} \in \mathbb{R}^{n \times d}$,
    $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$, and
    $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$ across $k$ devices. Tensor
    parallelism (:numref:`fig_splitting`, :cite:`shoeybi2019megatron`) splits
    $\mathbf{W}_1$ column-wise and $\mathbf{W}_2$ row-wise.
    1. Show that each device can compute a partial $n \times d$ output without
       communication and that summing the $k$ partial outputs gives
       $\mathbf{Y}$. Which collective does the forward pass need, and which
       does the backward pass need for $\nabla_{\mathbf{X}}$?
    1. Compare the values sent per device and step under tensor parallelism
       versus data parallelism, and find the token count $n$ below which
       tensor parallelism sends less data. Why is tensor parallelism
       nevertheless restricted to fast intra-node NVLink domains?
    1. Using `parallelize_module` with `ColwiseParallel()` and
       `RowwiseParallel()` on a 1D `DeviceMesh`, count the forward and
       backward collectives with `CommDebugMode`.
   :exlabel:`ex_mgp-tensor`
:end_tab:

:begin_tab:`jax`
2. [extended] **Sharded optimizer state.** ● Implement ZeRO Stage 1
    :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` declaratively starting from
    `train_step` (:numref:`sec_multi_gpu_concise`) with `optax.adam`. Keep the
    model parameters replicated (`P()`), and place the leaves of
    `nnx.state(opt)` with `fsdp_sharding`.
    1. Ensure parameters remain replicated at the end of `train_step` via
       `jax.sharding.reshard` if needed, and verify that multi-step parameter
       updates match replicated Adam.
    1. Inspect the compiled step with `collective_megabytes` and compare its collectives and live/peak memory at $k = 2$ and $k = 4$ against :numref:`tab_mgp-zero`.

3. [code] **Tensor versus data parallelism.** ● Consider the MLP block
    $\mathbf{Y} = \phi(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$ with
    $\mathbf{X} \in \mathbb{R}^{n \times d}$,
    $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$, and
    $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$ across $k$ devices. Tensor
    parallelism (:numref:`fig_splitting`, :cite:`shoeybi2019megatron`) splits
    $\mathbf{W}_1$ column-wise (`P(None, 'data')`) and $\mathbf{W}_2$ row-wise
    (`P('data', None)`).
    1. Show that each device can compute a partial $n \times d$ output without
       communication and that summing the $k$ partial outputs gives
       $\mathbf{Y}$. Which collective does the forward pass need, and which
       does the backward pass need for $\nabla_{\mathbf{X}}$?
    1. Compare the values sent per device and step under tensor parallelism
       versus data parallelism, and find the token count $n$ below which
       tensor parallelism sends less data. Why is tensor parallelism
       nevertheless restricted to fast intra-node NVLink domains?
    1. Place $\mathbf{W}_1$ with `P(None, 'data')`, $\mathbf{W}_2$ with
       `P('data', None)`, and $\mathbf{X}$ replicated on an `Auto` mesh, jit
       the forward and backward pass, and inspect the emitted collectives.
   :exlabel:`ex_mgp-tensor`
:end_tab:

4. **Prefetching the next block.** In the forward pass under FSDP, let $g$ be
   the all-gather time of one block and $c$ its compute time. Without overlap,
   $L$ blocks take $L(g + c)$; prefetching issues the all-gather of block
   $l + 1$ while block $l$ computes.
    1. Show that prefetching reduces the forward pass time to
       $g + (L-1)\max(g, c) + c$, and explain why no schedule can do better
       than $L\max(g, c)$.
    1. Evaluate $g$ and $c$ for one block of the 36-block GPT ($50$M
       parameters, 2,048 tokens/GPU, 83 TFLOP/s) on our PCIe link versus
       NVLink 5 (900 GB/s), and state how much prefetching saves on each
       interconnect.

[Discussions](https://d2l.discourse.group/)

[Discussions](https://d2l.discourse.group/t/36886)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.7]{.kicker}

ZeRO and Fully Sharded Data Parallelism (FSDP)<br>**16N state bottleneck · ZeRO stages 1–3 · all-gather & reduce-scatter · FSDP lifecycle · communication vs. memory**
:::
:::

::: {.slide title="When the 16N Parameter State Alone Exceeds GPU Memory"}
[Motivation]{.kicker}

Doubling our 1B-parameter GPT from $18$ to $36$ blocks (width $d = 2048$) produces a $1.92\text{B}$-parameter model whose `float32` Adam state ($16N$ bytes) no longer fits on a $24\text{ GiB}$ RTX 4090:

@!sharding-sharding-the-training-state-2

- **Why data parallelism cannot help:** Standard DDP replicates all $16\text{ B/param}$ ($28.6\text{ GiB}$) on every GPU, so four $24\text{ GiB}$ cards run out of memory before processing a single token just like one card does.
:::

::: {.slide title="Why Replicating State Across k GPUs Is Redundant"}
[Core Insight]{.kicker}

::: {.cols .vc}
::: {.col}
- **Weights ($\mathbf{W}_\ell$, $4\text{ B/param}$):** Needed only while block $\ell$ executes its forward or backward pass.
- **Gradients ($\nabla\mathbf{W}_\ell$, $4\text{ B/param}$):** Needed only from block $\ell$'s backward pass until the optimizer step.
- **Adam Moments ($\mathbf{m}, \mathbf{v}$, $8\text{ B/param}$):** Touched **only inside the elementwise Adam update** at the end of the step—each of the $k$ GPUs needs to own and update only $1/k$ of the parameters!
:::

::: {.col .fig .big}
@fig:ch13-perf-state-lifetime
:::
:::
:::

::: {.slide title="The Three ZeRO Stages: Memory per GPU vs. Stage"}
[ZeRO Taxonomy]{.kicker}

::: {.cols .vc}
::: {.col}
- **ZeRO-1 ($8 + 8/k\text{ B/param}$):** Shards Adam moments $\mathbf{m}, \mathbf{v}$ across $k$ GPUs ($21.4\text{ GiB}$ at $k=2$ for our $1.92\text{B}$ GPT).
- **ZeRO-2 ($4 + 12/k\text{ B/param}$):** Shards moments + gradients ($17.9\text{ GiB}$ at $k=2$) at the same $2\,\frac{k-1}{k}S$ comm volume as DDP.
- **ZeRO-3 / FSDP ($16/k\text{ B/param}$):** Shards weights, gradients, and moments ($14.3\text{ GiB}$ at $k=2$, leaving $\sim 10\text{ GiB}$ for activations).
:::

::: {.col .fig .big}
@fig:ch13-perf-zero-stages
:::
:::
:::

::: {.slide title="Collective Primitives Behind Sharded Training"}
[Collective Operations]{.kicker}

::: {.cols .vc}
::: {.col}
- **Ring Allreduce = Reduce-Scatter + All-Gather:** Decomposes a $\frac{2(k-1)}{k}S$ sum into two half-collectives of $\frac{k-1}{k}S$ bytes each.
- **Reduce-Scatter ($\text{Full } S \to \text{Summed } S/k$):** Sums gradients across GPUs and delivers shard $j$ to owner GPU $j$.
- **All-Gather ($\text{Shard } S/k \to \text{Full } S$):** Reconstructs full weights across GPUs just in time for a layer or after a sharded optimizer update.
:::

::: {.col .fig .big}
@fig:ch13-perf-four-collectives
:::
:::
:::

::: {.slide title="ZeRO Stage 1: Sharding Optimizer Moments at Free Comm"}
[ZeRO Stage 1]{.kicker}

::: {.cols .vc}
::: {.col}
- **Insert Adam between the two halves of allreduce:** (1) Reduce-scatter gradients ($\frac{k-1}{k}S$), (2) update owned $1/k$ weights with owned $1/k$ Adam moments, (3) all-gather updated weights ($\frac{k-1}{k}S$).
- **Zero extra communication bytes:** Total traffic per GPU is $\frac{2(k-1)}{k}S$—identical to standard DDP while cutting Adam state memory by $k\times$!
:::

::: {.col .fig .big}
@fig:ch13-perf-zero1-pipeline
:::
:::
:::

::: {.slide title="ZeRO Stage 2: Streaming Reduce-Scatter of Gradients"}
[ZeRO Stage 2]{.kicker}

Why hold all $4N$ bytes of gradients until the end of the backward pass if each GPU only updates $1/k$ of the parameters? **ZeRO Stage 2** fuses gradient release into the backward pass:

- **Bucketed reduce-scatter during `backward()`:** As soon as backpropagation finishes computing the gradients for a block or bucket, the GPUs immediately launch a **reduce-scatter** on that bucket.
- **Immediate deallocation of non-owned shards:** Each GPU retains only the summed $1/k$ gradient slice that it owns and immediately frees the other $(k-1)/k$ gradient slices before the next backward block runs.
- **Same communication volume as DDP:** Across the entire backward pass and post-step weight all-gather, each GPU still transmits $\frac{2(k-1)}{k}S$ bytes total, while never materializing the full $4N$ gradient tensor at once.
:::

::: {.slide title="ZeRO Stage 3 (FSDP): Block Weight Gather and Release"}
[Fully Sharded Data Parallel]{.kicker}

![](../img/mdl-perf-fsdp-lifecycle.svg){width=72%}

- **Sharding the weights ($16N/k$ total state):** Between uses, each GPU stores only $1/k$ of every block's weights.
- **Per-block lifecycle:** Before block $\ell$ runs in the **forward pass**, an **all-gather** assembles its full weights in a temporary buffer, which is freed as soon as block $\ell$ finishes. In the **backward pass**, a second **all-gather** re-assembles block $\ell$'s weights, followed by a **reduce-scatter** of block $\ell$'s gradients and immediate weight release.
:::

::: {.slide title="FSDP Communication Cost: 1.5× Data-Parallel Volume"}
[Communication Model]{.kicker}

::: {.cols .vc}
::: {.col}
$$t_{\textrm{comm (FSDP)}} \;=\; \frac{3(k-1)}{k}\,S\,\beta_{\textrm{link}}$$

- **$1.5\times$ DDP traffic:** Because FSDP frees gathered weights after the forward pass, it pays one extra all-gather in the backward pass ($3$ collective phases vs. $2$).
- **Scaling law as $k$ grows:** Adding GPUs divides steady-state memory ($16N/k \downarrow$), while per-GPU communication rises from $1.5\,S$ ($k=2$) toward $3S$ ($k \to \infty$).
:::

::: {.col .fig}
@fig:ch13-perf-fsdp-comm-scaling
:::
:::
:::

::: {.slide title="Measuring All-Gather Bandwidth on Two RTX 4090s"}
[Link Calibration]{.kicker}

Before running our 36-block GPT on $k = 2$ GPUs, we calibrate $\beta_{\textrm{link}}$ for a $256\text{ MiB}$ all-gather across two RTX 4090s:

@!sharding-memory-and-communication-1

- **Consistent host-staged link rate:** Each GPU sends $(k-1)/k = 1/2$ of the $256\text{ MiB}$ output ($128\text{ MiB}$) at a bus bandwidth of $\sim 2.2\text{--}2.3\text{ GB/s}$ ($\beta_{\textrm{link}} \approx 0.43\text{--}0.45\text{ ns/B}$), matching the bare allreduce link rate from §13.6.
:::

::: {.slide title="Predicting Memory, Comm, and Compute for the 1.9B GPT"}
[Step Prediction]{.kicker}

Now we predict the peak memory, communication time, and compute time of one FSDP step of our 36-block, $1.92\text{B}$-parameter GPT ($S = 4N = 7.7\text{ GB}$) on $k = 2$ GPUs with $b = 2$ sequences of $T = 1{,}024$ tokens per GPU and per-block activation checkpointing:

@!sharding-memory-and-communication-2

- **Memory prediction ($\le 17\text{ GiB} < 24\text{ GiB}$):** Sharded state $16N/2 = 14.3\text{ GiB}$ + checkpointed activations ($\sim 1\text{ MiB/token} \times 2{,}048 \approx 2\text{ GiB}$) + 1–2 live gathered blocks ($\sim 0.4\text{ GiB}$) $\approx 16.7\text{ GiB}$.
- **Time prediction ($\sim 5\text{ s}$ communication vs. $0.4\text{ s}$ compute):** At $k = 2$, each GPU transmits $1.5\,S = 11.5\text{ GB}$ over the $2.2\text{ GB/s}$ PCIe link ($\sim 5.0\text{ s}$), dwarfing the $0.4\text{ s}$ of `tf32` arithmetic ($8N$ FLOPs/token with checkpointing).
:::

::: {.slide title="Meta-Device Initialization and fully_shard in PyTorch" only="pytorch"}
[FSDP in PyTorch]{.kicker}

To avoid ever allocating the full $28.6\text{ GiB}$ state on any single GPU during startup, we construct the model on the zero-memory `'meta'` device, shard each block with `fully_shard`, and materialize only local shards via `to_empty`:

```{.python #sharding-exercises-1}
with torch.device('meta'):      # zero-memory shape/dtype construction
    model = d2l.GPT(**config)
for blk in model.blks:          # unit of all-gather & release: 1 Transformer block
    fully_shard(blk)
fully_shard(model)              # shard remaining embeddings and final RMSNorm
model.to_empty(device=device)   # allocate only this rank's 1/k parameter shards
...                             # initialize local shards in-place
opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
```

- **Why `fully_shard` wraps each block individually:** Wrapping at the block boundary ensures that only one $50\text{M}$-parameter block ($\sim 200\text{ MB}$) is all-gathered at a time rather than the entire $1.92\text{B}$-parameter model.
:::

::: {.slide title="Avoiding Host OOM During Multi-Process FSDP Startup" only="pytorch"}
[Initialization Discipline]{.kicker}

When `torchrun` launches $k$ worker processes on a single host machine, naive model initialization can crash host RAM or GPU VRAM before FSDP even starts:

- **Pitfall 1 — Materializing on GPU before `fully_shard`:** Calling `d2l.GPT(**config).to(device)` before `fully_shard` allocates $4N = 7.7\text{ GB}$ on the card; for larger models ($7\text{B+}$), this immediately triggers a CUDA OOM.
- **Pitfall 2 — $k$ copies in host RAM:** If all $k = 8$ processes construct and initialize a $30\text{ GB}$ model in CPU RAM simultaneously, they consume $8 \times 30 = 240\text{ GB}$ of host DRAM.
- **Production pattern:** Construct on `torch.device('meta')` $\to$ apply `fully_shard` per block $\to$ allocate local $1/k$ GPU storage with `model.to_empty(device=device)` $\to$ initialize or stream checkpoints directly into the sharded `DTensor`s.
:::

::: {.slide title="Declarative Parameter and Activation Sharding in JAX" only="jax"}
[FSDP in JAX]{.kicker}

In JAX, FSDP requires zero manual all-gather or reduce-scatter code: we shard each parameter tensor along its first axis divisible by $k$ (`fsdp_sharding`) and constrain each block's activation output to stay batch-sharded (`P('data')`):

::: {.cols .vc}
::: {.col}
```{.python #sharding-exercises-2}
def fsdp_sharding(x):  # shard first axis divisible by k
    for i, n in enumerate(x.shape):
        if n % k == 0:
            return NamedSharding(
                mesh, P(*[None] * i, 'data'))
    return NamedSharding(mesh, P())
```

- Because block weights are sharded on `'data'` and block activations are *also* sharded on `'data'`, XLA automatically emits an all-gather of each block's weights before the matmul and a reduce-scatter of its weight gradients in the backward pass.
:::

::: {.col .fig}
@!sharding-memory-and-communication-9

- **Exact traffic match:** In HLO result sizes ($2 \times$ sent bytes at $k=2$), all-gather produces $2S \approx 15.0\text{ GB}$ (sent $S = 7.5\text{ GB}$) and reduce-scatter produces $S/2 \approx 3.8\text{ GB}$ (sent $3.8\text{ GB}$), totaling $11.3\text{ GB} \approx 1.5\,S$!
:::
:::
:::

::: {.slide title="Inspecting XLA's Planned Per-GPU Memory and Donation" only="jax"}
[Compiler Memory Plan]{.kicker}

Querying `compiled.memory_analysis()` before running a single step confirms that XLA's static memory plan fits comfortably inside $24\text{ GiB}$ per GPU:

@!sharding-memory-and-communication-7

- **Buffer donation (`donate_argnums=(0, 1)`):** Marks the old sharded weight and Adam moment buffers ($10.7\text{ GiB} = 12N/2$) as reusable in place for the updated state; without buffer donation, holding both old and new states simultaneously during the optimizer update would require $2 \times 10.7 = 21.4\text{ GiB}$ and trigger an OOM!
:::

::: {.slide title="Measured FSDP Step: 16 GiB Peak, 90% Communication"}
[Experimental Verification]{.kicker}

Running our 36-block, $1.92\text{B}$-parameter GPT step on two $24\text{ GiB}$ RTX 4090s confirms both our memory and timing predictions:

@!sharding-memory-and-communication-11

- **Memory fits cleanly:** Each GPU peaks at **$\sim 16\text{ GiB}$** (below our $17\text{ GiB}$ bound), training a $28.6\text{ GiB}$-state model that neither card could hold alone.
- **Communication dominates over PCIe:** Each step takes **$\sim 5.5\text{ s}$**, with GPU 0 spending over **$90\%$ of the step ($\sim 5.0\text{ s}$) inside NCCL communication kernels** and only $\sim 0.4\text{ s}$ in arithmetic.
:::

::: {.slide title="FSDP Over Host-Staged PCIe vs. NVLink 5 Fabrics"}
[Interconnect Sensitivity]{.kicker}

::: {.cols .vc}
::: {.col}
- **Host-staged PCIe 4.0 ($\sim 2.2\text{ GB/s}$):** Moving $1.5\,S = 11.5\text{ GB}$ takes $\sim 5.0\text{ s}$ vs. $0.40\text{ s}$ compute, leaving the step deeply link-bound ($\sim 7\%$ compute share).
- **NVLink 4/5 ($450\text{--}900\text{ GB/s}$):** The same $11.5\text{ GB}$ finishes in $13\text{--}26\text{ ms}$, matching or beating block compute time ($14\text{--}32\text{ ms}$).
- **Block prefetching:** Overlapping block $\ell+1$'s all-gather with block $\ell$'s compute hides communication completely when $g \le c$.
:::

::: {.col .fig .big}
@fig:ch13-perf-fsdp-prefetch
:::
:::
:::

::: {.slide title="When to Shard vs. Replicate with Gradient Accumulation"}
[Decision Guide]{.kicker}

::: {.cols .vc}
::: {.col}
- **When $16N$ fits (e.g., 1B GPT on $24\text{ GiB}$):** Use **DDP + gradient accumulation**—GPUs communicate only **once per global step** after all $m$ micro-batches finish.
- **Why FSDP loses over PCIe on accumulated steps:** FSDP frees weights after every block, forcing it to re-gather all weights **twice on every single micro-batch** ($2m\times$ extra PCIe traffic).
- **Escalation ladder:** DDP $\to$ ZeRO-1/2 $\to$ ZeRO-3 (FSDP) only as memory demands require.
:::

::: {.col .fig .big}
@fig:ch13-perf-ddp-vs-fsdp-decision
:::
:::
:::

::: {.slide title="Concept Check: ZeRO Memory, FSDP Traffic, and Links"}
[Check Your Understanding]{.kicker}

Work out the memory and communication counts for three sharding scenarios ($S = 4N$ denotes the byte size of one full copy of `float32` weights or gradients):

1. A model with $N = 4 \times 10^9$ parameters ($16N = 64\text{ GB}$ of `float32` Adam state) is trained across $k = 4$ GPUs. How many gigabytes of training state does each GPU hold at the end of the backward pass under **DDP**, **ZeRO-1**, **ZeRO-2**, and **ZeRO-3 (FSDP)**?
2. Mixed-precision FSDP keeps each GPU's $1/k$ master weight and Adam moment shards in `float32`, but all-gathers weights in `bfloat16` ($2\text{ B}$) and reduce-scatters gradients in `bfloat16` ($2\text{ B}$). How many bytes per GPU does mixed-precision FSDP transmit per step compared to `float32` DDP?
3. As you scale FSDP from $k = 2$ GPUs to $k = 64$ GPUs, by what factor does per-GPU state memory change, and by what factor does per-GPU communication volume per step change?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **DDP ($16N$):** $\mathbf{64\text{ GB}}$; **ZeRO-1 ($(8 + 8/k)N = 10N$):** $\mathbf{40\text{ GB}}$; **ZeRO-2 ($(4 + 12/k)N = 7N$):** $\mathbf{28\text{ GB}}$; **ZeRO-3 / FSDP ($16N/k = 4N$):** $\mathbf{16\text{ GB}}$ per GPU.
2. **Less traffic than `float32` DDP!** Halving the collective dtype to `bfloat16` halves $S_{\textrm{comm}}$ from $4N \to 2N = S/2$, so 3 FSDP phases move $\frac{3(k-1)}{k} \frac{S}{2} = \mathbf{1.5\,\frac{k-1}{k}S}$ bytes—$25\%$ *fewer* bytes than `float32` DDP's $\frac{2(k-1)}{k}S$!
3. **Memory drops by $32\times$** ($16N/2 \to 16N/64$), while **communication per GPU grows by nearly $2\times$** (from $3(1/2)S = 1.5\,S$ at $k=2$ to $3(63/64)S \approx 2.95\,S$ at $k=64$).
:::
:::

::: {.slide title="Summary: ZeRO Stages 1–3 and FSDP Trade-Offs"}
[Key Takeaways]{.kicker}

| Strategy | Sharded Across $k$ GPUs | Memory / Param (`fp32` Adam) | Collectives per Step | Bytes Sent / GPU ($S = 4N$) |
| :--- | :--- | :---: | :--- | :---: |
| **Standard DDP** | Minibatch only | $16\text{ B}$ | 1 Allreduce | $\frac{2(k-1)}{k}S < 2S$ |
| **ZeRO Stage 1** | Optimizer moments ($\mathbf{m}, \mathbf{v}$) | $8 + \frac{8}{k}\text{ B}$ | 1 Reduce-Scatter + 1 All-Gather | $\frac{2(k-1)}{k}S < 2S$ |
| **ZeRO Stage 2** | Moments + Gradients ($\nabla \mathbf{W}$) | $4 + \frac{12}{k}\text{ B}$ | Bucketed Reduce-Scatter + 1 All-Gather | $\frac{2(k-1)}{k}S < 2S$ |
| **ZeRO Stage 3 (FSDP)** | Moments + Grads + Weights ($\mathbf{W}$) | **$\frac{16}{k}\text{ B}$** | 2 All-Gathers + 1 Reduce-Scatter | **$\frac{3(k-1)}{k}S < 3S$** |
| **`bf16`-Comm FSDP** | All state (master shards in `fp32`) | $\frac{16}{k}\text{ B}$ (+ live block) | 2 `bf16` All-Gathers + 1 `bf16` RS | $\frac{1.5(k-1)}{k}S < 1.5S$ |
:::
