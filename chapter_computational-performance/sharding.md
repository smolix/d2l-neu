# Sharding the Training State
:label:`sec_sharding`

<!-- WRITER: new section 13.7, split from multi-gpu-practice.md (structure.md §3.8). The prose below is moved verbatim; the two opening paragraphs repeat each other and merge into one; the predictions, the FSDP experiment (sharding code step), Summary, exercises and deck follow the design. -->

Data parallelism also leaves one cost untouched: every GPU keeps a full
copy of the training state. The GPT of :numref:`sec_memory_precision` with
36 blocks instead of 18 has 1.9 billion parameters. Its weights, their
gradients, and Adam's moment estimates take 29 GiB, more than the 24 GiB
of an RTX 4090. Data parallelism would place these 29 GiB on every one of
our four GPUs, so four GPUs cannot train this model any more than one can.

```{.python .input #sharding-setup}
%%tab jax
# XLA reads its flags once, when JAX starts, so this cell runs before the
# imports. The latency-hiding scheduler orders the step with its memory in
# view. Without it, XLA starts every reduce-scatter of the sharded step as
# soon as its gradient exists and finishes them all at the end, so the full
# gradients of all blocks are alive at once: about 8 GB of temporaries per
# GPU instead of about 5, more than JAX may allocate by default. The
# notebook also runs with NCCL_LOCAL_REGISTER=0: NCCL's buffer registration
# fails on GPUs without peer-to-peer access, such as ours, and would print a
# warning at every collective.
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

<!-- WRITER: the cell below prints the opening's numbers (1.92 billion parameters, 16N = 28.6 GiB; PyTorch: one GPU holds 23.6 GiB; JAX: the bytes_limit JAX may allocate, 17.6 GiB by default, 22.2 GiB under the scheduler's 0.94 fraction). The opening can say "as the cell shows" instead of asserting it. -->

```{.python .input #sharding-sharding-the-training-state-2}
%%tab pytorch
config = dict(vocab_size=50257, num_hiddens=2048, num_heads=16,
              num_blks=36, pos='learned')
k = 2                       # GPUs
assert d2l.num_gpus() >= k, 'This section needs two GPUs'
with torch.device('meta'):  # Shapes and dtypes only, no data
    N = d2l.num_params(d2l.GPT(**config))
total = torch.cuda.mem_get_info()[1]
print(f'{N / 1e9:.2f} billion parameters, 16N = {16 * N / 2**30:.1f} GiB; '
      f'one GPU holds {total / 2**30:.1f} GiB')
```

```{.python .input #sharding-sharding-the-training-state-2}
%%tab jax
config = dict(vocab_size=50257, num_hiddens=2048, num_heads=16,
              num_blks=36, pos='learned')
k = 2                       # GPUs
assert jax.device_count() >= k, 'This section needs two GPUs'
key = d2l.get_key()
N = d2l.num_params(nnx.eval_shape(lambda: d2l.GPT(**config,
                                                  rngs=nnx.Rngs(key))))
limit = jax.devices()[0].memory_stats()['bytes_limit']
print(f'{N / 1e9:.2f} billion parameters, 16N = {16 * N / 2**30:.1f} GiB; '
      f'JAX may allocate {limit / 2**30:.1f} GiB of one GPU')
```

## Partitioning the Optimizer State, Gradients, and Weights
:label:`subsec_mgp-fsdp`

Data parallelism makes training faster, but it makes no room for a larger
model: every GPU holds all of the weights, the gradients, and the
optimizer state. With Adam in fp32 these take 16 bytes per parameter
(:numref:`subsec_mp-anatomy`), so a 24 GiB card holds at most 1.6 billion
parameters before it stores a single activation. The 36-block GPT from the
start of this section has 1.9 billion parameters. At 16 bytes each they
take 29 GiB, which no single GPU of ours holds, and data parallelism would
need these 29 GiB on every GPU.

Each GPU needs a full copy of the weights only while it computes with them,
and it never needs the full gradient or the full optimizer state. The ZeRO
family of methods :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` removes the
redundant copies in three stages. Stage 1 partitions the optimizer state:
each GPU keeps Adam's moment estimates for $1/k$ of the parameters and
updates only those parameters. Stage 2 also partitions the gradients, and
stage 3 also partitions the weights. :numref:`tab_mgp-zero` lists what each
GPU then holds.

:Training state per GPU with Adam in fp32 on $k$ GPUs, and for the GPT with 36 blocks and 1.9 billion parameters on $k = 4$.
:label:`tab_mgp-zero`

| each GPU holds | weights | gradients | Adam state | bytes per parameter | GPT, $k = 4$ |
|---|---|---|---|---|---|
| data parallelism | all | all | all | 16 | 28.6 GiB |
| ZeRO stage 1 | all | all | $1/k$ | $8 + 8/k$ | 17.9 GiB |
| ZeRO stage 2 | all | $1/k$ | $1/k$ | $4 + 12/k$ | 12.5 GiB |
| ZeRO stage 3 (FSDP) | $1/k$ | $1/k$ | $1/k$ | $16/k$ | 7.1 GiB |

The collectives follow from the identity of :numref:`subsec_mg-ring`: an
allreduce is a reduce-scatter followed by an all-gather. In stages 1 and
2 the reduce-scatter leaves each GPU with the summed gradient of the
parameters it owns, which is all that its optimizer needs. Each GPU then
updates its $1/k$ of the parameters, and an all-gather assembles the
updated parameters on every GPU. The two phases move the same bytes as the
allreduce of data parallelism, now separated by the update. Stages 1 and 2
therefore reduce the memory per GPU at no cost in communication. For our
GPT, stage 1 on four GPUs already brings the training state below the
24 GiB of one card.

Stage 3 is known as *fully sharded data parallelism* (FSDP)
:cite:`Zhao.Gu.Varma.ea.2023`. Each GPU keeps only its shard of every
block's weights, and an all-gather assembles a block's weights just before
the block computes and releases them right after
(:numref:`fig_fsdp_lifecycle`). The backward pass gathers each block's
weights once more, releases them after the block's backward pass, and
reduce-scatters the block's gradients. A step thus performs two all-gathers
and one reduce-scatter of $N$ bytes, three phases where an allreduce has
two, so FSDP communicates 1.5 times as much as data parallelism. No full
copy of a block lives longer than the block's computation, so the memory
per GPU is $16/k$ bytes per parameter plus the activations and one or two
gathered blocks.

![The life of one block's weights under FSDP on one GPU. Between uses the
block is sharded, and each GPU holds $1/k$ of it; the other blocks stay
sharded throughout. An all-gather assembles the whole block for its forward
pass, after which it is released. A second all-gather assembles it for the
backward pass, after which it is released again, and a reduce-scatter
leaves each GPU with its share of the
gradients.](../img/mdl-perf-fsdp-lifecycle.svg)
:label:`fig_fsdp_lifecycle`

:numref:`tab_collectives` lists the four collectives that data-parallel and
sharded training use.

:The collective operations behind data-parallel and sharded training.
:label:`tab_collectives`

| collective | what every GPU ends with | where it appears |
|---|---|---|
| allreduce | the full sum | gradients in data parallelism |
| reduce-scatter | one part of the sum | gradients in ZeRO and FSDP |
| all-gather | every part, concatenated | weights in ZeRO and FSDP |
| all-to-all | a different part from each GPU | tokens routed to the experts of a mixture of experts (:numref:`sec_moe`) |

:begin_tab:`pytorch`
In PyTorch, `fully_shard` from `torch.distributed.fsdp` applies FSDP to a
module. It is called once for every transformer block and once for the
whole model, so that every block becomes a unit that is gathered and
released on its own. The older `FullyShardedDataParallel` wrapper still
works, and PyTorch's documentation recommends `fully_shard` for new code.
Stage 1 alone is available as
`torch.distributed.optim.ZeroRedundancyOptimizer`. DDP itself needs more
than the first row of :numref:`tab_mgp-zero`: its buckets hold a second
copy of the gradients, 4 more bytes per parameter, unless the argument
`gradient_as_bucket_view=True` makes the gradients views into the buckets.
:end_tab:

:begin_tab:`jax`
In JAX, sharding the training state is a change of placement, as data
parallelism was. Instead of replicating the weights and the optimizer state
with `P()`, we split one axis of each array across the mesh, for example
with `P('data')` on the leading axis. Splitting the weights alone, however,
lets the partitioner choose a different scheme, such as matrix products on
split weights followed by an allreduce of the activations. FSDP's all-gathers and
reduce-scatters appear only when every block's activations are also
constrained to stay split along the batch, with
`jax.lax.with_sharding_constraint(x, P('data'))`.
:end_tab:

## Memory and Communication

Sharding reduces memory and adds communication, and the cost model says
how long the communication takes. For our GPT, the fp32 gradients take
$N$ = 7.7 GB, four bytes per parameter. On $k$ GPUs, a ring allreduce
makes every GPU send $2(k-1)N/k$ bytes and receive as many
(:numref:`subsec_mg-ring`), $1.5N$ in each direction at $k = 4$ against
$N$ at $k = 2$. If every GPU keeps the bandwidth $\beta$ of about 5 GB/s
that we measured on two, one allreduce of data parallelism takes
$3N/\beta$, about 5 s, and the three phases of FSDP take $4.5N/\beta$,
about 7 s.

<!-- WRITER: this paragraph is false for the run, which is at k = 2 (structure.md 3.8): FSDP sends (k-1)S/k per GPU in each of its three phases, S = 4N = 7.7 GB the fp32 weights or gradients; at k = 2 that is S/2 = 3.8 GB per phase and 11.5 GB per step. beta_link is measured by the cell below on an all-gather, the collective FSDP uses (dev, GPUs 0 and 1: 2.2-2.3 GB/s in PyTorch, 2.1-2.2 in JAX), not "about 5 GB/s" (the old beta, an allreduce's 2S/t). The symbol N names the parameter count in the cell output; the prose used N for gradient bytes (MGP-15): write S for bytes. -->

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
print(json.dumps({'beta_link': (k - 1) / k * y.nbytes / t}))  # Sent per GPU
dist.destroy_process_group()
'''
beta_link = min(r['beta_link'] for r in d2l.torchrun(GATHER, k))
print(f'beta_link = {beta_link / 1e9:.1f} GB/s in an all-gather on {k} GPUs')
```

```{.python .input #sharding-memory-and-communication-1}
%%tab jax
mesh = jax.make_mesh((k,), ('data',), devices=jax.devices()[:k],
                     axis_types=(jax.sharding.AxisType.Auto,))
x = jnp.zeros(2**26, device=NamedSharding(mesh, P('data')))  # 256 MiB
# Asking for the result on every GPU makes the compiler gather the parts
gather = jax.jit(lambda x: x, out_shardings=NamedSharding(mesh, P()))
t = d2l.Benchmark(lambda: gather(x), repeats=10).time
beta_link = (k - 1) / k * x.nbytes / t  # Sent per GPU
del x                       # Frees its memory before the training step
print(f'beta_link = {beta_link / 1e9:.1f} GB/s in an all-gather on {k} GPUs')
```

The computation of a step is much shorter. The forward and the backward
pass take about six floating-point operations per parameter and token
(:numref:`sec_scaling-laws`). Two sequences of 1,024 tokens per GPU, which
fit next to the 7.1 GiB of stage 3 with the activation checkpointing of
:numref:`subsec_mp-checkpointing`, take about
$6 \cdot 1.9 \cdot 10^9 \cdot 2048 \approx 2.4 \times 10^{13}$
operations. At the RTX 4090's full tf32 rate of about 80 TFLOP/s that is
0.3 s, and 0.4 s with the forward pass that checkpointing repeats. On our
machine, sharding lets the model train on four GPUs, at a few percent of
the speed that its computation would allow. Over NVLink 5, at 900 GB/s in
each direction, the same three phases take about 20 ms, and FSDP is the
standard way to train models whose state does not fit on one GPU.

<!-- WRITER: the paragraph above quotes four GPUs and "a few percent"; the run is on k = 2. The prediction cell below prints the three numbers the prose should derive before the run: state 16N/k = 14.3 GiB per GPU; communication 3(k-1)S/(k beta_link) (dev: 5.0 s PyTorch, 5.3-5.4 s JAX); arithmetic 8N bT/P at the tf32 peak, 6N per token times 4/3 for the forward pass that checkpointing repeats (0.38 s for 2,048 tokens per GPU). Activations with per-block checkpointing: each block keeps its input, 8 KiB per token in fp32, 36 x 2,048 x 8 KiB = 0.6 GB, plus the logits of 2 x 1,024 tokens over 50,257 words, 0.4 GB per copy. -->

```{.python .input #sharding-memory-and-communication-2}
%%tab pytorch, jax
b, T = 2, 1024        # Sequences of T tokens per GPU and step
S = 4 * N             # Bytes of the fp32 weights, and of the gradients
peak = {'tf32': 82.6e12, 'bf16': 165.2e12}  # FLOP/s of the RTX 4090
print(f'state per GPU, 16N/k: {16 * N / k / 2**30:.1f} GiB')
print(f'communication, 3(k-1)S/(k beta_link): '
      f'{3 * (k - 1) / k * S / beta_link:.1f} s per step')
print(f'arithmetic at the peak, 8N bT/P: '
      f'{8 * N * b * T / peak["tf32"]:.2f} s per step')
```

<!-- WRITER (PyTorch tab note): the script below builds the model on the meta device, wraps every block so that torch.utils.checkpoint recomputes it (Checkpointed), calls fully_shard once per block and once for the whole model, and only then allocates this GPU's shards with to_empty and initializes them as d2l.GPT does (norms to one, embeddings N(0, 0.02^2), then init_weights); random initialization of a sharded tensor draws the values that the whole tensor would get, given the same seed in every process. fused=True runs Adam's update as one kernel over all tensors instead of the default foreach update, whose temporaries are as large as the gradients (about 3.6 GiB here). The JSON line: peak memory, three timed steps after one warm-up, and the device seconds of NCCL kernels in one profiled step. -->

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

```{.python .input #sharding-memory-and-communication-4}
%%tab pytorch
ranks = d2l.torchrun(TRAIN_FSDP, k, json.dumps(config), b, T)
peak_bytes = max(r['peak'] for r in ranks)
times, comm = ranks[0]['times'], ranks[0]['comm']  # GPU 0
```

<!-- WRITER (JAX tab note): the jitted create builds the model and the optimizer under a sharding constraint on their state, so that each GPU computes only its own shards (device memory after create: 10.7 GiB per GPU, the 12N/k bytes of weights and moment estimates); the key is an argument, since a constant key would make XLA compute the weights while compiling (dev: 312 s instead of 71 s). The 36 blocks still take about a minute to compile. fsdp_sharding splits the first axis that k divides: the leading axis of every weight matrix and norm; the embedding table, whose 50,257 rows k = 2 does not divide, along its width. The loss function repeats d2l.GPT's forward pass with two changes: nnx.remat recomputes each block in the backward pass, and with_sharding_constraint keeps each block's output split by batch. Donation lets the step write the new state into the buffers of the old (alias 10.7 GiB in the memory analysis); without it the step would hold both, 2 x 10.7 GiB plus temporaries, more than a card. -->

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
def create(key):  # Passed in: XLA would fold a constant key while compiling
    model = d2l.GPT(**config, rngs=nnx.Rngs(key))
    opt = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
    state = nnx.state((model, opt))
    nnx.update((model, opt), jax.lax.with_sharding_constraint(
        state, jax.tree.map(fsdp_sharding, state)))
    return model, opt

model, opt = create(d2l.get_key())  # Each GPU computes only its shards
```

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

```{.python .input #sharding-memory-and-communication-9}
%%tab jax
for kind, mb in sorted(collective_megabytes(compiled).items(),
                       key=lambda kv: -kv[1]):
    print(f'{kind}: {mb / 1e3:.2f} GB')
```

```{.python .input #sharding-memory-and-communication-10}
%%tab jax
train_step(model, opt, X).block_until_ready()  # Warm-up, finished
times = [d2l.Benchmark(lambda: train_step(model, opt, X), warmup=0,
                       repeats=1).time for _ in range(3)]
kernels = d2l.kernel_times(lambda: train_step(model, opt, X), steps=1)
comm = sum(ms for name, ms in kernels.items() if 'nccl' in name.lower()) / 1e3
peak_bytes = jax.devices()[0].memory_stats()['peak_bytes_in_use']
```

```{.python .input #sharding-memory-and-communication-11}
%%tab pytorch, jax
print(f'per GPU: peak memory {peak_bytes / 2**30:.1f} GiB; '
      f'{np.median(times):.2f} s per step ({min(times):.2f} to '
      f'{max(times):.2f}), of which GPU 0 spends {comm:.2f} s in '
      f'communication kernels')
```

<!-- WRITER (measured, dev runs on GPUs 0 and 1, 2026-11-20, not captured; the capture runs on the same pair): PyTorch beta_link 2.2-2.3 GB/s, predicted communication 5.0 s; peak 16.0 GiB per GPU (prediction about 17); 5.72 s per step (5.71 to 5.75; a second run 5.70 to 5.73), 5.69 s of it in NCCL kernels on GPU 0. JAX beta_link 2.1-2.2 GB/s, predicted 5.3-5.4 s; weights and moments 10.7 GiB, all of it donated; temporaries 4.3 GiB in one run, 5.2 GiB in another (XLA's schedule differs between compilations), peak 15.1-16.1 GiB; 5.40 s per step (5.38 to 5.55), 5.09 s in NCCL kernels. Compiled collectives (JAX): all-gather results 14.92 GB = the 36 blocks' weights twice (2 x 7.25 GB) plus the embedding table once (0.41 GB); reduce-scatter results 3.83 GB = S/k, so the full gradient S was reduce-scattered once; all-to-all 0.03 GB and a scalar all-reduce come from the embedding table, which is split along its width (fallback a of the design: the blocks follow FSDP exactly, the table does not). Bytes each GPU sends: 14.92/2 + 3.83 = 11.3 GB against 3 S/2 = 11.5 GB predicted. Without the activation constraint (dev, 4 blocks) the partitioner mixes allreduces, all-to-alls and collective-permutes and moves 6.7 GB instead of 2.7 GB: the JAX tab note's claim is true (good exercise material). So: the step is the communication, both tabs, within 15% of 3(k-1)S/(k beta_link); arithmetic at the tf32 peak is 0.38 s, 1/15 (PyTorch) to 1/14 (JAX) of the step, not "a twelfth" (the protected sentence takes the measured fraction). Not printed: the device time of the computing kernels rises from about 0.6 s (12-block model on one GPU, scaled to 36) to 2.2 s (JAX) and 3.5 s (PyTorch) while NCCL kernels run beside them; it overlaps the communication and does not set the step. -->

The arithmetic also says when to shard. A model whose training state fits
on one GPU next to its activations trains fastest with data parallelism.
On a 24 GiB card with Adam in fp32 that limit lies at about a billion
parameters, fewer for long sequences. The 18-block GPT of
:numref:`sec_memory_precision` sits at this limit. Stage 3 would shrink
its 15.1 GiB per GPU to 3.8 GiB at $k = 4$ and free 11 GiB for
activations, but on our machine the extra communication would take far
more time than the larger micro-batches save.

Past the limit, ZeRO stages 1 and 2 reduce the memory at no cost in
communication, and stage 3 removes the remaining copies with 1.5 times
the communication. Splitting the weight matrices of every layer across
GPUs instead, *tensor parallelism* (exercise :exref:`ex_mgp-tensor`), and
the forms of parallelism that span several machines belong to the
Language Models part.

## Summary

ZeRO and FSDP divide the 16 bytes per parameter of the training state among
the GPUs, stages 1 and 2 at no cost in communication and stage 3 at 1.5 times
the communication of data parallelism.

## Exercises

<!-- WRITER: exercises moved from multi-gpu-practice.md (old 3, 6, 8); exercise 2 still starts from TRAIN_DDP / train_step of sec_multi_gpu_concise, and exercise 1 part 3 still says "the beta that you measured" on four GPUs. -->

1. **Cost of sharding.** ● Count communication per GPU and step in units
   of $N$, the bytes of one full set of fp32 gradients.
    1. A variant of ZeRO stage 3 keeps every gathered block from its
       forward pass until its backward pass. Which collective does this
       remove, and how much of the memory saving of stage 3 does it give
       up, in bytes per parameter of :numref:`tab_mgp-zero`?
    1. Mixed-precision FSDP gathers the weights in bf16 and reduce-scatters
       the gradients in bf16, while every GPU keeps its shard of the fp32
       weights and of Adam's state. Compare its bytes per step with those
       of data parallelism in fp32, and its memory per parameter with the
       last row of :numref:`tab_mgp-zero`.
    1. For the GPT with 36 blocks and 1.9 billion parameters on our four
       GPUs, and with the $\beta$ that you measured, find the number of
       tokens per GPU and step at which the computation, 6 operations
       per parameter and token at 80 TFLOP/s, takes as long as the communication of
       stage 3. How many sequences of 1,024 tokens is that, and do their
       activations fit next to the training state
       (:numref:`subsec_mp-anatomy`)? How does gradient accumulation
       change the answer when FSDP gathers the weights anew for every
       micro-batch?


:begin_tab:`pytorch`
2. [extended] **Sharded optimizer state.** ● Implement ZeRO stage 1
    :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` by hand and measure what it
    saves. Start from `TRAIN_DDP` with `torch.optim.Adam` in place of SGD.
    1. Assign every parameter tensor to one owner process, so that the
       processes own nearly equal numbers of values, and give each process
       an Adam optimizer over its own tensors only. After the backward pass
       and the owners' steps, send each tensor from its owner to all
       processes with `dist.broadcast`. Check that after a few steps the
       parameters match those of DDP with a full Adam from the same
       initialization, and compare with
       `torch.distributed.optim.ZeroRedundancyOptimizer`.
    1. Read `torch.cuda.memory_allocated` after initialization, before the
       first optimizer step, and after it, with and without sharding at
       $k = 2$ and $k = 4$, and account for the differences with
       :numref:`tab_mgp-zero`; then read `torch.cuda.max_memory_allocated`
       after the second step and compare its change under sharding with
       the change in live memory.
    1. Measure the step time with and without sharding. DDP's allreduce
       hands every process the full summed gradient, of which stage 1 uses
       only the part that the process owns. Count the bytes per GPU that the
       allreduce and the broadcasts of your implementation send together,
       and compare them with the reduce-scatter and all-gather of stage 1
       in :numref:`subsec_mgp-fsdp`.

    *Adapted from Stanford CS336,
    [Assignment 2 (Systems)](https://github.com/stanford-cs336/assignment2-systems),
    Problems optimizer_state_sharding and
    optimizer_state_sharding_accounting.*

3. [code] **Tensor versus data parallelism.** ● Consider the MLP block
    $\mathbf{Y} = \phi(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$ with
    $\mathbf{X} \in \mathbb{R}^{n \times d}$ holding the $n$ tokens of one
    step, $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$,
    $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$, and an elementwise
    activation $\phi$, on $k$ devices. Tensor
    parallelism (:numref:`fig_splitting`) splits $\mathbf{W}_1$ by columns and
    $\mathbf{W}_2$ by rows :cite:`shoeybi2019megatron`.
    1. Show that each device can compute, without communication, an
       $n \times d$ partial result from its columns of $\mathbf{W}_1$ and
       its rows of $\mathbf{W}_2$, and that $\mathbf{Y}$ is the sum of the
       $k$ partial results. Which collective forms that sum, and which
       collective does the backward pass need for the gradient with
       respect to $\mathbf{X}$?
    1. Data parallelism on the same $k$ devices allreduces the gradients
       of $\mathbf{W}_1$ and $\mathbf{W}_2$ once per step. Count the values
       that each scheme sends per device and step, and find the number of
       tokens $n$ below which tensor parallelism sends less.
    1. Tensor parallelism is in practice confined to the GPUs of one NVLink
       domain even where it sends less. Explain why, using when each
       scheme's communication happens relative to the computation that
       needs its result.
    1. In a script launched with `d2l.torchrun`, apply `parallelize_module`
       to `nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))`
       on `init_device_mesh("cuda", (k,))`, with `ColwiseParallel()` for
       submodule `"0"` and `RowwiseParallel()` for submodule `"2"`. Count
       the collectives of the forward and the backward pass with
       `CommDebugMode` from `torch.distributed.tensor.debug`, once for an
       input with `requires_grad=True` and once without, and explain the
       difference.

    *Adapted from Stanford CS336,
    [Assignment 2 (Systems)](https://github.com/stanford-cs336/assignment2-systems),
    Problems data_parallel_calcs and tp_calcs.*
   :exlabel:`ex_mgp-tensor`
:end_tab:

:begin_tab:`jax`
2. [extended] **Sharded optimizer state.** ● Implement ZeRO stage 1
    :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` declaratively and measure
    what it saves. Use `optax.adam` in `nnx.Optimizer` instead of SGD, keep
    the model replicated, and place every leaf of `nnx.state(opt)` with a
    `NamedSharding` that splits the first of its axes whose length is
    divisible by $k$, replicating the leaves that have no such axis.
    1. Run `train_step` once and show the layout of a weight afterwards.
       Stage 1 needs replicated parameters at the start of every step; if
       the compiler did not keep them replicated, restore the layout inside
       the step with `jax.sharding.reshard` and a replicated
       `NamedSharding`. Check that after a few steps the parameters match
       those of replicated Adam from the same initialization.
    1. Read the collectives of the compiled step. Which collective is new
       compared with the replicated step, and is the gradient sum an
       allreduce or a reduce-scatter? Compare the bytes per GPU of the
       compiled step with the reduce-scatter and all-gather that stage 1
       needs (:numref:`subsec_mgp-fsdp`).
    1. In fresh processes, read `'bytes_in_use'` and `'peak_bytes_in_use'`
       from `jax.devices()[0].memory_stats()` after a completed step, and
       measure the step time, with and without sharding at $k = 2$ and
       $k = 4$. Account for the differences in live memory with
       :numref:`tab_mgp-zero`, and compare them with the change in the
       peak.

    *Adapted from Stanford CS336,
    [Assignment 2 (Systems)](https://github.com/stanford-cs336/assignment2-systems),
    Problems optimizer_state_sharding and
    optimizer_state_sharding_accounting.*

3. [code] **Tensor versus data parallelism.** ● Consider the MLP block
    $\mathbf{Y} = \phi(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$ with
    $\mathbf{X} \in \mathbb{R}^{n \times d}$ holding the $n$ tokens of one
    step, $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$,
    $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$, and an elementwise
    activation $\phi$, on $k$ devices. Tensor
    parallelism (:numref:`fig_splitting`) splits $\mathbf{W}_1$ by columns and
    $\mathbf{W}_2$ by rows :cite:`shoeybi2019megatron`.
    1. Show that each device can compute, without communication, an
       $n \times d$ partial result from its columns of $\mathbf{W}_1$ and
       its rows of $\mathbf{W}_2$, and that $\mathbf{Y}$ is the sum of the
       $k$ partial results. Which collective forms that sum, and which
       collective does the backward pass need for the gradient with
       respect to $\mathbf{X}$?
    1. Data parallelism on the same $k$ devices allreduces the gradients
       of $\mathbf{W}_1$ and $\mathbf{W}_2$ once per step. Count the values
       that each scheme sends per device and step, and find the number of
       tokens $n$ below which tensor parallelism sends less.
    1. Tensor parallelism is in practice confined to the GPUs of one NVLink
       domain even where it sends less. Explain why, using when each
       scheme's communication happens relative to the computation that
       needs its result.
    1. Build the mesh with `jax.make_mesh` and
       `axis_types=(jax.sharding.AxisType.Auto,)`, so that the compiler
       chooses the layout of each product. Place $\mathbf{W}_1$ with
       `P(None, 'data')`, $\mathbf{W}_2$ with `P('data', None)`, and
       $\mathbf{X}$ replicated, and show the layout of $\mathbf{W}_1$ with
       `jax.debug.visualize_array_sharding`. Then `jit` the block and its
       gradient with respect to all three inputs, and find the collectives
       of the first sub-problem in the compiled programs.

    *Adapted from Stanford CS336,
    [Assignment 2 (Systems)](https://github.com/stanford-cs336/assignment2-systems),
    Problems data_parallel_calcs and tp_calcs.*
   :exlabel:`ex_mgp-tensor`
:end_tab:

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.7]{.kicker}

Sharding the Training State<br>**ZeRO stages 1–3 · FSDP · memory and communication**
:::
:::

::: {.slide title="Data Parallelism Keeps k Copies of 16 Bytes per Parameter"}
[Sharding the training state]{.kicker}

Per GPU, Adam in fp32, $k$ GPUs; the 36-block GPT (1.9B) on $k = 4$:

| each GPU holds | weights | grads | Adam | bytes/param | GPT |
|---|---|---|---|---|---|
| data parallel | all | all | all | 16 | 28.6 GiB |
| ZeRO-1 | all | all | $1/k$ | $8 + 8/k$ | 17.9 GiB |
| ZeRO-2 | all | $1/k$ | $1/k$ | $4 + 12/k$ | 12.5 GiB |
| ZeRO-3 (FSDP) | $1/k$ | $1/k$ | $1/k$ | $16/k$ | 7.1 GiB |

::: {.d2l-note .rule}
On a 24 GiB card, data parallelism cannot train it; ZeRO-1 already can.
:::
:::

::: {.slide title="ZeRO-1 and ZeRO-2 Move the Same Bytes as an Allreduce"}
[Sharding the training state]{.kicker}

allreduce = reduce-scatter + all-gather. Split them:

1. reduce-scatter: each GPU gets the summed gradient of the parameters it
   owns;
2. each GPU updates its $1/k$ with its $1/k$ of Adam's state;
3. all-gather: the updated weights reach every GPU.

::: {.d2l-note .rule}
Same two phases, same bytes: less memory at no cost in communication.
:::
:::

::: {.slide title="FSDP Gathers Each Block Twice: 1.5× the Communication"}
[Sharding the training state]{.kicker}

![](../img/mdl-perf-fsdp-lifecycle.svg){width=80%}

All-gather (forward), all-gather (backward), reduce-scatter: three phases
where an allreduce has two. Memory: $16/k$ bytes per parameter plus a gathered block or two.
:::

::: {.slide title="Check: The 36-Block GPT with FSDP on Our Four GPUs"}
[Check]{.kicker}

1.9B parameters, gradients $N$ = 7.7 GB, $k = 4$, $\beta \approx$ 5 GB/s
(measured on two GPUs), 6 operations per parameter and token at 80 TFLOP/s, 2,048
tokens per GPU and step. Communication and computation per step?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Ring at $k = 4$: $1.5N$ each way per GPU, so an allreduce
takes $3N/\beta$ and FSDP $4.5N/\beta \approx 7$ s. Compute:
$6 \cdot 1.9 \cdot 10^9 \cdot 2048 / 8 \cdot 10^{13} \approx 0.3$ s. It
trains, at a few percent of its compute speed; NVLink 5: about 20 ms.
:::
:::
:::
