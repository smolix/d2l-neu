# Sharding the Training State
:label:`sec_sharding`

In :numref:`sec_memory_precision` we counted what a training step keeps in
memory and found one part that none of its techniques reduces: 16 bytes
for every parameter, for the weight, its gradient, and Adam's two moment
estimates. For the billion-parameter GPT of that section these bytes fill
15 GiB of a 24 GiB card in fp32 before the model has read a single token.
The two sections on data parallelism did nothing about them. They copied
all 16 bytes onto every GPU.

Let us make that GPT twice as deep, with 36 blocks instead of 18. It then
has almost two billion parameters, and its training state no longer fits
on any of our cards. Data parallelism on four GPUs cannot train it any
more than one GPU can. Can several GPUs train a model that none of them
can hold?

They can, because a GPU needs most of its state only for a moment. It
needs the full weights of a block only while it computes with that block,
and it never needs the full gradient or the full optimizer state at all.
Each of $k$ GPUs can therefore keep $1/k$ of everything, gather a block's
weights when the block computes, and scatter each gradient as soon as it
exists. The state then fits, but every step must move weights and
gradients between the GPUs. On our machine, whose GPUs talk to each other
through host memory, the question is how long that takes. We first divide
the state, then predict the memory and the time of a step for the 36-block
GPT on two GPUs, and finally run the step and compare.

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

The cells of this section need two GPUs. Their stored outputs come from
two RTX 4090s without peer-to-peer access, which exchange data through host
memory. We count the parameters $N$ of the 36-block GPT without allocating
memory for them, and compare its $16N$ bytes of training state with the
memory of one GPU.

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

At 28.6 GiB the state alone is several GiB larger than one GPU offers, and
the activations still have to fit beside it.

## Partitioning the Optimizer State, Gradients, and Weights
:label:`subsec_mgp-fsdp`

Let us look at when a GPU in data parallelism actually uses each part of
its state. The weights of a block are used twice per step, in the block's
forward pass and in its backward pass. A gradient is used once, by the
optimizer, and Adam's moment estimates are used only inside the update.
Moreover, all $k$ GPUs compute the same update from the same summed
gradient, so they hold $k$ identical copies of the optimizer state and do
the same arithmetic $k$ times. The ZeRO family of methods
:cite:`Rajbhandari.Rasley.Ruwase.ea.2020`, developed at Microsoft to
train models with tens of billions of parameters on GPUs of 32 GB,
removes the redundant copies in three stages. Stage 1 partitions the
optimizer state: each GPU keeps Adam's moment estimates for $1/k$ of the
parameters and updates only those parameters. Stage 2 also partitions the
gradients, and stage 3 also the weights. Stage 3 is also known as *fully
sharded data parallelism* (FSDP) :cite:`Zhao.Gu.Varma.ea.2023`.
:numref:`tab_mgp-zero` lists what each GPU then holds at the end of the
backward pass.

:Training state per GPU at the end of the backward pass, with Adam in fp32 on $k$ GPUs, and for the GPT with 36 blocks and 1.92 billion parameters on $k = 2$.
:label:`tab_mgp-zero`

| each GPU holds | weights | gradients | Adam state | bytes per parameter | GPT, $k = 2$ |
|---|---|---|---|---|---|
| data parallelism | all | all | all | 16 | 28.6 GiB |
| ZeRO stage 1 | all | all | $1/k$ | $8 + 8/k$ | 21.4 GiB |
| ZeRO stage 2 | all | $1/k$ | $1/k$ | $4 + 12/k$ | 17.9 GiB |
| ZeRO stage 3 (FSDP) | $1/k$ | $1/k$ | $1/k$ | $16/k$ | 14.3 GiB |

During the backward pass a GPU briefly holds more than the table says: in
stage 2 a bucket of gradients that have not yet been summed and scattered,
and in stage 3 also one or two gathered blocks. On two GPUs, stage 1
already brings the 36-block GPT below 24 GiB, but it leaves less than
3 GiB of the card for everything else. Stage 3 leaves almost 10.

:begin_tab:`pytorch`
DDP needs more than the first row of the table: its buckets hold a second
copy of the gradients, 4 more bytes per parameter, unless the argument
`gradient_as_bucket_view=True` makes the gradients views into the buckets.
:end_tab:

A GPU in stage 1 updates only its $1/k$ of the parameters, so how do the
other GPUs get the updated values? The identity of
:numref:`subsec_mg-ring` answers it: an allreduce is a reduce-scatter
followed by an all-gather. The reduce-scatter leaves each GPU with the
summed gradient of the parameters it owns, which is all that its optimizer
needs. Each GPU then updates its $1/k$ of the parameters, and an
all-gather assembles the updated parameters on every GPU. The two phases
move the same bytes as the allreduce of data parallelism, now separated by
the update, so stage 1 reduces the memory per GPU at no cost in bytes. It
is not quite free in time. The all-gather now sits between the update and
the next forward pass, and unless the implementation overlaps it with
that forward pass, block by block, or with the backward pass, bucket by
bucket, it adds to the step.

Stage 2 needs a word of explanation. Every GPU runs a full backward pass,
which produces every gradient, so how can a GPU hold only $1/k$ of them? It
never holds them all at once. The tools of :numref:`sec_multi_gpu_concise`
already exchange the gradients in *buckets*, groups of tensors sent as
soon as the backward pass has produced them. Stage 2 reduce-scatters each
bucket instead of allreducing it, so every GPU keeps the summed gradient
of the parameters it owns and frees the rest. The full gradient exists
only one bucket at a time. The collectives are those of stage 1, so
stage 2, too, saves memory at no cost in bytes.

:begin_tab:`jax`
XLA keeps to this order only with its latency-hiding scheduler, which the
first cell of this notebook turns on. Without it, XLA starts each
reduce-scatter as soon as its gradient exists but finishes them all at the
end of the backward pass, so the full gradients of all blocks are alive
together and the step needs more memory than JAX allows itself by default.
:end_tab:

In FSDP each GPU keeps only its shard of every block's weights. An
all-gather assembles a block's weights just before the block computes,
and they are released right after (:numref:`fig_fsdp_lifecycle`). The
backward pass gathers each block's weights once more, releases them after
the block's backward pass, and reduce-scatters the block's gradients. Let
$S = 4N$ be the bytes of the fp32 weights, which is also the size of the
gradients. A step then performs two all-gathers and one reduce-scatter of
$S$ bytes each, three phases where an allreduce has two, so FSDP
communicates 1.5 times as much as data parallelism. No full copy of a
block lives longer than the block's computation, so each GPU holds $16/k$
bytes per parameter plus the activations and one or two gathered blocks.

![The life of one block's weights under FSDP on one GPU. Between uses the
block is sharded, and each GPU holds $1/k$ of it; the other blocks stay
sharded throughout. An all-gather assembles the whole block for its forward
pass, after which it is released. A second all-gather assembles it for the
backward pass, after which it is released again, and a reduce-scatter
leaves each GPU with its share of the
gradients.](../img/mdl-perf-fsdp-lifecycle.svg)
:label:`fig_fsdp_lifecycle`

:numref:`tab_collectives` lists the collectives of this chapter, and one
more that the mixtures of experts of :numref:`sec_moe` need.

:The collective operations behind data-parallel and sharded training.
:label:`tab_collectives`

| collective | what every GPU ends with | where it appears |
|---|---|---|
| allreduce | the full sum | gradients in data parallelism |
| reduce-scatter | one part of the sum | gradients in ZeRO and FSDP |
| all-gather | every part, concatenated | weights in ZeRO and FSDP |
| all-to-all | a different part from each GPU | tokens routed to the experts of a mixture of experts (:numref:`sec_moe`) |

## Memory and Communication

Before we run FSDP, let us predict what a step of the 36-block GPT should
cost on two GPUs, the fewest on which its sharded state fits. Each GPU
processes two sequences of 1,024 tokens per step, and every block is
checkpointed as in :numref:`subsec_mp-checkpointing`, so that it keeps
only its input for the backward pass.

Memory is the easy part. The training state takes $16N/2 = 14.3$ GiB per
GPU. For the 18-block GPT, :numref:`subsec_mp-checkpointing` bounded the
activations under checkpointing by about 1 MiB per token: the input that
each block keeps, 8 KiB per token in fp32, plus one recomputed block, the
output layer, and the loss. Eighteen more stored inputs add only 144 KiB,
so the bound stays at about 1 MiB per token, or 2 GiB for 2,048 tokens.
One or two gathered blocks of 0.2 GB each bring each GPU to at most about
17 GiB, well under the 24 GiB of a card.

Communication sets the time of the step. In an all-gather or a
reduce-scatter of $S$ bytes, each of the $k$ GPUs sends $(k-1)S/k$ bytes:
it is one of the two phases of the ring allreduce of
:numref:`subsec_mg-ring`. FSDP runs three such phases per step. Let
$\beta_\textrm{link}$ be the rate at which each GPU sends during a large
collective (:numref:`subsec_mg-accounting`). The communication of a step
then takes

$$
t_\textrm{comm} = \frac{3(k-1)}{k}\,\frac{S}{\beta_\textrm{link}}.
$$
:eqlabel:`eq_fsdp_comm`

The size of the model sets this time. The number of GPUs, at a fixed
$\beta_\textrm{link}$, changes it by at most a factor of two: each GPU
sends $1.5S$ per step at $k = 2$, and close to $3S$ when $k$ is large.
Adding GPUs therefore divides the memory of the state by $k$, while the
bytes that each GPU sends per step grow instead of falling. If you remember
one property of FSDP from this section, let it be this one. Only more
tokens per GPU, or a faster link, lower the share of the step that goes to
communication. For our GPT, $1.5S$ is 11.5 GB. The rate depends on the
collective and on which GPUs take part, so we measure it for the
collective that FSDP uses most, an all-gather, on the two GPUs of the run.

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

Each GPU sends at about 2.2 GB/s, close to what the allreduces of
:numref:`sec_multi_gpu_concise` reached between the same two GPUs through
host memory. The 11.5 GB of a step should therefore take about 5 s.

The arithmetic takes far less time. A forward and a backward pass take
about six floating-point operations per parameter and token
(:numref:`sec_scaling-laws`), and checkpointing repeats the forward pass,
two of the six, for $8N$ per token. For 2,048 tokens per GPU at the tf32
rate of the RTX 4090, about 83 TFLOP/s, that is
$8 \cdot 1.92 \times 10^{9} \cdot 2048 / (8.3 \times 10^{13}) \approx 0.4$ s.
The cell evaluates the three predictions.

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

At best, overlap hides the 0.4 s of arithmetic under the 5 s of
communication. No schedule can hide 5 s of communication under 0.4 s of
arithmetic. The prediction for a step is therefore about 5 s, of which the
arithmetic would fill about a fourteenth.

## Sharded Training on Two GPUs

Now we run the training step. Both tabs build the model without ever
placing all of it on one GPU, divide the weights, gradients, and moment
estimates of every block between the two GPUs, and take one warm-up step
and three timed ones. One more step, profiled, shows how much of its time
GPU 0 spends in communication kernels. The tokens are random, since only
the shapes of a step decide its memory and its time, and five steps are
far too few to see a loss fall. What we test is whether two GPUs can hold
the step, and how long it takes.

:begin_tab:`pytorch`
In PyTorch, `fully_shard` from `torch.distributed.fsdp` applies FSDP to a
module. `TRAIN_FSDP` calls it once for every block, so that each block is
gathered and released on its own, and once for the whole model, which
covers the embeddings and the final norm. The model is built on the meta
device, which records shapes but stores nothing; `to_empty` then allocates
only this GPU's shards, and we initialize them as `d2l.GPT` does. With
`fused=True` Adam updates all tensors in a few fused kernels, without the
temporaries of the default update, which are as large as the gradients.
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
Like the scripts of :numref:`sec_multi_gpu_concise`, `TRAIN_FSDP` is a
complete program, and `d2l.torchrun` starts one copy of it per GPU.
:end_tab:

```{.python .input #sharding-memory-and-communication-4}
%%tab pytorch
ranks = d2l.torchrun(TRAIN_FSDP, k, json.dumps(config), b, T)
peak_bytes = max(r['peak'] for r in ranks)
times, comm = ranks[0]['times'], ranks[0]['comm']  # GPU 0
```

:begin_tab:`jax`
In JAX, sharding the training state is a change of placement, as data
parallelism was. `fsdp_sharding` splits each array with `P('data')` along
its first axis whose length is a multiple of $k$. For most weight matrices
that is the leading axis. Two kinds of array are split along their width
instead, because two GPUs cannot share their rows evenly: the output matrix
of every feed-forward network, with 5,461 rows, and the embedding table,
with 50,257. `create` builds the model and the optimizer inside `nnx.jit`
under this placement, so that each GPU computes only its own shards. Split
weights alone would let the partitioner choose another scheme, such as
products with split weights followed by an allreduce of the activations.
The constraint on the output of every block in `loss_fn` keeps the
activations split along the batch, and the compiled step below shows that
FSDP's collectives then follow.
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

:begin_tab:`jax`
Each GPU keeps 10.7 GiB of weights and moment estimates, the $12N/k$ bytes
of its shards, all of it donated so that the update can write into it.
The other $4N/k$ bytes of the table's last row, the gradient shards, live
among the temporaries, about 5 GiB, together with the activations and the
gathered blocks. The compiled program also tells which collectives the
compiler wrote.
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

```{.python .input #sharding-memory-and-communication-9}
%%tab jax
for kind, mb in sorted(collective_megabytes(compiled).items(),
                       key=lambda kv: -kv[1]):
    print(f'{kind}: {mb / 1e3:.2f} GB')
```

:begin_tab:`jax`
The compiler wrote FSDP's collectives and little else. The sizes are those
of what each collective returns: the all-gathers return the weights of the
36 blocks twice and the embedding table once, and the reduce-scatter
returns this GPU's half of the gradient, $S/2$. At $k = 2$ a GPU sends half
of each gathered result, and in the reduce-scatter as many bytes as it
keeps, so it sends $7.5 + 3.8 = 11.3$ GB per step, close to the $1.5S$ of
the prediction. The small all-to-all comes from the embedding table, which
is split along its width.
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

```{.python .input #sharding-memory-and-communication-11}
%%tab pytorch, jax
print(f'per GPU: peak memory {peak_bytes / 2**30:.1f} GiB; '
      f'{np.median(times):.2f} s per step ({min(times):.2f} to '
      f'{max(times):.2f})\nof each step, GPU 0 spends {comm:.2f} s in '
      f'communication kernels')
```

Two GPUs hold the step. Each peaks at about 16 GiB, under the bound of
17 GiB. A step takes about 5.5 s, a little longer than the predicted
communication, and the three timed steps agree within a few percent. GPU 0
spends nearly all of the step in communication kernels, so the arithmetic
adds little beside them, and the step lasts about as long as its
communication. Three byte counts and one measured rate predicted it to
within about a tenth. On our machine, sharding lets the model train on two
GPUs, at about 7% of the speed that its arithmetic would allow.

What 7% means becomes clear over a whole run. A compute-optimal run of
this model, at 20 tokens per parameter (:numref:`sec_scaling-laws`), reads
about 38 billion tokens. At 4,096 tokens per step and about 5.5 s per step
our two cards would need about 600 days. Their arithmetic alone, at the
peak rate, would finish in about six weeks. This is why nobody trains a
model of this size over host memory.

The same three phases over NVLink 5, at 900 GB/s in each direction
(:numref:`tab_gpu_specs`), take about 13 ms, a thirtieth of the arithmetic
of our step. FSDP is the standard way to train models whose state does
not fit on one GPU, and on such links it costs little.

The arithmetic also says when to shard. A model whose training state fits
on one GPU next to its activations trains fastest with data parallelism.
On a 24 GiB card with Adam in fp32 that limit lies at about a billion
parameters, fewer for long sequences. The 18-block GPT of
:numref:`sec_memory_precision` sits at this limit. Stage 3 on two GPUs
would cut its 15.1 GiB per GPU to 7.5 GiB and free 7.5 GiB for
activations. Larger micro-batches would then save at most the accumulator
traffic of :numref:`subsec_mp-accumulation`, a fraction $2I^*/n$ of the
arithmetic, which at micro-batches of 2,048 tokens in tf32 is 8%, about a
hundredth of a second per micro-batch. FSDP, on the other hand, gathers
the weights twice for every micro-batch, about 2 s on our link. Data
parallelism with gradient accumulation exchanges the gradient once per
optimizer step, however many micro-batches the step takes. On our machine
we therefore shard the 36-block GPT, and not the 18-block one.

FSDP still gathers each block whole before the block computes, so a block
must fit on one GPU. *Tensor parallelism* removes that limit by splitting
every weight matrix across the GPUs (exercise :exref:`ex_mgp-tensor`). It
belongs, with the forms of parallelism that span several machines, to
:numref:`chap_lm_parallelism`. The case study that ends this chapter
returns to a model that fits on one GPU, where the causes of a slow step
that we took apart one at a time act together.

## Summary

Data parallelism keeps all 16 bytes per parameter of the training state on
every GPU, so it cannot train a model whose state exceeds one card. ZeRO
divides the state among the $k$ GPUs: stage 1 the optimizer state, and
stage 2 also the gradients, reduce-scattered bucket by bucket, both at no
cost in bytes. Stage 3, FSDP, also divides the weights and gathers each
block for its forward and backward pass, at 1.5 times the communication of
data parallelism. More GPUs divide its memory per GPU, but not the bytes
that each GPU sends per step. Three phases of $S$ bytes and one measured
link rate predict the step. On two of our GPUs the 36-block GPT, which no
card of ours holds, runs at about 5.5 s per step, nearly all of it
communication through host memory, or about 7% of its arithmetic speed.
Sharding is worth its communication where the state does not fit, and over
a fast link the same bytes take milliseconds.

## Exercises

1. **Cost of sharding.** ● Count communication per GPU and step in units
   of $S$, the bytes of one full set of fp32 weights or gradients.
    1. A variant of ZeRO stage 3 keeps every gathered block from its
       forward pass until its backward pass. Which collective does this
       remove, and how much of the memory saving of stage 3 does it give
       up, in bytes per parameter of :numref:`tab_mgp-zero`?
    1. Mixed-precision FSDP gathers the weights in bf16 and reduce-scatters
       the gradients in bf16, while every GPU keeps its shard of the fp32
       weights and of Adam's state. Compare its bytes per step with those
       of data parallelism in fp32, and its memory per parameter with the
       last row of :numref:`tab_mgp-zero`.
    1. For the 36-block GPT on $k = 2$ GPUs, with the $\beta_\textrm{link}$
       measured in this section (the stored output gives 2.2 GB/s if you
       have no two GPUs), find the number of tokens per GPU and step at
       which the computation, $8N$ operations per token with checkpointing
       at 83 TFLOP/s, takes as long as the communication
       of :eqref:`eq_fsdp_comm`. How many sequences of 1,024 tokens is that,
       and do their activations fit next to the training state
       (:numref:`subsec_mp-checkpointing`)? How does gradient accumulation
       change the answer when FSDP gathers the weights anew for every
       micro-batch?


:begin_tab:`pytorch`
2. [extended] **Sharded optimizer state.** ● This section ran stage 3 with
    `fully_shard`; here you implement ZeRO stage 1
    :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` by hand and measure what it
    saves. Start from `TRAIN_DDP` of :numref:`sec_multi_gpu_concise`, with
    `torch.optim.Adam` in place of SGD.
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
2. [extended] **Sharded optimizer state.** ● This section ran stage 3 by
    placement; here you implement ZeRO stage 1
    :cite:`Rajbhandari.Rasley.Ruwase.ea.2020` declaratively and measure
    what it saves. Start from the `train_step` of
    :numref:`sec_multi_gpu_concise` and use `optax.adam` in
    `nnx.Optimizer` instead of SGD. Keep the model replicated, and place
    every leaf of `nnx.state(opt)` with a `NamedSharding` that splits the
    first of its axes whose length is divisible by $k$, replicating the
    leaves that have no such axis, as `fsdp_sharding` does.
    1. Run `train_step` once and show the layout of a weight afterwards.
       Stage 1 needs replicated parameters at the start of every step; if
       the compiler did not keep them replicated, restore the layout inside
       the step with `jax.sharding.reshard` and a replicated
       `NamedSharding`. Check that after a few steps the parameters match
       those of replicated Adam from the same initialization.
    1. Read the collectives of the compiled step with
       `collective_megabytes`. Which collective is new compared with the
       replicated step, and is the gradient sum an allreduce or a
       reduce-scatter? Compare the bytes per GPU of the compiled step with
       the reduce-scatter and all-gather that stage 1 needs
       (:numref:`subsec_mgp-fsdp`).
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

4. **Other machines, other counts.** Use :eqref:`eq_fsdp_comm`, the bound
   of about 1 MiB of activations per token, and $8N$ operations per token
   at 83 TFLOP/s for the 36-block GPT with two sequences of 1,024 tokens
   per GPU.
    1. Predict the training state and the peak memory per GPU, and the
       seconds per step, on four of our GPUs, assuming that
       $\beta_\textrm{link}$ keeps its two-GPU value. Which assumption of
       this prediction is the least safe on a machine whose GPUs all
       exchange data through the same host memory
       (:numref:`subsec_hw-interconnects`)?
    1. Predict the seconds per step on four GPUs joined by NVLink 4, at
       450 GB/s in each direction, keeping the arithmetic rate of an
       RTX 4090 so that only the link changes. Which resource now limits
       the step?
    1. Now hold the global minibatch at 64 sequences of 1,024 tokens,
       divided among the $k$ GPUs. For each of the two links, find the $k$
       at which the communication of a step equals its arithmetic,
       ignoring whether the activations fit and treating the tokens per
       GPU as continuous. What
       does the answer say about adding GPUs to a sharded run on our
       machine?

5. **Prefetching the next block.** In the forward pass under FSDP, let $g$
   be the time of the all-gather of one block and $c$ the time that the
   block computes. Without overlap, the forward pass of $L$ blocks takes
   $L(g + c)$. FSDP can instead issue the all-gather of the next block
   while the current one computes.
    1. Show that the forward pass then takes about
       $g + (L-1)\max(g, c) + c$. Which term of the prediction shrinks,
       and to what? Why can no schedule make the forward pass shorter than
       $L\max(g, c)$?
    1. Compute $g$ and $c$ for one block of the 36-block GPT on our two
       GPUs: $12d^2 \approx 50$ million parameters per block, 2,048 tokens
       per GPU, two operations per parameter and token in the forward pass
       at 83 TFLOP/s, and the $\beta_\textrm{link}$ of this section. How
       much can prefetching save per step, and why does the measured step
       leave so little room for it?
    1. Repeat the computation for NVLink 5 at 900 GB/s in each direction.
       Which of $g$ and $c$ is now larger, and what fraction of the forward
       pass does prefetching remove?

[Discussions](https://d2l.discourse.group/)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.7]{.kicker}

Sharding the Training State<br>**ZeRO stages 1–3 · FSDP · a model no card holds, run on two GPUs**
:::
:::

::: {.slide title="29 GiB of State, 24 GiB per Card"}
[The problem]{.kicker}

@!sharding-sharding-the-training-state-2

- The 1B GPT made twice as deep: 36 blocks, width 2,048.
- Adam in fp32: weight, gradient, two moments, 16 bytes per parameter.
- Data parallelism puts all 16 bytes on every GPU.

::: {.d2l-note .warn}
Can $k$ GPUs train a model that none of them holds?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- size the memory per GPU of ZeRO stages 1, 2, and 3;
- explain how stage 2 avoids ever holding the full gradient;
- count the bytes that FSDP sends per step, and how they change with $k$;
- predict a sharded step's memory and seconds, and check them;
- decide when sharding is worth its communication.

**Running example:** the 36-block GPT, $N = 1.92$ billion parameters, on
two RTX 4090s that exchange data through host memory.
:::

::: {.slide title="Data Parallelism Copied the State; Now We Divide It"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Performance Model | roofline, three regimes, timing, profiling |
| Hardware | where peak arithmetic and bandwidth come from |
| Computation Graphs and Compilation | fusion and kernel replay |
| Memory and Precision | 16 bytes per parameter, activations, checkpointing |
| Multi-GPU from First Principles | the ring allreduce, its cost model |
| Data Parallelism in Practice | one process per GPU, buckets, scaling |
| **Sharding the Training State** | **ZeRO and FSDP, measured** |
| Case Study: Optimizing a Transformer | the whole method on one GPT |
:::

::: {.col .narrow}
- **Uses:** reduce-scatter and all-gather, $\beta_\textrm{link}$,
  checkpointing.
- **Skips:** tensor, pipeline, and expert parallelism; several machines.
:::
:::
:::

::: {.slide title="A GPU Needs Each Part of Its State Only Briefly"}
[The idea]{.kicker}

| state | bytes per parameter | needed |
|---|---|---|
| weights | 4 | in its block's forward and backward pass |
| gradients | 4 | once, by the optimizer |
| Adam's two moments | 8 | only inside the update |

All $k$ GPUs of data parallelism run the same update: $k$ identical
copies.

::: {.d2l-note .rule}
Keep $1/k$ of each; gather a block when it computes, scatter each gradient
as soon as it exists.
:::
:::

::: {.slide title="Each ZeRO Stage Divides One More Kind of State by k"}
[ZeRO stages]{.kicker}

Per GPU at the end of the backward pass, Adam in fp32, $k$ GPUs:

| each GPU holds | weights | grads | Adam | bytes/param | GPT, $k = 2$ |
|---|---|---|---|---|---|
| data parallel | all | all | all | 16 | 28.6 GiB |
| ZeRO-1 | all | all | $1/k$ | $8 + 8/k$ | 21.4 GiB |
| ZeRO-2 | all | $1/k$ | $1/k$ | $4 + 12/k$ | 17.9 GiB |
| ZeRO-3 (FSDP) | $1/k$ | $1/k$ | $1/k$ | $16/k$ | 14.3 GiB |

::: {.d2l-note .rule}
On 24 GiB cards, ZeRO-1 leaves under 3 GiB free; ZeRO-3 almost 10.
:::
:::

::: {.slide title="Stage 1 Moves the Same Bytes as an Allreduce"}
[ZeRO stage 1]{.kicker}

allreduce = reduce-scatter + all-gather. Split them:

1. reduce-scatter: each GPU gets the summed gradient of the parameters
   it owns;
2. each GPU updates its $1/k$ with its $1/k$ of Adam's state;
3. all-gather: the updated weights reach every GPU.

::: {.d2l-note .rule}
Same bytes, less memory. The all-gather sits between the update and the
next forward pass, and adds to the step unless it is overlapped.
:::
:::

::: {.slide title="Stage 2 Scatters Each Bucket as Soon as It Exists"}
[ZeRO stage 2]{.kicker}

- Every GPU's backward pass produces every gradient.
- Gradients leave in buckets, as soon as the backward pass makes them.
- Reduce-scatter each bucket: keep the summed $1/k$ that this GPU owns,
  free the rest.

::: {.d2l-note .rule}
The full gradient never exists at once: one bucket at a time. The
collectives, and the bytes, are those of stage 1.
:::
:::

::: {.slide title="FSDP Gathers Each Block Twice: 1.5× the Communication"}
[ZeRO stage 3]{.kicker}

![](../img/mdl-perf-fsdp-lifecycle.svg){width=70%}

Per step: two all-gathers and one reduce-scatter of $S = 4N$ bytes. With
$\beta_\textrm{link}$ the bytes each GPU sends per second in a large
collective:

$$t_\textrm{comm} = \frac{3(k-1)}{k}\,\frac{S}{\beta_\textrm{link}}$$

::: {.d2l-note .rule}
From $1.5S$ per GPU at $k = 2$ towards $3S$: more GPUs divide the memory,
not the bytes each one sends.
:::
:::

::: {.slide title="Check: Predict Memory and Time on Two GPUs"}
[Check]{.kicker}

$N = 1.92$ B, $S = 4N = 7.7$ GB, $k = 2$, $\beta_\textrm{link} =
2.2$ GB/s sent per GPU, 2,048 tokens per GPU, checkpointing at about
1 MiB per token, $8N$ FLOPs per token at 83 TFLOP/s. Peak memory,
communication, arithmetic?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $16N/2 = 14.3$ GiB, plus 2 GiB of activations and a block or
two: at most 17 GiB. Each phase sends $S/2$, so
$1.5S/\beta_\textrm{link} = 11.5/2.2 \approx 5.2$ s. Arithmetic:
$8 \cdot 1.92 \cdot 10^9 \cdot 2048 / (8.3 \cdot 10^{13}) \approx 0.4$ s.
:::
:::
:::

::: {.slide title="The Step Runs at About 7% of Its Arithmetic Speed"}
[Measured]{.kicker}

@!sharding-memory-and-communication-11

- Peak about 16 GiB per GPU, under the predicted 17.
- About 5.5 s per step, a little over the predicted 5.2 s.
- Nearly all of it in communication kernels; the 0.4 s of arithmetic adds
  little.

::: {.d2l-note .rule}
A compute-optimal run, 20 tokens per parameter: about 600 days on these two
cards, six weeks for their arithmetic alone.
:::
:::

::: {.slide title="fully_shard per Block Makes Each Block a Unit" only="pytorch"}
[FSDP in PyTorch]{.kicker}

```{.python #sharding-exercises-1}
with torch.device('meta'):      # shapes only, no memory
    model = d2l.GPT(**config)
for blk in model.blks:          # each block gathered on its own
    fully_shard(blk)
fully_shard(model)              # embeddings and final norm
model.to_empty(device=device)   # this GPU's shards only
...                             # initialize them as d2l.GPT does
opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
```

- One process per GPU, started by `torchrun`.
- `fused=True`: no Adam temporaries as large as the gradients.
:::

::: {.slide title="Keep Activations Split, and XLA Writes FSDP's Collectives" only="jax"}
[FSDP in JAX]{.kicker}

::: {.cols .vc}
::: {.col}
```{.python #sharding-exercises-2}
def fsdp_sharding(x):  # first axis that k divides
    for i, n in enumerate(x.shape):
        if n % k == 0:
            return NamedSharding(
                mesh, P(*[None] * i, 'data'))
    return NamedSharding(mesh, P())
```

Every block's output constrained to stay split by batch.
:::

::: {.col}
@!sharding-memory-and-communication-9

- Sizes are results: weights gathered twice, gradient shard $S/2$.
- Sent per GPU: $7.5 + 3.8 = 11.3$ GB, against $1.5S = 11.5$ GB.
:::
:::
:::

::: {.slide title="Shard Only When the State Does Not Fit"}
[When to shard]{.kicker}

- Fits next to its activations: data parallelism is fastest.
- 24 GiB card, Adam in fp32: the limit is about 1B parameters.
- 18-block GPT: ZeRO-3 on two GPUs cuts 15.1 GiB to 7.5 GiB per GPU.
- Larger micro-batches would save at most 8% of the arithmetic in
  accumulator traffic, about 0.01 s per micro-batch.
- FSDP gathers the weights twice per micro-batch, about 2 s on our link;
  data parallelism with accumulation exchanges once per step.

::: {.d2l-note}
On our machine: shard the 36-block GPT, not the 18-block one.
:::
:::

::: {.slide title="Check: The Same Step over NVLink"}
[Check]{.kicker}

The same 11.5 GB per GPU and step, $k = 2$, over NVLink 5 at 900 GB/s
per direction. How long is the communication, and what limits the step?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $11.5 / 900 \approx 13$ ms, a thirtieth of the 0.4 s of
arithmetic.

| link | $\beta_\textrm{link}$ | communication per step |
|---|---|---|
| ours: PCIe through host memory | 2.2 GB/s | 5.2 s |
| NVLink 5 | 900 GB/s | 13 ms |

The arithmetic limits the step; on such links FSDP costs little.
:::
:::
:::

::: {.slide title="Divide the State by k; Every Step Gathers It Back"}
[Recap]{.kicker}

- Data parallelism: 16 bytes per parameter on every GPU; 28.6 GiB does
  not fit.
- ZeRO-1 and ZeRO-2: optimizer state, then gradients, at no cost in
  bytes.
- FSDP: also the weights, 14.3 GiB per GPU at $k = 2$, at 1.5× the
  communication; more GPUs do not cut the bytes each one sends.
- Measured: about 16 GiB and 5.5 s per step, 7% of the arithmetic speed.

**Next:** one GPU again, and a GPT step whose causes of slowness act all at once.
:::
