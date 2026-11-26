# Memory and Precision
:label:`sec_memory_precision`

The previous sections asked how fast a training step runs; a step that does not
fit in GPU memory does not run at all. Our 1B-parameter GPT trained with Adam
in fp32 stores four fp32 values per parameter—the weight, its gradient, and
Adam's two moment estimates—consuming $16N \approx 15.1$ GiB of a 24 GiB
RTX 4090 before reading a single token. The remaining ~9 GiB must hold the
*activations* saved by the forward pass for backpropagation, which take
several megabytes per token. As a result, only two or three sequences of 1,024
tokens fit on the card, whereas large language models train on global batches
of half a million to several million tokens :cite:`brown2020language`.

The memory of a training step consists of a fixed parameter term proportional
to $N$ and an activation term proportional to the number of tokens $n$. In
this section we count both for our GPT, verify the count against measured peak
memory, and study three techniques that act on the activation term or the
micro-batch size: *mixed precision* (computing in 16 bits), *activation
checkpointing* (recomputing activations during the backward pass instead of
storing them), and *gradient accumulation* (splitting a large global batch
into sequential micro-batches). None of the three shrinks the $16N$ bytes of
parameter state; dividing that state across GPUs is the subject of
:numref:`sec_sharding`.

```{.python .input #memory-precision-memory-and-precision}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #memory-precision-memory-and-precision}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #memory-precision-memory-and-precision-1}
%%tab jax
import re
```

```{.python .input #memory-precision-memory-and-precision-1}
%%tab pytorch
import os
import re
from torch.utils.checkpoint import checkpoint
# Detach PyTorch's profiler from the GPU when a profile ends; without this,
# every kernel launch after the first profile stays about 30% slower
os.environ['TEARDOWN_CUPTI'] = '1'
```

## Memory of a Training Step
:label:`subsec_mp-anatomy`

### Memory per Parameter

Parameter memory depends on the optimizer: plain fp32 SGD stores a weight and
gradient (8 bytes/parameter), SGD with momentum adds a velocity (12 bytes),
and Adam stores two moment estimates (**16 bytes/parameter**,
:numref:`subsec_memory_during_training`). Mixed-precision training keeps all
four in fp32 and holds a 16-bit weight copy during the step (**18
bytes/parameter**).

Our reference model is `d2l.GPT` (:numref:`sec_gpt`) with width $d = 2048$, 16
heads, $L = 18$ pre-norm SwiGLU blocks, vocabulary $V = 50{,}257$, and learned
positional embeddings. Each block holds ~$12d^2 \approx 50$M parameters
(~0.9B across 18 blocks, plus ~0.1B in the tied token embedding). Inspecting
the model on a meta/shape-only device confirms $N$ without allocating 4 GB of
weights (:numref:`subsec_count_without_alloc`):

```{.python .input #memory-precision-memory-per-parameter}
%%tab pytorch
config = dict(vocab_size=50257, num_hiddens=2048, num_heads=16,
              num_blks=18, pos='learned')
with torch.device('meta'):  # Shapes and dtypes only, no data
    model = d2l.GPT(**config)
N = d2l.num_params(model)
print(f'{N / 1e9:.2f} billion parameters, 16N = {16 * N / 2**30:.1f} GiB')
```

```{.python .input #memory-precision-memory-per-parameter}
%%tab jax
config = dict(vocab_size=50257, num_hiddens=2048, num_heads=16,
              num_blks=18, pos='learned')
key = d2l.get_key()
model = nnx.eval_shape(lambda: d2l.GPT(**config, rngs=nnx.Rngs(key)))
N = d2l.num_params(model)
print(f'{N / 1e9:.2f} billion parameters, 16N = {16 * N / 2**30:.1f} GiB')
```

### Memory per Token

During the forward pass, layers save intermediate activations needed by the
backward pass; during the backward pass, these activations are freed layer by
layer as parameter gradients are produced (:numref:`fig_memory_anatomy`).
Memory therefore peaks at the start of the backward pass.

![The memory of a training step with Adam, in units of $4N$ bytes. The
weights ($4N$) and Adam's moment estimates ($8N$) stay for the whole step.
Activations accumulate during the forward pass and are released during the
backward pass, while the gradients ($4N$) appear. Memory peaks at the start
of the backward pass.](../img/mdl-perf-memory-anatomy.svg)
:label:`fig_memory_anatomy`

Three rules determine what a Transformer block saves for backpropagation:

1. A linear layer $\mathbf{y} = \mathbf{W}\mathbf{x}$ saves its input
   $\mathbf{x}$ (since $\nabla_{\mathbf{W}} = (\nabla_{\mathbf{y}})\mathbf{x}^\top$).
2. An elementwise nonlinearity $f$ saves its input or output.
3. An elementwise product $\mathbf{p} \odot \mathbf{q}$ saves both factors.

Counting per token in units of width $d = 2048$:

* **Attention sublayer:** Pre-attention RMSNorm saves its input ($d$); the QKV
  projection saves the normalized output ($d$); fused causal attention saves
  $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ ($3d$) and its output ($d$, plus only a
  scalar log-normalizer per head and token instead of the $T \times T$ score
  matrix); the output projection saves the attention output ($0$ if aliased, or
  $d$ if copied when merging heads). Total: $6d\text{--}7d$.
* **SwiGLU FFN sublayer:** Pre-FFN RMSNorm saves its input ($d$); the gate and
  up-projections share the normalized output ($d$); $\textrm{SiLU}(\mathbf{g})
  \odot \mathbf{u}$ and the down-projection save the gate $\mathbf{g}$, the
  activated gate $\textrm{SiLU}(\mathbf{g})$, the up-projection $\mathbf{u}$,
  and their product—four vectors of width $\tfrac{8}{3}d \approx 10.7d$.
  Total: ~$12.7d$.

In total, one fused block saves $18.7d\text{--}19.7d \approx \mathbf{20d}$
values per token (**160 KiB** in fp32 at $d = 2048$; compare the $16d$ count
for un-gated GPT-3 blocks in :citet:`Korthikanti.Casper.Lym.ea.2022`). Because
fused attention recomputes its $T \times T$ weights on chip during the
backward pass, nothing in this count grows with sequence length $T$. Finally,
the cross-entropy loss saves $V = 50{,}257$ fp32 log-probabilities per token
(**196 KiB**)—more than an entire Transformer block.

To verify this tally empirically, we first define `saved_bytes` to total the
tensor bytes retained for the backward pass of `gpt_loss`:

```{.python .input #memory-precision-memory-per-token-1}
%%tab pytorch
def saved_bytes(f, *args):
    """Bytes of the tensors that autograd keeps for the backward pass of f."""
    storages = {}
    def pack(t):  # Called on every tensor saved for the backward pass
        storages[t.untyped_storage().data_ptr()] = (
            t.untyped_storage().nbytes())
        # A detached alias shares the storage. Returning t itself would form
        # a reference cycle between a saved output and the graph that made
        # it, and the graph, with all its tensors, would never be freed
        return t.detach()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        f(*args)
    return sum(storages.values())

def gpt_loss(model, X, bf16=False):
    """Cross-entropy of a GPT's predictions, with the inputs as targets."""
    with torch.autocast(X.device.type, torch.bfloat16, enabled=bf16):
        return F.cross_entropy(model(X).flatten(0, 1), X.flatten())
```

```{.python .input #memory-precision-memory-per-token-1}
%%tab jax
def saved_bytes(f, *args):
    """Bytes of the arrays that the backward pass of f keeps."""
    _, f_vjp = jax.vjp(f, *args)  # The leaves of f_vjp are the saved arrays
    return sum(x.nbytes for x in jax.tree_util.tree_leaves(f_vjp))

def gpt_loss(model, X):
    """Cross-entropy of a GPT's predictions, with the inputs as targets."""
    logits = model(X).astype(jnp.float32)
    return optax.softmax_cross_entropy_with_integer_labels(logits, X).mean()
```

By evaluating `saved_bytes` on 1-block and 2-block GPT models across batch
sizes $b \in \{1, 2\}$, `bytes_per_token` isolates the marginal activation
memory per token of a single Transformer block (`block`) from that of the
embeddings, output head, and loss (`head`):

```{.python .input #memory-precision-memory-per-token-2}
%%tab pytorch
def bytes_per_token(config, bf16=False, num_steps=1024):
    """Bytes stored per token by one block, and by the rest of a GPT."""
    gpu, per_token = d2l.try_gpu(), []
    for L in (1, 2):  # GPTs with one and with two blocks
        model = d2l.GPT(**{**config, 'num_blks': L}).to(gpu)
        X1, X2 = (torch.randint(0, config['vocab_size'], (b, num_steps),
                                device=gpu) for b in (1, 2))
        per_token.append((saved_bytes(gpt_loss, model, X2, bf16)
                          - saved_bytes(gpt_loss, model, X1, bf16))
                         / num_steps)
    return per_token[1] - per_token[0], 2 * per_token[0] - per_token[1]

block, head = bytes_per_token(config)
print(f'one block: {block / 2**10:.0f} KiB per token '
      f'({block / (4 * config["num_hiddens"]):.1f} fp32 values of width d)')
print(f'embedding, output layer, and loss: {head / 2**10:.0f} KiB per token')
```

```{.python .input #memory-precision-memory-per-token-2}
%%tab jax
def to_bf16(model):  #@save
    """Make the linear layers of `model` compute in bf16; parameters stay
    fp32."""
    for _, layer in nnx.iter_graph(model):
        if isinstance(layer, nnx.Linear):
            layer.dtype = jnp.bfloat16
    return model

def bytes_per_token(config, bf16=False, num_steps=1024):
    """Bytes stored per token by one block, and by the rest of a GPT."""
    per_token = []
    for L in (1, 2):  # GPTs with one and with two blocks
        model = d2l.GPT(**{**config, 'num_blks': L},
                        rngs=nnx.Rngs(d2l.get_key()))
        model = to_bf16(model) if bf16 else model
        # The structure, the weights, and the other state of the model;
        # nnx.merge puts them back together, so the loss is a function of
        # the weights
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)
        loss = lambda params, X: gpt_loss(nnx.merge(graphdef, params, rest), X)
        X1, X2 = (jax.random.randint(d2l.get_key(), (b, num_steps), 0,
                                     config['vocab_size']) for b in (1, 2))
        per_token.append((saved_bytes(loss, params, X2)
                          - saved_bytes(loss, params, X1)) / num_steps)
    return per_token[1] - per_token[0], 2 * per_token[0] - per_token[1]

block, head = bytes_per_token(config)
print(f'one block: {block / 2**10:.0f} KiB per token '
      f'({block / (4 * config["num_hiddens"]):.1f} fp32 values of width d)')
print(f'embedding, output layer, and loss: {head / 2**10:.0f} KiB per token')
```

:begin_tab:`pytorch`
Because `scaled_dot_product_attention`, RMSNorm, and SiLU run as single fused
kernels, PyTorch stores exactly the theoretical minimum: $18.7d$ values (149
KiB) per block and 212 KiB for the embedding, output projection, and loss.
:end_tab:

:begin_tab:`jax`
Before XLA compilation, `jax.vjp` records $29.0d$ values (232 KiB) per block
because `nnx.RMSNorm` and `jax.nn.silu` are written as composite elementwise
expressions whose intermediate arrays remain in the raw VJP residuals. Whether
the compiled program keeps all of them is a question for XLA's memory plan,
which we read below.
:end_tab:

### The Memory Budget

Combining parameter state ($16N$ bytes when gradients persist across
micro-batches) with $a$ bytes per token per block and $a_\textrm{out}$ bytes
per token for the embedding, output head, and loss yields the step memory
budget for $n$ tokens:

$$
M(n) \approx 16N + n \, (L a + a_\textrm{out}) \;\leq\; M_\textrm{GPU}.
$$
:eqlabel:`eq_mp-budget`

For our 18-block GPT on a 24 GiB card ($16N = 15.1$ GiB, leaving 8.9 GiB for
activations), the raw count predicts:

:begin_tab:`pytorch`
$La + a_\textrm{out} \approx 2.8$ MiB/token (~3,200 tokens, or 3 sequences of
1,024).
:end_tab:

:begin_tab:`jax`
$La + a_\textrm{out} \approx 4.3$ MiB/token (~2,100 tokens, or 2 sequences of
1,024).
:end_tab:

### Measured Peak Memory
:label:`subsec_mp-measuring`

To see what the backward pass temporarily allocates on top of saved
activations, we measure peak memory above the weights on a 2-block version of
our GPT across 1,024 to 8,192 tokens.

```{.python .input #memory-precision-measuring-memory-1}
%%tab pytorch
gpu, V = d2l.try_gpu(), config['vocab_size']
model = d2l.GPT(**{**config, 'num_blks': 2}).to(gpu)
gpt_loss(model, torch.randint(0, V, (1, 1024), device=gpu)).backward()
peaks = {}  # The step above was a warm-up
for n in (1024, 2048, 4096, 8192):  # Tokens per step, in sequences of 1,024
    X = torch.randint(0, V, (n // 1024, 1024), device=gpu)
    model.zero_grad(set_to_none=True)
    peaks[n] = d2l.peak_memory(lambda: gpt_loss(model, X).backward())
    print(f'{n:5d} tokens: peak {peaks[n] / 2**30:.2f} GiB above the weights')
for n in (2048, 4096):  # Growth per token from n to 2n tokens
    slope = (peaks[2 * n] - peaks[n]) / n
    print(f'{n:5d} to {2 * n:5d} tokens: measured {slope / 2**10:.0f} KiB '
          f'per token; counted {(2 * block + head) / 2**10:.0f} KiB')
```

```{.python .input #memory-precision-measuring-memory-3}
%%tab jax
key = d2l.get_key()
model = nnx.eval_shape(
    lambda: d2l.GPT(**{**config, 'num_blks': 2}, rngs=nnx.Rngs(key)))
graphdef, params, rest = nnx.split(model, nnx.Param, ...)
grad_fn = jax.jit(jax.grad(
    lambda params, rest, X: gpt_loss(nnx.merge(graphdef, params, rest), X)))
peaks = {}
for n in (1024, 2048, 4096, 8192):  # Tokens per step, in sequences of 1,024
    X = jax.ShapeDtypeStruct((n // 1024, 1024), jnp.int32)
    plan = grad_fn.lower(params, rest, X).compile().memory_analysis()
    peaks[n] = plan.output_size_in_bytes + plan.temp_size_in_bytes
    print(f'{n:5d} tokens: peak {peaks[n] / 2**30:.2f} GiB above the weights')
for n in (2048, 4096):  # Growth per token from n to 2n tokens
    slope = (peaks[2 * n] - peaks[n]) / n
    print(f'{n:5d} to {2 * n:5d} tokens: planned {slope / 2**10:.0f} KiB '
          f'per token; counted {(2 * block + head) / 2**10:.0f} KiB')
```

At small sizes fixed overheads still show. Once they are negligible, the peak
grows in proportion to the number of tokens, and the growth per token is the
number to compare with our count.

:begin_tab:`pytorch`
The measured peak grows by about 905 KiB per token, 393 KiB more than the
511 KiB we counted. The excess arises at the start of the backward pass,
while all forward activations are still live: the cross-entropy backward pass
materializes the gradient with respect to the log-probabilities and the
gradient with respect to the logits—two temporaries of $V = 50{,}257$ fp32
values per token ($8V \approx 393$ KiB/token above the saved-tensor count).
:end_tab:

:begin_tab:`jax`
From 4,096 to 8,192 tokens XLA's planned peak grows by about 690 KiB per
token, the same as the 692 KiB of the raw VJP count. The compiled program therefore needs as much memory per
token as `jax.vjp` recorded, with no measurable excess at the loss, and the
count can serve as the budget.
:end_tab:

:begin_tab:`pytorch`
We can visualize the entire allocation timeline of an Adam step on 4,096
tokens using PyTorch's allocator history API:
:end_tab:

```{.python .input #memory-precision-measured-peak-memory-1}
%%tab pytorch
X = torch.randint(0, V, (4, 1024), device=gpu)
opt = torch.optim.Adam(model.parameters())
gpt_loss(model, X).backward()  # A first step creates the moment estimates
opt.step()
opt.zero_grad(set_to_none=True)

# _snapshot, _record_memory_history and the keys of their records are a
# private API of PyTorch 2.11 and may change between versions
events = lambda: torch.cuda.memory._snapshot()['device_traces'][0]
start = torch.cuda.memory_allocated()
torch.cuda.memory._record_memory_history()  # Record from here on
loss = gpt_loss(model, X)
forward_end = len(events())
loss.backward()
backward_end = len(events())
opt.step()
opt.zero_grad(set_to_none=True)
trace = events()
torch.cuda.memory._record_memory_history(enabled=None)
```

:begin_tab:`pytorch`
Accumulating the signed byte changes (`alloc` minus `free_completed`) across
`trace` and segmenting by `forward_end` and `backward_end` plots the live
memory trajectory across the three phases of the step:
:end_tab:

```{.python .input #memory-precision-measuring-memory-2}
%%tab pytorch
sign = {'alloc': 1, 'free_completed': -1}  # Other actions change nothing
sizes = torch.tensor([sign.get(e['action'], 0) * e['size'] for e in trace])
series = ((start + sizes.cumsum(0)) / 2**30).tolist()
bounds = [0, forward_end, backward_end, len(trace)]
phases = list(zip(bounds[:-1], bounds[1:]))
d2l.plot([list(range(a, b)) for a, b in phases],
         [series[a:b] for a, b in phases], 'allocation event',
         'allocated (GiB)', legend=['forward', 'backward', 'optimizer step'])
```

:begin_tab:`pytorch`
The trace starts at $12N$ (weights and two Adam moments), climbs linearly
across the blocks during the forward pass, spikes at the logits and the start
of the backward pass, and steps down as activations are freed and gradients
are allocated.
:end_tab:

## Mixed Precision
:label:`subsec_mp-mixed`

Mixed-precision training (:numref:`sec_numerics`) keeps master weights,
gradients, optimizer moments, normalizations, the residual stream, and the
loss in fp32 while running matrix multiplications and intermediate activations
in 16-bit floating point (bf16, which shares fp32's 8-bit exponent range and
needs no loss scaling, or fp16 with loss scaling, :numref:`subsec_loss_scaling`).

Because the two residual-stream inputs to a block's normalizations stay in
fp32 (4 bytes) while the remaining 18 of the ~$20d$ saved values per block
drop to bf16 (2 bytes), a block's saved activations shrink to roughly
$(2 \cdot 4 + 18 \cdot 2) / (20 \cdot 4) = 0.55$ of their fp32 size:

```{.python .input #memory-precision-mixed-precision-1}
%%tab pytorch
block16, head16 = bytes_per_token(config, bf16=True)
print(f'one block: {block16 / 2**10:.0f} KiB per token in bf16, '
      f'{block16 / block:.2f} of fp32')
print(f'embedding, output layer, and loss: {head16 / 2**10:.0f} KiB per token')
```

```{.python .input #memory-precision-mixed-precision-1}
%%tab jax
block16, head16 = bytes_per_token(config, bf16=True)
print(f'one block: {block16 / 2**10:.0f} KiB per token in bf16, '
      f'{block16 / block:.2f} of fp32')
print(f'embedding, output layer, and loss: {head16 / 2**10:.0f} KiB per token')
```

One block drops to $0.58\text{--}0.60$ of its fp32 activation bytes. However,
holding bf16 copies of the weights during the step increases fixed parameter
memory from $16N = 15.1$ GiB to $18N = 17.0$ GiB, leaving only 7.0 GiB for
activations. The activation savings and extra weight copies largely offset
each other, expanding the maximum token capacity on a 24 GiB card by only
15–27%.

The primary benefit of bf16 mixed precision is speed rather than capacity: on
the RTX 4090, bf16 tensor cores run at 165 TFLOP/s, twice the 82.6 TFLOP/s of
tf32 and fp32. To separate the effect of tensor cores (tf32 vs. scalar fp32)
from the effect of 16-bit operands (bf16 vs. tf32), we benchmark one SGD step
of a compute-bound three-layer MLP of width 8,192 on a batch of 2,048 examples
($1.1 \times 10^{12}$ FLOPs/step: roofline floor 13.3 ms at 82.6 TFLOP/s and
6.7 ms at 165 TFLOP/s, plus ~1.3 ms for the bandwidth-bound SGD update).

```{.python .input #memory-precision-mixed-precision-2}
%%tab pytorch
sizes = [(64 * 64, 8192), (8192, 8192), (8192, 10)]  # (inputs, outputs)
mlp = nn.Sequential(nn.Linear(*sizes[0], bias=False), nn.ReLU(),
                    nn.Linear(*sizes[1], bias=False), nn.ReLU(),
                    nn.Linear(*sizes[2], bias=False)).to(gpu)
for layer in mlp[::2]:  # Weights drawn from N(0, 0.02^2)
    nn.init.normal_(layer.weight, std=0.02)
X_mlp = torch.randn(2048, 64 * 64, device=gpu)
y_mlp = torch.randint(0, 10, (2048,), device=gpu)
opt = torch.optim.SGD(mlp.parameters(), lr=0.01)

def step(bf16):
    with torch.autocast(X_mlp.device.type, torch.bfloat16, enabled=bf16):
        l = F.cross_entropy(mlp(X_mlp), y_mlp)
    l.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
```

```{.python .input #memory-precision-mixed-precision-2}
%%tab jax
def net_fn(params, X, dtype):
    h = X.astype(dtype)
    for W in params[:-1]:
        h = jax.nn.relu(h @ W.astype(dtype))
    return h @ params[-1].astype(dtype)

sizes = [(64 * 64, 8192), (8192, 8192), (8192, 10)]  # (inputs, outputs)
mlp_params = [jax.random.normal(k, s) * 0.02
              for k, s in zip(jax.random.split(d2l.get_key(), 3), sizes)]
X_mlp = jax.random.normal(d2l.get_key(), (2048, 64 * 64))
y_mlp = jax.random.randint(d2l.get_key(), (2048,), 0, 10)

def loss_fn(params, X, y, dtype):
    logits = net_fn(params, X, dtype).astype(jnp.float32)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

def sgd_step(params, X, y, dtype):
    grads = jax.grad(loss_fn)(params, X, y, dtype)
    return [p - 0.01 * g for p, g in zip(params, grads)]

step = jax.jit(sgd_step, static_argnums=3)
args = mlp_params, X_mlp, y_mlp
```

Next, we count the exact multiply-accumulate operations (`macs`) and total
`flops` for the forward and backward matrix products of the MLP and define
`report` to benchmark each precision format in milliseconds and TFLOP/s:

```{.python .input #memory-precision-mixed-precision-3}
%%tab pytorch, jax
# Two FLOPs per multiply-add, in the forward pass, the weight gradients, and
# the input gradients of every layer but the first
macs = sum(i * o for i, o in sizes)  # Per example and pass
flops = 2 * len(X_mlp) * (3 * macs - sizes[0][0] * sizes[0][1])
print(f'{flops:.2e} FLOPs per step')

def report(desc, f):
    """Print the time per call of f and the arithmetic rate it achieves."""
    t = d2l.Benchmark(f).time
    print(f'{desc}: {1000 * t:5.2f} ms, {flops / t / 1e12:3.0f} TFLOP/s')
```

We now run `report` across strict scalar `'fp32'`, tensor-core `'tf32'`, and
mixed-precision `'bf16'`:

```{.python .input #memory-precision-mixed-precision}
%%tab pytorch
torch.set_float32_matmul_precision('highest')  # True fp32
report('fp32', lambda: step(False))
torch.set_float32_matmul_precision('high')  # fp32 matmuls may use tf32
report('tf32', lambda: step(False))
report('bf16', lambda: step(True))
```

```{.python .input #memory-precision-mixed-precision}
%%tab jax
with jax.default_matmul_precision('highest'):  # True fp32
    report('fp32', lambda: step(*args, jnp.float32))
report('tf32', lambda: step(*args, jnp.float32))
report('bf16', lambda: step(*args, jnp.bfloat16))
```

True fp32 on scalar units reaches ~60% of its 82.6 TFLOP/s peak because every
multiply-add and address calculation competes for instruction issue slots;
tf32 on the tensor cores reaches ~90% of the same peak, cutting step time by
a third without changing stored types. Moving from tf32 to bf16 speeds up the
step by another $1.5\text{--}1.75\times$ (close to the $6.7 + 1.3 = 8.0$ ms
floor plus update, since the fp32 SGD update and weight casts do not shrink).

## Activation Checkpointing
:label:`subsec_mp-checkpointing`

To multiply the number of tokens that fit on a card, a step must store fewer
activations. *Activation checkpointing* (:numref:`subsec_activation_checkpointing`,
:cite:`Chen.Xu.Zhang.ea.2016`) saves only the input of each Transformer block
($d$ fp32 values, or $4d$ bytes per token) during the forward pass and
recomputes the block's internal activations one block at a time during the
backward pass. This trades one extra forward pass (~33% more compute, or $8N$
FLOPs/token instead of $6N$) for reducing peak per-token memory across $L$
blocks from $L a + a_\textrm{out} + \Delta$ to at most

$$
L \cdot 4d + a + a_\textrm{out} + \Delta,
$$

where $a$ is the single block currently being recomputed and $\Delta$ is the
excess of the measured peak over the count (:numref:`subsec_mp-measuring`).

```{.python .input #memory-precision-activation-checkpointing-1}
%%tab pytorch, jax
M_gpu, d, L = 24 * 2**30, config['num_hiddens'], config['num_blks']
slope = (peaks[8192] - peaks[4096]) / 4096  # Growth per token, measured
excess = slope - (2 * block + head)  # Measured peak above the count
budgets = {  # Configuration: (fixed bytes, bytes per token at the peak)
    'fp32, count': (16 * N, L * block + head),
    'fp32, measured': (16 * N, L * block + head + excess),
    'bf16, measured': (18 * N, L * block16 + head16 + excess),
    'fp32, checkpointing': (16 * N, L * 4 * d + block + head + excess),
    'bf16, checkpointing': (18 * N, L * 4 * d + block16 + head16 + excess)}
for name, (fixed, per_token) in budgets.items():  # M(n) <= M_gpu
    print(f'{name:19s}: {per_token / 2**20:.2f} MiB per token, room for '
          f'{(M_gpu - fixed) / per_token:6,.0f} tokens')
```

With per-block checkpointing, the 18 blocks together store only
$18 \times 8 = 144$ KiB/token (less than the output layer and loss alone), raising
the token budget more than threefold (to 10,000–15,000 tokens). Notice that under checkpointing, bf16 actually fits *fewer*
tokens than fp32: the extra $2N$ bytes of bf16 weight copies, together with
(in PyTorch) a larger bf16 output layer and loss, outweigh the activation
savings on the single recomputed block. At that point, the
output logits and cross-entropy loss dominate per-token memory; computing the
output projection and loss in token chunks inside a checkpoint of its own
eliminates that bottleneck for only $2dV / (6N) \approx 3\%$ extra FLOPs
(Exercise :exref:`ex_mp-chunked-loss`).

Let us measure per-block checkpointing on a 4-block stack with two 1,024-token
sequences (which should store $5 \times 4d = 40$ KiB/token across the four
block inputs and final output):

```{.python .input #memory-precision-activation-checkpointing}
%%tab pytorch
blocks = d2l.GPT(**{**config, 'num_blks': 4}).to(gpu).blks
def stack(H, remat):
    for blk in blocks:
        H = checkpoint(blk, H, use_reentrant=False) if remat else blk(H)
    return H.square().mean()

d = config['num_hiddens']
H1, H2 = (torch.randn(b, 1024, d, device=gpu, requires_grad=True)
          for b in (1, 2))
for remat in (False, True):
    stored = (saved_bytes(stack, H2, remat)
              - saved_bytes(stack, H1, remat)) / 1024
    stack(H2, remat).backward()  # Warm-up; allocates the gradients
    peak = d2l.peak_memory(lambda: stack(H2, remat).backward())
    t = d2l.Benchmark(lambda: stack(H2, remat).backward()).time
    print(f'checkpointing={remat!s:5}: stores {stored / 2**10:4.0f} KiB per '
          f'token, peak {peak / 2**30:.2f} GiB, {1000 * t:.0f} ms per step')
```

```{.python .input #memory-precision-activation-checkpointing}
%%tab jax
blocks = [nnx.split(blk, nnx.Param, ...) for blk in d2l.GPT(
    **{**config, 'num_blks': 4}, rngs=nnx.Rngs(d2l.get_key())).blks]
blk_params = [p for _, p, _ in blocks]
def stack(params, H, remat):
    def block(graphdef, p, rest, H):  # A block as a function of its weights
        return nnx.merge(graphdef, p, rest)(H)
    if remat:
        # On a CPU, XLA's default scheduler runs the recomputations early,
        # so that they coexist and the saving vanishes; setting XLA_FLAGS to
        # --xla_cpu_enable_concurrency_optimized_scheduler=false before JAX
        # starts restores it there
        block = jax.checkpoint(block, static_argnums=0)
    for (graphdef, _, rest), p in zip(blocks, params):
        H = block(graphdef, p, rest, H)
    return (H ** 2).mean()

d = config['num_hiddens']
H1, H2 = (jax.random.normal(d2l.get_key(), (b, 1024, d)) for b in (1, 2))
for remat in (False, True):
    f = lambda params, H: stack(params, H, remat)
    stored = (saved_bytes(f, blk_params, H2)
              - saved_bytes(f, blk_params, H1)) / 1024
    stack_grad = jax.jit(jax.grad(f))
    peak = d2l.peak_memory(stack_grad, blk_params, H2)
    t = d2l.Benchmark(lambda: stack_grad(blk_params, H2)).time
    print(f'checkpointing={remat!s:5}: stores {stored / 2**10:4.0f} KiB per '
          f'token, peak {peak / 2**30:.2f} GiB, {1000 * t:.0f} ms per step')
```

Checkpointing reduces the saved tensors to the exact 40 KiB/token predicted,
cuts peak activation memory substantially, and lengthens step time by ~25%
(within the 33% bound of one extra forward pass). Instead of recomputing an
entire block, *selective recomputation* :cite:`Korthikanti.Casper.Lym.ea.2022`
saves the expensive matrix-multiplication outputs and recomputes only cheap
elementwise activations such as the SwiGLU gate product (Exercise
:exref:`ex_mp-selective`).

## Gradient Accumulation
:label:`subsec_mp-accumulation`

To reach a global batch $B_\textrm{global} = m \, B_\textrm{micro}$ of
hundreds of thousands of tokens without storing more activations, *gradient
accumulation* splits the global batch into $m$ sequential micro-batches,
accumulates their gradients (dividing each micro-batch mean loss by $m$), and
runs the optimizer once at the end.

While gradient accumulation performs the same arithmetic as a single large
batch, each micro-batch after the first reads the running gradient accumulator
and the new gradient and writes their sum ($12N$ bytes of memory traffic),
while the micro-batch itself performs $6Nn$ FLOPs on $n$ tokens. Relative to
peak compute time, this accumulator traffic costs

$$
\frac{12N / \beta}{6Nn / P} = \frac{2 I^*}{n},
$$

where $I^* = P/\beta$ is the ridge point (:numref:`sec_perf_model`). For tf32
($I^* = 82$) and micro-batches of $n = 1{,}024$ tokens, that overhead is ~16%
on each of the $m - 1$ accumulating micro-batches; at $n = 4{,}096$ tokens, it
drops to 4%.

We verify both numerical equivalence and memory/time overhead on our 2-block
GPT for 4,096 tokens processed as 1 batch versus $m = 4$ micro-batches of
1,024 tokens:

```{.python .input #memory-precision-accumulated-step}
%%tab pytorch
X = torch.randint(0, V, (4, 1024), device=gpu)  # 4,096 tokens per step

def accumulate(model, X, m):
    """Gradient of the loss on X in .grad, accumulated over m micro-batches."""
    model.zero_grad(set_to_none=True)
    for Xb in X.chunk(m):
        (gpt_loss(model, Xb) / m).backward()  # / m: the loss averages

accumulate(model, X, 1)
full = [p.grad.clone() for p in model.parameters()]  # One-batch gradient
for m in (1, 4):
    model.zero_grad(set_to_none=True)
    peak = d2l.peak_memory(accumulate, model, X, m)
    assert max((p.grad - g).norm() / g.norm()
               for p, g in zip(model.parameters(), full)) < 1e-2
    f = lambda: accumulate(model, X, m)
    times = 1000 * np.array([d2l.Benchmark(f).time for _ in range(5)])  # ms
    kernels = d2l.kernel_times(f)        # Device ms per step, per kernel
    busy = sum(kernels.values())
    products = sum(ms for name, ms in kernels.items()
                   if re.search(d2l.PRODUCT_KERNELS, name, re.I))
    print(f'{m} x {4096 // m:4d} tokens: peak {peak / 2**30:.2f} GiB above '
          f'the weights, {np.median(times):.0f} ms per step '
          f'({times.min():.0f} to {times.max():.0f}), device busy '
          f'{busy:.0f} ms\n{"":16s}matrix products {products:.0f} ms, '
          f'other kernels {busy - products:.0f} ms')
```

```{.python .input #memory-precision-accumulated-step}
%%tab jax
model = d2l.GPT(**{**config, 'num_blks': 2}, rngs=nnx.Rngs(d2l.get_key()))
graphdef, params, rest = nnx.split(model, nnx.Param, ...)
X = jax.random.randint(d2l.get_key(), (4, 1024), 0, config['vocab_size'])

def accumulate(params, rest, X, m):
    """Gradient of the loss on X, accumulated over m micro-batches."""
    loss = lambda params, Xb: gpt_loss(nnx.merge(graphdef, params, rest), Xb)
    def add(g, Xb):  # One micro-batch; / m because the loss averages
        g_b = jax.grad(loss)(params, Xb)
        return jax.tree.map(lambda a, b: a + b / m, g, g_b), None
    zeros = jax.tree.map(jnp.zeros_like, params)
    return jax.lax.scan(add, zeros, X.reshape(m, -1, X.shape[-1]))[0]

accumulate = jax.jit(accumulate, static_argnums=3)
full = accumulate(params, rest, X, 1)  # One-batch gradient
for m in (1, 4):
    plan = accumulate.lower(params, rest, X, m).compile().memory_analysis()
    peak = plan.output_size_in_bytes + plan.temp_size_in_bytes
    assert max(jax.tree.leaves(jax.tree.map(
        lambda a, b: jnp.linalg.norm(a - b) / jnp.linalg.norm(b),
        accumulate(params, rest, X, m), full))) < 1e-2
    f = lambda: accumulate(params, rest, X, m)
    times = 1000 * np.array([d2l.Benchmark(f).time for _ in range(5)])  # ms
    kernels = d2l.kernel_times(f)        # Device ms per step, per kernel
    busy = sum(kernels.values())
    products = sum(ms for name, ms in kernels.items()
                   if re.search(d2l.PRODUCT_KERNELS, name, re.I))
    print(f'{m} x {4096 // m:4d} tokens: peak {peak / 2**30:.2f} GiB above '
          f'the weights, {np.median(times):.0f} ms per step '
          f'({times.min():.0f} to {times.max():.0f}), device busy '
          f'{busy:.0f} ms\n{"":16s}matrix products {products:.0f} ms, '
          f'other kernels {busy - products:.0f} ms')
```

Peak memory above the weights drops by roughly half (since the $4N = 0.77$ GiB
of persistent gradients remain allocated across micro-batches while activation
memory drops fourfold), and step time increases by 10–20% due to accumulator
additions and slightly lower tensor-core efficiency on smaller micro-batches.
The optimal micro-batch is therefore always the largest one that fits in
memory.

## Choosing a Memory Technique
:label:`subsec_mp-ladder`

:numref:`tab_mp-techniques` summarizes the memory techniques by the symptom
that calls for each.

:Memory techniques by symptom, with their effect on :eqref:`eq_mp-budget` and their cost.
:label:`tab_mp-techniques`

| symptom | technique | effect on memory | cost |
|---|---|---|---|
| matrix products dominate step time | bf16 mixed precision | activations $\times 0.6$, fixed state $+2N$ for weight copies | 8-bit significand rounding (fp16 also needs loss scaling) |
| a useful micro-batch does not fit | activation checkpointing | stored activations per block from $a$ to $4d$ bytes/token | one extra forward pass (~25–33% more time) |
| output layer and loss dominate | chunked cross-entropy loss | $a_\textrm{out}$ and $\Delta$ for one token chunk at a time | one extra pass through the output projection (~3% FLOPs) |
| global batch exceeds micro-batch capacity | gradient accumulation | none; global batch grows $m$-fold | accumulator traffic $2I^*/n$; repeated weight reads per micro-batch |
| $16N$ parameter state alone fills the card | sharding weights, gradients, and optimizer state (:numref:`sec_sharding`) | $16N/k$ on each of $k$ GPUs | inter-GPU collective communication |

For our 1B-parameter GPT on a single RTX 4090, bf16 mixed precision fits two
or three sequences of 1,024 tokens without checkpointing, and accumulating 256
micro-batches reaches a global batch of 524,288 tokens. At $6N$ FLOPs/token,
however, one global step requires $3.2 \times 10^{15}$ FLOPs—at least 19 s per
step on one card, or more than three weeks for 100,000 steps (52B tokens). A
single GPU can hold a step of our GPT, but finishing the run requires multiple
GPUs (:numref:`sec_multi_gpu`).

## Summary

Training step memory splits into a fixed parameter term ($16N$ bytes for fp32
Adam, $18N$ with bf16 weight copies) and a per-token activation term. By three
differentiation rules (linear layers keep inputs, elementwise functions keep
inputs or outputs, products keep both factors), a fused SwiGLU Transformer
block stores ~$20d$ values per token, and the cross-entropy loss stores $V$
log-probabilities; the peak can exceed this count by backward temporaries at
the loss, which is why we measure it. Mixed precision accelerates
matrix products on tensor cores while leaving total memory capacity nearly
unchanged; activation checkpointing stores only block inputs ($4d$ bytes/token)
at the cost of ~25% extra time; and gradient accumulation builds large global
batches from micro-batches that fit, adding an accumulator traffic fraction of
$2I^*/n$.

## Exercises

1. **Where the peak lands.** :numref:`fig_memory_anatomy` places the peak
   of a training step at the start of the backward pass. Consider a stack of
   $n$ identical layers where each layer's parameters occupy $w$ bytes and
   its forward pass stores $a$ bytes of activations. Assume no gradients
   exist when the backward pass starts, and each layer's backward pass frees
   its $a$ bytes of activations and allocates $w$ bytes of parameter
   gradients.
    1. Write the bytes held by activations and gradients after the backward
       pass has processed $j$ of the $n$ layers. Show that the peak lies at
       the start of the backward pass when $a > w$ and at its end when
       $a < w$.
    1. A block of our GPT holds ~$12d^2$ parameters and stores ~$20d$ fp32
       values per token. Above what token count per step does the peak lie at
       the start of the backward pass?
    1. Under gradient accumulation over $m$ micro-batches, the gradients of
       the first micro-batch remain allocated while later micro-batches run.
       Find the peak with accumulation and the values of $m$ for which
       accumulation lowers the peak below a single full-batch step.
   :exlabel:`ex_mp-peak`

:begin_tab:`pytorch`
2. [code] **Optimizer-step memory.** Build the two-block GPT of
   :numref:`subsec_mp-measuring` with three `torch.optim.Adam` configurations:
   `foreach=False`, the default, and `fused=True`.
    1. Run two steps with each optimizer. On the second step, measure the
       peak extra bytes allocated during `opt.step()` in units of the gradient
       size $4N$.
    1. Explain the three measurements from how each implementation updates
       parameter tensors (one tensor at a time, batched multi-tensor
       elementwise ops, or a single fused kernel). When can the optimizer step
       set the overall peak of a training step?
:end_tab:

:begin_tab:`jax`
2. [code] **Optimizer-step memory.** Extend the compiled gradient of
   :numref:`subsec_mp-measuring` into a full Adam update using `optax.adam`.
    1. Compile the step for 512 tokens and report `argument_size_in_bytes`,
       `output_size_in_bytes`, and `temp_size_in_bytes` from
       `memory_analysis()` in units of $4N$.
    1. Recompile with `donate_argnums` covering the parameters and optimizer
       state, report `alias_size_in_bytes`, and explain how buffer donation
       changes the peak memory.
:end_tab:

3. **Clipping under accumulation.** Gradient clipping to norm $\tau$
   replaces $\mathbf{g}$ by $\mathbf{g} \min(1, \tau / \|\mathbf{g}\|)$. Let
   $\mathbf{g}_1, \ldots, \mathbf{g}_m$ be micro-batch gradients scaled by
   $1/m$ so their sum $\mathbf{g}$ is the global-batch gradient.
    1. If a loop clips each $\mathbf{g}_i$ before adding it to the
       accumulator, what is the maximum norm the accumulated gradient can
       reach, and when does per-micro-batch clipping match global clipping?
    1. Under fp16 loss scaling with factor $S$, the accumulator holds
       $S \mathbf{g}$ after the final micro-batch. In what order must the loop
       unscale by $S$, clip, and step? What goes wrong if $S$ changes between
       micro-batches?

:begin_tab:`pytorch`
4. [code] **fp16 gradient window.** ● The 5-bit exponent of fp16 gives it
   normal numbers from $2^{-14}$ to 65,504 and subnormals down to $2^{-24}$.
   Loss scaling :cite:`Micikevicius.Narang.Alben.ea.2018` multiplies the loss
   by $S$ before `backward()` and divides gradients by $S$ afterward. Using
   `mlp`, `X_mlp`, and `y_mlp` from :numref:`subsec_mp-mixed` with true fp32
   reference gradients:
    1. Plot a histogram of $\log_2 |g|$ over all nonzero reference gradient
       entries and find the scale factors $S$ that place 99.9% of scaled
       entries $Sg$ inside the normal range of fp16.
    1. For $S = 2^0, 2^2, \ldots, 2^{24}$, compute gradients under
       `torch.autocast('cuda', dtype=torch.float16)`, unscale by $S$, and plot
       the relative error $\|\hat{\mathbf{g}} - \mathbf{g}\| / \|\mathbf{g}\|$.
       Repeat with `torch.bfloat16`. In which window of $S$ is fp16 more
       accurate than bf16, and why?
    1. How does the valid window of $S$ shift if the batch size grows from
       2,048 to 32,768?
:end_tab:

:begin_tab:`jax`
4. [code] **fp16 gradient window.** ● The 5-bit exponent of fp16 gives it
   normal numbers from $2^{-14}$ to 65,504 and subnormals down to $2^{-24}$.
   Loss scaling :cite:`Micikevicius.Narang.Alben.ea.2018` multiplies the loss
   by $S$ before differentiation and divides gradients by $S$ afterward. Using
   `mlp_params`, `X_mlp`, `y_mlp`, and `loss_fn` from
   :numref:`subsec_mp-mixed` with true fp32 reference gradients:
    1. Plot a histogram of $\log_2 |g|$ over all nonzero reference gradient
       entries and find the scale factors $S$ that place 99.9% of scaled
       entries $Sg$ inside the normal range of fp16.
    1. For $S = 2^0, 2^2, \ldots, 2^{24}$, compute gradients with
       `jnp.float16`, unscale by $S$, and plot the relative error
       $\|\hat{\mathbf{g}} - \mathbf{g}\| / \|\mathbf{g}\|$. Repeat with
       `jnp.bfloat16`. In which window of $S$ is fp16 more accurate than bf16,
       and why?
    1. How does the valid window of $S$ shift if the batch size grows from
       2,048 to 32,768?
:end_tab:

:begin_tab:`pytorch`
5. [code] **Selective recomputation.** ● Per-block checkpointing stores only
   each block's input. Passing
   `torch.utils.checkpoint.create_selective_checkpoint_contexts` via the
   `context_fn` argument of `checkpoint` caches the outputs of selected ops
   (such as `torch.ops.aten.mm.default`) and recomputes the rest.
    1. For one block of our GPT, list the saved tensors and the recomputed
       matrix products under three policies: full per-block checkpointing,
       caching every `mm` output, and caching all `mm` outputs except the FFN
       gate and up-projection.
    1. Implement the latter two policies on the 4-block stack of
       :numref:`subsec_mp-checkpointing` and compare saved bytes/token, peak
       memory, and step time.
   :exlabel:`ex_mp-selective`
:end_tab:

:begin_tab:`jax`
5. [code] **Selective recomputation.** ● `jax.checkpoint` accepts a `policy`
   argument such as
   `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`, which saves the
   outputs of weight matrix multiplications and recomputes elementwise ops.
    1. For one block of our GPT, list the saved residuals and recomputed
       matrix products under no checkpointing, default `jax.checkpoint`, and
       `dots_with_no_batch_dims_saveable`, verifying with
       `jax.ad_checkpoint.print_saved_residuals`.
    1. Add this policy to the 4-block benchmark of
       :numref:`subsec_mp-checkpointing` and compare saved bytes/token,
       planned peak memory, and step time.
   :exlabel:`ex_mp-selective`
:end_tab:

6. [extended] **The loss in chunks.** ●● Under per-block checkpointing, the
   output layer and cross-entropy loss of our GPT store more memory per token
   than all 18 blocks combined because the loss materializes $V = 50{,}257$
   log-probabilities and their gradients per token.
    1. Write the output projection and cross-entropy loss to process tokens in
       chunks of size $n_c = 1024$, wrapping each chunk in a checkpoint so
       that logits and log-probabilities exist for only one chunk at a time.
       Verify that the loss and gradients match the unchunked version.
    1. Predict the maximum micro-batch size of our 18-block GPT on a 24 GiB
       card with per-block checkpointing and the chunked loss, and measure the
       per-token memory slope and step time on the 2-block GPT.
   :exlabel:`ex_mp-chunked-loss`

[Discussions](https://d2l.discourse.group/t/36790)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.4]{.kicker}

Memory Hierarchy, Precision, and Checkpointing<br>**parameter & activation budgets · mixed precision (`bf16`/`tf32`) · activation checkpointing · gradient accumulation**
:::
:::

::: {.slide title="A 1B-Parameter GPT Fills 15 of 24 GiB Before Any Data"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Fixed parameter state ($16N = 15.1\text{ GiB}$):** On a 18-block, 1B GPT, `fp32` weights ($4N$), gradients ($4N$), and Adam moments ($8N$) leave only $8.9\text{ GiB}$ on a $24\text{ GiB}$ RTX 4090.
- **Per-token activations ($\sim 3\text{--}4\text{ MiB/tok}$):** Retaining 18 blocks of forward activations fits only **2–3 sequences** of $1{,}024$ tokens per micro-batch.
- **Closing the batch gap:** Reaching $0.5\text{M+}$ token global batches requires mixed precision, activation checkpointing, and gradient accumulation.
:::

::: {.col .fig .big}
@fig:ch13-perf-24gib-squeeze
:::
:::
:::

::: {.slide title="Fixed Training State: 16 Bytes per Parameter in Adam"}
[Parameter Memory]{.kicker}

The memory occupied by model parameters, gradients, and optimizer state depends only on parameter count $N$, optimizer choice, and precision—never on batch size:

| Optimizer & Precision Configuration | Stored Arrays per Parameter | Bytes / Param | 1.01B GPT State |
| :--- | :--- | :---: | :---: |
| **Plain SGD (`float32`)** | Weight ($4\text{ B}$) + Gradient ($4\text{ B}$) | $8\text{ B}$ | $7.5\text{ GiB}$ |
| **SGD + Momentum (`float32`)** | Weight + Gradient + Velocity ($4\text{ B}$) | $12\text{ B}$ | $11.3\text{ GiB}$ |
| **Adam / AdamW (`float32`)** | Weight + Gradient + 1st & 2nd Moments ($8\text{ B}$) | **$16\text{ B}$** | **$15.1\text{ GiB}$** |
| **Adam, Mixed Precision (`bf16`)** | All four in `fp32` ($16\text{ B}$) + `bf16` weight copy ($2\text{ B}$) | **$18\text{ B}$** | **$17.0\text{ GiB}$** |

@!memory-precision-memory-per-parameter
:::

::: {.slide title="Activation Lifecycle: Peak at Backward Pass Start"}
[Activation Memory]{.kicker}

![](../img/mdl-perf-memory-anatomy.svg){width=76%}

- **Forward accumulation, backward release:** Each forward layer stashes activations needed by its backward VJP; during backpropagation, each layer consumes and frees its saved activations as it materializes parameter gradients.
- **Where memory peaks:** Total allocated memory reaches its maximum right at the **transition from the forward loss to the start of the backward pass**.
:::

::: {.slide title="Three Differentiation Rules Govern Saved Activations"}
[Backpropagation Storage]{.kicker}

::: {.cols .vc}
::: {.col}
- **Linear layer ($\mathbf{Y} = \mathbf{X}\mathbf{W}$):** Must save input $\mathbf{X}$ because the weight gradient $\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^\top(\nabla_{\mathbf{Y}}\mathcal{L})$ multiplies by $\mathbf{X}$.
- **Elementwise nonlinearity ($\mathbf{y} = f(\mathbf{x})$):** Saves $\mathbf{x}$ (or $\mathbf{y}$) to evaluate the derivative $f'(\mathbf{x})$.
- **Hadamard product ($\mathbf{z} = \mathbf{p} \odot \mathbf{q}$):** Saves **both** $\mathbf{p}$ and $\mathbf{q}$, which is why SwiGLU alone stores four hidden vectors ($\approx 10.7d$ values/token).
:::

::: {.col .fig .big}
@fig:ch13-perf-vjp-save-rules
:::
:::
:::

::: {.slide title="Tallying a Transformer Block: ~20d Values per Token"}
[Per-Token Tally]{.kicker}

::: {.cols .vc}
::: {.col}
- **Norms & Attention ($8d\text{--}9d$):** Two RMSNorms + residual inputs save $4d$, while fused FlashAttention saves $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}$ ($4d\text{--}5d$, avoiding $T \times T$).
- **SwiGLU FFN ($\approx 10.7d$):** Saves gate $\mathbf{g}$, $\textrm{SiLU}(\mathbf{g})$, up $\mathbf{u}$, and their product at width $\frac{8}{3}d$, bringing each block to **$a \approx 18.7d\text{--}20d$** ($149\text{--}160\text{ KiB/tok}$ in `fp32`).
- **Output head ($a_{\textrm{out}} + \Delta$):** Final norm, $V$ `fp32` log-probs ($212\text{ KiB/tok}$), plus whatever excess $\Delta$ the measured peak shows above this count.
:::

::: {.col .fig .big}
@fig:ch13-perf-block-activation-tally
:::
:::
:::

::: {.slide title="Verifying Saved Bytes: 18.7d per Fused Block" only="pytorch"}
[Autograd Storage Hook]{.kicker}

By attaching `torch.autograd.graph.saved_tensors_hooks` and taking differences between 1 and 2 sequences and between 1 and 2 blocks, we isolate the exact bytes saved per token by one block ($a$) and by the output head ($a_{\textrm{out}}$):

@!memory-precision-memory-per-token-2

- **Exact match to theory:** Because PyTorch runs `scaled_dot_product_attention`, `RMSNorm`, and `F.silu` as single fused kernels, each block stores the theoretical minimum of **$18.7d = 149\text{ KiB/token}$**.
:::

::: {.slide title="Verifying Saved Bytes: Unfused Residuals vs. XLA" only="jax"}
[VJP Residual Tally]{.kicker}

Inspecting the raw VJP residuals returned by `jax.vjp` before XLA compilation and subtracting across 1 and 2 sequences and 1 and 2 blocks measures the uncompiled residual footprint:

@!memory-precision-memory-per-token-2

- **Why raw `jax.vjp` records $29.0d$ ($232\text{ KiB/token}$):** Because `nnx.RMSNorm` and `jax.nn.silu` are written in Python as composite elementwise expressions, their internal intermediates appear in the uncompiled VJP tree (XLA later fuses and recomputes many of them during `jax.jit`).
:::

::: {.slide title="The Step Memory Budget on a 24 GiB GPU"}
[Memory Budget]{.kicker}

Combining the fixed parameter state ($16N$ bytes when gradients persist across micro-batches) with $L$ blocks of $a$ bytes/token and output head $a_{\textrm{out}}$ bytes/token yields the total memory budget for $n$ tokens:

$$M(n) \;\approx\; 16N \;+\; n\,(L a + a_{\textrm{out}}) \;\le\; M_{\textrm{GPU}}$$

- **Fixed parameter floor ($N = 1.01 \times 10^9$):** $16N = 15.1\text{ GiB}$, leaving only $M_{\textrm{GPU}} - 16N = \mathbf{8.9\text{ GiB}}$ for activations and backward temporaries on a $24\text{ GiB}$ RTX 4090.
- **Maximum micro-batch token count ($L = 18$ blocks):** At $La + a_{\textrm{out}} \approx 2.8\text{--}4.3\text{ MiB/token}$, at most **$2{,}000\text{--}3{,}200$ tokens** (2 or 3 sequences of length $1{,}024$) can fit in a single micro-batch.
- **Doubling depth to 36 blocks ($N = 1.92\text{B}$):** Fixes $16N = 28.6\text{ GiB} > 24\text{ GiB}$—not even a single token fits without sharding the $16N$ state across multiple GPUs (§{sec_sharding}).
:::

::: {.slide title="Measured Peak Memory: The 8V Cross-Entropy Excess" only="pytorch"}
[Peak Memory vs. Tally]{.kicker}

Measuring actual peak memory (`torch.cuda.max_memory_allocated`) across $1{,}024$ to $8{,}192$ tokens on a 2-block GPT reveals an extra $\Delta \approx 393\text{ KiB/token}$ above the saved-activation tally:

@!memory-precision-measuring-memory-1

- **Where the extra $393\text{ KiB/token}$ comes from:** At the very start of `backward()`, while all forward activations and the $V$ saved log-probabilities are still live, the cross-entropy backward pass materializes two `float32` gradient tensors of width $V = 50{,}257$ (for log-probabilities and logits: $2 \times 4V = 8V \approx 393\text{ KiB/token}$).
:::

::: {.slide title="Planned Peak Memory via XLA memory_analysis()" only="jax"}
[Peak Memory vs. Tally]{.kicker}

In JAX, XLA plans static buffer assignments ahead of time; querying `compile().memory_analysis()` on `ShapeDtypeStruct` placeholders predicts peak memory without allocating a single GPU tensor:

@!memory-precision-measuring-memory-3

- **The plan matches the count:** From $4{,}096$ to $8{,}192$ tokens the planned peak grows by about $690\text{ KiB/token}$, the same as the $692\text{ KiB}$ of the raw VJP count ($2 \times 29d$ plus the output head), so $\Delta \approx 0$ and the count can serve as the budget.
:::

::: {.slide title="Allocation Timeline Across Forward, Backward, and Step" only="pytorch"}
[Allocator Trace]{.kicker}

Recording every CUDA caching-allocator event across one Adam step on $4{,}096$ tokens visualizes the sawtooth lifecycle of activations and gradients:

@!memory-precision-measuring-memory-2

- **Three distinct phases:** Memory starts at $12N$ (weights + two Adam moments), climbs staircase-fashion across the blocks during the **forward pass**, spikes at the logits and initial **backward** loss gradients, steps down as activations are freed, and briefly allocates temporaries during the **optimizer step**.
:::

::: {.slide title="Mixed Precision: Master fp32 State with bf16 Compute"}
[Mixed Precision]{.kicker}

::: {.cols .vc}
::: {.col}
- **Linear & Attention Matmuls in `bfloat16` ($2\text{ B}$):** Doubles Tensor Core throughput to $165\text{ TFLOP/s}$ and cuts per-block activation storage by $\sim 40\%$.
- **Master State & Reductions in `float32` ($4\text{ B}$):** Prevents small updates ($\eta g \sim 10^{-5}w$) from rounding to zero and preserves RMSNorm, residual, and cross-entropy accuracy.
- **No loss scaling needed:** `bfloat16` shares the 8-bit exponent of `float32` ($\sim 10^{\pm 38}$ range).
:::

::: {.col .fig .big}
@fig:ch13-perf-mixed-precision-flow
:::
:::
:::

::: {.slide title="bf16 Cuts Block Activations by 40% but Adds 2N State"}
[Precision & Capacity]{.kicker}

In `bfloat16` mixed precision, the two residual-stream inputs per block remain in `float32` ($4\text{ B}$) while the other $18$ of the $\sim 20d$ saved vectors drop to `bfloat16` ($2\text{ B}$), predicting $(2 \cdot 4 + 18 \cdot 2) / (20 \cdot 4) = 0.55$ of the `float32` block activation size:

@!memory-precision-mixed-precision-1

- **Why total token capacity barely grows ($\sim 15\text{--}27\%$):** Caching `bfloat16` weight copies during the step raises fixed parameter memory from $16N = 15.1\text{ GiB}$ to $18N = 17.0\text{ GiB}$ (shrinking activation headroom from $8.9 \to 7.0\text{ GiB}$), and the large `float32` loss log-probabilities ($a_{\textrm{out}}$) do not shrink.
:::

::: {.slide title="Tensor Core Speedup: Scalar fp32 vs. tf32 vs. bf16"}
[Precision & Throughput]{.kicker}

Mixed precision is primarily a **speed** optimization. Benchmarking one SGD step of a compute-bound 3-layer width-8192 MLP ($1.1 \times 10^{12}\text{ FLOPs/step}$; compute floor $13.3\text{ ms}$ at $82.6\text{ TFLOP/s}$ and $6.7\text{ ms}$ at $165.2\text{ TFLOP/s}$, plus $1.3\text{ ms}$ for the `fp32` SGD update) separates Tensor Core execution from 16-bit operands:

@!memory-precision-mixed-precision

- **Scalar `fp32` $\to$ Tensor Core `tf32` ($\sim 1.5\times$):** Same $82.6\text{ TFLOP/s}$ theoretical ceiling and 32-bit storage, but Tensor Core tile instructions reach $\sim 90\%$ utilization vs. $\sim 60\%$ on scalar ALUs.
- **Tensor Core `tf32` $\to$ `bf16` ($1.5\text{--}1.75\times$):** Matmul time halves to $\sim 6.7\text{ ms}$, while the bandwidth-bound `fp32` SGD update ($1.3\text{ ms}$) and weight casts remain constant.
:::

::: {.slide title="Activation Checkpointing: O(L) Inputs, One Block Live"}
[Activation Checkpointing]{.kicker}

::: {.cols .vc}
::: {.col}
$$\textrm{Peak/tok} \;=\; L \cdot 4d \;+\; a \;+\; a_{\textrm{out}} \;+\; \Delta$$

- **Forward pass:** Stores only the `fp32` input at each of the $L = 18$ block boundaries ($18 \times 8\text{ KiB} = \mathbf{144\text{ KiB/tok}}$, down from $2.7\text{ MiB/tok}$).
- **Backward pass:** Re-runs block $\ell$'s forward pass just before backpropagating through block $\ell$, keeping only **1 block's** internal activations ($a$) live at a time for $\le 33\%$ extra FLOPs.
:::

::: {.col .fig}
@fig:ch13-perf-activation-checkpointing
:::
:::
:::

::: {.slide title="Token Capacity Across Precision and Checkpointing"}
[Capacity Comparison]{.kicker}

Evaluating $M(n) \le 24\text{ GiB}$ for our 18-block, 1B-parameter GPT shows how checkpointing transforms activation capacity:

@!memory-precision-activation-checkpointing-1

- **More than $3\times$ larger micro-batches:** Per-block checkpointing expands capacity from $\sim 2{,}800$ tokens to **$\sim 10{,}000\text{--}15{,}000$ tokens**.
- **Surprising interaction:** Under full per-block checkpointing, `bf16` fits *fewer* tokens than `fp32` because its extra $2N = 1.9\text{ GiB}$ of `bf16` weight copies (and, in PyTorch, a larger output layer and loss) outweigh what the single recomputed block saves.
:::

::: {.slide title="Measuring Checkpointing: 40 KiB/Token and +25% Time"}
[Checkpointing Benchmark]{.kicker}

On a 4-block stack ($d = 2048$ in `float32`), checkpointing stores only the $4$ block inputs plus the final output ($5 \times 4d = 40\text{ KiB/token}$):

@!memory-precision-activation-checkpointing

- **Exact storage & modest overhead:** Saved bytes drop to the exact $40\text{ KiB/token}$ prediction, peak memory falls sharply, and step time increases by only **$\sim 25\%$** (below the $33\%$ FLOP bound because fused backward kernels absorb elementwise recomputation).
:::

::: {.slide title="Selective Recomputation and Chunked Cross-Entropy Loss"}
[Targeted Recomputation]{.kicker}

Once per-block checkpointing reduces all 18 blocks to $144\text{ KiB/token}$, two refinements target the remaining bottlenecks:

- **Selective recomputation (Korthikanti et al., 2022):** Matrix multiplications take most of a block's FLOPs while elementwise ops (SwiGLU's $\sim 5d$ activated gate and product, or unfused $T \times T$ attention scores) take most of its activation bytes. Caching `mm` outputs and recomputing only cheap elementwise ops saves significant memory for almost zero extra FLOPs.
- **Chunked output projection & cross-entropy loss:** Because the output logits and loss ($a_{\textrm{out}} + \Delta \approx 605\text{ KiB/token}$) now use **$4\times$ more memory per token than all 18 checkpointed blocks combined**, evaluating the output projection and cross-entropy on chunks of $1{,}024$ tokens inside a checkpoint eliminates the $V$-wide spike for only $2dV / (6N) \approx 3\%$ extra FLOPs.
:::

::: {.slide title="Gradient Accumulation and the 2I*/n Traffic Overhead"}
[Gradient Accumulation]{.kicker}

::: {.cols .vc}
::: {.col}
$$\frac{t_{\textrm{accumulator traffic}}}{t_{\textrm{micro-batch compute}}} \;=\; \frac{12N / \beta}{6Nn / P} \;=\; \frac{2 I^*}{n}$$

- **Reaching large global batches ($B_{\textrm{global}} = m\,B_{\textrm{micro}}$):** Sums $\frac{1}{m}\nabla\mathcal{L}_i$ across $m$ sequential micro-batches of $n$ tokens before stepping Adam once.
- **The $2I^*/n$ traffic tax:** Each micro-batch reads old grads ($4N$), new grads ($4N$), and writes their sum ($4N$); always use the largest micro-batch $n$ that fits without extra checkpointing.
:::

::: {.col .fig}
@fig:ch13-perf-grad-accumulation
:::
:::
:::

::: {.slide title="Measuring Accumulation: Halved Peak, Exact Gradients"}
[Accumulation Benchmark]{.kicker}

Splitting $4{,}096$ tokens on our 2-block GPT into $m = 4$ micro-batches of $1{,}024$ tokens reproduces the exact full-batch gradient while cutting activation memory fourfold:

@!memory-precision-accumulated-step

- **Peak memory & step time:** Peak memory above the weights drops by roughly half (to the $4N = 0.77\text{ GiB}$ persistent gradient buffer plus $1{,}024$ tokens of activations), and step time increases by $\sim 10\text{--}18\%$ as predicted by $2I^*/n$.
:::

::: {.slide title="Concept Check: Memory Budgets, Remat, and Accumulation"}
[Check Your Understanding]{.kicker}

Apply the memory and timing formulas of this section to two design questions for our 18-block, 1B-parameter GPT ($d = 2048$, `bf16` ridge $I^* = 164\text{ FLOP/byte}$):

1. Suppose we want a global batch of $524{,}288$ tokens on one $24\text{ GiB}$ RTX 4090 in `bfloat16`. Micro-batches of **2 sequences** ($n = 2{,}048$ tokens, $m = 256$) fit *without* checkpointing, whereas **4 sequences** ($n = 4{,}096$ tokens, $m = 128$) require per-block checkpointing ($\sim 25\%$ recomputation overhead). Which recipe finishes the global step faster?
2. At $6N\text{ FLOPs/token}$ and $165\text{ TFLOP/s}$ peak, what is the minimum compute time of one $524{,}288$-token global step on a single RTX 4090?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **2 sequences without checkpointing ($m = 256$) is faster:** Its accumulator overhead is $\frac{2 I^*}{n} = \frac{2 \times 164}{2048} \approx \mathbf{16\%}$, whereas 4 sequences with checkpointing pays $8\%$ accumulator overhead plus $\sim 25\%$ recomputation ($\approx \mathbf{33\%}$ total overhead). Never turn on checkpointing if a large enough micro-batch already fits.
2. **$6 \times (1.01 \times 10^9) \times 524{,}288 \approx 3.18 \times 10^{15}\text{ FLOPs/step}$**, taking at least **$\ge 19.2\text{ s/step}$** at $100\%$ peak ($\sim 22\text{ s}$ with accumulation), or over 25 days for $100{,}000$ steps. One GPU can hold the step, but finishing the run requires multiple GPUs.
:::
:::

::: {.slide title="Summary: Memory Breakdown and Optimization Techniques"}
[Key Takeaways]{.kicker}

| Technique / Component | Effect on Step Memory $M(n)$ | Time / Compute Cost | When to Use It |
| :--- | :--- | :--- | :--- |
| **Fixed Adam State** | $16N\text{ B}$ (`fp32`), $18N\text{ B}$ (`bf16`) | Fixed per step | Baseline parameter + gradient + 2 moment buffers |
| **Block Activations ($a$)** | $\approx 20d$ values/token ($160\text{ KiB}$ `fp32`) | Baseline forward/backward | Linear inputs ($4d$) + attn ($4d$) + SwiGLU ($11d$) |
| **`bf16` Mixed Precision** | Block activations $\times 0.6$, state $+2N$ | **$1.5\text{--}1.75\times$ faster** | Always on Tensor Core GPUs (no loss scaling needed) |
| **Block Checkpointing** | Block storage $La \to L \cdot 4d + a$ | $+25\text{--}33\%$ step time | When useful micro-batch $n$ does not fit in $M_{\textrm{GPU}}$ |
| **Chunked Cross-Entropy** | Eliminates $V$-wide $a_{\textrm{out}} + 8V$ peak | $+3\%$ FLOPs ($2dV$) | When vocab $V$ dominates checkpointed memory |
| **Gradient Accumulation** | Runs $B_{\textrm{global}} = m \, B_{\textrm{micro}}$ in $M(B_{\textrm{micro}})$ | $+\frac{2I^*}{n}$ accumulator traffic | To reach large global batches on limited GPUs |
:::
