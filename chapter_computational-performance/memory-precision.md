# Memory and Precision
:label:`sec_memory_precision`

The previous sections asked how fast a training step runs. A step that
does not fit in the GPU's memory does not run at all. We have met the
question once before, for a small MLP whose step took a sliver of a card
(:numref:`subsec_memory_during_training`), and :numref:`sec_hardware`
remarked in passing that our billion-parameter GPT, trained with Adam in
fp32, leaves an RTX 4090 less than ten gigabytes for everything else. How
much is that, counted in tokens? And how can a model that barely fits on a
card still train on large minibatches?

The arithmetic is short and sobering. The optimizer keeps four numbers for
every parameter: the weight, its gradient, and Adam's two moment
estimates. At four bytes each that is 15 GiB of the card's 24 GiB before
the model has read a single token. The remaining 9 GiB must hold what the
forward pass stores for the backward pass, and for this model that is
several megabytes for every token of the minibatch. Two or three sequences
of 1,024 tokens fit. GPT-3, for comparison, was trained on minibatches of
half a million to 3.2 million tokens :cite:`brown2020language`. A run that
does not fit does not start, and one that barely fits learns from too few
examples per step.

The memory of a step has two parts. One is proportional to the number of
parameters and fixed by the model and the optimizer. The other is
proportional to the number of tokens in the step: the tensors that the
forward pass keeps for the backward pass, called *activations*. We count
both for our GPT and check the count against a measurement of the peak.
Then we study three techniques that act on the second part. Mixed
precision stores activations in 16 bits. Activation checkpointing
recomputes activations instead of storing them, which takes extra time.
Gradient accumulation splits a large minibatch into pieces that fit one
at a time. None of the three reduces the memory per parameter, and that
part, together with the time of a step, is what will send us to several
GPUs.

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
%%tab pytorch
from torch.utils.checkpoint import checkpoint
```

## Memory of a Training Step
:label:`subsec_mp-anatomy`

We count the two parts in turn, first per parameter and then per token,
combine them into a budget, and then check the budget against what the
GPU actually holds.

### Memory per Parameter

What a parameter costs depends on the optimizer. Plain SGD keeps a weight
and its gradient, 8 bytes in fp32. Momentum adds a velocity, 12 bytes.
Adam keeps two moment estimates instead, 16 bytes per parameter, the count
that :numref:`subsec_memory_during_training` measured. None of these
numbers depends on the data: a model with $N$ parameters trained with Adam
holds $16N$ bytes whether it processes one token or a million. Mixed
precision, below, keeps all four numbers in fp32 and adds a 16-bit copy of
every weight for the duration of a step, $18N$ bytes in all.

Our model is `d2l.GPT`, the decoder-only transformer of :numref:`sec_gpt`,
which stacks pre-norm blocks of causal attention and a SwiGLU feed-forward
network between a token embedding and a tied output layer. We give it
width $d = 2048$, 16 attention heads, 18 blocks, and the vocabulary of
$V = 50{,}257$ tokens and the learned positions of GPT-2. A block holds
about $12d^2 = 50$ million parameters, so the blocks hold 0.9 billion and
the embedding table another 0.1 billion. These are the shapes of the GPT
that the chapter has been calling ours. Building it without its data, as in
:numref:`subsec_count_without_alloc`, confirms the count without
allocating 4 GB of weights:

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

Backpropagation needs intermediate results of the forward pass to compute
the gradients, so every layer stores some of its inputs or outputs until
the backward pass has used them. :numref:`fig_memory_anatomy` follows
these activations through one step. They accumulate during the forward
pass and are released one layer at a time during the backward pass, while
the gradients appear. Memory therefore peaks at the start of the backward
pass, when every activation is still stored.

![The memory of a training step with Adam, in units of $4N$ bytes. The
weights ($4N$) and Adam's moment estimates ($8N$) stay for the whole step.
Activations accumulate during the forward pass and are released during the
backward pass, while the gradients ($4N$) appear. Memory peaks at the start
of the backward pass.](../img/mdl-perf-memory-anatomy.svg)
:label:`fig_memory_anatomy`

Which tensors does a layer keep? Three rules cover a transformer block. A
linear layer $\mathbf{y} = \mathbf{W}\mathbf{x}$ keeps its input, because
the gradient of its weights is the product of the gradient of $\mathbf{y}$
with $\mathbf{x}^\top$. An elementwise function keeps its input or its
output, whichever its derivative is written in. A product of two
activations keeps both factors, because the gradient with respect to each
factor is the other one. The weights are needed as well, but they are
already counted in $16N$.

Let us apply the rules to one block, counting values per token in units of
the width $d$. The normalization before the attention keeps its input, $d$
values, and the projection to queries, keys, and values keeps the
normalized output, another $d$. The attention keeps the queries, keys, and
values, $3d$, and its output, $d$, from which a fused kernel computes the
derivative of the softmax. The projection after the attention keeps the
same output as its input. That is a second copy, $d$ more, if rearranging
the heads into one row per token copies the array, and nothing more if it
does not.

The feed-forward half of the block follows the same rules. Its
normalization keeps its input, $d$, and the gate and the up-projection
both read the same normalized output, which needs to be kept only once,
another $d$. The SwiGLU network computes a gate $\mathbf{g}$ and an
up-projection $\mathbf{u}$, each of width $\tfrac{8}{3}d$, and multiplies $\textrm{SiLU}(\mathbf{g})$ by
$\mathbf{u}$. The SiLU keeps $\mathbf{g}$, the product keeps both of its
factors, and the down-projection keeps the product: four vectors of width
$\tfrac{8}{3}d$, about $11d$. In all, a block keeps $19.7d$ values per
token with the second copy and $18.7d$ without. We round to $20d$, which is
160 KiB in fp32 at $d = 2048$. :citet:`Korthikanti.Casper.Lym.ea.2022`
count the same way for the block of GPT-3 without dropout and arrive at
$16d$ values besides the attention weights; our feed-forward network keeps
more because it has a gate.

Nothing in this count grows with the length $T$ of the sequence, because
the attention keeps no $T \times T$ array. The fused attention kernel of
:numref:`sec_attention-at-scale` keeps a single number per head and token,
the logarithm of the softmax normalizer, and recomputes the attention
weights during the backward pass. Whether a framework calls such a kernel
depends on the device, the dtypes, and the arguments, and a framework can
also reach the same result by recomputing the weights in other ways. On a
GPU, in the configuration used here, neither framework stores the
weights, and the count below checks it: an attention that stored its
weights would keep at least another $n_h T$ values per token for $n_h$
heads, 16,384 at $T = 1024$. That is $8d$, 40% of the rest of the block,
too large to miss.

The output layer adds a term of its own. The loss needs the
log-probability of every token of the vocabulary at every position, so it
keeps $V$ values per token, in fp32 for numerical safety
(:numref:`sec_numerics`): 196 KiB, more than a whole block. The largest
single term of the tally is not in the network at all; it is in the loss.

We check the tally by counting the bytes that the backward pass keeps. The
function `saved_bytes` runs the forward pass of a function and adds up the
sizes of the arrays it stores for the backward pass. The function
`gpt_loss` is the cross-entropy of a GPT's predictions. It uses the input
tokens as targets, which avoids loading a dataset and leaves the memory
unchanged.

:begin_tab:`pytorch`
`torch.autograd.graph.saved_tensors_hooks` calls its first function on
every tensor that autograd saves for the backward pass. Ours records the
size of the memory block, the *storage*, behind it, once per storage,
since several saved tensors can be views of one.
:end_tab:

:begin_tab:`jax`
`jax.vjp(f, *args)` runs the forward pass of `f` and returns the backward
pass as a function, `f_vjp`. That function is a pytree whose leaves are
the arrays the backward pass needs, so their sizes add up to what
the forward pass stores.
:end_tab:

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

Two subtractions isolate the terms. The count for a minibatch of two
sequences minus the count for one sequence gives the bytes of one sequence:
the weights, which the backward pass also needs, cancel. The bytes per
sequence of a GPT with two blocks minus those of a GPT with one block give
the bytes of one block. What remains of the one-block model is the
embedding, the output layer, and the loss.

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

The tally is the least that an implementation stores when it recomputes
nothing. An implementation reaches it when each normalization and each
activation function is a single fused kernel, which keeps what the rules
ask for and nothing else. An implementation that writes them as several
operations keeps intermediate results of each operation and stores more.
This is the memory side of the fusion of :numref:`subsec_comp-fusion`,
where the same intermediate arrays cost time.

:begin_tab:`pytorch`
PyTorch is the fused case. `scaled_dot_product_attention` runs a fused
kernel, and RMSNorm and SiLU are single kernels, so a block keeps 18.7
values of width $d$ per token: the tally without the second copy, since
the projection's input shares its memory with the attention's output. The
embedding, the output layer, and the loss keep 212 KiB per token.
:end_tab:

:begin_tab:`jax`
A block keeps 29.0 values of width $d$ per token, half again the tally.
The attention of `d2l.GPT` runs `jax.nn.dot_product_attention` under
`jax.checkpoint` (:numref:`sec_gpt`), so its backward pass keeps only the
queries, keys, and values and recomputes the weights, and nothing it keeps
grows with the length of the sequence. The surplus comes from the layers
that Flax writes as several operations. Each `nnx.RMSNorm` keeps four
vectors of width $d$ where the tally counts two, among them the squared
input and a copy of the input. `jax.nn.silu`, computed as
`x * jax.nn.sigmoid(x)`, keeps three arrays of width $\tfrac{8}{3}d$ where
the tally counts one. And the gate and the up-projection each keep their
own copy of the normalized input, one $d$ more. The embedding, the output
layer, and the loss keep 228 KiB per token.
:end_tab:

### The Memory Budget

The two counts give the memory of a training step with $n$ tokens. With $a$
bytes per token for each of the $L$ blocks and $a_\textrm{out}$ bytes per
token for the embedding, the output layer, and the loss,

$$
M(n) \approx 16N + n \, (L a + a_\textrm{out}),
$$
:eqlabel:`eq_mp-budget`

and a step fits on a GPU with $M_\textrm{GPU}$ bytes if $M(n) \leq
M_\textrm{GPU}$. The first term is fixed by the model and the optimizer;
the second grows with the minibatch. The largest minibatch is therefore
$n_{\max} = (M_\textrm{GPU} - 16N) / (L a + a_\textrm{out})$.

The equation charges the gradients, $4N$ bytes, for the whole step. That
is right under gradient accumulation (:numref:`subsec_mp-accumulation`),
where the gradients of one piece of the minibatch stay in memory while the
next piece runs. A single step creates its gradients while the
activations are being released and peaks lower, at $12N$ bytes plus the activations
(:numref:`fig_memory_anatomy`; Exercise :exref:`ex_mp-peak`). A model of
this size trains with accumulation, so we budget for it.

:begin_tab:`pytorch`
`zero_grad(set_to_none=True)` releases the gradients after each update, so
a step without accumulation reaches $12N$ plus the activations.
:end_tab:

:begin_tab:`jax`
`jax.grad` returns the gradients as outputs of the compiled program, and
XLA allocates the outputs of a program for the whole time that it runs, so
a compiled gradient computation holds them from its first operation on.
:end_tab:

For our GPT, $16N = 15.1$ GiB and the card has 24 GiB, which leaves 8.9 GiB
for the activations. A cell in :numref:`subsec_mp-checkpointing` evaluates
:eqref:`eq_mp-budget` for every configuration of this section; for the
first one the arithmetic fits in a line.

:begin_tab:`pytorch`
With $a = 149$ KiB and $a_\textrm{out} = 212$ KiB from the count, each
token needs $La + a_\textrm{out} \approx 2.8$ MiB, which leaves room for
about 3,200 tokens, three sequences of 1,024 tokens.
:end_tab:

:begin_tab:`jax`
With $a = 232$ KiB and $a_\textrm{out} = 228$ KiB from the count, each
token needs $La + a_\textrm{out} \approx 4.3$ MiB, which leaves room for
about 2,100 tokens, two sequences of 1,024 tokens.
:end_tab:

These estimates are optimistic. The allocator rounds and fragments its
blocks, and libraries reserve workspace of their own. Worse, as the
measurement below shows, the backward pass creates large arrays that no
count of stored tensors can see.

### Measured Peak Memory
:label:`subsec_mp-measuring`

A count is a model of what the framework stores, and a model can leave
things out. We check :eqref:`eq_mp-budget` on a GPT that is small enough
to measure: the width, the vocabulary, and the block of our GPT, but only
two blocks. We measure the peak memory of the forward and backward pass
with 1,024 to 8,192 tokens per step, in sequences of 1,024 tokens as in
the count, and compare its growth per token with the count of the
previous cell. With few tokens, other parts of the step can
set the peak, so we read the growth from 2,048 tokens on.

:begin_tab:`pytorch`
`d2l.peak_memory` (:numref:`sec_use_gpu`) reads the highest count of
allocated bytes during a call, relative to the memory allocated before it,
which here holds the weights.
:end_tab:

:begin_tab:`jax`
XLA decides before a step runs where every array of the compiled program
lives (:numref:`sec_compilation`), and `memory_analysis()` reports this
plan. Compiling needs only shapes, so we compile against
`jax.ShapeDtypeStruct`s without allocating the model, and take the plan's
outputs and temporaries, the memory beyond the weights.
:end_tab:

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

Why should the peak hold more than what is stored? The count is what the
backward pass keeps. The peak also holds what the backward pass creates
while the kept tensors are still in memory, and the largest of these
arrays belongs, once again, to the loss. Its backward pass begins with
two arrays of $V$ fp32 values per token: the gradient with respect to the
log-probabilities, and from it the gradient with respect to the logits. For a moment both exist next to the stored log-probabilities,
and an implementation that writes them to memory needs $8V$ bytes, 393 KiB,
per token more than the count.

:begin_tab:`pytorch`
The measured slope exceeds the count by 392 to 394 KiB per token, the $8V$
bytes of the two gradients. The first of them is all zeros but one: to
mark which one of 50,257 log-probabilities the loss read, the backward
pass of `F.cross_entropy` writes 50,257 numbers per token. With 1,024
tokens the peak comes later in the step, when the gradients of the
weights exist, and it lies above the line through the other points. For our GPT the peak needs about 3.2 MiB per
token, which leaves room for about 2,800 tokens instead of 3,200.
:end_tab:

:begin_tab:`jax`
XLA's plan follows the compiled program, which need not store what the
count of `jax.vjp` reports. A fused backward kernel can recompute a cheap
elementwise result instead of reading it from memory, which saves bytes,
and the plan also holds the gradient of the logits and other temporaries
of the backward pass, which adds bytes. From 2,048 to 4,096 tokens the plan
grows faster than the count; from 4,096 to 8,192 tokens the two effects
happen to cancel, and it grows by the count itself. Since XLA plans anew
for each shape, the plan need not be linear in the tokens. The plan, not
the count, is what the GPU must hold, so it is worth reading for the shape
one actually trains with. For our GPT the count of 4.3 MiB per token and
2,100 tokens stands.
:end_tab:

:begin_tab:`pytorch`
The allocator's own record shows the course of :numref:`fig_memory_anatomy`
in a real step. `torch.cuda.memory._record_memory_history()` records every
allocation and release, and `torch.cuda.memory._snapshot()` returns the
record. We record one training step with Adam on 4,096 tokens, after a
first step has created the moment estimates:
:end_tab:

:begin_tab:`jax`
The plan reports one peak for the whole program; since the step runs as one
compiled program, Python cannot read the allocator in between.
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
Adding up the sizes of the allocations and subtracting those of the
releases, starting from the memory allocated before the step, gives the
allocated memory after every event:
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
The trace starts at the weights and moment estimates, $12N$, climbs
through the forward pass, and jumps at the logits. It peaks at the start
of the backward pass, with the two gradients of the loss, then falls as
the activations go and the weight gradients arrive. The optimizer step
briefly allocates temporaries of the size of the gradients.
:end_tab:

## Mixed Precision
:label:`subsec_mp-mixed`

Mixed-precision training, introduced in :numref:`sec_numerics`, keeps the
weights in fp32 and runs the matrix multiplications of the forward and
backward pass in a 16-bit format, with the loss and other reductions in
fp32. bf16 has the exponent range of fp32, so its gradients do not
underflow; fp16 has a narrower range and needs loss scaling
(:numref:`subsec_loss_scaling`). Numerically the recipe risks little. The
weights, their updates, and the loss stay in fp32. What is rounded to 8
significant bits are the matrix products, which accumulate in fp32, and
the activations passed between them, and training tolerates that
rounding well. We first ask what the recipe does to memory, then to time.

If every stored value moved to bf16, a block would keep half of its fp32
bytes. It cannot quite: the residual stream, which carries the running sum
from block to block, stays in fp32, and the input of each normalization is
that stream. Of the 20 values of the tally, 2 then keep their 4 bytes and
18 take 2 bytes, so a block should store

$$
\frac{2 \cdot 4 + 18 \cdot 2}{20 \cdot 4} = 0.55
$$

of its fp32 bytes. The count of the previous subsection, repeated with bf16
arithmetic, tests the prediction:

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

A block stores slightly more than predicted, because an implementation
keeps a few more values in fp32, or a second bf16 copy of one. The loss
keeps its log-probabilities in fp32, so $a_\textrm{out}$ does not shrink.
Where the output layer runs in bf16, its bf16 logits are stored as well,
and $a_\textrm{out}$ grows.

The fixed term moves the other way. The 16-bit copies of the weights, $2N$
bytes, stay in memory until the backward pass has used them, so for our
GPT the fixed term grows from $16N = 15.1$ GiB to $18N = 17.0$ GiB, and the
room for activations shrinks from 8.9 to 7.0 GiB. The smaller activations
and the added weight copies largely offset each other. Unfortunately for
anyone who hoped to halve the memory, mixed precision raises the largest
minibatch by about a quarter at best.

:begin_tab:`pytorch`
Under `torch.autocast` the normalizations run in fp32 and pass their
output to the products cast to bf16; the gate and the up-projection of the
feed-forward network each cast it again and keep their own copy. A block
stores 0.58 of its fp32 bytes. The output layer runs in bf16, and
$a_\textrm{out}$ grows from 212 to 306 KiB. By the bf16 count plus the
excess measured in fp32, a token of our GPT needs about 2.2 MiB at the
peak instead of 3.2 MiB, and the largest minibatch grows by 15%, to about
3,300 tokens.
:end_tab:

:begin_tab:`jax`
`to_bf16`, which `bytes_per_token` applies, sets the compute dtype `dtype`
of the linear layers to bf16 (:numref:`sec_numerics`) and leaves their
`param_dtype`, the embedding, and the normalizations in fp32. The backward
pass keeps the fp32 outputs of the normalizations as well, and a block
stores 0.60 of its fp32 bytes. The output layer, which reuses the
embedding table, computes in fp32, so $a_\textrm{out}$ stays at 228 KiB. A
token of our GPT needs 2.7 instead of 4.3 MiB, and the largest minibatch
grows by about a quarter, to some 2,700 tokens.
:end_tab:

Memory is the smaller of the two gains. The larger one is time: the tensor
cores of the RTX 4090 multiply bf16 matrices at 165 TFLOP/s, twice their
rate for tf32 (:numref:`sec_hardware`). Whether a plain fp32 matrix
multiplication runs on the tensor cores as tf32 is a framework default
(:numref:`sec_numerics`), so a speedup quoted against true fp32 mixes two
effects. To separate them, we time one SGD step in three configurations:
true fp32, tf32, and bf16. A GPT would make a poor stopwatch here, since
attention, normalizations, and the loss blur the picture. We take instead a
three-layer MLP of width 8192 on a minibatch of 2,048 examples, whose
FLOPs are easy to count and whose large products lie far above the ridge
point, so that the step is compute-bound and the roofline floor is a fair
prediction.

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

The roofline gives each configuration a floor. At the tf32 peak of
83 TFLOP/s the $1.1 \times 10^{12}$ FLOPs of the step take at least
13.3 ms, and at the bf16 peak of 165 TFLOP/s at least 6.7 ms. True fp32 runs
on the ordinary arithmetic units, whose peak on this card is also
83 TFLOP/s (:numref:`sec_hardware`), so its floor is the same 13.3 ms. We
should still expect it to come in slower. On the ordinary units every
multiply-add is an instruction of its own, and loading its operands and
computing their addresses take instructions too, which compete for the
same issue slots; a tensor core does a whole tile per instruction. The
SGD update adds the same in every configuration: it reads each weight and
its gradient and writes the weight, 12 bytes for each of the 0.1 billion
weights, about 1.3 ms at 0.9 TB/s.

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

The step gets faster in two stages. True fp32 reaches about three fifths
of its peak, tf32 about nine tenths of the same number. The two formats
share a peak on this card; the tensor cores simply come much closer to
theirs. tf32 shortens the step by about a third without changing a single
stored byte, and its time is within a few percent of the floor plus the
update.

bf16 then runs 1.5 to 1.75 times as fast as tf32. Why not twice? The floor
plus the update is $6.7 + 1.3 = 8.0$ ms, and the step takes 8.4 to 9.6 ms.
The update moves bytes, not FLOPs, so it takes as long as before: about a
tenth of the tf32 step, it is a seventh of the bf16 one. The casts of
weights and gradients between fp32 and bf16 move bytes as well. Whatever
does not shrink with the format takes a larger share of a shorter step.
The fair baseline for a speedup from mixed precision is tf32, not true
fp32.

:begin_tab:`pytorch`
For matrix multiplications PyTorch defaults to `'highest'`, true fp32;
`torch.set_float32_matmul_precision('high')` allows tf32. The bf16 step
needs no `GradScaler`, since bf16 gradients do not underflow; with fp16, it
would (:numref:`subsec_loss_scaling`).
:end_tab:

:begin_tab:`jax`
On GPUs with tensor cores, JAX's default precision for fp32 matrix
multiplications is tf32. `jax.default_matmul_precision('highest')`
requests true fp32 for the code traced inside it, and `jit` compiles a
separate program for it.
:end_tab:

## Activation Checkpointing
:label:`subsec_mp-checkpointing`

Mixed precision raised the largest minibatch of our GPT by a quarter at
best. To raise it severalfold, a step must store less. Activation
checkpointing (:numref:`subsec_activation_checkpointing`)
:cite:`Chen.Xu.Zhang.ea.2016` stores only the input of each segment of the
network and recomputes the rest of the segment during the backward pass,
at the cost of one more forward pass: at most a third more arithmetic,
since a backward pass costs about two forward passes.

A transformer comes in natural segments, its blocks. With one segment per
block, each block stores only its input, $d$ values or $4d$ bytes per token
in fp32, instead of its $a$ bytes of activations. During the backward pass
one block at a time is recomputed, so the activations of a single block
return while it is processed. The memory per token at the peak falls to at
most

$$
L \cdot 4d + a + a_\textrm{out} + \Delta
$$

bytes. The four terms are the stored inputs of all blocks, one recomputed
block, the output layer and the loss, and the excess $\Delta$ of the
measured peak over the count (:numref:`subsec_mp-measuring`). The sum is
an upper bound, because the gradient of the logits is released before the
first block is recomputed.
For our GPT in fp32 the first term is $18 \times 8$ KiB $= 144$ KiB. The
next cell evaluates :eqref:`eq_mp-budget` for every configuration of this
section:

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

The output layer and the loss now store more per token than all 18 blocks
together. Surprisingly, under checkpointing bf16 even lowers the room for
tokens. It can save only on the single recomputed block, and its $2N$
bytes of weight copies outweigh that.

:begin_tab:`pytorch`
Checkpointing lowers the memory per token at the peak from 3.2 to 0.9 MiB
and raises the largest minibatch to about 10,000 tokens, more than three
times as many as without it. In bf16 the bytes per token even rise, from
0.88 to 0.91 MiB, because the stored bf16 logits add more to
$a_\textrm{out}$ than the recomputed block saves.
:end_tab:

:begin_tab:`jax`
Checkpointing lowers the memory per token from 4.3 to 0.6 MiB and leaves
room for about 15,000 tokens, seven times as many as without it. The number
extrapolates the plan of the two-block model, which need not grow linearly.
:end_tab:

The loss is therefore the next term to attack. A loss computed in chunks
of tokens runs the output layer and the cross-entropy on one chunk at a
time, inside a checkpoint of its own, so that the logits, the
log-probabilities, and their gradients exist for one chunk only. The
backward pass recomputes the logits of each chunk, which costs one more
pass through the output layer: $2dV$ FLOPs per token, about 3% of the
$6N$ FLOPs per token of a training step (two per parameter in the forward
pass and four in the backward pass, :numref:`sec_scaling-laws`). Exercise
:exref:`ex_mp-chunked-loss` builds it.

We measure checkpointing on a stack of four blocks of our GPT, with two
sequences of 1,024 tokens. With checkpointing, the stack should store five
vectors of width $d$ per token, 40 KiB: the inputs of the four blocks and
the output that the loss reads. The cell reports the bytes stored per
token, the peak memory beyond the weights and gradients, and the time of a
forward and backward pass.

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

Checkpointing reduces what the four blocks store to the predicted 40 KiB
per token. The peak falls by less, because the backward pass recomputes one
block at a time and holds its activations while it does. The step takes
about a quarter longer, within the bound of a third for one more forward
pass. In a compiled program the compiler decides when the recomputation
runs, so checkpointing guarantees what is stored between the two passes,
not where the peak lies. For a fixed minibatch, checkpointing always adds
time; it is the right choice only when memory, not speed, limits the
minibatch.

:begin_tab:`pytorch`
`checkpoint(blk, H, use_reentrant=False)` runs `blk` without storing its
intermediate results and keeps what it needs to rerun it, including the
state of the random number generator, so that dropout draws the same mask
twice. The formula predicts the peak: 2,048 tokens times $40 + 149$ KiB is
about 0.37 GiB with checkpointing, and 2,048 times 606 KiB is about
1.18 GiB without, against 0.42 and 1.25 GiB measured. The peak falls by a
factor of about three.
:end_tab:

:begin_tab:`jax`
`jax.checkpoint` makes the derivative of a function keep only the
function's inputs; we apply it to each block written as a function of its
weights, with its structure `graphdef` as a static argument. Without
checkpointing the count predicts 2,048 times 936 KiB, about 1.83 GiB, but
XLA plans 0.78 GiB, because its fused kernels recompute cheap elementwise
results instead of storing them (:numref:`subsec_mp-measuring`).
Checkpointing lowers the plan by a factor of about 1.7.
:end_tab:

Per-block checkpointing recomputes everything in a block, including the
matrix multiplications that take most of its time, while much of its memory
sits in results that are cheap to compute. *Selective recomputation*
stores the expensive results and recomputes the cheap ones.
:citet:`Korthikanti.Casper.Lym.ea.2022` recompute only the scores
$\mathbf{Q}\mathbf{K}^\top$, their softmax, and its product with the
values, which store $T \times T$ values per head for little arithmetic;
the fused attention kernel does the same inside one kernel. Our block
already has that saving, and its remaining candidates are the activated
gate and the product of the SwiGLU network, two vectors of width
$\tfrac{8}{3}d$, about $5d$ of the tally's $20d$. Exercise
:exref:`ex_mp-selective` writes a policy that keeps the outputs of the
matrix multiplications and recomputes the elementwise operations.

## Gradient Accumulation
:label:`subsec_mp-accumulation`

With checkpointing, our GPT fits ten thousand tokens or more per step,
still only two or three percent of the minibatches it should train with.
Gradient accumulation closes the rest of the gap without storing more
activations. It splits the minibatch that the optimizer should see, the
*global batch*, into $m$ *micro-batches*, runs the forward and backward
pass on each while adding up their gradients, and takes one optimizer step
at the end: $B_{\textrm{global}} = m \, B_{\textrm{micro}}$. If the loss
averages over the examples, as the cross-entropy does, each micro-batch
loss is divided by $m$, so that the sum of the gradients becomes the
average over the global batch.

We run it on the two-block GPT of :numref:`subsec_mp-measuring`, with
4,096 tokens as a single batch and as four micro-batches of 1,024. An
`assert` checks that the two gradients agree; without the division by $m$
they would differ by a factor of four.

What should we expect? The peak should fall to about half, not to a
quarter. A micro-batch stores the activations of a quarter of the tokens,
less than 1 GiB, but the gradients of all weights, $4N = 0.77$ GiB for the
0.21 billion parameters of this model, now stay in memory from the first
micro-batch to the last. They are the $4N$ that :eqref:`eq_mp-budget`
charges for the whole step.

The arithmetic is the same in both cases, but accumulation adds memory
traffic. Each micro-batch after the first reads the gradient so far and
its own and writes their sum, $12N$ bytes, 2.5 GB, about 2.5 ms at the
1.0 TB/s of :numref:`tab_gpu_specs`. A micro-batch of $n$ tokens performs
$6Nn$ FLOPs, so at the peak rate $P$ and bandwidth $\beta$ the additions
take a fraction

$$
\frac{12N / \beta}{6Nn / P} = \frac{2 I^*}{n}
$$

of its arithmetic time, where $I^* = P / \beta$ is the ridge point of
:numref:`sec_perf_model`. In tf32, with $I^* = 82$ FLOP/byte, that is 16%
for $n = 1{,}024$. Three of the four micro-batches incur it, about 7 ms in
all, so the step should take about a tenth longer.

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
    print(f'{m} x {4096 // m:4d} tokens: peak {peak / 2**30:.2f} GiB above '
          f'the weights, {np.median(times):.0f} ms per step '
          f'({times.min():.0f} to {times.max():.0f})')
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
    print(f'{m} x {4096 // m:4d} tokens: peak {peak / 2**30:.2f} GiB above '
          f'the weights, {np.median(times):.0f} ms per step '
          f'({times.min():.0f} to {times.max():.0f})')
```

The peak falls to about half, and the gradients that persist across the
micro-batches are a large part of what remains. The time grows by a tenth
to a fifth, far more than the spread of the five timings: accumulation over
micro-batches of 1,024 tokens is not free. The fraction $2I^*/n$ also says
how to make it cheap. At micro-batches of 4,096 tokens it is 4%.

:begin_tab:`pytorch`
The two-block GPT runs in tf32 here, because the timing cell of
:numref:`subsec_mp-mixed` left `'high'` set. The step takes 17 ms longer,
more than twice the 7 ms of accumulator traffic that we predicted. A
profile of the two steps (:numref:`sec_perf_model`), which this cell does
not run, puts the rest in the additions into `.grad`, which run below the
full bandwidth, and in idle time between the kernels of the smaller
micro-batches.
:end_tab:

:begin_tab:`jax`
The step takes 9 ms longer, close to the 7 ms of accumulator traffic that
we predicted at the full specified bandwidth, which a copy never quite
reaches.
:end_tab:

The two gradients agree, but the equivalence rests on conditions that this
test meets without effort. The loss must add up over examples, and the
micro-batches must have equal sizes, or their losses must be weighted by
their sizes. Layers that couple the examples of a batch, batch
normalization above all, see only the micro-batch. And everything that
acts on the whole gradient, such as clipping and the optimizer step, must
wait for the last micro-batch.

A micro-batch too small to keep the tensor cores busy costs more again,
since each micro-batch reads all the weights in the forward and in the
backward pass (:numref:`sec_perf_model`). Both costs shrink as the
micro-batch grows, so the best micro-batch is the largest one that fits.
Accumulation never makes a step faster: it changes when the optimizer
steps, not how much work each token needs.

## Choosing a Memory Technique
:label:`subsec_mp-ladder`

Each technique changes one term of :eqref:`eq_mp-budget` at a cost, and
:numref:`tab_mp-techniques` lists them by the symptom that calls for them,
together with the one of :numref:`sec_sharding` that acts on the fixed
term. Mixed precision heads the table for its speed; its effect on memory
is small.

:Memory techniques by symptom, with their effect on :eqref:`eq_mp-budget` and their cost.
:label:`tab_mp-techniques`

| symptom | technique | effect on memory | cost |
|---|---|---|---|
| the matrix products set the time of a step | bf16 mixed precision | small: activations to about 0.6, and $+2N$ for weight copies | rounding to 8 significant bits; fp16 also needs loss scaling |
| one sequence barely fits, or not at all | activation checkpointing | stored activations per block from $a$ to $4d$ bytes per token | one more forward pass, about a quarter more time |
| the output layer and the loss dominate | a loss computed in chunks of tokens | $a_\textrm{out}$ and $\Delta$ for one chunk at a time | one more pass through the output layer |
| the minibatch is too small for training | gradient accumulation | none; the batch per update grows $m$-fold | accumulator traffic, $2I^*/n$ of the arithmetic; micro-batches too small to keep the tensor cores busy |
| $16N$ alone fills the card | sharding the weights, gradients, and optimizer state over GPUs | $16N/k$ on each of $k$ GPUs | communication |

For our GPT on one RTX 4090 the techniques combine into one recipe, and
the arithmetic of this section sets its numbers. bf16 makes the products
faster. Two sequences of 1,024 tokens fit in bf16 without checkpointing,
with some room; a third would leave little or none. So a micro-batch
holds two sequences, and the step accumulates 256 of them, a global batch
of 524,288 tokens. Checkpointing would allow micro-batches of four
sequences and halve the accumulator traffic, from $2I^*/n \approx 16\%$ to
8% with the bf16 ridge point of 164, but its recomputation costs about a
quarter, more than it saves. It helps only once a micro-batch of useful
size no longer fits.

Now we can ask what a step costs. At $6N$ FLOPs per token, a global batch
takes $6 \times 1.01 \times 10^9 \times 524{,}288 \approx 3.2 \times
10^{15}$ FLOPs. At the bf16 peak of 165 TFLOP/s, which no real step
reaches, that is 19 s, and with the accumulator traffic at least 22 s per
optimizer step. A run of 100,000 steps reads 52 billion tokens, about 50
per parameter: more than the compute-optimal 20 of Chinchilla and far
fewer than small models meant for deployment read
(:numref:`sec_scaling-laws`). On one card it takes almost a month even at
the peak, and more than a month at the rates measured above.

Data parallelism, the subject of the next section, shares the work of each
step among several GPUs. It also places a full copy of the 17 GiB of state
on every one of them: the weights, gradients, moment estimates, and bf16
weight copies that none of the techniques of this section touches.
:numref:`sec_sharding` divides that state among the GPUs instead. One card
can now hold a step of our GPT. It cannot finish the run.

## Summary

The memory of a training step has a part that grows with the parameters,
$16N$ bytes with Adam in fp32, and a part that grows with the tokens of the
minibatch, the activations that the forward pass keeps for the backward
pass. Three rules give the count of the second part: a linear layer keeps
its input, an elementwise function its input or its output, and a product
both factors. By this count a transformer block whose attention recomputes
its weights keeps about $20d$ values per token, and the loss $V$ more;
implementations that do not fuse their normalizations and activations
keep more. The peak also holds what the
backward pass creates while the stored tensors still exist, above all the
gradient of the logits, and on a 24 GiB card a billion-parameter GPT has
room for two or three sequences of 1,024 tokens. Mixed precision speeds up
the matrix products but changes memory little, because it adds 16-bit
copies of the weights. Activation checkpointing stores only the input of
each block, which fits several times as many tokens for about a quarter
more time, and gradient accumulation builds a global batch of half a
million tokens from micro-batches that fit, at an extra cost of $2I^*/n$
for micro-batches of $n$ tokens.

## Exercises

1. **Where the peak lands.** :numref:`fig_memory_anatomy` places the peak
   of a training step at the start of the backward pass. Consider a stack of
   $n$ identical layers. The parameters of each layer occupy $w$ bytes, and
   its forward pass stores $a$ bytes of activations. Assume that no
   gradients exist when the backward pass starts, and that the backward pass
   of a layer frees the activations of that layer and creates its parameter
   gradient of $w$ bytes. Leave out the parameters and the optimizer
   states, whose sizes stay constant.
    1. Write the bytes held by activations and gradients after the backward
       pass has processed $j$ of the $n$ layers. Show that the peak lies at
       the start of the backward pass when $a > w$ and at its end when
       $a < w$.
    1. A block of our GPT holds about $12d^2$ parameters and stores about
       $20d$ fp32 values per token. For how many tokens per step does the
       peak lie at the start of the backward pass? Is that a realistic
       number of tokens for training?
    1. Under gradient accumulation over $m$ micro-batches, the gradients of
       the first micro-batch stay allocated while the later micro-batches
       run. Find the peak with accumulation, and the values of $m$ for which
       accumulation lowers the peak below that of a single step on the full
       batch.
   :exlabel:`ex_mp-peak`

:begin_tab:`pytorch`
2. [code] **Optimizer-step memory.** The trace of
   :numref:`subsec_mp-measuring` shows that the optimizer step allocates
   temporaries of its own. Build the two-block GPT of that subsection three
   times, each with a different `torch.optim.Adam`: with
   `foreach=False`, with the default arguments, and with `fused=True`.
    1. Run two training steps with each optimizer. On the second step,
       record `torch.cuda.memory_allocated()` just before `opt.step()`,
       reset the peak statistics, and record
       `torch.cuda.max_memory_allocated()` after `opt.step()`. Express the
       extra bytes of each update in units of the gradient size $4N$.
    1. The three implementations update one parameter tensor at a time,
       apply each elementwise operation to all parameter tensors at once,
       or run the whole update in one kernel per group of tensors. Explain
       the three measurements from this difference. When does the optimizer
       step, rather than the start of the backward pass, set the peak of a
       training step?
:end_tab:

:begin_tab:`jax`
2. [code] **Optimizer-step memory.** A JAX training step returns new
   parameters and optimizer states instead of updating them in place.
   Extend the compiled gradient of :numref:`subsec_mp-measuring` into a
   full Adam step with `optax.adam`, a function of the parameters, the
   optimizer state, and `X` that returns the updated parameters and the
   updated optimizer state.
    1. Compile the step for 512 tokens and report `argument_size_in_bytes`,
       `output_size_in_bytes`, and `temp_size_in_bytes` of its
       `memory_analysis()` in units of the parameter size $4N$. Estimate
       the peak memory of one step from these numbers.
    1. Compile the step again with `donate_argnums` covering the
       parameters and the optimizer state, and report
       `alias_size_in_bytes` as well. How does the estimated peak change,
       and how does it compare with $16N$?
:end_tab:

3. **Clipping under accumulation.** Gradient clipping to norm $\tau$
   replaces a gradient $\mathbf{g}$ by
   $\mathbf{g} \min(1, \tau / \|\mathbf{g}\|)$. Let
   $\mathbf{g}_1, \ldots, \mathbf{g}_m$ be the micro-batch gradients
   after the scaling by $1/m$ of :numref:`subsec_mp-accumulation`, so that
   their sum $\mathbf{g}$ is the gradient of the global batch.
    1. A training loop clips each $\mathbf{g}_i$ before adding it to the
       accumulator. Find the largest norm that the accumulated gradient can
       reach, and a condition under which the loop gives the same result as
       clipping $\mathbf{g}$ once before the optimizer step.
    1. Under fp16 loss scaling with factor $S$, the accumulator holds
       $S \mathbf{g}$ once the last micro-batch has been added. In which
       order must the loop divide by $S$, clip, and step? What goes wrong
       if it clips before dividing, and what goes wrong if $S$ changes
       before the last micro-batch has been added?

    *Adapted from the PyTorch documentation,
    [Automatic Mixed Precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
    sections "Gradient clipping" and "Gradient accumulation".*

:begin_tab:`pytorch`
4. [code] **fp16 gradient window.** ● The 5-bit exponent of fp16 gives it
   normal numbers from $2^{-14}$ to 65,504 and subnormal numbers down to
   $2^{-24}$. Loss scaling :cite:`Micikevicius.Narang.Alben.ea.2018`
   multiplies the loss by a factor $S$ before the backward pass and divides
   the gradients by $S$ afterwards. Use `mlp`, `X_mlp`, and `y_mlp` as
   defined in :numref:`subsec_mp-mixed`, and take as reference the fp32
   parameter gradients computed under
   `torch.set_float32_matmul_precision('highest')`.
    1. Extend the measurement of :numref:`subsec_loss_scaling` to all
       parameter gradients of `mlp`: plot a histogram of $\log_2 |g|$ over
       the nonzero entries $g$ of the reference gradients, and find the
       factors $S$ that place 99.9% of the scaled entries $S g$ in the
       normal range of fp16.
    1. For $S = 2^0, 2^2, \ldots, 2^{24}$, compute the gradients of $S$
       times the loss under `torch.autocast('cuda', dtype=torch.float16)`,
       divide them by $S$, and plot their relative error
       $\|\hat{\mathbf{g}} - \mathbf{g}\| / \|\mathbf{g}\|$ with respect
       to the reference $\mathbf{g}$, marking the runs with non-finite
       entries. Repeat the sweep with `torch.bfloat16`. In which range of
       $S$ is fp16 more accurate than bf16, and why is that possible?
    1. `torch.amp.GradScaler` starts at $S = 2^{16}$. Where does this
       factor lie in the window of the previous sub-problem? The gradient
       of the loss with respect to a logit is the difference between the
       predicted and the target probability, divided by the batch size,
       here 2048. How does the window move if the batch grows to 32,768
       examples? Explain why a static loss scale that works for one
       workload can fail for another.
:end_tab:

:begin_tab:`jax`
4. [code] **fp16 gradient window.** ● The 5-bit exponent of fp16 gives it
   normal numbers from $2^{-14}$ to 65,504 and subnormal numbers down to
   $2^{-24}$. Loss scaling :cite:`Micikevicius.Narang.Alben.ea.2018`
   multiplies the loss by a factor $S$ before the backward pass and divides
   the gradients by $S$ afterwards. Use `mlp_params`, `X_mlp`, `y_mlp`, and
   `loss_fn` as defined in :numref:`subsec_mp-mixed`, and take as reference
   the fp32
   gradients computed under `jax.default_matmul_precision('highest')`.
    1. Extend the measurement of :numref:`subsec_loss_scaling` to all
       parameter gradients of the MLP: plot a histogram of $\log_2 |g|$
       over the nonzero entries $g$ of the reference gradients, and find
       the factors $S$ that place 99.9% of the scaled entries $S g$ in the
       normal range of fp16.
    1. For $S = 2^0, 2^2, \ldots, 2^{24}$, compute the gradients of $S$
       times the loss with `jnp.float16` as the compute dtype, divide them
       by $S$, and plot their relative error
       $\|\hat{\mathbf{g}} - \mathbf{g}\| / \|\mathbf{g}\|$ with respect
       to the reference $\mathbf{g}$, marking the runs with non-finite
       entries. Repeat the sweep with `jnp.bfloat16`. In which range of $S$
       is fp16 more accurate than bf16, and why is that possible?
    1. At the time of writing, optax ships no automatic loss scaler.
       Choose a static factor $S$ from the window of the previous
       sub-problem, and justify the choice.
       The gradient of the loss with respect to a logit is the difference
       between the predicted and the target probability, divided by the
       batch size of 2048. How does the window move if the batch grows to
       32,768 examples? Explain why a static loss scale that works for one
       workload can fail for another.
:end_tab:

5. **Checkpoint placement.** ● Consider a stack of $n$ identical residual
   blocks, and measure memory in units of the input of one block. The
   backward pass of one block needs $c$ units of stored tensors, its input
   included. A schedule with segments of $s$ consecutive blocks stores only
   the input of each segment during the forward pass. During the backward
   pass it recomputes one segment at a time, while the stored inputs of the
   earlier segments remain in memory.
    1. Derive the peak activation memory as a function of $s$, and the $s$
       that minimizes it. For $c = 1$ the answer is the $s = \sqrt{n}$
       of :numref:`subsec_activation_checkpointing`. Show that for $c > n$
       the minimizing $s$ is below one, so that per-block checkpointing is
       the best schedule. A block of our GPT has $c \approx 20$, and the
       model has $n = 18$ blocks. Compare storing everything with per-block
       checkpointing, in memory and in the number of block forward
       evaluations per step.
    1. :citet:`Chen.Xu.Zhang.ea.2016` reduce the memory further by
       recursion. The schedule stores the input of the stack and the input
       of its second half, processes the second half recursively, and then
       processes the first half recursively from its stored input. For $n$
       a power of two, derive the peak memory and the total number of block
       forward evaluations, the original forward pass included. For
       $n = 16$ and $n = 1024$ with $c = 1$, compare both quantities with
       those of the best segment length.

:begin_tab:`pytorch`
6. [code] **Selective recomputation.** ● Per-block checkpointing stores
   only the input of each block. The function
   `torch.utils.checkpoint.create_selective_checkpoint_contexts`, passed to
   `checkpoint` through its `context_fn` argument, instead caches the
   outputs of chosen operations during the forward pass and recomputes the
   remaining operations. Its policy function receives each operation
   together with its arguments. The linear layers of `d2l.GPT` have no
   biases, and each dispatches to `torch.ops.aten.mm.default`.
    1. For one block of our GPT, list the tensors stored
       between the forward and the backward pass, and the matrix
       multiplications that the backward pass recomputes, under three
       policies: per-block checkpointing, caching the outputs of every
       `mm`, and caching all outputs except those of the feed-forward
       network's gate and up-projection.
    1. Implement the last two policies for the stack of four blocks of
       :numref:`subsec_mp-checkpointing`, and measure the stored bytes per
       token, the peak memory, and the step time as that subsection does.
       Which policy would you choose for our GPT, and why?
   :exlabel:`ex_mp-selective`
:end_tab:

:begin_tab:`jax`
6. [code] **Selective recomputation.** ● `jax.checkpoint` accepts a
   `policy` argument that names the intermediate values it may keep, and
   `jax.checkpoint_policies.dots_with_no_batch_dims_saveable` keeps the
   outputs of matrix multiplications with the weights and recomputes
   everything else.
    1. For one block of our GPT, list the values stored
       between the forward and the backward pass, and the matrix
       multiplications that the backward pass recomputes, under three
       configurations: no checkpointing, `jax.checkpoint` without a
       policy, and `jax.checkpoint` with
       `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`. Check
       your lists with `jax.ad_checkpoint.print_saved_residuals`.
    1. Add the third configuration to the measurement of
       :numref:`subsec_mp-checkpointing`, and compare the stored bytes per
       token, the planned memory, and the step time of all three. Which
       configuration would you choose for our GPT, and why?
   :exlabel:`ex_mp-selective`
:end_tab:

7. [code] **Cost of accumulation.** ● Split the global batch of 2048
   examples of the bf16 configuration of :numref:`subsec_mp-mixed` into $m$
   equal micro-batches, and accumulate their gradients, for
   $m = 1, 2, 4, \ldots, 64$.
    1. Let the network have $N$ parameters. A micro-batch of
       $B_{\textrm{micro}}$ examples performs about
       $6 N B_{\textrm{micro}}$ floating-point operations in its forward and
       backward passes. Assume an implementation that adds into the
       accumulator inside the backward pass, so that the gradient of a
       micro-batch is never stored on its own; it then moves at least
       $12N$ bytes: it reads the bf16 weights in each of the two passes,
       and it reads and writes the fp32 gradient accumulator. Ignore the
       activations. Take from
       :numref:`tab_gpu_specs` the peak arithmetic rate and the memory
       bandwidth of the RTX 4090, predict the time per global batch as a
       function of $m$, and find the $m$ beyond which the micro-batches are
       bandwidth-bound.
    1. Measure the time per global batch for every $m$, and plot it
       together with the prediction. Identify the weight and gradient
       traffic of the bf16 configuration that the count of $12N$ bytes
       leaves out.
    1. Suppose that activation checkpointing let each micro-batch hold four
       times as many examples, with a third more arithmetic.
       Under the model of the first sub-problem, for which $m \geq 4$ of the
       sweep does accumulation over $m/4$ checkpointed micro-batches take
       less time than accumulation over $m$ plain ones?

8. [extended] **The loss in chunks.** ●● Under per-block checkpointing,
   the output layer and the loss of our GPT store more per
   token than all 18 blocks together, because the loss keeps $V = 50{,}257$
   log-probabilities per token and the backward pass creates their
   gradient.
    1. Write the output layer and the cross-entropy so that they process the
       tokens of a step in chunks of $n_c$ tokens, and wrap each chunk in a
       checkpoint, so that the log-probabilities of only one chunk exist at
       a time. Check that the loss and the gradients match the unchunked
       version up to rounding.
    1. Using the counts of this section, predict the largest micro-batch of
       our GPT on a 24 GiB card, in fp32 with per-block
       checkpointing, with and without the chunked loss, for $n_c = 1024$.
       Which term of :eqref:`eq_mp-budget` limits the micro-batch now?
    1. On the two-block GPT of :numref:`subsec_mp-measuring`, measure the
       growth of the peak memory per token with the chunked loss, as that
       subsection does, and compare it with your prediction. How much time
       does the chunked loss add per step?
   :exlabel:`ex_mp-chunked-loss`

[Discussions](https://d2l.discourse.group/t/36790)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §13.4]{.kicker}

Memory and Precision<br>
**memory per parameter and per token · the memory budget · mixed precision · checkpointing · accumulation**
:::
:::

::: {.slide title="A 1B-Parameter GPT Fills 15 of 24 GiB before It Reads a Token"}
[The problem]{.kicker}

GPT with $N = 1.01 \times 10^9$ parameters, Adam in fp32, one RTX 4090
(24 GiB).

- Weight, gradient, two Adam moments: $16N$ bytes = **15.1 GiB**.
- The forward pass stores **several MiB per token** for the backward pass.
- The remaining 9 GiB hold **two or three sequences** of 1,024 tokens.
- GPT-3 trained on **0.5M to 3.2M tokens** per step.

::: {.d2l-note .warn}
A step that does not fit does not run. A step that barely fits learns from
too few tokens.
:::

[Brown et al. (2020), Table 2.1.]{.small}
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- count the bytes per parameter for an optimizer and a precision;
- count the bytes per token of a transformer block from three rules;
- predict the largest micro-batch that fits, and read a measured peak
  against the prediction;
- predict what bf16, checkpointing, and accumulation change in memory and
  in time, and the time of a step of the whole recipe.

**The idea:** memory = (fixed per parameter) + (tokens × bytes per token).
**Our GPT:** $d = 2048$, 16 heads, 18 blocks,
$V = 50{,}257$, on one RTX 4090.
:::

::: {.slide title="Memory Decides Whether the Step Runs at All"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| The Performance Model | roofline, three regimes, timing, profiling |
| Hardware | where peak arithmetic and bandwidth come from |
| Computation Graphs and Compilation | fusion and kernel replay |
| **Memory and Precision** | **what fits: bytes per parameter and per token** |
| Multi-GPU from First Principles | data parallelism by hand, its cost model |
| Data Parallelism in Practice | one process per GPU, overlap |
| Sharding the Training State | dividing the $16N$ among GPUs |
| Case Study: Optimizing a Transformer | the whole method on one GPT |
:::

::: {.col .narrow}
- **Uses:** 16 bytes per parameter, checkpointing (Computation chapter);
  the formats bf16 and tf32; the GPT; fused attention; the roofline.
- **Skips:** several GPUs (next sections); fp8 training and quantization
  (Language Models part).
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Memory of a Training Step]{.dtitle}

[per parameter · per token · the budget · measurement]{.dsub}
:::
:::

::: {.slide title="Every Parameter Takes 16 Bytes with Adam in fp32"}
[Memory per parameter]{.kicker}

| optimizer and precision | bytes per parameter |
|---|---|
| SGD: weight + gradient | 8 |
| SGD with momentum: + velocity | 12 |
| Adam: + two moment estimates | **16** |
| Adam, mixed precision: + bf16 weight copies | 18 |

@!memory-precision-memory-per-parameter

The count is fixed by model and optimizer, whatever the minibatch.
:::

::: {.slide title="Activations Peak at the Start of the Backward Pass"}
[Memory per token]{.kicker}

![](../img/mdl-perf-memory-anatomy.svg){width=78%}

The forward pass stores activations for the backward pass, which releases
them layer by layer while the gradients appear.
:::

::: {.slide title="Three Rules Say What a Layer Keeps for the Backward Pass"}
[Memory per token]{.kicker}

| layer | keeps | because |
|---|---|---|
| linear $\mathbf{y} = \mathbf{W}\mathbf{x}$ | its input $\mathbf{x}$ | weight gradient $= \partial_{\mathbf{y}} \, \mathbf{x}^\top$ |
| elementwise $f$ | its input or its output | $f'$ is written in one of them |
| product $\mathbf{p} \odot \mathbf{q}$ | both factors | the gradient for each is the other |

Example, SwiGLU: SiLU keeps the gate $\mathbf{g}$; the product keeps
$\textrm{SiLU}(\mathbf{g})$ and $\mathbf{u}$; the down-projection keeps the
product. Four vectors of width $\tfrac{8}{3}d$.
:::

::: {.slide title="A Block Stores about 20 Values of Width d per Token"}
[Memory per token]{.kicker}

| stored for the backward pass | values per token |
|---|---|
| inputs of the two normalizations | $2d$ |
| normalized outputs, kept by the projections that read them | $2d$ |
| queries, keys, values; attention output | $4d$ |
| attention output again, if merging the heads copies it | $0$ or $d$ |
| SwiGLU: gate, up, activated gate, product | $4 \cdot \tfrac{8}{3}d \approx 11d$ |
| **block** | $\approx 20d$ = **160 KiB** in fp32 |
| loss: log-probabilities of $V$ tokens, fp32 | $V$ = **196 KiB** |

Fused attention recomputes its $T \times T$ weights: nothing grows with $T$.
:::

::: {.slide title="Fused Kernels Store the Tally: 18.7d per Block" only="pytorch"}
[Memory per token]{.kicker}

Count the saved tensors for 1 and 2 sequences, for 1 and 2 blocks, and
subtract: the weights cancel, then the rest of the model.

@!memory-precision-memory-per-token-2

- RMSNorm, SiLU, and attention are single kernels: they keep what the
  rules ask for.
- 18.7d is the tally without the second copy of the attention output.

::: {.d2l-note}
Without recomputation, the tally is the least a block can store.
:::
:::

::: {.slide title="Unfused Layers Store Half Again the Tally: 29d per Block" only="jax"}
[Memory per token]{.kicker}

Count the saved arrays for 1 and 2 sequences, for 1 and 2 blocks, and
subtract: the weights cancel, then the rest of the model.

@!memory-precision-memory-per-token-2

- `nnx.RMSNorm` keeps 4 vectors of width $d$, not 2; `jax.nn.silu` keeps
  3 arrays of width $\tfrac{8}{3}d$, not 1.
- Gate and up-projection each keep their own copy of their input: $d$ more.

::: {.d2l-note}
Without recomputation, the tally is the least a block can store.
:::
:::

::: {.slide title="The Budget Leaves Room for Two or Three Sequences"}
[The memory budget]{.kicker}

$$M(n) \approx 16N + n\,(L a + a_\textrm{out}) \le M_\textrm{GPU}$$

$a$: bytes per token of a block; $L = 18$ blocks; $a_\textrm{out}$:
embedding, output layer, and loss.

- $16N$ = 15.1 GiB of $M_\textrm{GPU}$ = 24 GiB: **8.9 GiB** for activations.
- $La + a_\textrm{out}$: **about 3 to 4 MiB per token** by the count.
- $n_{\max} = 8.9\textrm{ GiB} / (La + a_\textrm{out})$: **two or three
  sequences** of 1,024 tokens.
- $16N$ charges the gradients for the whole step, as under accumulation;
  a single step peaks at $12N$ + activations.
:::

::: {.slide title="Check: Twice as Deep, Does It Still Train?"}
[Check]{.kicker}

The same GPT with 36 blocks instead of 18, on the same 24 GiB card.

. . .

- $N \approx 36 \times 50\textrm{M} + 0.1\textrm{B} \approx 1.9\textrm{B}$.
- $16N \approx 29$ GiB **> 24 GiB**: not one token fits.
- No activation technique helps: only dividing the $16N$ among GPUs does.
:::

::: {.slide title="The Measured Peak Adds 8V Bytes per Token to the Count" only="pytorch"}
[Measured peak memory]{.kicker}

Two-block GPT, peak of forward + backward, 1,024 to 8,192 tokens:

@!memory-precision-measuring-memory-1

The loss's backward pass creates two fp32 arrays of $V$ values per token,
the gradients for the log-probabilities and for the logits, while the
log-probabilities are still stored: $8V$ bytes = 393 KiB. The first one
marks the one target among 50,257 entries; the other 50,256 are zero.
:::

::: {.slide title="XLA's Plan Grows by the Count between 4,096 and 8,192 Tokens" only="jax"}
[Measured peak memory]{.kicker}

Two-block GPT, XLA's plan for forward + backward, compiled from shapes only:

@!memory-precision-measuring-memory-3

- Fused kernels recompute cheap elementwise results: fewer bytes.
- The plan holds the gradient of the $V$ logits: more bytes.
- From 4,096 to 8,192 tokens the two happen to cancel.

::: {.d2l-note}
The plan, not the count, is what the GPU must hold: read
`memory_analysis()` for the shape you train with.
:::
:::

::: {.slide title="The Measured Trace Peaks at the Start of the Backward Pass" only="pytorch"}
[Measured peak memory]{.kicker}

@!memory-precision-measuring-memory-2

Forward: activations, then the logits. Backward: the loss's two gradients,
then activations out, gradients in. Step: temporaries, then `zero_grad`.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Three Techniques]{.dtitle}

[mixed precision · activation checkpointing · gradient accumulation]{.dsub}
:::
:::

::: {.slide title="bf16 Stores a Little over Half the Activations and Adds 2N"}
[Mixed precision]{.kicker}

Prediction: the residual stream stays fp32, 2 of the 20 values:
$(2 \cdot 4 + 18 \cdot 2)/(20 \cdot 4) = 0.55$.

@!memory-precision-mixed-precision-1

- Fixed term: $16N \to 18N$, 15.1 to 17.0 GiB; room 8.9 to 7.0 GiB.
- The log-probabilities stay fp32: $a_\textrm{out}$ does not shrink.
- Largest micro-batch: **about a quarter larger at best**.
:::

::: {.slide title="bf16 Runs 1.5 to 1.75 Times as Fast as tf32, Not Twice"}
[Mixed precision]{.kicker}

One SGD step, 3-layer MLP of width 8192, 2,048 examples, $1.1 \times
10^{12}$ FLOPs. Floors: **13.3 ms** at 83 TFLOP/s (tf32), **6.7 ms** at
165 (bf16).

@!memory-precision-mixed-precision

- fp32 and tf32 share the 83 TFLOP/s peak: ordinary units reach about
  0.6 of it, the tensor cores about 0.9.
- bf16: floor + update = 6.7 + 1.3 = 8.0 ms. The update and the casts move
  bytes and do not shrink.
- Baseline is tf32, not true fp32. bf16 needs no loss scaling.
:::

::: {.slide title="Checkpointing Each Block Leaves the Loss as the Largest Term"}
[Activation checkpointing]{.kicker}

Store each block's input; recompute the block in the backward pass.

$$L \cdot 4d + a + a_\textrm{out} + \Delta \textrm{ bytes per token}$$

$\Delta$: measured peak minus count.

@!memory-precision-activation-checkpointing-1

- The output layer and the loss now store more than all 18 blocks.
- Under checkpointing, bf16 lowers the room: its $2N$ of weight copies
  outweigh the most it can save, on one recomputed block.
:::

::: {.slide title="Check: What Do Four Checkpointed Blocks Store?"}
[Check]{.kicker}

Four blocks of our GPT, 2,048 tokens, $d = 2048$, fp32. Per token, how
many bytes does checkpointing keep, and how much longer is the step?

. . .

- Four block inputs and the output the loss reads: $5 \times 4d$ bytes
  = **40 KiB**.
- The peak keeps one recomputed block on top of that.
- One more forward pass: **at most a third** more time.
:::

::: {.slide title="Four Blocks: 40 KiB per Token, about a Quarter More Time"}
[Activation checkpointing]{.kicker}

@!memory-precision-activation-checkpointing

- Stored bytes match the prediction; the peak falls by less.
- A compiler decides when the recomputation runs: checkpointing fixes what
  is stored, not where the peak is.
:::

::: {.slide title="Recompute What Is Cheap, Store What Is Expensive"}
[Activation checkpointing]{.kicker}

- Time sits in a few matrix products; memory in many cheap results.
- Selective recomputation (Korthikanti et al.): recompute only
  $QK^\top$, softmax, times $V$: $T \times T$ values per head, little
  arithmetic.
- Fused attention does the same inside one kernel.
- Left in our block: SwiGLU's activated gate and product, about $5d$ of
  the $20d$.
:::

::: {.slide title="A Chunked Loss Removes the Largest Term for 3% More Arithmetic"}
[Activation checkpointing]{.kicker}

Under checkpointing, the output layer and the loss store more per token
than all 18 blocks.

- Run the output layer and the cross-entropy on one chunk of tokens at a
  time, each chunk in its own checkpoint.
- Logits, log-probabilities, and their gradients exist for one chunk only.
- Cost: the logits are recomputed, $2dV$ FLOPs per token, about 3% of a
  step's $6N$.
:::

::: {.slide title="Accumulation Halves the Peak and Costs a Tenth to a Fifth More Time"}
[Gradient accumulation]{.kicker}

$B_\textrm{global} = m\,B_\textrm{micro}$; two-block GPT, 4,096 tokens
as 1 or 4 micro-batches. Predicted:

- peak: activations of 1,024 tokens + $4N$ = 0.77 GiB of gradients;
- time: $12N$ bytes of additions per micro-batch, $2I^*/n$ = 16% at
  $n = 1024$ ($I^* = 82$, tf32), about 7 ms for three micro-batches.

@!memory-precision-accumulated-step

Divide each loss by $m$; clip and step after the last micro-batch.
:::

::: {.slide title="Check: Two Sequences with m = 256, or Four with m = 128?"}
[Check]{.kicker}

Both give 524,288 tokens per step for our GPT in bf16. Two sequences fit
without checkpointing; four need it. Which is faster?

. . .

- Two, $m = 256$: accumulator $2 \times 164 / 2048 \approx 16\%$.
- Four, $m = 128$: accumulator 8%, recomputation about 25%.
- **Two sequences** win: checkpoint only when memory limits the
  micro-batch.
:::

::: {.slide title="Each Technique Changes One Term at a Cost"}
[Choosing a technique]{.kicker}

| symptom | technique | cost |
|---|---|---|
| products set the time | bf16 mixed precision | rounding; $+2N$ |
| one sequence barely fits | checkpoint every block | about a quarter more time |
| the loss dominates | loss in chunks | one more pass through the head |
| minibatch too small | accumulate $m$ micro-batches | $2I^*/n$ of the arithmetic |
| $16N$ alone fills the card | shard over GPUs | communication |
:::

::: {.slide title="One Card Holds a Step of Our GPT and Cannot Finish the Run"}
[Wrap-up]{.kicker}

- Memory = $16N$ (Adam, fp32) + tokens × bytes per token.
- A block: about $20d$ per token by three rules; the loss $V$ more, and
  the peak $8V$ bytes beyond the count where the loss writes its gradients.
- 1B GPT on 24 GiB: 15.1 GiB fixed, two or three sequences per step.
- bf16: 1.5 to 1.75 times as fast as tf32; memory hardly changes.
- Recipe: bf16, 256 micro-batches of 2,048 tokens: $3.2 \times 10^{15}$
  FLOPs, **at least 22 s per step**, a month or more for 100,000 steps.

**Next:** data parallelism shares the work of a step among GPUs but copies
the 17 GiB of state to each; sharding divides it.
:::
