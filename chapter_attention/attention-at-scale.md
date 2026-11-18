# The Cost of Attention
:label:`sec_attention-at-scale`

The language model of :numref:`sec_positional-information` read contexts of
128 bytes, and we never asked what its attention cost. Language models in
use today read whole reports, code repositories and books, often tens or
hundreds of thousands of tokens at once. Self-attention lets every token read
every other token in a single layer, or every earlier one in a language
model, and that is what makes it expensive: it computes one score for every
pair of positions. How does this cost grow with the length of the context,
and how much of it can be avoided without changing what the layer computes?

Doubling the context quadruples the number of pairs. A context of 8,192
tokens already has 67 million query--key pairs per head, and storing their
scores in single precision takes 256 MiB for one head of one sequence.
Arithmetic and memory both grow with the square of the sequence length, and
at long contexts this quadratic term dominates the cost of the attention
layer.

For several years this cost was taken to be a property of attention itself,
and from about 2019 on a whole literature approximated attention to escape
it. Then Rabe and Staats
showed, in a paper titled "Self-attention does not need $O(n^2)$ memory"
:cite:`Rabe.Staats.2021`, that the exact computation never needed the
matrix. Their tool was a trick that Milakov and Gimelshein had published
three years earlier to make the softmax alone faster
:cite:`Milakov.Gimelshein.2018`. The quadratic memory turned out to be a
property of one way of computing attention. The quadratic arithmetic is
another matter, and this section shows the difference.

We first count the cost, in operations and in bytes, and check both counts
against measurements on a GPU. Then we ask whether the score matrix has to
be stored at all, and what a GPU gains when it is not. Everything in this
section computes attention exactly; the next section asks what must be given
up to compute less. The cells time GPU kernels and need a CUDA GPU with a few
GiB of free memory.

```{.python .input #attention-at-scale-the-cost-of-attention}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #attention-at-scale-the-cost-of-attention}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #attention-at-scale-the-cost-of-attention-1}
%%tab pytorch
from torch.nn.attention import SDPBackend, sdpa_kernel
```

## Quadratic Time and Memory

### Counting FLOPs

Self-attention connects every pair of positions in one layer, where a
recurrence needs $n$ steps and a stack of convolutions many layers
(:numref:`sec_multihead-attention` compares the three). The cost shows in
the operation count. For a layer of width $d$ with $H$ heads on $n$ tokens,
:eqref:`eq_multihead-flops` gave the two leading terms, and the softmax adds
a third:

$$
\underbrace{8 n d^2}_{\textrm{projections}} \;+\;
\underbrace{4 n^2 d}_{\textrm{scores and mixing}} \;+\; O(Hn^2)
\quad \textrm{FLOPs}.
$$
:eqlabel:`eq_attn-layer-flops`

The quadratic term comes from the two matrix products inside each head. A
head of width $d_h$ forms $n^2$ dot products of length $d_h$ and mixes $n^2$
values, $4n^2 d_h$ FLOPs when one multiply--add counts as two, and with
$d_h = d/H$ the $H$ heads add up to $4n^2 d$. The softmax adds a few
operations per score, the term $O(Hn^2)$. The counts in this section are per
layer and in terms of the model width $d$; the measurements run one head at
a time and use $d_h$.

Which term matters depends on the context. Setting $4n^2d = 8nd^2$ gives
the crossover: the quadratic part dominates as soon as $n > 2d$. GPT-2
small, with $d = 768$ and contexts of 1,024 tokens, sits below the
crossover, so its projections still cost more than its scores. A model with
$d = 4096$ at $n = 131{,}072$ sits far beyond it. There $n/2d = 16$, so an
attention layer does sixteen times as much score-and-mix work as projection
work. The rule $n > 2d$ is worth remembering because it needs only two
numbers that every model description states. It compares attention with its
own projections, though. :numref:`chap_transformers` adds to every layer a
feed-forward network that typically costs twice as much as the projections,
which moves the crossover for a whole transformer layer to about $n = 6d$.

### Counting Memory
:label:`subsec_attn-counting-memory`

Arithmetic is only one part of the cost, and at long contexts memory is
often what runs out first. A naive implementation writes out the score
matrix and its softmax. These are two $n \times n$ buffers, $8n^2$ bytes in
single precision, per head and per sequence in the batch. The projections,
by contrast, keep only $O(nd)$ activations. Let us consider one head of
width $d_h = 64$, the head width of GPT-2 small, at $n = 8192$. The inputs
$\mathbf{Q}, \mathbf{K}, \mathbf{V}$ occupy 6 MiB together, while the two
score buffers occupy 512 MiB. At a context of $n = 131{,}072$ tokens a
*single* attention map in fp32 would take $n^2 \times 4$ bytes $= 64$ GiB
per head and sequence, which is why no system at that scale ever stores one.

Training makes matters worse, because the weights of every layer and every
head must be kept until the backward pass reaches them. Run GPT-2 small, 12
layers of 12 heads, at 8,192 tokens, and its saved fp32 weights alone take
$144 \times 256$ MiB $= 36$ GiB per sequence. That is more than the 24 GB
of the GPU this book runs on (:numref:`sec_hardware`), for a model whose 124
million parameters fit in half a gigabyte.

We check the formula on one head of width 64 as $n$ doubles. The function
`attention` computes
$\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_h})\mathbf{V}$ through
the full score matrix, and `d2l.peak_memory` of :numref:`sec_use_gpu`
reports the temporary memory that a call needs. The prediction is the two
$n \times n$ float buffers, $8n^2$ bytes.

:begin_tab:`pytorch`
Here `d2l.peak_memory` reads the high-water mark of PyTorch's caching
allocator.
:end_tab:

:begin_tab:`jax`
XLA reserves a large part of the GPU's memory at startup and decides at
compile time which buffers exist, so `d2l.peak_memory` asks the compiler how
much temporary memory the compiled function reserves.
:end_tab:

```{.python .input #attention-at-scale-counting-memory-1}
%%tab pytorch
def attention(Q, K, V):
    """softmax(Q K^T / sqrt(d_h)) V through the full score matrix."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    return torch.softmax(scores, dim=-1) @ V
```

```{.python .input #attention-at-scale-counting-memory-1}
%%tab jax
def attention(Q, K, V):
    """softmax(Q K^T / sqrt(d_h)) V through the full score matrix."""
    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(Q.shape[-1])
    return jax.nn.softmax(scores, axis=-1) @ V
```

```{.python .input #attention-at-scale-counting-memory-3}
%%tab pytorch
d_h = 64
for n in [2048, 4096, 8192, 16384]:
    Q = torch.randn(n, d_h, device=d2l.try_gpu())
    mem = d2l.peak_memory(attention, Q, Q, Q)  # Allocator high-water mark
    print(f'n = {n:5d}: measured {mem/2**20:7.1f}'
          f' MiB, predicted 8n^2 B = {8*n*n/2**20:7.1f} MiB')
```

```{.python .input #attention-at-scale-counting-memory-3}
%%tab jax
d_h = 64
for n in [2048, 4096, 8192, 16384]:
    Q = jax.random.normal(d2l.get_key(), (n, d_h))
    mem = d2l.peak_memory(attention, Q, Q, Q)  # XLA's compile-time scratch
    print(f'n = {n:5d}: XLA temp {mem/2**20:7.1f}'
          f' MiB, predicted 8n^2 B = {8*n*n/2**20:7.1f} MiB')
```

The measurement agrees with $8n^2$ bytes to within 10 MiB of workspace at
every length, and doubling $n$ quadruples it. At $n = 16{,}384$, a single
fp32 head already needs 2 GiB of temporary storage.

### Measuring Time

If the running time follows the operation count, each doubling of $n$
should multiply it by four. We time the forward pass as $n$ doubles, after
warm-up calls. On an accelerator, kernel launches return before the work is
done, so we synchronize before reading the clock. `d2l.Benchmark` of
:numref:`sec_use_gpu` does both and reports the average time per call.

```{.python .input #attention-at-scale-counting-memory-2}
%%tab pytorch
for n in [2048, 4096, 8192, 16384]:
    Q = torch.randn(n, d_h, device=d2l.try_gpu())
    print(d2l.Benchmark(lambda: attention(Q, Q, Q), desc=f'n = {n:5d}'))
```

```{.python .input #attention-at-scale-counting-memory-2}
%%tab jax
attention_jit = jax.jit(attention)
for n in [2048, 4096, 8192, 16384]:
    Q = jax.random.normal(d2l.get_key(), (n, d_h))
    print(d2l.Benchmark(lambda: attention_jit(Q, Q, Q), desc=f'n = {n:5d}'))
```

Between 8,192 and 16,384 tokens, where every call lasts more than a
millisecond, doubling $n$ multiplies the time by about four, as the
$4n^2 d_h$ operation count predicts. The shorter calls are poorer guides. At
2,048 and 4,096 tokens a call lasts a fraction of a millisecond, short
enough that the fixed cost of launching kernels weighs in, and the ratio
between neighboring lengths changes from run to run. The growth is half of
what the timing tells. The other half is in the size of the times, which we
can read once we know where a GPU keeps its data.

## Exact Attention Without the Matrix

### Blockwise Online Softmax

Must the $n \times n$ matrix exist at all? Each output row is a weighted
average of the values, with weights
$\exp(a_j)/\sum_{j'}\exp(a_{j'})$ that depend on *all* scores of the row
through the normalizer. This looks like an obstacle to computing attention
one block of keys at a time, because the weights of the first block are not
known until the last block has been seen. It turns out that they need not be
known early. It suffices to carry the unnormalized sums along and divide
once at the end, the way one computes a running mean from a running total
and a count.

One complication remains. A softmax subtracts the row maximum before
exponentiating, so that no exponent exceeds zero. Without it, a score above
about 88 overflows single precision
(:numref:`subsec_softmax-implementation-revisited`), and float16, one of the
two 16-bit formats in which attention runs, overflows at much smaller
scores. A blockwise computation does not know the row maximum in advance. It
keeps the largest score seen so far instead, and corrects the past whenever
a larger one arrives.

The correction is easiest to see in the smallest case: one query, four keys
in two blocks of two, with scores $(1, 3)$ in the first block and $(5, 2)$
in the second. After the first block the running maximum is $m = 3$, and the
running sum of exponentials, taken relative to that maximum, is
$s = e^{1-3} + e^{3-3} \approx 0.135 + 1 = 1.135$. The second block raises
the maximum to $m' = 5$. Every term of $s$ was computed relative to 3 and
must now be relative to 5, which means multiplying it by
$e^{3-5} \approx 0.135$. The old sum becomes 0.154, and the new block adds
$e^{0} + e^{-3} \approx 1.050$, so $s' \approx 1.203$. This is
$\sum_j e^{a_j - 5}$ over all four scores, as if we had known the maximum
from the start, and the largest score receives the weight
$1/1.203 \approx 0.83$. One multiplication repaired the whole first block,
however many keys it had held.

In general, we process the keys in blocks of $c$ and keep three running
quantities per query: the maximum $m$ seen so far, the sum $s$ of
exponentials relative to that maximum, and the output accumulator
$\mathbf{o}$, which sums the values with the same unnormalized weights. A
new block of scores $\{a_j\}$ with values $\{\mathbf{v}_j\}$ updates them as

$$
\begin{aligned}
m' &= \max\big(m, \max_j a_j\big), \\
s' &= s\, e^{m - m'} + \sum\nolimits_j e^{a_j - m'}, \\
\mathbf{o}' &= \mathbf{o}\, e^{m - m'} + \sum\nolimits_j e^{a_j - m'}\, \mathbf{v}_j,
\end{aligned}
$$
:eqlabel:`eq_online-softmax`

and after the last block, $\mathbf{o}/s$ is exactly
$\mathrm{softmax}(\mathbf{a})\mathbf{V}$. The factor $e^{m - m'}$ is the
multiplication that repaired the first block in the example. It equals one
when a block leaves the maximum unchanged, and otherwise it rescales
everything accumulated under the old maximum, so no approximation is made.
This way of computing a softmax is
called *online softmax* :cite:`Milakov.Gimelshein.2018`.
:numref:`fig_online-softmax` shows the resulting schedule for all queries at
once. One $n \times c$ stripe of the score matrix is in memory at a time,
next to a running state of size $O(n d_h)$ :cite:`Rabe.Staats.2021`.

![Online softmax processes the score matrix one key block at a time. Of the scores, only the current $n \times c$ stripe is in memory, next to the running maxima, sums and outputs $\mathbf{m}, \mathbf{s}, \mathbf{O}$ of all queries; the full $n \times n$ matrix is never stored.](../img/mdl-attention-online-softmax.svg)
:label:`fig_online-softmax`

### A Chunked Implementation
:label:`subsec_attn-chunked`

From here on we work in the causal setting of a language model, for two
reasons: the language models of this book are causal, and a causal kernel
can skip work, which will matter for the timing below. `d2l.causal_mask`
supplies the mask, and excluded scores are overwritten with the most
negative finite value of the dtype rather than with $-\infty$, as in
:numref:`sec_attention-scoring-functions`. Here the finite value does a
second job. If the first block that a query sees held only masked scores,
$-\infty$ would leave the running maximum at $-\infty$, and the exponent
$a_j - m'$ would become $-\infty + \infty$, which is undefined (Exercise
:exref:`ex_attention-at-scale-running-maximum`d traces this case).

We first implement a naive reference, which accepts any number of leading
axes such as heads, and then a chunked version for one head. The chunked
implementation applies :eqref:`eq_online-softmax` to all queries at once and
processes one key block per iteration.

```{.python .input #attention-at-scale-a-chunked-implementation-1}
%%tab pytorch
def causal_attention(Q, K, V):
    """Reference: causal attention with the full score matrix."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    mask = d2l.causal_mask(Q.shape[-2], device=Q.device)
    scores.masked_fill_(~mask, torch.finfo(Q.dtype).min)
    return torch.softmax(scores, dim=-1) @ V

def chunked_attention(Q, K, V, chunk_size=512):
    """Exact causal attention, one n-by-chunk_size block at a time."""
    n, d_h = Q.shape
    m = torch.full((n, 1), torch.finfo(Q.dtype).min, device=Q.device)
    s = torch.zeros(n, 1, device=Q.device)
    O = torch.zeros(n, V.shape[-1], device=Q.device)
    pos_q = torch.arange(n, device=Q.device)[:, None]
    for start in range(0, n, chunk_size):
        Kc, Vc = K[start:start + chunk_size], V[start:start + chunk_size]
        pos_k = torch.arange(start, start + Kc.shape[0], device=Q.device)
        scores = Q @ Kc.T / math.sqrt(d_h)
        scores.masked_fill_(pos_k[None, :] > pos_q, torch.finfo(Q.dtype).min)
        m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
        scale = torch.exp(m - m_new)                  # Rescale the past
        P = torch.exp(scores - m_new)
        s = s * scale + P.sum(dim=-1, keepdim=True)
        O = O * scale + P @ Vc
        m = m_new
    return O / s
```

```{.python .input #attention-at-scale-a-chunked-implementation-1}
%%tab jax
def causal_attention(Q, K, V):
    """Reference: causal attention with the full score matrix."""
    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(Q.shape[-1])
    mask = d2l.causal_mask(Q.shape[-2])
    scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    return jax.nn.softmax(scores, axis=-1) @ V

def chunked_attention(Q, K, V, chunk_size=512):
    """Exact causal attention, one n-by-chunk_size block at a time."""
    (n, d_h), d_v = Q.shape, V.shape[-1]
    assert n % chunk_size == 0, 'lax.scan needs equal-sized blocks'
    pos_q = jnp.arange(n)[:, None]
    def block(carry, chunk):
        m, s, O = carry
        Kc, Vc, start = chunk
        pos_k = start + jnp.arange(chunk_size)
        scores = Q @ Kc.T / math.sqrt(d_h)
        scores = jnp.where(pos_k[None, :] > pos_q,
                           jnp.finfo(scores.dtype).min, scores)
        m_new = jnp.maximum(m, scores.max(axis=-1, keepdims=True))
        scale = jnp.exp(m - m_new)                    # Rescale the past
        P = jnp.exp(scores - m_new)
        s = s * scale + P.sum(axis=-1, keepdims=True)
        O = O * scale + P @ Vc
        return (m_new, s, O), None
    init = (jnp.full((n, 1), jnp.finfo(Q.dtype).min), jnp.zeros((n, 1)),
            jnp.zeros((n, d_v)))
    chunks = (K.reshape(-1, chunk_size, d_h), V.reshape(-1, chunk_size, d_v),
              jnp.arange(0, n, chunk_size))           # Blocks, their starts
    (m, s, O), _ = jax.lax.scan(block, init, chunks)
    return O / s
```

The chunked version is meant to be exact, so we check it against the
reference.

```{.python .input #attention-at-scale-a-chunked-implementation-2}
%%tab pytorch
n, d_h = 2048, 64
Q, K, V = (torch.randn(n, d_h, device=d2l.try_gpu()) for _ in range(3))
err = (chunked_attention(Q, K, V) - causal_attention(Q, K, V)).abs().max()
print(f'maximum deviation: {float(err):.2e}')
```

```{.python .input #attention-at-scale-a-chunked-implementation-2}
%%tab jax
n, d_h = 2048, 64
Q, K, V = (jax.random.normal(k, (n, d_h))
           for k in jax.random.split(d2l.get_key(), 3))
# Full fp32 matrix products: the GPU default (TF32) would perturb the two
# computations differently, at the 1e-4 level
with jax.default_matmul_precision('highest'):
    err = jnp.abs(chunked_attention(Q, K, V)
                  - causal_attention(Q, K, V)).max()
print(f'maximum deviation: {float(err):.2e}')
```

The two computations agree up to floating-point rounding, at the level of
$10^{-7}$ to $10^{-6}$ in fp32. Chunking changes the order of evaluation,
not the definition of attention. The chunked version holds $n \times c$
scores at a time instead of $n \times n$, so its footprint should grow
linearly in $n$ rather than quadratically.

```{.python .input #attention-at-scale-a-chunked-implementation-3}
%%tab pytorch
lengths, mems = [2048, 4096, 8192, 16384], [[], []]
for n in lengths:
    Q = torch.randn(n, d_h, device=d2l.try_gpu())
    mems[0].append(d2l.peak_memory(causal_attention, Q, Q, Q) / 2**20)
    mems[1].append(d2l.peak_memory(chunked_attention, Q, Q, Q) / 2**20)
print(f'n = {n:,}: full matrix {mems[0][-1]:6.1f} MiB, '
      f'chunked {mems[1][-1]:6.1f} MiB')
d2l.plot(lengths, mems, 'sequence length n', 'peak memory (MiB)',
         legend=['full matrix', 'chunked'], xscale='log', yscale='log')
```

```{.python .input #attention-at-scale-a-chunked-implementation-3}
%%tab jax
lengths, mems = [2048, 4096, 8192, 16384], [[], []]
for n in lengths:
    Q = jax.random.normal(d2l.get_key(), (n, d_h))
    mems[0].append(d2l.peak_memory(causal_attention, Q, Q, Q) / 2**20)
    mems[1].append(d2l.peak_memory(chunked_attention, Q, Q, Q) / 2**20)
print(f'n = {n:,}: full matrix {mems[0][-1]:6.1f} MiB, '
      f'chunked {mems[1][-1]:6.1f} MiB')
d2l.plot(lengths, mems, 'sequence length n', 'XLA temp memory (MiB)',
         legend=['full matrix', 'chunked'], xscale='log', yscale='log')
```

At $n = 16{,}384$ the full-matrix implementation needs 2 GiB of scratch
memory or a little more, the score matrix of our count. The chunked one, as
the printed line shows, needs more than an order of magnitude less, and the
gap doubles with every further doubling of $n$.

:begin_tab:`pytorch`
The full-matrix line exceeds 2 GiB by about 256 MiB: the reference also
holds its boolean causal mask, one byte for each of the $n^2$ pairs.
:end_tab:

### Memory Traffic and Fused Kernels

Chunking saves memory. Can it save time as well? To answer, we need to know
where a GPU keeps its data. Its *main memory* holds gigabytes and is what
`peak_memory` measured. Next to the arithmetic units sits a much smaller and
much faster memory, a few hundred kilobytes for each group of units, that
holds only what is being worked on. A GPU computes far faster than it can
read from main memory. The RTX 4090 used for this book reads its main
memory at about 0.9 TB/s, while it performs about 80 trillion operations
per second on 32-bit matrices and 165 trillion on 16-bit ones
(:numref:`chap_performance` measures these numbers). An operation that
performs little arithmetic per byte it moves therefore spends most of its
time waiting for data.

With these numbers, the timing of one head at $n = 16{,}384$ can be read for
its size. The score matrix holds 1 GiB in fp32, so a single pass over it,
one write or one read, takes about 1.2 ms. The arithmetic,
$4n^2 d_h \approx 7 \times 10^{10}$ operations, takes under a millisecond at
peak. The call took several milliseconds: the time of several passes over
the matrix, and several times what the arithmetic needs. The reason lies in
what naive attention does with the matrix. It writes the $n^2$ scores to
main memory, reads them back for the softmax, writes the weights, and reads
them again for the value mixing. These are four passes at the least, with
far fewer operations per byte moved than the GPU needs to keep its
arithmetic units busy (Exercise
:exref:`ex_attention-at-scale-arithmetic-intensity` counts them).

Our `chunked_attention` does not avoid these passes. Every operation in its
loop writes its result to main memory and the next one reads it back, so it
saves memory but not traffic (Exercise
:exref:`ex_attention-at-scale-tiled-traffic`a counts both). Saving the
traffic takes a single kernel that runs the whole loop and keeps each piece
of the score matrix in fast memory until all work on it is done. A stripe is
far too large for that: at $n = 8{,}192$ and $c = 512$ it holds 16 MiB in
fp32. The rows of the score matrix do not interact, however, since each
query carries its own $m$, $s$ and $\mathbf{o}$. So the queries can be cut
into tiles as well, and a tile of queries meets the key blocks one at a
time, through a tile of scores small enough to stay in fast memory.

FlashAttention :cite:`Dao.Fu.Ermon.ea.2022` is this schedule fitted to the
hardware: tiles of queries and keys sized to the fast memory, so that every
tile of scores is used where it is computed and never reaches main memory.
The result is exact attention that is faster *and* needs memory linear in
$n$.

Training adds a difficulty. The backward pass needs the attention weights,
which the fused kernel never stored. It recomputes them, tile by tile, from
the inputs and from one number per query that the forward pass keeps,
$\ell = m + \log s$, the logarithm of the final normalizer. Any weight is
then $e^{a_j - \ell}$, computed from its score alone. Recomputing every score
sounds wasteful, but on a GPU the extra arithmetic takes less time than the
memory traffic it avoids. For GPT-2 small at 8,192 tokens, the 36 GiB of
saved weights become 4.5 MiB of such numbers.

Frameworks provide this schedule as a fused kernel, one kernel that performs
all steps without writing intermediate results to main memory:
`torch.nn.functional.scaled_dot_product_attention` in PyTorch and
`jax.nn.dot_product_attention` with `implementation='cudnn'` in JAX. We
compare it with the naive reference on a minibatch of 2 sequences with 8
heads of width 64 in half precision at $n = 8192$. The memory can be
predicted before running. The naive reference holds two $n \times n$
half-precision buffers for each of the 16 heads, $16 \cdot 4n^2$ bytes
$= 4$ GiB, while the fused kernel needs little beyond its output,
$2 \cdot 8 \cdot 8192 \cdot 64$ numbers of two bytes, or 16 MiB. A third
run switches the mask off. The fused kernel then computes every tile, as
the naive reference does, and the comparison separates what skipping tiles
saves from what the schedule saves.

```{.python .input #attention-at-scale-the-bottleneck-is-memory-traffic}
%%tab pytorch
def fused_attention(Q, K, V, is_causal=True):
    """Attention in PyTorch's fused FlashAttention kernel."""
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

B, H, n = 2, 8, 8192
X = torch.randn(B, H, n, d_h, device=d2l.try_gpu(), dtype=torch.float16)
# The unmasked kernel computes every tile: it shows what skipping saves
for name, f in [('naive', causal_attention), ('fused', fused_attention),
                ('fused, no mask',
                 lambda Q, K, V: fused_attention(Q, K, V, is_causal=False))]:
    t = d2l.Benchmark(lambda: f(X, X, X)).time
    mem = d2l.peak_memory(f, X, X, X)
    print(f'{name:>14}: {t*1e3:6.2f} ms, peak memory {mem/2**20:7.1f} MiB')
```

```{.python .input #attention-at-scale-the-bottleneck-is-memory-traffic}
%%tab jax
def fused_attention(Q, K, V, is_causal=True):
    """Attention in the fused cuDNN kernel."""
    # The kernel takes (batch, length, heads, width): swap axes 1 and 2
    Q, K, V = (A.swapaxes(1, 2) for A in (Q, K, V))
    O = jax.nn.dot_product_attention(Q, K, V, is_causal=is_causal,
                                     implementation='cudnn')
    return O.swapaxes(1, 2)

B, H, n = 2, 8, 8192
X = jax.random.normal(d2l.get_key(), (B, H, n, d_h), dtype=jnp.float16)
# The unmasked kernel computes every tile: it shows what skipping saves
for name, f in [('naive', causal_attention), ('fused', fused_attention),
                ('fused, no mask',
                 lambda Q, K, V: fused_attention(Q, K, V, is_causal=False))]:
    f_jit = jax.jit(f)
    t = d2l.Benchmark(lambda: f_jit(X, X, X)).time
    mem = d2l.peak_memory(f, X, X, X)
    print(f'{name:>14}: {t*1e3:6.2f} ms, XLA temp {mem/2**20:7.1f} MiB')
```

The prediction holds: about 4 GiB of scratch memory for the naive
reference, and for the fused kernel a small multiple of its 16 MiB output.
The fused kernel is also several times faster. One might suspect the mask
of doing the work, since a causal kernel skips the tiles above the
diagonal, about half of them. The third line settles this. Skipping saves
about a factor of 1.5, not the full two: the causal kernel does half the
arithmetic but at a lower rate. And without the mask, computing exactly as
many scores as the naive reference, the fused kernel is still several times
faster. The tiles explain a little of the gap.

The rest is traffic, and the hardware numbers above show it. Without the
mask the fused kernel performs
$4 \cdot 16 \cdot n^2 \cdot 64 \approx 2.7 \times 10^{11}$ operations, which
would take 1.7 ms at the card's 16-bit peak, and the measured time is close
to that. Its arithmetic sets its pace;
memory hardly enters. The naive reference moves 2 GiB with every pass over
its 16 score matrices, about 2.4 ms per pass, and its measured time is that
of about four passes or more, while the same arithmetic would fill less
than a fifth of it. Naive attention spends most of its time writing and
reading a matrix that it never needed to keep.

:begin_tab:`pytorch`
`sdpa_kernel(SDPBackend.FLASH_ATTENTION)` makes
`scaled_dot_product_attention` use the FlashAttention kernel, which takes
half and bfloat16 inputs, instead of letting PyTorch choose a backend.
:end_tab:

:begin_tab:`jax`
`implementation='cudnn'` selects cuDNN's fused kernel for half and bfloat16
inputs on NVIDIA GPUs. The default implementation is XLA's own, which
writes out the full score matrix.
:end_tab:

Exact attention in a fused kernel is what nearly every language model runs
today, and it gives up nothing: the function is the same, only the schedule
differs. It removes the stored matrix and most of the traffic, but not the
arithmetic. Every query still scores every earlier key, so the quadratic
term of :eqref:`eq_attn-layer-flops`, halved by the causal mask, still grows
fourfold with each doubling of the context. Doing less than that means
computing something other than dense attention, and the next section asks
what such a change costs.

## Summary

Dense self-attention scores every pair of positions, so a layer costs
$4n^2d$ FLOPs beside the $8nd^2$ of its projections, and the quadratic term
dominates once the context exceeds twice the model width. A naive
implementation also stores two $n \times n$ buffers per head, which our
measurements confirm at $8n^2$ bytes, and spends most of its time moving
them through the GPU's main memory. Online softmax removes the matrix
without changing the result: it processes keys in blocks and carries a
running maximum, normalizer and output, rescaling them whenever the maximum
grows. FlashAttention runs this schedule on tiles that stay in the GPU's
fast memory, which makes exact attention several times faster and linear in
memory, but leaves the $n^2$ arithmetic for the next section.

## Exercises

1. **Online softmax by hand.** One query meets four keys in two blocks of
   two. Its scores are $(0, 2)$ in the first block and $(1, 3)$ in the
   second, and the values are the numbers $1, 2$ in the first block and
   $3, 4$ in the second.
    1. Apply :eqref:`eq_online-softmax` block by block, starting from
       $m = -\infty$, $s = 0$ and $o = 0$. Record $m$, $s$ and $o$ after
       each block, and divide to obtain the output.
    1. Compute the four softmax weights directly, and check that they give
       the same output.
    1. After the last block, form $\ell = m + \log s$ and recompute each of
       the four weights as $e^{a_j - \ell}$ from its score alone. Check that
       they agree with the weights of the previous part and sum to one, and
       explain why a fused kernel's backward pass can keep $\ell$ instead of
       $m$ and $s$.

1. [code] **The running maximum.** Remove the running maximum from
   `chunked_attention`, so that the state accumulates plain sums of
   $e^{a_j}$. Run the result with $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$
   and the state $s$, $\mathbf{O}$ in float16, on queries scaled up by a
   factor of $10$.
    1. Before running, compute the largest exponent $x$ for which $e^x$ is
       finite in float16, and estimate the fraction of scaled scores that
       exceed it when queries and keys of width $64$ have independent
       standard normal entries.
    1. Run the modified function, and report at which operation the first
       infinite or undefined value appears.
    1. Explain why carrying $m$ bounds every exponent by zero. A term
       $e^{a_j - m'}$ that is smaller than the smallest positive float16
       number rounds to zero. Bound the total contribution of such terms to
       $s$, relative to the term of the largest score, for a row of $n$
       keys.
    1. `chunked_attention` initializes $m$ with the most negative finite
       value of the dtype and masks scores with the same value. Take
       $n = 2c$ for blocks of $c$ keys, and let the loop visit the two key
       blocks in reverse order, so that every score of the first visited
       block is masked for a query in the first block. Trace $m$, $s$, and
       $\mathbf{O}$ for that query through both iterations, and show that
       the final output is still exact.
   :exlabel:`ex_attention-at-scale-running-maximum`

1. **Merging partial states.** Fix one query with scores $a_j$ and values
   $\mathbf{v}_j$. For a set $B$ of keys, call the triple $(m_B, s_B,
   \mathbf{o}_B)$ that :eqref:`eq_online-softmax` produces from the keys in
   $B$ alone the partial state of $B$, so that $\mathbf{o}_B / s_B$ is the
   attention output restricted to $B$.
    1. Derive the partial state of $B \cup B'$ for disjoint sets $B$ and
       $B'$ from the two partial states, and show that
       :eqref:`eq_online-softmax` is the special case in which $B'$ is a
       single block.
    1. Show that this merge is associative and commutative. Conclude that
       the key blocks can be processed in any order, or in parallel and
       merged in a tree of depth $\lceil \log_2 (n/c) \rceil$ for blocks of
       $c$ keys, and explain why this matters when a single query attends
       to a long cache during generation.
    1. Replace the pair $(m_B, s_B)$ by the single number $\ell_B = m_B +
       \log s_B$ and the triple by $(\ell_B, \bar{\mathbf{o}}_B)$ with
       $\bar{\mathbf{o}}_B = \mathbf{o}_B / s_B$. Express the merged
       $\ell$ and $\bar{\mathbf{o}}$ in these terms, and show that
       $\bar{\mathbf{o}}$ is a convex combination of
       $\bar{\mathbf{o}}_B$ and $\bar{\mathbf{o}}_{B'}$ whose weights
       depend only on $\ell_B$ and $\ell_{B'}$.

1. **Arithmetic intensity of attention.** Consider one head with $n$
   queries and keys of width $d_h$ and values of width $d_v$. Its two matrix
   products, $\mathbf{Q}\mathbf{K}^\top$ and the product of the weights with
   $\mathbf{V}$, cost $2n^2 d_h$ and $2n^2 d_v$ floating-point operations,
   and the softmax costs a few operations per score.
    1. Suppose that the score matrix is written to memory once and read
       once by the softmax, whose output is written once and read once by
       the value product, all in single precision. Count the bytes moved by
       these four passes, and by reading $\mathbf{Q}$, $\mathbf{K}$,
       $\mathbf{V}$ and writing the output.
    1. Neglecting the softmax arithmetic, form the ratio of operations to
       bytes and let $n \to \infty$ at fixed $d_h$ and $d_v$. Show that the
       limit is $(d_h + d_v)/8$, and state how the softmax arithmetic
       changes it.
    1. Redo the limit for inputs and scores in a 16-bit format. Take the
       16-bit throughput and the memory bandwidth of an H100 or a B200 from
       :numref:`tab_gpu_specs`, compute the ratio of operations per byte,
       and decide whether attention with $d_h = d_v = 64$ is limited by
       arithmetic or by memory traffic on it.
   :exlabel:`ex_attention-at-scale-arithmetic-intensity`

1. **Traffic of a tiled schedule.** ● By Exercise
   :exref:`ex_attention-at-scale-arithmetic-intensity`a,
   the naive layer moves $4n^2$ floats to and from main memory beyond
   the terms linear in $n$, since the score matrix and the weight matrix
   are each written once and read once. Ignore all traffic that is linear
   in $n$.
    1. Suppose that `chunked_attention` runs one operation at a time, each
       reading its inputs from main memory and writing its output back.
       Count, as functions of $n$, $c$, and $d_h$, the floats moved for the
       score stripes, the floats moved for $\mathbf{Q}$ and the accumulator
       $\mathbf{O}$, and the number of operations launched. Which of these
       quantities does chunking reduce relative to the naive layer?
    1. A fused kernel instead holds a tile of $b$ queries, their output
       accumulator, and one tile each of $b$ keys and $b$ values in a fast
       memory of $M$ floats, so that $4bd_h \leq M$; the scores of a tile
       and the running $m$ and $s$ stay in fast memory as well. It loads
       each query tile once, streams
       all key and value tiles past it, and writes each output tile once.
       Show that its traffic beyond the linear terms is $2n^2 d_h / b$
       floats and that the largest admissible tile gives
       $\Theta(n^2 d_h^2 / M)$, the bound that
       :citet:`Dao.Fu.Ermon.ea.2022` prove for FlashAttention.
    1. Compute the ratio of the fused traffic to the naive traffic for
       $d_h \in \{64, 128, 256\}$ with $M = 2^{16}$, and state how the head
       width limits the benefit of tiling.
   :exlabel:`ex_attention-at-scale-tiled-traffic`

[Discussions](https://d2l.discourse.group/t/36772)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.5]{.kicker}

The Cost of Attention<br>
**quadratic cost · online softmax · fused kernels**
:::
:::

::: {.slide title="One Score Matrix at 131,072 Tokens Takes 64 GiB"}
[The problem]{.kicker}

Self-attention scores every pair of positions: $n^2$ scores per head.

| context $n$ | scores per head | fp32 bytes |
|--:|--:|--:|
| 1,024 (GPT-2 small) | 1.0 M | 4 MiB |
| 8,192 | 67 M | 256 MiB |
| 131,072 | 17 G | 64 GiB |

::: {.d2l-note}
Time and memory grow with $n^2$. How do long contexts fit at all?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- count the FLOPs and bytes of one attention layer and find where the
  $n^2$ term takes over;
- predict the scratch memory of naive attention, and check it;
- compute exact attention one key block at a time (online softmax), by hand
  and in code;
- explain why a fused kernel is faster although it computes the same thing.

**Running example:** one head of GPT-2 small: width 64, $d = 768$, contexts
of 1,024 tokens, and the same head at 8,192 tokens.
:::

::: {.slide title="Exact Attention Comes Before Cheaper Attention"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Queries, Keys, and Values | attention as a soft lookup |
| Attention Scoring and Masking | scaled dot products, masks |
| Self-Attention and Multiple Heads | a sequence reads itself; several heads |
| Positional Information | where a token sits |
| **The Cost of Attention** | **quadratic cost, fused kernels** |
| Windowed and Linear Attention | fewer scores, at a price |
| Attention Circuits | what trained heads compute |
:::

::: {.col .narrow}
- **Uses:** multi-head attention and its FLOP count, causal masks, the
  stable softmax.
- **Skips:** windows and linear attention (next deck), serving engines.
:::
:::
:::

::: {.slide title="Scores Cost More Than Projections Once n > 2d"}
[Counting the cost]{.kicker}

Width $d$, $n$ tokens, one multiply--add = 2 FLOPs:

- scores $\mathbf{Q}\mathbf{K}^\top$: $2n^2d$; mixing with $\mathbf{V}$:
  another $2n^2d$;
- projections $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$:
  $8nd^2$, for any number of heads;
- ratio $\dfrac{4n^2d}{8nd^2} = \dfrac{n}{2d}$.

::: {.d2l-note}
$d = 4096$ at $n = 131{,}072$: the scores cost 16× the projections. With a
transformer's feed-forward network the crossover moves to about $n = 6d$.
:::
:::

::: {.slide title="Check: GPT-2 Small at Its Own Context"}
[Check]{.kicker}

GPT-2 small: $d = 768$, context $n = 1{,}024$. Which costs more per layer,
the projections or the scores? At which context do they cost the same?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $n/2d = 1024/1536 \approx 0.67$: the projections still cost
more. The two match at $n = 2d = 1{,}536$.
:::
:::
:::

::: {.slide title="Score Buffers Take 8n² Bytes per Head"}
[Counting the cost]{.kicker}

Naive attention keeps the scores and their softmax: two $n \times n$ fp32
buffers, $8n^2$ bytes per head and sequence. One head of width 64:

@!attention-at-scale-counting-memory-3

::: {.d2l-note}
Doubling $n$ quadruples it: 2 GiB at 16,384 tokens. Training keeps every
layer's weights: GPT-2 small at 8,192 tokens saves 36 GiB per sequence.
:::
:::

::: {.slide title="Time Quadruples with Each Doubling of n"}
[Counting the cost]{.kicker}

Forward pass of the same head ($d_h = 64$), after warm-up, synchronized:

@!attention-at-scale-counting-memory-2

::: {.d2l-note}
8,192 → 16,384 tokens, calls above 1 ms: ×4, the $4n^2d_h$ count.
Sub-millisecond calls: launch costs blur the ratio.
:::
:::

::: {.slide title="A Running Maximum Keeps Every Exponent at Most Zero"}
[Exact attention without the matrix]{.kicker}

- One query with scores $a_j$: a weight $e^{a_j}/\sum_{j'} e^{a_{j'}}$
  needs the row's sum and, for safety, its maximum.
- Without the maximum, $e^{a}$ overflows fp32 above $a \approx 88$, fp16
  much sooner.
- Blockwise: keep the largest score so far, **repair the past** when it
  grows.
:::

::: {.slide title="One Multiplication Repairs the First Block"}
[Exact attention without the matrix]{.kicker}

One query, scores $(1, 3)$ then $(5, 2)$; $s$ is relative to $m$.

| | after block 1 | after block 2 |
|:--|--:|--:|
| running max $m$ | 3 | 5 |
| old sum $\times\, e^{m - m'}$ | | $1.135 \times 0.135 = 0.154$ |
| new terms | $e^{-2} + e^{0} = 1.135$ | $e^{0} + e^{-3} = 1.050$ |
| sum $s$ | 1.135 | 1.203 |

::: {.d2l-note}
$1.203 = \sum_j e^{a_j - 5}$: as if the maximum had been known from the
start. Weight of the score 5: $1/1.203 \approx 0.83$.
:::
:::

::: {.slide title="Check: The Blocks in Reverse Order"}
[Check]{.kicker}

Same scores, but block $(5, 2)$ comes first and $(1, 3)$ second. What is
the rescaling factor at the second block, and what is the final $s$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The maximum stays 5, so $e^{m - m'} = 1$.
$s = 1.050 + e^{-4} + e^{-2} \approx 1.203$: the same sum.
:::
:::
:::

::: {.slide title="Online Softmax Carries Three Statistics per Query"}
[Exact attention without the matrix]{.kicker}

Keys in blocks, values $\mathbf{v}_j$ (rows of $\mathbf{V}$); per query
the running maximum $m$, the sum $s$, and the output $\mathbf{o}$:

$$m' = \max\big(m, \max_j a_j\big)$$
$$s' = s\,e^{m-m'} + \textstyle\sum_j e^{a_j - m'}$$
$$\mathbf{o}' = \mathbf{o}\,e^{m-m'} + \textstyle\sum_j e^{a_j-m'}\mathbf{v}_j$$

::: {.d2l-note}
After the last block $\mathbf{o}/s = \mathrm{softmax}(\mathbf{a})\mathbf{V}$
exactly. No approximation anywhere.
:::
:::

::: {.slide title="Only One Stripe of Scores Is Ever in Memory"}
[Exact attention without the matrix]{.kicker}

![](../img/mdl-attention-online-softmax.svg){width=78%}

Memory: one $n \times c$ stripe ($c$ keys per block) plus $O(n d_h)$
running state, instead of $n \times n$.
:::

::: {.slide title="Chunked Attention Is Exact in Linear Memory"}
[Exact attention without the matrix]{.kicker}

@!attention-at-scale-a-chunked-implementation-2

@!attention-at-scale-a-chunked-implementation-3

::: {.d2l-note}
Same answer to fp32 rounding, same FLOPs. Chunking changes the order of
evaluation, not the definition of attention.
:::
:::

::: {.slide title="Naive Attention Waits for the Score Matrix"}
[Memory traffic]{.kicker}

- **Main memory:** gigabytes, 0.9 TB/s. **Fast memory** beside the
  arithmetic units: kilobytes, much faster.
- Naive: write scores, read for softmax, write weights, read for mixing.
- One fp32 head, $n = 16{,}384$: a pass over the 1 GiB matrix 1.2 ms;
  arithmetic under 1 ms at peak; measured: several ms.

::: {.d2l-note}
Our chunked loop writes every stripe out too: less memory, not less traffic.
:::
:::

::: {.slide title="FlashAttention Keeps the Tiles in Fast Memory"}
[Memory traffic]{.kicker}

- Rows do not interact: cut queries into tiles too; each score tile
  stays in fast memory.
- No $n \times n$ buffer in main memory: memory linear in $n$.
- Backward **recomputes** score tiles from one number per query,
  $\ell = m + \log s$: weight $= e^{a_j - \ell}$.
- `scaled_dot_product_attention` (PyTorch); `dot_product_attention(...,
  implementation='cudnn')` (JAX).

[Dao et al. 2022]{.small}
:::

::: {.slide title="Check: Predict the Scratch Memory"}
[Check]{.kicker}

2 sequences × 8 heads of width 64, $n = 8{,}192$, fp16, causal. How much
scratch memory do the naive reference and the fused kernel need?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Naive: 16 heads × two $n \times n$ buffers × 2 bytes
$= 16 \cdot 4n^2$ B $= 4$ GiB. Fused: about the output,
$2 \cdot 8 \cdot 8192 \cdot 64 \cdot 2$ B $= 16$ MiB.
:::
:::
:::

::: {.slide title="The Fused Kernel Is Limited by Arithmetic, the Naive One by Memory"}
[Memory traffic]{.kicker}

@!attention-at-scale-the-bottleneck-is-memory-traffic

- Fused, no mask: $2.7 \times 10^{11}$ FLOPs, 1.7 ms at 165 TFLOP/s;
  measured close to that.
- Naive: 2 GiB per pass over the scores, 2.4 ms; about four passes or more.
- Skipping the tiles above the diagonal: about 1.5×, not 2×.

::: {.d2l-note}
Naive attention spends most of its time moving a matrix it never needed to
keep.
:::
:::

::: {.slide title="Fused Kernels Remove the Matrix but Not the n² Arithmetic"}
[What remains]{.kicker}

- Same function, new schedule: what nearly every language model runs.
- Every query still scores every earlier key: ×4 per doubling.
- At $d = 4096$ and 131,072 tokens: still 8× the projections with the
  causal mask, 16× without.

::: {.d2l-note}
Cheaper than this means computing something other than dense attention.
:::
:::

::: {.slide title="Check: Score Memory of GPT-2 Small"}
[Check]{.kicker}

One layer of GPT-2 small: 12 heads of width 64, $n = 1{,}024$, one
sequence. How much temporary memory do the naive layer (fp32 scores) and the
fused kernel (fp16) need? And at $n = 8{,}192$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Naive: $12 \cdot 8n^2$ bytes $= 96$ MiB at 1,024 tokens and
6 GiB at 8,192. Fused: the output plus one number $\ell$ per query,
about 1.5 MiB and 12 MiB, linear in $n$.
:::
:::
:::

::: {.slide title="Exact Attention Costs n² Arithmetic and Linear Memory"}
[Wrap-up]{.kicker}

- **Count:** $8nd^2 + 4n^2d$ FLOPs; scores dominate once $n > 2d$ (GPT-2
  small at 1,024 tokens: not yet).
- **Memory:** naive $8n^2$ bytes per head, 512 MiB at 8,192 tokens.
- **Online softmax:** carry $m, s, \mathbf{o}$; rescale by $e^{m - m'}$;
  exact.
- **Fused kernel:** no stored matrix, several times faster; the $n^2$
  arithmetic stays (halved by the causal mask).

**Next:** give up exactness to score fewer pairs: windows and linear
attention.
:::
