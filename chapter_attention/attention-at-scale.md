# The Cost of Attention
:label:`sec_attention-at-scale`

Long-context language models process tens or hundreds of thousands of tokens
in a single pass. Because dense self-attention evaluates a score for every
pair of positions, both its arithmetic and its naive memory footprint grow
quadratically with sequence length $n$. At $n = 8{,}192$, a single head
evaluates $6.7 \times 10^7$ query--key pairs (256 MiB in float32).

For several years, $O(n^2)$ memory was widely assumed to be an unavoidable
cost of exact attention. Then :citet:`Rabe.Staats.2021` demonstrated—building
on the *online softmax* algorithm of :citet:`Milakov.Gimelshein.2018`—that
exact attention never requires materializing the $n \times n$ score matrix,
and *FlashAttention* :cite:`Dao.Fu.Ermon.ea.2022` turned tiled online softmax
into an IO-aware GPU kernel that is both faster than naive attention and
linear in memory.

This section counts the FLOPs and memory of dense attention, benchmarks them
on a GPU, derives blockwise online softmax, and compares a chunked
implementation and fused FlashAttention kernels against naive matrix
materialization.

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

For a multi-head self-attention layer of width $d$ with $H$ heads of width
$d_h = d/H$ over $n$ tokens, :eqref:`eq_multihead-flops` together with the
row-wise softmax gives

$$
\underbrace{8 n d^2}_{\textrm{projections}} \;+\;
\underbrace{4 n^2 d}_{\textrm{scores and mixing}} \;+\; O(Hn^2)
\quad \textrm{FLOPs}.
$$
:eqlabel:`eq_attn-layer-flops`

The quadratic $4n^2 d$ term ($2n^2 d_h$ for $\mathbf{Q}\mathbf{K}^\top$ and
$2n^2 d_h$ for $\mathbf{A}\mathbf{V}$ across $H$ heads) overtakes the linear
$8nd^2$ projection term once $n > 2d$. For GPT-2 small ($d = 768$, $n =
1{,}024$), projections still dominate; for a model with $d = 4{,}096$ at
$n = 131{,}072$, $n / (2d) = 16$, so scoring and mixing require $16\times$ as
many FLOPs as the attention projections (and roughly $5\times$ as many as the
entire transformer block once the feed-forward network of
:numref:`chap_transformers` is included, which moves the block crossover to
$n \approx 6d$).

### Counting Memory
:label:`subsec_attn-counting-memory`

At long contexts, memory often becomes the bottleneck before arithmetic. A
naive implementation materializes both the $n \times n$ score matrix and its
$n \times n$ softmax weight matrix, consuming $8n^2$ bytes per head and
sequence in float32 (versus $O(nd)$ for projections). For one head of width
$d_h = 64$ at $n = 8{,}192$, $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ occupy just
6 MiB together while the two $n \times n$ buffers consume 512 MiB; at $n =
131{,}072$, a single float32 attention matrix would require 64 GiB. During
training, saving attention weights across all 12 layers and 12 heads of GPT-2
small at $n = 8{,}192$ would require $144 \times 256\textrm{ MiB} = 36\textrm{
GiB}$ per sequence.

We verify the $8n^2$-byte scratch footprint on one float32 head ($d_h = 64$)
using `d2l.peak_memory` (:numref:`sec_use_gpu`):

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

### Measuring Time

Timing the same forward pass with `d2l.Benchmark` (:numref:`sec_use_gpu`)
confirms the quadratic runtime growth once $n$ is large enough to amortize
kernel launch latency ($n \ge 8{,}192$):

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

## Exact Attention Without the Matrix

### Blockwise Online Softmax

At first glance, the softmax denominator $\sum_{j'} \exp(a_{j'})$ and the
numerical-stability row maximum $\max_{j'} a_{j'}$
(:numref:`subsec_softmax-implementation-revisited`) appear to require all $n$
scores before any value can be weighted. *Online softmax*
:cite:`Milakov.Gimelshein.2018` avoids this two-pass requirement by
processing keys in blocks of size $c$ and maintaining three running statistics
per query: the running maximum score $m$, the running denominator sum $s =
\sum_j e^{a_j - m}$, and the unnormalized output accumulator $\mathbf{o} =
\sum_j e^{a_j - m} \mathbf{v}_j$.

When a new block of scores $\{a_j\}$ and values $\{\mathbf{v}_j\}$ arrives,
updating the running maximum to $m' = \max(m, \max_j a_j)$ requires rescaling
the previously accumulated sum $s$ and output $\mathbf{o}$ by $e^{m - m'}$:

$$
\begin{aligned}
m' &= \max\big(m, \max_j a_j\big), \\
s' &= s\, e^{m - m'} + \sum\nolimits_j e^{a_j - m'}, \\
\mathbf{o}' &= \mathbf{o}\, e^{m - m'} + \sum\nolimits_j e^{a_j - m'}\, \mathbf{v}_j.
\end{aligned}
$$
:eqlabel:`eq_online-softmax`

For example, if block 1 has scores $(1, 3)$, then $m = 3$ and $s = e^{-2} +
e^0 \approx 1.135$. If block 2 has scores $(5, 2)$, the new maximum is $m' =
5$, so the block-1 sum is rescaled by $e^{3 - 5} = e^{-2}$ and incremented by
$e^{5 - 5} + e^{2 - 5}$, yielding $s' = e^{-4} + e^{-2} + e^0 + e^{-3}
\approx 1.203$—identical to the standard softmax denominator computed with
the global maximum $5$ known in advance. After the final block,
$\mathbf{o} / s$ equals $\mathrm{softmax}(\mathbf{a})\mathbf{V}$ up to
floating-point round-off, while holding only one $n \times c$ score stripe in
memory at a time (:numref:`fig_online-softmax`, :cite:`Rabe.Staats.2021`).

![Online softmax processes the score matrix one key block at a time. Of the scores, only the current $n \times c$ stripe is in memory, next to the running maxima, sums and outputs $\mathbf{m}, \mathbf{s}, \mathbf{O}$ of all queries; the full $n \times n$ matrix is never stored.](../img/mdl-attention-online-softmax.svg)
:label:`fig_online-softmax`

### A Chunked Implementation
:label:`subsec_attn-chunked`

We compare full-matrix `causal_attention` against `chunked_attention`, which
steps through key--value blocks of width `chunk_size=512` using
:eqref:`eq_online-softmax` (initializing $m$ and masking future positions with
`finfo(dtype).min` to avoid $-\infty - (-\infty) = \mathrm{NaN}$ when an
entire block is causally masked; see Exercise
:exref:`ex_attention-at-scale-running-maximum`d).

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

`chunked_attention` matches `causal_attention` to float32 precision ($\sim
10^{-6}$) while reducing peak memory from $O(n^2)$ to $O(nc)$:

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

### Memory Traffic and Fused Kernels

A GPU's compute units execute matrix arithmetic orders of magnitude faster
than its off-chip high-bandwidth memory (HBM) can transfer data (e.g., on an
RTX 4090, $\sim 165$ TFLOP/s in float16 versus $\sim 0.9$ TB/s HBM bandwidth;
see :numref:`chap_performance`). Naive attention is *memory-bandwidth bound*:
it writes the $n \times n$ score matrix to HBM, reads it back for softmax,
writes the $n \times n$ weights to HBM, and reads them again to multiply by
$\mathbf{V}$—moving $4n^2$ floats across HBM with low arithmetic intensity
(Exercises :exref:`ex_attention-at-scale-arithmetic-intensity` and
:exref:`ex_attention-at-scale-tiled-traffic`). Because a Python loop like
`chunked_attention` still launches separate GPU kernels that read and write
every $n \times c$ stripe to HBM, it reduces peak memory allocation without
reducing total HBM traffic.

*FlashAttention* :cite:`Dao.Fu.Ermon.ea.2022` tiles both queries and keys so
that each score block and its running statistics $(m, s, \mathbf{o})$ reside
entirely in fast on-chip SRAM throughout the loop, never touching HBM. For
the backward pass, instead of saving $n \times n$ attention weights, the
forward pass stores only the scalar log-normalizer $\ell = m + \log s$ per
query ($O(n)$ memory) and recomputes the score tiles on chip during
backpropagation.

We benchmark the framework-provided fused FlashAttention kernel
(`F.scaled_dot_product_attention` in PyTorch; `jax.nn.dot_product_attention`
with `implementation='cudnn'` in JAX) against `causal_attention` on $B = 2$
sequences, $H = 8$ heads, $d_h = 64$, and $n = 8{,}192$ in float16. Including
an unmasked fused run separates the $O(n^2)$ HBM-traffic speedup from the
$\sim 1.5\times$ speedup of skipping upper-triangular causal tiles:

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

The fused kernel reduces temporary memory from $\sim 4$ GiB ($16 \cdot 4n^2$
bytes) to a small multiple of the 16 MiB output and runs several times faster
close to the GPU's peak matrix throughput. Even with fused kernels, however,
exact attention still performs $\Theta(n^2 d)$ arithmetic operations;
achieving linear time complexity requires changing the attention mechanism
itself (:numref:`sec_windowed-linear-attention`).

## Summary

* Dense self-attention requires $8nd^2 + 4n^2d$ FLOPs per layer; the
  quadratic score-and-mix term dominates the projections once $n > 2d$.
* Materializing the $n \times n$ score and weight matrices consumes $8n^2$
  bytes per head in float32 and makes naive attention memory-bandwidth bound.
* *Online softmax* computes exact attention block by block by maintaining a
  running maximum $m$, normalizer $s$, and value accumulator $\mathbf{o}$,
  rescaling prior sums by $e^{m - m'}$ whenever $m$ increases.
* *FlashAttention* fuses tiled online softmax in on-chip SRAM, reducing
  memory to $O(n)$ and achieving large wall-clock speedups while leaving the
  $\Theta(n^2 d)$ FLOP count intact.

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

Attention at Scale: Memory Bandwidth, Online Softmax, and FlashAttention<br>**quadratic FLOP & memory scaling · arithmetic intensity · blockwise online softmax · SRAM tiling · FlashAttention**
:::
:::

::: {.slide title="Why Long-Context Attention Hits a Memory Wall"}
[Motivation]{.kicker}

Long-context language models process $32\text{K}$ to $128\text{K}+$ tokens in a single pass, where materializing an $n \times n$ score matrix per head exhausts GPU memory long before compute saturates:

| Context Length $n$ | Score Pairs $n^2$ per Head | One `float32` Score Matrix ($4n^2$ B) | Score + Weight Buffers ($8n^2$ B) | All 144 Heads of GPT-2 Small (Training) |
| ---: | ---: | ---: | ---: | ---: |
| **$1{,}024$ (GPT-2)** | $1.05 \times 10^6$ | $4.0\text{ MiB}$ | $8.0\text{ MiB}$ | $0.56\text{ GiB}$ |
| **$8{,}192$** | $6.71 \times 10^7$ | $256.0\text{ MiB}$ | $512.0\text{ MiB}$ | $36.0\text{ GiB}$ |
| **$131{,}072$ ($128\text{K}$)** | $1.72 \times 10^{10}$ | **$64.0\text{ GiB}$** | **$128.0\text{ GiB}$** | **$9.0\text{ TiB}$** |

::: {.d2l-note}
For years, $O(n^2)$ memory was thought unavoidable for exact attention. Online softmax (Milakov & Gimelshein, 2018; Rabe & Staats, 2021) and **FlashAttention** (Dao et al., 2022) compute *exact* attention in **$O(n)$ memory**.
:::
:::

::: {.slide title="Counting FLOPs: When Scores Overtake Projections"}
[Arithmetic Scaling]{.kicker}

A multi-head self-attention layer of width $d$ with $H$ heads of width $d_h = d/H$ over $n$ tokens executes:

$$\textrm{FLOPs} = \underbrace{8 n d^2}_{\textrm{Q, K, V, O projections}} \;+\; \underbrace{4 n^2 d}_{\textrm{scores } \mathbf{Q}\mathbf{K}^\top \textrm{ and mixing } \mathbf{A}\mathbf{V}} \;+\; O(H n^2)$$

- **The Crossover Ratio:** Dividing the quadratic score-and-mix work by the linear projection work gives:
  $$\frac{4 n^2 d}{8 n d^2} = \frac{n}{2d}$$
- **Short vs. Long Context Regimes:** In GPT-2 small ($d = 768, n = 1{,}024$), $n / (2d) = 0.67$, so linear projections still dominate; at $d = 4{,}096$ and $n = 131{,}072$, $n / (2d) = 16$, so attention scoring and value mixing cost **$16\times$** more FLOPs than all four projections combined!
:::

::: {.slide title="Counting Memory: The 8n²-Byte Scratch Footprint"}
[Memory Allocation]{.kicker}

Why does naive attention allocate $8n^2$ bytes of temporary memory per head in `float32`?

::: {.cols .vc}
::: {.col}
- **Two $n \times n$ Materializations:** `scores = Q @ K.T / sqrt(d_h)` allocates $4n^2$ bytes, and `softmax(scores)` allocates a second $4n^2$-byte buffer for the weights before multiplying by `V`.
- **Inputs Are Tiny by Comparison:** At $n = 8{,}192, d_h = 64$, the input tensors $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ occupy only $3 \times 8{,}192 \times 64 \times 4\text{ B} = 6\text{ MiB}$, while the two $n \times n$ buffers consume **$512\text{ MiB}$** ($85\times$ larger)!
:::

::: {.col .fig}
@attention-at-scale-counting-memory-1
:::
:::
:::

::: {.slide title="Measuring Quadratic Memory and Runtime on a GPU"}
[Empirical Benchmarks]{.kicker}

Measuring peak scratch memory (`d2l.peak_memory`) and wall-clock execution time (`d2l.Benchmark`) on a single `float32` head ($d_h = 64$) confirms the exact $8n^2$ byte footprint and $4\times$ time jump per doubling of $n$:

@!attention-at-scale-counting-memory-3

@!attention-at-scale-counting-memory-2
:::

::: {.slide title="Why Naive Softmax Seems to Require Two Passes"}
[The Two-Pass Bottleneck]{.kicker}

To compute row-wise softmax weights $\alpha_j = e^{a_j - m} / \sum_{j'=1}^n e^{a_{j'} - m}$ without floating-point overflow, standard softmax needs two global statistics over the entire row of $n$ keys:

$$m = \max_{1 \le j \le n} a_j, \qquad s = \sum_{j=1}^n \exp(a_j - m), \qquad \mathbf{o} = \frac{1}{s}\sum_{j=1}^n \exp(a_j - m)\,\mathbf{v}_j$$

- **Pass 1 (Row Maximum & Normalizer):** Scan all $n$ scores to find $m$ and accumulate the denominator $s$.
- **Pass 2 (Value Accumulation):** Divide each $\exp(a_j - m)$ by $s$ and accumulate $\sum_j \alpha_j \mathbf{v}_j$.
- **Why Must We Keep $m$?** Without subtracting $m$, $\exp(a_j)$ overflows `float16` above $a_j \approx 11.09$ and `float32` above $a_j \approx 88.7$.
- **The Online Insight:** We can process keys in blocks and *rescale* running sums whenever a new block contains a larger maximum $m' > m$!
:::

::: {.slide title="Deriving Blockwise Online Softmax"}
[Online Softmax Recurrence]{.kicker}

Maintain three running statistics per query across key blocks: the running maximum $m$, the unnormalized denominator sum $s = \sum_j e^{a_j - m}$, and the unnormalized output vector $\mathbf{o} = \sum_j e^{a_j - m}\mathbf{v}_j$. When a new block of keys $\{a_j, \mathbf{v}_j\}$ arrives:

$$m' = \max\!\big(m,\, \max_j a_j\big), \qquad s' = s\,\exp(m - m') + \sum_j \exp(a_j - m')$$

$$\mathbf{o}' = \mathbf{o}\,\exp(m - m') + \sum_j \exp(a_j - m')\,\mathbf{v}_j$$

- **Exact Multiplicative Repair:** Because $e^{a_{\textrm{old}} - m} \cdot e^{m - m'} = e^{a_{\textrm{old}} - m'}$, multiplying the old accumulators $s$ and $\mathbf{o}$ by $\exp(m - m') \le 1$ updates every earlier term to the new reference maximum $m'$ in $O(1)$ operations!
- **Final Output:** After visiting all key blocks, dividing $\mathbf{o} / s$ yields the **exact** attention output $\mathrm{softmax}(\mathbf{a})\mathbf{V}$.
:::

::: {.slide title="Worked Example: Online Softmax Across Two Blocks"}
[Step-by-Step Numerical Trace]{.kicker}

Trace one query against four keys arriving in two blocks of $c = 2$: Block 1 has scores $(1, 3)$ and Block 2 has scores $(5, 2)$:

| Statistic / Operation | After Block 1: scores $(1, 3)$ | Block 2 Arrives: scores $(5, 2)$ | Final Exact Verification |
| :--- | :--- | :--- | :--- |
| **Running maximum $m$** | $m_1 = \max(1, 3) = \mathbf{3}$ | $m_2 = \max(3, 5, 2) = \mathbf{5}$ | Global max $\max(1,3,5,2) = 5$ |
| **Rescale factor $e^{m - m'}$** | — | $e^{3 - 5} = e^{-2} \approx \mathbf{0.1353}$ | Shifts reference from $3 \to 5$ |
| **Rescaled old sum $s \cdot e^{m-m'}$** | $0$ | $(e^{-2} + e^0) \cdot e^{-2} = e^{-4} + e^{-2} \approx 0.1537$ | Exact Block-1 terms relative to $5$ |
| **New block contribution** | $e^{1-3} + e^{3-3} = e^{-2} + 1 \approx 1.1353$ | $e^{5-5} + e^{2-5} = 1 + e^{-3} \approx 1.0498$ | Exact Block-2 terms relative to $5$ |
| **Updated running sum $s$** | $s_1 \approx \mathbf{1.1353}$ | $s_2 = 0.1537 + 1.0498 \approx \mathbf{1.2034}$ | $e^{-4} + e^{-2} + 1 + e^{-3} \approx \mathbf{1.2034}$ |

::: {.d2l-note .rule}
Online softmax is an **associative, commutative reduction**: processing Block 2 before Block 1 keeps $m = 5$ ($e^{m-m'} = 1$) and yields the exact same sum $1.2034$.
:::
:::

::: {.slide title="Visualizing Striped Online Softmax Memory"}
[Linear Working Memory]{.kicker}

::: {.cols .vc}
::: {.col}
- **One Stripe at a Time:** Instead of materializing the $n \times n$ score matrix, we hold only one $n \times c$ score stripe for the current block of $c$ keys.
- **Running Query State:** Alongside the stripe, we store $\mathbf{m} \in \mathbb{R}^{n \times 1}$, $\mathbf{s} \in \mathbb{R}^{n \times 1}$, and $\mathbf{O} \in \mathbb{R}^{n \times d_v}$ ($O(n d_v)$ floats total).
- **Memory Reduction:** Peak score memory drops from $O(n^2)$ to $O(nc)$, which is **linear in $n$** for fixed block size $c$!
:::

::: {.col .fig}
![Online softmax sweeps an n × c stripe across key blocks while updating m, s, and O.](../img/mdl-attention-online-softmax.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Implementing Chunked Causal Attention"}
[Code Implementation]{.kicker}

`chunked_attention` steps through key–value blocks of width `chunk_size=512`, initializing `m` and masking future keys with `finfo(dtype).min` so fully masked future blocks never trigger $-\infty - (-\infty) = \mathrm{NaN}$:

@attention-at-scale-a-chunked-implementation-1
:::

::: {.slide title="Verifying Exactness and Linear Memory Growth"}
[Chunked vs. Full-Matrix Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
- **Exact to Machine Precision:** At $n = 2{,}048, d_h = 64$, `chunked_attention` matches `causal_attention` within $\sim 10^{-7}$ (`float32` rounding):

@!attention-at-scale-a-chunked-implementation-2

- **Quadratic $\to$ Linear Memory:** At $n = 16{,}384$, full-matrix attention consumes $2{,}048\text{ MiB}$ ($2\text{ GiB}$), whereas `chunked_attention` ($c = 512$) uses only $\approx 68\text{ MiB}$ ($30\times$ less)!
:::

::: {.col .fig}
@!attention-at-scale-a-chunked-implementation-3
:::
:::
:::

::: {.slide title="The Hardware Bottleneck: GPU HBM vs. On-Chip SRAM"}
[GPU Memory Hierarchy]{.kicker}

Why is a Python loop like `chunked_attention` still slow even though it uses linear memory? Because modern GPUs are **memory-bandwidth bound** on elementwise and softmax operations:

| Memory Tier / Compute Unit | Capacity (Typical GPU) | Bandwidth / Throughput | Role in Naive vs. Fused Attention |
| :--- | :--- | :--- | :--- |
| **Tensor Cores (FP16/BF16)** | — | $\sim 165$–$1{,}000\text{ TFLOP/s}$ | Executes $\mathbf{Q}\mathbf{K}^\top$ and $\mathbf{P}\mathbf{V}$ matrix multiplies |
| **On-Chip SRAM (Registers/L1)** | $\sim 128$–$256\text{ KiB}$ per SM ($\sim 20$–$50\text{ MiB}$ total) | **$\sim 19\text{ TB/s}$** ($10$–$20\times$ HBM) | Holds active tiles $(m, s, \mathbf{o}, \mathbf{Q}_i, \mathbf{K}_j, \mathbf{V}_j)$ in **FlashAttention** |
| **Off-Chip HBM (Main VRAM)** | $24$–$80\text{ GiB}$ | **$\sim 0.9$–$3.3\text{ TB/s}$** | Naive & Python-chunked attention read/write every score here! |

::: {.d2l-note}
At $165\text{ TFLOP/s}$ and $0.9\text{ TB/s}$ HBM bandwidth, the GPU can perform $\sim 180$ floating-point operations in the time it takes to transfer a single byte from HBM!
:::
:::

::: {.slide title="Arithmetic Intensity of Naive Attention"}
[Roofline Analysis]{.kicker}

Naive attention launches separate kernels that move $n \times n$ matrices back and forth across slow HBM:

- **Four HBM Passes Over $n^2$ Floats:** (1) Write score matrix $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top/\sqrt{d_h}$ to HBM; (2) read $\mathbf{S}$ from HBM for softmax; (3) write weight matrix $\mathbf{A}$ to HBM; (4) read $\mathbf{A}$ from HBM to multiply by $\mathbf{V}$—moving $4n^2$ floats ($8n^2$ bytes in 16-bit, $16n^2$ bytes in 32-bit).
- **Low Arithmetic Intensity:** Dividing total FLOPs $2n^2(d_h + d_v)$ by HBM bytes moved $4n^2 \cdot \text{bytes\_per\_float}$ gives:
  $$\textrm{Arithmetic Intensity} \;\xrightarrow{\;n \to \infty\;}\; \frac{d_h + d_v}{8}\;\textrm{FLOPs/byte (fp32)} \quad\textrm{or}\quad \frac{d_h + d_v}{4}\;\textrm{FLOPs/byte (fp16)}$$
- **Memory-Bound Regime:** For $d_h = d_v = 64$ in 16-bit, intensity is only $32\text{ FLOPs/byte}$—well below the $\sim 180$–$300\text{ FLOPs/byte}$ ridge point of modern Tensor Core GPUs.
:::

::: {.slide title="FlashAttention: Tiling Queries and Keys in SRAM"}
[IO-Aware Kernel Fusion]{.kicker}

**FlashAttention** (Dao et al., 2022) fuses $\mathbf{Q}\mathbf{K}^\top$, masking, online softmax, and $\mathbf{P}\mathbf{V}$ into a *single* GPU kernel that tiles both queries and keys of block size $b$:

- **Zero HBM Score Traffic:** An outer loop loads a query tile $\mathbf{Q}_i \in \mathbb{R}^{b \times d_h}$ into fast on-chip SRAM along with $(m_i, s_i, \mathbf{O}_i)$; an inner loop streams key–value tiles $(\mathbf{K}_j, \mathbf{V}_j)$ from HBM, updates $(m_i, s_i, \mathbf{O}_i)$ entirely in SRAM, and writes $\mathbf{O}_i / s_i$ to HBM only once at the end.
- **HBM Traffic Bound:** Choosing the largest tile $b \propto M / d_h$ that fits in SRAM of size $M$ cuts quadratic HBM traffic from $4n^2$ floats down to $\Theta(n^2 d_h^2 / M)$ floats (a $4\times$ to $16\times$ traffic reduction).
- **$O(n)$ Backward Pass via Log-Normalizer Recomputation:** Instead of saving the $n \times n$ attention matrix for backpropagation, the forward pass saves only the single scalar $\ell_i = m_i + \log s_i$ per query ($O(n)$ memory) and reconstructs weights $\alpha_{ij} = \exp(a_{ij} - \ell_i)$ on chip during the backward pass!
:::

::: {.slide title="Benchmarking Fused FlashAttention vs. Naive Attention"}
[Code & Wall-Clock Speedup]{.kicker}

We benchmark `causal_attention` against the framework's fused FlashAttention kernel on $B = 2$ sequences, $H = 8$ heads, $d_h = 64$, and $n = 8{,}192$ in `float16`:

@attention-at-scale-the-bottleneck-is-memory-traffic
:::

::: {.slide title="Analyzing the Fused Kernel Speedup and Causal Skipping"}
[Dissecting the Benchmark]{.kicker}

Comparing `naive`, `fused` (causal), and `fused, no mask` separates two distinct sources of speedup:

- **1. Eliminating HBM Score Traffic (`naive` $\to$ `fused, no mask`):** Cuts peak temporary memory from $\sim 4{,}096\text{ MiB}$ ($16 \times 4n^2$ bytes) down to the $16\text{ MiB}$ output tensor ($> 200\times$ memory reduction) and speeds up execution several-fold by running near peak Tensor Core throughput ($2.7 \times 10^{11}\text{ FLOPs}$).
- **2. Causal Tile Skipping (`fused, no mask` $\to$ `fused`):** Because a causal mask zeroes all entries above the diagonal ($j > i$), a tiled kernel skips the entire upper triangle of empty tiles without computing them.
- **Why Causal Skipping Gives $\sim 1.5\times$ Instead of $2\times$:** Diagonal tiles straddle $j = i$ and must apply elementwise masking, and thread-block scheduling across variable-length causal rows incurs small load-balancing overhead.
:::

::: {.slide title="What Fused Kernels Solve—and What Remains O(n²)"}
[Architectural Perspective]{.kicker}

FlashAttention is now the default implementation of exact attention in every modern LLM training and inference stack, but it is crucial to distinguish what it changes from what it leaves intact:

| Resource / Property | Naive Materialized Attention | Python `chunked_attention` | Fused **FlashAttention** |
| :--- | :--- | :--- | :--- |
| **Mathematical Function** | Exact $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_h})\mathbf{V}$ | Exact (up to fp round-off) | Exact (up to fp round-off) |
| **Peak Working Memory** | $\Theta(n^2)$ ($8n^2$ B in fp32) | $\Theta(nc)$ (linear in $n$) | **$\Theta(n d_h)$ (linear in $n$)** |
| **Quadratic HBM Traffic** | $4n^2$ floats moved | $2n^2 + O(n^2 d_h / c)$ floats | **$\Theta(n^2 d_h^2 / M)$ floats** (in SRAM) |
| **Total Arithmetic FLOPs** | $\Theta(n^2 d_h)$ | $\Theta(n^2 d_h)$ | **$\Theta(n^2 d_h)$ ($4\times$ per $2\times$ length)** |

::: {.d2l-note}
Removing the $\Theta(n^2 d)$ arithmetic itself requires changing the attention function—via **sliding windows** or **linear attention** (§10.6).
:::
:::

::: {.slide title="Concept Check: Online Softmax and Memory Traffic"}
[Check Your Understanding]{.kicker}

Work through these three systems and algorithmic questions:

1. **Log-Sum-Exp Backward Reconstruction:** After the final block of online softmax produces $(m, s, \mathbf{o})$, how can the backward pass recover the exact softmax weight $\alpha_j$ of key $j$ from its score $a_j$ and the single stored scalar $\ell = m + \log s$?
2. **Float16 Overflow Without Running Max:** In `float16` (max finite value $65{,}504 \approx e^{11.09}$), what happens if you accumulate raw $\sum_j e^{a_j}$ without subtracting the running maximum $m$ when a scaled score reaches $a_j = 12$?
3. **Tile Size vs. Head Width:** Using FlashAttention's HBM traffic bound $\Theta(n^2 d_h^2 / M)$ for on-chip SRAM size $M$, how does doubling the head width from $d_h = 64$ to $d_h = 128$ (at fixed total width $d$) affect the maximum tile size $b$ and HBM traffic?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **$\alpha_j = \exp(a_j - \ell)$**, because $\exp(a_j - m - \log s) = \exp(a_j - m) / s = \alpha_j$, requiring only $1$ float per query instead of $(m, s)$ or all $n$ weights!
2. **Immediate overflow to `inf`**, followed by `inf / inf = NaN` in the output normalizer. Carrying $m$ guarantees $a_j - m' \le 0$ so every exponential lies in $(0, 1]$.
3. **Tile size $b \le M / (4d_h)$ halves**, so twice as many passes over $\mathbf{K}, \mathbf{V}$ are needed and quadratic HBM traffic increases (by $2\times$ per head, or $4\times$ at fixed head count).
:::
:::

::: {.slide title="Summary: The Computational Cost of Attention"}
[Key Takeaways]{.kicker}

| Concept / Algorithm | Mathematical Formula / Bound | Scaling with Length $n$ | Systems Impact |
| :--- | :--- | :--- | :--- |
| **Layer FLOP Count** | $8nd^2 + 4n^2d$ | Crossover at $n = 2d$ ($\approx 6d$ with FFN) | Projections dominate short contexts; scores dominate long contexts |
| **Naive Scratch Memory** | $2 \times n^2 \times \text{sizeof(float)}$ per head | $\Theta(n^2)$ ($64\text{ GiB}$ at $n=128\text{K}$, fp32) | Exhausts GPU VRAM at long contexts before compute saturates |
| **Arithmetic Intensity** | $\to \dfrac{d_h + d_v}{4}$ FLOPs/byte (fp16) | $O(1)$ independent of $n$ | Naive attention is severely HBM-bandwidth bound |
| **Online Softmax Update** | $s' = s e^{m-m'} + \sum_j e^{a_j-m'}$ | Processes $n \times c$ stripes in $O(nc)$ memory | Exact one-pass softmax via $e^{m-m'}$ rescaling |
| **Parallel State Merge** | Associative & commutative on $(m_B, s_B, \mathbf{o}_B)$ | $O(\log(n/c))$ parallel tree depth | Enables FlashDecoding across long KV caches during generation |
| **FlashAttention** | Tiled SRAM loop + save $\ell = m + \log s$ | **$\Theta(n)$ memory**, $\Theta(n^2 d_h^2/M)$ HBM IO | Default exact attention kernel; $3$–$10\times$ faster, zero $n \times n$ storage |
:::
