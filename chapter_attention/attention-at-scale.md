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
| Windowed and Linear Attention | linear cost, and what it gives up |
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
