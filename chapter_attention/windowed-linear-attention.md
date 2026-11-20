# Windowed and Linear Attention
:label:`sec_windowed-linear-attention`

Fused kernels (:numref:`sec_attention-at-scale`) eliminate the $O(n^2)$
intermediate score matrix, yet both their $\Theta(n^2 d)$ FLOP count and
their key--value cache still grow with sequence length $n$. When processing a
100,000-token document, however, most tokens depend primarily on local
context, while only occasional lookups require long-range retrieval.

Two primary architectural strategies reduce attention's computational cost to
$O(n)$:

1. **Sliding-window attention** :cite:`child2019generating,beltagy2020longformer`
   restricts each query to its $w$ most recent keys, reducing per-layer work
   to $O(nw)$ and bounding the key--value cache at $w$ tokens.
2. **Linear attention** :cite:`Katharopoulos.Vyas.Pappas.ea.2020` replaces
   the softmax kernel $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ with a
   factorized kernel $\phi(\mathbf{q})^\top \phi(\mathbf{k})$, allowing keys
   and values to be accumulated into a fixed-size $d_h \times d_v$ recurrent
   state matrix.

Both approaches compute a different function from dense softmax attention.
This section analyzes how sliding windows trade single-hop reach for tile
sparsity, derives the parallel, recurrent, and chunked forms of linear
attention, and demonstrates why factorized feature maps lack the sharpening
temperature of softmax.

```{.python .input #windowed-linear-attention-windowed-and-linear-attention-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #windowed-linear-attention-windowed-and-linear-attention-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #windowed-linear-attention-windowed-and-linear-attention-2}
%%tab pytorch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
```

## Sliding-Window Attention

### The Band Mask

In causal *sliding-window attention* of width $w$, query $i$ attends to key
$j$ only if $j \le i$ and $i - j < w$. This conjunction of causality and
locality replaces the lower triangle of $n(n+1)/2$ active pairs with a
diagonal band of at most $nw$ pairs:

```{.python .input #attention-at-scale-attention-through-a-window}
%%tab pytorch
def window_mask(n, w, device=None):
    """Causal band: query i may attend to keys i - w + 1, ..., i."""
    i = torch.arange(n, device=device)
    return d2l.causal_mask(n, device=device) & (i[:, None] - i[None, :] < w)
```

```{.python .input #attention-at-scale-attention-through-a-window}
%%tab jax
def window_mask(n, w):
    """Causal band: query i may attend to keys i - w + 1, ..., i."""
    i = jnp.arange(n)
    return d2l.causal_mask(n) & (i[:, None] - i[None, :] < w)
```

### Receptive Field Growth with Depth
:label:`subsec_attn-receptive-field`

Just as stacked convolutions expand their receptive field with depth
(:numref:`chap_cnn`), stacking $L$ sliding-window layers of width $w$ allows
information to propagate $w - 1$ steps backward per layer, yielding a
theoretical receptive field of

$$1 + L(w - 1)$$

tokens. For example, Mistral 7B :cite:`Jiang.Sablayrolles.Mensch.ea.2023`
uses $w = 4{,}096$ across $L = 32$ layers, spanning up to $131{,}041$
tokens. However, theoretical reach differs from direct retrieval: whereas
full self-attention connects any two tokens in a single hop, a stack of
windows requires $\lceil \Delta / (w - 1) \rceil$ sequential hops across
layers to relay a fact across distance $\Delta$. Under uniform window
averaging ($w = 4$, $L = 4$), a token at the boundary $\Delta = 12$ arrives
 along a single four-hop path with weight $(1/4)^4 = 1/256$, compared to
$1/13$ in a single global layer.

### Global Layers

To preserve one-hop long-range retrieval while retaining most of the memory
and compute savings of local windows, modern long-context architectures
interleave $r$ windowed layers with one full (*global*) attention layer—for
example, Gemma 3 uses $r = 5$ windowed layers ($w = 1{,}024$) per global
layer :cite:`Gemma.Team.2025`. Because a windowed layer caches only the last
$\min(n, w)$ keys and values during generation (:numref:`sec_kv-cache`),
interleaving reduces the total KV cache to the fraction

$$
\frac{r \min(n, w) + n}{(r + 1)\, n} \;\xrightarrow{n \gg w}\; \frac{1}{r + 1}
$$

of an all-global stack ($\approx 27\%$ at $n = 8{,}192$ and $1/6$
asymptotically for $r = 5, w = 1{,}024$).

### Tiles in a Fused Kernel

Applying `window_mask` inside a dense $n \times n$ matrix multiplication
still computes all $n^2$ scores. To realize $O(nw)$ time and memory, an
implementation must evaluate only the active band—for instance, by grouping
queries into blocks of size $w$ and attending only to the current and
immediately preceding key blocks ($2nw$ scores total):

```{.python .input #attention-at-scale-a-linear-cost-implementation-1}
%%tab pytorch
def windowed_attention(Q, K, V, w):
    """Causal sliding-window attention in O(nw) time and memory."""
    n, d_h = Q.shape
    assert n % w == 0, 'blocks must tile the sequence exactly'
    Qb, Kb, Vb = (X.reshape(n // w, w, -1) for X in (Q, K, V))
    # Each query block attends to its own key block and the one before it
    prev = lambda X: torch.cat([torch.zeros_like(X[:1]), X[:-1]])
    Kb, Vb = torch.cat([prev(Kb), Kb], 1), torch.cat([prev(Vb), Vb], 1)
    i = torch.arange(w, device=Q.device)[:, None]      # Query in its block
    j = torch.arange(-w, w, device=Q.device)[None, :]  # Key offset from it
    first = torch.arange(n // w, device=Q.device)[:, None, None] == 0
    mask = (j <= i) & (i - j < w) & ~(first & (j < 0))  # No keys before 0
    scores = Qb @ Kb.transpose(-1, -2) / math.sqrt(d_h)
    return (d2l.masked_softmax(scores, mask) @ Vb).reshape(n, -1)
```

```{.python .input #attention-at-scale-a-linear-cost-implementation-1}
%%tab jax
def windowed_attention(Q, K, V, w):
    """Causal sliding-window attention in O(nw) time and memory."""
    n, d_h = Q.shape
    assert n % w == 0, 'blocks must tile the sequence exactly'
    Qb, Kb, Vb = (X.reshape(n // w, w, -1) for X in (Q, K, V))
    # Each query block attends to its own key block and the one before it
    prev = lambda X: jnp.concatenate([jnp.zeros_like(X[:1]), X[:-1]])
    Kb = jnp.concatenate([prev(Kb), Kb], 1)
    Vb = jnp.concatenate([prev(Vb), Vb], 1)
    i = jnp.arange(w)[:, None]                         # Query in its block
    j = jnp.arange(-w, w)[None, :]                     # Key offset from it
    first = jnp.arange(n // w)[:, None, None] == 0
    mask = (j <= i) & (i - j < w) & ~(first & (j < 0))  # No keys before 0
    scores = Qb @ Kb.swapaxes(-1, -2) / math.sqrt(d_h)
    return (d2l.masked_softmax(scores, mask) @ Vb).reshape(n, -1)
```

In a tiled GPU kernel such as FlashAttention or FlexAttention
:cite:`Dong.Feng.Guessous.ea.2025`, computation proceeds in $b \times b$
tiles (typically $b = 128$), and a tile is skipped only when *all* $b^2$ of
its entries are masked out. At $n = 2{,}048$ and $b = w = 128$, the causal
mask visits $T(T+1)/2$ of the $T^2$ tiles ($53.1\%$ for $T = n/b = 16$),
whereas the sliding window visits only the diagonal tile and its left
neighbor in each tile row ($2T - 1 = 31$ tiles, or $12.1\%$):

```{.python .input #attention-at-scale-a-linear-cost-implementation-3}
%%tab pytorch
def tiles_visited(mask, b):
    """Fraction of b-by-b tiles of a mask that hold at least one True."""
    T = mask.shape[0] // b
    return mask.reshape(T, b, T, b).any(dim=(1, 3)).float().mean().item()

n, b, w = 2048, 128, 128
print(f'causal:  {tiles_visited(d2l.causal_mask(n), b):.1%} of tiles')
print(f'window:  {tiles_visited(window_mask(n, w), b):.1%} of tiles')
```

```{.python .input #attention-at-scale-a-linear-cost-implementation-3}
%%tab jax
def tiles_visited(mask, b):
    """Fraction of b-by-b tiles of a mask that hold at least one True."""
    T = mask.shape[0] // b
    return float(mask.reshape(T, b, T, b).any(axis=(1, 3)).mean())

n, b, w = 2048, 128, 128
print(f'causal:  {tiles_visited(d2l.causal_mask(n), b):.1%} of tiles')
print(f'window:  {tiles_visited(window_mask(n, w), b):.1%} of tiles')
```

Both `windowed_attention` and the framework's fused sliding-window kernel
(`flex_attention` with `create_block_mask` in PyTorch;
`dot_product_attention` with `local_window_size=(w - 1, 0)` in JAX) match the
dense masked definition up to float32 round-off:

```{.python .input #attention-at-scale-a-linear-cost-implementation-2}
%%tab pytorch
n, w, d_h = 2048, 128, 64
Q, K, V = (torch.randn(n, d_h, device=d2l.try_gpu()) for _ in range(3))
reference = d2l.masked_softmax(Q @ K.T / math.sqrt(d_h),
                               window_mask(n, w, Q.device)) @ V
# create_block_mask evaluates the band at every position and records which
# tiles are empty, full or cut by it; compiled, the kernel skips the empty ones
band = lambda batch, head, q, k: (k <= q) & (q - k < w)
block_mask = create_block_mask(band, None, None, n, n, device=Q.device)
fused = torch.compile(flex_attention)(Q[None, None], K[None, None],
                                      V[None, None], block_mask=block_mask)
for name, O in (('blocked', windowed_attention(Q, K, V, w)),
                ('fused kernel', fused[0, 0])):
    print(f'{name:>12}: maximum deviation '
          f'{float((O - reference).abs().max()):.2e}')
```

```{.python .input #attention-at-scale-a-linear-cost-implementation-2}
%%tab jax
n, w, d_h = 2048, 128, 64
Q, K, V = (jax.random.normal(k, (n, d_h))
           for k in jax.random.split(d2l.get_key(), 3))
with jax.default_matmul_precision('highest'):
    reference = d2l.masked_softmax(Q @ K.T / math.sqrt(d_h),
                                   window_mask(n, w)) @ V
    # The window as an argument: w - 1 keys back, none ahead. The default
    # XLA implementation runs in float32; cuDNN's needs 16-bit inputs
    fused = jax.nn.dot_product_attention(
        Q[None, :, None], K[None, :, None], V[None, :, None],
        is_causal=True, local_window_size=(w - 1, 0))
    for name, O in (('blocked', windowed_attention(Q, K, V, w)),
                    ('fused kernel', fused[0, :, 0])):
        print(f'{name:>12}: maximum deviation '
              f'{float(jnp.abs(O - reference).max()):.2e}')
```

### Windowed and Dense Kernels Compared

With $T = n / 128$ tile rows, the ratio of active causal tiles to active
windowed tiles ($w = b = 128$) is $T(T + 1) / (2(2T - 1)) \approx T/4$,
predicting a $64\times$ theoretical tile reduction at $n = 32{,}768$ ($T =
256$). We benchmark both masks inside the same fused bfloat16 kernel across
$n \in [512, 32{,}768]$ ($H = 32$ heads, $d_h = 64$):

```{.python .input #attention-at-scale-the-price-of-attention-measured}
%%tab pytorch
flex = torch.compile(flex_attention)
w = b = 128  # Window width; tile size of the kernel
masks = {'dense causal': lambda batch, head, q, k: k <= q,
         f'window (w={w})': lambda batch, head, q, k: (k <= q) & (q - k < w)}
H, lengths = 32, [512, 1024, 2048, 4096, 8192, 16384, 32768]
times = {name: [] for name in masks}
for n in lengths:
    X = torch.randn(1, H, n, d_h, device=d2l.try_gpu(), dtype=torch.bfloat16)
    for name, rule in masks.items():
        block_mask = create_block_mask(rule, None, None, n, n, device=X.device)
        times[name].append(d2l.Benchmark(
            lambda: flex(X, X, X, block_mask=block_mask)).time * 1e3)
    T = n // b  # Rows of b x b tiles; causal T(T+1)/2, window 2T - 1
    dense, window = (t[-1] for t in times.values())
    print(f'n = {n:6,}: window {dense / window:4.1f}x faster, '
          f'tile count predicts {T * (T + 1) / 2 / (2 * T - 1):4.1f}x')
d2l.plot(lengths, list(times.values()), 'sequence length n', 'time (ms)',
         legend=list(times), xscale='log', yscale='log', figsize=(4.5, 3))
```

```{.python .input #attention-at-scale-the-price-of-attention-measured}
%%tab jax
# cuDNN takes (batch, length, heads, width); it skips the tiles outside the
# band only together with is_causal=True
w = b = 128  # Window width; tile size of the kernel
masks = {'dense causal': {},
         f'window (w={w})': {'local_window_size': (w - 1, 0)}}
kernels = {name: jax.jit(lambda X, kw=kw: jax.nn.dot_product_attention(
    X, X, X, is_causal=True, implementation='cudnn', **kw))
    for name, kw in masks.items()}
H, lengths = 32, [512, 1024, 2048, 4096, 8192, 16384, 32768]
times = {name: [] for name in masks}
for n in lengths:
    X = jax.random.normal(d2l.get_key(), (1, n, H, d_h), dtype=jnp.bfloat16)
    for name, f in kernels.items():
        times[name].append(d2l.Benchmark(lambda: f(X)).time * 1e3)
    T = n // b  # Rows of b x b tiles; causal T(T+1)/2, window 2T - 1
    dense, window = (t[-1] for t in times.values())
    print(f'n = {n:6,}: window {dense / window:4.1f}x faster, '
          f'tile count predicts {T * (T + 1) / 2 / (2 * T - 1):4.1f}x')
d2l.plot(lengths, list(times.values()), 'sequence length n', 'time (ms)',
         legend=list(times), xscale='log', yscale='log', figsize=(4.5, 3))
```

Beyond $n \approx 2{,}048$, dense causal runtime scales quadratically while
windowed runtime scales linearly, reaching a $> 30\times$ speedup at $n =
32{,}768$ (slightly below the $64\times$ raw tile ratio because every active
tile in a width-$b$ band straddles the mask boundary and incurs elementwise
masking overhead, whereas interior tiles of the causal triangle run
unmasked).

## Linear Attention

### Kernelizing the Score

Rather than restricting which pairs are scored, *linear attention*
:cite:`Katharopoulos.Vyas.Pappas.ea.2020` replaces the exponential similarity
$\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ with a factorized kernel
$K(\mathbf{q}, \mathbf{k}) = \phi(\mathbf{q})^\top \phi(\mathbf{k})$ using a
nonnegative elementwise feature map $\phi: \mathbb{R}^{d_h} \to
\mathbb{R}_+^{d_h}$. By the associativity of matrix multiplication,

$$
\bigl(\phi(\mathbf{Q})\,\phi(\mathbf{K})^\top\bigr)\,\mathbf{V}
= \phi(\mathbf{Q})\,\bigl(\phi(\mathbf{K})^\top \mathbf{V}\bigr),
$$

avoiding the $n \times n$ matrix $\phi(\mathbf{Q})\phi(\mathbf{K})^\top$ in
favor of a $d_h \times d_v$ summary matrix. Under a causal mask ($s \le t$),
factoring $\phi(\mathbf{q}_t)^\top$ out of the numerator and denominator sums
yields

$$
\mathbf{o}_t^\top
= \frac{\sum_{s \leq t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_s)\,
        \mathbf{v}_s^\top}
       {\sum_{s \leq t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_s)}
= \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}
       {\phi(\mathbf{q}_t)^\top \mathbf{z}_t},
\qquad
\mathbf{S}_t = \sum_{s \leq t} \phi(\mathbf{k}_s) \mathbf{v}_s^\top,
\quad
\mathbf{z}_t = \sum_{s \leq t} \phi(\mathbf{k}_s).
$$
:eqlabel:`eq_linear-attn`

Following :citet:`Katharopoulos.Vyas.Pappas.ea.2020`, we choose $\phi(x) =
\mathrm{elu}(x) + 1 > 0$ elementwise (with a small $\epsilon = 10^{-6}$ in
the denominator against float32 underflow). The prefix sums $(\mathbf{S}_t,
\mathbf{z}_t)$ satisfy the linear recurrence

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\, \mathbf{v}_t^\top,
\qquad
\mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t),
$$
:eqlabel:`eq_linear-attn-recurrence`

turning causal attention into an RNN whose hidden state is a $d_h \times d_v$
associative memory matrix updated by outer products (a *fast weight
programmer*, :cite:`Schlag.Irie.Schmidhuber.2021`). We can evaluate
:eqref:`eq_linear-attn` either in parallel via cumulative sums (`cumsum`) or
recurrently step by step in $O(d_h d_v)$ memory:

```{.python .input #attention-at-scale-kernelizing-the-score-1}
%%tab pytorch
def phi(X):
    return F.elu(X) + 1

def linear_attention_parallel(Q, K, V, eps=1e-6):
    """Training form: all outputs at once via cumulative sums."""
    Qp, Kp = phi(Q), phi(K)
    S = torch.cumsum(Kp[:, :, None] * V[:, None, :], dim=0)  # (n, d_h, d_v)
    z = torch.cumsum(Kp, dim=0)                              # (n, d_h)
    den = (Qp * z).sum(-1, keepdim=True) + eps
    return (Qp[:, :, None] * S).sum(1) / den

def linear_attention_recurrent(Q, K, V, eps=1e-6):
    """Generation form: a (d_h, d_v) state, updated token by token."""
    Qp, Kp = phi(Q), phi(K)
    S = torch.zeros(Q.shape[-1], V.shape[-1], device=Q.device)
    z = torch.zeros(Q.shape[-1], device=Q.device)
    outputs = []
    for t in range(Q.shape[0]):
        S = S + Kp[t][:, None] * V[t][None, :]   # Outer-product write
        z = z + Kp[t]
        outputs.append(Qp[t] @ S / (Qp[t] @ z + eps))  # Query the memory
    return torch.stack(outputs)
```

```{.python .input #attention-at-scale-kernelizing-the-score-1}
%%tab jax
def phi(X):
    return jax.nn.elu(X) + 1

def linear_attention_parallel(Q, K, V, eps=1e-6):
    """Training form: all outputs at once via cumulative sums."""
    Qp, Kp = phi(Q), phi(K)
    S = jnp.cumsum(Kp[:, :, None] * V[:, None, :], axis=0)  # (n, d_h, d_v)
    z = jnp.cumsum(Kp, axis=0)                              # (n, d_h)
    den = (Qp * z).sum(-1, keepdims=True) + eps
    return (Qp[:, :, None] * S).sum(1) / den

def linear_attention_recurrent(Q, K, V, eps=1e-6):
    """Generation form: a (d_h, d_v) state, updated token by token."""
    Qp, Kp = phi(Q), phi(K)
    def step(carry, qkv):
        S, z = carry
        q, k, v = qkv
        S = S + k[:, None] * v[None, :]          # Outer-product write
        z = z + k
        return (S, z), q @ S / (q @ z + eps)     # Query the memory
    init = (jnp.zeros((Q.shape[-1], V.shape[-1])), jnp.zeros(Q.shape[-1]))
    _, outputs = jax.lax.scan(step, init, (Qp, Kp, V))
    return outputs
```

### Chunked Computation

During training on GPUs, materializing all $n$ intermediate matrices
$\mathbf{S}_t \in \mathbb{R}^{d_h \times d_v}$ in `linear_attention_parallel`
consumes $O(n d_h d_v)$ memory (256 MiB per head at $n = 16{,}384, d_h = d_v
= 64$), whereas `linear_attention_recurrent` executes $n$ sequential vector
steps that underutilize GPU matrix cores. Partitioning the sequence into
chunks of length $c$ :cite:`Yang.Wang.Shen.ea.2024` combines both:for a
chunk starting at $t_0$ with pre-chunk state $(\mathbf{S}, \mathbf{z})$,

$$
\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}
  + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)\,
    \mathbf{v}_s^\top}
  {\phi(\mathbf{q}_t)^\top \mathbf{z}
  + \sum_{t_0 \le s \le t} \phi(\mathbf{q}_t)^\top\phi(\mathbf{k}_s)},
$$
:eqlabel:`eq_linear-attn-chunked`

so both intra-chunk attention ($c \times c$) and inter-chunk state propagation
use dense matrix multiplications in $O(nc(d_h + d_v) + n d_h d_v)$ FLOPs and
store only $n/c$ chunk states:

```{.python .input #attention-at-scale-chunked-computation-1}
%%tab pytorch
def linear_attention_chunked(Q, K, V, c=64, eps=1e-6):
    """Training form in chunks: quadratic within, recurrent across."""
    n, d_h = Q.shape
    assert n % c == 0, 'chunks must tile the sequence exactly'
    Qp, Kp = phi(Q).reshape(n // c, c, d_h), phi(K).reshape(n // c, c, d_h)
    Vc = V.reshape(n // c, c, -1)
    # The state each chunk starts from: sums over all earlier chunks
    S = torch.cumsum(Kp.transpose(-1, -2) @ Vc, dim=0)   # (n/c, d_h, d_v)
    z = torch.cumsum(Kp.sum(1), dim=0)                   # (n/c, d_h)
    S = torch.cat([torch.zeros_like(S[:1]), S[:-1]])
    z = torch.cat([torch.zeros_like(z[:1]), z[:-1]])
    A = torch.tril(Qp @ Kp.transpose(-1, -2))            # (n/c, c, c)
    num = Qp @ S + A @ Vc                                # Past + this chunk
    den = Qp @ z[..., None] + A.sum(-1, keepdim=True) + eps
    return (num / den).reshape(n, -1)
```

```{.python .input #attention-at-scale-chunked-computation-1}
%%tab jax
def linear_attention_chunked(Q, K, V, c=64, eps=1e-6):
    """Training form in chunks: quadratic within, recurrent across."""
    n, d_h = Q.shape
    assert n % c == 0, 'chunks must tile the sequence exactly'
    Qp, Kp = phi(Q).reshape(n // c, c, d_h), phi(K).reshape(n // c, c, d_h)
    Vc = V.reshape(n // c, c, -1)
    # The state each chunk starts from: sums over all earlier chunks
    S = jnp.cumsum(Kp.swapaxes(-1, -2) @ Vc, axis=0)     # (n/c, d_h, d_v)
    z = jnp.cumsum(Kp.sum(1), axis=0)                    # (n/c, d_h)
    S = jnp.concatenate([jnp.zeros_like(S[:1]), S[:-1]])
    z = jnp.concatenate([jnp.zeros_like(z[:1]), z[:-1]])
    A = jnp.tril(Qp @ Kp.swapaxes(-1, -2))               # (n/c, c, c)
    num = Qp @ S + A @ Vc                                # Past + this chunk
    den = Qp @ z[..., None] + A.sum(-1, keepdims=True) + eps
    return (num / den).reshape(n, -1)
```

All three forms compute the exact same outputs up to float32 rounding, and
the recurrent state occupies a constant 16 KiB per head ($64 \times 64 + 64$
floats) regardless of sequence length:

```{.python .input #attention-at-scale-kernelizing-the-score-2}
%%tab pytorch
n = 512
Q, K, V = (torch.randn(n, d_h, device=d2l.try_gpu()) for _ in range(3))
recurrent = linear_attention_recurrent(Q, K, V)
for name, f in (('parallel', linear_attention_parallel),
                ('chunked', linear_attention_chunked)):
    err = (f(Q, K, V) - recurrent).abs().max()
    print(f'{name:>8}: maximum deviation {float(err):.2e}')
print(f'recurrent state: {d_h}x{d_h} + {d_h} floats '
      f'= {(d_h * d_h + d_h) * 4 / 1024:.0f} KiB at any sequence length')
```

```{.python .input #attention-at-scale-kernelizing-the-score-2}
%%tab jax
n = 512
Q, K, V = (jax.random.normal(k, (n, d_h))
           for k in jax.random.split(d2l.get_key(), 3))
with jax.default_matmul_precision('highest'):
    recurrent = linear_attention_recurrent(Q, K, V)
    for name, f in (('parallel', linear_attention_parallel),
                    ('chunked', linear_attention_chunked)):
        err = jnp.abs(f(Q, K, V) - recurrent).max()
        print(f'{name:>8}: maximum deviation {float(err):.2e}')
print(f'recurrent state: {d_h}x{d_h} + {d_h} floats '
      f'= {(d_h * d_h + d_h) * 4 / 1024:.0f} KiB at any sequence length')
```

### Capacity and Sharpness

Compressing an arbitrary number of key--value pairs into a $d_h \times d_v$
state matrix incurs two fundamental costs:

1. **Finite associative capacity:** Because $\phi(\mathbf{k}_s) \in
   \mathbb{R}_+^{d_h}$ is nonnegative, a query can isolate a single stored
   value without cross-talk only if key feature vectors are nearly orthogonal
   (nearly disjoint supports), limiting exact retrieval to at most $d_h$
   keys.
2. **Lack of temperature sharpening:** In softmax attention, scaling a query
   by $\beta > 1$ exponentiates score differences ($e^{\beta \Delta}$),
   driving the distribution toward a one-hot lookup as $\beta \to \infty$. In
   linear attention with $\phi(x) = \mathrm{elu}(x) + 1$, scaling $\mathbf{q}$
   by large $\beta$ yields $\phi(\beta \mathbf{q}) \approx \beta\,
   \mathrm{relu}(\mathbf{q})$, so the scale factor $\beta$ cancels out between
   numerator and denominator (Exercise
   :exref:`ex_attention-at-scale-feature-map`):

```{.python .input #windowed-linear-attention-sharpening}
%%tab pytorch
Q = torch.randn(1000, d_h, device=d2l.try_gpu())
K = torch.randn(8, d_h, device=d2l.try_gpu())
for beta in (1, 4, 16):  # Scale the queries up; 8 keys: uniform is 0.125
    soft = torch.softmax(beta * Q @ K.T / math.sqrt(d_h), dim=-1)
    lin = phi(beta * Q) @ phi(K).T
    lin = lin / lin.sum(-1, keepdim=True)
    print(f'queries x{beta:<2}: mean largest weight, softmax '
          f'{soft.amax(-1).mean():.2f}, elu+1 {lin.amax(-1).mean():.2f}')
```

```{.python .input #windowed-linear-attention-sharpening}
%%tab jax
kq, kk = jax.random.split(d2l.get_key())
Q, K = jax.random.normal(kq, (1000, d_h)), jax.random.normal(kk, (8, d_h))
for beta in (1, 4, 16):  # Scale the queries up; 8 keys: uniform is 0.125
    soft = jax.nn.softmax(beta * Q @ K.T / math.sqrt(d_h), axis=-1)
    lin = phi(beta * Q) @ phi(K).T
    lin = lin / lin.sum(-1, keepdims=True)
    print(f'queries x{beta:<2}: mean largest weight, softmax '
          f'{soft.max(-1).mean():.2f}, elu+1 {lin.max(-1).mean():.2f}')
```

## Choosing a Mechanism

* **Exact fused attention** is the default for short-to-medium contexts,
  providing exact one-hop retrieval in $O(n)$ working memory.
* **Interleaved sliding-window and global layers** are the standard for
  long-context transformers (e.g., Gemma 3), cutting KV-cache memory and tile
  FLOPs on local layers while preserving one-hop global routing on periodic
  global layers.
* **Linear attention** requires data-dependent decay gates (so old entries
  can be forgotten; see Exercise :exref:`ex_attention-at-scale-decaying-state`
  and :numref:`chap_modern_rnn`) and hybrid interleaving with full softmax
  layers (e.g., MiniMax-01, :cite:`MiniMax.2025`) to handle exact associative
  recall :cite:`Arora.Eyuboglu.Zhang.ea.2024,Yang.Wang.Shen.ea.2024`.

## Summary

* Sliding-window attention restricts each query to the $w$ most recent keys,
  achieving $O(nw)$ time and $O(w)$ cache memory; $L$ stacked layers span
  $1 + L(w - 1)$ tokens via multi-hop relays and are typically interleaved
  with periodic global attention layers.
* Tiled GPU kernels skip work only on completely empty $b \times b$ tiles,
  yielding speedups proportional to the fraction of skipped tiles (tempered by
  boundary masking overhead).
* Linear attention factorizes the similarity kernel as $\phi(\mathbf{q})^\top
  \phi(\mathbf{k})$, turning causal attention into an $O(n)$ recurrence with
  a constant-size $d_h \times d_v$ matrix state that can be trained in chunks.
* Because a fixed-size state with nonnegative features cannot sharpen weights
  via query scaling or store more than $O(d_h)$ orthogonal keys without
  interference, practical linear-attention models add gating and hybrid
  softmax layers.

## Exercises

1. [code] **Tiles of a fused kernel.** A fused kernel works on tiles of
   $b \times b$ scores and skips a tile only if the mask excludes all of it.
   Let $T = n/b$.
    1. Show that a causal mask leaves $T(T+1)/2$ of the $T^2$ tiles, and
       that a window of width $w = kb$, for a whole number $k < T$, leaves
       $T(k + 1) - k(k + 1)/2$. Check both counts with `tiles_visited`.
       How does the count change when $w$ is not a multiple of $b$?
    1. Several documents are packed into one sequence of $n = 2048$ tokens,
       and the mask combines causality with "same document" as in the
       masks of :numref:`sec_attention-scoring-functions`. Using
       `tiles_visited` with $b = 128$, compare four documents of 512 tokens
       with documents of 1,000, 600, 300, and 148 tokens. Explain why
       documents whose boundaries do not fall on tile boundaries cost more
       tiles.

1. [code] **The feature map.** The feature map $\phi(x) =
   \mathrm{elu}(x) + 1$ in `phi` is one choice among many. Compare it with
   $\phi(x) = \mathrm{relu}(x)$ and $\phi(x) = \exp(x)$, both applied
   elementwise, on $n = 512$ random queries and keys of width `d_h`.
    1. For each map, compute the causal linear-attention weight matrix
       with entries
       $$\frac{\phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_s)}{\sum_{s' \leq t} \phi(\mathbf{q}_t)^\top \phi(\mathbf{k}_{s'})}, \qquad s \leq t,$$
       and measure its maximum absolute difference from the causal softmax
       weight matrix `d2l.masked_softmax(Q @ K.T / math.sqrt(d_h),
       d2l.causal_mask(n))`, and the mean entropy of its rows.
    1. Under which condition on $\mathbf{q}_t$ and the keys
       $\mathbf{k}_1, \ldots, \mathbf{k}_t$ does the ReLU map produce an
       undefined row? Derive the probability of this event for row $t$
       when all entries are independent standard normal, and evaluate it
       for $t = 1$ at width $64$ and at width $4$.
    1. The section scaled the queries by $\beta \in \{1, 4, 16\}$ and
       watched the largest weight. Do the same for the ReLU and exponential
       maps, measuring the mean row entropy instead. Derive, for each of
       the three maps, the limit of the normalized weights as
       $\beta \to \infty$, and explain why none of them becomes one-hot
       where softmax does.
   :exlabel:`ex_attention-at-scale-feature-map`

1. **An exact feature map.** ● The softmax kernel $K(\mathbf{q},
   \mathbf{k}) = \exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d_h})$ also
   factorizes, but only with infinitely many features.
    1. Expand $K$ in its Taylor series, and show that the degree-$p$
       term equals $\phi_p(\mathbf{q})^\top \phi_p(\mathbf{k})$ for a
       feature map $\phi_p$ of dimension $d_h^p$ built from the $p$-fold
       tensor product of its argument.
    1. Truncate the series after degree $2$. Show that the truncated kernel
       is positive for every pair of $\mathbf{q}$ and $\mathbf{k}$, so the
       normalizer of :eqref:`eq_linear-attn` cannot vanish, and give the
       dimension of the feature map that concatenates $\phi_0$, $\phi_1$,
       and $\phi_2$. For $d_h = 64$ and values of width $64$, compute the
       size of the state $(\mathbf{S}, \mathbf{z})$ in single precision and
       the length $n$ at which the key--value cache of a softmax head
       reaches the same size. A truncation of this kind is the feature map
       of Based :cite:`Arora.Eyuboglu.Zhang.ea.2024`.
    1. Take as given that the $n \times n$ matrix with entries
       $K(\mathbf{x}_i, \mathbf{x}_j)$ is positive definite for
       distinct $\mathbf{x}_1, \ldots, \mathbf{x}_n$. Show that no feature
       map of dimension $D < n$ reproduces $K$ exactly on these points
       when queries and keys coincide.

1. [extended] **Attention variants in a language model.** ● Whether the
   quality gap between exact and linear attention matters depends on the
   task. Measure it on `TinyCharLM` of
   :numref:`sec_positional-information`, with `pos='learned'` for every
   variant so that position enters the same way.
    1. Subclass `TinyCharLM` and override `attend` twice: once with a
       causal band mask of width $w$ in place of the causal mask, and once
       with the causal linear-attention weights of :eqref:`eq_linear-attn`
       for $\phi(x) = \mathrm{elu}(x) + 1$, computed explicitly as a full
       weight matrix over the context so that the method still returns its
       weights. Add $\epsilon = 10^{-6}$ to the normalizer, as in
       `linear_attention_parallel`; without it, a row whose features
       underflow divides zero by zero and training ends in NaN.
    1. Train dense attention, windows of width $w \in \{4, 16, 64\}$, and
       linear attention with the protocol of
       :numref:`subsec_pos-experiment`, and report validation perplexity at
       context $128$. Relate the window results to the reach $1 + 2(w-1)$
       of the two-block model.
    1. Build validation sequences of $128$ bytes, each a random
       $64$-byte excerpt of the validation text followed by the same
       excerpt again, and compare the mean loss on the second half with
       that on the first half for every variant. At the budget of
       :numref:`subsec_pos-experiment`, expect even dense attention to
       gain little from the copy. Train the dense model and the window
       $w = 16$ longer, or on data that repeats, until the dense model's
       second-half loss falls clearly below its first-half loss, and
       report the budget at which that happens. Which variants then
       exploit the repetition, and how does the answer relate to the reach
       of the windowed models and to the fixed-size state of linear
       attention?

1. [code] **A decaying state.** ● Give the state of
   :eqref:`eq_linear-attn-recurrence` a scalar decay $\gamma \in (0, 1)$,
   $$\mathbf{S}_t = \gamma\, \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\,\mathbf{v}_t^\top, \qquad \mathbf{z}_t = \gamma\, \mathbf{z}_{t-1} + \phi(\mathbf{k}_t).$$
    1. Show that the parallel form now weights the contribution of position
       $s$ to the output at $t$ by $\gamma^{t-s}$, and implement it by
       rescaling the cumulative sums of `linear_attention_parallel` with
       powers of $\gamma$.
    1. Add the decay to `linear_attention_recurrent`, and verify that the
       two forms agree for $n = 512$ and $\gamma = 0.99$.
    1. The rescaling multiplies position $s$ by $\gamma^{-s}$ before the
       cumulative sum. For which $n$ does this overflow single precision at
       $\gamma = 0.9$ and at $\gamma = 0.5$?
    1. Add the decay to `linear_attention_chunked`: rescale each position
       relative to the start of its chunk, and carry the state from one
       chunk to the next with the decayed recurrence. Show that no factor
       larger than $\gamma^{-c}$ appears, run the result for $\gamma = 0.5$,
       $n = 512$, and $c = 64$, and compare it with the recurrent form.
    1. Compute the lag at which the weight $\gamma^{t-s}$ falls below
       $0.01$ for $\gamma = 0.99$. A task requires recalling one token
       from $2000$ positions back, while the tokens in between are filler.
       Explain why no fixed $\gamma$ serves both that token and the
       filler, and what a decay computed from each token's own input could
       do instead.
   :exlabel:`ex_attention-at-scale-decaying-state`

[Discussions](https://d2l.discourse.group/t/36882)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §10.6]{.kicker}

Windowed and Linear Attention<br>
**sliding windows · tiles · measured cost · linear attention**
:::
:::

::: {.slide title="Exact Fused Attention Still Visits 32,896 Tiles at 32K Tokens"}
[The problem]{.kicker}

A fused kernel no longer stores the $n \times n$ score matrix, but it still
computes every score, in tiles of $128 \times 128$.

| $n = 32{,}768$ tokens, $T = 256$ rows of tiles | tiles visited |
|:--|--:|
| dense causal: $T(T+1)/2$ | 32,896 |
| window of the last 128 keys: $2T - 1$ | 511 |

::: {.d2l-note}
The score matrix is gone; the arithmetic and the cache remain.
:::
:::

::: {.slide title="Goals: Two Routes Below n², and What Each Gives Up"}
[Goals]{.kicker}

- compute the **reach** of stacked windows and the **cache** they save;
- **count tiles**, predict a fused kernel's speedup, and **time** it;
- derive **linear attention** as a recurrence;
- show why scaling the query **does not sharpen** it;
- **choose** a mechanism.

Running example: one head of width $d_h = 64$, as in GPT-2 small.
:::

::: {.slide title="This Deck Covers Cheaper Attention, Not Learned Sparsity"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Queries, Keys, and Values | attention as a soft lookup |
| Attention Scoring and Masking | scaled dot products, masks |
| Self-Attention and Multiple Heads | a sequence reads itself; several heads |
| Positional Information | where a token sits |
| The Cost of Attention | quadratic cost, fused kernels |
| **Windowed and Linear Attention** | **linear cost, and what it gives up** |
| Attention Circuits | what trained heads compute |
:::

::: {.col .narrow}
- **Uses:** masks, fused kernels, the recurrent state of an RNN.
- **Skips:** sparsity chosen from content, random-feature approximations
  of softmax, gated recurrences (the state space chapter).
:::
:::
:::

::: {.slide title="Two Routes Below n²: Score Fewer Pairs, or Factorize the Score"}
[Idea]{.kicker}

| route | change | cost | gives up |
|:--|:--|:--|:--|
| sliding window | each query sees its $w$ latest keys | $O(nw)$ | direct long-range paths |
| linear attention | $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h}) \to \phi(\mathbf{q})^\top\phi(\mathbf{k})$ | $O(n)$, fixed state | sharp recall |

$n$: sequence length; $\phi$: a feature map.
:::

::: {.slide title="A Sliding Window Turns the Causal Triangle into a Band"}
[Windows]{.kicker}

Query $i$ may read key $j$ if $j \le i$ and $i - j < w$:

@attention-at-scale-attention-through-a-window

12 positions: causal keeps 78 of 144 pairs, a window of $w = 4$ keeps 42.
Large $n$: about $nw$ pairs instead of $n^2/2$.
:::

::: {.slide title="Depth Restores the Reach: 1 + L(w − 1) Positions"}
[Windows]{.kicker}

Each layer moves information up to $w - 1$ positions back.

| layers $L$ | 1 | 2 | 4 | 32 |
|:--|--:|--:|--:|--:|
| reach $1 + L(w-1)$, $w = 4$ | 4 | 7 | 13 | 97 |
| reach, $w = 4096$ (Mistral 7B) | 4,096 | 8,191 | 16,381 | 131,041 |

[Jiang et al. 2023]{.small}
:::

::: {.slide title="Reach Is Not Use: Averaging Windows Pass On 1/256"}
[Windows]{.kicker}

- Full attention: any two positions in **one** layer.
- Windows: about $n/w$ hops; 100,000 tokens at $w = 4096$ need at least
  25 of Mistral's 32 layers.
- Equal-weight windows, $w = 4$, 4 layers: the token 12 back gets
  $(1/4)^4 = 1/256$; one global layer gives it $1/13$.

::: {.d2l-note}
A distant fact arrives only if every hop puts most of its weight on it.
:::
:::

::: {.slide title="Global Layers Keep One-Hop Paths; Windows Shrink the Cache"}
[Windows]{.kicker}

$r$ windowed layers per global layer; KV cache relative to all-global:
$$\frac{r\min(n, w) + n}{(r+1)\,n} \;\to\; \frac{1}{r+1}$$

| Gemma 3: $r = 5$, $w = 1024$ | 8K | 32K | 128K |
|:--|--:|--:|--:|
| cache vs. all-global | 0.27 | 0.19 | 0.17 |

[Gemma Team 2025]{.small}
:::

::: {.slide title="Check: Reach and Cache of a 24-Layer Model"}
[Check]{.kicker}

24 layers, windows of $w = 512$. How far back does the last layer reach?
With one global layer after every three windowed ones, what fraction of
the cache remains at $n = 4096$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $1 + 24 \cdot 511 = 12{,}265$ positions.
$(3 \cdot 512 + 4096)/(4 \cdot 4096) \approx 0.34$, tending to $1/4$.
:::
:::
:::

::: {.slide title="A Fused Kernel Skips Work Only in Whole Tiles"}
[Tiles]{.kicker}

Tiles of $b \times b = 128 \times 128$ scores at $n = 2048$:

@!attention-at-scale-a-linear-cost-implementation-3

- A tile is skipped only if the mask excludes **all** of it.
- Window: the diagonal tile and its left neighbor per row.

::: {.d2l-note}
Time grows as $nw$, rounded up to whole tiles.
:::
:::

::: {.slide title="Check: Tiles for a Narrower and a Wider Window"}
[Check]{.kicker}

$b = 128$, $n = 2048$, so 16 rows of tiles. A window of $w = 128$ touches 31
of the 256 tiles. How many does $w = 32$ touch? And $w = 512$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $w = 32$: still 31; narrowing below a tile saves nothing.
$w = 512$: up to 5 tiles per row, $16 \cdot 5 - 10 = 70$.
:::
:::
:::

::: {.slide title="Fused Kernels Take the Window as an Argument and Match the Mask"}
[Tiles]{.kicker}

Blocked `windowed_attention` and a fused window kernel, against the masked
softmax definition:

@!attention-at-scale-a-linear-cost-implementation-2

PyTorch: FlexAttention takes the band as a rule on positions; its block
mask records which tiles to skip. JAX: `local_window_size=(w - 1, 0)` with
`is_causal=True`.
:::

::: {.slide title="A Window Is Over 30× Faster at 32K Tokens, Short of the Tiles' 64×"}
::: {.cols .vc}
::: {.col .narrow}
Both arms run in one fused kernel: one sequence, 32 heads of width 64,
bfloat16. The tile count predicts the speedup; the measured one falls short
of it at every length.
:::

::: {.col}
@!attention-at-scale-the-price-of-attention-measured
:::
:::
:::

::: {.slide title="Every Tile the Band Visits Is a Masked Tile"}
[Measured]{.kicker}

| | tiles visited | tiles on the mask's edge |
|:--|--:|:--|
| dense causal | $T(T+1)/2$ | only the $T$ diagonal tiles |
| window, $w = b$ | $2T - 1$ | all of them |

- Short sequences: a fixed cost per call, no saving.
- Long sequences: dense 3–4× per doubling of $n$, window about 2×.

::: {.d2l-note}
Tile counts predict the shape; masked tiles explain most of the shortfall.
:::
:::

::: {.slide title="Check: A Window Two Tiles Wide"}
[Check]{.kicker}

At $n = 32{,}768$ with tiles of 128 ($T = 256$), the window grows to
$w = 256$. How many tiles does the kernel visit, what speedup over dense
causal does the tile count predict, and are all of them masked?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Up to 3 per row: $3 \cdot 256 - 3 = 765$ tiles, so the count
predicts $32{,}896 / 765 \approx 43\times$. Not all are masked: the tile
left of the diagonal holds key distances 1 to 255, all inside the band, so
255 tiles run without a mask.
:::
:::
:::

::: {.slide title="Moving the Brackets Removes the n × n Matrix"}
[Linear attention]{.kicker}

Replace $\exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$ by
$\phi(\mathbf{q})^\top\phi(\mathbf{k})$. Without a mask, up to the
normalization:

$$\bigl(\phi(\mathbf{Q})\,\phi(\mathbf{K})^\top\bigr)\mathbf{V}
= \phi(\mathbf{Q})\bigl(\phi(\mathbf{K})^\top\mathbf{V}\bigr)$$

Left: an $n \times n$ matrix. Right: a $d_h \times d_v$ matrix, formed once
($d_v$: width of a value).

::: {.d2l-note}
Same product, other brackets: $n^2$ becomes $n$.
:::

[Katharopoulos et al. 2020]{.small}
:::

::: {.slide title="Causal Linear Attention Is an RNN with a Matrix State"}
[Linear attention]{.kicker}

$\phi = \mathrm{elu} + 1 > 0$, with $\mathrm{elu}(x) = x$ for $x > 0$ and
$e^x - 1$ otherwise. The query factors out of the running sums:

$$\mathbf{o}_t^\top = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}{\phi(\mathbf{q}_t)^\top \mathbf{z}_t}$$
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\,\mathbf{v}_t^\top, \qquad \mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t)$$

::: {.d2l-note}
State $\mathbf{S}$: a $d_h \times d_v$ matrix, written by outer products,
read with the query.
:::
:::

::: {.slide title="Three Forms, One Function: Parallel, Recurrent, Chunked"}
[Linear attention]{.kicker}

| form | stores | work | used for |
|:--|:--|:--|:--|
| parallel (cumulative sums) | $n$ states | elementwise products, a cumulative sum | short training sequences |
| recurrent | one state | $n$ sequential steps | generation |
| chunked, chunks of $c$ | one state per chunk | matrix products | training |

Chunked: a causally masked $c \times c$ product within a chunk, the carried
state $\mathbf{S}$ across chunks. Parallel at $n = 16{,}384$, one head:
256 MiB of states.
:::

::: {.slide title="The State Stays 16 KiB While a KV Cache Grows"}
[Linear attention]{.kicker}

@!attention-at-scale-kernelizing-the-score-2

- The three forms agree to within $4 \times 10^{-7}$.
- A softmax head of width 64 caches 512 bytes per token: larger than the
  state from the 33rd token, 4 MiB at 8,192.

::: {.d2l-note}
The price of constant size: features are positive, so every read mixes in
every write.
:::
:::

::: {.slide title="Scaling the Query Sharpens Softmax but Not elu + 1"}
[Linear attention]{.kicker}

1,000 random queries, 8 random keys (uniform weight 0.125), queries
multiplied by $\beta$:

@!windowed-linear-attention-sharpening

- Softmax: weight ratios $e^{\beta\Delta}$ grow without bound.
- $\phi(\beta\mathbf{q}) \approx \beta\,\mathrm{relu}(\mathbf{q})$: $\beta$
  cancels in the normalization.
:::

::: {.slide title="Check: The ReLU Feature Map"}
[Check]{.kicker}

Take $\phi(x) = \mathrm{relu}(x)$. What happens to the linear-attention
weights of a query when it is multiplied by 16?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Nothing. $\phi(16\mathbf{q}) = 16\,\phi(\mathbf{q})$, and the
factor 16 cancels exactly between numerator and normalizer.
:::
:::
:::

::: {.slide title="Default to Exact Fused Attention; Interleave Windows for Long Contexts"}
[Choice]{.kicker}

| mechanism | cost | cache | used |
|:--|:--|:--|:--|
| exact, fused kernel | $n^2$ time | all keys | the default |
| windows + few global layers | mostly $nw$ | mostly $w$ keys | long-context models |
| plain linear attention | $n$ | one matrix | not as is |

- Linear attention improved by a decay that depends on the input, plus a
  few softmax layers (MiniMax-01: one in eight).

::: {.d2l-note}
Scaling the query does not sharpen linear attention, and it recalls poorly.
:::

[Arora et al. 2024; Yang et al. 2024; MiniMax 2025]{.small}
:::

::: {.slide title="Windows and Linear Attention Trade Reach and Sharpness for Linear Cost"}
[Wrap-up]{.kicker}

- Window: $O(nw)$ work, reach $1 + L(w - 1)$; Gemma 3 keeps about a
  quarter of the cache at 8K.
- Tiles at 2,048 tokens: causal about half, window about one in eight.
- At 32K tokens the window is more than 30× faster, short of the tiles' 64.
- Linear attention: 16 KiB of state per head; a larger query does not
  sharpen its weights.

Next: what a trained head computes.
:::
